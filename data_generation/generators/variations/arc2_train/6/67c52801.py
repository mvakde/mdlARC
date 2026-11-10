import numpy as np


_SOURCE_PAIRS = (
    (
        ("00880", "00000", "30333", "33333"),
        ("00000", "08000", "38333", "33333"),
    ),
    (
        (
            "00000000",
            "22200000",
            "22200000",
            "00000550",
            "00000550",
            "00000000",
            "80080008",
            "88888888",
        ),
        (
            "00000000",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
            "05502220",
            "85582228",
            "88888888",
        ),
    ),
    (
        (
            "000000050000",
            "000000050000",
            "000000000000",
            "660077000000",
            "660077000000",
            "000077000000",
            "000000000000",
            "101001100011",
            "111111111111",
        ),
        (
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "050660077700",
            "151661177711",
            "111111111111",
        ),
    ),
    (
        (
            "000000",
            "022000",
            "000033",
            "000033",
            "000000",
            "101001",
            "111111",
        ),
        (
            "000000",
            "000000",
            "000000",
            "000000",
            "020330",
            "121331",
            "111111",
        ),
    ),
    (
        (
            "22066000000",
            "22000000000",
            "22000001100",
            "22000001100",
            "00000000000",
            "50500500005",
            "55555555555",
        ),
        (
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
            "06011022220",
            "56511522225",
            "55555555555",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    input_key = _grid_key(input_grid)
    output_key = _grid_key(output_grid)
    return any(
        input_key == _rows_key(source_input)
        and output_key == _rows_key(source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)

    if num_train is not None and num_test is not None:
        train_count = int(num_train)
        test_count = int(num_test)
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    seed_value = int(seed)

    if num_train is None and num_test is None:
        test_count = 1 + seed_value % 4
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed_value // 4) % number_of_train_choices
        )
        return train_count, test_count

    if num_train is None:
        test_count = int(num_test)
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + seed_value % number_of_train_choices
        )
        return train_count, test_count

    train_count = int(num_train)
    maximum_test_count = min(4, train_count - 1)
    test_count = 1 + seed_value % maximum_test_count
    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    # These affect sampling only. The learned convention is the fixed rack rule;
    # literal colors never carry meaning across pairs.
    return {
        "compact_chance": float(rng.uniform(0.25, 0.55)),
        "reuse_chance": float(rng.uniform(0.2, 0.55)),
        "stack_chance": float(rng.uniform(0.3, 0.65)),
    }


def _random_composition(
    total: int,
    parts: int,
    rng: np.random.Generator,
) -> list[int]:
    result = [0] * parts
    if total > 0:
        destinations = rng.integers(0, parts, size=total)
        for destination in destinations:
            result[int(destination)] += 1
    return result


def _choose_colors(
    rng: np.random.Generator,
    widths: list[int],
    reuse_chance: float,
    prior: dict | None,
) -> tuple[int, int, dict[int, int]]:
    background = 0 if rng.random() < 0.58 else int(rng.integers(0, 10))
    if prior is not None and rng.random() < 0.32:
        background = int(prior["background"])
    rack_choices = [c for c in range(10) if c != background]
    if prior is not None and rng.random() < 0.45:
        prior_payload = [c for c in prior["colors"].values() if c != background]
        if prior_payload:
            rack = int(rng.choice(prior_payload))
        else:
            rack = int(rng.choice(rack_choices))
    else:
        rack = int(rng.choice(rack_choices))
    available = [c for c in range(10) if c not in (background, rack)]
    colors: dict[int, int] = {}
    for width in widths:
        if colors and rng.random() < reuse_chance:
            colors[width] = int(rng.choice(list(colors.values())))
        else:
            colors[width] = int(rng.choice(available))
    # Two training demonstrations explicitly reuse a visible payload color at
    # a different size, so a fixed color-to-width interpretation is refuted.
    if prior is not None:
        candidates = [
            (width, color)
            for width in widths
            for old_width, color in prior["colors"].items()
            if old_width != width and color not in (background, rack)
        ]
        if candidates:
            width, color = candidates[int(rng.integers(len(candidates)))]
            colors[width] = color
    return background, rack, colors


def _slot_order(
    rng: np.random.Generator,
    widths: list[int],
    requested: str | None,
) -> list[int]:
    if requested is None:
        requested = str(rng.choice(["ascending", "descending", "shuffled"], p=[0.42, 0.25, 0.33]))
    if requested == "ascending":
        return sorted(widths)
    if requested == "descending":
        return sorted(widths, reverse=True)
    for _ in range(12):
        order = [int(x) for x in rng.permutation(widths)]
        if len(order) < 3 or (order != sorted(widths) and order != sorted(widths, reverse=True)):
            return order
    return sorted(widths, reverse=True) if len(widths) == 2 else widths[1:] + widths[:1]


def _separators(rng: np.random.Generator, widths: list[int]) -> list[int]:
    count = len(widths) + 1
    if len(widths) == 1 and rng.random() < 0.45:
        return [int(x) for x in rng.integers(1, 4, size=2)]
    extras = [int(x) for x in rng.integers(0, 7, size=count)]
    capacity = 30 - sum(widths) - count
    while sum(extras) > capacity:
        choices = [i for i, value in enumerate(extras) if value]
        extras[int(rng.choice(choices))] -= 1
    return [value + 1 for value in extras]


def _conflicts(a: dict, b: dict) -> bool:
    return (
        a["top"] <= b["top"] + b["height"]
        and b["top"] <= a["top"] + a["height"]
        and a["left"] <= b["left"] + b["width"]
        and b["left"] <= a["left"] + a["width"]
    )


def _place_pieces(
    rng: np.random.Generator,
    specs: list[dict],
    grid_height: int,
    grid_width: int,
    stack: bool,
) -> list[dict]:
    available_height = grid_height - 3
    placed: list[dict] = []
    ordered = [specs[int(i)] for i in rng.permutation(len(specs))]
    remaining = ordered
    if stack and len(ordered) >= 2:
        ordered.sort(key=lambda spec: spec["height"], reverse=True)
        first, second = ordered[:2]
        if first["height"] + second["height"] + 1 <= available_height:
            left = int(rng.integers(0, grid_width - max(first["width"], second["width"]) + 1))
            top = int(rng.integers(0, available_height - first["height"] - second["height"]))
            placed = [
                {**first, "top": top, "left": left},
                {**second, "top": top + first["height"] + 1, "left": left},
            ]
            remaining = ordered[2:]
    for spec in remaining:
        found = False
        for _ in range(180):
            candidate = {
                **spec,
                "top": int(rng.integers(0, available_height - spec["height"] + 1)),
                "left": int(rng.integers(0, grid_width - spec["width"] + 1)),
            }
            if all(not _conflicts(candidate, old) for old in placed):
                placed.append(candidate)
                found = True
                break
        if not found:
            break
    if len(placed) == len(specs):
        return placed

    # Guaranteed construction: every piece fits in its own column band.
    occupied = sum(spec["width"] for spec in ordered) + len(ordered) - 1
    slack = grid_width - occupied
    if slack < 0:
        raise RuntimeError("rack is too narrow for fallback placement")
    padding = _random_composition(slack, len(ordered) + 1, rng)
    left = padding[0]
    placed = []
    for index, spec in enumerate(ordered):
        placed.append({
            **spec,
            "top": int(rng.integers(0, available_height - spec["height"] + 1)),
            "left": left,
        })
        left += spec["width"]
        if index < len(ordered) - 1:
            left += 1 + padding[index + 1]
    return placed


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    pair_index: int,
    widths: list[int],
    order_kind: str | None,
    prior_colors: dict | None,
    flip_three: bool | None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    background, rack, color_by_width = _choose_colors(
        rng, widths, puzzle_parameters["reuse_chance"], prior_colors
    )
    slot_widths = _slot_order(rng, widths, order_kind)
    separator_lengths = _separators(rng, widths)

    grid_width = int(sum(slot_widths) + sum(separator_lengths))
    piece_specs = []
    for width in widths:
        if width == 2:
            piece_height, piece_width = 2, 2
        else:
            rotate = bool(rng.integers(0, 2)) if width != 3 or flip_three is None else flip_three
            piece_height, piece_width = (width, 2) if rotate else (2, width)
        piece_specs.append({
            "normalized_width": width,
            "height": piece_height,
            "width": piece_width,
            "color": color_by_width[width],
        })
    minimum_height = max(spec["height"] for spec in piece_specs) + 3
    stack = len(widths) >= 2 and rng.random() < puzzle_parameters["stack_chance"]
    if stack:
        tallest = sorted((spec["height"] for spec in piece_specs), reverse=True)
        minimum_height = max(minimum_height, tallest[0] + tallest[1] + 4)
    compact_probability = 0.75 if len(widths) == 1 else puzzle_parameters["compact_chance"]
    if rng.random() < compact_probability:
        grid_height = minimum_height + int(rng.integers(0, 2))
    else:
        extra = 4 if stack else (5 if len(widths) <= 2 else 7)
        grid_height = minimum_height + int(rng.integers(2, extra + 1))
    grid_height = min(grid_height, 30)

    input_grid = np.full(
        (grid_height, grid_width),
        background,
        dtype=np.int8,
    )
    input_grid[-1, :] = rack
    input_grid[-2, :] = rack

    openings: dict[int, tuple[int, int]] = {}
    cursor = 0
    for slot_index, slot_width in enumerate(slot_widths):
        cursor += separator_lengths[slot_index]
        start = cursor
        end = start + slot_width
        input_grid[-2, start:end] = background
        openings[slot_width] = (start, end)
        cursor = end

    if cursor + separator_lengths[-1] != grid_width:
        raise RuntimeError("internal rack construction error")

    placed_pieces = _place_pieces(rng, piece_specs, grid_height, grid_width, stack)
    for piece in placed_pieces:
        top, left = piece["top"], piece["left"]
        input_grid[top:top + piece["height"], left:left + piece["width"]] = piece["color"]

    output_grid = input_grid.copy()

    for piece in placed_pieces:
        top = int(piece["top"])
        left = int(piece["left"])
        piece_height = int(piece["height"])
        piece_width = int(piece["width"])
        output_grid[
            top : top + piece_height,
            left : left + piece_width,
        ] = background

    for piece in placed_pieces:
        normalized_width = int(piece["normalized_width"])
        color = int(piece["color"])
        start, end = openings[normalized_width]
        output_grid[-3, start:end] = color
        output_grid[-2, start:end] = color

    return input_grid, output_grid


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must be identical",
    )

    height, width = input_grid.shape
    _require(height >= 4, "grid is too short to contain a rack")

    rack = int(input_grid[-1, 0])
    _require(
        np.all(input_grid[-1, :] == rack),
        "bottom rack base must be completely filled",
    )

    background = int(input_grid[-3, 0])
    _require(
        background != rack,
        "background and rack colors must differ",
    )
    _require(
        np.all(input_grid[-3, :] == background),
        "input must have a blank row separating pieces from the rack",
    )

    opening_row = input_grid[-2, :]
    opening_values = set(int(x) for x in np.unique(opening_row))
    _require(
        opening_values.issubset({background, rack}),
        "rack opening row contains an unrelated color",
    )
    _require(
        int(opening_row[0]) == rack
        and int(opening_row[-1]) == rack,
        "every opening must be bounded by rack-colored separators",
    )

    openings = []
    column = 0
    while column < width:
        value = int(opening_row[column])
        if value == rack:
            column += 1
            continue

        start = column
        while (
            column < width
            and int(opening_row[column]) == background
        ):
            column += 1
        end = column

        _require(
            start > 0 and end < width,
            "an opening is not bounded on both sides",
        )
        openings.append(
            {
                "start": start,
                "end": end,
                "width": end - start,
            }
        )

    _require(len(openings) >= 1, "rack must contain an opening")

    opening_widths = [opening["width"] for opening in openings]
    _require(
        len(set(opening_widths)) == len(opening_widths),
        "opening widths must be unique",
    )

    upper_rows = height - 2
    visited = np.zeros((upper_rows, width), dtype=bool)
    pieces = []

    for row in range(upper_rows):
        for col in range(width):
            if visited[row, col]:
                continue
            if int(input_grid[row, col]) == background:
                visited[row, col] = True
                continue

            stack = [(row, col)]
            visited[row, col] = True
            cells = []

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))

                for delta_row, delta_col in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (
                        0 <= next_row < upper_rows
                        and 0 <= next_col < width
                    ):
                        continue
                    if visited[next_row, next_col]:
                        continue
                    if (
                        int(input_grid[next_row, next_col])
                        == background
                    ):
                        continue

                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            component_colors = {
                int(input_grid[r, c]) for r, c in cells
            }
            _require(
                len(component_colors) == 1,
                "a detached component must be monochrome",
            )
            color = next(iter(component_colors))
            _require(
                color != rack,
                "detached pieces may not use the rack color",
            )

            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells) + 1
            left = min(c for _, c in cells)
            right = max(c for _, c in cells) + 1
            piece_height = bottom - top
            piece_width = right - left

            _require(
                len(cells) == piece_height * piece_width,
                "a detached piece must be a solid rectangle",
            )
            _require(
                np.all(input_grid[top:bottom, left:right] == color),
                "a detached piece has a hole or mixed color",
            )
            _require(
                piece_height == 2 or piece_width == 2,
                "each piece must have one dimension equal to two",
            )

            normalized_width = (
                piece_width if piece_height == 2 else piece_height
            )
            pieces.append(
                {
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "height": piece_height,
                    "width": piece_width,
                    "normalized_width": normalized_width,
                    "color": color,
                }
            )

    _require(
        len(pieces) == len(openings),
        "piece and opening counts must match",
    )

    piece_widths = [
        piece["normalized_width"] for piece in pieces
    ]
    _require(
        len(set(piece_widths)) == len(piece_widths),
        "normalized piece widths must be unique",
    )
    _require(
        set(piece_widths) == set(opening_widths),
        "piece widths and opening widths must match exactly",
    )

    opening_by_width = {
        opening["width"]: opening for opening in openings
    }

    expected_output = input_grid.copy()
    for piece in pieces:
        expected_output[
            piece["top"] : piece["bottom"],
            piece["left"] : piece["right"],
        ] = background

    for piece in pieces:
        opening = opening_by_width[piece["normalized_width"]]
        start = opening["start"]
        end = opening["end"]
        color = piece["color"]
        expected_output[-3, start:end] = color
        expected_output[-2, start:end] = color

    _require(
        np.array_equal(output_grid, expected_output),
        "output is not the uniquely reconstructed rack-filling result",
    )

    return {
        "background": background,
        "rack": rack,
        "pieces": pieces,
        "openings": openings,
        "piece_widths": tuple(sorted(piece_widths)),
        "slot_widths": tuple(opening_widths),
    }


def _is_ascending(widths: tuple[int, ...]) -> bool:
    return len(widths) > 1 and widths == tuple(sorted(widths))


def _is_nonascending(widths: tuple[int, ...]) -> bool:
    return len(widths) > 1 and widths != tuple(sorted(widths))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[split], list) for split in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         "illegal grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors must be 0..9")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            _require(not _is_source_pair(pair["input"], pair["output"]),
                     "exact source-pair copy")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(pair["input"], pair["output"])
            _require(1 <= len(info["pieces"]) <= 5, "unsupported piece count")
            _require(max(info["piece_widths"]) <= (5 if split == "train" else 6),
                     "width exceeds regime range")
            infos[split].append(info)

    train_infos, test_infos = infos["train"], infos["test"]
    first_two = train_infos[:2]
    _require(all(len(info["pieces"]) >= 2 for info in first_two),
             "first two demonstrations need multiple pieces")
    _require(any({1, 2, 3}.issubset(info["piece_widths"]) for info in first_two),
             "training must show the source widths together")
    training_widths = set().union(*(set(info["piece_widths"]) for info in train_infos))
    _require({1, 2, 3}.issubset(training_widths),
             "training must cover widths 1, 2, and 3")
    _require(any(_is_ascending(info["slot_widths"]) for info in first_two)
             and any(_is_nonascending(info["slot_widths"]) for info in first_two),
             "training must show both ascending and reordered racks")
    orientations = {
        (piece["height"], piece["width"])
        for info in first_two for piece in info["pieces"]
        if piece["normalized_width"] == 3
    }
    _require({(2, 3), (3, 2)}.issubset(orientations),
             "training must show both orientations of a non-square piece")
    _require(any(piece["normalized_width"] == 2 for info in first_two
                 for piece in info["pieces"]), "training must show a square piece")
    largest_learned = max(training_widths)
    for info in test_infos:
        unseen = set(info["piece_widths"]) - training_widths
        _require(not unseen or unseen == {largest_learned + 1},
                 "test introduced a new width without a size progression")
        _require(max(info["piece_widths"]) <= largest_learned + 1,
                 "test width exceeds learned progression")


def _training_plan(rng: np.random.Generator, train_count: int) -> list[dict]:
    full_index = int(rng.integers(0, 2))
    first = [1, 2, 3]
    if rng.random() < 0.22:
        first.append(4)
    other = [3, int(rng.choice([1, 2]))]
    if rng.random() < 0.45:
        other.append(1 if 1 not in other else 2)
    if rng.random() < 0.18:
        other.append(4)
    widths = [None, None]
    widths[full_index] = sorted(set(first))
    widths[1 - full_index] = sorted(set(other))
    ascending_index = int(rng.integers(0, 2))
    first_flip = bool(rng.integers(0, 2))
    plan = []
    for index in range(2):
        if index == ascending_index:
            order = "ascending"
        elif len(widths[index]) == 2:
            order = "descending"
        else:
            order = "descending" if rng.random() < 0.5 else "shuffled"
        plan.append({"widths": widths[index], "order": order,
                     "flip_three": first_flip if index == 0 else not first_flip})
    for _ in range(2, train_count):
        count = int(rng.choice([1, 2, 3, 4, 5], p=[0.24, 0.21, 0.28, 0.19, 0.08]))
        if count == 1:
            chosen = [int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5]))]
        else:
            chosen = sorted(int(x) for x in rng.choice(np.arange(1, 6), size=count, replace=False))
        plan.append({"widths": chosen, "order": None, "flip_three": None})
    return plan


def _test_widths(rng: np.random.Generator, learned: set[int]) -> list[int]:
    novel = max(learned) + 1 <= 6 and rng.random() < 0.58
    count = int(rng.choice([1, 2, 3, 4, 5], p=[0.12, 0.24, 0.33, 0.23, 0.08]))
    count = min(count, len(learned) + int(novel))
    if novel:
        count = max(2, count)
        chosen = [int(x) for x in rng.choice(sorted(learned), size=count - 1, replace=False)]
        chosen.append(max(learned) + 1)
    else:
        chosen = [int(x) for x in rng.choice(sorted(learned), size=count, replace=False)]
    return sorted(chosen)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    puzzle_parameters = sample_puzzle_parameters(rng)
    plan = _training_plan(rng, train_count)
    learned = set().union(*(set(entry["widths"]) for entry in plan))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    first_colors = None
    for split, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            if split == "train":
                entry = plan[pair_index]
                widths, order, flip_three = entry["widths"], entry["order"], entry["flip_three"]
            else:
                widths, order, flip_three = _test_widths(rng, learned), None, None
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1, dtype=np.int64)))
            for _ in range(80):
                input_grid, output_grid = generate_pair(
                    pair_rng, "source_train" if split == "train" else "target_test",
                    puzzle_parameters, pair_index, widths, order,
                    first_colors if split == "train" and pair_index == 1 else None,
                    flip_three,
                )
                if _grid_key(input_grid) in used_inputs or _is_source_pair(input_grid, output_grid):
                    continue
                pair = {"input": input_grid, "output": output_grid}
                puzzle[split].append(pair)
                used_inputs.add(_grid_key(input_grid))
                if split == "train" and pair_index == 0:
                    info = _analyze_pair(input_grid, output_grid)
                    first_colors = {
                        "background": info["background"], "rack": info["rack"],
                        "colors": {piece["normalized_width"]: piece["color"]
                                   for piece in info["pieces"]},
                    }
                break
            else:
                raise ValueError("could not generate a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False
    for section_name in ("train", "test"):
        if len(first[section_name]) != len(second[section_name]):
            return False
        for first_pair, second_pair in zip(
            first[section_name],
            second[section_name],
        ):
            if not np.array_equal(
                first_pair["input"], second_pair["input"]
            ):
                return False
            if not np.array_equal(
                first_pair["output"], second_pair["output"]
            ):
                return False
    return True


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            section_name,
            tuple(
                (
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
                for pair in puzzle[section_name]
            ),
        )
        for section_name in ("train", "test")
    )


def _predict_from_demonstrations(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # Input-only reconstruction. The training outputs first establish that the
    # same relocation explains all demonstrations; no generated metadata or
    # test label is passed to this function.
    def reconstruct(grid: np.ndarray) -> np.ndarray:
        background = int(grid[-3, 0])
        rack = int(grid[-1, 0])
        openings = {}
        col = 0
        while col < grid.shape[1]:
            if int(grid[-2, col]) == rack:
                col += 1
                continue
            start = col
            while col < grid.shape[1] and int(grid[-2, col]) == background:
                col += 1
            openings[col - start] = (start, col)
        result = grid.copy()
        seen = np.zeros(grid.shape, dtype=bool)
        for row in range(grid.shape[0] - 2):
            for col in range(grid.shape[1]):
                if seen[row, col] or int(grid[row, col]) == background:
                    continue
                color = int(grid[row, col])
                frontier = [(row, col)]
                seen[row, col] = True
                cells = []
                while frontier:
                    y, x = frontier.pop()
                    cells.append((y, x))
                    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if (0 <= yy < grid.shape[0] - 2 and 0 <= xx < grid.shape[1]
                                and not seen[yy, xx] and int(grid[yy, xx]) == color):
                            seen[yy, xx] = True
                            frontier.append((yy, xx))
                rows = [cell[0] for cell in cells]
                cols = [cell[1] for cell in cells]
                piece_height = max(rows) - min(rows) + 1
                piece_width = max(cols) - min(cols) + 1
                normalized_width = piece_width if piece_height == 2 else piece_height
                for y, x in cells:
                    result[y, x] = background
                start, end = openings[normalized_width]
                result[-3:-1, start:end] = color
        return result

    for pair in train_pairs:
        _require(np.array_equal(reconstruct(pair["input"]), pair["output"]),
                 "visible rule does not explain a training demonstration")
    return reconstruct(test_input)


def self_test() -> None:
    for seed, train_count, test_count in ((700, 2, 1), (701, 10, 1), (702, 10, 4),
                                          (703, 5, 2), (704, 8, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit counts were ignored")
        validate_generated_puzzle(puzzle)
    for seed, train_count, test_count in ((800, None, 1), (801, None, 4),
                                          (802, 2, None), (803, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count,
                 "partial train count was ignored")
        _require(test_count is None or len(puzzle["test"]) == test_count,
                 "partial test count was ignored")
        validate_generated_puzzle(puzzle)
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}, {"num_train": True},
                    {"num_test": 1.5}):
        try:
            generate_puzzle(1000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count was accepted: {request}")
    _require(_puzzles_equal(generate_puzzle(123456, 7, 3),
                            generate_puzzle(123456, 7, 3)),
             "explicit generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(654321), generate_puzzle(654321)),
             "automatic generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(0), generate_puzzle(1)),
             "different seeds produced identical puzzles")

    train_counts, test_counts, puzzle_keys = set(), set(), set()
    grid_keys, shapes, layouts, role_pairs = set(), set(), set(), set()
    orders, piece_counts, dimensions = set(), set(), set()
    repeated_color_pairs = shared_column_pairs = long_separator_pairs = 0
    compact_pairs = role_reuse_puzzles = varying_background_puzzles = 0
    learned_tests = novel_tests = combined_pairs = 0
    varying_count_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        train_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        learned = set().union(*(set(info["piece_widths"]) for info in train_infos))
        all_infos = train_infos + test_infos
        if len({len(info["pieces"]) for info in all_infos}) >= 2:
            varying_count_puzzles += 1
        if len({info["background"] for info in all_infos}) >= 2:
            varying_background_puzzles += 1
        if any(info["rack"] in {piece["color"] for previous in all_infos[:i]
                                 for piece in previous["pieces"]}
               for i, info in enumerate(all_infos)):
            role_reuse_puzzles += 1
        for test_pair, info in zip(puzzle["test"], test_infos):
            predicted = _predict_from_demonstrations(puzzle["train"], test_pair["input"])
            _require(np.array_equal(predicted, test_pair["output"]),
                     "input-only test prediction failed")
            if set(info["piece_widths"]).issubset(learned):
                learned_tests += 1
            else:
                novel_tests += 1
        for split, pairs, infos in (("train", puzzle["train"], train_infos),
                                     ("test", puzzle["test"], test_infos)):
            for pair, info in zip(pairs, infos):
                grid = pair["input"]
                grid_keys.add(_grid_key(grid))
                shapes.add(tuple(grid.shape))
                role_pairs.add((info["background"], info["rack"]))
                piece_counts.add(len(info["pieces"]))
                orders.add(info["slot_widths"])
                for piece in info["pieces"]:
                    dimensions.add((piece["normalized_width"], piece["height"], piece["width"]))
                layout = tuple(sorted((p["normalized_width"], p["height"], p["width"],
                                       p["top"], p["left"]) for p in info["pieces"]))
                layouts.add((split, tuple(grid.shape), info["slot_widths"], layout))
                colors = [piece["color"] for piece in info["pieces"]]
                repeated = len(set(colors)) < len(colors)
                if repeated:
                    repeated_color_pairs += 1
                shared = any(max(a["left"], b["left"]) < min(a["right"], b["right"])
                             for i, a in enumerate(info["pieces"])
                             for b in info["pieces"][i + 1:])
                if shared:
                    shared_column_pairs += 1
                if repeated and shared and _is_nonascending(info["slot_widths"]):
                    combined_pairs += 1
                gaps = [info["openings"][0]["start"]]
                gaps += [b["start"] - a["end"] for a, b in
                         zip(info["openings"], info["openings"][1:])]
                gaps.append(grid.shape[1] - info["openings"][-1]["end"])
                if max(gaps) > 3:
                    long_separator_pairs += 1
                if grid.shape[0] <= 5:
                    compact_pairs += 1
                _require(not _is_source_pair(pair["input"], pair["output"]),
                         "source pair was copied")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack range")
    _require(len(puzzle_keys) >= 40 and len(grid_keys) >= 150,
             "puzzle or grid diversity is insufficient")
    _require(len(shapes) >= 25 and len(layouts) >= 150 and len(role_pairs) >= 25,
             "geometric or color diversity is insufficient")
    _require(piece_counts == {1, 2, 3, 4, 5}, "piece count range is incomplete")
    _require(any(_is_ascending(order) for order in orders)
             and any(order == tuple(sorted(order, reverse=True)) for order in orders if len(order) > 1)
             and any(len(order) >= 3 and not _is_ascending(order)
                     and order != tuple(sorted(order, reverse=True)) for order in orders),
             "ascending, descending, and mixed racks were not all observed")
    _require((4, 4, 2) in dimensions and (4, 2, 4) in dimensions,
             "width-four orientations were not both observed")
    _require(repeated_color_pairs >= 12 and shared_column_pairs >= 12
             and long_separator_pairs >= 12 and compact_pairs >= 1,
             "review-requested pair layouts lack coverage")
    _require(role_reuse_puzzles >= 8 and varying_background_puzzles >= 20,
             "local semantic colors lack coverage")
    _require(learned_tests >= 12 and novel_tests >= 12,
             "learned-only and larger-width tests lack coverage")
    _require(varying_count_puzzles >= 25 and combined_pairs >= 1,
             "within-puzzle and combined variation is insufficient")
