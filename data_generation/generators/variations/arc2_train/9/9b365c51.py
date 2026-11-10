import numpy as np


def _rows_to_array(rows):
    return np.array(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (
        _rows_to_array(input_rows),
        _rows_to_array(output_rows),
    )
    for input_rows, output_rows in (
        (
            (
                "010607088800888",
                "010607088800888",
                "010607000088888",
                "010607000088000",
                "010607000088000",
                "010607000088000",
            ),
            (
                "000000011100777",
                "000000011100777",
                "000000000066777",
                "000000000066000",
                "000000000066000",
                "000000000066000",
            ),
        ),
        (
            (
                "04030200000000000000",
                "04030200888800088880",
                "04030200888800088880",
                "04030200888888088880",
                "04030200888888088880",
                "04030200000088000000",
                "04030200000088000000",
                "04030200000000000000",
            ),
            (
                "00000000000000000000",
                "00000000444400022220",
                "00000000444400022220",
                "00000000444433022220",
                "00000000444433022220",
                "00000000000033000000",
                "00000000000033000000",
                "00000000000000000000",
            ),
        ),
        (
            (
                "030204070880008800",
                "030204070888808800",
                "030204070008808800",
                "030204070008808888",
                "030204070000008888",
                "030204070000000088",
                "030204070000000000",
            ),
            (
                "000000000330004400",
                "000000000332204400",
                "000000000002204400",
                "000000000002204477",
                "000000000000004477",
                "000000000000000077",
                "000000000000000000",
            ),
        ),
        (
            (
                "0103020406070888000088800000",
                "0103020406070888008888800088",
                "0103020406070888008800088888",
                "0103020406070000888800088888",
                "0103020406070000880000088888",
                "0103020406070000880000088800",
                "0103020406070000880000088800",
            ),
            (
                "0000000000000111000044400000",
                "0000000000000111002244400077",
                "0000000000000111002200066677",
                "0000000000000000332200066677",
                "0000000000000000330000066677",
                "0000000000000000330000066600",
                "0000000000000000330000066600",
            ),
        ),
    )
)


def _checked_optional_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train_count = _checked_optional_count(num_train, "num_train")
    test_count = _checked_optional_count(num_test, "num_test")

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    selector = int(seed)

    if train_count is None and test_count is None:
        test_count = 1 + selector % 4
        legal_train_counts = list(range(max(2, test_count + 1), 11))
        index = (selector // 4) % len(legal_train_counts)
        train_count = legal_train_counts[index]
    elif train_count is None:
        legal_train_counts = list(range(max(2, test_count + 1), 11))
        train_count = legal_train_counts[selector % len(legal_train_counts)]
    elif test_count is None:
        legal_test_counts = list(range(1, min(4, train_count - 1) + 1))
        if not legal_test_counts:
            raise ValueError("no legal test count exists for num_train")
        test_count = legal_test_counts[selector % len(legal_test_counts)]

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng):
    roles = tuple(int(value) for value in rng.permutation(10))
    mode = ("source_like", "stable", "varying", "varying")[
        int(rng.integers(0, 4))
    ]
    return {
        "mode": mode,
        "base_roles": roles[:2],
        "source_palette": roles[2:8],
    }



def _sample_roles(rng, parameters, train_count, test_count):
    total = train_count + test_count
    base = parameters["base_roles"]
    if parameters["mode"] != "varying":
        return [base] * total

    roles = [base]
    alternatives = [
        (background, template)
        for background in range(10)
        for template in range(10)
        if background != template
        and background != base[0]
        and template != base[1]
    ]
    roles.append(alternatives[int(rng.integers(0, len(alternatives)))])
    for _ in range(total - 2):
        permutation = rng.permutation(10)
        roles.append((int(permutation[0]), int(permutation[1])))
    return roles


def _sample_lengths(rng, parameters, train_count, test_count):
    if parameters["mode"] == "source_like":
        train_lengths = [3, 3]
        if train_count >= 3:
            train_lengths.append(4)
        train_lengths.extend(
            int(rng.integers(2, 6)) for _ in range(train_count - len(train_lengths))
        )
        return train_lengths, [6] * test_count

    train_lengths = [int(rng.integers(3, 7)), int(rng.integers(3, 7))]
    train_lengths.extend(
        int(rng.integers(2, 9)) for _ in range(train_count - 2)
    )
    if parameters["mode"] == "stable" and train_count >= 3 and rng.random() < 0.6:
        train_lengths[2] = train_lengths[0]
    test_lengths = [int(rng.integers(2, 9)) for _ in range(test_count)]
    return train_lengths, test_lengths


def _sample_key_order(rng, roles, length, permitted, preferred=None):
    available = [color for color in permitted if color not in roles]
    assert len(available) >= length
    order = []
    if preferred is not None and preferred in available:
        order.append(preferred)
    remaining = [color for color in available if color not in order]
    extra = rng.choice(remaining, size=length - len(order), replace=False)
    order.extend(int(color) for color in extra)
    rng.shuffle(order)
    return tuple(order)


def _make_pair_specs(rng, parameters, train_count, test_count):
    roles = _sample_roles(rng, parameters, train_count, test_count)
    train_lengths, test_lengths = _sample_lengths(
        rng, parameters, train_count, test_count
    )
    lengths = train_lengths + test_lengths
    specifications = []
    training_colors = set()

    for index, (role_pair, length) in enumerate(zip(roles, lengths)):
        if parameters["mode"] == "source_like":
            permitted = parameters["source_palette"]
            if index == 0:
                order = _sample_key_order(rng, role_pair, 3, permitted[:3])
            elif index == 1:
                order = _sample_key_order(rng, role_pair, 3, permitted[3:])
            else:
                order = _sample_key_order(rng, role_pair, length, permitted)
        else:
            permitted = tuple(range(10))
            unseen = [
                color for color in permitted
                if color not in role_pair and color not in training_colors
            ]
            preferred = None
            if index >= train_count and unseen and rng.random() < 0.45:
                preferred = unseen[int(rng.integers(0, len(unseen)))]
            order = _sample_key_order(
                rng, role_pair, length, permitted, preferred
            )
            if (
                parameters["mode"] == "stable"
                and index == 2
                and length == len(specifications[0]["key_colors"])
                and rng.random() < 0.7
            ):
                order = specifications[0]["key_colors"]

        if index == 1 and order == specifications[0]["key_colors"]:
            order = tuple(reversed(order))
        if index < train_count:
            training_colors.update(order)
        specifications.append({"roles": role_pair, "key_colors": order})

    return specifications[:train_count], specifications[train_count:]


def _sample_relations(rng, count, width_limit, first_training=False):
    if first_training:
        return tuple(
            "edge" if index % 2 == 0 else "corner"
            for index in range(count - 1)
        )

    minimum_width = 3 * count + 1
    maximum_gaps = min(count - 1, width_limit - minimum_width)
    assert maximum_gaps >= 0
    category = ("touching", "touching", "mixed", "mixed", "separated")[
        int(rng.integers(0, 5))
    ]
    if category == "touching":
        gap_count = 0
    elif category == "separated":
        gap_count = maximum_gaps
    elif maximum_gaps == 0:
        gap_count = 0
    else:
        gap_count = int(rng.integers(1, maximum_gaps + 1))

    gap_positions = set()
    if gap_count:
        gap_positions = set(
            int(value)
            for value in rng.choice(count - 1, size=gap_count, replace=False)
        )
    relations = [
        "gap" if index in gap_positions
        else ("edge" if rng.random() < 0.65 else "corner")
        for index in range(count - 1)
    ]
    if gap_count == 0 and count >= 5 and rng.random() < 0.55:
        start = int(rng.integers(0, count - 3))
        relations[start:start + 3] = ["edge"] * 3
    return tuple(relations)


def _sample_intervals(rng, height, relations):
    candidates = [
        (top, top + block_height - 1)
        for block_height in range(1, height)
        for top in range(height - block_height + 1)
    ]
    intervals = []
    for index in range(len(relations) + 1):
        if index == 0 or relations[index - 1] == "gap":
            valid = candidates
        else:
            previous_top, previous_bottom = intervals[-1]
            if relations[index - 1] == "edge":
                valid = [
                    candidate for candidate in candidates
                    if candidate != intervals[-1]
                    and max(previous_top, candidate[0])
                    <= min(previous_bottom, candidate[1])
                ]
            else:
                valid = [
                    candidate for candidate in candidates
                    if candidate[0] == previous_bottom + 1
                    or candidate[1] + 1 == previous_top
                ]
        assert valid
        desired_height = int(rng.integers(1, height))
        distances = [
            abs(candidate[1] - candidate[0] + 1 - desired_height)
            for candidate in valid
        ]
        best = min(distances)
        nearest = [
            candidate for candidate, distance in zip(valid, distances)
            if distance == best
        ]
        intervals.append(nearest[int(rng.integers(0, len(nearest)))])
    return tuple(intervals)


def _sample_horizontal_layout(rng, count, relations, width_limit):
    prefix = 2 * count + 1
    widths = [1] * count
    gaps = [int(relation == "gap") for relation in relations]
    margins = [0, 0]
    minimum_width = prefix + sum(widths) + sum(gaps)
    target_width = int(rng.integers(minimum_width, width_limit + 1))
    slack = target_width - minimum_width

    while slack:
        slots = (
            [("width", index)] * 3
            for index, block_width in enumerate(widths) if block_width < 8
        )
        weighted = [slot for group in slots for slot in group]
        weighted.extend(
            ("gap", index)
            for index, gap in enumerate(gaps) if 0 < gap < 6
            for _ in range(2)
        )
        weighted.extend(
            ("margin", index)
            for index, margin in enumerate(margins)
            if margin < (8 if index == 0 else 10)
            for _ in range(4)
        )
        assert weighted
        kind, index = weighted[int(rng.integers(0, len(weighted)))]
        if kind == "width":
            widths[index] += 1
        elif kind == "gap":
            gaps[index] += 1
        else:
            margins[index] += 1
        slack -= 1

    rectangle_columns = []
    left = prefix + margins[0]
    for index, block_width in enumerate(widths):
        right = left + block_width - 1
        rectangle_columns.append((left, right))
        if index < count - 1:
            left = right + 1 + gaps[index]
    return target_width, tuple(rectangle_columns)


def generate_pair(rng, regime, parameters, specification, variant_index):
    count = len(specification["key_colors"])
    source_like = parameters["mode"] == "source_like"
    if source_like and regime == "source_train":
        height = int(rng.integers(6, 11))
        width_limit = 20
    elif source_like:
        height = int(rng.integers(7, 14))
        width_limit = 30
    else:
        height = int(rng.integers(5, 21))
        width_limit = 30

    relations = _sample_relations(
        rng, count, width_limit,
        first_training=regime == "source_train" and variant_index == 0,
    )
    intervals = _sample_intervals(rng, height, relations)
    width, columns = _sample_horizontal_layout(
        rng, count, relations, width_limit
    )
    background, template = specification["roles"]
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    for index, color in enumerate(specification["key_colors"]):
        input_grid[:, 1 + 2 * index] = color
    for index, (rows, horizontal) in enumerate(zip(intervals, columns)):
        top, bottom = rows
        left, right = horizontal
        input_grid[top:bottom + 1, left:right + 1] = template
        output_grid[top:bottom + 1, left:right + 1] = (
            specification["key_colors"][index]
        )
    return input_grid, output_grid


def _grid_signature(grid):
    return (
        int(grid.shape[0]), int(grid.shape[1]),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _is_exact_source_pair(input_grid, output_grid):
    return any(
        np.array_equal(input_grid, source_input)
        and np.array_equal(output_grid, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _analyze_pair(pair):
    input_grid = pair["input"]
    output_grid = pair["output"]
    height, width = input_grid.shape
    background = int(input_grid[0, 0])
    assert np.all(input_grid[:, 0] == background)

    key_colors = []
    column = 1
    while column + 1 < width:
        color = int(input_grid[0, column])
        if not (
            color != background
            and np.all(input_grid[:, column] == color)
            and np.all(input_grid[:, column + 1] == background)
        ):
            break
        assert color not in key_colors
        key_colors.append(color)
        column += 2
    assert 2 <= len(key_colors) <= 8
    workspace_start = column
    remaining = set(int(value) for value in np.unique(input_grid))
    remaining -= {background, *key_colors}
    assert len(remaining) == 1
    template = next(iter(remaining))
    assert template != background

    footprints = []
    for workspace_column in range(workspace_start, width):
        rows = np.flatnonzero(input_grid[:, workspace_column] == template)
        if rows.size == 0:
            footprints.append(None)
        else:
            top, bottom = int(rows[0]), int(rows[-1])
            assert rows.size == bottom - top + 1
            footprints.append((top, bottom))

    rectangles = []
    offset = 0
    while offset < len(footprints):
        footprint = footprints[offset]
        if footprint is None:
            offset += 1
            continue
        end = offset + 1
        while end < len(footprints) and footprints[end] == footprint:
            end += 1
        rectangles.append((
            footprint[0], footprint[1],
            workspace_start + offset, workspace_start + end - 1,
        ))
        offset = end
    assert len(rectangles) == len(key_colors)

    reconstructed = np.full(input_grid.shape, background, dtype=np.int16)
    expected = np.full(input_grid.shape, background, dtype=np.int16)
    for index, color in enumerate(key_colors):
        reconstructed[:, 1 + 2 * index] = color
    for index, (top, bottom, left, right) in enumerate(rectangles):
        assert bottom - top + 1 < height
        reconstructed[top:bottom + 1, left:right + 1] = template
        expected[top:bottom + 1, left:right + 1] = key_colors[index]
    assert np.array_equal(input_grid, reconstructed)
    assert np.array_equal(output_grid, expected)

    relations = []
    gaps = []
    for previous, following in zip(rectangles, rectangles[1:]):
        gap = following[2] - previous[3] - 1
        assert gap >= 0
        gaps.append(gap)
        if gap:
            relations.append("gap")
        elif max(previous[0], following[0]) <= min(previous[1], following[1]):
            assert previous[:2] != following[:2]
            relations.append("edge")
        else:
            assert previous[1] + 1 == following[0] or following[1] + 1 == previous[0]
            relations.append("corner")

    return {
        "background": background,
        "template": template,
        "key_colors": tuple(key_colors),
        "workspace_start": workspace_start,
        "rectangles": tuple(rectangles),
        "relations": tuple(relations),
        "gaps": tuple(gaps),
        "shape": input_grid.shape,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    signatures = set()
    train_info, test_info = [], []
    for split, destination in (("train", train_info), ("test", test_info)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            assert not np.array_equal(input_grid, output_grid)
            signature = _grid_signature(input_grid)
            assert signature not in signatures
            signatures.add(signature)
            assert not _is_exact_source_pair(input_grid, output_grid)
            info = _analyze_pair(pair)
            assert 5 <= info["shape"][0] <= 20
            destination.append(info)

    assert len({info["key_colors"] for info in train_info}) >= 2
    assert "edge" in {relation for info in train_info for relation in info["relations"]}
    assert "corner" in {relation for info in train_info for relation in info["relations"]}
    for role in ("background", "template"):
        train_values = {info[role] for info in train_info}
        test_values = {info[role] for info in test_info}
        if len(test_values | train_values) > 1:
            assert len(train_values) > 1
    if any(
        len(info["key_colors"]) >= 6 and all(gap == 0 for gap in info["gaps"])
        for info in test_info
    ):
        assert any(
            sum(gap == 0 for gap in info["gaps"]) >= 2
            for info in train_info
        )
    assert len({(info["shape"], info["rectangles"]) for info in train_info}) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    train_specs, test_specs = _make_pair_specs(
        rng, parameters, train_count, test_count
    )
    puzzle = {"train": [], "test": []}
    used_inputs = set()

    for split, specifications in (("train", train_specs), ("test", test_specs)):
        regime = "source_train" if split == "train" else "target_test"
        for index, specification in enumerate(specifications):
            for _ in range(128):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                input_grid, output_grid = generate_pair(
                    pair_rng, regime, parameters, specification, index
                )
                signature = _grid_signature(input_grid)
                if signature in used_inputs or _is_exact_source_pair(input_grid, output_grid):
                    continue
                used_inputs.add(signature)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise RuntimeError("could not generate a unique non-source pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a["input"], b["input"])
            and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    key_lengths, shapes, layouts, role_pairs = set(), set(), set(), set()
    all_touch_tests = long_edge_tests = dense_six_tests = 0
    mixed_tests = separated_tests = novel_color_puzzles = 0
    source_like_puzzles = reused_order_puzzles = 0
    role_change_puzzles = stable_puzzles = 0
    taller_blocks = wider_blocks = thin_blocks = broad_gaps = wide_left_margins = 0
    equal_or_short_tests = shorter_tests = large_train = 0
    within_puzzle_count_changes = within_puzzle_width_changes = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
        roles = {(info["background"], info["template"]) for info in train_info}
        role_change_puzzles += len(roles) > 1
        stable_puzzles += len(roles) == 1
        all_info = train_info + test_info
        within_puzzle_count_changes += len({len(info["key_colors"]) for info in all_info}) > 1
        within_puzzle_width_changes += len({info["shape"][1] for info in all_info}) > 1
        large_train += any(len(info["key_colors"]) >= 6 for info in train_info)
        shorter_tests += any(
            len(info["key_colors"]) < max(len(train["key_colors"]) for train in train_info)
            for info in test_info
        )
        equal_or_short_tests += any(
            len(info["key_colors"]) <= max(len(train["key_colors"]) for train in train_info)
            for info in test_info
        )
        training_colors = {color for info in train_info for color in info["key_colors"]}
        novel_color_puzzles += any(
            set(info["key_colors"]) - training_colors for info in test_info
        )
        source_like_puzzles += (
            len(train_info) >= 3
            and [len(info["key_colors"]) for info in train_info[:3]] == [3, 3, 4]
            and all(len(info["key_colors"]) == 6 for info in test_info)
            and all(info["shape"][1] <= 20 for info in train_info)
            and len(roles) == 1
        )
        reused_order_puzzles += any(
            first["key_colors"] == second["key_colors"]
            and first["rectangles"] != second["rectangles"]
            for index, first in enumerate(all_info)
            for second in all_info[index + 1:]
        )
        for info in test_info:
            all_touch_tests += all(gap == 0 for gap in info["gaps"])
            dense_six_tests += (
                len(info["key_colors"]) == 6
                and all(gap == 0 for gap in info["gaps"])
            )
            mixed_tests += any(gap == 0 for gap in info["gaps"]) and any(
                gap > 0 for gap in info["gaps"]
            )
            separated_tests += sum(gap > 0 for gap in info["gaps"]) >= max(
                1, len(info["gaps"]) - 1
            )
            current_run = longest_run = 0
            for relation in info["relations"]:
                current_run = current_run + 1 if relation == "edge" else 0
                longest_run = max(longest_run, current_run)
            long_edge_tests += longest_run >= 3
        for info in train_info + test_info:
            count = len(info["key_colors"])
            key_lengths.add(count)
            shapes.add(info["shape"])
            layouts.add((info["shape"], info["rectangles"]))
            role_pairs.add((info["background"], info["template"]))
            for top, bottom, left, right in info["rectangles"]:
                taller_blocks += bottom - top + 1 >= 6
                wider_blocks += right - left + 1 >= 5
                thin_blocks += bottom == top and right - left + 1 >= 2
            broad_gaps += any(gap >= 3 for gap in info["gaps"])
            wide_left_margins += info["rectangles"][0][2] - info["workspace_start"] >= 3

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert {2, 3, 4, 5, 6, 7, 8} <= key_lengths
    assert len(shapes) >= 35 and len(layouts) >= 180
    assert len(role_pairs) >= 15
    assert all_touch_tests >= 8 and long_edge_tests >= 2
    assert dense_six_tests >= 3 and mixed_tests >= 5 and separated_tests >= 5
    assert role_change_puzzles >= 8 and stable_puzzles >= 5
    assert taller_blocks >= 20 and wider_blocks >= 10 and thin_blocks >= 3
    assert broad_gaps >= 8 and wide_left_margins >= 5
    assert equal_or_short_tests >= 10 and shorter_tests >= 8 and large_train >= 8
    assert novel_color_puzzles >= 2 and source_like_puzzles >= 2
    assert reused_order_puzzles >= 2
    assert within_puzzle_count_changes >= 30 and within_puzzle_width_changes >= 30

    for seed, train_count, test_count in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 7, 3),
        (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None),
    ):
        puzzle = generate_puzzle(seed, num_train=train_count, num_test=test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    assert _puzzles_equal(
        generate_puzzle(987654, num_train=8, num_test=3),
        generate_puzzle(987654, num_train=8, num_test=3),
    )
    assert not _puzzles_equal(generate_puzzle(3000), generate_puzzle(3001))
    for request in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 2.5, "num_test": 1},
        {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(4000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request did not fail: {request}")
