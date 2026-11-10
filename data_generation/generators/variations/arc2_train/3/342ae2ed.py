import numpy as np


_SOURCE_INPUT_KEYS = {
    tuple(
        """7777997777777777
7777997777777333
7777777777777333
7777777777777333
7777777799777777
7777777799777777
7711177777777744
7711177333777744
7711177333777777
7777777333777777
7777777777777777
7777777777777777
7777777777777777
7777777771117777
7777774471117777
7777774471117777""".splitlines()
    ),
    tuple(
        """7777777777777777
7777777777777777
7788877777774477
7788877777774477
7788877777777777
7777777777777777
7777777777777777
7777777777777777
7777777777777777
7997744777777777
7997744777777777
7777777777788877
7777777777788877
7777777777788877
7777779977777777
7777779977777777""".splitlines()
    ),
    tuple(
        """0077777777777777
0077777799977777
7777777799977777
7777777799977777
7777007777777777
7777007777777777
7777777777777755
7799977337777755
7799977337777777
7799977777777777
7777777777777777
7777777777777777
7777777777773377
7777777557773377
7777777557777777
7777777777777777""".splitlines()
    ),
    tuple(
        """7777777774477777
7777777774477777
7777777777777777
7777777777117777
7777744777117777
7766744777777777
7766777788877777
7777777788877711
7777777788877711
7777777777777777
7777777777777337
7777777766777337
7777777766777777
7888777777777777
7888777773377777
7888777773377777""".splitlines()
    ),
}

def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _check_optional_count(value, name: str, low: int, high: int):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed_value: int,
    num_train,
    num_test,
) -> tuple[int, int]:
    train_count = _check_optional_count(num_train, "num_train", 2, 10)
    test_count = _check_optional_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed_value % 4
        number_of_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed_value // 4) % number_of_choices
        )
    elif train_count is None:
        lowest_train = max(2, test_count + 1)
        number_of_choices = 11 - lowest_train
        train_count = lowest_train + seed_value % number_of_choices
    elif test_count is None:
        largest_test = min(4, train_count - 1)
        test_count = 1 + seed_value % largest_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_dimensions(rng: np.random.Generator, pair_count: int) -> tuple[int, int]:
    low = max(7, 4 + 2 * pair_count)
    mode = int(rng.choice(4, p=[0.34, 0.16, 0.16, 0.34]))
    compact_high = min(30, low + 7)
    if mode == 0:
        return (int(rng.integers(low, compact_high + 1)),
                int(rng.integers(low, compact_high + 1)))
    if mode == 1 and low + 5 <= 30:
        return (int(rng.integers(low, compact_high + 1)),
                int(rng.integers(low + 5, 31)))
    if mode == 2 and low + 5 <= 30:
        return (int(rng.integers(low + 5, 31)),
                int(rng.integers(low, compact_high + 1)))
    return (int(rng.integers(low, 31)), int(rng.integers(low, 31)))


def _sample_puzzle_parameters(rng: np.random.Generator, largest_count: int) -> dict:
    # Background and canvas are local visual properties. Some puzzles repeat
    # one or both, while others change them between demonstrations.
    return {
        "background": int(rng.integers(0, 10)),
        "shape": _sample_dimensions(rng, largest_count),
        "same_background": bool(rng.random() < 0.28),
        "same_shape": bool(rng.random() < 0.28),
    }


def _sample_scene_counts(rng: np.random.Generator, train_count: int,
                         test_count: int) -> tuple[list[int], list[int]]:
    train_values = np.arange(1, 7)
    test_values = np.arange(1, 10)
    train_weights = [0.11, 0.15, 0.26, 0.23, 0.17, 0.08]
    test_weights = [0.06, 0.08, 0.13, 0.17, 0.25, 0.15, 0.08, 0.05, 0.03]
    train = [int(x) for x in rng.choice(train_values, train_count, p=train_weights)]
    test = [int(x) for x in rng.choice(test_values, test_count, p=test_weights)]
    if max(train) == 1:
        train[int(rng.integers(train_count))] = 2
    return train, test


def _sample_sizes(rng: np.random.Generator, count: int, height: int,
                  width: int) -> list[int]:
    maximum = min(5, max(1, (min(height, width) - 1) // 2))
    weights = np.array([0.10, 0.46, 0.29, 0.11, 0.04])[:maximum]
    if count >= 7:
        weights *= np.array([2.0, 1.4, 0.7, 0.35, 0.2])[:maximum]
    weights /= weights.sum()
    sizes = [int(x) for x in rng.choice(np.arange(1, maximum + 1),
                                        count, p=weights)]
    return sizes


def _sample_gap(rng: np.random.Generator, size: int, height: int,
                width: int, count: int) -> int:
    maximum = min(22, min(height, width) - 2 * size)
    if maximum < 1:
        raise ValueError("canvas too small for a diagonal pair")
    draw = rng.random()
    if draw < 0.31:
        lo, hi = 1, min(2, maximum)
    elif draw < (0.88 if count >= 7 else 0.76):
        lo, hi = min(3, maximum), min(8, maximum)
    else:
        lo, hi = min(9, maximum), maximum
    return int(rng.integers(lo, hi + 1))


def _candidate_layouts(height: int, width: int, size: int, slope: int,
                       gap: int) -> list[tuple]:
    extent = 2 * size + gap
    if extent > min(height, width):
        return []
    displacement = size + gap
    layouts = []
    for row in range(height - extent + 1):
        columns = (range(width - extent + 1) if slope == 1
                   else range(displacement, width - size + 1))
        for column in columns:
            other_row = row + displacement
            other_column = column + slope * displacement
            bridge = tuple((row + size + step,
                            column + size + step if slope == 1
                            else column - 1 - step)
                           for step in range(gap))
            square_mask = 0
            for top, left in ((row, column), (other_row, other_column)):
                for rr in range(top, top + size):
                    for cc in range(left, left + size):
                        square_mask |= 1 << (rr * width + cc)
            full_mask = square_mask
            for rr, cc in bridge:
                full_mask |= 1 << (rr * width + cc)
            layouts.append((row, column, other_row, other_column,
                            bridge, full_mask, square_mask))
    return layouts


def _alternative_connections(layout: tuple, size: int, gap: int,
                             placed: list[tuple], square_mask: int,
                             width: int) -> int:
    opportunities = 0
    endpoints = ((layout[0], layout[1]), (layout[2], layout[3]))
    for old_layout, old_size, old_gap in placed:
        if old_size != size:
            continue
        for top1, left1 in endpoints:
            for top2, left2 in ((old_layout[0], old_layout[1]),
                                (old_layout[2], old_layout[3])):
                if top1 > top2:
                    top1a, left1a, top2a, left2a = top2, left2, top1, left1
                else:
                    top1a, left1a, top2a, left2a = top1, left1, top2, left2
                displacement = top2a - top1a
                if displacement <= size or abs(left2a - left1a) != displacement:
                    continue
                alternative_gap = displacement - size
                if alternative_gap >= max(gap, old_gap):
                    continue
                slope = 1 if left2a > left1a else -1
                cells = ((top1a + size + step,
                          left1a + size + step if slope == 1
                          else left1a - 1 - step)
                         for step in range(alternative_gap))
                if all(square_mask & (1 << (rr * width + cc)) == 0
                       for rr, cc in cells):
                    opportunities += 1
    return opportunities


def _choose_layouts(rng: np.random.Generator, height: int, width: int,
                    designs: list[tuple[int, int, int]],
                    seek_decoy: bool, cluster: bool) -> list[tuple] | None:
    count = len(designs)
    candidate_lists = [_candidate_layouts(height, width, *design)
                       for design in designs]
    if any(not candidates for candidates in candidate_lists):
        return None
    for _ in range(8):
        order = list(range(count))
        rng.shuffle(order)
        order.sort(key=lambda i: 2 * designs[i][0] + designs[i][2],
                   reverse=True)
        chosen = [None] * count
        used_mask = 0
        square_mask = 0
        placed = []
        bounds = None
        for index in order:
            size, slope, gap = designs[index]
            candidates = candidate_lists[index]
            eligible = []
            for raw in rng.permutation(len(candidates)):
                candidate = candidates[int(raw)]
                if candidate[5] & used_mask:
                    continue
                eligible.append(candidate)
                if len(eligible) == 48:
                    break
            if not eligible:
                break
            if seek_decoy and placed:
                scores = [_alternative_connections(candidate, size, gap,
                          placed, square_mask | candidate[6], width)
                          for candidate in eligible]
                best = max(scores)
                if best:
                    eligible = [candidate for candidate, score in
                                zip(eligible, scores) if score == best]
            if cluster and bounds is not None and len(eligible) > 1:
                top, left, bottom, right = bounds
                areas = []
                for candidate in eligible:
                    n_top = min(top, candidate[0])
                    n_left = min(left, candidate[1], candidate[3])
                    n_bottom = max(bottom, candidate[2] + size - 1)
                    n_right = max(right, candidate[1] + size - 1,
                                  candidate[3] + size - 1)
                    areas.append((n_bottom - n_top + 1) *
                                 (n_right - n_left + 1))
                cutoff = sorted(areas)[max(0, len(areas) // 4)]
                eligible = [candidate for candidate, area in
                            zip(eligible, areas) if area <= cutoff]
            choice = eligible[int(rng.integers(len(eligible)))]
            chosen[index] = choice
            used_mask |= choice[5]
            square_mask |= choice[6]
            placed.append((choice, size, gap))
            top = choice[0]
            left = min(choice[1], choice[3])
            bottom = choice[2] + size - 1
            right = max(choice[1], choice[3]) + size - 1
            if bounds is None:
                bounds = (top, left, bottom, right)
            else:
                bounds = (min(bounds[0], top), min(bounds[1], left),
                          max(bounds[2], bottom), max(bounds[3], right))
        if all(layout is not None for layout in chosen):
            return chosen
    return None


def _trim_margins(rng: np.random.Generator, input_grid: np.ndarray,
                  output_grid: np.ndarray, background: int) -> tuple:
    rows, columns = np.where(input_grid != background)
    top = max(0, int(rows.min()) - int(rng.integers(0, 6)))
    left = max(0, int(columns.min()) - int(rng.integers(0, 6)))
    bottom = min(input_grid.shape[0], int(rows.max()) + 1 +
                 int(rng.integers(0, 6)))
    right = min(input_grid.shape[1], int(columns.max()) + 1 +
                int(rng.integers(0, 6)))
    trimmed_input = input_grid[top:bottom, left:right].copy()
    trimmed_output = output_grid[top:bottom, left:right].copy()
    counts = np.bincount(trimmed_input.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) == 1 and \
            int(np.argmax(counts)) == background:
        return trimmed_input, trimmed_output
    return input_grid, output_grid


def _generate_pair(rng: np.random.Generator, regime: str,
                   puzzle_parameters: dict, object_count: int) -> tuple:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    if not 1 <= object_count <= 9:
        raise ValueError("pair count must be 1 through 9")
    for attempt in range(50):
        height, width = (puzzle_parameters["shape"] if
                         puzzle_parameters["same_shape"] else
                         _sample_dimensions(rng, object_count))
        background = (puzzle_parameters["background"] if
                      puzzle_parameters["same_background"] else
                      int(rng.integers(0, 10)))
        sizes = _sample_sizes(rng, object_count, height, width)
        slopes = [1 if rng.random() < 0.5 else -1
                  for _ in range(object_count)]
        gaps = [_sample_gap(rng, size, height, width, object_count)
                for size in sizes]
        designs = list(zip(sizes, slopes, gaps))
        layouts = _choose_layouts(rng, height, width, designs,
                                  seek_decoy=rng.random() < 0.47,
                                  cluster=rng.random() < 0.43)
        if layouts is None:
            continue
        colors = [int(x) for x in rng.permutation(
            [color for color in range(10) if color != background]
        )[:object_count]]
        input_grid = np.full((height, width), background, dtype=np.int8)
        for color, size, layout in zip(colors, sizes, layouts):
            row, column, other_row, other_column = layout[:4]
            input_grid[row:row + size, column:column + size] = color
            input_grid[other_row:other_row + size,
                       other_column:other_column + size] = color
        counts = np.bincount(input_grid.ravel(), minlength=10)
        if np.count_nonzero(counts == counts.max()) != 1 or \
                int(np.argmax(counts)) != background:
            continue
        output_grid = input_grid.copy()
        for color, layout in zip(colors, layouts):
            for row, column in layout[4]:
                output_grid[row, column] = color
        if not puzzle_parameters["same_shape"]:
            input_grid, output_grid = _trim_margins(
                rng, input_grid, output_grid, background)
        return input_grid, output_grid
    # Every legal count has ample room in this deterministic fallback.
    height = width = 30
    background = (puzzle_parameters["background"] if
                  puzzle_parameters["same_background"] else
                  int(rng.integers(0, 10)))
    designs = [(2, 1 if i % 2 else -1, 1) for i in range(object_count)]
    layouts = _choose_layouts(rng, height, width, designs, False, False)
    if layouts is None:
        raise ValueError("could not place square pairs")
    colors = [int(x) for x in rng.permutation(
        [color for color in range(10) if color != background]
    )[:object_count]]
    input_grid = np.full((height, width), background, dtype=np.int8)
    for color, layout in zip(colors, layouts):
        for row, column in ((layout[0], layout[1]),
                            (layout[2], layout[3])):
            input_grid[row:row + 2, column:column + 2] = color
    output_grid = input_grid.copy()
    for color, layout in zip(colors, layouts):
        for row, column in layout[4]:
            output_grid[row, column] = color
    return input_grid, output_grid


def _canonical_grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(value) for value in grid.ravel()))


def _row_strings(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _components_for_color(grid: np.ndarray, color: int) -> list[set]:
    unseen = {(int(row), int(column)) for row, column in np.argwhere(grid == color)}
    components = []
    while unseen:
        start = unseen.pop()
        component = {start}
        stack = [start]
        while stack:
            row, column = stack.pop()
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _describe_square(grid: np.ndarray, color: int, component: set) -> tuple:
    rows = [point[0] for point in component]
    columns = [point[1] for point in component]
    top, bottom = min(rows), max(rows)
    left, right = min(columns), max(columns)
    size = bottom - top + 1
    assert size == right - left + 1
    assert len(component) == size * size
    assert np.all(grid[top:bottom + 1, left:right + 1] == color)
    return top, left, size


def _infer_background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 2
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(colors[int(np.argmax(counts))])


def _inspect_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    background = _infer_background(input_grid)
    colors = [int(color) for color in np.unique(input_grid)
              if int(color) != background]
    assert 1 <= len(colors) <= 9
    expected = input_grid.copy()
    bridge_cells = set()
    objects = []
    for color in colors:
        components = _components_for_color(input_grid, color)
        assert len(components) == 2
        first, second = sorted(_describe_square(input_grid, color, component)
                               for component in components)
        row, column, size = first
        other_row, other_column, other_size = second
        assert size == other_size
        displacement = other_row - row
        assert displacement > size
        assert abs(other_column - column) == displacement
        slope = 1 if other_column > column else -1
        gap = displacement - size
        for step in range(gap):
            rr = row + size + step
            cc = (column + size + step if slope == 1
                  else column - 1 - step)
            assert 0 <= rr < input_grid.shape[0]
            assert 0 <= cc < input_grid.shape[1]
            assert int(input_grid[rr, cc]) == background
            assert (rr, cc) not in bridge_cells
            bridge_cells.add((rr, cc))
            expected[rr, cc] = color
        objects.append({"color": color, "size": size, "slope": slope,
                        "gap": gap, "top_lefts": ((row, column),
                                                   (other_row, other_column))})
    assert np.array_equal(expected, output_grid)
    assert int(np.count_nonzero(input_grid != output_grid)) == len(bridge_cells)
    assert set(int(x) for x in np.unique(input_grid)) == \
        set(int(x) for x in np.unique(output_grid))
    return {"background": background, "shape": tuple(input_grid.shape),
            "num_pairs": len(colors), "sizes": {x["size"] for x in objects},
            "slopes": {x["slope"] for x in objects},
            "changed_cells": len(bridge_cells), "objects": objects,
            "non_background_cells": int(np.count_nonzero(input_grid != background))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    train_information = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9
            key = _canonical_grid_key(pair["input"])
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert _row_strings(pair["input"]) not in _SOURCE_INPUT_KEYS
            information = _inspect_pair(pair["input"], pair["output"])
            assert information["changed_cells"] >= 1
            if split == "train":
                train_information.append(information)
    # The only learned convention is to connect same-colored square corners.
    # At least one demonstration has multiple colors, making color pairing
    # explicit. Canvas size and background color are visible in each input.
    assert any(item["num_pairs"] >= 2 for item in train_information)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    train_scene_counts, test_scene_counts = _sample_scene_counts(
        rng, train_count, test_count)
    parameters = _sample_puzzle_parameters(
        rng, max(train_scene_counts + test_scene_counts))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, counts in (("train", "source_train", train_scene_counts),
                                  ("test", "target_test", test_scene_counts)):
        for count in counts:
            for _ in range(30):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                input_grid, output_grid = _generate_pair(
                    pair_rng, regime, parameters, count)
                key = _canonical_grid_key(input_grid)
                if key not in seen and _row_strings(input_grid) not in _SOURCE_INPUT_KEYS:
                    seen.add(key)
                    puzzle[split].append({"input": input_grid,
                                          "output": output_grid})
                    break
            else:
                raise ValueError("could not generate unique grid")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for left, right in zip(first[split], second[split]):
            if not np.array_equal(left["input"], right["input"]) or \
                    not np.array_equal(left["output"], right["output"]):
                return False
    return True


def _review_scene_features(input_grid: np.ndarray, information: dict) -> tuple:
    objects = information["objects"]
    squares = [(row, column, item["size"], item["color"], item["gap"])
               for item in objects for row, column in item["top_lefts"]]
    background = information["background"]
    alternatives = nearer = touching = crossings = 0
    for first_index in range(len(squares)):
        for second_index in range(first_index + 1, len(squares)):
            first, second = squares[first_index], squares[second_index]
            r1, c1, s1, color1, actual_gap1 = first
            r2, c2, s2, color2, actual_gap2 = second
            if color1 == color2:
                continue
            horizontal_touch = (r1 < r2 + s2 and r2 < r1 + s1 and
                                (c1 + s1 == c2 or c2 + s2 == c1))
            vertical_touch = (c1 < c2 + s2 and c2 < c1 + s1 and
                              (r1 + s1 == r2 or r2 + s2 == r1))
            touching += horizontal_touch or vertical_touch
            if s1 != s2:
                continue
            if r1 > r2:
                r1, c1, r2, c2 = r2, c2, r1, c1
            displacement = r2 - r1
            if displacement <= s1 or abs(c2 - c1) != displacement:
                continue
            slope = 1 if c2 > c1 else -1
            gap = displacement - s1
            if all(int(input_grid[r1 + s1 + step,
                                  c1 + s1 + step if slope == 1
                                  else c1 - 1 - step]) == background
                   for step in range(gap)):
                alternatives += 1
                nearer += gap < max(actual_gap1, actual_gap2)
    for first_index in range(len(objects)):
        for second_index in range(first_index + 1, len(objects)):
            first, second = objects[first_index], objects[second_index]
            if first["slope"] == second["slope"] or \
                    min(first["gap"], second["gap"]) < 2:
                continue
            plus = first if first["slope"] == 1 else second
            minus = second if first["slope"] == 1 else first
            pr, pc = plus["top_lefts"][0]
            mr, mc = minus["top_lefts"][0]
            pr += plus["size"]
            pc += plus["size"]
            mr += minus["size"]
            mc -= 1
            twice_crossing_row = (pr - pc) + (mr + mc)
            if twice_crossing_row % 2 and \
                    2 * pr < twice_crossing_row < 2 * (pr + plus["gap"] - 1) and \
                    2 * mr < twice_crossing_row < 2 * (mr + minus["gap"] - 1):
                crossings += 1
    return alternatives, nearer, touching, crossings


def self_test() -> None:
    explicit = generate_puzzle(123456, 6, 3)
    assert _puzzles_equal(explicit, generate_puzzle(123456, 6, 3))
    assert not _puzzles_equal(explicit, generate_puzzle(123457, 6, 3))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1),
                              (1003, 10, 4), (2001, None, 1),
                              (2002, None, 4), (2003, 2, None),
                              (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
    train_counts, test_counts = set(), set()
    backgrounds, shapes, pair_counts, sizes, gaps = set(), set(), set(), set(), set()
    slope_sets, size_compositions, layout_fingerprints = set(), set(), set()
    varying_backgrounds = varying_shapes = 0
    test_fewer = all_parallel = 0
    same_background = same_shape = 0
    alternative_scenes = nearer_scenes = multiple_alternatives = 0
    touching_scenes = crossing_scenes = combined_variations = 0
    short_scenes = thirty_scenes = rectangular_scenes = 0
    lower_test_occupancy = familiar_increase = overlapping_counts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pairs = [pair for split in ("train", "test") for pair in puzzle[split]]
        infos = [_inspect_pair(pair["input"], pair["output"])
                 for pair in pairs]
        train_infos = infos[:len(puzzle["train"])]
        test_infos = infos[len(puzzle["train"]):]
        varying_backgrounds += len({x["background"] for x in infos}) > 1
        varying_shapes += len({x["shape"] for x in infos}) > 1
        same_background += len({x["background"] for x in infos}) == 1
        same_shape += len({x["shape"] for x in infos}) == 1
        test_fewer += min(x["num_pairs"] for x in test_infos) < \
            max(x["num_pairs"] for x in train_infos)
        lower_test_occupancy += any(
            x["non_background_cells"] <= max(
                y["non_background_cells"] for y in train_infos)
            for x in test_infos)
        familiar_increase += (any(x["num_pairs"] == 5 for x in test_infos)
                              and any(x["num_pairs"] in (3, 4)
                                      for x in train_infos))
        overlapping_counts += any(x["num_pairs"] == y["num_pairs"]
                                  for x in train_infos for y in test_infos)
        for pair, info in zip(pairs, infos):
            backgrounds.add(info["background"])
            shapes.add(info["shape"])
            pair_counts.add(info["num_pairs"])
            slope_sets.add(tuple(sorted(info["slopes"])))
            all_parallel += len(info["slopes"]) == 1
            short_scenes += min(info["shape"]) <= 10
            thirty_scenes += 30 in info["shape"]
            rectangular_scenes += max(info["shape"]) / min(info["shape"]) >= 1.5
            objects = info["objects"]
            size_compositions.add(tuple(sorted(x["size"] for x in objects)))
            sizes.update(x["size"] for x in objects)
            gaps.update(x["gap"] for x in objects)
            combined_variations += (len(info["sizes"]) >= 3 and
                                    len(info["slopes"]) == 2 and
                                    max(x["gap"] for x in objects) >= 12)
            alternatives, nearer, touching, crossings = _review_scene_features(
                pair["input"], info)
            alternative_scenes += alternatives > 0
            nearer_scenes += nearer > 0
            multiple_alternatives += alternatives >= 2
            touching_scenes += touching > 0
            crossing_scenes += crossings > 0
            layout_fingerprints.add((info["shape"], tuple(sorted(
                (x["size"], x["slope"], x["gap"], x["top_lefts"])
                for x in objects))))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) == 10 and len(shapes) >= 40
    assert pair_counts == set(range(1, 10))
    assert sizes >= {1, 2, 3, 4}
    assert 1 in gaps and max(gaps) >= 12
    assert (-1,) in slope_sets and (1,) in slope_sets and (-1, 1) in slope_sets
    assert len(size_compositions) >= 25 and len(layout_fingerprints) >= 100
    assert varying_backgrounds >= 10 and varying_shapes >= 10
    assert test_fewer >= 5 and all_parallel >= 20
    assert same_background >= 3 and same_shape >= 3
    assert (2, 2, 2, 3) in size_compositions
    assert (2, 2, 2, 2, 3) in size_compositions
    assert sum(len(set(composition)) == 1 for composition in size_compositions) >= 4
    assert short_scenes >= 10 and thirty_scenes >= 1
    assert rectangular_scenes >= 10 and combined_variations >= 10
    assert alternative_scenes >= 20 and nearer_scenes >= 15
    assert multiple_alternatives >= 5
    assert touching_scenes >= 20 and crossing_scenes >= 10
    assert lower_test_occupancy >= 5
    assert familiar_increase >= 5 and overlapping_counts >= 5
