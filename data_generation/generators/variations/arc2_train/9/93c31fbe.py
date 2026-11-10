import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "c65175aa337000acf0239c6428fb81aec97fd5c2a76f32696cc01a453406a79c",
    "8e3d834061a8164fc0007a85c86c02e88c37b264219e1665980f4b6ddcb6d55a",
    "e9bd2b5f010fb7244dbe7c54adb45319f6cb88e1f962898ac5040661da919cf1",
    "166901095f5f09f48cf29aafc7390eab234ec87b568bb8b017c285feb6945de5",
})

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validate_requested_count(name, value, low, high):
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


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_requested_count("num_train", num_train, 2, 10)
    num_test = _validate_requested_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees broad count diversity for
        # consecutive seeds while keeping every combination legal.
        seed_value = int(seed)
        num_test = 1 + seed_value % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed_value // 4) % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists")
        num_test = int(rng.integers(1, maximum_test + 1))

    if not (2 <= num_train <= 10):
        raise ValueError("num_train must be between 2 and 10")
    if not (1 <= num_test <= 4):
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _corner_cells(rect):
    top, bottom, left, right = rect
    return {
        (top, left),
        (top, left + 1),
        (top + 1, left),

        (top, right),
        (top, right - 1),
        (top + 1, right),

        (bottom, left),
        (bottom, left + 1),
        (bottom - 1, left),

        (bottom, right),
        (bottom, right - 1),
        (bottom - 1, right),
    }


def _four_connected_components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components = []

    for start_row, start_col in np.argwhere(mask):
        start_row = int(start_row)
        start_col = int(start_col)
        if seen[start_row, start_col]:
            continue

        stack = [(start_row, start_col)]
        seen[start_row, start_col] = True
        component = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for delta_row, delta_col in _NEIGHBORS:
                new_row = row + delta_row
                new_col = col + delta_col
                if (
                    0 <= new_row < height
                    and 0 <= new_col < width
                    and mask[new_row, new_col]
                    and not seen[new_row, new_col]
                ):
                    seen[new_row, new_col] = True
                    stack.append((new_row, new_col))

        components.append(component)

    return components


def _decode_structural_mask(mask):
    assert isinstance(mask, np.ndarray)
    assert mask.ndim == 2
    assert mask.dtype == np.bool_ or np.issubdtype(mask.dtype, np.bool_)
    assert np.any(mask)

    markers = []

    for component in _four_connected_components(mask):
        assert len(component) == 3
        cells = set(component)

        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]
        top = min(rows)
        bottom = max(rows)
        left = min(cols)
        right = max(cols)

        assert bottom == top + 1
        assert right == left + 1

        square = {
            (top, left),
            (top, right),
            (bottom, left),
            (bottom, right),
        }
        missing = square - cells
        assert len(missing) == 1
        missing_cell = next(iter(missing))

        if missing_cell == (bottom, right):
            marker = ("TL", top, left)
        elif missing_cell == (bottom, left):
            marker = ("TR", top, right)
        elif missing_cell == (top, right):
            marker = ("BL", bottom, left)
        elif missing_cell == (top, left):
            marker = ("BR", bottom, right)
        else:
            raise AssertionError("invalid L marker")

        markers.append(marker)

    assert len(markers) >= 4
    assert len(markers) % 4 == 0
    assert len(set(markers)) == len(markers)

    marker_set = set(markers)
    by_kind = {
        kind: [marker for marker in markers if marker[0] == kind]
        for kind in ("TL", "TR", "BL", "BR")
    }

    chamber_count = len(markers) // 4
    for kind in by_kind:
        assert len(by_kind[kind]) == chamber_count

    candidates = []

    for _, top, left in by_kind["TL"]:
        for _, other_top, right in by_kind["TR"]:
            if other_top != top or right <= left:
                continue

            for _, bottom, other_left in by_kind["BL"]:
                if other_left != left or bottom <= top:
                    continue

                if ("BR", bottom, right) not in marker_set:
                    continue

                height = bottom - top + 1
                width = right - left + 1
                if height >= 5 and width >= 5 and height != width:
                    candidates.append((top, bottom, left, right))

    assert len(candidates) == chamber_count
    assert len(set(candidates)) == chamber_count

    used_markers = []
    for top, bottom, left, right in candidates:
        used_markers.extend([
            ("TL", top, left),
            ("TR", top, right),
            ("BL", bottom, left),
            ("BR", bottom, right),
        ])

    assert len(set(used_markers)) == 4 * chamber_count
    assert set(used_markers) == marker_set

    rectangles = sorted(candidates)

    for index, first in enumerate(rectangles):
        first_top, first_bottom, first_left, first_right = first
        for second in rectangles[index + 1:]:
            second_top, second_bottom, second_left, second_right = second
            disjoint = (
                first_bottom < second_top
                or second_bottom < first_top
                or first_right < second_left
                or second_right < first_left
            )
            assert disjoint

    expected_cells = set()
    for rectangle in rectangles:
        cells = _corner_cells(rectangle)
        assert len(cells) == 12
        assert expected_cells.isdisjoint(cells)
        expected_cells.update(cells)

    actual_cells = {
        (int(row), int(col))
        for row, col in np.argwhere(mask)
    }
    assert actual_cells == expected_cells

    return rectangles


def _infer_roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3

    background_index = int(np.argmax(counts))
    background = int(values[background_index])
    assert int(counts[background_index]) > grid.size // 2

    other_colors = [
        int(value)
        for value in values
        if int(value) != background
    ]

    structural_candidates = []
    for color in other_colors:
        try:
            rectangles = _decode_structural_mask(grid == color)
        except AssertionError:
            continue
        structural_candidates.append((color, rectangles))

    assert len(structural_candidates) == 1
    structure, rectangles = structural_candidates[0]
    payload = next(color for color in other_colors if color != structure)

    return background, structure, payload, rectangles


def _solve_grid(input_grid, background, structure, payload):
    rectangles = _decode_structural_mask(input_grid == structure)
    output_grid = input_grid.copy()
    output_grid[output_grid == payload] = background

    payload_positions = [
        (int(row), int(col))
        for row, col in np.argwhere(input_grid == payload)
    ]

    for top, bottom, left, right in rectangles:
        height = bottom - top + 1
        width = right - left + 1

        for row, col in payload_positions:
            if not (top <= row <= bottom and left <= col <= right):
                continue

            output_grid[row, col] = payload

            if width > height:
                reflected_row = row
                reflected_col = left + right - col
            else:
                reflected_row = top + bottom - row
                reflected_col = col

            assert input_grid[reflected_row, reflected_col] == background
            output_grid[reflected_row, reflected_col] = payload

    return output_grid


def _sample_pair_shape(rng, chamber_count, regime):
    minimum = 18 if chamber_count <= 2 else (23 if chamber_count == 3 else 25)
    if regime == "target_test" and rng.random() < 0.45:
        minimum = max(minimum, 24)
    return (int(rng.integers(minimum, 31)),
            int(rng.integers(minimum, 31)))


def _candidate_rectangle(rng, height, width, chamber_count, orientation):
    short = int(rng.integers(5, 8))
    maximum_long = 14 if chamber_count <= 2 else (13 if chamber_count == 3 else 11)
    long = int(rng.integers(max(7, short + 1), maximum_long + 1))
    rect_height, rect_width = (short, long) if orientation == "H" else (long, short)
    if rect_height > height - 2 or rect_width > width - 2:
        return None
    top = int(rng.integers(1, height - rect_height))
    left = int(rng.integers(1, width - rect_width))
    return (top, top + rect_height - 1, left, left + rect_width - 1)


def _separated(first, second):
    a, b, c, d = first
    e, f, g, h = second
    return b + 1 < e or f + 1 < a or d + 1 < g or h + 1 < c


def _place_rectangles(rng, height, width, chamber_count, required_orientation):
    for _ in range(250):
        orientations = [required_orientation] if required_orientation else []
        orientations += ["H" if rng.random() < 0.5 else "V"
                         for _ in range(chamber_count - len(orientations))]
        rng.shuffle(orientations)
        rectangles = []
        for orientation in orientations:
            for _ in range(180):
                rect = _candidate_rectangle(rng, height, width,
                                            chamber_count, orientation)
                if rect is not None and all(_separated(rect, old) for old in rectangles):
                    rectangles.append(rect)
                    break
            else:
                break
        if len(rectangles) != chamber_count:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for rect in rectangles:
            for row, col in _corner_cells(rect):
                mask[row, col] = True
        try:
            decoded = _decode_structural_mask(mask)
        except AssertionError:
            continue
        if decoded == sorted(rectangles):
            return decoded
    raise RuntimeError("could not place unambiguous chambers")


def _sample_motif(rng, grid, rectangle, background, payload):
    top, bottom, left, right = rectangle
    horizontal = right - left > bottom - top
    sign = -1 if rng.random() < 0.5 else 1
    candidates = []
    for row in range(top + 1, bottom):
        for col in range(left + 1, right):
            delta = (2 * col - left - right) if horizontal else (2 * row - top - bottom)
            other = (row, left + right - col) if horizontal else (top + bottom - row, col)
            if delta * sign > 0 and grid[row, col] == background and grid[other] == background:
                candidates.append((row, col))
    assert len(candidates) >= 2
    upper = min(12, len(candidates))
    size = int(rng.integers(2, upper + 1))
    mode = int(rng.choice(4, p=[0.15, 0.25, 0.45, 0.15]))
    remaining = set(candidates)
    selected = []
    if mode == 0:
        # A connected stroke or branch.
        first = candidates[int(rng.integers(len(candidates)))]
        selected.append(first)
        remaining.remove(first)
        while len(selected) < size:
            frontier = [p for p in sorted(remaining)
                        if any((p[0] + dr, p[1] + dc) in selected
                               for dr, dc in _NEIGHBORS)]
            choices = frontier if frontier else sorted(remaining)
            chosen = choices[int(rng.integers(len(choices)))]
            selected.append(chosen)
            remaining.remove(chosen)
    elif mode == 1:
        # Two or three independently placed small groups.
        groups = min(int(rng.integers(2, 4)), size)
        for group in range(groups):
            choices = sorted(remaining)
            chosen = choices[int(rng.integers(len(choices)))]
            selected.append(chosen)
            remaining.remove(chosen)
        while len(selected) < size:
            frontier = [p for p in sorted(remaining)
                        if any((p[0] + dr, p[1] + dc) in selected
                               for dr, dc in _NEIGHBORS)]
            choices = frontier if frontier and rng.random() < 0.7 else sorted(remaining)
            chosen = choices[int(rng.integers(len(choices)))]
            selected.append(chosen)
            remaining.remove(chosen)
    elif mode == 2:
        # Widely scattered dots over the available half.
        indices = rng.choice(len(candidates), size=size, replace=False)
        selected = [candidates[int(i)] for i in indices]
    else:
        # A long row or column with occasional separated additions.
        first = candidates[int(rng.integers(len(candidates)))]
        selected = [first]
        remaining.remove(first)
        direction = _NEIGHBORS[int(rng.integers(4))]
        while len(selected) < size:
            end = selected[-1]
            next_cell = (end[0] + direction[0], end[1] + direction[1])
            if next_cell in remaining and rng.random() < 0.8:
                chosen = next_cell
            else:
                choices = sorted(remaining)
                chosen = choices[int(rng.integers(len(choices)))]
            selected.append(chosen)
            remaining.remove(chosen)
    for row, col in selected:
        grid[row, col] = payload


def _add_exterior_distractors(rng, grid, rectangles, background, payload,
                              require_noise):
    exterior = np.ones(grid.shape, dtype=bool)
    for top, bottom, left, right in rectangles:
        exterior[top:bottom + 1, left:right + 1] = False
    free = {(int(r), int(c)) for r, c in np.argwhere(exterior & (grid == background))}
    count = int(rng.integers(0, 19))
    if require_noise:
        count = max(1, count)
    chosen = []
    allow_clusters = rng.random() < 0.42
    while len(chosen) < count and free:
        remaining = count - len(chosen)
        cluster = allow_clusters and remaining >= 2 and rng.random() < 0.55
        near = [p for p in free if any(
            top - 3 <= p[0] <= bottom + 3 and left - 3 <= p[1] <= right + 3
            for top, bottom, left, right in rectangles)]
        options = sorted(near if near and rng.random() < 0.55 else free)
        start = options[int(rng.integers(len(options)))]
        chosen.append(start)
        free.remove(start)
        if cluster:
            group_size = min(remaining, int(rng.integers(2, 5)))
            group = [start]
            while len(group) < group_size:
                frontier = [p for p in sorted(free)
                            if any((p[0] + dr, p[1] + dc) in group
                                   for dr, dc in _NEIGHBORS)]
                if not frontier:
                    break
                point = frontier[int(rng.integers(len(frontier)))]
                chosen.append(point)
                group.append(point)
                free.remove(point)
    for row, col in chosen:
        grid[row, col] = payload


def _sample_puzzle_parameters(rng, num_train, num_test):
    colors = [int(c) for c in rng.permutation(10)]
    return {"base_background": colors[0], "base_payload": colors[1],
            "markers": colors[2:6], "other_colors": colors[6:],
            "role_variation": float(rng.choice(
                [0.0, 0.15, 0.35, 0.55], p=[0.2, 0.25, 0.35, 0.2]))}


def _sample_colors(rng, parameters, regime):
    markers = parameters["markers"]
    if regime == "source_train":
        structure = markers[int(rng.integers(0, 3))]
    else:
        structure = markers[int(rng.integers(0, 4))]
    background = parameters["base_background"]
    payload = parameters["base_payload"]
    if rng.random() < parameters["role_variation"]:
        background = parameters["other_colors"][int(rng.integers(4))]
    if rng.random() < parameters["role_variation"]:
        choices = [c for c in parameters["other_colors"] if c != background]
        payload = choices[int(rng.integers(len(choices)))]
    if payload == background:
        payload = parameters["base_payload"]
    assert len({background, structure, payload}) == 3
    return background, structure, payload


def _generate_pair(rng, regime, parameters, pair_index):
    if regime == "source_train":
        chamber_count = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.42, 0.28, 0.05]))
        required_orientation = "H" if pair_index == 0 else ("V" if pair_index == 1 else None)
    else:
        chamber_count = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.25, 0.35, 0.25]))
        required_orientation = None
    height, width = _sample_pair_shape(rng, chamber_count, regime)
    rectangles = _place_rectangles(rng, height, width, chamber_count,
                                   required_orientation)
    background, structure, payload = _sample_colors(rng, parameters, regime)
    input_grid = np.full((height, width), background, dtype=np.int8)
    for rect in rectangles:
        for row, col in _corner_cells(rect):
            input_grid[row, col] = structure
    for rect in rectangles:
        _sample_motif(rng, input_grid, rect, background, payload)
    _add_exterior_distractors(rng, input_grid, rectangles, background, payload,
                              regime == "source_train" and pair_index == 0)
    output_grid = _solve_grid(input_grid, background, structure, payload)
    return input_grid, output_grid


def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, regime, count in (("train", "source_train", num_train),
                                   ("test", "target_test", num_test)):
        for index in range(count):
            for _ in range(10):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                input_grid, output_grid = _generate_pair(pair_rng, regime,
                                                         parameters, index)
                key = _grid_key(input_grid)
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("duplicate inputs after retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background, structure, payload, rectangles = _infer_roles(input_grid)
    assert set(int(c) for c in np.unique(output_grid)) == {background, structure, payload}
    assert np.array_equal(input_grid == structure, output_grid == structure)
    assert np.count_nonzero(input_grid == structure) == 12 * len(rectangles)
    assert np.array_equal(output_grid, _solve_grid(input_grid, background, structure, payload))
    assert not np.array_equal(input_grid, output_grid)
    inside = np.zeros(input_grid.shape, dtype=bool)
    orientations = set()
    motif_sizes = []
    disconnected = []
    crossing = []
    long_sides = []
    signs = []
    for top, bottom, left, right in rectangles:
        horizontal = right - left > bottom - top
        orientations.add("H" if horizontal else "V")
        long_sides.append(max(bottom - top + 1, right - left + 1))
        crossing.append(top <= (input_grid.shape[0]-1)/2 <= bottom or
                        left <= (input_grid.shape[1]-1)/2 <= right)
        inside[top:bottom + 1, left:right + 1] = True
        local = input_grid[top:bottom + 1, left:right + 1] == payload
        positions = [(top + int(r), left + int(c)) for r, c in np.argwhere(local)]
        assert len(positions) >= 2
        motif_sizes.append(len(positions))
        disconnected.append(len(_four_connected_components(local)) > 1)
        end_signs = set()
        for row, col in positions:
            delta = 2 * col - left - right if horizontal else 2 * row - top - bottom
            assert delta != 0
            end_signs.add(np.sign(delta))
            mirror = (row, left + right - col) if horizontal else (top + bottom - row, col)
            assert input_grid[mirror] == background
        assert len(end_signs) == 1
        signs.append(int(next(iter(end_signs))))
    assert not np.any((output_grid == payload) & ~inside)
    noise = (input_grid == payload) & ~inside
    noise_sizes = [len(c) for c in _four_connected_components(noise)]
    assert np.all(output_grid[noise] == background)
    assert int(np.count_nonzero((input_grid == background) & (output_grid == payload))) == sum(motif_sizes)
    assert int(np.count_nonzero((input_grid == payload) & (output_grid == background))) == sum(noise_sizes)
    return {"background": background, "structure": structure, "payload": payload,
            "rectangles": rectangles, "orientations": orientations,
            "long_sides": long_sides, "crossing": crossing,
            "motif_sizes": motif_sizes, "disconnected": disconnected,
            "signs": signs, "noise_sizes": noise_sizes}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    information = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            digest = hashlib.sha256(pair["input"].astype(np.int8).tobytes()
                                    + b"|" + pair["output"].astype(np.int8).tobytes()).hexdigest()
            assert digest not in _SOURCE_PAIR_HASHES
            information[section].append(_inspect_pair(pair["input"], pair["output"]))
    demonstrated = set().union(*(item["orientations"] for item in information["train"]))
    required = set().union(*(item["orientations"] for item in information["test"]))
    assert demonstrated == {"H", "V"} and required <= demonstrated
    assert any(item["noise_sizes"] for item in information["train"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    first = generate_puzzle(17)
    assert _puzzles_equal(first, generate_puzzle(17))
    assert not _puzzles_equal(first, generate_puzzle(18))
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3),
                        (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        assert _puzzles_equal(puzzle, generate_puzzle(12345, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(40, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts = set(), set()
    layouts, shapes, palettes = set(), set(), set()
    chamber_counts, long_sides = set(), set()
    crossings = large_motifs = disconnected = clusters = no_noise = 0
    one_orientation = mixed_orientation = color_change = repeated_test_marker = new_test_marker = 0
    same_or_smaller_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {section: [_inspect_pair(p["input"], p["output"])
                           for p in puzzle[section]] for section in ("train", "test")}
        train_markers = {x["structure"] for x in infos["train"]}
        for item in infos["test"]:
            repeated_test_marker += item["structure"] in train_markers
            new_test_marker += item["structure"] not in train_markers
        all_pairs = puzzle["train"] + puzzle["test"]
        all_infos = infos["train"] + infos["test"]
        color_change += len({(x["background"], x["payload"]) for x in all_infos}) > 1
        maximum_train_area = max(p["input"].size for p in puzzle["train"])
        same_or_smaller_test += any(p["input"].size <= maximum_train_area for p in puzzle["test"])
        for pair, item in zip(all_pairs, all_infos):
            chamber_counts.add(len(item["rectangles"]))
            long_sides.update(item["long_sides"])
            crossings += sum(item["crossing"])
            large_motifs += sum(size >= 10 for size in item["motif_sizes"])
            disconnected += sum(item["disconnected"])
            clusters += any(size >= 3 for size in item["noise_sizes"])
            no_noise += not item["noise_sizes"]
            one_orientation += len(item["orientations"]) == 1
            mixed_orientation += len(item["orientations"]) == 2
            shapes.add(pair["input"].shape)
            palettes.add(tuple(sorted(int(x) for x in np.unique(pair["input"]))))
            layouts.add((pair["input"].shape, tuple(item["rectangles"])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 100 and len(palettes) >= 30
    assert chamber_counts == {1, 2, 3, 4}
    assert {7, 13, 14} <= long_sides
    assert crossings >= 20 and large_motifs >= 10 and disconnected >= 30
    assert clusters >= 20 and no_noise >= 5
    assert one_orientation >= 30 and mixed_orientation >= 10
    assert color_change >= 20 and repeated_test_marker >= 10 and new_test_marker >= 10
    assert same_or_smaller_test >= 15
