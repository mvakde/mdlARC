"""Generate colored copies of a damaged, shared D4-symmetric square design."""

import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "d03cbf792449ef900974e0ff5f1b50c7760d2abd93f583bef5e8ce8691827795",
    "d9d9050b50b1d149f4fe48f1e1f46c2ba458648b98fdecbbdd35e37dfa817ed7",
    "e3049e588d62f1fe320f6f52b9b6af9f2b7bdf16d8af128d80af94945f66e0e0",
    "90aa9819fcb361c5a7f4a7f68fe161b69eb1011684b05bc012286ae2e6baf9a6",
    "b16a64330869f6180ab8351425d4a53fb1cbfba6c8976a47a72c90830665e9eb",
}


def _validated_optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validated_optional_count("num_train", num_train, 2, 10)
    num_test = _validated_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _orbit(side, row, col):
    radius = side - 1
    dr, dc = 2 * row - radius, 2 * col - radius
    return frozenset(
        ((sign_r * a + radius) // 2, (sign_c * b + radius) // 2)
        for a, b in ((dr, dc), (dc, dr))
        for sign_r in (-1, 1)
        for sign_c in (-1, 1)
    )


def _all_orbits(side):
    seen = set()
    orbits = []
    for row in range(side):
        for col in range(side):
            if (row, col) not in seen:
                orbit = _orbit(side, row, col)
                orbits.append(orbit)
                seen.update(orbit)
    return orbits


def _closure(points, side):
    return set().union(*(_orbit(side, row, col) for row, col in points))


def _sample_template(rng, side, force_center=False):
    orbits = _all_orbits(side)
    boundary = [orbit for orbit in orbits if any(row == 0 for row, _ in orbit)]
    anchor = boundary[int(rng.integers(len(boundary)))]
    center = frozenset({(side // 2, side // 2)}) if side % 2 else None
    selected = [anchor]
    if center is not None and (force_center or rng.random() < 0.58):
        selected.append(center)
    remaining = [orbit for orbit in orbits if orbit not in selected]
    maximum = min(6, len(orbits))
    desired = int(rng.integers(max(2, len(selected)), maximum + 1))
    order = rng.permutation(len(remaining))
    selected.extend(remaining[int(i)] for i in order[:desired - len(selected)])
    template = set().union(*selected)
    assert _closure(template, side) == template
    assert {min(r for r, _ in template), max(r for r, _ in template)} == {0, side - 1}
    return template, selected


def _choose_slots(rng, count, rows, cols):
    from itertools import combinations

    all_slots = [(row, col) for row in range(rows) for col in range(cols)]
    valid = [
        slots for slots in combinations(all_slots, count)
        if len({r for r, _ in slots}) == rows and len({c for _, c in slots}) == cols
    ]
    choice = valid[int(rng.integers(len(valid)))]
    return [choice[int(i)] for i in rng.permutation(count)]


def _layout(rng, count, side):
    # Independent row and column gaps and local offsets break rigid slot layouts.
    candidates = []
    for rows in range(1, count + 1):
        for cols in range(1, count + 1):
            if max(rows, cols) <= count <= rows * cols and rows * cols <= count + 2:
                if rows * (side + 2) + rows - 1 <= 30 and cols * (side + 2) + cols - 1 <= 30:
                    candidates.append((rows, cols))
    rows, cols = candidates[int(rng.integers(len(candidates)))]
    slots = _choose_slots(rng, count, rows, cols)
    room = 2
    for _ in range(20):
        row_gaps = [int(rng.integers(1, 6)) for _ in range(rows - 1)]
        col_gaps = [int(rng.integers(1, 6)) for _ in range(cols - 1)]
        height_used = rows * (side + room) + sum(row_gaps)
        width_used = cols * (side + room) + sum(col_gaps)
        if height_used <= 30 and width_used <= 30:
            break
    else:
        row_gaps = [1] * (rows - 1)
        col_gaps = [1] * (cols - 1)
        height_used = rows * (side + room) + sum(row_gaps)
        width_used = cols * (side + room) + sum(col_gaps)
    height = int(rng.integers(max(11, height_used), min(30, height_used + 8) + 1))
    width = int(rng.integers(max(11, width_used), min(30, width_used + 8) + 1))
    top_margin = int(rng.integers(0, height - height_used + 1))
    left_margin = int(rng.integers(0, width - width_used + 1))
    row_starts = [top_margin]
    col_starts = [left_margin]
    for gap in row_gaps:
        row_starts.append(row_starts[-1] + side + room + gap)
    for gap in col_gaps:
        col_starts.append(col_starts[-1] + side + room + gap)
    positions = [
        (row_starts[r] + int(rng.integers(0, room + 1)),
         col_starts[c] + int(rng.integers(0, room + 1)))
        for r, c in slots
    ]
    return height, width, positions


def _select_palette(rng, count, previous):
    if previous is None:
        background = int(rng.integers(10))
    else:
        prior_background, prior_foreground = previous
        mode = rng.random()
        if mode < 0.24:
            background = prior_background
        elif mode < 0.56:
            background = int(rng.choice(prior_foreground))
        else:
            background = int(rng.integers(10))
    available = [color for color in range(10) if color != background]
    colors = [int(c) for c in rng.choice(available, size=count, replace=False)]
    if previous is not None and rng.random() < 0.36 and previous[0] != background:
        colors[int(rng.integers(count))] = previous[0]
        # Sampling without replacement might already have included the old background.
        if len(set(colors)) != count:
            colors = [previous[0]] + [c for c in colors if c != previous[0]]
            for color in available:
                if len(colors) == count:
                    break
                if color not in colors:
                    colors.append(color)
    return background, colors


def _damage_copies(rng, template, orbits, count, side, special):
    copies = [set(template) for _ in range(count)]
    reference = int(rng.integers(count))
    target = (reference + 1) % count
    center = (side // 2, side // 2)
    if special == "center" and center in template:
        copies[target].remove(center)
    elif special == "boundary":
        boundary_row = int(rng.choice([0, side - 1]))
        copies[target].difference_update({p for p in template if p[0] == boundary_row})
    elif special == "orbit":
        eligible = [orbit for orbit in orbits if orbit <= template and len(orbit) <= 4]
        if eligible:
            copies[target].difference_update(eligible[int(rng.integers(len(eligible)))])

    for index, surviving in enumerate(copies):
        if index == reference:
            budget = int(rng.integers(0, 3))
        else:
            budget = int(rng.integers(1, 7))
        budget = max(budget, len(template - surviving))
        budget = min(budget, max(1, len(template) - 4))
        shuffled = [tuple(p) for p in rng.permutation(np.asarray(sorted(surviving)))]
        for point in shuffled:
            if len(template - surviving) >= budget:
                break
            trial = surviving - {point}
            if index == reference:
                if _closure(trial, side) != template:
                    continue
                rows = [r for r, _ in trial]
                cols = [c for _, c in trial]
                if (min(rows), max(rows), min(cols), max(cols)) != (0, side - 1, 0, side - 1):
                    continue
            if len(trial) >= 4:
                surviving.remove(point)
    if all(copy == template for copy in copies):
        copies[target].remove(next(iter(template)))
    return copies


def _dominant_color(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(np.argmax(counts))


def _points(grid, color):
    return {(int(r), int(c)) for r, c in np.argwhere(grid == color)}


def _box(points):
    return (min(r for r, _ in points), max(r for r, _ in points),
            min(c for _, c in points), max(c for _, c in points))


def _solve_from_input(input_grid):
    """Find a template and copy alignments using the input alone."""
    background = _dominant_color(input_grid)
    colors = [int(c) for c in np.unique(input_grid) if c != background]
    points = {color: _points(input_grid, color) for color in colors}
    side = max(max(b - t + 1, r - l + 1) for t, b, l, r in map(_box, points.values()))
    height, width = input_grid.shape
    solutions = {}
    for reference in colors:
        top, bottom, left, right = _box(points[reference])
        if (bottom - top + 1, right - left + 1) != (side, side):
            continue
        local = {(r - top, c - left) for r, c in points[reference]}
        template = _closure(local, side)
        if any(not (0 <= r < side and 0 <= c < side) for r, c in template):
            continue
        placements = {}
        ambiguous_placement = False
        for color in colors:
            t, b, l, rr = _box(points[color])
            possible = []
            for row in range(max(0, b - side + 1), min(t, height - side) + 1):
                for col in range(max(0, rr - side + 1), min(l, width - side) + 1):
                    if all((r - row, c - col) in template for r, c in points[color]):
                        possible.append((row, col))
            if not possible:
                break
            if len(possible) > 1:
                ambiguous_placement = True
            placements[color] = possible[0]
        if len(placements) != len(colors):
            continue
        if ambiguous_placement:
            return None
        output = np.full(input_grid.shape, background, dtype=np.int8)
        for color, (row, col) in placements.items():
            for dr, dc in template:
                if output[row + dr, col + dc] != background:
                    break
                output[row + dr, col + dc] = color
            else:
                continue
            break
        else:
            if all(output[r, c] == color for color in colors for r, c in points[color]):
                solutions[output.tobytes()] = output
    if len(solutions) == 1:
        return next(iter(solutions.values()))
    return None


def _generate_pair(rng, side, count, previous, special):
    background, colors = _select_palette(rng, count, previous)
    for _ in range(100):
        template, orbits = _sample_template(rng, side, special == "center")
        effective_special = special
        if special == "center" and (side % 2 == 0 or (side // 2, side // 2) not in template):
            effective_special = "boundary"
        copies = _damage_copies(rng, template, orbits, count, side, effective_special)
        height, width, positions = _layout(rng, count, side)
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = input_grid.copy()
        for color, (top, left), surviving in zip(colors, positions, copies):
            for row, col in template:
                output_grid[top + row, left + col] = color
            for row, col in surviving:
                input_grid[top + row, left + col] = color
        if not np.any(input_grid != output_grid):
            continue
        solved = _solve_from_input(input_grid)
        if solved is not None and np.array_equal(solved, output_grid):
            return input_grid, output_grid, (background, colors)
    raise ValueError("could not construct a uniquely recoverable pair")


def _array_key(array):
    return (array.shape, array.tobytes())


def _template_key(output_grid):
    background = _dominant_color(output_grid)
    color = next(int(c) for c in np.unique(output_grid) if c != background)
    coords = _points(output_grid, color)
    top, bottom, left, right = _box(coords)
    return (bottom - top + 1, frozenset((r - top, c - left) for r, c in coords))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    previous = None
    seen = set()
    seen_templates = set()
    for section, pair_count in (("train", num_train), ("test", num_test)):
        for index in range(pair_count):
            if section == "train":
                side = 5 if index == 0 else 6 if index == 1 else int(rng.choice([5, 5, 6, 6, 7]))
                special = "center" if index == 0 else "boundary" if index == 1 else str(rng.choice(["none", "center", "boundary", "orbit"]))
            else:
                side = (5 if index == 0 else 7) if pair_count >= 2 and index < 2 else int(rng.choice([5, 6, 7, 7]))
                special = str(rng.choice(["none", "center", "boundary", "orbit"]))
            count = int(rng.integers(2, 5))
            for _ in range(10):
                pair_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
                input_grid, output_grid, palette = _generate_pair(
                    np.random.default_rng(pair_seed), side, count, previous, special
                )
                key = _array_key(input_grid)
                template_key = _template_key(output_grid)
                if key not in seen and template_key not in seen_templates:
                    break
            else:
                raise ValueError("could not generate unique input")
            seen.add(key)
            seen_templates.add(template_key)
            puzzle[section].append({"input": input_grid, "output": output_grid})
            previous = palette
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background = _dominant_color(input_grid)
    assert _dominant_color(output_grid) == background
    input_colors = set(int(c) for c in np.unique(input_grid)) - {background}
    output_colors = set(int(c) for c in np.unique(output_grid)) - {background}
    assert input_colors == output_colors
    assert 2 <= len(input_colors) <= 4
    assert np.any(input_grid != output_grid)
    assert np.all(input_grid[input_grid != background] == output_grid[input_grid != background])
    assert np.all(input_grid[input_grid != output_grid] == background)
    boxes = []
    templates = []
    sides = []
    missing_orbits = 0
    intact = 0
    for color in sorted(input_colors):
        input_points = _points(input_grid, color)
        output_points = _points(output_grid, color)
        assert input_points <= output_points
        top, bottom, left, right = _box(output_points)
        side = bottom - top + 1
        assert side == right - left + 1
        local_output = {(r - top, c - left) for r, c in output_points}
        local_input = {(r - top, c - left) for r, c in input_points}
        assert _closure(local_output, side) == local_output
        missing_orbits += sum(orbit <= local_output and not orbit & local_input for orbit in _all_orbits(side))
        intact += input_points == output_points
        boxes.append((top, bottom, left, right))
        templates.append(local_output)
        sides.append(side)
    assert len(set(sides)) == 1
    assert all(template == templates[0] for template in templates)
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            assert (first[1] + 1 < second[0] or second[1] + 1 < first[0]
                    or first[3] + 1 < second[2] or second[3] + 1 < first[2])
    assert set().union(*(
        {(r - box[0], c - box[2]) for r, c in _points(input_grid, color)}
        for color, box in zip(sorted(input_colors), boxes)
    )) <= templates[0]
    solved = _solve_from_input(input_grid)
    assert solved is not None and np.array_equal(solved, output_grid)
    digest = hashlib.sha256(input_grid.astype(np.int8).tobytes() + output_grid.astype(np.int8).tobytes()).hexdigest()
    assert digest not in _SOURCE_PAIR_DIGESTS
    return {"background": background, "colors": input_colors, "side": sides[0],
            "count": len(input_colors), "orbits": sum(orbit <= templates[0] for orbit in _all_orbits(sides[0])),
            "missing_orbits": missing_orbits, "intact": intact,
            "damage": sum(len(_points(output_grid, c) - _points(input_grid, c)) for c in input_colors),
            "shape": input_grid.shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    seen_templates = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            template_key = _template_key(pair["output"])
            assert template_key not in seen_templates
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_templates.add(template_key)
            analyses[section].append(_analyze_pair(pair["input"], pair["output"]))
    assert analyses["train"][0]["side"] == 5
    assert analyses["train"][1]["side"] == 6
    assert analyses["train"][0]["missing_orbits"] >= 1
    assert analyses["train"][1]["missing_orbits"] >= 1 or analyses["train"][1]["damage"] >= 2
    if len(analyses["test"]) >= 2:
        assert analyses["test"][0]["side"] == 5
        assert analyses["test"][1]["side"] == 7


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(314159)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(271828, 4, 2)
    assert _puzzles_equal(explicit, generate_puzzle(271828, 4, 2))
    assert not _puzzles_equal(explicit, generate_puzzle(271829, 4, 2))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(9000 + (ntrain or 0) * 10 + (ntest or 0), ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, shapes, backgrounds, patterns, layouts = (set() for _ in range(6))
    shared = missing = intact = rich = simple = role_exchange = same_background = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = [_analyze_pair(pair["input"], pair["output"])
                 for section in ("train", "test") for pair in puzzle[section]]
        shapes.update(info["shape"] for info in infos)
        backgrounds.update(info["background"] for info in infos)
        patterns.add(tuple((info["side"], info["orbits"]) for info in infos))
        layouts.add(tuple((info["shape"], info["count"]) for info in infos))
        shared += len(infos)
        missing += sum(info["missing_orbits"] > 0 for info in infos)
        intact += sum(info["intact"] > 0 for info in infos)
        rich += sum(info["orbits"] >= 5 for info in infos)
        simple += sum(info["orbits"] <= 3 for info in infos)
        role_exchange += any(info["background"] in prior["colors"]
                             for prior, info in zip(infos, infos[1:]))
        same_background += any(info["background"] == prior["background"]
                               for prior, info in zip(infos, infos[1:]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(backgrounds) == 10
    assert len(patterns) >= 30 and len(layouts) >= 30
    assert shared >= 200 and missing >= 42 and intact > 0
    assert rich > 20 and simple > 20
    assert role_exchange > 10 and same_background > 10
