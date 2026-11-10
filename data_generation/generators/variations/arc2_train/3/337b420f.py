"""Generate panel-composition ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ("44888088888088886", "84488058885088888", "88448055888088888",
     "48844088888088668", "44884088888088866"),
    ("88888088888011181", "22282088888088888", "88888066686088888",
     "88888066686088888", "88888088888088888"),
    ("88888088887088844", "88888087778088884", "68886087878088488",
     "68888087778088888", "66688088888088888"),
    ("88888088888098888", "81888088558088888", "88888088888089988",
     "88881088858089888", "88111088888088888"),
    ("66688088888088844", "86888018188088844", "88888011188088888",
     "88888088888044444", "88866081888088888"),
    ("88855088228033888", "88888088822033388", "88888028888083888",
     "55588028888088833", "55588088888088833"),
}
_ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_input(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUTS


def _components(cells):
    rest = set(cells)
    found = []
    while rest:
        start = rest.pop()
        component = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for dr, dc in _ORTHOGONAL:
                neighbor = row + dr, col + dc
                if neighbor in rest:
                    rest.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        found.append(frozenset(component))
    return sorted(found, key=lambda part: (-len(part), tuple(sorted(part))))


def _normal(shape):
    top = min(row for row, _ in shape)
    left = min(col for _, col in shape)
    return frozenset((row - top, col - left) for row, col in shape)


def _variants(shape):
    result = set()
    for mirror in range(2):
        for turns in range(4):
            points = set(shape)
            if mirror:
                points = {(row, -col) for row, col in points}
            for _ in range(turns):
                points = {(col, -row) for row, col in points}
            result.add(_normal(points))
    return tuple(result)


def _shape(rng, size, height, width):
    """Sample a shape before attempting any assembly."""
    styles = ("growth", "growth", "growth", "line", "solid", "bend",
              "step", "ring")
    for _ in range(35):
        style = styles[int(rng.integers(len(styles)))]
        if style == "line":
            points = {(0, col) for col in range(size)}
        elif style == "ring":
            side = 3 if size == 8 else 4 if size == 12 else 0
            if not side:
                continue
            points = {(row, col) for row in range(side) for col in range(side)
                      if row in (0, side - 1) or col in (0, side - 1)}
        elif style == "solid":
            factors = [(a, size // a) for a in range(2, size + 1)
                       if size % a == 0 and size // a >= 2]
            if not factors:
                continue
            a, b = factors[int(rng.integers(len(factors)))]
            points = {(row, col) for row in range(a) for col in range(b)}
        elif style == "bend":
            if size < 3:
                continue
            arm = int(rng.integers(2, size))
            points = {(0, col) for col in range(arm)}
            points.update((row, arm - 1) for row in range(size - arm + 1))
        elif style == "step":
            points = {(index // 2, index // 2 + index % 2) for index in range(size)}
            # The staircase above touches only diagonally at some turns.
            if len(_components(points)) != 1:
                continue
        else:
            points = {(0, 0)}
            while len(points) < size:
                frontier = sorted({
                    (row + dr, col + dc)
                    for row, col in points
                    for dr, dc in _ORTHOGONAL
                    if (row + dr, col + dc) not in points
                })
                points.add(frontier[int(rng.integers(len(frontier)))])
        variants = _variants(points)
        feasible = [item for item in variants
                    if max(row for row, _ in item) < height
                    and max(col for _, col in item) < width]
        if feasible:
            return feasible[int(rng.integers(len(feasible)))]
    snake = [(row, col) for row in range(height)
             for col in (range(width) if row % 2 == 0 else range(width - 1, -1, -1))]
    return _normal(snake[:size])


def _placements(shape, height, width):
    shape_height = max(row for row, _ in shape) + 1
    shape_width = max(col for _, col in shape) + 1
    return [
        frozenset((row + dr, col + dc) for row, col in shape)
        for dr in range(height - shape_height + 1)
        for dc in range(width - shape_width + 1)
    ]


def _contact(first, second):
    edge = sum((row + dr, col + dc) in second
               for row, col in first for dr, dc in _ORTHOGONAL)
    corner = sum((row + dr, col + dc) in second
                 for row, col in first
                 for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
    return edge, corner


def _direct_layout(rng, shapes, height, width, contact_mode):
    pieces = []
    occupied = frozenset()
    for index, shape in enumerate(shapes):
        options = [candidate for candidate in _placements(shape, height, width)
                   if not candidate & occupied]
        if index:
            if contact_mode == "corner" or (
                contact_mode == "mixed" and bool(rng.integers(2))
            ):
                preferred = [item for item in options
                             if _contact(item, occupied)[0] == 0
                             and _contact(item, occupied)[1] > 0]
            else:
                preferred = [item for item in options
                             if _contact(item, occupied)[0] > 0]
            if not preferred:
                return None
            options = preferred
        if not options:
            return None
        chosen = options[int(rng.integers(len(options)))]
        pieces.append(chosen)
        occupied = occupied | chosen
    return pieces


def _overlap(pieces):
    return len(set().union(*pieces)) != sum(map(len, pieces))


def _minimal_assembly(pieces, height, width, max_move=2):
    """Return only a unique minimum-Manhattan-displacement placement."""
    if not _overlap(pieces):
        return list(pieces), 0, True
    choices = []
    for piece in pieces:
        offsets = []
        for dr in range(-max_move, max_move + 1):
            for dc in range(-max_move, max_move + 1):
                cost = abs(dr) + abs(dc)
                if cost > max_move:
                    continue
                moved = frozenset((row + dr, col + dc) for row, col in piece)
                if all(0 <= row < height and 0 <= col < width
                       for row, col in moved):
                    offsets.append((cost, moved))
        choices.append(offsets)
    for budget in range(1, max_move + 1):
        answers = []

        def visit(index, cost, occupied, chosen):
            if len(answers) > 1:
                return
            if index == len(choices):
                if cost == budget:
                    answers.append(tuple(chosen))
                return
            for delta, moved in choices[index]:
                if cost + delta <= budget and not moved & occupied:
                    visit(index + 1, cost + delta, occupied | moved,
                          chosen + [moved])

        visit(0, 0, frozenset(), [])
        if answers:
            return list(answers[0]), budget, len(answers) == 1
    return None, None, False


def _collision_layout(rng, direct, height, width):
    indices = rng.permutation(len(direct))
    directions = list(_ORTHOGONAL)
    for index in indices:
        for direction in rng.permutation(4):
            dr, dc = directions[int(direction)]
            raw = list(direct)
            raw[int(index)] = frozenset(
                (row + dr, col + dc) for row, col in raw[int(index)]
            )
            if not all(0 <= row < height and 0 <= col < width
                       for row, col in raw[int(index)]):
                continue
            if not _overlap(raw):
                continue
            placed, distance, unique = _minimal_assembly(raw, height, width)
            if unique and distance == 1:
                return raw, placed
    return None


def _grow_distractor(rng, allowed, size, prefer_span=False):
    allowed = set(allowed)
    if len(allowed) < size:
        return None
    candidates = []
    for _ in range(28):
        start = sorted(allowed)[int(rng.integers(len(allowed)))]
        points = {start}
        while len(points) < size:
            frontier = sorted({
                (row + dr, col + dc)
                for row, col in points for dr, dc in _ORTHOGONAL
                if (row + dr, col + dc) in allowed
                and (row + dr, col + dc) not in points
            })
            if not frontier:
                break
            points.add(frontier[int(rng.integers(len(frontier)))])
        if len(points) == size:
            candidates.append(frozenset(points))
    if not candidates:
        return None
    if prefer_span:
        return max(candidates, key=lambda item:
                   (max(col for _, col in item) - min(col for _, col in item)
                    + max(row for row, _ in item) - min(row for row, _ in item)))
    return candidates[int(rng.integers(len(candidates)))]


def _longer_span_distractor(rng, allowed, piece):
    piece_span = (max(row for row, _ in piece) - min(row for row, _ in piece)
                  + max(col for _, col in piece) - min(col for _, col in piece))
    available = set(allowed)
    choices = []
    for size in range(3, min(len(piece) - 1, 6) + 1):
        if size - 1 <= piece_span:
            continue
        for row, col in sorted(available):
            for dr, dc in ((1, 0), (0, 1)):
                line = frozenset((row + dr * step, col + dc * step)
                                 for step in range(size))
                if line <= available:
                    choices.append(line)
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _distractors(rng, pieces, height, width, difficulty):
    result = []
    has_non_singleton = False
    has_singleton = False
    close_gap = False
    span_win = False
    for index, piece in enumerate(pieces):
        if difficulty == "easy":
            count = 1
        elif difficulty == "selection" and index == 0:
            count = 1
        else:
            count = int(rng.choice((0, 1, 1, 1, 2)))
        forbidden = set(piece)
        for row, col in piece:
            forbidden.update((row + dr, col + dc) for dr, dc in _ORTHOGONAL)
        used = set()
        panel_distractors = []
        for item_index in range(count):
            allowed = {(row, col) for row in range(height) for col in range(width)
                       if (row, col) not in forbidden and (row, col) not in used}
            if not allowed:
                if difficulty in ("easy", "selection") and index == 0:
                    return None
                break
            maximum = min(len(piece) - 1, 6, len(allowed))
            if maximum < 1:
                return None
            if difficulty == "easy":
                sizes = [1]
            elif difficulty == "selection" and index == 0:
                sizes = list(range(maximum, 1, -1))
            else:
                target = int(rng.integers(1, maximum + 1))
                sizes = [target] + [n for n in range(maximum, 0, -1)
                                    if n != target]
            chosen = (_longer_span_distractor(rng, allowed, piece)
                      if difficulty != "easy" and rng.integers(5) == 0
                      else None)
            if chosen is None:
                for size in sizes:
                    chosen = _grow_distractor(
                        rng, allowed, size,
                        prefer_span=bool(rng.integers(6) == 0),
                    )
                    if chosen is not None:
                        break
            if chosen is None:
                if difficulty in ("easy", "selection") and index == 0:
                    return None
                break
            panel_distractors.append(chosen)
            used.update(chosen)
            for row, col in chosen:
                forbidden.update((row + dr, col + dc) for dr, dc in _ORTHOGONAL)
            has_non_singleton |= len(chosen) > 1
            has_singleton |= len(chosen) == 1
            close_gap |= len(piece) - len(chosen) <= 2
            piece_span = max(row for row, _ in piece) - min(row for row, _ in piece) + max(
                col for _, col in piece) - min(col for _, col in piece)
            loser_span = max(row for row, _ in chosen) - min(row for row, _ in chosen) + max(
                col for _, col in chosen) - min(col for _, col in chosen)
            span_win |= loser_span > piece_span
        result.append(panel_distractors)
    if difficulty == "selection" and not has_non_singleton:
        return None
    return result


def _compose(pieces, distractors, colors, background, divider,
             height, width, orientation):
    count = len(pieces)
    if orientation == "horizontal":
        grid = np.full((height, count * width + count - 1),
                       background, dtype=np.int8)
        for index in range(count - 1):
            grid[:, (index + 1) * width + index] = divider
    else:
        grid = np.full((count * height + count - 1, width),
                       background, dtype=np.int8)
        for index in range(count - 1):
            grid[(index + 1) * height + index, :] = divider
    for index, (piece, losers, color) in enumerate(zip(pieces, distractors, colors)):
        row_base = index * (height + 1) if orientation == "vertical" else 0
        col_base = index * (width + 1) if orientation == "horizontal" else 0
        for row, col in set(piece).union(*losers):
            grid[row_base + row, col_base + col] = color
    return grid


def _render(pieces, colors, background, height, width):
    result = np.full((height, width), background, dtype=np.int8)
    for piece, color in zip(pieces, colors):
        for row, col in piece:
            result[row, col] = color
    return result


def _count(name, value, lower, upper):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not lower <= int(value) <= upper
    ):
        raise ValueError(f"{name} must be an integer from {lower} to {upper}")


def _resolve_counts(seed, rng, num_train, num_test):
    _count("num_train", num_train, 2, 10)
    _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _pair(rng, count, orientation, collision, difficulty, dimensions=None):
    cap = (30 - count + 1) // count
    if dimensions is not None:
        height, width = dimensions
    elif orientation == "horizontal":
        width = int(rng.integers(5, min(8, cap) + 1))
        height = int(rng.integers(5, 9))
    else:
        height = int(rng.integers(5, min(8, cap) + 1))
        width = int(rng.integers(5, 9))
    background, divider = [int(x) for x in rng.choice(10, 2, replace=False)]
    palette = [color for color in range(10) if color not in (background, divider)]
    colors = [int(x) for x in rng.choice(palette, count, replace=False)]
    contact_mode = ("edge", "corner", "mixed")[int(rng.integers(3))]
    limit = min(12 if count == 2 else 10 if count == 3 else 7,
                height * width // count + 3)
    for _ in range(140):
        sizes = [int(rng.integers(2, limit + 1)) for _ in range(count)]
        if sum(sizes) > int(height * width * 0.77):
            continue
        if difficulty == "selection" and sizes[0] < 4:
            continue
        shapes = [_shape(rng, size, height, width) for size in sizes]
        direct = _direct_layout(rng, shapes, height, width, contact_mode)
        if direct is None:
            continue
        if collision:
            resolved = _collision_layout(rng, direct, height, width)
            if resolved is None:
                continue
            pieces, placed = resolved
        else:
            pieces, placed = direct, direct
        losers = _distractors(rng, pieces, height, width, difficulty)
        if losers is None:
            continue
        input_grid = _compose(pieces, losers, colors, background, divider,
                              height, width, orientation)
        output_grid = _render(placed, colors, background, height, width)
        if _source_input(input_grid):
            continue
        return {"input": input_grid, "output": output_grid}
    raise ValueError("Could not construct the requested pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    layout_mode = ("fixed", "shifted", "mixed")[int(rng.integers(3))]
    orientation_mode = (("horizontal", "vertical")[int(rng.integers(2))]
                        if layout_mode != "mixed" else "mixed")
    fixed_count = int(rng.integers(2, 5)) if layout_mode == "fixed" else None
    if fixed_count is not None:
        maximum = min(8, (30 - fixed_count + 1) // fixed_count)
        fixed_dimensions = (int(rng.integers(5, maximum + 1)),
                            int(rng.integers(5, maximum + 1)))
    else:
        fixed_dimensions = None
    collision_level = int(rng.choice((0, 1, 2)))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_orientations = []
    train_counts = []
    for index in range(train_count):
        if orientation_mode == "mixed":
            orientation = ("horizontal", "vertical")[index % 2]
        else:
            orientation = orientation_mode
        count = (fixed_count if fixed_count is not None else
                 3 if index == 0 else
                 (2 if seed % 2 else 4) if index == 1 else
                 int(rng.integers(2, 5)))
        collision = collision_level > 0 and index >= 1 and (
            index == 1 or (collision_level == 2 and bool(rng.integers(2)))
        )
        difficulty = "easy" if index == 0 else "selection" if index == 1 else "varied"
        for _ in range(16):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            try:
                pair = _pair(pair_rng, count, orientation, collision, difficulty,
                             fixed_dimensions)
            except ValueError:
                continue
            sig = _signature(pair["input"])
            if sig not in seen:
                break
        else:
            raise ValueError("Could not construct a distinct training pair")
        puzzle["train"].append(pair)
        seen.add(sig)
        train_orientations.append(orientation)
        train_counts.append(count)
    for index in range(test_count):
        orientation = train_orientations[int(rng.integers(len(train_orientations)))]
        count = train_counts[int(rng.integers(len(train_counts)))]
        collision = collision_level > 0 and bool(rng.integers(3) == 0)
        difficulty = ("easy", "selection", "varied")[int(rng.integers(3))]
        for _ in range(16):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            try:
                pair = _pair(pair_rng, count, orientation, collision, difficulty,
                             fixed_dimensions)
            except ValueError:
                continue
            sig = _signature(pair["input"])
            if sig not in seen:
                break
        else:
            raise ValueError("Could not construct a distinct test pair")
        puzzle["test"].append(pair)
        seen.add(sig)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(grid):
    options = []
    for orientation in ("horizontal", "vertical"):
        length = grid.shape[1] if orientation == "horizontal" else grid.shape[0]
        other = grid.shape[0] if orientation == "horizontal" else grid.shape[1]
        for count in (2, 3, 4):
            if (length - count + 1) % count:
                continue
            panel_length = (length - count + 1) // count
            if not 5 <= panel_length <= 8 or not 5 <= other <= 8:
                continue
            height = other if orientation == "horizontal" else panel_length
            width = panel_length if orientation == "horizontal" else other
            separators = [
                grid[:, (i + 1) * width + i] if orientation == "horizontal"
                else grid[(i + 1) * height + i, :]
                for i in range(count - 1)
            ]
            if not separators or not all(np.all(line == separators[0][0])
                                         for line in separators):
                continue
            divider = int(separators[0][0])
            panels = [
                grid[:, i * (width + 1):i * (width + 1) + width]
                if orientation == "horizontal" else
                grid[i * (height + 1):i * (height + 1) + height, :]
                for i in range(count)
            ]
            color_sets = [set(int(x) for x in np.unique(panel)) for panel in panels]
            if any(divider in colors or len(colors) != 2 for colors in color_sets):
                continue
            common = set.intersection(*color_sets)
            if len(common) != 1:
                continue
            background = common.pop()
            colors = [next(iter(items - {background})) for items in color_sets]
            if len(set(colors)) != count:
                continue
            pieces = []
            losers = []
            valid = True
            for panel, color in zip(panels, colors):
                components = _components({
                    (row, col) for row in range(height) for col in range(width)
                    if int(panel[row, col]) == color
                })
                if not components or (len(components) > 1 and
                                      len(components[0]) <= len(components[1])):
                    valid = False
                    break
                pieces.append(components[0])
                losers.append(components[1:])
            if valid:
                options.append((orientation, height, width, background,
                                divider, colors, pieces, losers))
    assert len(options) == 1
    return options[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    training = []
    testing = []
    for split, records in (("train", training), ("test", testing)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= extent <= 30 for extent in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert not _source_input(pair["input"])
            sig = _signature(pair["input"])
            assert sig not in seen
            seen.add(sig)
            info = _analyze(pair["input"])
            orientation, height, width, background, divider, colors, pieces, losers = info
            assert pair["output"].shape == (height, width)
            assert background != divider and divider not in colors
            assert all(all(len(loser) < len(piece) for loser in panel_losers)
                       for piece, panel_losers in zip(pieces, losers))
            placed, movement, unique = _minimal_assembly(pieces, height, width)
            assert unique and placed is not None
            assert np.array_equal(pair["output"],
                                  _render(placed, colors, background, height, width))
            records.append((orientation, len(pieces), movement, losers, pieces))
    assert any(any(loser for panel in item[3] for loser in panel)
               for item in training)
    assert any(any(len(loser) > 1 for panel in item[3] for loser in panel)
               for item in training)
    assert any(item[2] == 0 for item in training)
    train_orientations = {item[0] for item in training}
    train_counts = {item[1] for item in training}
    train_collision = any(item[2] > 0 for item in training)
    for item in testing:
        assert item[0] in train_orientations
        assert item[1] in train_counts
        assert item[2] == 0 or train_collision


def self_test() -> None:
    counts = set()
    test_counts = set()
    dimensions = set()
    layouts = set()
    colors = set()
    shapes = set()
    collisions = 0
    test_collisions = 0
    corner_only = 0
    non_singletons = 0
    zero_losers = 0
    two_losers = 0
    close = 0
    large_gap = 0
    larger_span = 0
    rings = 0
    large_solid = 0
    test_without_triomino = 0
    test_outside_old_cell_range = 0
    fixed_puzzles = 0
    shifted_puzzles = 0
    mixed_puzzles = 0
    no_collision_puzzles = 0
    several_collision_puzzles = 0
    varying_background = 0
    varying_divider = 0
    combined_variations = 0
    contact_lengths = set()
    panel_counts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        repeat = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k])
                   for split in ("train", "test")
                   for a, b in zip(puzzle[split], repeat[split])
                   for k in ("input", "output"))
        local_geometry = set()
        local_background = set()
        local_divider = set()
        local_orientations = set()
        local_train_collisions = 0
        for split in ("train", "test"):
            for pair in puzzle[split]:
                orientation, height, width, background, divider, palette, pieces, losers = _analyze(pair["input"])
                local_geometry.add((orientation, height, width, len(pieces)))
                local_orientations.add(orientation)
                local_background.add(background)
                local_divider.add(divider)
                dimensions.add((orientation, height, width, len(pieces)))
                panel_counts.add(len(pieces))
                colors.add((background, divider))
                layouts.add(tuple(tuple(sorted(piece)) for piece in pieces))
                shapes.update(_normal(piece) for piece in pieces)
                collisions += int(_overlap(pieces))
                local_train_collisions += int(split == "train" and _overlap(pieces))
                test_collisions += int(split == "test" and _overlap(pieces))
                union = set().union(*pieces) if not _overlap(pieces) else set()
                if union and len(_components(union)) > 1:
                    corner_only += 1
                placed, _, unique = _minimal_assembly(pieces, height, width)
                assert unique
                for i in range(len(placed)):
                    for j in range(i + 1, len(placed)):
                        contact_lengths.add(_contact(placed[i], placed[j])[0])
                non_singletons += sum(len(loser) > 1 for panel in losers for loser in panel)
                zero_losers += sum(not panel for panel in losers)
                two_losers += sum(len(panel) >= 2 for panel in losers)
                close += sum(len(piece) - len(loser) <= 2
                             for piece, panel in zip(pieces, losers) for loser in panel)
                large_gap += sum(len(piece) - len(loser) >= 4
                                 for piece, panel in zip(pieces, losers) for loser in panel)
                for piece, panel in zip(pieces, losers):
                    piece_span = (max(r for r, _ in piece) - min(r for r, _ in piece)
                                  + max(c for _, c in piece) - min(c for _, c in piece))
                    box_area = ((max(r for r, _ in piece) - min(r for r, _ in piece) + 1)
                                * (max(c for _, c in piece) - min(c for _, c in piece) + 1))
                    rings += int(len(piece) == 8 and box_area == 9
                                 and _normal(piece) == frozenset(
                                     (r, c) for r in range(3) for c in range(3)
                                     if (r, c) != (1, 1)))
                    large_solid += int(len(piece) >= 6 and len(piece) == box_area)
                    for loser in panel:
                        loser_span = (max(r for r, _ in loser) - min(r for r, _ in loser)
                                      + max(c for _, c in loser) - min(c for _, c in loser))
                        larger_span += int(loser_span > piece_span)
                if split == "test":
                    test_without_triomino += int(all(len(piece) != 3 for piece in pieces))
                    test_outside_old_cell_range += int(
                        sum(map(len, pieces)) not in (12, 13, 14))
                combined_variations += int(
                    _overlap(pieces) and any(len(loser) > 1
                                             for panel in losers for loser in panel)
                    and height != width)
        fixed_puzzles += len(local_geometry) == 1
        shifted_puzzles += len(local_geometry) > 1 and len(local_orientations) == 1
        mixed_puzzles += len(local_orientations) > 1
        no_collision_puzzles += local_train_collisions == 0
        several_collision_puzzles += local_train_collisions >= 2
        varying_background += len(local_background) > 1
        varying_divider += len(local_divider) > 1
    assert len(counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 20 and len(colors) >= 12
    assert len(layouts) >= 100 and len(shapes) >= 30
    assert collisions >= 8 and test_collisions >= 1
    assert corner_only >= 5 and non_singletons >= 20
    assert zero_losers >= 5 and close >= 10
    assert panel_counts == {2, 3, 4}
    assert two_losers >= 5 and large_gap >= 20 and larger_span >= 3
    assert rings >= 3 and large_solid >= 10
    assert test_without_triomino >= 10 and test_outside_old_cell_range >= 10
    assert fixed_puzzles >= 3 and shifted_puzzles >= 3 and mixed_puzzles >= 3
    assert no_collision_puzzles >= 3 and several_collision_puzzles >= 3
    assert varying_background >= 30 and varying_divider >= 30
    assert combined_variations >= 3 and max(contact_lengths) >= 3
    ambiguous = [frozenset({(1, 1)}), frozenset({(1, 1)}),
                 frozenset({(4, 4)})]
    assert not _minimal_assembly(ambiguous, 5, 5)[2]
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(8000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        generate_puzzle(8100 + (train or 0) * 10 + (test or 0), train, test)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2.0, 1)):
        try:
            generate_puzzle(9000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
