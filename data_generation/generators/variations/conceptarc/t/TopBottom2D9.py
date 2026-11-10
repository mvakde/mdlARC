import numpy as np


_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_ORIGINAL_INPUTS = {
    ("111111", "000000", "111111", "000000", "111111", "000000"),
    ("200200", "020020", "002002", "200200", "020020", "002002"),
    ("040404", "440404", "000404", "444404", "000004", "444444"),
    ("088800", "880008", "000888", "088800", "880008", "000888"),
    ("900990", "990099", "099009", "009900", "900990", "990099"),
    ("000777", "777700", "700007", "007777", "777000", "000077"),
    ("003058", "333058", "000058", "555558", "000008", "888888"),
}


def _grid_key(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _components(grid):
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy in range(max(0, y - 1), min(h, y + 2)):
                    for xx in range(max(0, x - 1), min(w, x + 2)):
                        if not seen[yy, xx] and int(grid[yy, xx]) == color:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
            result.append((color, cells))
    return result


def _selected_component(grid):
    objects = _components(grid)
    if len(objects) < 2:
        raise ValueError("At least two objects are required")
    bottoms = [max(r for r, _ in cells) for _, cells in objects]
    if bottoms.count(min(bottoms)) != 1:
        raise ValueError("The highest bottommost cell must be unique")
    return objects[bottoms.index(min(bottoms))], objects


def _solve(grid):
    selected, _ = _selected_component(grid)
    out = grid.copy()
    for r, c in selected[1]:
        out[r, c] = 5
    return out


def _choose_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ValueError("Pair count outside the allowed range")
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _size(rng):
    if rng.random() < 0.36:
        return 6, 6
    return int(rng.integers(6, 11)), int(rng.integers(6, 11))


def _can_add(grid, mask, r, c, color):
    if grid[r, c] or mask[r, c]:
        return False
    h, w = grid.shape
    return not np.any(grid[max(0, r - 1):min(h, r + 2),
                           max(0, c - 1):min(w, c + 2)] == color)


def _object_mask(rng, grid, color, top, bottom, desired, diagonal=False, prefer=None):
    h, w = grid.shape
    starts = list(int(x) for x in rng.permutation(w))
    if prefer is not None:
        starts.sort(key=lambda c: -abs(c - prefer) + float(rng.random()) * 0.15)
    for start in starts:
        mask = np.zeros((h, w), dtype=bool)
        if not _can_add(grid, mask, top, start, color):
            continue
        mask[top, start] = True
        col = start
        for row in range(top + 1, bottom + 1):
            steps = (-1, 1) if diagonal else (-1, 0, 1)
            choices = [col + step for step in steps
                       if 0 <= col + step < w and _can_add(grid, mask, row, col + step, color)]
            if not choices:
                break
            col = int(rng.choice(choices))
            mask[row, col] = True
        else:
            if not diagonal:
                for _ in range(max(0, desired - int(mask.sum()))):
                    frontier = []
                    for r in range(top, bottom + 1):
                        for c in range(w):
                            if not _can_add(grid, mask, r, c, color):
                                continue
                            if np.any(mask[max(0, r - 1):min(h, r + 2),
                                           max(0, c - 1):min(w, c + 2)]):
                                frontier.append((r, c))
                    if not frontier:
                        break
                    mask[frontier[int(rng.integers(len(frontier)))]] = True
            return mask
    return None


def _simple_pair(rng, shape, color, style):
    h, w = shape
    grid = np.zeros(shape, dtype=np.int8)
    if style == "stripe":
        row = int(rng.integers(0, h - 3))
        lower = int(rng.integers(row + 2, h))
        grid[row, :] = color
        width = int(rng.integers(max(2, w // 2), w + 1))
        col = int(rng.integers(w - width + 1))
        grid[lower, col:col + width] = color
    else:
        left = bool(rng.integers(2))
        low = int(rng.integers(1, h - 2))
        a = slice(0, 2) if left else slice(w - 2, w)
        b = slice(w - 2, w) if left else slice(0, 2)
        for r in range(low + 1):
            grid[r, a.start + int(rng.integers(2))] = color
        for r in range(h):
            grid[r, b.start + int(rng.integers(2))] = color
    return grid


def _pair_spec(rng, kind):
    if kind == "diagonal":
        return dict(diagonal=True, counter=False, existing=False, mono=True,
                    count=2, dense=False, small=False)
    if kind == "counter":
        return dict(diagonal=False, counter=True, existing=True, mono=False,
                    count=3, dense=False, small=True)
    mono = bool(rng.random() < 0.36)
    existing = not mono and bool(rng.random() < 0.48)
    count = int(rng.integers(2 if mono else 3, 4 if mono else 6))
    return dict(diagonal=bool(rng.random() < 0.16),
                counter=bool(rng.random() < 0.36), existing=existing,
                mono=mono, count=count, dense=not mono and bool(rng.random() < 0.42),
                small=bool(rng.random() < 0.35))


def _construct(rng, spec, target_color, other_colors):
    for _ in range(350):
        shape = _size(rng)
        h, w = shape
        if (spec["mono"] and spec["count"] == 2 and not spec["diagonal"]
                and not spec["counter"] and rng.random() < 0.23):
            grid = _simple_pair(rng, shape, target_color,
                                "stripe" if rng.integers(2) else "lane")
            try:
                selected, objects = _selected_component(grid)
            except ValueError:
                continue
            if len(objects) == 2 and selected[0] == target_color:
                return grid, _solve(grid)
            continue
        target_bottom = int(rng.integers(2 if spec["counter"] else 1, h - 1))
        target_top = (int(rng.integers(1, target_bottom + 1)) if spec["counter"]
                      else int(rng.integers(target_bottom + 1)))
        if spec["diagonal"] and target_top == target_bottom:
            target_top = target_bottom - 1
        grid = np.zeros(shape, dtype=np.int8)
        target_area = (target_bottom - target_top + 1 if spec["diagonal"] else
                       target_bottom - target_top + 1 + int(rng.integers(1, max(3, w // 2 + 2))))
        target = _object_mask(rng, grid, target_color, target_top, target_bottom,
                              target_area, diagonal=spec["diagonal"])
        if target is None:
            continue
        grid[target] = target_color
        target_center = float(np.mean(np.nonzero(target)[1]))
        masks = [target]
        colors = [target_color]
        remaining = spec["count"] - 1
        if spec["existing"]:
            bottom = int(rng.integers(target_bottom + 1, h))
            area = bottom + 1 + int(rng.integers(0, max(2, w // 2 + 1)))
            mask = _object_mask(rng, grid, 5, 0, bottom, area, prefer=target_center)
            if mask is None:
                continue
            grid[mask] = 5
            masks.append(mask)
            colors.append(5)
            remaining -= 1
        failed = False
        for j in range(remaining):
            color = (target_color if spec["mono"] else
                     int(other_colors[j % len(other_colors)]))
            bottom = int(rng.integers(target_bottom + 1, h))
            top = (0 if spec["counter"] and not spec["existing"] and j == 0 else
                   int(rng.integers(bottom + 1)))
            if spec["small"] and j == remaining - 1:
                top = bottom
                area = int(rng.integers(1, 3))
            else:
                area = bottom - top + 1 + int(rng.integers(1, max(3, w // 2 + 3)))
                if spec["dense"]:
                    area += int(rng.integers(2, w + 1))
            mask = _object_mask(rng, grid, color, top, bottom, area,
                                prefer=target_center if spec["mono"] else None)
            if mask is None:
                failed = True
                break
            grid[mask] = color
            masks.append(mask)
            colors.append(color)
        if failed:
            continue
        try:
            selected, objects = _selected_component(grid)
        except ValueError:
            continue
        if len(objects) != spec["count"] or selected[0] != target_color:
            continue
        if set(selected[1]) != set(zip(*np.nonzero(target))):
            continue
        if spec["counter"] and not any(min(r for r, _ in cells) < target_top
                                         for _, cells in objects if cells is not selected[1]):
            continue
        if spec["dense"] and (np.any(np.all(grid == 0, axis=0)) or
                              np.any(np.all(grid == 0, axis=1))):
            continue
        values, counts = np.unique(grid, return_counts=True)
        if int(values[int(np.argmax(counts))]) != 0:
            continue
        return grid, _solve(grid)
    raise RuntimeError("Could not construct a valid object arrangement")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _choose_counts(rng, num_train, num_test)
    train_colors = [int(x) for x in rng.choice(_COLORS,
                    size=min(n_train, int(rng.integers(2, 5))), replace=False)]
    kinds = ["diagonal", "counter"] + ["free"] * (n_train - 2)
    rng.shuffle(kinds)
    puzzle = {"train": [], "test": []}
    used = set()
    for index, kind in enumerate(kinds):
        target_color = train_colors[index % len(train_colors)]
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
            spec = _pair_spec(pair_rng, kind)
            choices = [c for c in train_colors if c != target_color]
            try:
                inp, out = _construct(pair_rng, spec, target_color, choices)
            except RuntimeError:
                continue
            key = _grid_key(inp)
            if key not in used and key not in _ORIGINAL_INPUTS:
                used.add(key)
                puzzle["train"].append({"input": inp, "output": out})
                break
        else:
            raise RuntimeError("Could not construct a distinct training pair")
    seen_colors = {int(v) for pair in puzzle["train"] for v in np.unique(pair["input"])} - {0, 5}
    novel_colors = [c for c in _COLORS if c not in seen_colors]
    for _ in range(n_test):
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
            spec = _pair_spec(pair_rng, "free")
            regime = int(pair_rng.integers(3))
            if regime == 0:
                target_color = int(pair_rng.choice(novel_colors))
                others = [int(pair_rng.choice(tuple(seen_colors)))]
            elif regime == 1:
                target_color = int(pair_rng.choice(tuple(seen_colors)))
                others = [int(pair_rng.choice(novel_colors))]
            else:
                target_color = int(pair_rng.choice(tuple(seen_colors)))
                others = [int(pair_rng.choice(tuple(seen_colors - {target_color} or seen_colors)))]
            if spec["mono"]:
                others = [target_color]
            try:
                inp, out = _construct(pair_rng, spec, target_color, others)
            except RuntimeError:
                continue
            key = _grid_key(inp)
            if key not in used and key not in _ORIGINAL_INPUTS:
                used.add(key)
                puzzle["test"].append({"input": inp, "output": out})
                break
        else:
            raise RuntimeError("Could not construct a distinct test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _orthogonally_disconnected(cells):
    remaining = set(cells)
    reached = {next(iter(remaining))}
    frontier = list(reached)
    while frontier:
        r, c = frontier.pop()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in remaining and point not in reached:
                reached.add(point)
                frontier.append(point)
    return len(reached) != len(remaining)


def _predict_from_examples(train, test_input):
    changed_to = set()
    for pair in train:
        mask = pair["input"] != pair["output"]
        changed_to.update(int(x) for x in np.unique(pair["output"][mask]))
    assert changed_to == {5}
    selected, _ = _selected_component(test_input)
    predicted = test_input.copy()
    for r, c in selected[1]:
        predicted[r, c] = next(iter(changed_to))
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert all(type(puzzle[s]) is list for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_selected_colors = set()
    mono_train = mixed_train = diagonal_train = counter_train = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            values, counts = np.unique(inp, return_counts=True)
            assert int(values[int(np.argmax(counts))]) == 0
            key = _grid_key(inp)
            assert key not in seen and key not in _ORIGINAL_INPUTS
            seen.add(key)
            selected, objects = _selected_component(inp)
            assert selected[0] != 5 and len(objects) >= 2
            assert np.array_equal(out, _solve(inp)) and not np.array_equal(inp, out)
            if split == "train":
                train_selected_colors.add(selected[0])
                nonfive = set(int(x) for x in values) - {0, 5}
                mono_train |= len(nonfive) == 1 and 5 not in values
                mixed_train |= len(nonfive) >= 2 and 5 in values
                diagonal_train |= _orthogonally_disconnected(selected[1])
                target_top = min(r for r, _ in selected[1])
                counter_train |= target_top > 0 and any(
                    min(r for r, _ in cells) < target_top for _, cells in objects
                    if cells is not selected[1])
            else:
                assert np.array_equal(_predict_from_examples(puzzle["train"], inp.copy()), out)
    assert len(train_selected_colors) >= 2
    assert mono_train and mixed_train and diagonal_train and counter_train


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        puzzles.append(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, signatures, palettes, structures = set(), set(), set(), set()
    no_separator = many = lower_start = five_top = novel_distractor = 0
    seen_target = close_bottom = selected_larger = novel_target = 0
    bbox_overlap = near_miss = stripe = lane = five_beside = 0
    mixed_test = mono_test = varied_shape = varied_bottom = varied_count = 0
    for puzzle in puzzles:
        training_colors = {int(v) for pair in puzzle["train"]
                           for v in np.unique(pair["input"])} - {0, 5}
        puzzle_shapes, puzzle_bottoms, puzzle_counts = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                selected, objects = _selected_component(grid)
                shapes.add(grid.shape)
                puzzle_shapes.add(grid.shape)
                puzzle_bottoms.add(max(r for r, _ in selected[1]))
                puzzle_counts.add(len(objects))
                palettes.add(tuple(int(v) for v in np.unique(grid)))
                structures.add((grid.shape, tuple(int(v != 0) for v in grid.flat)))
                signatures.add((len(objects), tuple(sorted(len(c) for _, c in objects)),
                                max(r for r, _ in selected[1])))
                no_separator += int(not np.any(np.all(grid == 0, axis=0)) and
                                    not np.any(np.all(grid == 0, axis=1)))
                many += int(len(objects) >= 4)
                lower_start += int(min(r for r, _ in selected[1]) > 0)
                five_top += int(any(color == 5 and min(r for r, _ in cells) == 0
                                    for color, cells in objects))
                bottoms = sorted(max(r for r, _ in cells) for _, cells in objects)
                close_bottom += int(bottoms[1] - bottoms[0] == 1)
                selected_larger += int(any(len(selected[1]) > len(cells)
                                           for _, cells in objects if cells is not selected[1]))
                stripe += int(any(len({r for r, _ in cells}) == 1 and
                                  len(cells) >= grid.shape[1] // 2 for _, cells in objects))
                lane += int(len(objects) == 2 and any(
                    np.all(grid[:, c] == 0) and np.any(grid[:, :c] != 0) and
                    np.any(grid[:, c + 1:] != 0)
                    for c in range(1, grid.shape[1] - 1)))
                five_beside += int(any(color == 5 and any(
                    max(abs(r - s), abs(c - d)) == 1
                    for r, c in cells for other_color, other_cells in objects
                    if other_color != 5 for s, d in other_cells)
                    for color, cells in objects))
                overlapping = touching_near = False
                for i, (color_a, cells_a) in enumerate(objects):
                    for color_b, cells_b in objects[i + 1:]:
                        if color_a != color_b:
                            continue
                        rows_a, cols_a = zip(*cells_a)
                        rows_b, cols_b = zip(*cells_b)
                        overlapping |= (max(min(rows_a), min(rows_b)) <=
                                        min(max(rows_a), max(rows_b)) and
                                        max(min(cols_a), min(cols_b)) <=
                                        min(max(cols_a), max(cols_b)))
                        touching_near |= min(max(abs(r - s), abs(c - d))
                                             for r, c in cells_a for s, d in cells_b) == 2
                bbox_overlap += int(overlapping)
                near_miss += int(touching_near)
                if split == "test":
                    seen_target += int(selected[0] in training_colors)
                    novel_target += int(selected[0] not in training_colors)
                    novel_distractor += int(selected[0] in training_colors and any(
                        color not in training_colors and color != 5
                        for color, cells in objects if cells is not selected[1]))
                    nonfive = set(int(v) for v in np.unique(grid)) - {0, 5}
                    mixed_test += int(len(nonfive) >= 2)
                    mono_test += int(len(nonfive) == 1 and 5 not in grid)
        varied_shape += len(puzzle_shapes) >= 2
        varied_bottom += len(puzzle_bottoms) >= 2
        varied_count += len(puzzle_counts) >= 2
    assert len(shapes) >= 10 and len(structures) >= 100
    assert len(signatures) >= 30 and len(palettes) >= 20
    assert min(no_separator, many, lower_start, five_top, novel_distractor,
               seen_target, close_bottom, selected_larger, novel_target,
               bbox_overlap, near_miss, stripe, lane, five_beside,
               mixed_test, mono_test) > 0
    assert min(varied_shape, varied_bottom, varied_count) >= 30
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(1234, *counts)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(1234, *counts))
        for given, actual in zip(counts, (len(puzzle["train"]), len(puzzle["test"]))):
            assert given is None or given == actual
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(counts)
