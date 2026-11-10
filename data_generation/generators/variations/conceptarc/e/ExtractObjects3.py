import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3),
    (5, 2), (5, 3), (6, 4), (7, 3), (8, 4), (10, 4),
)

_SOURCE_INPUTS = {
    "0000333330000/0222322232220/0202320232020/0222322232220/0000333330000/0222022202220/0202020202020/0222022202220/0000000000000/0222022202220/0202020202020/0222022202220/0000000000000",
    "66666000000/63336033330/63006030030/63006030030/66666033330/00000000000/00333303333/00300303003/00300303003/00333303333/00000000000",
    "0000000000000000000/0444440444440444440/0400040400040400040/0400040400040400040/0400040400040400040/0444440444440444440/0000000000000000000/4444404444404444400/4000404000404000400/4000404000404077770/4000404000404070470/4444404444404474470/0000000000000077770/0004444404444400000/0004000404000400000/0004000404000400000/0004000404000400000/0004444404444400000/0000000000000000000",
    "00000000000000/00000000000000/00800000080000/08080000808000/80008555555800/08080500805000/00800500085000/00000500005000/00008500085000/00080555555000/00800088000800/00080800808000/00008000080000/00000000000000",
    "999999999999/900000000009/900000000009/977707770009/970707070009/977707770009/900000000009/900000000009/900777000009/900707000009/900777000009/999999999999",
}


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _rows(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _frame(grid, top, left, size, color):
    grid[top, left:left + size] = color
    grid[top + size - 1, left:left + size] = color
    grid[top:top + size, left] = color
    grid[top:top + size, left + size - 1] = color


def _output(size, background, target):
    result = np.full((size, size), background, dtype=np.int8)
    _frame(result, 0, 0, size, target)
    return result


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    _need(len(order) >= 2 and counts[order[-1]] > counts[order[-2]],
          "background is not uniquely dominant")
    return int(colors[order[-1]])


def _square_candidates(grid, background):
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        rows, cols = np.where(grid == color)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(cols.min()), int(cols.max())
        size = bottom - top + 1
        if size < 3 or right - left + 1 != size:
            continue
        border = np.zeros(grid.shape, dtype=bool)
        _frame(border, top, left, size, True)
        if np.array_equal(grid == color, border):
            candidates.append((color, top, left, size))
    return candidates


def _visible_frame_sizes(grid, color):
    sizes = []
    height, width = grid.shape
    for size in range(3, 7):
        border = np.zeros((size, size), dtype=bool)
        _frame(border, 0, 0, size, True)
        for top in range(height - size + 1):
            for left in range(width - size + 1):
                patch = grid[top:top + size, left:left + size]
                if np.all(patch[border] == color) and np.all(patch[~border] != color):
                    sizes.append(size)
    return sizes


def _frame_count(grid, color):
    return len(_visible_frame_sizes(grid, color))


def _lattice(grid, rng, color, plain):
    height, width = grid.shape
    maximum = min(6, (min(height, width) - 2) // 2)
    choices = np.arange(3, maximum + 1)
    weights = np.arange(1, len(choices) + 1, dtype=float)
    size = int(rng.choice(choices, p=weights / weights.sum()))
    gap = int(rng.integers(1, 4))
    if 2 * size + gap > min(height, width):
        gap = 1
    row_start = int(rng.integers(0, min(3, height - 2 * size - gap + 1)))
    col_start = int(rng.integers(0, min(3, width - 2 * size - gap + 1)))
    rows = list(range(row_start, height - size + 1, size + gap))
    cols = list(range(col_start, width - size + 1, size + gap))
    if len(rows) < 2 or len(cols) < 2:
        rows = [0, height - size]
        cols = [0, width - size]
    sites = [(row, col) for row in rows for col in cols]
    rng.shuffle(sites)
    for index, (row, col) in enumerate(sites):
        if not plain and index >= 3 and rng.random() < 0.22:
            continue
        actual = size
        if not plain and index >= 3 and rng.random() < 0.4:
            actual = int(rng.integers(3, min(7, size + gap + 1,
                                              height - row + 1, width - col + 1)))
        _frame(grid, row, col, actual, color)
    return size


def _scattered(grid, rng, color, count, inset=0):
    height, width = grid.shape
    for _ in range(count):
        kind = int(rng.integers(0, 4))
        if kind == 0:
            size = int(rng.integers(3, min(7, height - 2 * inset + 1,
                                           width - 2 * inset + 1)))
            top = int(rng.integers(inset, height - inset - size + 1))
            left = int(rng.integers(inset, width - inset - size + 1))
            _frame(grid, top, left, size, color)
        elif kind == 1:
            radius = 2 if min(height, width) - 2 * inset < 11 else int(rng.integers(2, 4))
            cy = int(rng.integers(inset + radius, height - inset - radius))
            cx = int(rng.integers(inset + radius, width - inset - radius))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) + abs(dx) == radius:
                        grid[cy + dy, cx + dx] = color
        elif kind == 2:
            length = int(rng.integers(3, min(7, height - 2 * inset + 1,
                                             width - 2 * inset + 1)))
            top = int(rng.integers(inset, height - inset - length + 1))
            left = int(rng.integers(inset, width - inset - length + 1))
            grid[top:top + length, left] = color
            grid[top + length - 1, left:left + length] = color
        else:
            length = int(rng.integers(3, min(7, height - 2 * inset + 1,
                                             width - 2 * inset + 1)))
            top = int(rng.integers(inset, height - inset - length + 1))
            left = int(rng.integers(inset, width - inset - length + 1))
            for offset in range(length):
                grid[top + offset, left + offset] = color
                grid[top + offset, left + length - offset - 1] = color


def _position(rng, height, width, size):
    option = int(rng.integers(0, 5))
    if option == 0:
        return int(rng.integers(0, height - size + 1)), int(rng.integers(0, width - size + 1))
    if option == 1:
        if rng.random() < 0.5:
            return int(rng.choice((0, height - size))), int(rng.integers(0, width - size + 1))
        return int(rng.integers(0, height - size + 1)), int(rng.choice((0, width - size)))
    if option == 2:
        return int(rng.integers((height - size) // 2, height - size + 1)), int(rng.integers(0, width - size + 1))
    if option == 3:
        return int(rng.integers(0, height - size + 1)), int(rng.integers((width - size) // 2, width - size + 1))
    return max(0, min(height - size, height // 2 - size // 2 + int(rng.integers(-2, 3)))), max(0, min(width - size, width // 2 - size // 2 + int(rng.integers(-2, 3))))


def _pair(rng, background, target, distractor, mode, first_train=False,
          force_cleanup=False):
    if mode == "outer":
        size = int(rng.integers(11, 21))
        height = width = size
        top = left = 0
    else:
        size = int(rng.integers(4, 9)) if first_train else int(rng.integers(3, 12))
        compact = not first_train and rng.random() < 0.42 and size <= 8
        lower = max(size + 2, 9 if compact else 12)
        upper = 12 if compact else 30
        height = int(rng.integers(lower, upper + 1))
        width = int(rng.integers(lower, upper + 1))
        top, left = _position(rng, height, width, size)
    grid = np.full((height, width), background, dtype=np.int8)
    if mode in ("lattice", "mixed"):
        _lattice(grid, rng, distractor, plain=(mode == "lattice" and first_train))
    if mode in ("scattered", "mixed", "outer"):
        _scattered(grid, rng, distractor,
                   int(rng.integers(3, 8)) if mode != "outer" else int(rng.integers(2, 5)),
                   inset=1 if mode == "outer" else 0)
    intrusion = "many" if force_cleanup or mode == "outer" else rng.choice(
        ("clean", "small", "many"), p=(0.20, 0.30, 0.50))
    interior = grid[top + 1:top + size - 1, left + 1:left + size - 1]
    if intrusion in ("clean", "small"):
        interior[:] = background
    if intrusion == "small":
        index = int(rng.integers(0, interior.size))
        interior.flat[index] = distractor
    elif intrusion == "many":
        desired = min(interior.size, max(2, interior.size // 3))
        if np.count_nonzero(interior == distractor) < desired:
            cells = rng.choice(interior.size, size=desired, replace=False)
            interior.flat[cells] = distractor
    _frame(grid, top, left, size, target)
    return grid, _output(size, background, target)


def _resolve_counts(seed, rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(name + " outside its legal range")
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[int(seed) % len(_AUTO_COUNTS)]
    if num_train is None:
        return int(rng.integers(num_test + 1, 11)), int(num_test)
    if num_test is None:
        return int(num_train), int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(seed, rng, num_train, num_test)
    palette = [int(v) for v in rng.choice(10, size=3, replace=False)]
    base_background, first_target, second_target = palette
    puzzle = {"train": [], "test": []}
    used = set()
    for section, count in (("train", n_train), ("test", n_test)):
        for index in range(count):
            if section == "train" and index < 2:
                background = base_background
                target, distractor = ((first_target, second_target) if index == 0
                                      else (second_target, first_target))
            else:
                background = (base_background if rng.random() < 0.55 else
                              int(rng.choice([c for c in range(10) if c != base_background])))
                colors = [c for c in range(10) if c != background]
                target, distractor = (int(v) for v in rng.choice(colors, size=2, replace=False))
            if section == "train" and index == 0:
                mode = "lattice"
            else:
                modes = ("lattice", "scattered", "mixed", "outer")
                weights = ((0.35, 0.28, 0.28, 0.09) if section == "train" else
                           (0.25, 0.28, 0.27, 0.20))
                mode = str(rng.choice(modes, p=weights))
            for attempt in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                grid, output = _pair(pair_rng, background, target, distractor,
                                     mode, first_train=(section == "train" and index == 0),
                                     force_cleanup=(section == "train" and index < 2))
                key = (grid.shape, grid.tobytes())
                if key in used or _rows(grid) in _SOURCE_INPUTS:
                    continue
                if section == "train" and index == 1 and output.shape == puzzle["train"][0]["output"].shape:
                    continue
                try:
                    bg = _background(grid)
                    candidates = _square_candidates(grid, bg)
                    if bg != background or len(candidates) != 1:
                        continue
                    if section == "train" and index == 0 and _frame_count(grid, distractor) < 2:
                        continue
                    if mode != "outer":
                        rows, cols = np.where(grid == distractor)
                        _, square_top, square_left, square_size = candidates[0]
                        if not any(not (square_top <= row < square_top + square_size and
                                        square_left <= col < square_left + square_size)
                                   for row, col in zip(rows, cols)):
                            continue
                except AssertionError:
                    continue
                used.add(key)
                puzzle[section].append({"input": grid, "output": output})
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "pair counts")
    seen = set()
    roles = []
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, output = pair["input"], pair["output"]
            for item in (grid, output):
                _need(isinstance(item, np.ndarray) and item.ndim == 2, "grid type")
                _need(all(1 <= dimension <= 30 for dimension in item.shape), "grid size")
                _need(np.issubdtype(item.dtype, np.integer), "grid dtype")
                _need(bool(np.all((item >= 0) & (item <= 9))), "grid colors")
            key = (grid.shape, grid.tobytes())
            _need(key not in seen and _rows(grid) not in _SOURCE_INPUTS, "duplicate or source input")
            seen.add(key)
            background = _background(grid)
            colors = {int(c) for c in np.unique(grid)}
            _need(len(colors) == 3, "input must have background and two foreground colors")
            candidates = _square_candidates(grid, background)
            _need(len(candidates) == 1, "input lacks a unique complete square color")
            target, top, left, size = candidates[0]
            distractor = next(iter(colors - {background, target}))
            _need(np.array_equal(output, _output(size, background, target)),
                  "output is not the cleaned target crop")
            _need(not np.array_equal(grid, output), "unchanged pair")
            if section == "train" and index < 2:
                interior = grid[top + 1:top + size - 1, left + 1:left + size - 1]
                _need(bool(np.any(interior == distractor)), "training lacks cleanup evidence")
                roles.append((background, target, distractor))
            if section == "train" and index == 0:
                _need(_frame_count(grid, distractor) >= 2, "first training pair lacks repeated outlines")
    _need(roles[0][0] == roles[1][0] and roles[0][1] == roles[1][2] and
          roles[0][2] == roles[1][1], "training does not demonstrate color-role swap")
    _need(puzzle["train"][0]["output"].shape != puzzle["train"][1]["output"].shape,
          "first two demonstrations repeat the target size")


def _same_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    train_sizes, test_sizes, backgrounds, shapes, layouts = set(), set(), set(), set(), set()
    clean, intrusive, varying_background = 0, 0, 0
    compact = {"train": 0, "test": 0}
    outer = {"train": 0, "test": 0}
    edge = {"train": 0, "test": 0}
    repeated = {"train": 0, "test": 0}
    relative = {"train": set(), "test": set()}
    shared_sizes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_backgrounds = set()
        puzzle_sizes = {"train": set(), "test": set()}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid, output = pair["input"], pair["output"]
                background = _background(grid)
                puzzle_backgrounds.add(background)
                backgrounds.add(background)
                target, top, left, size = _square_candidates(grid, background)[0]
                _need(np.array_equal(output, _output(size, background, target)),
                      "input-only transformation failure")
                (train_sizes if section == "train" else test_sizes).add(size)
                puzzle_sizes[section].add(size)
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple(int(v != background) for v in grid.flat)))
                distractor = next(c for c in np.unique(grid) if c not in (background, target))
                compact[section] += max(grid.shape) <= 11
                outer[section] += grid.shape == (size, size) and top == 0 and left == 0
                edge[section] += (top == 0 or left == 0 or
                                  top + size == grid.shape[0] or
                                  left + size == grid.shape[1])
                frames = _visible_frame_sizes(grid, distractor)
                repeated[section] += len(frames) >= 2
                if frames:
                    typical = max(set(frames), key=frames.count)
                    relative[section].add(int(np.sign(size - typical)))
                interior = grid[top + 1:top + size - 1, left + 1:left + size - 1]
                amount = int(np.count_nonzero(interior == distractor))
                clean += amount == 0
                intrusive += amount >= 2
        varying_background += len(puzzle_backgrounds) > 1
        shared_sizes += bool(puzzle_sizes["train"] & puzzle_sizes["test"])
    _need(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _need(len(train_sizes) >= 6 and len(test_sizes) >= 6, "size diversity")
    _need(len(backgrounds) >= 7 and varying_background >= 8, "background diversity")
    _need(len(shapes) >= 40 and len(layouts) >= 150, "structural diversity")
    _need(clean >= 5 and intrusive >= 50, "interference diversity")
    _need(all(compact[s] >= 5 and outer[s] >= 5 and edge[s] >= 15 and
              repeated[s] >= 20 and relative[s] == {-1, 0, 1}
              for s in ("train", "test")), "regime and scale diversity")
    _need(shared_sizes >= 8, "training and test scales never overlap")
    for counts in ((2, 1), (10, 1), (10, 4), (6, 3),
                   (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(117, *counts)
        second = generate_puzzle(117, *counts)
        _need(_same_puzzles(first, second), "requested-count nondeterminism")
        validate_generated_puzzle(first)
    _need(_same_puzzles(generate_puzzle(19), generate_puzzle(19)),
          "automatic-count nondeterminism")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(5, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
