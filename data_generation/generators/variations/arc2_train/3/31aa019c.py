"""Generate grids for the global singleton and centered marker ARC rule."""

import numpy as np


# Exact source inputs, used only to reject accidental copies.
_SOURCE_INPUT_ROWS = (
    ("2771030003", "0009000037", "0001000609", "0000000100",
     "9000000000", "0000200030", "0507300010", "4400010005",
     "0000000530", "0000450000"),
    ("0001000500", "0000000000", "2000020001", "0010000005",
     "0080000000", "0000000000", "0400000000", "0510100000",
     "0810001030", "0000000300"),
    ("6000800000", "0000000028", "0700205020", "0901000000",
     "0900000001", "0000060000", "0107000000", "0000000000",
     "0000003000", "0050000000"),
    ("0000257000", "0005602000", "0000003000", "0080300008",
     "7477400004", "0008007000", "0000090400", "5503006707",
     "0030000002", "1010000067"),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _sample_shape(rng, band="any"):
    if band == "small":
        return int(rng.integers(5, 11)), int(rng.integers(5, 11))
    if band == "large":
        if rng.random() < 0.10:
            return 30, 30
        return int(rng.integers(20, 31)), int(rng.integers(20, 31))
    mode = int(rng.choice(4, p=[0.18, 0.20, 0.46, 0.16]))
    if mode == 0:
        return int(rng.integers(5, 10)), int(rng.integers(5, 10))
    if mode == 1:
        shape = (int(rng.integers(5, 10)), int(rng.integers(17, 31)))
        return shape if rng.random() < 0.5 else shape[::-1]
    if mode == 2:
        return int(rng.integers(8, 23)), int(rng.integers(8, 23))
    return _sample_shape(rng, "large")


def _background(rng, excluded):
    available = [color for color in range(10) if color not in excluded]
    # Familiar black backgrounds remain common, while every literal color can
    # also play the background role.
    if 0 in available and rng.random() < 0.38:
        return 0
    return int(rng.choice(available))


def _singleton(grid, background):
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    candidates = [color for color in range(10)
                  if color != background and counts[color] == 1]
    if len(candidates) != 1:
        raise ValueError("there must be exactly one nonbackground singleton color")
    color = candidates[0]
    row, col = np.argwhere(grid == color)[0]
    return color, int(row), int(col), counts


def _output_from_input(grid, background, frame):
    color, row, col, _ = _singleton(grid, background)
    if not (1 <= row < grid.shape[0] - 1 and 1 <= col < grid.shape[1] - 1):
        raise ValueError("target must have a one-cell margin")
    output = np.full(grid.shape, background, dtype=np.int8)
    output[row - 1:row + 2, col - 1:col + 2] = frame
    output[row, col] = color
    return output


def _sample_density(rng, split, source_like):
    if source_like:
        bounds = (0.13, 0.29) if split == "train" else (0.34, 0.48)
    else:
        weights = [0.22, 0.42, 0.36] if split == "train" else [0.19, 0.40, 0.41]
        mode = int(rng.choice(3, p=weights))
        bounds = ((0.10, 0.19), (0.19, 0.34), (0.34, 0.48))[mode]
    return float(rng.uniform(*bounds))


def _distractor_counts(rng, colors, total):
    counts = np.full(len(colors), 2, dtype=np.int64)
    extra = total - int(counts.sum())
    if extra == 0:
        return counts
    mode = int(rng.choice(3, p=[0.42, 0.29, 0.29]))
    if mode == 0:  # ordinary scatter with variable counts
        weights = rng.dirichlet(np.ones(len(colors)))
        counts += rng.multinomial(extra, weights)
    elif mode == 1:  # leave several exact doubletons
        order = rng.permutation(len(colors))
        fixed = max(1, min(len(colors) - 1, int(rng.integers(1, 5))))
        receivers = order[fixed:]
        if len(receivers) == 0:
            receivers = order[-1:]
        counts[receivers] += rng.multinomial(extra, np.full(len(receivers), 1 / len(receivers)))
    else:  # a conspicuously frequent repeated color
        major = int(rng.integers(len(colors)))
        counts[major] += max(1, int(round(0.72 * extra)))
        remaining = total - int(counts.sum())
        if remaining:
            counts += rng.multinomial(remaining, np.full(len(colors), 1 / len(colors)))
    return counts


def _sites(rng, height, width, row, col, total, neighbors, style):
    near = [(r, c) for r in range(row - 1, row + 2)
            for c in range(col - 1, col + 2) if (r, c) != (row, col)]
    chosen_near = [near[int(i)] for i in rng.choice(8, size=neighbors, replace=False)]
    outside = [(r, c) for r in range(height) for c in range(width)
               if (r, c) != (row, col) and (r, c) not in near]
    need = total - neighbors
    run_groups = []
    selected = []
    if style == "runs" and need >= 2:
        available = set(outside)
        group_goal = min(3, need // 2)
        for _ in range(group_goal):
            if need - len(selected) < 2:
                break
            length = min(int(rng.integers(2, 5)), need - len(selected))
            candidates = []
            for r, c in available:
                for dr, dc in ((1, 0), (0, 1)):
                    segment = [(r + dr * j, c + dc * j) for j in range(length)]
                    if all(point in available for point in segment):
                        candidates.append(segment)
            if not candidates:
                break
            group = candidates[int(rng.integers(len(candidates)))]
            selected.extend(group)
            run_groups.append(group)
            available.difference_update(group)
    remaining = need - len(selected)
    pool = [point for point in outside if point not in selected]
    if remaining:
        if style == "clusters":
            anchors = [pool[int(i)] for i in rng.choice(len(pool),
                       size=min(int(rng.integers(2, 5)), len(pool)), replace=False)]
            scores = [min(abs(r - ar) + abs(c - ac) for ar, ac in anchors)
                      + float(rng.uniform(0, 2.6)) for r, c in pool]
            indices = np.argsort(scores)[:remaining]
        else:
            indices = rng.choice(len(pool), size=remaining, replace=False)
        selected.extend(pool[int(i)] for i in indices)
    return chosen_near + selected, run_groups


def _color_sites(rng, sites, colors, counts, style, run_groups):
    assignments = {}
    left = {int(color): int(count) for color, count in zip(colors, counts)}
    if style == "runs":
        for group in run_groups:
            eligible = [color for color in colors if left[color] >= len(group)]
            if not eligible:
                continue
            color = int(rng.choice(eligible))
            for point in group:
                assignments[point] = color
            left[color] -= len(group)
    remaining = [point for point in sites if point not in assignments]
    if style == "clusters":
        order = list(rng.permutation(colors))
        for color in order:
            color = int(color)
            count = left[color]
            if count == 0:
                continue
            anchors = [remaining[int(i)] for i in rng.choice(len(remaining),
                       size=min(2 if rng.random() < 0.45 else 1, len(remaining)),
                       replace=False)]
            scores = [min(abs(r - ar) + abs(c - ac) for ar, ac in anchors)
                      + float(rng.uniform(0, 1.5)) for r, c in remaining]
            indices = set(int(i) for i in np.argsort(scores)[:count])
            for i in indices:
                assignments[remaining[i]] = color
            remaining = [point for i, point in enumerate(remaining) if i not in indices]
    else:
        values = np.repeat(np.asarray(colors, dtype=np.int8),
                           np.asarray([left[color] for color in colors], dtype=np.int64))
        rng.shuffle(values)
        rng.shuffle(remaining)
        for point, value in zip(remaining, values):
            assignments[point] = int(value)
    return assignments


def _make_pair(rng, split, frame, target, mandatory, shape, source_like):
    height, width = shape
    area = height * width
    background = _background(rng, {frame, target, *mandatory})
    required = {color for color in mandatory if color not in (background, target)}
    force_full_palette = source_like and split == "test"
    if force_full_palette:
        required = set(range(10)) - {background, target}
    elif rng.random() < 0.52:
        required.add(frame)
    density = _sample_density(rng, split, source_like)
    max_foreground = (area - 1) // 2  # background strictly exceeds half the canvas
    min_foreground = 1 + 2 * len(required)
    desired = int(round(area * density))
    foreground = min(max_foreground, max(min_foreground, desired))
    if foreground < min_foreground:
        raise ValueError("shape cannot accommodate mandatory repeated colors")
    max_distractors = min(8, (foreground - 1) // 2)
    min_distractors = max(1, len(required))
    if force_full_palette:
        distractor_count = 8
    elif rng.random() < 0.23:
        distractor_count = max_distractors
    else:
        distractor_count = int(rng.integers(min_distractors, max_distractors + 1))
    available = [color for color in range(10)
                 if color not in {background, target, *required}]
    extras = list(rng.choice(available, size=distractor_count - len(required),
                             replace=False))
    distractors = sorted(int(color) for color in required.union(extras))
    counts = _distractor_counts(rng, distractors, foreground - 1)
    row = int(rng.integers(1, height - 1))
    col = int(rng.integers(1, width - 1))
    max_neighbors = min(8, foreground - 1)
    choices = np.arange(max_neighbors + 1)
    weights = np.asarray([0.12, 0.14, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08, 0.08])[:len(choices)]
    weights = weights / weights.sum()
    neighbors = int(rng.choice(choices, p=weights))
    style = str(rng.choice(["scatter", "clusters", "runs"], p=[0.43, 0.31, 0.26]))
    sites, run_groups = _sites(rng, height, width, row, col,
                               foreground - 1, neighbors, style)
    assignments = _color_sites(rng, sites, distractors, counts, style, run_groups)
    grid = np.full((height, width), background, dtype=np.int8)
    for (r, c), color in assignments.items():
        grid[r, c] = color
    grid[row, col] = target
    return {"input": grid, "output": _output_from_input(grid, background, frame)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    frame = int(rng.integers(10))
    possible_targets = [color for color in range(10) if color != frame]
    rng.shuffle(possible_targets)
    pool_size = int(rng.integers(2, min(4, num_train) + 1))
    train_pool = possible_targets[:pool_size]
    train_targets = train_pool.copy()
    train_targets.extend(int(rng.choice(train_pool)) for _ in range(num_train - pool_size))
    rng.shuffle(train_targets)
    source_like = bool(rng.random() < 0.24)
    novel_pool = [color for color in possible_targets if color not in train_pool]
    test_targets = []
    for _ in range(num_test):
        familiar = not source_like and rng.random() < 0.48
        test_targets.append(int(rng.choice(train_pool if familiar else novel_pool)))

    # Training target colors recur as distractors in another demonstration.
    # Each new test target also occurs as a repeated training distractor, but
    # does not have to occur in every training input.
    mandatory = [set() for _ in range(num_train)]
    for index, target in enumerate(train_targets):
        mandatory[index].add(train_pool[(train_pool.index(target) + 1) % pool_size])
    for index, target in enumerate(sorted(set(test_targets) - set(train_pool))):
        mandatory[index % num_train].add(target)

    shared_shape = bool(rng.random() < 0.19)
    size_shift = int(rng.choice(3, p=[0.60, 0.20, 0.20])) if not shared_shape else 0
    if shared_shape:
        common_shape = _sample_shape(rng, "large" if source_like and rng.random() < 0.25
                                     else "any")
        if source_like and common_shape[0] * common_shape[1] < 50:
            common_shape = _sample_shape(rng, "large")
    pairs = {"train": [], "test": []}
    input_keys = set()
    for split, targets in (("train", train_targets), ("test", test_targets)):
        for index, target in enumerate(targets):
            words = rng.integers(0, 2**32, size=4, dtype=np.uint32)
            pair_rng = np.random.default_rng(np.random.SeedSequence(
                [int(word) for word in words] + [index, 0 if split == "train" else 1]))
            if shared_shape:
                shape = common_shape
            elif size_shift == 1:
                shape = _sample_shape(pair_rng, "small" if split == "train" else "large")
            elif size_shift == 2:
                shape = _sample_shape(pair_rng, "large" if split == "train" else "small")
            else:
                shape = _sample_shape(pair_rng)
            if source_like and split == "test" and shape[0] * shape[1] < 50:
                shape = _sample_shape(pair_rng, "large")
            # A unique input is overwhelmingly likely; a bounded redraw handles
            # rare collisions on tiny canvases without a seed-dependent failure.
            for _ in range(8):
                pair = _make_pair(pair_rng, split, frame, target,
                                  mandatory[index] if split == "train" else set(),
                                  shape, source_like)
                key = _grid_key(pair["input"])
                if key not in input_keys and not _source_input(pair["input"]):
                    break
            else:
                raise ValueError("could not draw a unique input")
            input_keys.add(key)
            pairs[split].append(pair)
    validate_generated_puzzle(pairs)
    return pairs


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _source_input(grid):
    if grid.shape != (10, 10):
        return False
    return tuple("".join(str(int(value)) for value in row) for row in grid) in _SOURCE_INPUT_ROWS


def _pair_analysis(pair, label):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             f"{label} must contain input and output")
    for role in ("input", "output"):
        grid = pair[role]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 f"{label}.{role} must be a two-dimensional ndarray")
        _require(all(1 <= side <= 30 for side in grid.shape), f"{label}.{role} has invalid shape")
        _require(np.issubdtype(grid.dtype, np.integer), f"{label}.{role} is not integer")
        _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label}.{role} has invalid color")
    grid, output = pair["input"], pair["output"]
    _require(grid.shape == output.shape, f"{label} changes dimensions")
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > grid.size // 2, f"{label} lacks a dominant background")
    try:
        target, row, col, counts = _singleton(grid, background)
    except ValueError as exc:
        raise AssertionError(f"{label}: {exc}") from exc
    _require(1 <= row < grid.shape[0] - 1 and 1 <= col < grid.shape[1] - 1,
             f"{label} target lacks marker margin")
    _require(all(counts[color] == 0 or counts[color] >= 2
                 for color in range(10) if color not in (background, target)),
             f"{label} has another singleton")
    ring = output[row - 1:row + 2, col - 1:col + 2].copy()
    ring[1, 1] = -1
    frame = int(ring[0, 0])
    _require(frame != background and frame != target, f"{label} conflates color roles")
    _require(np.array_equal(output, _output_from_input(grid, background, frame)),
             f"{label} violates the marker transformation")
    _require(not np.array_equal(grid, output), f"{label} has identical input and output")
    _require(not _source_input(grid), f"{label} copies a source input")
    neighbors = int(np.count_nonzero(grid[row - 1:row + 2, col - 1:col + 2] != background)) - 1
    return {"background": background, "frame": frame, "target": target,
            "counts": counts, "neighbors": neighbors, "shape": grid.shape,
            "density": (grid.size - counts[background]) / grid.size}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[split], list) for split in ("train", "test")),
             "train and test must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "invalid pair counts")
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            analysis = _pair_analysis(pair, label)
            key = _grid_key(pair["input"])
            _require(key not in seen, f"{label} duplicates an input")
            seen.add(key)
            analyses[split].append(analysis)
    frames = {a["frame"] for split in analyses for a in analyses[split]}
    _require(len(frames) == 1, "frame color must be learned once per puzzle")
    train_targets = {a["target"] for a in analyses["train"]}
    _require(len(train_targets) >= 2, "training needs multiple target colors")
    for target in train_targets:
        _require(any(a["target"] != target and a["counts"][target] >= 2
                     for a in analyses["train"]),
                 "a training target color never recurs as a distractor")
    for target in {a["target"] for a in analyses["test"]} - train_targets:
        _require(any(a["counts"][target] >= 2 for a in analyses["train"]),
                 "a new test target color was not shown as a distractor")


def _predict_from_training(train, test_input):
    # Independent of the test label and generation state. This is a local
    # recoverability check, not a protected reference solver.
    frames = []
    for example in train:
        grid = example["input"]
        counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
        background = int(np.argmax(counts))
        _, row, col, _ = _singleton(grid, background)
        frames.append(int(example["output"][row - 1, col - 1]))
    _require(len(set(frames)) == 1, "training does not establish one frame color")
    counts = np.bincount(test_input.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    return _output_from_input(test_input, background, frames[0])


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(first[split], second[split]) for role in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    frames, backgrounds, shapes, structures = set(), set(), set(), set()
    neighbors_by_split = {"train": set(), "test": set()}
    shared, mixed, bigger, smaller = 0, 0, 0, 0
    full_train, reduced_test, sparse_test, familiar_test, novel_test = 0, 0, 0, 0, 0
    absent_frame, present_frame, doubletons, skewed = 0, 0, 0, 0
    low_density_test, high_density_test = 0, 0
    target_zero, target_two, small, narrow, large, extent_30, full_30 = 0, 0, 0, 0, 0, 0, 0
    edge_contact, orthogonal_single, diagonal_single, runs_of_three = 0, 0, 0, 0
    mixed_backgrounds, repeated_coordinates = 0, 0
    combined_small, combined_large = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_targets = set()
        analyses = {"train": [], "test": []}
        coordinates = []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                a = _pair_analysis(pair, f"{split}[{index}]")
                analyses[split].append(a)
                frames.add(a["frame"])
                backgrounds.add(a["background"])
                shapes.add(a["shape"])
                structures.add(_grid_key(pair["input"]))
                neighbors_by_split[split].add(a["neighbors"])
                grid = pair["input"]
                height, width = grid.shape
                _, row, col, _ = _singleton(grid, a["background"])
                coordinates.append((row, col))
                target_zero += int(a["target"] == 0)
                target_two += int(a["target"] == 2)
                small += int(height <= 8 and width <= 8)
                narrow += int(max(height, width) >= 2 * min(height, width))
                large += int(height >= 20 and width >= 20)
                extent_30 += int(height == 30 or width == 30)
                full_30 += int(height == 30 and width == 30)
                edge_contact += int(row in (1, height - 2) or col in (1, width - 2))
                if a["neighbors"] == 1:
                    near = [(r, c) for r in range(row - 1, row + 2)
                            for c in range(col - 1, col + 2)
                            if (r, c) != (row, col) and grid[r, c] != a["background"]]
                    orthogonal_single += int(abs(near[0][0] - row) + abs(near[0][1] - col) == 1)
                    diagonal_single += int(abs(near[0][0] - row) + abs(near[0][1] - col) == 2)
                run = any(grid[r, c] == grid[r, c + 1] == grid[r, c + 2] != a["background"]
                          for r in range(height) for c in range(width - 2))
                run |= any(grid[r, c] == grid[r + 1, c] == grid[r + 2, c] != a["background"]
                           for r in range(height - 2) for c in range(width))
                runs_of_three += int(run)
                if a["counts"][a["frame"]] >= 2:
                    present_frame += 1
                else:
                    absent_frame += 1
                repeated = [int(a["counts"][color]) for color in range(10)
                            if color not in (a["background"], a["target"])
                            and a["counts"][color] >= 2]
                doubletons += sum(count == 2 for count in repeated)
                skewed += int(len(repeated) >= 2 and max(repeated) >= 3 * min(repeated))
                combined_small += int(height <= 8 and width <= 8 and a["neighbors"] == 1
                                      and a["counts"][a["frame"]] == 0)
                combined_large += int(height >= 20 and width >= 20
                                      and len(np.unique(grid)) == 10
                                      and len(repeated) >= 2
                                      and max(repeated) >= 3 * min(repeated)
                                      and run)
                if split == "train":
                    train_targets.add(a["target"])
                    full_train += int(len(np.unique(pair["input"])) == 10)
                else:
                    _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                            pair["output"]), "test output is not recoverable")
                    reduced_test += int(len(np.unique(pair["input"])) <= 8)
                    sparse_test += int(a["density"] < 0.22)
                    familiar_test += int(a["target"] in train_targets)
                    novel_test += int(a["target"] not in train_targets)
                    low_density_test += int(a["density"] < min(x["density"] for x in analyses["train"]))
                    high_density_test += int(a["density"] > max(x["density"] for x in analyses["train"]))
        all_shapes = {a["shape"] for split in analyses for a in analyses[split]}
        mixed_backgrounds += int(len({a["background"] for split in analyses
                                      for a in analyses[split]}) > 1)
        repeated_coordinates += int(len(set(coordinates)) < len(coordinates))
        shared += int(len(all_shapes) == 1)
        mixed += int(len(all_shapes) > 1)
        train_areas = [np.prod(a["shape"]) for a in analyses["train"]]
        test_areas = [np.prod(a["shape"]) for a in analyses["test"]]
        bigger += int(min(test_areas) > max(train_areas))
        smaller += int(max(test_areas) < min(train_areas))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(frames) >= 7 and len(backgrounds) >= 8, "color roles lack diversity")
    _require(len(shapes) >= 55 and len(structures) >= 300, "layouts or sizes lack diversity")
    _require(shared >= 2 and mixed >= 20 and bigger >= 2 and smaller >= 2,
             "size regimes lack diversity")
    _require({0, 1, 2, 8}.issubset(neighbors_by_split["train"] | neighbors_by_split["test"]),
             "target neighborhoods lack requested range")
    _require(full_train >= 2 and reduced_test >= 5 and sparse_test >= 2,
             "palette or density regimes lack diversity")
    _require(familiar_test >= 3 and novel_test >= 3 and low_density_test >= 2
             and high_density_test >= 2, "train/test shifts lack diversity")
    _require(absent_frame >= 5 and present_frame >= 5 and doubletons >= 20
             and skewed >= 8, "distractor regimes lack diversity")
    _require(target_zero >= 2 and target_two >= 2 and mixed_backgrounds >= 20,
             "literal target colors or within-puzzle backgrounds lack diversity")
    _require(small >= 5 and narrow >= 5 and large >= 5 and extent_30 >= 2
             and full_30 >= 1
             and edge_contact >= 10 and repeated_coordinates >= 2,
             "shape or position coverage is too narrow")
    _require(orthogonal_single >= 2 and diagonal_single >= 2 and runs_of_three >= 10,
             "contact direction or distractor runs lack diversity")
    _require(combined_small >= 1 and combined_large >= 2,
             "compatible variation axes do not combine")

    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(200000 + 10 * train_count + test_count,
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit counts were ignored")
    for offset, (train_count, test_count) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(300000 + offset, train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require(train_count is None or len(puzzle["train"]) == train_count,
                 "partial train count was ignored")
        _require(test_count is None or len(puzzle["test"]) == test_count,
                 "partial test count was ignored")
    first = generate_puzzle(123456, 7, 3)
    _require(_puzzles_equal(first, generate_puzzle(123456, 7, 3)), "nondeterministic call")
    _require(not _puzzles_equal(first, generate_puzzle(123457, 7, 3)),
             "different seeds produced the same puzzle")
    bad = {split: [{role: pair[role].copy() for role in ("input", "output")}
                   for pair in first[split]] for split in ("train", "test")}
    a = _pair_analysis(bad["train"][0], "bad")
    color, row, col, _ = _singleton(bad["train"][0]["input"], a["background"])
    bad["train"][0]["input"][row, col] = a["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an erased singleton")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (2.5, 1), (True, 1)):
        try:
            generate_puzzle(400000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
