"""Complete, naturally cropped diamond wavefront puzzles."""

import hashlib

import numpy as np


MAX_EXPLANATION_RADIUS = 7


def _diamond(height, width, row, col, radius):
    return frozenset(
        (r, c)
        for r in range(max(0, row - radius), min(height, row + radius + 1))
        for c in range(max(0, col - radius), min(width, col + radius + 1))
        if abs(r - row) + abs(c - col) == radius
    )


def _update(height, width, row, col, radius, role):
    if role == "stationary":
        return _diamond(height, width, row, col, radius)
    if role == "contract":
        assert radius >= 1
        return _diamond(height, width, row, col, radius - 1)
    assert role == "expand"
    return _diamond(height, width, row, col, radius + 1)


def _cells(grid, color):
    rows, cols = np.where(grid == color)
    return frozenset(zip(rows.tolist(), cols.tolist()))


def _components(cells):
    unseen = set(cells)
    found = []
    while unseen:
        start = unseen.pop()
        group = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    other = (row + dr, col + dc)
                    if other in unseen:
                        unseen.remove(other)
                        group.add(other)
                        stack.append(other)
        found.append(frozenset(group))
    return found


def _descriptors(component, height, width):
    """Infer centers from visible cells alone, including clipped contours."""
    first_row, first_col = next(iter(component))
    result = []
    for row in range(height):
        for col in range(width):
            radius = abs(first_row - row) + abs(first_col - col)
            if radius > MAX_EXPLANATION_RADIUS:
                continue
            if any(abs(r - row) + abs(c - col) != radius for r, c in component):
                continue
            if _diamond(height, width, row, col, radius) == component:
                result.append((row, col, radius))
    return result


def _component_prediction(component, height, width, role):
    possible = {
        _update(height, width, row, col, radius, role)
        for row, col, radius in _descriptors(component, height, width)
        if role != "contract" or radius >= 1
    }
    if len(possible) != 1:
        return None
    return next(iter(possible))


def _predict_color(grid, color, role):
    height, width = grid.shape
    result = set()
    for component in _components(_cells(grid, color)):
        prediction = _component_prediction(component, height, width, role)
        if prediction is None or not prediction or result.intersection(prediction):
            return None
        result.update(prediction)
    return frozenset(result)


def _near(first, second):
    return any(
        abs(r - s) <= 1 and abs(c - d) <= 1
        for r, c in first
        for s, d in second
    )


class _Builder:
    def __init__(self, height, width, background):
        self.height = height
        self.width = width
        self.background = background
        self.objects = []

    def add(self, color, role, row, col, radius):
        source = _diamond(self.height, self.width, row, col, radius)
        target = _update(self.height, self.width, row, col, radius, role)
        if not source or not target or (role != "stationary" and source == target):
            return False
        if len(_components(source)) != 1 or len(_components(target)) != 1:
            return False
        if _component_prediction(source, self.height, self.width, role) != target:
            return False
        for old_color, old_source, old_target in self.objects:
            if source & old_source or target & old_target:
                return False
            if old_color == color and (_near(source, old_source) or _near(target, old_target)):
                return False
        self.objects.append((color, source, target))
        return True

    def finish(self):
        source = np.full((self.height, self.width), self.background, dtype=np.int8)
        target = source.copy()
        for color, input_cells, output_cells in self.objects:
            for row, col in input_cells:
                source[row, col] = color
            for row, col in output_cells:
                target[row, col] = color
        return {"input": source, "output": target}


def _clip_sides(row, col, radius, height, width):
    return (
        row - radius < 0,
        col + radius >= width,
        row + radius >= height,
        col - radius < 0,
    )


def _place(builder, rng, color, role, radius, clipping):
    height, width = builder.height, builder.width
    margin = radius + (role == "expand")
    candidates = []
    for row in range(height):
        for col in range(width):
            flags = _clip_sides(row, col, radius, height, width)
            clipped = sum(flags)
            if clipping == "interior":
                if any(_clip_sides(row, col, margin, height, width)):
                    continue
            elif clipping == "side":
                if clipped != 1:
                    continue
            elif clipping == "corner":
                if clipped != 2 or not any(flags[i] and flags[(i + 1) % 4] for i in range(4)):
                    continue
            else:
                raise AssertionError("Unknown clipping mode")
            candidates.append((row, col))
    if not candidates:
        return False
    for index in rng.permutation(len(candidates)):
        row, col = candidates[int(index)]
        if builder.add(color, role, row, col, radius):
            return True
    return False


def _random_clipping(rng, radius):
    if radius < 2:
        return "interior"
    return str(rng.choice(("interior", "interior", "interior", "side", "side", "corner")))


def _dimensions(rng, regime, index):
    if regime == "train" and index == 1:
        return int(rng.integers(23, 31)), int(rng.integers(23, 31))
    if regime == "train" and index == 0:
        if rng.random() < 0.5:
            return 16, 16
        return int(rng.integers(15, 22)), int(rng.integers(15, 22))
    category = int(rng.choice(4, p=(0.20, 0.27, 0.28, 0.25)))
    if category == 0:
        return 16, 16
    if category == 1:
        return int(rng.integers(12, 19)), int(rng.integers(12, 20))
    if category == 2:
        return int(rng.integers(18, 25)), int(rng.integers(17, 26))
    return int(rng.integers(24, 31)), int(rng.integers(23, 31))


def _stationary_schedule(rng, count):
    first = int(rng.integers(0, 3))
    if first == 0:
        result = [(0, 1), ()]
    elif first == 1:
        result = [(0,), (1,)]
    else:
        result = [(), (0, 1)]
    for _ in range(2, count):
        result.append(tuple(i for i in range(2) if rng.random() < 0.5))
    return result


def _specifications(rng, roles, regime, index, stationary_indices):
    contract, expand = roles["contract"], roles["expand"]
    specs = []
    if regime == "train" and index == 0:
        specs.extend(((contract, "contract", int(rng.integers(2, 5)), "interior"),
                      (expand, "expand", int(rng.integers(1, 4)), "interior")))
        extra_count = int(rng.integers(0, 3))
    elif regime == "train" and index == 1:
        clipped_role = str(rng.choice(("contract", "expand")))
        clipped_color = roles[clipped_role]
        specs.extend(((contract, "contract", 1, "interior"),
                      (expand, "expand", 0, "interior"),
                      (clipped_color, clipped_role, int(rng.integers(3, 7)),
                       str(rng.choice(("side", "side", "corner"))))))
        extra_count = int(rng.integers(0, 2))
    else:
        cr = int(rng.integers(1, 7))
        er = int(rng.integers(0, 6))
        specs.extend(((contract, "contract", cr, _random_clipping(rng, cr)),
                      (expand, "expand", er, _random_clipping(rng, er))))
        extra_count = int(rng.integers(0, 3))
    for _ in range(extra_count):
        role = str(rng.choice(("contract", "expand")))
        radius = int(rng.integers(1, 6)) if role == "contract" else int(rng.integers(0, 5))
        specs.append((roles[role], role, radius, _random_clipping(rng, radius)))
    for stationary_index in stationary_indices:
        radius = int(rng.choice((0, 0, 1, 2)))
        specs.append((roles["stationary"][stationary_index], "stationary", radius,
                      _random_clipping(rng, radius)))
    specs.sort(key=lambda spec: spec[2] + (spec[1] == "expand"), reverse=True)
    return specs


def _make_pair(rng, roles, regime, index, stationary_indices):
    for _ in range(80):
        height, width = _dimensions(rng, regime, index)
        builder = _Builder(height, width, roles["background"])
        specs = _specifications(rng, roles, regime, index, stationary_indices)
        if all(_place(builder, rng, *spec) for spec in specs):
            return builder.finish()
    raise RuntimeError("Could not place a distinct set of contours")


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(name + " is invalid")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(array):
    return array.shape, array.tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    colors = [int(color) for color in rng.permutation(10)[:5]]
    roles = {
        "background": colors[0],
        "contract": colors[1],
        "expand": colors[2],
        "stationary": tuple(colors[3:]),
    }
    schedule = _stationary_schedule(rng, train_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            stationary = schedule[index] if regime == "train" else tuple(
                i for i in range(2) if rng.random() < 0.5
            )
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, roles, regime, index, stationary)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a unique pair")
    return puzzle


def _modal(grid):
    return int(np.argmax(np.bincount(grid.ravel().astype(np.int64), minlength=10)))


def _infer_roles(training, background):
    foreground = set()
    for pair in training:
        foreground.update(np.unique(pair["input"]).tolist())
        foreground.update(np.unique(pair["output"]).tolist())
    foreground.discard(background)
    assert len(foreground) == 4
    roles = {"contract": [], "expand": [], "stationary": []}
    for color in foreground:
        observed = [pair for pair in training if _cells(pair["input"], color) or _cells(pair["output"], color)]
        assert observed and all(_cells(pair["input"], color) for pair in observed)
        if all(_cells(pair["input"], color) == _cells(pair["output"], color) for pair in observed):
            roles["stationary"].append(color)
            continue
        matches = [
            role for role in ("contract", "expand")
            if all(_predict_color(pair["input"], color, role) == _cells(pair["output"], color)
                   for pair in observed)
        ]
        assert len(matches) == 1
        roles[matches[0]].append(color)
    assert len(roles["contract"]) == len(roles["expand"]) == 1
    assert len(roles["stationary"]) == 2
    return {
        "contract": roles["contract"][0],
        "expand": roles["expand"][0],
        "stationary": tuple(sorted(roles["stationary"])),
    }


def _source_digest(grid):
    return hashlib.sha256(bytes([grid.shape[0], grid.shape[1]]) + grid.astype(np.uint8).tobytes()).hexdigest()


_SOURCE_INPUT_DIGESTS = frozenset((
    "9fbca5c30caed6c930f323427e353630807910ee8952f5872f7d34fa991125a7",
    "138395c4e4320862f33eae4b4c1a0f10f601b2f4fbdf274ee00240a510f19f68",
    "47b1ec668d942a276c4df3e434a6846e88d66fb5017c417a1b988e21e15dbbbf",
    "6f6782236e81ea47d2cc971f4b446f18cb681648dad7b5c2255852c2a56d0931",
))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    training, testing = puzzle["train"], puzzle["test"]
    assert isinstance(training, list) and isinstance(testing, list)
    assert 2 <= len(training) <= 10 and 1 <= len(testing) <= 4 and len(training) > len(testing)
    seen_inputs, seen_pairs = set(), set()
    all_pairs = training + testing
    backgrounds = set()
    for pair in all_pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        source, target = pair["input"], pair["output"]
        for grid in (source, target):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            backgrounds.add(_modal(grid))
        assert source.shape == target.shape and not np.array_equal(source, target)
        assert _source_digest(source) not in _SOURCE_INPUT_DIGESTS
        input_key = _key(source)
        pair_key = (input_key, _key(target))
        assert input_key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
    assert len(backgrounds) == 1
    background = next(iter(backgrounds))
    roles = _infer_roles(training, background)
    foreground = {roles["contract"], roles["expand"], *roles["stationary"]}
    train_stationary_seen = set()
    clean_contract = clean_expand = small_contract = seed_expand = clipped_train = False
    for split, pairs in (("train", training), ("test", testing)):
        for pair in pairs:
            source, target = pair["input"], pair["output"]
            present = set(np.unique(source).tolist()) - {background}
            assert present <= foreground and {roles["contract"], roles["expand"]} <= present
            assert set(np.unique(target).tolist()) == present | {background}
            for role in ("contract", "expand"):
                color = roles[role]
                assert _predict_color(source, color, role) == _cells(target, color)
            for color in roles["stationary"]:
                assert _predict_color(source, color, "stationary") == _cells(target, color)
                if split == "train" and color in present:
                    train_stationary_seen.add(color)
            if split == "train":
                height, width = source.shape
                for role in ("contract", "expand"):
                    for component in _components(_cells(source, roles[role])):
                        descriptions = _descriptors(component, height, width)
                        assert descriptions
                        for row, col, radius in descriptions:
                            clipped = any(_clip_sides(row, col, radius, height, width))
                            clipped_train |= clipped
                            if role == "contract":
                                small_contract |= radius == 1
                                clean_contract |= radius >= 2 and not clipped
                            else:
                                seed_expand |= radius == 0
                                clean_expand |= radius >= 1 and not any(
                                    _clip_sides(row, col, radius + 1, height, width)
                                )
    assert train_stationary_seen == set(roles["stationary"])
    assert clean_contract and clean_expand and small_contract and seed_expand and clipped_train


def _equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a[field], b[field])
            for a, b in zip(first[split], second[split])
            for field in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    one = generate_puzzle(777, 7, 3)
    two = generate_puzzle(777, 7, 3)
    validate_generated_puzzle(one)
    assert _equal(one, two)
    assert not _equal(one, generate_puzzle(778, 7, 3))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(901 + (train_count or 0) * 10 + (test_count or 0),
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(103, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid pair counts accepted")
    train_counts, test_counts, shapes, signatures, conventions = set(), set(), set(), set(), set()
    stationary_counts, clipped_sides = set(), set()
    dimension_classes = set()
    interior_tests = clipped_contract_tests = clipped_expand_tests = 0
    small_contract_tests = singleton_expand_tests = multi_action_tests = 0
    large_contract_train = adjacent_color_pairs = dense_pairs = 0
    varied_shapes = varied_stationary = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        background = _modal(puzzle["train"][0]["input"])
        roles = _infer_roles(puzzle["train"], background)
        conventions.add((background, roles["contract"], roles["expand"], roles["stationary"]))
        puzzle_shapes, puzzle_stationary = set(), set()
        for split, pair in (
            [("train", pair) for pair in puzzle["train"]]
            + [("test", pair) for pair in puzzle["test"]]
        ):
            source = pair["input"]
            shapes.add(source.shape)
            puzzle_shapes.add(source.shape)
            stationary_count = sum(bool(_cells(source, c)) for c in roles["stationary"])
            puzzle_stationary.add(stationary_count)
            stationary_counts.add(stationary_count)
            if source.shape == (16, 16):
                dimension_classes.add("source_size")
            elif max(source.shape) < 19:
                dimension_classes.add("small")
            elif max(source.shape) < 24:
                dimension_classes.add("medium")
            else:
                dimension_classes.add("large")
            dense_pairs += np.count_nonzero(source != background) / source.size >= 0.09
            sizes = tuple(sorted(len(component) for color in set(np.unique(source).tolist()) - {background}
                                 for component in _components(_cells(source, color))))
            signatures.add((source.shape, sizes))
            color_components = [
                (color, component)
                for color in set(np.unique(source).tolist()) - {background}
                for component in _components(_cells(source, color))
            ]
            adjacent_color_pairs += any(
                first_color != second_color and _near(first, second)
                for index, (first_color, first) in enumerate(color_components)
                for second_color, second in color_components[index + 1:]
            )
            clipped_roles = set()
            action_components = {"contract": 0, "expand": 0}
            for role in ("contract", "expand"):
                for component in _components(_cells(source, roles[role])):
                    action_components[role] += 1
                    descriptors = _descriptors(component, *source.shape)
                    assert len(descriptors) == 1
                    for row, col, radius in descriptors:
                        flags = _clip_sides(row, col, radius, *source.shape)
                        clipped_sides.update(i for i, flag in enumerate(flags) if flag)
                        if any(flags):
                            clipped_roles.add(role)
                        if split == "train" and role == "contract" and radius == 6:
                            large_contract_train += 1
                        if split == "test" and role == "contract" and radius == 1:
                            small_contract_tests += 1
                        if split == "test" and role == "expand" and radius == 0:
                            singleton_expand_tests += 1
            if split == "test":
                interior_tests += not clipped_roles
                clipped_contract_tests += "contract" in clipped_roles
                clipped_expand_tests += "expand" in clipped_roles
                multi_action_tests += max(action_components.values()) >= 2
        varied_shapes += len(puzzle_shapes) >= 2
        varied_stationary += len(puzzle_stationary) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(conventions) >= 30 and len(shapes) >= 20 and len(signatures) >= 80
    assert stationary_counts == {0, 1, 2} and clipped_sides == {0, 1, 2, 3}
    assert dimension_classes == {"source_size", "small", "medium", "large"}
    assert min(interior_tests, clipped_contract_tests, clipped_expand_tests) >= 4
    assert min(small_contract_tests, singleton_expand_tests, multi_action_tests) >= 4
    assert large_contract_train >= 3 and adjacent_color_pairs >= 10 and dense_pairs >= 10
    assert varied_shapes == 42 and varied_stationary >= 35
