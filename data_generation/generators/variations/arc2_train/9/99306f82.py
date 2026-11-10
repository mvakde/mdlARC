from __future__ import annotations

import numpy as np


# (height, width, background, frame, ordered key, frame top, frame left,
#  frame height, frame width). These describe the four original input grids
#  exactly; no whole geometry or palette is excluded merely for resembling one.
_SOURCE_SPECS = (
    (15, 15, 0, 1, (2, 6, 4), 3, 3, 9, 9),
    (10, 10, 0, 1, (3, 2), 2, 2, 6, 6),
    (15, 15, 0, 1, (8, 6, 4, 2), 4, 4, 10, 10),
    (20, 20, 0, 1, (2, 3, 9, 8, 7), 5, 5, 13, 14),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _valid_int(value: object, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _valid_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = None if num_train is None else _valid_int(num_train, "num_train", 2, 10)
    test = None if num_test is None else _valid_int(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 3 + (seed // 4) % 8)
    elif train is None:
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _construct(
    height: int, width: int, background: int, frame: int,
    key: tuple[int, ...], top: int, left: int, frame_height: int,
    frame_width: int,
) -> dict:
    grid = np.full((height, width), background, dtype=np.int8)
    for index, color in enumerate(key):
        grid[top - len(key) + index, left - len(key) + index] = color
    bottom = top + frame_height - 1
    right = left + frame_width - 1
    grid[top, left:right + 1] = frame
    grid[bottom, left:right + 1] = frame
    grid[top:bottom + 1, left] = frame
    grid[top:bottom + 1, right] = frame
    answer = grid.copy()
    interior_height = frame_height - 2
    interior_width = frame_width - 2
    # The last key entry saturates the whole residual rectangle.
    for row in range(interior_height):
        for col in range(interior_width):
            depth = min(row, col, interior_height - row - 1, interior_width - col - 1)
            answer[top + 1 + row, left + 1 + col] = key[min(depth, len(key) - 1)]
    return {"input": grid, "output": answer}


def _signature(grid: np.ndarray) -> tuple:
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes(order="C")


def _source_signatures() -> set[tuple]:
    return {
        (_signature(pair["input"]), _signature(pair["output"]))
        for pair in (_construct(*spec) for spec in _SOURCE_SPECS)
    }


def _roles(rng: np.random.Generator, previous: tuple[int, int] | None = None) -> tuple[int, int]:
    background = 0 if rng.random() < 0.23 else int(rng.integers(1, 10))
    frame = int(rng.choice([color for color in range(10) if color != background]))
    if previous is not None and (background, frame) == previous:
        frame = (frame + 1) % 10
        if frame == background:
            frame = (frame + 1) % 10
    return background, frame


def _palette(rng: np.random.Generator, roles: tuple[int, int], length: int) -> tuple[int, ...]:
    colors = [color for color in range(10) if color not in roles]
    return tuple(int(color) for color in rng.choice(colors, size=length, replace=False))


def _geometry(
    rng: np.random.Generator, key_length: int, *, square: bool | None = None,
    translated: bool | None = None, strip: bool = False, broad: bool = False,
    elongated: bool = False,
) -> tuple[int, int, int, int, int, int]:
    # For a k-color key, frame dimension = 2*k + terminal-core dimension.
    # Each axis has its own offset, core span, and outer margin.
    max_origin = min(1 if elongated else 4, max(0, (27 - 3 * key_length) // 2))
    if translated is True:
        origin_row = int(rng.integers(1, max_origin + 1))
        origin_col = int(rng.integers(1, max_origin + 1))
    elif translated is False:
        origin_row = origin_col = 0
    else:
        origin_row = int(rng.integers(0, max_origin + 1))
        origin_col = int(rng.integers(0, max_origin + 1))

    margin_cap = 2 if elongated else 6
    margin_row = int(rng.integers(1, min(margin_cap, 28 - 3 * key_length - origin_row) + 1))
    margin_col = int(rng.integers(1, min(margin_cap, 28 - 3 * key_length - origin_col) + 1))
    max_row_core = min(18, 30 - 3 * key_length - origin_row - margin_row)
    max_col_core = min(18, 30 - 3 * key_length - origin_col - margin_col)

    def draw(limit: int) -> int:
        if broad and limit >= 5:
            return int(rng.integers(max(4, limit // 2), limit + 1))
        return int(rng.integers(1, limit + 1))

    if elongated:
        if rng.random() < 0.5:
            core_row, core_col = 1, int(rng.integers(max(10, max_col_core - 3), max_col_core + 1))
        else:
            core_row, core_col = int(rng.integers(max(10, max_row_core - 3), max_row_core + 1)), 1
    elif strip:
        if rng.random() < 0.5:
            core_row, core_col = 1, draw(max_col_core)
            core_col = max(2, core_col)
        else:
            core_row, core_col = draw(max_row_core), 1
            core_row = max(2, core_row)
    elif square is True:
        side = draw(min(max_row_core, max_col_core))
        core_row = core_col = side
    else:
        core_row, core_col = draw(max_row_core), draw(max_col_core)
        if square is False and core_row == core_col:
            if core_col < max_col_core:
                core_col += 1
            elif core_row > 1:
                core_row -= 1
            else:
                core_row += 1

    frame_height = 2 * key_length + core_row
    frame_width = 2 * key_length + core_col
    top = origin_row + key_length
    left = origin_col + key_length
    height = top + frame_height + margin_row
    width = left + frame_width + margin_col
    return height, width, top, left, frame_height, frame_width


def _pair(
    rng: np.random.Generator, roles: tuple[int, int], key_length: int,
    geometry: tuple[int, int, int, int, int, int],
) -> dict:
    return _construct(*geometry[:2], *roles, _palette(rng, roles, key_length), *geometry[2:])


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    seed = _valid_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    fixed_roles = bool(rng.integers(0, 2))
    base_roles = _roles(rng)
    source = _source_signatures()
    puzzle: dict[str, list[dict]] = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    seen_structures: set[tuple] = set()

    # The first example is a square four-color anchor with a several-cell
    # terminal core. Its frame can be reused for a key-only test shift.
    anchor_side = int(rng.integers(11, 17))
    anchor_origin = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
    anchor_margins = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
    anchor_top, anchor_left = 4 + anchor_origin[0], 4 + anchor_origin[1]
    anchor_geometry = (
        anchor_top + anchor_side + anchor_margins[0],
        anchor_left + anchor_side + anchor_margins[1],
        anchor_top, anchor_left, anchor_side, anchor_side,
    )
    key_only_test = any((seed + index) % 4 == 1 for index in range(test_count))
    second_length = (2, 3)[seed % 2] if key_only_test else (2, 3, 5)[seed % 3]
    strip_puzzle = seed % 5 == 0
    second_geometry = _geometry(
        rng, second_length, square=False, translated=True,
        strip=strip_puzzle, broad=seed % 3 == 1,
    )
    train_specs = [(4, anchor_geometry), (second_length, second_geometry)]
    for _ in range(2, train_count):
        key_length = int(rng.integers(2, 5 if key_only_test else 6))
        geometry = _geometry(
            rng, key_length, square=bool(rng.integers(0, 2)),
            translated=None, strip=bool(rng.random() < 0.16),
            broad=bool(rng.random() < 0.38),
        )
        train_specs.append((key_length, geometry))
    rng.shuffle(train_specs)

    def add(split: str, key_length: int, geometry: tuple[int, ...], index: int) -> None:
        for attempt in range(24):
            roles = base_roles if fixed_roles else _roles(
                rng, base_roles if split == "train" and index == 1 else None,
            )
            pair = _pair(rng, roles, key_length, geometry)
            inp = _signature(pair["input"])
            full = (inp, _signature(pair["output"]))
            structural = (key_length, *geometry)
            if inp not in seen_inputs and full not in source and structural not in seen_structures:
                puzzle[split].append(pair)
                seen_inputs.add(inp)
                seen_structures.add(structural)
                return
            if attempt >= 1:
                geometry = _geometry(rng, key_length)
        raise RuntimeError("could not create a distinct pair")

    for index, (key_length, geometry) in enumerate(train_specs):
        add("train", key_length, geometry, index)

    familiar_lengths = [item[0] for item in train_specs]
    for index in range(test_count):
        mode = (seed + index) % 4
        if mode == 0:  # source-like combined length, scale, and shape shift
            key_length = 5 + (seed // 4 + index) % 3
            geometry = _geometry(
                rng, key_length, square=False, translated=None,
                strip=strip_puzzle and index % 2 == 1, broad=True,
            )
        elif mode == 1:  # new key length, familiar frame shape and dimensions
            key_length = 5
            top = anchor_origin[0] + key_length
            left = anchor_origin[1] + key_length
            geometry = (
                top + anchor_side + anchor_margins[0],
                left + anchor_side + anchor_margins[1],
                top, left, anchor_side, anchor_side,
            )
        elif mode == 2:  # familiar key length, new frame geometry
            key_length = 4
            elongated = (seed // 4 + index) % 2 == 0
            geometry = _geometry(
                rng, key_length, square=False, translated=None,
                strip=strip_puzzle and not elongated, broad=True,
                elongated=elongated,
            )
        else:  # familiar key length, novel placement and margins
            key_length = familiar_lengths[int(rng.integers(0, len(familiar_lengths)))]
            geometry = _geometry(
                rng, key_length, square=bool(rng.integers(0, 2)),
                translated=True, broad=bool(rng.integers(0, 2)),
            )
        add("test", key_length, geometry, index)

    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid: object, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{label} must have ARC dimensions")
    _require(grid.dtype.kind in ("i", "u"), f"{label} must have integer dtype")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
             f"{label} must have ARC colors")


def _analyze_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    background = int(inp[-1, -1])
    counts = np.bincount(inp.ravel(), minlength=10)
    counts[background] = 0
    frame = int(np.argmax(counts))
    _require(counts[frame] >= 16, "missing one-cell frame")
    rows, cols = np.where(inp == frame)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    frame_height, frame_width = bottom - top + 1, right - left + 1
    _require(bottom < height - 1 and right < width - 1,
             "frame needs bottom and right exterior margins")
    key_reverse = []
    row, col = top - 1, left - 1
    while row >= 0 and col >= 0 and int(inp[row, col]) != background:
        key_reverse.append(int(inp[row, col]))
        row -= 1
        col -= 1
    key = tuple(reversed(key_reverse))
    k = len(key)
    _require(2 <= k <= 7 and len(set(key)) == k, "invalid ordered key")
    _require(frame not in key and background not in key, "overlapping color roles")
    core_height, core_width = frame_height - 2 * k, frame_width - 2 * k
    _require(core_height >= 1 and core_width >= 1, "missing final-color core")
    expected = _construct(height, width, background, frame, key,
                          top, left, frame_height, frame_width)
    _require(np.array_equal(inp, expected["input"]), "input has stray cells or an invalid key/frame")
    _require(np.array_equal(out, expected["output"]), "output violates boundary-depth coloring")
    return {
        "background": background, "frame": frame, "key": key, "k": k,
        "height": height, "width": width, "top": top, "left": left,
        "origin_row": top - k, "origin_col": left - k,
        "frame_height": frame_height, "frame_width": frame_width,
        "core_height": core_height, "core_width": core_width,
        "bottom_margin": height - bottom - 1, "right_margin": width - right - 1,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
                 "puzzle must contain exactly train and test")
        _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
                 "splits must be lists")
        train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
                 "illegal pair counts")
        source = _source_signatures()
        inputs: set[tuple] = set()
        full_pairs: set[tuple] = set()
        structures: set[tuple] = set()
        metadata: dict[str, list[dict]] = {"train": [], "test": []}
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                label = f"{split}[{index}]"
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                         f"{label} must contain exactly input and output")
                inp, out = pair["input"], pair["output"]
                _check_grid(inp, label + ".input")
                _check_grid(out, label + ".output")
                _require(inp.shape == out.shape and inp.dtype == out.dtype,
                         f"{label} must preserve shape and dtype")
                input_signature = _signature(inp)
                full = (input_signature, _signature(out))
                _require(input_signature not in inputs and full not in full_pairs,
                         "duplicate pair or input")
                _require(full not in source, "exact copy of an original source pair")
                inputs.add(input_signature)
                full_pairs.add(full)
                info = _analyze_pair(pair)
                _require(not np.array_equal(inp, out), "unchanged pair")
                structure = (info["k"], info["height"], info["width"],
                             info["top"], info["left"], info["frame_height"],
                             info["frame_width"])
                _require(structure not in structures, "repeated structural layout")
                structures.add(structure)
                metadata[split].append(info)

        train = metadata["train"]
        _require(len({item["k"] for item in train}) >= 2,
                 "training needs varied key lengths")
        _require(any(item["k"] == 4 and min(item["core_height"], item["core_width"]) >= 2
                     for item in train), "training needs a long-key multi-cell core")
        _require(any(item["frame_height"] == item["frame_width"] for item in train),
                 "training needs a square frame")
        _require(any(item["frame_height"] != item["frame_width"] for item in train),
                 "training needs a rectangular frame")
        _require(any(item["origin_row"] > 0 and item["origin_col"] > 0 for item in train),
                 "training needs a translated key and frame")
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed puzzle: {exc}") from exc


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_signature(pair["input"]), _signature(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _predict_from_examples(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This check receives only visible examples and a test input. It does not
    # inspect the test answer, generation seed, or sampled geometry.
    for pair in train:
        info = _analyze_pair(pair)
        for index, color in enumerate(info["key"]):
            row = info["top"] + 1 + index
            col = info["left"] + 1 + index
            if row < info["top"] + info["frame_height"] - 1 and col < info["left"] + info["frame_width"] - 1:
                _require(int(pair["output"][row, col]) == color,
                         "demonstration does not reveal outer-to-inner order")
    # Read only the test input. The frame is its repeated non-background color.
    background = int(test_input[-1, -1])
    counts = np.bincount(test_input.ravel(), minlength=10)
    counts[background] = 0
    frame = int(np.argmax(counts))
    rows, cols = np.where(test_input == frame)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    reversed_key = []
    row, col = top - 1, left - 1
    while row >= 0 and col >= 0 and int(test_input[row, col]) != background:
        reversed_key.append(int(test_input[row, col]))
        row -= 1
        col -= 1
    key = tuple(reversed(reversed_key))
    answer = test_input.copy()
    for row in range(top + 1, bottom):
        for col in range(left + 1, right):
            depth = min(row - top - 1, bottom - row - 1,
                        col - left - 1, right - col - 1)
            answer[row, col] = key[min(depth, len(key) - 1)]
    return answer


def self_test() -> None:
    explicit = generate_puzzle(12345, 7, 3)
    _require(_puzzle_signature(explicit) == _puzzle_signature(generate_puzzle(12345, 7, 3)),
             "explicit-count generation is not deterministic")
    _require(_puzzle_signature(generate_puzzle(19)) == _puzzle_signature(generate_puzzle(19)),
             "automatic generation is not deterministic")
    _require(_puzzle_signature(generate_puzzle(888, 6, 2)) !=
             _puzzle_signature(generate_puzzle(889, 6, 2)), "seeds did not vary")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts were not honored")
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train, "partial train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "partial test count ignored")
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 2.0, "num_test": 1},
                      {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count was accepted: {arguments}")

    train_counts, test_counts, signatures = set(), set(), set()
    key_lengths, canvas_shapes, frame_shapes, origins = set(), set(), set(), set()
    core_shapes, margins, role_pairs = set(), set(), set()
    zero_backgrounds = variable_role_puzzles = strip_pairs = 0
    familiar_test_keys = longer_test_keys = rectangular_tests = square_tests = 0
    three_train_puzzles = four_even = four_odd = broad_cores = 0
    key_only_matches = familiar_key_new_shapes = translated_both = 0
    elongated_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _puzzle_signature(puzzle)
        _require(signature not in signatures, "duplicate whole puzzle across seeds")
        signatures.add(signature)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        three_train_puzzles += len(puzzle["train"]) == 3
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
        train_keys = {item["k"] for item in train_info}
        train_shapes = {(item["frame_height"], item["frame_width"])
                        for item in train_info}
        four_even += sum(item["k"] == 4 and item["frame_height"] % 2 == 0
                         for item in train_info)
        four_odd += sum(item["k"] == 4 and item["frame_height"] % 2 == 1
                        for item in train_info)
        translated_both += (
            any(item["origin_row"] > 0 and item["origin_col"] > 0
                for item in train_info)
            and any(item["origin_row"] > 0 and item["origin_col"] > 0
                    for item in test_info)
        )
        key_only_matches += any(
            item["k"] not in train_keys
            and (item["frame_height"], item["frame_width"]) in train_shapes
            for item in test_info
        )
        familiar_key_new_shapes += any(
            item["k"] in train_keys
            and (item["frame_height"], item["frame_width"]) not in train_shapes
            for item in test_info
        )
        roles_here = {(item["background"], item["frame"])
                      for item in train_info + test_info}
        variable_role_puzzles += len(roles_here) > 1
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], train_info if split == "train" else test_info):
                inp, out = pair["input"], pair["output"]
                _require(np.count_nonzero(inp != out) ==
                         (info["frame_height"] - 2) * (info["frame_width"] - 2),
                         "changed cells extend outside the interior")
                key_lengths.add(info["k"])
                canvas_shapes.add((info["height"], info["width"]))
                frame_shapes.add((info["frame_height"], info["frame_width"]))
                origins.add((info["origin_row"], info["origin_col"]))
                core_shapes.add((info["core_height"], info["core_width"]))
                margins.add((info["bottom_margin"], info["right_margin"]))
                role_pairs.add((info["background"], info["frame"]))
                zero_backgrounds += info["background"] == 0
                strip_pairs += min(info["core_height"], info["core_width"]) == 1
                broad_cores += info["core_height"] * info["core_width"] >= 30
                if split == "test":
                    familiar_test_keys += info["k"] in train_keys
                    longer_test_keys += info["k"] > max(train_keys)
                    rectangular_tests += info["frame_height"] != info["frame_width"]
                    square_tests += info["frame_height"] == info["frame_width"]
                    elongated_tests += (max(info["frame_height"], info["frame_width"])
                                        >= 2 * min(info["frame_height"], info["frame_width"]))
                    prediction = _predict_from_examples(
                        [{"input": example["input"].copy(),
                          "output": example["output"].copy()}
                         for example in puzzle["train"]], inp.copy())
                    _require(np.array_equal(prediction, out),
                             "visible training evidence did not predict test output")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(canvas_shapes) >= 35 and len(frame_shapes) >= 30,
             "canvas or frame shape diversity collapsed")
    _require(len(core_shapes) >= 25 and len(origins) >= 8 and len(margins) >= 10,
             "core or placement diversity collapsed")
    _require(len(role_pairs) >= 20 and zero_backgrounds > 0 and variable_role_puzzles >= 8,
             "color-role variation collapsed")
    _require({2, 3, 4, 5, 6, 7}.issubset(key_lengths), "key-length range collapsed")
    _require(strip_pairs >= 8 and familiar_test_keys >= 10 and longer_test_keys >= 8,
             "core strips or test key shifts are missing")
    _require(rectangular_tests >= 10 and square_tests >= 10,
             "test shape range collapsed")
    _require(three_train_puzzles >= 2 and four_even >= 10 and four_odd >= 10,
             "concise puzzles or even/odd four-color centers are missing")
    _require(broad_cores >= 30 and elongated_tests >= 8,
             "broad cores or elongated frames are missing")
    _require(key_only_matches >= 8 and familiar_key_new_shapes >= 10,
             "key-only or frame-only test shifts are missing")
    _require(translated_both >= 15,
             "translation is not demonstrated before testing")
