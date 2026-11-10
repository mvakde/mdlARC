import numpy as np


_SOURCE_PAIR_ROWS = frozenset({
    (
        ("910014", "111111", "018810", "018010", "111111", "210013"),
        ("94", "20"),
    ),
    (
        (
            "21000013",
            "11111111",
            "01080010",
            "01880810",
            "01008010",
            "01808810",
            "11111111",
            "41000016",
        ),
        ("0200", "2203", "0060", "4066"),
    ),
    (
        (
            "61000012",
            "11111111",
            "01080810",
            "01888010",
            "01808810",
            "01888010",
            "11111111",
            "71000014",
        ),
        ("0602", "6620", "7044", "7740"),
    ),
    (
        (
            "3100000014",
            "1111111111",
            "0108800010",
            "0188808010",
            "0100808010",
            "0108088010",
            "0188080810",
            "0108008010",
            "1111111111",
            "7100000015",
        ),
        ("033000", "333040", "003040", "070550", "770505", "070050"),
    ),
})

def _quadrants(
    height: int,
    width: int,
) -> tuple[tuple[slice, slice], ...]:
    half_h = height // 2
    half_w = width // 2
    return (
        (slice(0, half_h), slice(0, half_w)),
        (slice(0, half_h), slice(half_w, width)),
        (slice(half_h, height), slice(0, half_w)),
        (slice(half_h, height), slice(half_w, width)),
    )


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _puzzle_signature(puzzle: dict) -> tuple:
    return (
        tuple(_pair_signature(pair) for pair in puzzle["train"]),
        tuple(_pair_signature(pair) for pair in puzzle["test"]),
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _rows_key(input_grid),
        _rows_key(output_grid),
    ) in _SOURCE_PAIR_ROWS


def _validate_count(
    value: int,
    name: str,
    low: int,
    high: int,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = None
    test_count = None

    if num_train is not None:
        train_count = _validate_count(
            num_train, "num_train", 2, 10
        )
    if num_test is not None:
        test_count = _validate_count(
            num_test, "num_test", 1, 4
        )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = test_count + 1
        train_count = (
            minimum_train
            + (seed // 4) % (11 - minimum_train)
        )
    elif train_count is None:
        minimum_train = test_count + 1
        train_count = (
            minimum_train
            + seed % (11 - minimum_train)
        )
    else:
        largest_test = min(4, train_count - 1)
        test_count = 1 + seed % largest_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError(
            "the requested counts cannot form a legal puzzle"
        )

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    roles = rng.choice(
        np.arange(10, dtype=np.int8),
        size=3,
        replace=False,
    )
    return {
        "background": int(roles[0]),
        "separator": int(roles[1]),
        "active": int(roles[2]),
        "vary_roles": bool(rng.random() < 0.46),
    }


def _sample_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    anchor: bool = False,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if anchor:
        for rs, cs in _quadrants(height, width):
            block = mask[rs, cs]
            chosen = rng.choice(block.size, size=int(rng.integers(1, block.size)), replace=False)
            block.flat[chosen] = True
        return mask

    density = float(rng.uniform(0.13, 0.87))
    topology = int(rng.choice(4, p=(0.40, 0.27, 0.23, 0.10)))
    noise = rng.random((height, width))
    if topology == 1:  # Local clusters.
        scores = 0.35 * noise
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            scores += 0.13 * np.roll(noise, (dr, dc), axis=(0, 1))
    elif topology == 2:  # Horizontal and vertical runs.
        scores = 0.27 * noise + 0.40 * rng.random((height, 1))
        scores += 0.33 * rng.random((1, width))
    elif topology == 3:  # Alternating isolated candidates.
        yy, xx = np.indices((height, width))
        scores = 0.55 * noise + 0.45 * ((xx + yy + int(rng.integers(2))) % 2)
    else:
        scores = noise

    for rs, cs in _quadrants(height, width):
        block = mask[rs, cs]
        local_density = float(np.clip(density + rng.normal(0, 0.23), 0.03, 0.97))
        count = int(rng.binomial(block.size, local_density))
        if count:
            block.flat[np.argsort(scores[rs, cs].ravel())[-count:]] = True

    special = int(rng.choice(3, p=(0.72, 0.14, 0.14)))
    if special:
        rs, cs = _quadrants(height, width)[int(rng.integers(4))]
        mask[rs, cs] = special == 2
    elif height >= 4 and width >= 4 and rng.random() < 0.53:
        # Control adjacency across either midpoint, independently.
        crossing = int(rng.integers(4))
        half_h, half_w = height // 2, width // 2
        if crossing & 1:
            col = int(rng.integers(width))
            mask[half_h - 1, col] = mask[half_h, col] = True
        else:
            mask[half_h - 1, :] = False
        if crossing & 2:
            row = int(rng.integers(height))
            mask[row, half_w - 1] = mask[row, half_w] = True
        else:
            mask[:, half_w - 1] = False

    if not np.any(mask):
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    if np.all(mask):
        mask[int(rng.integers(height)), int(rng.integers(width))] = False

    return mask


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    central_shape: tuple[int, int],
    forced_keys: tuple[int, ...] | None = None,
    forced_mask: np.ndarray | None = None,
    anchor: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")

    height, width = central_shape
    background = puzzle_parameters["background"]
    separator = puzzle_parameters["separator"]
    active = puzzle_parameters["active"]

    unavailable = {background, separator, active}
    key_palette = np.array(
        [
            color
            for color in range(10)
            if color not in unavailable
        ],
        dtype=np.int8,
    )
    if forced_keys is not None:
        assert not set(forced_keys).intersection(unavailable)
        keys = tuple(forced_keys)
    elif anchor or rng.random() < 0.75:
        keys = tuple(int(value) for value in rng.choice(key_palette, size=4, replace=False))
    else:
        distinct = int(rng.choice((2, 3)))
        colors = [int(value) for value in rng.choice(key_palette, size=distinct, replace=False)]
        repeated = colors + [colors[int(rng.integers(distinct))] for _ in range(4 - distinct)]
        rng.shuffle(repeated)
        keys = tuple(repeated)

    mask = forced_mask.copy() if forced_mask is not None else _sample_mask(rng, height, width, anchor)

    input_grid = np.full(
        (height + 4, width + 4),
        background,
        dtype=np.int8,
    )

    input_grid[0, 0] = keys[0]
    input_grid[0, -1] = keys[1]
    input_grid[-1, 0] = keys[2]
    input_grid[-1, -1] = keys[3]

    input_grid[1, :] = separator
    input_grid[-2, :] = separator
    input_grid[:, 1] = separator
    input_grid[:, -2] = separator

    center = input_grid[2:-2, 2:-2]
    center[mask] = active

    output_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    for key, (row_slice, col_slice) in zip(
        keys,
        _quadrants(height, width),
    ):
        output_block = output_grid[
            row_slice,
            col_slice,
        ]
        output_block[
            mask[row_slice, col_slice]
        ] = key

    return input_grid, output_grid, keys


def _inspect_pair(pair: dict) -> dict:
    assert isinstance(pair, dict)
    assert set(pair.keys()) == {"input", "output"}

    for name in ("input", "output"):
        grid = pair[name]
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30
        assert 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))

    input_grid = pair["input"]
    output_grid = pair["output"]
    out_h, out_w = output_grid.shape

    assert out_h % 2 == 0
    assert out_w % 2 == 0
    assert input_grid.shape == (out_h + 4, out_w + 4)

    background = int(input_grid[0, 2])
    separator = int(input_grid[1, 0])
    keys = (
        int(input_grid[0, 0]),
        int(input_grid[0, -1]),
        int(input_grid[-1, 0]),
        int(input_grid[-1, -1]),
    )

    # Validate the outer key area and background channels.
    assert np.all(
        input_grid[0, 2:-2] == background
    )
    assert np.all(
        input_grid[-1, 2:-2] == background
    )
    assert np.all(
        input_grid[2:-2, 0] == background
    )
    assert np.all(
        input_grid[2:-2, -1] == background
    )

    # Validate the complete one-cell-inward separator.
    assert np.all(input_grid[1, :] == separator)
    assert np.all(input_grid[-2, :] == separator)
    assert np.all(input_grid[:, 1] == separator)
    assert np.all(input_grid[:, -2] == separator)

    center = input_grid[2:-2, 2:-2]
    center_values = {
        int(value)
        for value in np.unique(center)
    }

    assert background in center_values
    assert len(center_values) == 2

    active = next(
        value
        for value in center_values
        if value != background
    )

    assert len({background, separator, active}) == 3
    assert not set(keys).intersection({
        background,
        separator,
        active,
    })

    # Independently reconstruct the required output.
    mask = center == active
    expected = np.full(
        output_grid.shape,
        background,
        dtype=np.int8,
    )

    for key, (row_slice, col_slice) in zip(
        keys,
        _quadrants(out_h, out_w),
    ):
        expected_block = expected[
            row_slice,
            col_slice,
        ]
        expected_block[
            mask[row_slice, col_slice]
        ] = key

    assert np.array_equal(output_grid, expected)

    return {
        "background": background,
        "separator": separator,
        "active": active,
        "keys": keys,
        "mask": mask,
        "central_shape": output_grid.shape,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    seen_inputs = set()
    seen_pairs = set()
    train_metadata = []
    test_metadata = []

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            metadata = _inspect_pair(pair)
            input_signature = _grid_signature(pair["input"])
            pair_signature = _pair_signature(pair)

            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            assert not _is_source_pair(
                pair["input"],
                pair["output"],
            )

            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            if split_name == "train":
                train_metadata.append(metadata)
            else:
                test_metadata.append(metadata)

    # One four-key demonstration reveals every positional association.
    anchor = train_metadata[0]
    assert anchor["central_shape"] == (4, 4)
    assert len(set(anchor["keys"])) == 4
    assert all(np.any(anchor["mask"][rs, cs]) for rs, cs in _quadrants(4, 4))
    assert len({
        metadata["keys"]
        for metadata in train_metadata
    }) >= 2
    train_roles = {(m["background"], m["separator"], m["active"]) for m in train_metadata}
    all_roles = {(m["background"], m["separator"], m["active"])
                 for m in train_metadata + test_metadata}
    assert len(all_roles) == 1 or len(train_roles) >= 2


def _draw_roles(rng: np.random.Generator, forbidden=()) -> dict:
    palette = [color for color in range(10) if color not in forbidden]
    colors = rng.choice(palette, size=3, replace=False)
    return {"background": int(colors[0]), "separator": int(colors[1]),
            "active": int(colors[2])}


def _shape(rng: np.random.Generator, regime: str, train_shapes=()) -> tuple[int, int]:
    if regime == "source_train":
        band = int(rng.choice(3, p=(0.25, 0.58, 0.17)))
        options = ((2, 4, 6), (4, 6, 8), (8, 10, 12, 14))[band]
    else:
        band = int(rng.choice(4, p=(0.19, 0.20, 0.48, 0.13)))
        if band == 1:
            return tuple(train_shapes[int(rng.integers(len(train_shapes)))])
        options = ((2, 4, 6), (), (6, 8, 10, 12, 14),
                   (16, 18, 20, 22, 24, 26))[band]
    return int(rng.choice(options)), int(rng.choice(options))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError(
            "seed must be a non-negative integer"
        )

    seed = int(seed)
    if seed < 0:
        raise ValueError(
            "seed must be a non-negative integer"
        )

    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        seed,
        num_train,
        num_test,
    )

    puzzle_parameters = sample_puzzle_parameters(rng)
    vary_roles = puzzle_parameters.pop("vary_roles")
    train_shapes = [(4, 4)]  # A complete four-key positional demonstration.
    source_progression = bool(rng.random() < 0.22)
    train_shapes.append((2, 2) if source_progression or rng.random() < 0.42
                        else _shape(rng, "source_train"))
    while len(train_shapes) < train_count:
        train_shapes.append(_shape(rng, "source_train"))
    comparison = int(rng.choice(3, p=(0.50, 0.25, 0.25))) if train_count >= 3 else 0
    if comparison:
        train_shapes[2] = (4, 4)
    test_shapes = [_shape(rng, "target_test", train_shapes) for _ in range(test_count)]
    if source_progression:
        test_shapes[0] = (6, 6)

    puzzle = {
        "train": [],
        "test": [],
    }

    seen_inputs = set()
    seen_pairs = set()
    anchor_mask = None
    anchor_keys = None

    for split_name, regime, shapes in (
        ("train", "source_train", train_shapes),
        ("test", "target_test", test_shapes),
    ):
        for index, central_shape in enumerate(shapes):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            pair_rng = np.random.default_rng(pair_seed)
            for _ in range(64):
                roles = puzzle_parameters
                if vary_roles and not (split_name == "train" and index == 0):
                    roles = _draw_roles(pair_rng)
                    if split_name == "train" and index == 1 and roles == puzzle_parameters:
                        roles = _draw_roles(pair_rng, puzzle_parameters.values())
                forced_keys = None
                forced_mask = None
                if split_name == "train" and index == 2 and comparison:
                    if comparison == 1:
                        forced_mask = anchor_mask
                        forced_keys = tuple(anchor_keys[int(i)] for i in pair_rng.permutation(4))
                        if forced_keys == anchor_keys:
                            forced_keys = anchor_keys[1:] + anchor_keys[:1]
                    else:
                        forced_keys = anchor_keys
                    if set(forced_keys).intersection(roles.values()):
                        roles = _draw_roles(pair_rng, forced_keys)
                elif split_name == "train" and index >= 3 and pair_rng.random() < 0.18:
                    if not set(anchor_keys).intersection(roles.values()):
                        forced_keys = anchor_keys
                input_grid, output_grid, keys = _generate_pair(
                    pair_rng,
                    regime,
                    roles,
                    central_shape,
                    forced_keys=forced_keys,
                    forced_mask=forced_mask,
                    anchor=(split_name == "train" and index == 0),
                )
                if split_name == "train" and index == 1 and keys == anchor_keys:
                    continue
                if split_name == "train" and index == 2 and comparison == 2:
                    if np.array_equal(input_grid[2:-2, 2:-2] != roles["background"], anchor_mask):
                        continue
                input_signature = _grid_signature(
                    input_grid
                )
                pair_signature = (
                    input_signature,
                    _grid_signature(output_grid),
                )

                if input_signature in seen_inputs:
                    continue
                if pair_signature in seen_pairs:
                    continue
                if _is_source_pair(
                    input_grid,
                    output_grid,
                ):
                    continue
                break
            else:
                raise RuntimeError(
                    "could not generate a unique valid pair"
                )

            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            puzzle[split_name].append({
                "input": input_grid,
                "output": output_grid,
            })
            if split_name == "train" and index == 0:
                anchor_keys = keys
                anchor_mask = input_grid[2:-2, 2:-2] == roles["active"]

    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    automatic_puzzles = []

    # Forty-two fixed seeds exercise automatic count selection.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        automatic_puzzles.append(puzzle)

    train_counts = {
        len(puzzle["train"])
        for puzzle in automatic_puzzles
    }
    test_counts = {
        len(puzzle["test"])
        for puzzle in automatic_puzzles
    }

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    # Whole-puzzle determinism with explicit and automatic counts.
    deterministic_a = generate_puzzle(12345, 7, 3)
    deterministic_b = generate_puzzle(12345, 7, 3)
    assert (
        _puzzle_signature(deterministic_a)
        == _puzzle_signature(deterministic_b)
    )

    automatic_a = generate_puzzle(31415)
    automatic_b = generate_puzzle(31415)
    assert (
        _puzzle_signature(automatic_a)
        == _puzzle_signature(automatic_b)
    )

    # Different seeds should overwhelmingly produce different puzzles.
    automatic_signatures = {
        _puzzle_signature(puzzle)
        for puzzle in automatic_puzzles
    }
    assert len(automatic_signatures) >= 38

    # Legal boundaries with both counts supplied.
    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )
    for index, (train_count, test_count) in enumerate(
        boundary_cases
    ):
        puzzle = generate_puzzle(
            500 + index,
            train_count,
            test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    # Partial count specifications.
    partial_cases = (
        (601, None, 1),
        (602, None, 4),
        (603, 2, None),
        (604, 10, None),
    )
    for seed, requested_train, requested_test in partial_cases:
        puzzle = generate_puzzle(
            seed,
            requested_train,
            requested_test,
        )

        if requested_train is not None:
            assert (
                len(puzzle["train"])
                == requested_train
            )
        if requested_test is not None:
            assert (
                len(puzzle["test"])
                == requested_test
            )

        validate_generated_puzzle(puzzle)

    # Illegal count combinations and values.
    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": True, "num_test": 1},
    )

    for arguments in illegal_cases:
        try:
            generate_puzzle(700, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal counts were accepted: "
                f"{arguments}"
            )

    role_assignments = set()
    input_shapes = set()
    mask_layouts = set()
    key_assignments = set()
    all_input_signatures = set()
    total_pairs = 0

    for puzzle in automatic_puzzles:
        first_metadata = _inspect_pair(
            puzzle["train"][0]
        )
        role_assignments.add((
            first_metadata["background"],
            first_metadata["separator"],
            first_metadata["active"],
        ))

        local_inputs = set()
        local_pairs = set()

        for split_name in ("train", "test"):
            for pair in puzzle[split_name]:
                metadata = _inspect_pair(pair)
                total_pairs += 1

                input_shapes.add(pair["input"].shape)
                mask_layouts.add((
                    metadata["central_shape"],
                    tuple(
                        bool(value)
                        for value
                        in metadata["mask"].ravel()
                    ),
                ))
                key_assignments.add(metadata["keys"])

                input_signature = _grid_signature(
                    pair["input"]
                )
                pair_signature = _pair_signature(pair)

                assert input_signature not in local_inputs
                assert pair_signature not in local_pairs

                local_inputs.add(input_signature)
                local_pairs.add(pair_signature)
                all_input_signatures.add(input_signature)

                assert not _is_source_pair(
                    pair["input"],
                    pair["output"],
                )

    # Meaningful diversity across the fixed-seed population.
    assert len(role_assignments) >= 20
    assert len(input_shapes) >= 8
    assert len(mask_layouts) >= 100
    assert len(key_assignments) >= 100
    assert (
        len(all_input_signatures)
        >= int(0.85 * total_pairs)
    )

    # Review-specific measurements use reconstructed masks and visible keys.
    train_rectangles = test_squares = small_tests = large_tests = 0
    train_blank = train_solid = test_blank = test_solid = 0
    repeated_corner_colors = variable_role_puzzles = fixed_role_puzzles = 0
    same_scale_tests = repeated_keys_new_mask = same_mask_new_keys = 0
    densities = {"train": set(), "test": set()}
    test_crossings = set()
    for puzzle in automatic_puzzles:
        infos = {split: [_inspect_pair(pair) for pair in puzzle[split]]
                 for split in ("train", "test")}
        all_infos = infos["train"] + infos["test"]
        triples = {(m["background"], m["separator"], m["active"]) for m in all_infos}
        variable_role_puzzles += len(triples) > 1
        fixed_role_puzzles += len(triples) == 1
        train_shapes = {m["central_shape"] for m in infos["train"]}
        same_scale_tests += any(m["central_shape"] in train_shapes for m in infos["test"])
        for a_index, a in enumerate(infos["train"]):
            for b in infos["train"][a_index + 1:]:
                if a["keys"] == b["keys"] and (a["central_shape"] != b["central_shape"]
                    or not np.array_equal(a["mask"], b["mask"])):
                    repeated_keys_new_mask += 1
                if (a["central_shape"] == b["central_shape"]
                    and np.array_equal(a["mask"], b["mask"])
                    and a["keys"] != b["keys"]
                    and set(a["keys"]) == set(b["keys"])):
                    same_mask_new_keys += 1
        for split in ("train", "test"):
            for m in infos[split]:
                h, w = m["central_shape"]
                mask = m["mask"]
                value = float(np.mean(mask))
                densities[split].add("sparse" if value < 0.4 else "dense" if value > 0.6 else "intermediate")
                repeated_corner_colors += len(set(m["keys"])) < 4
                if split == "train":
                    train_rectangles += h != w
                else:
                    test_squares += h == w
                    small_tests += max(h, w) <= 6
                    large_tests += max(h, w) >= 16
                    hh, ww = h // 2, w // 2
                    test_crossings.add((bool(np.any(mask[hh - 1] & mask[hh])),
                                        bool(np.any(mask[:, ww - 1] & mask[:, ww]))))
                if min(h, w) >= 4:
                    blocks = [mask[rs, cs] for rs, cs in _quadrants(h, w)]
                    if split == "train":
                        train_blank += any(not np.any(block) for block in blocks)
                        train_solid += any(np.all(block) for block in blocks)
                    else:
                        test_blank += any(not np.any(block) for block in blocks)
                        test_solid += any(np.all(block) for block in blocks)
    assert train_rectangles >= 10 and test_squares >= 5
    assert small_tests >= 5 and large_tests >= 2 and same_scale_tests >= 2
    assert min(train_blank, train_solid, test_blank, test_solid) >= 2
    assert repeated_corner_colors >= 10
    assert variable_role_puzzles >= 5 and fixed_role_puzzles >= 5
    assert repeated_keys_new_mask >= 1 and same_mask_new_keys >= 1
    assert densities["train"] == densities["test"] == {"sparse", "intermediate", "dense"}
    assert len(test_crossings) == 4

    # The separator endpoints and recoloring are independently checked by the validator.
    example = generate_puzzle(800, 2, 1)
    damaged = {split: [dict(pair) for pair in example[split]] for split in ("train", "test")}
    damaged["train"][0]["input"] = example["train"][0]["input"].copy()
    damaged["train"][0]["input"][0, 1] = damaged["train"][0]["input"][0, 2]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("separator endpoint corruption accepted")
