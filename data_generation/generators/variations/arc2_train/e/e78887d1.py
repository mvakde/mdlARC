import numpy as np


# Exact source pairs, used only to ensure that generated pairs are not copies.
_SOURCE_PAIR_ROWS = (
    (
        (
            "000000000000000",
            "222030001000505",
            "002030000110505",
            "002033301000505",
            "000000000000000",
        ),
        (
            "200030001010555",
            "200003301010005",
            "222030001010005",
        ),
    ),
    (
        (
            "00000000000",
            "20203000111",
            "02003330010",
            "20200030010",
            "00000000000",
            "20003330101",
            "22200300010",
            "00200300101",
            "00000000000",
        ),
        (
            "22203030100",
            "02000300111",
            "02003030001",
        ),
    ),
    (
        (
            "000000000000000",
            "200030301010050",
            "222003001110555",
            "002030300100050",
            "000000000000000",
            "202030300100500",
            "020033301110555",
            "202003000100005",
            "000000000000000",
        ),
        (
            "202003001000505",
            "222033301110050",
            "020003000010505",
        ),
    ),
    (
        (
            "00000000000",
            "20200100333",
            "20201110000",
            "22200100333",
            "00000000000",
            "02001110303",
            "22200000303",
            "02001110333",
            "00000000000",
            "22201010030",
            "00001010333",
            "22201110030",
            "00000000000",
        ),
        (
            "20200100333",
            "20201110000",
            "22200100333",
        ),
    ),
    (
        (
            "000000000000000",
            "220033001010050",
            "020000301010555",
            "222033001010050",
            "000000000000000",
            "220030300100550",
            "002030301110050",
            "220030300100555",
            "000000000000000",
        ),
        (
            "202003001100550",
            "202033300100005",
            "202003001110550",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _coerce_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _coerce_count("num_train", num_train, 2, 10)
    test_count = _coerce_count("num_test", num_test, 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This deterministic schedule guarantees broad count diversity for
        # consecutive seeds while always satisfying the legal bounds.
        seed_number = int(seed)
        test_count = 1 + seed_number % 4
        span = 10 - test_count
        train_count = test_count + 1 + (seed_number // 4) % span
        return train_count, test_count

    if train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    if maximum_test < 1:
        raise ValueError("no legal num_test exists for the requested num_train")
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


def _mask_key(mask: np.ndarray) -> int:
    key = 0
    for index, value in enumerate(mask.reshape(-1)):
        if bool(value):
            key |= 1 << index
    return key


def _mask_from_key(key: int) -> np.ndarray:
    return np.array([(key >> bit) & 1 for bit in range(9)], dtype=bool).reshape(3, 3)


def _mask_is_connected(mask: np.ndarray) -> bool:
    cells = {tuple(int(v) for v in cell) for cell in np.argwhere(mask)}
    seen: set[tuple[int, int]] = set()
    pending = [next(iter(cells))]
    while pending:
        row, col = pending.pop()
        if (row, col) in seen:
            continue
        seen.add((row, col))
        pending.extend(
            neighbor for neighbor in
            ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            if neighbor in cells and neighbor not in seen
        )
    return len(seen) == len(cells)


_MASKS_BY_SIZE = {
    size: tuple(key for key in range(1, 512) if key.bit_count() == size)
    for size in range(1, 10)
}


def _sample_mask_collection(
    rng: np.random.Generator,
    slot_count: int,
    familiar_keys: set[int],
    reuse_mode: bool,
) -> list[np.ndarray]:
    # Dense, recognizable glyphs dominate, but sparse and nearly full glyphs
    # remain possible. Reuse is across pairs only; within a pair masks differ.
    for _ in range(64):
        keys: list[int] = []
        blocked: set[int] = set()
        for slot in range(slot_count):
            use_familiar = bool(familiar_keys) and reuse_mode and (
                (slot == 0) or rng.random() < 0.25
            )
            available = tuple(familiar_keys - blocked)
            if use_familiar and available:
                key = int(rng.choice(available))
            else:
                density_group = int(rng.choice(3, p=[0.13, 0.78, 0.09]))
                sizes = ((1, 2, 3), (4, 5, 6, 7), (8, 9))[density_group]
                size = int(rng.choice(sizes))
                choices = tuple(
                    key for key in _MASKS_BY_SIZE[size]
                    if key not in blocked and (reuse_mode or key not in familiar_keys)
                )
                if not choices:
                    choices = tuple(
                        key for key in _MASKS_BY_SIZE[6]
                        if key not in blocked and (reuse_mode or key not in familiar_keys)
                    )
                key = int(rng.choice(choices))
            keys.append(key)
            blocked.add(key)
        masks = [_mask_from_key(key) for key in keys]
        union = np.logical_or.reduce(masks)
        if np.all(np.any(union, axis=0)) and np.all(np.any(union, axis=1)):
            return masks
    raise RuntimeError("could not sample a well-segmented mask collection")


def _shift_masks_left(masks: list[np.ndarray]) -> list[np.ndarray]:
    # New slot j receives the old mask from j + 1. The old leftmost mask
    # therefore wraps into the rightmost slot.
    return list(masks[1:]) + [masks[0]]


def _render_state(
    masks: list[np.ndarray],
    carriers: tuple[int, ...],
    background: int,
) -> np.ndarray:
    slot_count = len(masks)
    width = 4 * slot_count - 1
    state = np.full((3, width), background, dtype=np.int8)

    for slot, (mask, carrier) in enumerate(zip(masks, carriers)):
        start = 4 * slot
        block = state[:, start : start + 3]
        block[mask] = carrier

    return state


def _construct_pair(
    base_masks: list[np.ndarray],
    carriers: tuple[int, ...],
    history_length: int,
    background: int,
) -> dict:
    states: list[list[np.ndarray]] = [list(base_masks)]
    for _ in range(1, history_length):
        states.append(_shift_masks_left(states[-1]))

    width = 4 * len(base_masks) - 1
    input_grid = np.full(
        (4 * history_length + 1, width),
        background,
        dtype=np.int8,
    )

    for time_index, masks in enumerate(states):
        row_start = 1 + 4 * time_index
        input_grid[row_start : row_start + 3, :] = _render_state(
            masks,
            carriers,
            background,
        )

    output_masks = _shift_masks_left(states[-1])
    output_grid = _render_state(output_masks, carriers, background)

    return {
        "input": input_grid,
        "output": output_grid,
    }


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    familiar_masks: set[int],
    required_layout: tuple[int, int] | None = None,
    reuse_mode: bool = False,
) -> dict:
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown generation regime: {regime}")
    if required_layout is not None:
        slot_count, history_length = required_layout
    else:
        # Most examples use the source's 3-4 slots; longer rows are occasional
        # extrapolations of exactly the same cyclic rule.
        slot_count = int(rng.choice((3, 4, 5, 6, 7), p=(0.33, 0.37, 0.15, 0.09, 0.06)))
        if regime == "source_train":
            history_length = int(rng.choice((1, 2, 3), p=(0.28, 0.48, 0.24)))
        else:
            history_length = int(rng.choice((1, 2, 3), p=(0.38, 0.36, 0.26)))

    background = 0 if rng.random() < 0.72 else int(rng.integers(1, 10))
    palette = [color for color in range(10) if color != background]
    carriers = tuple(int(color) for color in rng.choice(palette, size=slot_count, replace=False))
    masks = _sample_mask_collection(rng, slot_count, familiar_masks, reuse_mode)

    return _construct_pair(
        masks,
        carriers,
        history_length,
        background,
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(size) for size in grid.shape),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _grid_as_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    key = (_grid_as_rows(input_grid), _grid_as_rows(output_grid))
    return key in _SOURCE_PAIR_ROWS


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        rng,
        seed,
        num_train,
        num_test,
    )
    puzzle = {
        "train": [],
        "test": [],
    }

    familiar_masks: set[int] = set()
    used_input_keys: set[tuple] = set()
    demonstration_index = int(rng.integers(0, train_count))
    # Choose which tests use familiar glyphs independently of their layout.
    test_reuse = [bool(rng.random() < 0.55) for _ in range(test_count)]

    for train_index in range(train_count):
        pair_seed = int(
            rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        pair_rng = np.random.default_rng(pair_seed)

        for _ in range(64):
            pair = _generate_pair(
                pair_rng,
                "source_train",
                familiar_masks,
                required_layout=(4, 2) if train_index == demonstration_index else None,
                reuse_mode=bool(rng.random() < 0.45),
            )
            input_key = _grid_key(pair["input"])

            if input_key in used_input_keys:
                continue
            if _is_source_pair(pair["input"], pair["output"]):
                continue

            used_input_keys.add(input_key)
            puzzle["train"].append(pair)
            analysis = _analyze_pair(pair)
            familiar_masks.update(analysis["initial_masks"])
            break
        else:
            raise RuntimeError("could not generate a unique training pair")

    for test_index in range(test_count):
        pair_seed = int(
            rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        pair_rng = np.random.default_rng(pair_seed)

        for _ in range(64):
            pair = _generate_pair(
                pair_rng,
                "target_test",
                familiar_masks,
                reuse_mode=test_reuse[test_index],
            )
            input_key = _grid_key(pair["input"])

            if input_key in used_input_keys:
                continue
            if _is_source_pair(pair["input"], pair["output"]):
                continue

            used_input_keys.add(input_key)
            puzzle["test"].append(pair)
            familiar_masks.update(_analyze_pair(pair)["initial_masks"])
            break
        else:
            raise RuntimeError(
                f"could not generate unique test pair {test_index}"
            )

    validate_generated_puzzle(puzzle)
    return puzzle


def _decode_state(
    state: np.ndarray,
    slot_count: int,
    background: int,
) -> tuple[list[np.ndarray], tuple[int, ...]]:
    masks: list[np.ndarray] = []
    colors: list[int] = []

    for slot in range(slot_count):
        start = 4 * slot
        glyph = state[:, start : start + 3]
        mask = glyph != background

        _require(np.any(mask), "every glyph must be nonempty")
        nonzero_colors = np.unique(glyph[mask])
        _require(
            nonzero_colors.size == 1,
            "a glyph slot must use exactly one carrier color",
        )

        masks.append(mask)
        colors.append(int(nonzero_colors[0]))

    return masks, tuple(colors)


def _analyze_pair(pair: dict) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    _require(output_grid.shape[0] == 3, "output must be one state high")
    _require(
        input_grid.shape[1] == output_grid.shape[1],
        "input and output widths must match",
    )

    width = input_grid.shape[1]
    _require(
        (width + 1) % 4 == 0,
        "width must encode 3-column glyphs with 1-column gaps",
    )
    slot_count = (width + 1) // 4
    _require(3 <= slot_count <= 7, "there must be three to seven slots")

    input_height = input_grid.shape[0]
    _require(
        (input_height - 1) % 4 == 0,
        "input height must encode bounded three-row states",
    )
    history_length = (input_height - 1) // 4
    _require(
        history_length in (1, 2, 3),
        "history length must be one, two, or three",
    )

    background = int(input_grid[0, 0])
    for separator_row in range(0, input_height, 4):
        _require(
            np.all(input_grid[separator_row, :] == background),
            "chronological states must be separated and bounded by blank rows",
        )

    for gap_column in range(3, width, 4):
        _require(
            np.all(input_grid[:, gap_column] == background),
            "input slot gaps must be blank",
        )
        _require(
            np.all(output_grid[:, gap_column] == background),
            "output slot gaps must be blank",
        )

    states: list[list[np.ndarray]] = []
    carriers: tuple[int, ...] | None = None

    for time_index in range(history_length):
        row_start = 1 + 4 * time_index
        state = input_grid[row_start : row_start + 3, :]

        _require(
            np.all(np.any(state != background, axis=1)),
            "each row of a displayed state must contain a glyph cell",
        )

        masks, state_carriers = _decode_state(state, slot_count, background)
        if carriers is None:
            carriers = state_carriers
            _require(
                len(set(carriers)) == slot_count,
                "carrier colors must be distinct across slots",
            )
        else:
            _require(
                state_carriers == carriers,
                "carrier colors must remain fixed at their slots",
            )
        states.append(masks)

    _require(
        np.all(np.any(output_grid != background, axis=1)),
        "each output row must contain a glyph cell",
    )
    output_masks, output_carriers = _decode_state(
        output_grid,
        slot_count,
        background,
    )
    _require(
        output_carriers == carriers,
        "output must use the same positional carrier colors",
    )

    initial_keys = tuple(_mask_key(mask) for mask in states[0])
    _require(
        len(set(initial_keys)) == slot_count,
        "glyph masks within a pair must be distinct",
    )

    for time_index in range(history_length - 1):
        expected_next = _shift_masks_left(states[time_index])
        for slot in range(slot_count):
            _require(
                np.array_equal(
                    states[time_index + 1][slot],
                    expected_next[slot],
                ),
                "successive states must shift complete masks one slot left",
            )

    expected_output_masks = _shift_masks_left(states[-1])
    for slot in range(slot_count):
        _require(
            np.array_equal(output_masks[slot], expected_output_masks[slot]),
            "output masks must be the next cyclic state",
        )

    expected_output = _render_state(
        expected_output_masks,
        carriers,
        background,
    )
    _require(
        np.array_equal(output_grid, expected_output),
        "output must exactly match the independently reconstructed state",
    )

    return {
        "slots": slot_count,
        "history": history_length,
        "carriers": carriers,
        "background": background,
        "initial_masks": initial_keys,
        "cell_counts": tuple(
            int(np.count_nonzero(mask)) for mask in states[0]
        ),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _require(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _require(isinstance(puzzle["train"], list), "train must be a list")
    _require(isinstance(puzzle["test"], list), "test must be a list")

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _require(2 <= train_count <= 10, "illegal training-pair count")
    _require(1 <= test_count <= 4, "illegal test-pair count")
    _require(train_count > test_count, "train count must exceed test count")

    all_pairs = puzzle["train"] + puzzle["test"]
    input_keys: set[tuple] = set()
    pair_keys: set[tuple] = set()

    for pair in all_pairs:
        _require(isinstance(pair, dict), "each pair must be a dictionary")
        _require(
            set(pair.keys()) == {"input", "output"},
            "each pair must contain exactly input and output",
        )

        for grid_name in ("input", "output"):
            grid = pair[grid_name]
            _require(
                isinstance(grid, np.ndarray),
                f"{grid_name} must be a NumPy array",
            )
            _require(grid.ndim == 2, f"{grid_name} must be two-dimensional")
            _require(
                1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                f"{grid_name} dimensions are outside ARC limits",
            )
            _require(
                np.issubdtype(grid.dtype, np.integer),
                f"{grid_name} must have an integer dtype",
            )
            _require(
                int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9,
                f"{grid_name} contains an invalid ARC color",
            )

        input_key = _grid_key(pair["input"])
        pair_key = (input_key, _grid_key(pair["output"]))
        _require(
            input_key not in input_keys,
            "duplicate input grid found within puzzle",
        )
        _require(
            pair_key not in pair_keys,
            "duplicate input/output pair found within puzzle",
        )
        input_keys.add(input_key)
        pair_keys.add(pair_key)

        _require(
            not _is_source_pair(pair["input"], pair["output"]),
            "generated pair exactly duplicates a source pair",
        )

    train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
    test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
    all_info = train_info + test_info

    _require(
        any(
            info["slots"] >= 3 and info["history"] >= 2
            for info in train_info
        ),
        "training must demonstrate a visible leftward step and wraparound",
    )
    # Slot colors and background are observable in each input. They may change
    # between pairs; only the transfer rule is shared across the puzzle.
    for info in all_info:
        _require(
            len(set(info["carriers"])) == info["slots"]
            and info["background"] not in info["carriers"],
            "slot colors must be distinct from each other and the background",
        )
    direction = _direction_from_training(puzzle["train"])
    for pair in puzzle["test"]:
        _require(
            np.array_equal(_prediction_from_input(pair["input"], direction), pair["output"]),
            "test output cannot be recovered from training and its input",
        )


def _puzzle_key(puzzle: dict) -> tuple:
    return (
        tuple(
            (_grid_key(pair["input"]), _grid_key(pair["output"]))
            for pair in puzzle["train"]
        ),
        tuple(
            (_grid_key(pair["input"]), _grid_key(pair["output"]))
            for pair in puzzle["test"]
        ),
    )



def _prediction_from_input(input_grid: np.ndarray, direction: int) -> np.ndarray:
    """Rebuild the next band from visible pixels, with no generation metadata."""
    background = int(input_grid[0, 0])
    width = input_grid.shape[1]
    slots = (width + 1) // 4
    last = input_grid[-4:-1, :]
    result = np.full((3, width), background, dtype=np.int8)
    for source in range(slots):
        destination = (source + direction) % slots
        source_block = last[:, 4 * source:4 * source + 3]
        destination_block = last[:, 4 * destination:4 * destination + 3]
        ink = np.unique(destination_block[destination_block != background])
        _require(ink.size == 1, "cannot infer destination carrier color")
        result[:, 4 * destination:4 * destination + 3][source_block != background] = int(ink[0])
    return result


def _direction_from_training(train: list[dict]) -> int:
    consistent = []
    for direction in (-1, 1):
        if all(np.array_equal(
            _prediction_from_input(pair["input"], direction), pair["output"]
        ) for pair in train):
            consistent.append(direction)
    _require(consistent == [-1], "training must identify the leftward rule uniquely")
    return -1


def self_test() -> None:
    signatures = set()
    train_counts, test_counts = set(), set()
    layouts, sizes, backgrounds, palettes, masks = set(), set(), set(), set(), set()
    dense = sparse = total = reused_test_puzzles = fresh_test_puzzles = 0
    one_state_tests = long_slot_tests = three_state_four_slot = 0
    carrier_changes = 0
    first_layouts = set()
    symmetric = asymmetric = connected = disconnected = 0
    familiar_new_position = familiar_new_color = 0
    zero_ink_pairs = first_slot_two = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair) for pair in puzzle["test"]]
        all_info = train_info + test_info
        first_layouts.add((train_info[0]["slots"], train_info[0]["history"]))
        carrier_changes += len({info["carriers"] for info in all_info}) > 1
        training_masks = {key for info in train_info for key in info["initial_masks"]}
        training_glyphs = {}
        for info in train_info:
            for slot, key in enumerate(info["initial_masks"]):
                training_glyphs.setdefault(key, []).append((slot, info["carriers"][slot]))
        if any(training_masks.intersection(info["initial_masks"]) for info in test_info):
            reused_test_puzzles += 1
        if any(training_masks.isdisjoint(info["initial_masks"]) for info in test_info):
            fresh_test_puzzles += 1
        one_state_tests += any(info["history"] == 1 for info in test_info)
        long_slot_tests += any(info["slots"] >= 5 for info in test_info)
        three_state_four_slot += any(
            info["slots"] == 4 and info["history"] == 3 for info in all_info
        )
        for info in test_info:
            for slot, key in enumerate(info["initial_masks"]):
                prior = training_glyphs.get(key, ())
                familiar_new_position += any(slot != old_slot for old_slot, _ in prior)
                familiar_new_color += any(info["carriers"][slot] != old_color
                                        for _, old_color in prior)
        for pair, info in zip(puzzle["train"] + puzzle["test"], all_info):
            layouts.add((info["slots"], info["history"]))
            sizes.add(tuple(pair["input"].shape))
            backgrounds.add(info["background"])
            palettes.add(info["carriers"])
            zero_ink_pairs += 0 in info["carriers"]
            first_slot_two += info["carriers"][0] == 2
            masks.update(info["initial_masks"])
            for key, count in zip(info["initial_masks"], info["cell_counts"]):
                total += 1
                dense += 4 <= count <= 7
                sparse += count <= 3
                mask = _mask_from_key(key)
                is_symmetric = any(np.array_equal(mask, transformed) for transformed in
                                   (np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2)))
                symmetric += is_symmetric
                asymmetric += not is_symmetric
                is_connected = _mask_is_connected(mask)
                connected += is_connected
                disconnected += not is_connected
        direction = _direction_from_training(puzzle["train"])
        for pair in puzzle["test"]:
            _require(np.array_equal(
                _prediction_from_input(pair["input"].copy(), direction),
                pair["output"]
            ), "input-only prediction disagrees with test output")

    _require(len(signatures) == 42, "cross-seed puzzles repeated")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(layouts) >= 12 and len(sizes) >= 12,
             "slot/history layouts lack diversity")
    _require(len(first_layouts) >= 4, "training schedules remain fixed")
    _require(len(backgrounds) >= 5 and len(palettes) >= 40,
             "pair color assignments lack diversity")
    _require(carrier_changes >= 40, "carrier colors are fixed within puzzles")
    _require(len(masks) >= 200 and dense / total >= 0.65 and sparse > 0,
             "glyph masks lack source-like density and variety")
    _require(min(symmetric, asymmetric, connected, disconnected) >= 40,
             "glyph symmetry or connectivity variety is missing")
    _require(reused_test_puzzles >= 8 and fresh_test_puzzles >= 8,
             "familiar and fresh test masks must both occur")
    _require(familiar_new_position >= 12 and familiar_new_color >= 12,
             "familiar test glyphs must move into new positions and colors")
    _require(zero_ink_pairs >= 8 and first_slot_two >= 8,
             "background and first-carrier color range is too narrow")
    _require(one_state_tests >= 5 and long_slot_tests >= 5
             and three_state_four_slot >= 2,
             "harder slot/history combinations are missing")

    for seed, expected in ((731, None), (9123, (7, 3))):
        kwargs = {} if expected is None else {"num_train": expected[0], "num_test": expected[1]}
        a, b = generate_puzzle(seed, **kwargs), generate_puzzle(seed, **kwargs)
        _require(_puzzle_key(a) == _puzzle_key(b), "generation is not deterministic")
        if expected is not None:
            _require((len(a["train"]), len(a["test"])) == expected,
                     "explicit pair counts not honored")
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(10000 + index, *counts)
        _require((len(puzzle["train"]), len(puzzle["test"])) == counts,
                 "boundary counts not honored")
    for index, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(11000 + index, *counts)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "partial train count not honored")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "partial test count not honored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(12000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count specification did not raise ValueError")
