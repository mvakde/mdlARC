import hashlib

import numpy as np


_RELATIONS = (
    "above", "below", "left", "right",
    "above_left", "above_right", "below_left", "below_right",
)
_SOURCE_DIGESTS = {
    "4e0e02ee81d5bc0272875723db1c023c69e8ded0dac82411a43800106cccb0ea",
    "30d4b7d901e58ca757a3ad4f81a3042578d7c389858caf22ef93b4067835ebe0",
    "9af8334b3a75d32d014029b61e0a5495ce6c746689bfad84862a92b1a1d23015",
    "f5ed812aa41cd91f01db368ce0e4af29e9c20364c1a48dff234a3cfac326f0dd",
}

_PLUS_MASK = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)
_ARMS_MASK = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool
)


def _checked_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
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


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _checked_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = minimum_train + (
            (seed // 4 + 3 * seed) % span
        )

    elif num_train is None:
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = minimum_train + (
            (seed + 17 * num_test) % span
        )

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + (
            (seed + 13 * num_train) % maximum_test
        )

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in mask.ravel())


def _has_full_bbox(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        and mask[-1].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    transforms = []

    for turns in range(4):
        rotated = np.rot90(mask, turns)
        transforms.append(rotated)
        transforms.append(np.fliplr(rotated))

    original = _mask_key(mask)
    matches = sum(
        _mask_key(transformed) == original
        for transformed in transforms
    )
    return matches == 1


def _component_count(mask: np.ndarray) -> int:
    remaining = {tuple(int(v) for v in point) for point in np.argwhere(mask)}
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    return count


def _build_mask_catalog() -> dict[str, list[np.ndarray]]:
    catalog = {name: [] for name in (
        "two_corners", "near_full", "ring", "full", "arms", "plus", "general"
    )}
    for bits in range(512):
        mask = np.array(
            [(bits >> index) & 1 for index in range(9)], dtype=bool
        ).reshape(3, 3)
        if not _has_full_bbox(mask):
            continue
        count = int(mask.sum())
        if count == 2:
            category = "two_corners"
        elif count == 8 and not mask[1, 1]:
            category = "ring"
        elif count == 8:
            category = "near_full"
        elif count == 9:
            category = "full"
        elif np.array_equal(mask, _ARMS_MASK):
            category = "arms"
        elif np.array_equal(mask, _PLUS_MASK):
            category = "plus"
        else:
            category = "general"
        catalog[category].append(mask)
    return catalog


_MASK_CATALOG = _build_mask_catalog()


def _sample_mask(
    rng: np.random.Generator,
    used: set[tuple[int, ...]],
) -> np.ndarray:
    categories = tuple(_MASK_CATALOG)
    weights = (0.12, 0.09, 0.05, 0.10, 0.11, 0.06, 0.47)
    preferred = categories[int(rng.choice(len(categories), p=weights))]
    available = [mask for mask in _MASK_CATALOG[preferred]
                 if _mask_key(mask) not in used]
    if not available:
        available = [mask for masks in _MASK_CATALOG.values()
                     for mask in masks if _mask_key(mask) not in used]
    if not available:
        raise RuntimeError("exhausted miniature designs")
    return available[int(rng.integers(len(available)))].copy()


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [int(value) for value in rng.permutation(10)]
    return {
        "palette": colors,
        "stable_colors": bool(rng.random() < 0.5),
        "source_like": bool(rng.random() < 0.34),
    }


def _sample_relations(rng: np.random.Generator, count: int) -> list[str]:
    relations = []
    while len(relations) < count:
        relations.extend(_RELATIONS[int(i)] for i in rng.permutation(len(_RELATIONS)))
    return relations[:count]


def _sample_colors(
    rng: np.random.Generator, parameters: dict, index: int, regime: str
) -> tuple[int, int, int]:
    palette = parameters["palette"]
    common_accent = palette[2]
    if regime == "source_train" and index < 2:
        accent = common_accent
    elif rng.random() < 0.48:
        accent = common_accent
    else:
        accent = palette[int(rng.integers(2, 9))]
    if parameters["stable_colors"]:
        return palette[0], palette[1], accent
    if regime == "source_train" and index < 2:
        return ((palette[0], palette[1], accent) if index == 0
                else (palette[1], palette[0], accent))
    remaining = [color for color in palette if color != accent]
    chosen = rng.choice(remaining, size=2, replace=False)
    return int(chosen[0]), int(chosen[1]), accent


def _layout_objects(
    rng: np.random.Generator,
    outer_side: int,
    relation: str,
) -> tuple[int, int, int, int, int, int]:
    assert relation in _RELATIONS
    # At scale seven, a detached 3 by 3 mask and one-cell margins still fit.
    max_gap = min(5, 25 - outer_side)
    gap_y = int(rng.integers(1, max_gap + 1))
    gap_x = int(rng.integers(1, max_gap + 1))
    frame_y = frame_x = 0
    if relation in {"above", "below"}:
        mask_x = int(rng.integers(0, outer_side - 2))
        mask_y = -3 - gap_y if relation == "above" else outer_side + gap_y
    elif relation in {"left", "right"}:
        mask_y = int(rng.integers(0, outer_side - 2))
        mask_x = -3 - gap_x if relation == "left" else outer_side + gap_x
    else:
        vertical, horizontal = relation.split("_")
        mask_y = -3 - gap_y if vertical == "above" else outer_side + gap_y
        mask_x = -3 - gap_x if horizontal == "left" else outer_side + gap_x

    low_y, low_x = min(frame_y, mask_y), min(frame_x, mask_x)
    frame_y, mask_y = frame_y - low_y, mask_y - low_y
    frame_x, mask_x = frame_x - low_x, mask_x - low_x
    object_height = max(frame_y + outer_side, mask_y + 3)
    object_width = max(frame_x + outer_side, mask_x + 3)
    extra_height = int(rng.integers(0, 30 - object_height - 1))
    extra_width = int(rng.integers(0, 30 - object_width - 1))
    top = int(rng.integers(1, extra_height + 2))
    left = int(rng.integers(1, extra_width + 2))
    return (
        object_height + 2 + extra_height,
        object_width + 2 + extra_width,
        frame_y + top, frame_x + left,
        mask_y + top, mask_x + left,
    )


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    colors: tuple[int, int, int],
    mask: np.ndarray,
    scale: int,
    relation: str,
    accent: int,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in {"source_train", "target_test"}:
        raise ValueError("unknown pair regime")
    if not 1 <= scale <= 7 or relation not in _RELATIONS:
        raise ValueError("illegal scale or relation")
    background, frame, accent_color = colors
    assert accent == accent_color
    assert len(set(colors)) == 3
    outer_side = 3 * scale + 2

    (
        height,
        width,
        frame_y,
        frame_x,
        mask_y,
        mask_x,
    ) = _layout_objects(rng, outer_side, relation)

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    input_grid[
        frame_y,
        frame_x : frame_x + outer_side,
    ] = frame
    input_grid[
        frame_y + outer_side - 1,
        frame_x : frame_x + outer_side,
    ] = frame
    input_grid[
        frame_y : frame_y + outer_side,
        frame_x,
    ] = frame
    input_grid[
        frame_y : frame_y + outer_side,
        frame_x + outer_side - 1,
    ] = frame

    for row, column in np.argwhere(mask):
        input_grid[
            mask_y + int(row),
            mask_x + int(column),
        ] = accent

    output_grid = np.full(
        (outer_side, outer_side),
        background,
        dtype=np.int8,
    )
    output_grid[0, :] = frame
    output_grid[-1, :] = frame
    output_grid[:, 0] = frame
    output_grid[:, -1] = frame

    scaled_mask = np.repeat(
        np.repeat(mask, scale, axis=0),
        scale,
        axis=1,
    )
    output_grid[1:-1, 1:-1] = np.where(
        scaled_mask,
        accent,
        background,
    )

    return input_grid, output_grid


def _grid_key(
    grid: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, ...]]:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _source_digest(input_grid: np.ndarray, output_grid: np.ndarray) -> str:
    contents = repr((input_grid.tolist(), output_grid.tolist())).encode()
    return hashlib.sha256(contents).hexdigest()


def _bbox(
    grid: np.ndarray,
    color: int,
) -> tuple[int, int, int, int] | None:
    coordinates = np.argwhere(grid == color)
    if len(coordinates) == 0:
        return None

    return (
        int(coordinates[:, 0].min()),
        int(coordinates[:, 0].max()),
        int(coordinates[:, 1].min()),
        int(coordinates[:, 1].max()),
    )


def _inspect_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    colors = [int(value) for value in np.unique(input_grid)]
    assert len(colors) == 3

    candidates = []

    for frame in colors:
        box = _bbox(input_grid, frame)
        assert box is not None
        r0, r1, c0, c1 = box

        side_height = r1 - r0 + 1
        side_width = c1 - c0 + 1

        if side_height != side_width:
            continue
        if side_height < 5:
            continue
        if (side_height - 2) % 3 != 0:
            continue

        frame_region = input_grid[
            r0 : r1 + 1,
            c0 : c1 + 1,
        ]

        perimeter = np.zeros(
            frame_region.shape,
            dtype=bool,
        )
        perimeter[0, :] = True
        perimeter[-1, :] = True
        perimeter[:, 0] = True
        perimeter[:, -1] = True

        if not np.all(frame_region[perimeter] == frame):
            continue

        if int(np.count_nonzero(input_grid == frame)) != int(
            perimeter.sum()
        ):
            continue

        interior_colors = np.unique(
            frame_region[1:-1, 1:-1]
        )
        if len(interior_colors) != 1:
            continue

        background = int(interior_colors[0])
        if background == frame:
            continue

        accents = [
            color
            for color in colors
            if color not in {background, frame}
        ]
        if len(accents) != 1:
            continue

        accent = accents[0]
        mask_box = _bbox(input_grid, accent)
        assert mask_box is not None
        mr0, mr1, mc0, mc1 = mask_box

        if (mr1 - mr0 + 1, mc1 - mc0 + 1) != (3, 3):
            continue

        mask = (
            input_grid[
                mr0 : mr1 + 1,
                mc0 : mc1 + 1,
            ]
            == accent
        )

        if not _has_full_bbox(mask):
            continue

        expected_input = np.full(
            input_grid.shape,
            background,
            dtype=input_grid.dtype,
        )

        expected_input[r0, c0 : c1 + 1] = frame
        expected_input[r1, c0 : c1 + 1] = frame
        expected_input[r0 : r1 + 1, c0] = frame
        expected_input[r0 : r1 + 1, c1] = frame

        for row, column in np.argwhere(mask):
            expected_input[
                mr0 + int(row),
                mc0 + int(column),
            ] = accent

        if not np.array_equal(expected_input, input_grid):
            continue

        row_relation = (
            "above" if mr1 < r0 else
            "below" if r1 < mr0 else ""
        )
        column_relation = (
            "left" if mc1 < c0 else
            "right" if c1 < mc0 else ""
        )
        row_gap = max(r0 - mr1 - 1, mr0 - r1 - 1, 0)
        column_gap = max(c0 - mc1 - 1, mc0 - c1 - 1, 0)
        if max(row_gap, column_gap) < 1:
            continue
        relation = "_".join(
            part for part in (row_relation, column_relation) if part
        )
        if relation not in _RELATIONS:
            continue

        if min(r0, c0, mr0, mc0) < 1:
            continue
        if max(r1, mr1) >= input_grid.shape[0] - 1:
            continue
        if max(c1, mc1) >= input_grid.shape[1] - 1:
            continue

        scale = (side_height - 2) // 3

        expected_output = np.full(
            (side_height, side_height),
            background,
            dtype=output_grid.dtype,
        )
        expected_output[0, :] = frame
        expected_output[-1, :] = frame
        expected_output[:, 0] = frame
        expected_output[:, -1] = frame

        scaled_mask = np.repeat(
            np.repeat(mask, scale, axis=0),
            scale,
            axis=1,
        )
        expected_output[1:-1, 1:-1] = np.where(
            scaled_mask,
            accent,
            background,
        )

        if not np.array_equal(
            expected_output,
            output_grid,
        ):
            continue

        candidates.append(
            {
                "background": background,
                "frame": frame,
                "accent": accent,
                "scale": scale,
                "relation": relation,
                "mask": mask.copy(),
                "frame_box": (r0, r1, c0, c1),
                "mask_box": (mr0, mr1, mc0, mc1),
            }
        )

    assert len(candidates) == 1
    return candidates[0]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer")

    seed = int(seed)
    num_train, num_test = _resolve_counts(
        seed,
        num_train,
        num_test,
    )

    rng = np.random.default_rng(seed)
    parameters = _sample_puzzle_parameters(rng)

    pair_rngs = [
        np.random.default_rng(int(rng.integers(0, 2**63)))
        for _ in range(num_train + num_test)
    ]
    scale_pool = (1, 2, 3) if parameters["source_like"] else (1, 2, 3, 4, 5)
    train_scales = [int(value) for value in rng.permutation(scale_pool)]
    while len(train_scales) < num_train:
        train_scales.append(int(rng.choice(scale_pool)))
    train_scales = train_scales[:num_train]
    train_relations = _sample_relations(rng, num_train)

    train = []
    train_masks = []
    used_train_masks = set()
    for index in range(num_train):
        pair_rng = pair_rngs[index]
        mask = _sample_mask(pair_rng, used_train_masks)
        used_train_masks.add(_mask_key(mask))
        train_masks.append(mask)
        colors = _sample_colors(pair_rng, parameters, index, "source_train")
        input_grid, output_grid = _generate_pair(
            pair_rng, "source_train", colors, mask,
            train_scales[index], train_relations[index], colors[2],
        )
        train.append({"input": input_grid, "output": output_grid})

    test = []
    used_test_masks = set()
    for index in range(num_test):
        pair_rng = pair_rngs[num_train + index]
        repeat_probability = 0.18 if index == 0 else 0.28
        if pair_rng.random() < repeat_probability:
            options = [mask for mask in train_masks
                       if _mask_key(mask) not in used_test_masks]
            mask = (options[int(pair_rng.integers(len(options)))].copy()
                    if options else _sample_mask(
                        pair_rng, used_train_masks | used_test_masks
                    ))
        else:
            mask = _sample_mask(pair_rng, used_train_masks | used_test_masks)
        used_test_masks.add(_mask_key(mask))
        if parameters["source_like"] and index == 0:
            scale = 4
        elif pair_rng.random() < 0.39:
            scale = int(pair_rng.choice(train_scales))
        else:
            scale = int(pair_rng.integers(4, 8))
        if pair_rng.random() < 0.35:
            relation = train_relations[int(pair_rng.integers(num_train))]
        else:
            relation = _RELATIONS[int(pair_rng.integers(len(_RELATIONS)))]
        colors = _sample_colors(pair_rng, parameters, index, "target_test")
        input_grid, output_grid = _generate_pair(
            pair_rng, "target_test", colors, mask, scale, relation, colors[2]
        )
        test.append({"input": input_grid, "output": output_grid})

    puzzle = {
        "train": train,
        "test": test,
    }

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()

    analyses = {
        "train": [],
        "test": [],
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(
                    grid.dtype,
                    np.integer,
                )
                assert np.all((grid >= 0) & (grid <= 9))

            assert _source_digest(input_grid, output_grid) not in _SOURCE_DIGESTS

            input_key = _grid_key(input_grid)
            pair_key = (
                input_key,
                _grid_key(output_grid),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            assert not np.array_equal(
                input_grid,
                output_grid,
            )

            analyses[split].append(
                _inspect_pair(
                    input_grid,
                    output_grid,
                )
            )

    train_info = analyses["train"]
    test_info = analyses["test"]
    all_info = train_info + test_info

    assert len({info["scale"] for info in train_info}) >= 2
    assert len({info["relation"] for info in train_info}) >= 2
    assert all(1 <= info["scale"] <= 7 for info in all_info)
    assert all(len({info["background"], info["frame"], info["accent"]}) == 3
               for info in all_info)

    train_masks = [
        _mask_key(info["mask"])
        for info in train_info
    ]
    test_masks = [
        _mask_key(info["mask"])
        for info in test_info
    ]
    assert len(set(train_masks)) == len(train_masks)
    assert len(set(test_masks)) == len(test_masks)


def _puzzles_equal(
    left: dict,
    right: dict,
) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False

    for split in ("train", "test"):
        if len(left[split]) != len(right[split]):
            return False

        for left_pair, right_pair in zip(
            left[split],
            right[split],
        ):
            if not np.array_equal(
                left_pair["input"],
                right_pair["input"],
            ):
                return False
            if not np.array_equal(
                left_pair["output"],
                right_pair["output"],
            ):
                return False

    return True


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                (
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    automatic = generate_puzzle(12345)
    explicit = generate_puzzle(
        12345,
        num_train=7,
        num_test=3,
    )

    validate_generated_puzzle(automatic)
    validate_generated_puzzle(explicit)

    assert _puzzles_equal(
        automatic,
        generate_puzzle(12345),
    )
    assert _puzzles_equal(
        explicit,
        generate_puzzle(
            12345,
            num_train=7,
            num_test=3,
        ),
    )

    assert not _puzzles_equal(
        generate_puzzle(100),
        generate_puzzle(101),
    )

    for train_count, test_count in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            200 + train_count + test_count,
            num_train=train_count,
            num_test=test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    partial_cases = (
        (301, None, 1),
        (302, None, 4),
        (303, 2, None),
        (304, 10, None),
    )

    for seed, train_count, test_count in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )

        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

        validate_generated_puzzle(puzzle)

    illegal_calls = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    )

    for arguments in illegal_calls:
        try:
            generate_puzzle(9, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal count arguments accepted: {arguments}"
            )

    train_counts = set()
    test_counts = set()
    color_role_pairs = set()
    canvas_shapes = set()
    motifs = set()
    layouts = set()
    puzzle_keys = set()
    total_pairs = 0
    scales = {"train": set(), "test": set()}
    relations = {"train": set(), "test": set()}
    occupied = {"train": set(), "test": set()}
    components = {"train": set(), "test": set()}
    asymmetry = {"train": set(), "test": set()}
    combinations = set()
    gaps = set()
    aligned_offsets = set()
    center_holes = {"train": 0, "test": 0}
    arm_masks = {"train": 0, "test": 0}
    novel_test_puzzles = 0
    repeated_test_puzzles = 0
    stable_color_puzzles = 0
    reassigned_color_puzzles = 0
    trained_side_tests = 0
    trained_scale_tests = 0
    shared_accent_tests = 0
    independent_scale_accent_puzzles = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))

        current_input_keys = set()
        infos = {
            split: [_inspect_pair(pair["input"], pair["output"])
                    for pair in puzzle[split]]
            for split in ("train", "test")
        }
        train_masks = {_mask_key(info["mask"]) for info in infos["train"]}
        test_masks = {_mask_key(info["mask"]) for info in infos["test"]}
        novel_test_puzzles += bool(test_masks - train_masks)
        repeated_test_puzzles += bool(test_masks & train_masks)
        role_pairs = {(info["background"], info["frame"])
                      for info in infos["train"] + infos["test"]}
        stable_color_puzzles += len(role_pairs) == 1
        reassigned_color_puzzles += len(role_pairs) > 1
        trained_side_tests += bool(
            {info["relation"] for info in infos["train"]}
            & {info["relation"] for info in infos["test"]}
        )
        trained_scale_tests += bool(
            {info["scale"] for info in infos["train"]}
            & {info["scale"] for info in infos["test"]}
        )
        shared_accent_tests += bool(
            {info["accent"] for info in infos["train"]}
            & {info["accent"] for info in infos["test"]}
        )
        independent_scale_accent_puzzles += bool(
            any(left["accent"] == right["accent"] and
                left["scale"] != right["scale"]
                for left in infos["train"] for right in infos["train"])
        )

        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                total_pairs += 1

                input_grid = pair["input"]
                output_grid = pair["output"]

                current_input_keys.add(
                    _grid_key(input_grid)
                )
                scales[split].add(info["scale"])
                relations[split].add(info["relation"])
                occupied[split].add(int(info["mask"].sum()))
                components[split].add(_component_count(info["mask"]))
                asymmetry[split].add(_is_fully_asymmetric(info["mask"]))
                center_holes[split] += bool(
                    not info["mask"][1, 1] and int(info["mask"].sum()) == 8
                )
                arm_masks[split] += bool(np.array_equal(info["mask"], _ARMS_MASK))
                combinations.add((split, info["scale"], info["relation"],
                                  int(info["mask"].sum())))
                fr0, fr1, fc0, fc1 = info["frame_box"]
                mr0, mr1, mc0, mc1 = info["mask_box"]
                if mr1 < fr0:
                    gaps.add(fr0 - mr1 - 1)
                if fr1 < mr0:
                    gaps.add(mr0 - fr1 - 1)
                if mc1 < fc0:
                    gaps.add(fc0 - mc1 - 1)
                if fc1 < mc0:
                    gaps.add(mc0 - fc1 - 1)
                if info["relation"] in {"above", "below"}:
                    aligned_offsets.add(mc0 - fc0)
                if info["relation"] in {"left", "right"}:
                    aligned_offsets.add(mr0 - fr0)
                color_role_pairs.add(
                    (
                        info["background"],
                        info["frame"],
                    )
                )
                canvas_shapes.add(input_grid.shape)
                motifs.add(_mask_key(info["mask"]))
                layouts.add(
                    (
                        split,
                        info["scale"],
                        info["relation"],
                        input_grid.shape,
                        info["frame_box"],
                        info["mask_box"],
                    )
                )

                assert _source_digest(input_grid, output_grid) not in _SOURCE_DIGESTS

        expected_pair_count = (
            len(puzzle["train"])
            + len(puzzle["test"])
        )
        assert len(current_input_keys) == expected_pair_count

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_keys) == 42

    assert len(color_role_pairs) >= 15
    assert len(canvas_shapes) >= 60
    assert len(motifs) >= 35
    assert len(layouts) >= max(
        100,
        total_pairs // 2,
    )
    assert scales["train"] >= {1, 2, 3, 4, 5}
    assert scales["test"] >= {1, 2, 3, 4, 5, 6, 7}
    assert relations["train"] == relations["test"] == set(_RELATIONS)
    assert occupied["train"] >= {2, 8, 9}
    assert occupied["test"] >= {2, 8, 9}
    assert len(components["train"]) >= 3 and len(components["test"]) >= 3
    assert asymmetry["train"] == asymmetry["test"] == {False, True}
    assert center_holes["train"] > 0 and center_holes["test"] > 0
    assert arm_masks["train"] > 0 and arm_masks["test"] > 0
    assert gaps >= {1, 2, 3, 4, 5} and len(aligned_offsets) >= 10
    assert novel_test_puzzles >= 25 and repeated_test_puzzles >= 4
    assert stable_color_puzzles >= 8 and reassigned_color_puzzles >= 8
    assert trained_side_tests >= 8 and trained_scale_tests >= 8
    assert shared_accent_tests >= 8
    assert independent_scale_accent_puzzles == 42
    assert len(combinations) >= 80
