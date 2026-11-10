"""Sparse corner templates for concentric two-cell-lattice square frames."""

from itertools import combinations

import numpy as np


_ORIENTATIONS = {
    "UL": (-1, -1),
    "UR": (-1, 1),
    "LL": (1, -1),
    "LR": (1, 1),
}
_SIZES = tuple(range(11, 30, 2))

# Exact descriptions of the four source pairs, used only to reject copies.
_SOURCE_SIGNATURES = {
    ("UR", ((8, 3, 0), (6, 4, 1))),
    ("UL", ((8, 8, 1), (6, 2, 0))),
    ("UL", ((8, 4, 3), (6, 1, 0), (4, 8, 0))),
    ("LL", ((8, 1, 3), (6, 2, 0), (4, 4, 8))),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name, low, high):
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


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
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


def _radii_for(size, phase):
    return tuple(range(2 + phase, size // 2, 2))


def _build_input(size, background, orientation, rings):
    grid = np.full((size, size), background, dtype=np.int8)
    center = size // 2
    row_sign, col_sign = _ORIENTATIONS[orientation]
    for radius, corner, side in rings:
        row = center + row_sign * radius
        col = center + col_sign * radius
        grid[row, col] = corner
        grid[row - 2 * row_sign, col] = side
        grid[row, col - 2 * col_sign] = side
    return grid


def _complete(input_grid, rings):
    output = input_grid.copy()
    center = input_grid.shape[0] // 2
    for radius, corner, side in rings:
        top, bottom = center - radius, center + radius
        left, right = center - radius, center + radius
        for position in range(-radius + 2, radius, 2):
            output[top, center + position] = side
            output[bottom, center + position] = side
            output[center + position, left] = side
            output[center + position, right] = side
        for row in (top, bottom):
            for col in (left, right):
                output[row, col] = corner
    return output


def _decode_input(grid):
    size = grid.shape[0]
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    _require(int(np.max(counts)) > grid.size // 2, "background is not dominant")
    center = size // 2
    candidates = []
    for orientation, (row_sign, col_sign) in _ORIENTATIONS.items():
        if any(
            int(grid[center + row_sign * radius, center + col_sign * radius])
            != background
            for radius in range(2, center)
        ):
            candidates.append(orientation)
    _require(len(candidates) == 1, "source corner is not unique")
    orientation = candidates[0]
    row_sign, col_sign = _ORIENTATIONS[orientation]
    rings = []
    for radius in range(2, center):
        row = center + row_sign * radius
        col = center + col_sign * radius
        corner = int(grid[row, col])
        if corner == background:
            continue
        vertical = int(grid[row - 2 * row_sign, col])
        horizontal = int(grid[row, col - 2 * col_sign])
        _require(vertical == horizontal, "inward arm colors disagree")
        rings.append((radius, corner, vertical))
    _require(1 <= len(rings) <= 4, "ring count outside supported range")
    _require(
        len({radius % 2 for radius, _, _ in rings}) == 1,
        "rings must share a two-cell lattice phase",
    )
    expected = _build_input(size, background, orientation, rings)
    _require(
        np.array_equal(grid, expected),
        "input has noise or marks outside the corner template",
    )
    return background, orientation, rings


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_signature(size, background, orientation, rings):
    return (
        size == 19
        and background == 0
        and (
            orientation,
            tuple(sorted(rings, reverse=True)),
        )
        in _SOURCE_SIGNATURES
    )


def _layout_key(size, background, orientation, rings):
    return (
        size,
        orientation,
        tuple(radius for radius, _, _ in rings),
        tuple(side != background for _, _, side in rings),
    )


def _sample_background(rng, forbidden=None):
    colors = [color for color in range(10) if color != forbidden]
    weights = np.array([0.46 if color == 0 else 0.06 for color in colors])
    weights /= weights.sum()
    return int(rng.choice(colors, p=weights))


def _sample_role_colors(rng, background, ring_count, visible_count, anchor):
    colors = [color for color in range(10) if color != background]
    slots = ring_count + visible_count
    if anchor:
        return [int(x) for x in rng.choice(colors, size=slots, replace=False)]
    if rng.random() < 0.36:
        roles = [int(x) for x in rng.choice(colors, size=slots, replace=False)]
    else:
        pool_size = int(rng.integers(1, min(8, slots) + 1))
        pool = rng.choice(colors, size=pool_size, replace=False)
        roles = [int(x) for x in rng.choice(pool, size=slots)]
    if background != 0 and rng.random() < 0.35:
        roles[int(rng.integers(slots))] = 0
    return roles


def _sample_pair(
    rng,
    regime,
    anchor=False,
    forbidden_size=None,
    forbidden_orientation=None,
    forbidden_background=None,
    allowed_orientations=None,
    minimum_rings=1,
    maximum_rings=4,
    minimum_visible=0,
    maximum_visible=4,
):
    sizes = [
        size for size in _SIZES
        if size != forbidden_size
        and (not anchor or size >= 13)
        and any(
            len(_radii_for(size, phase)) >= max(
                minimum_rings, minimum_visible, 2 if anchor else 1
            )
            for phase in (0, 1)
        )
    ]
    if regime == "target_test" and rng.random() < 0.27:
        large = [size for size in sizes if size >= 21]
        if large:
            sizes = large
    size = int(rng.choice(sizes))
    orientations = [
        name for name in (
            allowed_orientations
            if allowed_orientations is not None else _ORIENTATIONS
        )
        if name != forbidden_orientation
    ]
    orientation = str(rng.choice(orientations))
    background = _sample_background(rng, forbidden_background)
    phases = [
        phase for phase in (0, 1)
        if len(_radii_for(size, phase)) >= max(
            minimum_rings, minimum_visible, 2 if anchor else 1
        )
    ]
    phase = int(rng.choice(phases))
    available = _radii_for(size, phase)
    maximum = min(maximum_rings, len(available))
    if anchor:
        ring_count = 2
        visible_count = 1
    else:
        ring_count = int(rng.integers(
            max(minimum_rings, minimum_visible), maximum + 1
        ))
        visible_count = int(rng.integers(
            minimum_visible,
            min(maximum_visible, ring_count) + 1,
        ))
    radii = sorted(
        int(x) for x in rng.choice(available, size=ring_count, replace=False)
    )
    visible = set(
        int(x) for x in rng.choice(
            ring_count, size=visible_count, replace=False
        )
    )
    roles = _sample_role_colors(
        rng, background, ring_count, visible_count, anchor
    )
    side_cursor = ring_count
    rings = []
    for index, radius in enumerate(radii):
        side = background
        if index in visible:
            side = roles[side_cursor]
            side_cursor += 1
        rings.append((radius, roles[index], side))
    input_grid = _build_input(size, background, orientation, rings)
    output_grid = _complete(input_grid, rings)
    return (
        {"input": input_grid, "output": output_grid},
        _layout_key(size, background, orientation, rings),
        _source_signature(size, background, orientation, rings),
        (size, background, orientation),
    )


def _fallback_pair(
    used_layouts,
    used_inputs,
    regime,
    anchor=False,
    forbidden_size=None,
    forbidden_orientation=None,
    allowed_orientations=None,
    minimum_rings=1,
    maximum_rings=4,
    minimum_visible=0,
    maximum_visible=4,
):
    # A deterministic finite search is available if unlucky random draws collide.
    for size in _SIZES:
        if size == forbidden_size or (anchor and size < 13):
            continue
        for phase in (0, 1):
            available = _radii_for(size, phase)
            low_n = 2 if anchor else minimum_rings
            high_n = 2 if anchor else min(maximum_rings, len(available))
            for ring_count in range(low_n, high_n + 1):
                for radii in combinations(available, ring_count):
                    low_v = 1 if anchor else minimum_visible
                    high_v = 1 if anchor else min(maximum_visible, ring_count)
                    for visible_count in range(low_v, high_v + 1):
                        for visible_indices in combinations(
                            range(ring_count), visible_count
                        ):
                            for orientation in (
                                allowed_orientations
                                if allowed_orientations is not None
                                else _ORIENTATIONS
                            ):
                                if orientation == forbidden_orientation:
                                    continue
                                background = 0
                                corners = (1, 2, 3, 4)
                                sides = (5, 6, 7, 8)
                                rings = [
                                    (
                                        radius,
                                        corners[index],
                                        sides[visible_indices.index(index)]
                                        if index in visible_indices else background,
                                    )
                                    for index, radius in enumerate(radii)
                                ]
                                layout = _layout_key(
                                    size, background, orientation, rings
                                )
                                if layout in used_layouts or _source_signature(
                                    size, background, orientation, rings
                                ):
                                    continue
                                input_grid = _build_input(
                                    size, background, orientation, rings
                                )
                                if _grid_key(input_grid) in used_inputs:
                                    continue
                                return {
                                    "input": input_grid,
                                    "output": _complete(input_grid, rings),
                                }, layout
    raise RuntimeError(f"no unused {regime} layout exists")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    ring_shift = int(rng.choice((-1, 0, 1), p=(0.24, 0.52, 0.24)))
    side_shift = int(rng.choice((-1, 0, 1), p=(0.24, 0.52, 0.24)))
    if ring_shift == -1 and side_shift == 1:
        side_shift = 0
    pool_size = 2 if rng.random() < 0.63 else 3
    orientation_pool = tuple(
        str(x) for x in rng.permutation(tuple(_ORIENTATIONS))[:pool_size]
    )
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_layouts = set()
    first_size = first_orientation = first_background = None
    test_unseen_orientation = rng.random() < 0.38

    for regime, count in (("source_train", num_train), ("target_test", num_test)):
        split = "train" if regime == "source_train" else "test"
        for index in range(count):
            anchor = regime == "source_train" and index == 0
            accepted = None
            for _ in range(96):
                minimum_rings = 2 if regime == "source_train" and ring_shift == -1 else 1
                maximum_rings = 2 if regime == "source_train" and ring_shift == 1 else 4
                minimum_visible = 1 if regime == "source_train" and side_shift == -1 else 0
                maximum_visible = 1 if regime == "source_train" and side_shift == 1 else 4
                if regime == "target_test" and index == 0:
                    if ring_shift == -1:
                        maximum_rings = 1
                    elif ring_shift == 1:
                        minimum_rings = 3
                    if side_shift == -1:
                        maximum_visible = 0
                    elif side_shift == 1:
                        minimum_visible = 2
                allowed_orientations = (
                    orientation_pool if regime == "source_train" else None
                )
                if (
                    regime == "target_test"
                    and index == 0
                    and test_unseen_orientation
                ):
                    allowed_orientations = tuple(
                        name for name in _ORIENTATIONS
                        if name not in orientation_pool
                    )
                forbidden_background = None
                if (
                    regime == "source_train"
                    and index == 1
                    and rng.random() < 0.52
                ):
                    forbidden_background = first_background
                pair, layout, source_copy, properties = _sample_pair(
                    rng,
                    regime,
                    anchor=anchor,
                    forbidden_size=first_size
                    if regime == "source_train" and index == 1 else None,
                    forbidden_orientation=first_orientation
                    if regime == "source_train" and index == 1 else None,
                    forbidden_background=forbidden_background,
                    allowed_orientations=allowed_orientations,
                    minimum_rings=minimum_rings,
                    maximum_rings=maximum_rings,
                    minimum_visible=minimum_visible,
                    maximum_visible=maximum_visible,
                )
                input_key = _grid_key(pair["input"])
                if (
                    layout not in used_layouts
                    and input_key not in used_inputs
                    and not source_copy
                ):
                    accepted = pair, layout, properties
                    break
            if accepted is None:
                pair, layout = _fallback_pair(
                    used_layouts,
                    used_inputs,
                    regime,
                    anchor=anchor,
                    forbidden_size=first_size
                    if regime == "source_train" and index == 1 else None,
                    forbidden_orientation=first_orientation
                    if regime == "source_train" and index == 1 else None,
                    allowed_orientations=allowed_orientations,
                    minimum_rings=minimum_rings,
                    maximum_rings=maximum_rings,
                    minimum_visible=minimum_visible,
                    maximum_visible=maximum_visible,
                )
                properties = (
                    pair["input"].shape[0],
                    0,
                    _decode_input(pair["input"])[1],
                )
                accepted = pair, layout, properties
            pair, layout, properties = accepted
            if anchor:
                first_size, first_background, first_orientation = properties
            used_inputs.add(_grid_key(pair["input"]))
            used_layouts.add(layout)
            puzzle[split].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(
        isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
        "puzzle needs exactly train and test",
    )
    _require(
        isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
        "train and test must be lists",
    )
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(
        2 <= train_count <= 10 and 1 <= test_count <= 4
        and train_count > test_count,
        "illegal pair counts",
    )
    used_inputs, used_pairs, used_layouts = set(), set(), set()
    train_decoded = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(
                isinstance(pair, dict) and set(pair) == {"input", "output"},
                "pair needs exactly input and output",
            )
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                _require(
                    isinstance(grid, np.ndarray) and grid.ndim == 2,
                    "grid must be a 2D NumPy array",
                )
                _require(
                    all(1 <= extent <= 30 for extent in grid.shape)
                    and np.issubdtype(grid.dtype, np.integer)
                    and bool(np.all((grid >= 0) & (grid <= 9))),
                    "grid has invalid size, dtype, or color",
                )
            size = input_grid.shape[0]
            _require(
                size in _SIZES and output_grid.shape == input_grid.shape
                and output_grid.dtype == input_grid.dtype,
                "canvas must stay odd, square, and unchanged in size/dtype",
            )
            _require(
                not np.array_equal(input_grid, output_grid),
                "template completion must change the grid",
            )
            input_key = _grid_key(input_grid)
            pair_key = input_key, _grid_key(output_grid)
            _require(input_key not in used_inputs, "duplicate input")
            _require(pair_key not in used_pairs, "duplicate pair")
            used_inputs.add(input_key)
            used_pairs.add(pair_key)
            background, orientation, rings = _decode_input(input_grid)
            _require(
                np.array_equal(output_grid, _complete(input_grid, rings)),
                "output does not complete the encoded squares",
            )
            _require(
                not _source_signature(size, background, orientation, rings),
                "exact copy of a source pair",
            )
            layout = _layout_key(size, background, orientation, rings)
            _require(layout not in used_layouts, "repeated geometric layout")
            used_layouts.add(layout)
            if split == "train":
                train_decoded.append((size, background, orientation, rings))
    _require(
        len({size for size, _, _, _ in train_decoded}) >= 2,
        "training must show multiple canvas sizes",
    )
    _require(
        len({orientation for _, _, orientation, _ in train_decoded}) >= 2,
        "training must show multiple source corners",
    )
    _require(
        any(
            any(side == background for _, _, side in rings)
            and any(side != background for _, _, side in rings)
            for _, background, _, rings in train_decoded
        ),
        "training must directly show visible and corner-only rings together",
    )


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a["input"], b["input"])
            and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, backgrounds, orientations, ring_counts = set(), set(), set(), set()
    visible_counts, foreground_counts, layouts = set(), set(), set()
    odd_phases = even_phases = 0
    lower_train = upper_test = seen_test = unseen_test = 0
    zero_foreground = reused_role = same_corner_side = 0
    all_visible = all_invisible = 0
    ring_up = ring_down = side_up = side_down = 0
    varied_train_background = varied_train_rings = 0
    varied_train_visibility = varied_train_phase = 0
    source_like_test = new_test_color = combined_axes = 0
    radius_gaps = set()
    split_ring_counts = {"train": set(), "test": set()}
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(
            tuple(
                (split, tuple(_grid_key(pair["input"]) for pair in puzzle[split]))
                for split in ("train", "test")
            )
        )
        train_orientations = {
            _decode_input(pair["input"])[1] for pair in puzzle["train"]
        }
        train_descriptions = [
            _decode_input(pair["input"]) for pair in puzzle["train"]
        ]
        train_ring_counts = [len(rings) for _, _, rings in train_descriptions]
        train_visibilities = [
            sum(side != background for _, _, side in rings)
            for background, _, rings in train_descriptions
        ]
        train_colors = {
            color
            for background, _, rings in train_descriptions
            for _, corner, side in rings
            for color in (corner, side)
            if color != background
        }
        varied_train_background += len(
            {background for background, _, _ in train_descriptions}
        ) >= 2
        varied_train_rings += len(set(train_ring_counts)) >= 2
        varied_train_visibility += len(set(train_visibilities)) >= 2
        varied_train_phase += len(
            {rings[0][0] % 2 for _, _, rings in train_descriptions}
        ) >= 2
        lower_train += any(o in ("LL", "LR") for o in train_orientations)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background, orientation, rings = _decode_input(grid)
                size = grid.shape[0]
                visible = sum(side != background for _, _, side in rings)
                roles = [corner for _, corner, _ in rings] + [
                    side for _, _, side in rings if side != background
                ]
                sizes.add(size)
                backgrounds.add(background)
                orientations.add(orientation)
                ring_counts.add(len(rings))
                split_ring_counts[split].add(len(rings))
                visible_counts.add(visible)
                foreground_counts.add(len(set(roles)))
                layouts.add(_layout_key(size, background, orientation, rings))
                odd_phases += rings[0][0] % 2
                even_phases += 1 - rings[0][0] % 2
                zero_foreground += background != 0 and 0 in roles
                reused_role += len(set(roles)) < len(roles)
                same_corner_side += any(
                    side == corner and side != background
                    for _, corner, side in rings
                )
                radius_gaps.update(
                    later[0] - earlier[0]
                    for earlier, later in zip(rings, rings[1:])
                )
                combined_axes += (
                    size >= 21 and rings[0][0] % 2 == 1
                    and orientation in ("LL", "LR")
                    and background != 0 and 0 < visible < len(rings)
                    and len(set(roles)) < len(roles)
                )
                all_visible += visible == len(rings)
                all_invisible += visible == 0
                if split == "test":
                    upper_test += orientation in ("UL", "UR")
                    seen_test += orientation in train_orientations
                    unseen_test += orientation not in train_orientations
                    ring_up += len(rings) > max(train_ring_counts)
                    ring_down += len(rings) < min(train_ring_counts)
                    side_up += visible > max(train_visibilities)
                    side_down += visible < min(train_visibilities)
                    source_like_test += (
                        orientation in ("LL", "LR")
                        and len(rings) == 3 and visible == 2
                    )
                    new_test_color += bool(set(roles) - train_colors)
                # Rebuild from the returned input only; no generation metadata.
                _require(
                    np.array_equal(pair["output"], _complete(grid, rings)),
                    "self-test completion disagrees with input",
                )
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(puzzle_keys) == 42, "cross-seed puzzle diversity")
    _require(len(sizes) >= 8 and len(backgrounds) >= 5, "canvas/color diversity")
    _require(orientations == set(_ORIENTATIONS), "orientation diversity")
    _require(ring_counts == {1, 2, 3, 4}, "ring-count diversity")
    _require(
        split_ring_counts["train"] == {1, 2, 3, 4}
        and split_ring_counts["test"] == {1, 2, 3, 4},
        "both splits need the full ring-count range",
    )
    _require(odd_phases >= 20 and even_phases >= 20, "lattice phase diversity")
    _require(
        len(radius_gaps) >= 3, "selected radius gaps did not vary"
    )
    _require(len(layouts) >= 130, "structural layout diversity")
    _require(
        0 in visible_counts and max(visible_counts) >= 3
        and all_visible >= 10 and all_invisible >= 10,
        "visible-side diversity",
    )
    _require(
        min(foreground_counts) == 1 and max(foreground_counts) >= 6
        and zero_foreground >= 10 and reused_role >= 20
        and same_corner_side >= 8,
        "local color-role diversity",
    )
    _require(
        lower_train >= 5 and upper_test >= 10
        and seen_test >= 10 and unseen_test >= 10,
        "train/test orientation diversity",
    )
    _require(
        varied_train_background >= 20 and varied_train_rings >= 20
        and varied_train_visibility >= 20 and varied_train_phase >= 15,
        "within-puzzle pair-level diversity",
    )
    _require(
        ring_up >= 5 and ring_down >= 5
        and side_up >= 5 and side_down >= 5
        and source_like_test >= 1 and new_test_color >= 10,
        "train/test regime diversity",
    )
    _require(combined_axes >= 2, "compatible axes did not combine")
    for seed, train, test in (
        (1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
        (2001, None, 1), (2002, None, 4),
        (2003, 2, None), (2004, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(puzzle["train"]) == train, "num_train request ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "num_test request ignored")
    _require(
        _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3)),
        "fixed seed must be deterministic",
    )
    _require(
        not _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314160, 7, 3)),
        "different seeds must differ",
    )
    for request in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    ):
        try:
            generate_puzzle(999, **request)
        except ValueError:
            continue
        raise AssertionError(f"invalid count request accepted: {request}")
