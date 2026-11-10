import numpy as np


_DIRECTIONS = ("left", "right", "top", "bottom")
_HORIZONTAL = {"left", "right"}

_SOURCE_PAIR_ROWS = {
    (
        (
            "7577557757",
            "0007755775",
            "0007557577",
            "0005757577",
            "7755557555",
            "7575775575",
            "7577555777",
            "7005557575",
            "5007500757",
            "7005500577",
        ),
        (
            "7007500757",
            "7007700000",
            "7007777000",
            "7777777000",
            "7775577555",
            "7775777575",
            "7777577777",
            "7775577575",
            "5777577757",
            "7775577577",
        ),
    ),
    (
        (
            "77575757",
            "75775777",
            "77577000",
            "57757000",
            "77777000",
            "57575577",
        ),
        (
            "77575757",
            "75775777",
            "00077777",
            "00077777",
            "00077777",
            "57575577",
        ),
    ),
    (
        (
            "555700005557",
            "575500007555",
            "577700007577",
            "577557555775",
            "775755755775",
            "757555555775",
            "757755550000",
            "777577550000",
            "577777575755",
            "755575575575",
            "750575775755",
            "750755555755",
        ),
        (
            "550777775557",
            "570577777555",
            "577777777577",
            "577577775775",
            "777777775775",
            "757577775775",
            "000077777777",
            "000077777777",
            "577777775755",
            "757500005575",
            "757500005755",
            "757700005755",
        ),
    ),
    (
        (
            "77777777077577",
            "55777557077777",
            "77577577777000",
            "77577555577000",
            "77757577757775",
            "75775777755775",
            "00000077777577",
            "00000075577777",
            "00000077577777",
            "57577577755000",
            "00077777555575",
            "00075757777577",
            "57577777577777",
            "77777557577700",
            "75757775507500",
            "57700007707777",
            "77700007707775",
            "77700005507577",
        ),
        (
            "77700007707577",
            "55700007707777",
            "00000007707777",
            "00077777707777",
            "77777777777775",
            "75777777775775",
            "77777777000000",
            "77777777000000",
            "77777777000000",
            "00077777777777",
            "77777777777000",
            "77777777777000",
            "57577777777777",
            "00777777777777",
            "00777777777777",
            "57777777777777",
            "77777777077775",
            "77777775077577",
        ),
    ),
}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            cells = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))

                for nr, nc in (
                    (rr - 1, cc),
                    (rr + 1, cc),
                    (rr, cc - 1),
                    (rr, cc + 1),
                ):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            result.append(cells)

    return result


def _extract_movers(grid, mover):
    h, w = grid.shape
    components = _components(grid == mover)

    if not components:
        return None

    rectangles = []

    for cells in components:
        rows = [cell[0] for cell in cells]
        cols = [cell[1] for cell in cells]

        r0 = min(rows)
        r1 = max(rows) + 1
        c0 = min(cols)
        c1 = max(cols) + 1

        if len(cells) != (r1 - r0) * (c1 - c0):
            return None

        if not np.all(grid[r0:r1, c0:c1] == mover):
            return None

        touches = (
            r0 == 0,
            r1 == h,
            c0 == 0,
            c1 == w,
        )

        if sum(touches) != 1:
            return None

        side = ("top", "bottom", "left", "right")[touches.index(True)]
        rectangles.append((r0, r1, c0, c1, side))

    rectangles.sort()
    return rectangles


def _destination(rectangle, h, w):
    r0, r1, c0, c1, side = rectangle
    rect_h = r1 - r0
    rect_w = c1 - c0

    if side == "left":
        return r0, r1, w - rect_w, w
    if side == "right":
        return r0, r1, 0, rect_w
    if side == "top":
        return h - rect_h, h, c0, c1

    return 0, rect_h, c0, c1


def _rect_mask(rectangle, h, w):
    mask = np.zeros((h, w), dtype=bool)
    r0, r1, c0, c1 = rectangle[:4]
    mask[r0:r1, c0:c1] = True
    return mask


def _lane_mask_for(rectangle, h, w):
    mask = np.zeros((h, w), dtype=bool)
    r0, r1, c0, c1, side = rectangle

    if side in _HORIZONTAL:
        mask[r0:r1, :] = True
    else:
        mask[:, c0:c1] = True

    return mask


def _all_masks(rectangles, h, w):
    source = np.zeros((h, w), dtype=bool)
    lanes = np.zeros((h, w), dtype=bool)
    destinations = np.zeros((h, w), dtype=bool)

    for rectangle in rectangles:
        source |= _rect_mask(rectangle, h, w)
        lanes |= _lane_mask_for(rectangle, h, w)
        destinations |= _rect_mask(_destination(rectangle, h, w), h, w)

    return source, lanes, destinations


def _apply_rule(grid, mover, carrier):
    rectangles = _extract_movers(grid, mover)

    if rectangles is None:
        return None

    h, w = grid.shape
    output = grid.copy()

    for rectangle in rectangles:
        r0, r1, c0, c1, side = rectangle

        if side in _HORIZONTAL:
            output[r0:r1, :] = carrier
        else:
            output[:, c0:c1] = carrier

    for rectangle in rectangles:
        r0, r1, c0, c1 = _destination(rectangle, h, w)
        output[r0:r1, c0:c1] = mover

    return output


def _infer_pair_roles(input_grid, output_grid):
    colors = [int(value) for value in np.unique(input_grid)]

    if len(colors) != 3:
        return []

    matches = []

    for mover in colors:
        for carrier in colors:
            if mover == carrier:
                continue

            expected = _apply_rule(input_grid, mover, carrier)

            if expected is not None and np.array_equal(expected, output_grid):
                texture = next(
                    color
                    for color in colors
                    if color not in (mover, carrier)
                )
                matches.append((mover, carrier, texture))

    return matches


def _has_destination_lane_overlap(rectangles, h, w):
    for i, rectangle in enumerate(rectangles):
        destination = _rect_mask(_destination(rectangle, h, w), h, w)

        for j, other in enumerate(rectangles):
            if i == j:
                continue

            axes_differ = (
                (rectangle[4] in _HORIZONTAL)
                != (other[4] in _HORIZONTAL)
            )

            if axes_differ and np.any(
                destination & _lane_mask_for(other, h, w)
            ):
                return True

    return False


def _source_destination_overlap(rectangles, h, w):
    for i, rectangle in enumerate(rectangles):
        destination = _rect_mask(_destination(rectangle, h, w), h, w)
        for j, other in enumerate(rectangles):
            if i != j and np.any(destination & _rect_mask(other, h, w)):
                return True
    return False


def _sample_extent(rng, limit):
    limit = min(7, limit)
    if rng.random() < 0.28:
        return int(rng.integers(1, limit + 1))
    values = np.arange(1, limit + 1)
    weights = 1 / values.astype(float) ** 1.35
    return int(rng.choice(values, p=weights / weights.sum()))


def _candidate_rectangle(rng, h, w, side):
    if side in _HORIZONTAL:
        rect_h = _sample_extent(rng, h - 2)
        rect_w = _sample_extent(rng, w - 2)
        r0 = int(rng.integers(1, h - rect_h))
        c0 = 0 if side == "left" else w - rect_w
        return r0, r0 + rect_h, c0, c0 + rect_w, side
    rect_h = _sample_extent(rng, h - 2)
    rect_w = _sample_extent(rng, w - 2)
    c0 = int(rng.integers(1, w - rect_w))
    r0 = 0 if side == "top" else h - rect_h
    return r0, r0 + rect_h, c0, c0 + rect_w, side


def _place_rectangles(rng, h, w, orientations, require_priority=False,
                      require_source_overlap=False):
    for _ in range(100):
        rectangles = []
        occupied = np.zeros((h, w), dtype=bool)
        destinations = np.zeros((h, w), dtype=bool)
        order = list(orientations)
        rng.shuffle(order)
        for side in order:
            for _ in range(90):
                rectangle = _candidate_rectangle(rng, h, w, side)
                candidate = _rect_mask(rectangle, h, w)
                destination = _rect_mask(_destination(rectangle, h, w), h, w)
                near = occupied.copy()
                near[1:, :] |= occupied[:-1, :]
                near[:-1, :] |= occupied[1:, :]
                near[:, 1:] |= occupied[:, :-1]
                near[:, :-1] |= occupied[:, 1:]
                if not np.any(candidate & near) and not np.any(
                    destination & destinations
                ):
                    rectangles.append(rectangle)
                    occupied |= candidate
                    destinations |= destination
                    break
            else:
                break
        if len(rectangles) != len(orientations):
            continue
        if require_priority and not _has_destination_lane_overlap(
            rectangles, h, w
        ):
            continue
        if require_source_overlap and not _source_destination_overlap(
            rectangles, h, w
        ):
            continue
        extracted = _extract_movers(np.where(occupied, 1, 0), 1)
        if extracted is not None and len(extracted) == len(rectangles):
            return rectangles
    return None


def _sample_shape(rng, regime, count):
    if regime == "source_train":
        low, high = 6, 19
    else:
        low, high = 8, 25
    # Height and width are sampled separately; many movers are not assigned
    # a minimum on both axes.
    h = int(rng.integers(low, high))
    w = int(rng.integers(low, high))
    if count >= 7 and rng.random() < 0.38:
        if rng.random() < 0.5:
            h = int(rng.integers(10, 15))
        else:
            w = int(rng.integers(10, 15))
    return h, w


def _texture_field(rng, h, w, mode):
    if mode == "sparse":
        density = float(rng.uniform(0.025, 0.13))
    elif mode == "dense":
        density = float(rng.uniform(0.42, 0.72))
    else:
        density = float(rng.uniform(0.12, 0.43))
    texture = rng.random((h, w)) < density
    if mode == "cluster":
        for _ in range(int(rng.integers(1, 5))):
            rh = int(rng.integers(2, min(5, h) + 1))
            rw = int(rng.integers(2, min(5, w) + 1))
            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            texture[r0:r0 + rh, c0:c0 + rw] = (
                rng.random((rh, rw)) < float(rng.uniform(0.65, 0.95))
            )
    elif mode == "patch":
        period_r = int(rng.integers(2, 5))
        period_c = int(rng.integers(2, 5))
        rr, cc = np.indices((h, w))
        texture |= (rr % period_r == int(rng.integers(0, period_r))) & (
            cc % period_c == int(rng.integers(0, period_c))
        )
    return texture


def _generate_pair(rng, regime, puzzle_parameters, orientations,
                   require_priority=False, require_source_overlap=False,
                   forbid_source_overlap=False, texture_mode=None):
    mover = puzzle_parameters["mover"]
    carrier = puzzle_parameters["carrier"]
    texture = puzzle_parameters["texture"]
    modes = ("sparse", "medium", "dense", "cluster", "patch",
             "clean_lane", "all_swept")
    for _ in range(170):
        h, w = _sample_shape(rng, regime, len(orientations))
        rectangles = _place_rectangles(
            rng, h, w, orientations, require_priority,
            require_source_overlap
        )
        if rectangles is None:
            continue
        if forbid_source_overlap and _source_destination_overlap(
            rectangles, h, w
        ):
            continue
        source, lanes, destinations = _all_masks(rectangles, h, w)
        mode = texture_mode or str(rng.choice(modes))
        field_mode = "medium" if mode in ("clean_lane", "all_swept") else mode
        textured = _texture_field(rng, h, w, field_mode)
        textured[source] = False
        clean_mask = np.zeros((h, w), dtype=bool)
        if mode == "all_swept":
            textured[~lanes] = False
        if mode == "clean_lane" and len(rectangles) > 1:
            clean_index = int(rng.integers(0, len(rectangles)))
            clean_mask = _lane_mask_for(rectangles[clean_index], h, w)
            textured[clean_mask] = False
            possible = np.zeros((h, w), dtype=bool)
            for index, rectangle in enumerate(rectangles):
                if index != clean_index:
                    possible |= _lane_mask_for(rectangle, h, w)
            possible &= ~clean_mask & ~source
            cells = np.argwhere(possible)
            if len(cells) == 0:
                continue
            rr, cc = cells[int(rng.integers(0, len(cells)))]
            textured[rr, cc] = True
        visible_sweep = lanes & ~source & ~destinations
        if not np.any(textured & visible_sweep):
            cells = np.argwhere(visible_sweep & ~clean_mask)
            if len(cells) == 0:
                continue
            rr, cc = cells[int(rng.integers(0, len(cells)))]
            textured[rr, cc] = True
        input_grid = np.full((h, w), carrier, dtype=np.int8)
        input_grid[textured] = texture
        input_grid[source] = mover
        output_grid = _apply_rule(input_grid, mover, carrier)
        if output_grid is None or np.array_equal(input_grid, output_grid):
            continue
        if _infer_pair_roles(input_grid, output_grid) != [
            (mover, carrier, texture)
        ]:
            continue
        return input_grid, output_grid
    raise RuntimeError("could not generate a valid pair")


def _check_optional_count(value, name, low, high):
    if value is None:
        return

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} is outside its legal range")


def _resolve_counts(seed, num_train, num_test):
    _check_optional_count(
        num_train,
        "num_train",
        2,
        10,
    )
    _check_optional_count(
        num_test,
        "num_test",
        1,
        4,
    )

    seed_value = int(seed)

    if num_train is None and num_test is None:
        test_count = 1 + seed_value % 4
        lower_train = test_count + 1
        train_count = (
            lower_train
            + (seed_value // 4) % (11 - lower_train)
        )
    elif num_train is None:
        test_count = int(num_test)
        lower_train = test_count + 1
        train_count = (
            lower_train
            + seed_value % (11 - lower_train)
        )
    elif num_test is None:
        train_count = int(num_train)
        test_count = (
            1
            + seed_value % min(4, train_count - 1)
        )
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _array_key(array):
    return (
        array.shape,
        tuple(int(value) for value in array.ravel()),
    )


def _pair_key(pair):
    return (
        _array_key(pair["input"]),
        _array_key(pair["output"]),
    )


def _grid_rows(array):
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in array
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed,
        (int, np.integer),
    ):
        raise ValueError("seed must be an integer")

    train_count, test_count = _resolve_counts(
        int(seed),
        num_train,
        num_test,
    )

    rng = np.random.default_rng(int(seed))

    colors = [
        int(value)
        for value in rng.choice(
            10,
            size=3,
            replace=False,
        )
    ]

    puzzle_parameters = {
        "mover": colors[0],
        "carrier": colors[1],
        "texture": colors[2],
    }

    direction_order = [str(value) for value in rng.permutation(_DIRECTIONS)]
    first = direction_order[:3]
    if rng.random() < 0.40:
        first.append(str(rng.choice(first)))
    second = [direction_order[3]]
    if rng.random() < 0.52:
        second.append(str(rng.choice(_DIRECTIONS)))
    train_orientations = [first, second]
    for _ in range(2, train_count):
        mover_count = int(rng.choice([1, 2, 3, 4, 5],
                                     p=[0.24, 0.23, 0.22, 0.19, 0.12]))
        train_orientations.append([
            str(value) for value in rng.choice(
                _DIRECTIONS, size=mover_count, replace=True
            )
        ])
    test_orientations = []
    for _ in range(test_count):
        mover_count = int(rng.choice(
            [2, 3, 4, 5, 6, 7, 8, 9],
            p=[0.06, 0.09, 0.12, 0.17, 0.19, 0.17, 0.13, 0.07]
        ))
        test_orientations.append([
            str(value) for value in rng.choice(
                _DIRECTIONS, size=mover_count, replace=True
            )
        ])

    puzzle = {
        "train": [],
        "test": [],
    }
    used_inputs = set()

    for index, orientations in enumerate(train_orientations):
        pair_rng = np.random.default_rng(
            int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                )
            )
        )

        for _ in range(20):
            input_grid, output_grid = _generate_pair(
                pair_rng, "source_train", puzzle_parameters, orientations,
                require_priority=index == 0,
                require_source_overlap=index == 0,
                forbid_source_overlap=index == 1,
                texture_mode="medium" if index == 0 else None,
            )
            key = _array_key(input_grid)

            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["train"].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise RuntimeError(
                "could not make a unique training pair"
            )

    for orientations in test_orientations:
        pair_rng = np.random.default_rng(
            int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                )
            )
        )

        for _ in range(20):
            input_grid, output_grid = _generate_pair(
                pair_rng, "target_test", puzzle_parameters, orientations,
            )
            key = _array_key(input_grid)

            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["test"].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise RuntimeError(
                "could not make a unique test pair"
            )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    inputs_seen = set()
    pairs_seen = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

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
                assert np.all((0 <= grid) & (grid <= 9))

            assert input_grid.shape == output_grid.shape

            input_key = _array_key(input_grid)
            pair_key = _pair_key(pair)

            assert input_key not in inputs_seen
            assert pair_key not in pairs_seen

            inputs_seen.add(input_key)
            pairs_seen.add(pair_key)

            source_key = (
                _grid_rows(input_grid),
                _grid_rows(output_grid),
            )
            assert source_key not in _SOURCE_PAIR_ROWS

    train_roles = []

    for pair in puzzle["train"]:
        matches = _infer_pair_roles(
            pair["input"],
            pair["output"],
        )
        assert len(matches) == 1
        train_roles.append(matches[0])

    assert len(set(train_roles)) == 1

    roles = train_roles[0]
    mover, carrier, texture = roles
    assert len({mover, carrier, texture}) == 3

    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            input_grid, output_grid = pair["input"], pair["output"]
            assert {int(value) for value in np.unique(input_grid)} == set(roles)
            assert {int(value) for value in np.unique(output_grid)} <= set(roles)
            assert _infer_pair_roles(input_grid, output_grid) == [roles]
            rectangles = _extract_movers(input_grid, mover)
            assert rectangles is not None
            h, w = input_grid.shape
            if split == "train":
                assert 6 <= h <= 18 and 6 <= w <= 18
                assert 1 <= len(rectangles) <= 5
            else:
                assert 8 <= h <= 24 and 8 <= w <= 24
                assert 2 <= len(rectangles) <= 9
            assert np.array_equal(_apply_rule(input_grid, mover, carrier),
                                  output_grid)
            assert not np.array_equal(input_grid, output_grid)
            source, lanes, destinations = _all_masks(rectangles, h, w)
            assert np.all(np.isin(input_grid[~source], [carrier, texture]))
            assert np.array_equal(output_grid[~lanes], input_grid[~lanes])
            assert np.all(output_grid[lanes & ~destinations] == carrier)
            assert np.all(output_grid[destinations] == mover)
            assert np.any((input_grid == texture) & lanes & ~destinations)
            destination_sum = np.zeros((h, w), dtype=np.int8)
            for rectangle in rectangles:
                destination_sum += _rect_mask(_destination(rectangle, h, w),
                                              h, w)
            assert np.max(destination_sum) == 1
            details[split].append(rectangles)

    train_sides = {rectangle[4] for rectangles in details["train"]
                   for rectangle in rectangles}
    assert train_sides == set(_DIRECTIONS)
    first = details["train"][0]
    h, w = puzzle["train"][0]["input"].shape
    assert _has_destination_lane_overlap(first, h, w)
    assert _source_destination_overlap(first, h, w)
    second = details["train"][1]
    h, w = puzzle["train"][1]["input"].shape
    assert not _source_destination_overlap(second, h, w)
    assert len(first) != len(second)


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for first_pair, second_pair in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False

            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


def _puzzle_key(puzzle):
    return tuple(
        (
            split,
            tuple(
                _pair_key(pair)
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _independent_apply(grid, mover, carrier):
    h, w = grid.shape
    result = grid.copy()
    lane = np.zeros((h, w), dtype=bool)
    destinations = []
    for cells in _components(grid == mover):
        rows, cols = zip(*cells)
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1
        if len(cells) != (r1 - r0) * (c1 - c0):
            return None
        edges = (r0 == 0, r1 == h, c0 == 0, c1 == w)
        if sum(edges) != 1:
            return None
        if edges[0]:
            lane[:, c0:c1] = True
            destinations.append((h - (r1 - r0), h, c0, c1))
        elif edges[1]:
            lane[:, c0:c1] = True
            destinations.append((0, r1 - r0, c0, c1))
        elif edges[2]:
            lane[r0:r1, :] = True
            destinations.append((r0, r1, w - (c1 - c0), w))
        else:
            lane[r0:r1, :] = True
            destinations.append((r0, r1, 0, c1 - c0))
    if not destinations:
        return None
    result[lane] = carrier
    for r0, r1, c0, c1 in destinations:
        result[r0:r1, c0:c1] = mover
    return result


def _predict_from_demonstrations(train, test_input):
    possible = None
    for pair in train:
        grid = pair["input"]
        colors = [int(value) for value in np.unique(grid)]
        matches = set()
        for mover in colors:
            for carrier in colors:
                if mover == carrier:
                    continue
                prediction = _independent_apply(grid, mover, carrier)
                if prediction is not None and np.array_equal(
                    prediction, pair["output"]
                ):
                    matches.add((mover, carrier))
        possible = matches if possible is None else possible & matches
    assert possible is not None and len(possible) == 1
    mover, carrier = next(iter(possible))
    return _independent_apply(test_input, mover, carrier)


def self_test() -> None:
    automatic = [
        generate_puzzle(seed)
        for seed in range(42)
    ]

    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            prediction = _predict_from_demonstrations(
                puzzle["train"], pair["input"].copy()
            )
            assert np.array_equal(prediction, pair["output"])

    explicit = [
        generate_puzzle(101, 5, 2),
        generate_puzzle(102, 8, 3),
        generate_puzzle(103, 9, 4),
    ]

    for puzzle in explicit:
        validate_generated_puzzle(puzzle)

    fixed_a = generate_puzzle(98765, 8, 3)
    fixed_b = generate_puzzle(98765, 8, 3)
    assert _puzzles_equal(fixed_a, fixed_b)

    different_seed_keys = {
        _puzzle_key(generate_puzzle(seed, 6, 2))
        for seed in range(12)
    }
    assert len(different_seed_keys) >= 10

    boundary_calls = (
        (2, 1),
        (10, 1),
        (10, 4),
    )

    for index, (train_count, test_count) in enumerate(
        boundary_calls
    ):
        puzzle = generate_puzzle(
            200 + index,
            train_count,
            test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    partial_calls = (
        (None, 1),
        (None, 4),
        (2, None),
        (10, None),
    )

    for index, (train_count, test_count) in enumerate(
        partial_calls
    ):
        puzzle = generate_puzzle(
            300 + index,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)

    illegal_calls = (
        (1, None),
        (11, None),
        (None, 0),
        (None, 5),
        (2, 2),
        (3, 4),
        (4, 4),
    )

    for train_count, test_count in illegal_calls:
        try:
            generate_puzzle(
                400,
                train_count,
                test_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal counts did not raise ValueError"
            )

    train_counts = {
        len(puzzle["train"])
        for puzzle in automatic
    }
    test_counts = {
        len(puzzle["test"])
        for puzzle in automatic
    }

    role_assignments = set()
    dimension_profiles = set()
    layout_profiles = set()
    fingerprints = set()
    review = {split: {name: 0 for name in (
        "priority", "source_overlap", "repeated", "cross", "no_cross",
        "opposite", "opposite_shared", "cross_source", "broad",
        "long_shallow", "clean_dirty", "swept"
    )} for split in ("train", "test")}
    mover_counts = {"train": set(), "test": set()}
    thicknesses = set()
    depths = set()
    texture_densities = []
    compact_tests = 0

    for puzzle in automatic:
        first_pair = puzzle["train"][0]
        roles = _infer_pair_roles(
            first_pair["input"],
            first_pair["output"],
        )[0]

        role_assignments.add(roles)

        dimension_profiles.add(
            tuple(
                pair["input"].shape
                for split in ("train", "test")
                for pair in puzzle[split]
            )
        )

        layout_profile = []
        max_train_h = max(p["input"].shape[0] for p in puzzle["train"])
        max_train_w = max(p["input"].shape[1] for p in puzzle["train"])
        if any(p["input"].shape[0] <= max_train_h and
               p["input"].shape[1] <= max_train_w for p in puzzle["test"]):
            compact_tests += 1

        for split in ("train", "test"):
            for pair in puzzle[split]:
                rectangles = _extract_movers(
                    pair["input"],
                    roles[0],
                )
                expected = _apply_rule(
                    pair["input"],
                    roles[0],
                    roles[1],
                )

                assert rectangles is not None
                assert np.array_equal(
                    expected,
                    pair["output"],
                )
                h, w = pair["input"].shape
                source, _, _ = _all_masks(rectangles, h, w)
                sides = [r[4] for r in rectangles]
                axes = {s in _HORIZONTAL for s in sides}
                flags = {
                    "priority": _has_destination_lane_overlap(rectangles, h, w),
                    "source_overlap": _source_destination_overlap(rectangles, h, w),
                    "repeated": len(set(sides)) < len(sides),
                    "cross": len(axes) == 2,
                    "no_cross": len(axes) == 1,
                    "opposite": ("left" in sides and "right" in sides) or
                                ("top" in sides and "bottom" in sides),
                    "opposite_shared": any(
                        (x[4] == "left" and y[4] == "right" and
                         max(x[0], y[0]) < min(x[1], y[1])) or
                        (x[4] == "top" and y[4] == "bottom" and
                         max(x[2], y[2]) < min(x[3], y[3]))
                        for x in rectangles for y in rectangles
                    ),
                    "cross_source": any(
                        (x[4] in _HORIZONTAL) != (y[4] in _HORIZONTAL)
                        and np.any(_rect_mask(_destination(x, h, w), h, w)
                                   & _rect_mask(y, h, w))
                        for x in rectangles for y in rectangles if x != y
                    ),
                    "broad": any(r[1]-r[0] >= 4 and r[3]-r[2] >= 4
                                 for r in rectangles),
                    "long_shallow": any((r[1]-r[0] >= 4 and r[3]-r[2] == 1)
                                        or (r[3]-r[2] >= 4 and r[1]-r[0] == 1)
                                        for r in rectangles),
                    "clean_dirty": any(not np.any((pair["input"] == roles[2]) &
                                                    _lane_mask_for(r, h, w))
                                       for r in rectangles) and
                                   any(np.any((pair["input"] == roles[2]) &
                                              _lane_mask_for(r, h, w))
                                       for r in rectangles),
                    "swept": not np.any(pair["output"] == roles[2]),
                }
                for name, present in flags.items():
                    review[split][name] += int(present)
                mover_counts[split].add(len(rectangles))
                texture_densities.append(float(np.mean(
                    pair["input"][~source] == roles[2]
                )))
                for rectangle in rectangles:
                    if rectangle[4] in _HORIZONTAL:
                        thicknesses.add(rectangle[1]-rectangle[0])
                        depths.add(rectangle[3]-rectangle[2])
                    else:
                        thicknesses.add(rectangle[3]-rectangle[2])
                        depths.add(rectangle[1]-rectangle[0])

                layout_profile.append(
                    tuple(
                        sorted(
                            (
                                rectangle[4],
                                rectangle[1]
                                - rectangle[0],
                                rectangle[3]
                                - rectangle[2],
                            )
                            for rectangle in rectangles
                        )
                    )
                )

        layout_profiles.add(tuple(layout_profile))
        fingerprints.add(_puzzle_key(puzzle))

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(role_assignments) >= 15
    assert len(dimension_profiles) >= 30
    assert len(layout_profiles) >= 30
    assert len(fingerprints) >= 40
    assert mover_counts["train"] == {1, 2, 3, 4, 5}
    assert mover_counts["test"] >= {2, 3, 4, 5, 6, 7, 8}
    assert thicknesses >= {1, 2, 3, 4, 5, 6}
    assert depths >= {1, 2, 3, 4, 5, 6}
    assert compact_tests >= 8
    assert min(texture_densities) < 0.08
    assert max(texture_densities) > 0.55
    for split in ("train", "test"):
        for name in ("priority", "source_overlap", "repeated", "cross",
                     "opposite", "opposite_shared", "cross_source", "broad",
                     "long_shallow", "clean_dirty", "swept"):
            assert review[split][name] >= (4 if name in ("broad", "swept",
                                                     "cross_source")
                                           else 8)
        assert review[split]["no_cross"] >= 4
