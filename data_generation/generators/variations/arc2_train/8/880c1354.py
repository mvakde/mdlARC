import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

_SOURCE_ROWS = (
    (
        (
            "66744722",
            "67444472",
            "74444447",
            "44444447",
            "74447773",
            "17447333",
            "11747333",
            "11177333",
        ),
        (
            "11744766",
            "17444476",
            "74444447",
            "44444447",
            "74447772",
            "37447222",
            "33747222",
            "33377222",
        ),
    ),
    (
        (
            "11744766",
            "17444766",
            "74444766",
            "97444766",
            "99744766",
            "99974766",
            "99997766",
            "99999766",
        ),
        (
            "99744711",
            "97444711",
            "74444711",
            "67444711",
            "66744711",
            "66674711",
            "66667711",
            "66666711",
        ),
    ),
    (
        (
            "66666666",
            "66666666",
            "77777776",
            "74444447",
            "74444777",
            "74777555",
            "77555555",
            "75555555",
        ),
        (
            "55555555",
            "55555555",
            "77777775",
            "74444447",
            "74444777",
            "74777666",
            "77666666",
            "76666666",
        ),
    ),
    (
        (
            "66777000",
            "66744700",
            "67444700",
            "74444700",
            "44444700",
            "77774700",
            "33337000",
            "33337000",
        ),
        (
            "33777666",
            "33744766",
            "37444766",
            "74444766",
            "44444766",
            "77774766",
            "00007666",
            "00007666",
        ),
    ),
)


def _grid_signature(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.ravel())


def _source_signatures():
    inputs = set()
    pairs = set()

    for input_rows, output_rows in _SOURCE_ROWS:
        inp = np.array(
            [[int(ch) for ch in row] for row in input_rows],
            dtype=np.int8,
        )
        out = np.array(
            [[int(ch) for ch in row] for row in output_rows],
            dtype=np.int8,
        )
        input_signature = _grid_signature(inp)
        output_signature = _grid_signature(out)
        inputs.add(input_signature)
        pairs.add((input_signature, output_signature))

    return inputs, pairs


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = _N8 if diagonal else _N4
    result = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            result.append(np.asarray(cells, dtype=np.int16))

    return result


def _touches_boundary(mask):
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _has_divider_block(mask):
    return bool(
        np.any(
            mask[:-1, :-1]
            & mask[1:, :-1]
            & mask[:-1, 1:]
            & mask[1:, 1:]
        )
    )


def _has_solid_block(mask):
    return _has_divider_block(mask)


def _is_rectangle(mask):
    rows, cols = np.where(mask)
    if rows.size == 0:
        return False

    box_area = (
        int(rows.max()) - int(rows.min()) + 1
    ) * (
        int(cols.max()) - int(cols.min()) + 1
    )
    return int(mask.sum()) == box_area


def _clockwise_components(components, anchor_mask):
    anchor_rows, anchor_cols = np.where(anchor_mask)
    center_r = float(anchor_rows.mean())
    center_c = float(anchor_cols.mean())
    angular = []

    for component in components:
        rows = component[:, 0].astype(float)
        cols = component[:, 1].astype(float)

        north = center_r - rows
        east = cols - center_c
        lengths = np.hypot(north, east)

        assert np.all(lengths > 1e-9)

        mean_east = float(np.mean(east / lengths))
        mean_north = float(np.mean(north / lengths))
        resultant = float(np.hypot(mean_east, mean_north))

        assert resultant > 0.08

        angle = float(np.arctan2(mean_north, mean_east))
        angular.append((angle, component))

    angles = np.sort(
        np.asarray([item[0] for item in angular], dtype=float)
    )
    gaps = np.diff(
        np.concatenate((angles, angles[:1] + 2.0 * np.pi))
    )

    assert float(gaps.min()) > 0.18

    angular.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in angular]


def _analyze_pair(inp, out):
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.ndim == 2
    assert out.ndim == 2
    assert inp.shape == out.shape
    assert np.issubdtype(inp.dtype, np.integer)
    assert np.issubdtype(out.dtype, np.integer)

    input_palette = set(int(x) for x in np.unique(inp))
    output_palette = set(int(x) for x in np.unique(out))
    assert input_palette == output_palette

    h, w = inp.shape
    palette = sorted(input_palette)

    stable = [
        color
        for color in palette
        if np.array_equal(inp == color, out == color)
    ]
    assert len(stable) == 2

    solid_flags = [
        _has_solid_block(inp == color)
        for color in stable
    ]
    assert solid_flags.count(True) == 1

    anchor_color = stable[solid_flags.index(True)]
    divider_color = stable[solid_flags.index(False)]

    anchor_mask = inp == anchor_color
    divider_mask = inp == divider_color

    assert np.array_equal(anchor_mask, out == anchor_color)
    assert np.array_equal(divider_mask, out == divider_color)
    assert int(anchor_mask.sum()) >= 3
    assert int(divider_mask.sum()) >= 2
    assert len(_components(anchor_mask)) == 1
    assert len(
        _components(anchor_mask | divider_mask, diagonal=True)
    ) == 1
    assert not _has_divider_block(divider_mask)

    anchor_rows, anchor_cols = np.where(anchor_mask)
    assert (
        0.25 * (h - 1)
        <= float(anchor_rows.mean())
        <= 0.75 * (h - 1)
    )
    assert (
        0.25 * (w - 1)
        <= float(anchor_cols.mean())
        <= 0.75 * (w - 1)
    )

    sectors = _components(~(anchor_mask | divider_mask))
    sector_count = len(sectors)

    assert 2 <= sector_count <= 6
    assert min(len(component) for component in sectors) >= 3

    ordered = _clockwise_components(sectors, anchor_mask)
    input_colors = []
    output_colors = []

    for component in ordered:
        rows = component[:, 0]
        cols = component[:, 1]
        in_values = np.unique(inp[rows, cols])
        out_values = np.unique(out[rows, cols])

        assert in_values.size == 1
        assert out_values.size == 1

        input_colors.append(int(in_values[0]))
        output_colors.append(int(out_values[0]))

    assert len(set(input_colors)) == sector_count
    assert len(set(output_colors)) == sector_count
    assert set(input_colors) == set(output_colors)
    assert anchor_color not in input_colors
    assert divider_color not in input_colors
    assert len(palette) == sector_count + 2

    expected = [
        input_colors[(index - 1) % sector_count]
        for index in range(sector_count)
    ]
    assert output_colors == expected
    assert not np.array_equal(inp, out)

    return {
        "anchor_color": anchor_color,
        "divider_color": divider_color,
        "anchor_mask": anchor_mask,
        "divider_mask": divider_mask,
        "sector_count": sector_count,
        "sector_areas": tuple(
            len(component) for component in ordered
        ),
        "input_payload_colors": tuple(input_colors),
    }


def _clockwise_predictions(inp):
    """Enumerate outputs from roles and sectors visible in an input alone."""
    h, w = inp.shape
    palette = [int(x) for x in np.unique(inp)]
    predictions = []
    for anchor_color in palette:
        anchor = inp == anchor_color
        if not _has_solid_block(anchor) or len(_components(anchor)) != 1:
            continue
        rr, cc = np.where(anchor)
        if not (0.25 * (h - 1) <= rr.mean() <= 0.75 * (h - 1)):
            continue
        if not (0.25 * (w - 1) <= cc.mean() <= 0.75 * (w - 1)):
            continue
        for divider_color in palette:
            if divider_color == anchor_color:
                continue
            divider = inp == divider_color
            if _has_divider_block(divider) or not _touches_boundary(divider):
                continue
            if int(divider.sum()) < 2:
                continue
            if len(_components(anchor | divider, diagonal=True)) != 1:
                continue
            sectors = _components(~(anchor | divider))
            if not (2 <= len(sectors) <= 6 and len(sectors) == len(palette) - 2):
                continue
            if min(len(component) for component in sectors) < 3:
                continue
            try:
                ordered = _clockwise_components(sectors, anchor)
            except AssertionError:
                continue
            colors = []
            for cells in ordered:
                values = np.unique(inp[cells[:, 0], cells[:, 1]])
                if len(values) != 1:
                    break
                colors.append(int(values[0]))
            if len(colors) != len(sectors) or len(set(colors)) != len(colors):
                continue
            out = inp.copy()
            for index, cells in enumerate(ordered):
                out[cells[:, 0], cells[:, 1]] = colors[(index - 1) % len(colors)]
            predictions.append(out)
    return predictions


def _bresenham(r0, c0, r1, c1):
    points = []
    dx = abs(c1 - c0)
    sx = 1 if c0 < c1 else -1
    dy = -abs(r1 - r0)
    sy = 1 if r0 < r1 else -1
    error = dx + dy

    while True:
        points.append((r0, c0))

        if r0 == r1 and c0 == c1:
            return points

        twice = 2 * error

        if twice >= dy:
            error += dy
            c0 += sx

        if twice <= dx:
            error += dx
            r0 += sy


def _ray_endpoint(h, w, r, c, angle):
    dr = -float(np.sin(angle))
    dc = float(np.cos(angle))
    times = []

    if dr > 1e-9:
        times.append((h - 1 - r) / dr)
    elif dr < -1e-9:
        times.append((0 - r) / dr)

    if dc > 1e-9:
        times.append((w - 1 - c) / dc)
    elif dc < -1e-9:
        times.append((0 - c) / dc)

    time = min(value for value in times if value >= 0.0)

    end_r = int(
        np.clip(round(r + time * dr), 0, h - 1)
    )
    end_c = int(
        np.clip(round(c + time * dc), 0, w - 1)
    )
    return end_r, end_c


def _make_anchor(rng, h, w):
    """Grow a solid central mass with optional broad arms and an edge reach."""
    ah = int(rng.integers(2, 4))
    aw = int(rng.integers(2, 4))
    top = int(np.clip((h - ah) // 2 + rng.integers(-1, 2), 1, h - ah - 1))
    left = int(np.clip((w - aw) // 2 + rng.integers(-1, 2), 1, w - aw - 1))
    anchor = np.zeros((h, w), dtype=bool)
    anchor[top:top + ah, left:left + aw] = True

    size_fraction = float(rng.uniform(0.07, 0.32))
    target_area = max(4, int(round(size_fraction * h * w)))
    edge_reach = rng.random() < 0.42
    if edge_reach:
        side = int(rng.integers(0, 4))
        sr = int(rng.integers(top, top + ah))
        sc = int(rng.integers(left, left + aw))
        if side == 0:
            endpoint = (0, int(np.clip(sc + rng.integers(-2, 3), 1, w - 2)))
        elif side == 1:
            endpoint = (h - 1, int(np.clip(sc + rng.integers(-2, 3), 1, w - 2)))
        elif side == 2:
            endpoint = (int(np.clip(sr + rng.integers(-2, 3), 1, h - 2)), 0)
        else:
            endpoint = (int(np.clip(sr + rng.integers(-2, 3), 1, h - 2)), w - 1)
        for r, c in _bresenham(sr, sc, *endpoint):
            anchor[r, c] = True
            if rng.random() < 0.25:
                if side < 2:
                    anchor[r, int(np.clip(c + int(rng.choice([-1, 1])), 0, w - 1))] = True
                else:
                    anchor[int(np.clip(r + int(rng.choice([-1, 1])), 0, h - 1)), c] = True

    for _ in range(max(0, target_area - int(anchor.sum()))):
        cells = np.argwhere(anchor)
        candidates = []
        for r, c in cells:
            for dr, dc in _N4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not anchor[nr, nc]:
                    candidates.append((nr, nc))
        if not candidates:
            break
        nr, nc = candidates[int(rng.integers(len(candidates)))]
        anchor[nr, nc] = True

    return anchor


def _sample_geometry(
    rng,
    h,
    w,
    sector_count,
    regime,
):
    anchor = _make_anchor(rng, h, w)

    if len(_components(anchor)) != 1 or not _has_solid_block(anchor):
        return None
    if int(anchor.sum()) > 0.38 * h * w:
        return None

    rows, cols = np.where(anchor)
    center_r = float(rows.mean())
    center_c = float(cols.mean())
    ray_count = sector_count - int(_touches_boundary(anchor))
    spacing = 2.0 * np.pi / ray_count
    base = float(rng.uniform(0.0, 2.0 * np.pi))
    jitter_fraction = 0.28

    angles = [
        base
        + index * spacing
        + float(
            rng.uniform(
                -jitter_fraction * spacing,
                jitter_fraction * spacing,
            )
        )
        for index in range(ray_count)
    ]

    endpoints = [
        _ray_endpoint(
            h,
            w,
            center_r,
            center_c,
            angle,
        )
        for angle in angles
    ]

    if len(set(endpoints)) != ray_count:
        return None

    divider = np.zeros((h, w), dtype=bool)
    for angle, (end_r, end_c) in zip(angles, endpoints):
        dr = -np.sin(angle)
        dc = np.cos(angle)
        projections = (center_r - rows) * dr + (cols - center_c) * dc
        candidate_indices = np.argsort(projections)[-max(2, len(rows) // 3):]
        selected = int(rng.choice(candidate_indices))
        start_r, start_c = int(rows[selected]), int(cols[selected])
        points = _bresenham(start_r, start_c, end_r, end_c)

        if rng.random() < 0.52:
            fraction = float(rng.uniform(0.35, 0.65))
            mid_r = (
                start_r
                + fraction * (end_r - start_r)
            )
            mid_c = (
                start_c
                + fraction * (end_c - start_c)
            )

            vr = end_r - start_r
            vc = end_c - start_c
            length = max(
                1.0,
                float(np.hypot(vr, vc)),
            )

            shift = (
                float(
                    rng.choice(
                        np.asarray([-1, 1])
                    )
                )
                * float(rng.uniform(0.55, 2.0))
            )

            bend_r = int(
                np.clip(
                    round(mid_r - shift * vc / length),
                    1,
                    h - 2,
                )
            )
            bend_c = int(
                np.clip(
                    round(mid_c + shift * vr / length),
                    1,
                    w - 2,
                )
            )

            points = _bresenham(
                start_r,
                start_c,
                bend_r,
                bend_c,
            )
            points += _bresenham(
                bend_r,
                bend_c,
                end_r,
                end_c,
            )[1:]

        elif rng.random() < 0.25:
            bend = (start_r, end_c) if rng.random() < 0.5 else (end_r, start_c)
            points = _bresenham(start_r, start_c, *bend)
            points += _bresenham(*bend, end_r, end_c)[1:]

        for r, c in points:
            if not anchor[r, c]:
                divider[r, c] = True

    if _has_divider_block(divider):
        return None

    if len(_components(anchor)) != 1:
        return None

    if len(
        _components(anchor | divider, diagonal=True)
    ) != 1:
        return None

    sectors = _components(~(anchor | divider))

    if len(sectors) != sector_count:
        return None

    if min(len(component) for component in sectors) < 3:
        return None

    try:
        ordered = _clockwise_components(
            sectors,
            anchor,
        )
    except AssertionError:
        return None

    return anchor, divider, ordered


def _structure_signature(
    anchor_mask,
    divider_mask,
):
    return (
        tuple(anchor_mask.shape),
        tuple(bool(x) for x in anchor_mask.ravel()),
        tuple(bool(x) for x in divider_mask.ravel()),
    )


def _generate_pair(
    rng,
    regime,
    puzzle_parameters,
    sector_count,
    forbidden_inputs,
    forbidden_structures,
):
    h = puzzle_parameters["height"]
    w = puzzle_parameters["width"]
    anchor_color = puzzle_parameters["anchor_color"]
    divider_color = puzzle_parameters["divider_color"]

    payload_pool = np.asarray(
        [color for color in range(10) if color not in (anchor_color, divider_color)],
        dtype=np.int8,
    )

    for _ in range(2000):
        geometry = _sample_geometry(
            rng,
            h,
            w,
            sector_count,
            regime,
        )

        if geometry is None:
            continue

        anchor, divider, ordered = geometry
        structure = _structure_signature(
            anchor,
            divider,
        )

        if structure in forbidden_structures:
            continue

        payload_colors = rng.choice(
            payload_pool,
            size=sector_count,
            replace=False,
        ).astype(np.int8)

        rng.shuffle(payload_colors)

        inp = np.empty((h, w), dtype=np.int8)
        inp[anchor] = anchor_color
        inp[divider] = divider_color

        for component, color in zip(
            ordered,
            payload_colors,
        ):
            inp[
                component[:, 0],
                component[:, 1],
            ] = int(color)

        out = inp.copy()

        for index, component in enumerate(ordered):
            color = int(
                payload_colors[
                    (index - 1) % sector_count
                ]
            )
            out[
                component[:, 0],
                component[:, 1],
            ] = color

        input_signature = _grid_signature(inp)

        if input_signature in forbidden_inputs:
            continue

        try:
            _analyze_pair(inp, out)
        except AssertionError:
            continue

        predictions = _clockwise_predictions(inp)
        if not predictions or any(not np.array_equal(candidate, out) for candidate in predictions):
            continue

        return inp, out, structure

    raise RuntimeError(
        "Unable to construct a valid, distinct sector layout"
    )


def _validate_count(
    value,
    name,
    low,
    high,
):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return value


def _choose_counts(
    rng,
    num_train,
    num_test,
):
    train_count = _validate_count(
        num_train,
        "num_train",
        2,
        10,
    )
    test_count = _validate_count(
        num_test,
        "num_test",
        1,
        4,
    )

    if (
        train_count is not None
        and test_count is not None
    ):
        if train_count <= test_count:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return train_count, test_count

    if train_count is None and test_count is None:
        if rng.random() < 0.32:
            return 3, 1
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif train_count is None:
        train_count = int(
            rng.integers(
                max(2, test_count + 1),
                11,
            )
        )
    else:
        largest_test = min(
            4,
            train_count - 1,
        )
        test_count = int(
            rng.integers(
                1,
                largest_test + 1,
            )
        )

    if train_count <= test_count:
        raise ValueError(
            "no legal pair-count completion exists"
        )

    return train_count, test_count


def _sample_puzzle_parameters(rng):
    roles = rng.choice(np.arange(10), size=2, replace=False)
    return {
        "anchor_color": int(roles[0]),
        "divider_color": int(roles[1]),
        "height": int(rng.integers(7, 17)),
        "width": int(rng.integers(7, 17)),
        "vary_roles": bool(rng.random() < 0.62),
        "vary_size": bool(rng.random() < 0.75),
    }


def _training_sector_counts(rng, count):
    if count == 2:
        result = [3, 4 if rng.random() < 0.65 else 2]
    else:
        result = [2, 3, 4]
        result.extend(int(rng.integers(2, 5)) for _ in range(count - 3))

    rng.shuffle(result)
    return result


def _pair_parameters(rng, common, index, train_count, sector_count, prior_roles):
    if common["vary_roles"]:
        roles = tuple(int(x) for x in rng.choice(np.arange(10), size=2, replace=False))
        if index == 1 and roles == prior_roles:
            roles = (roles[1], roles[0])
    else:
        roles = (common["anchor_color"], common["divider_color"])

    if common["vary_size"]:
        h = int(rng.integers(7, 17))
        w = int(rng.integers(7, 17))
    else:
        h, w = common["height"], common["width"]
    if sector_count >= 5:
        h = max(h, 11)
        w = max(w, 11)

    return {
        "anchor_color": roles[0],
        "divider_color": roles[1],
        "height": h,
        "width": w,
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)

    train_count, test_count = _choose_counts(
        rng,
        num_train,
        num_test,
    )
    parameters = _sample_puzzle_parameters(rng)
    train_sector_counts = _training_sector_counts(
        rng,
        train_count,
    )
    test_sector_counts = [int(rng.choice(np.array([2, 3, 4, 5]), p=[0.20, 0.36, 0.34, 0.10])) for _ in range(test_count)]
    if 5 in test_sector_counts and 4 not in train_sector_counts:
        replacement = train_sector_counts.index(2) if 2 in train_sector_counts else 0
        train_sector_counts[replacement] = 4

    child_seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=train_count + test_count,
        dtype=np.uint64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    forbidden_inputs = set()
    forbidden_structures = set()
    first_roles = None

    for index in range(
        train_count + test_count
    ):
        if index < train_count:
            regime = "source_train"
            sector_count = train_sector_counts[index]
            split = "train"
        else:
            regime = "target_test"
            sector_count = test_sector_counts[index - train_count]
            split = "test"

        pair_rng = np.random.default_rng(
            int(child_seeds[index])
        )
        pair_parameters = _pair_parameters(
            pair_rng, parameters, index, train_count, sector_count, first_roles
        )
        if first_roles is None:
            first_roles = (
                pair_parameters["anchor_color"],
                pair_parameters["divider_color"],
            )

        for fallback in range(4):
            try:
                inp, out, structure = _generate_pair(
                    pair_rng,
                    regime,
                    pair_parameters,
                    sector_count,
                    forbidden_inputs,
                    forbidden_structures,
                )
                break
            except RuntimeError:
                pair_parameters = dict(pair_parameters)
                pair_parameters["height"] = max(pair_parameters["height"], 11 + fallback)
                pair_parameters["width"] = max(pair_parameters["width"], 11 + fallback)
        else:
            raise RuntimeError("Unable to construct pair after roomier fallbacks")

        puzzle[split].append(
            {
                "input": inp,
                "output": out,
            }
        )

        forbidden_inputs.add(
            _grid_signature(inp)
        )
        forbidden_structures.add(structure)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {
        "train",
        "test",
    }
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    source_inputs, source_pairs = (
        _source_signatures()
    )
    input_signatures = set()
    pair_signatures = set()
    structure_signatures = set()

    train_metadata = []
    test_metadata = []

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {
                "input",
                "output",
            }

            inp = pair["input"]
            out = pair["output"]

            for grid in (inp, out):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(
                    grid.dtype,
                    np.integer,
                )
                assert np.all(
                    (0 <= grid) & (grid <= 9)
                )

            assert inp.shape == out.shape

            input_signature = _grid_signature(inp)
            pair_signature = (
                input_signature,
                _grid_signature(out),
            )

            assert (
                input_signature
                not in input_signatures
            )
            assert (
                pair_signature
                not in pair_signatures
            )
            assert (
                input_signature
                not in source_inputs
            )
            assert pair_signature not in source_pairs

            input_signatures.add(input_signature)
            pair_signatures.add(pair_signature)

            metadata = _analyze_pair(inp, out)
            predictions = _clockwise_predictions(inp)
            assert predictions
            assert all(np.array_equal(candidate, out) for candidate in predictions)
            structure = _structure_signature(
                metadata["anchor_mask"],
                metadata["divider_mask"],
            )

            assert (
                structure
                not in structure_signatures
            )
            structure_signatures.add(structure)

            if split == "train":
                train_metadata.append(metadata)
                assert (
                    2
                    <= metadata["sector_count"]
                    <= 4
                )
            else:
                test_metadata.append(metadata)
                assert 2 <= metadata["sector_count"] <= 5

    # A three-sector training example makes clockwise
    # direction distinguishable from counterclockwise.
    assert any(
        metadata["sector_count"] == 3
        for metadata in train_metadata
    )
    assert any(
        metadata["sector_count"] >= 3
        for metadata in train_metadata
    )
    if any(metadata["sector_count"] == 5 for metadata in test_metadata):
        assert any(metadata["sector_count"] == 4 for metadata in train_metadata)

    train_structures = {
        _structure_signature(
            metadata["anchor_mask"],
            metadata["divider_mask"],
        )
        for metadata in train_metadata
    }

    for metadata in test_metadata:
        test_structure = _structure_signature(
            metadata["anchor_mask"],
            metadata["divider_mask"],
        )
        assert test_structure not in train_structures


def _puzzles_equal(first, second):
    if (
        set(first.keys()) != {"train", "test"}
        or set(second.keys())
        != {"train", "test"}
    ):
        return False

    for split in ("train", "test"):
        if (
            len(first[split])
            != len(second[split])
        ):
            return False

        for left, right in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                left["input"],
                right["input"],
            ):
                return False

            if not np.array_equal(
                left["output"],
                right["output"],
            ):
                return False

    return True


def _puzzle_signature(puzzle):
    return tuple(
        (
            split,
            _grid_signature(pair["input"]),
            _grid_signature(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(
            314159,
            **kwargs,
        )
    except ValueError:
        return

    raise AssertionError(
        f"expected ValueError for {kwargs}"
    )


def self_test() -> None:
    original = []
    for input_rows, output_rows in _SOURCE_ROWS:
        inp = np.asarray([[int(ch) for ch in row] for row in input_rows], dtype=np.int8)
        out = np.asarray([[int(ch) for ch in row] for row in output_rows], dtype=np.int8)
        metadata = _analyze_pair(inp, out)
        predictions = _clockwise_predictions(inp)
        assert predictions and all(np.array_equal(p, out) for p in predictions)
        original.append(metadata)
    assert [m["sector_count"] for m in original] == [4, 3, 2, 3]
    assert sum(_touches_boundary(m["anchor_mask"]) for m in original) == 3

    explicit = generate_puzzle(17, 6, 2)
    assert _puzzles_equal(explicit, generate_puzzle(17, 6, 2))
    validate_generated_puzzle(explicit)
    for a, b in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(700 + a + b, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for index, (a, b) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(800 + index, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    ):
        _expect_value_error(**kwargs)

    train_counts, test_counts = set(), set()
    dimensions, roles, structures, signatures = set(), set(), set(), set()
    train_zero_payload = test_without_zero_payload = zero_structure = 0
    edge_anchors = large_anchors = 0
    varied_sizes = varied_roles = combined = compact = 0
    test_sectors = set()
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        compact += (len(puzzle["train"]), len(puzzle["test"])) == (3, 1)
        local_sizes, local_roles = set(), set()
        local_edge = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                total_pairs += 1
                metadata = _analyze_pair(inp, out)
                predictions = _clockwise_predictions(inp)
                assert predictions and all(np.array_equal(p, out) for p in predictions)
                dimensions.add(inp.shape)
                local_sizes.add(inp.shape)
                role = (metadata["anchor_color"], metadata["divider_color"])
                roles.add(role)
                local_roles.add(role)
                structures.add(_structure_signature(metadata["anchor_mask"], metadata["divider_mask"]))
                edge = _touches_boundary(metadata["anchor_mask"])
                local_edge |= edge
                edge_anchors += edge
                large_anchors += int(metadata["anchor_mask"].sum()) >= 15
                zero_structure += 0 in role
                if split == "train":
                    train_zero_payload += 0 in metadata["input_payload_colors"]
                else:
                    test_sectors.add(metadata["sector_count"])
                    test_without_zero_payload += 0 not in metadata["input_payload_colors"]
        size_varies = len(local_sizes) > 1
        role_varies = len(local_roles) > 1
        varied_sizes += size_varies
        varied_roles += role_varies
        combined += size_varies and role_varies and local_edge
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 25 and len(roles) >= 20
    assert len(structures) >= int(0.85 * total_pairs)
    assert compact >= 3
    assert train_zero_payload > 0 and test_without_zero_payload > 0
    assert zero_structure > 0 and edge_anchors >= 12 and large_anchors >= 12
    assert varied_sizes >= 10 and varied_roles >= 10 and combined >= 3
    assert {2, 3, 4}.issubset(test_sectors)
