import numpy as np


_SOURCE_PAIR_CODES = (
    (
        "00000/33000/30000/33033/03330/00000/00000/00000/00000/"
        "00000/00000/00000/00000/00000/00000/22222/00000",
        "00000/00000/00000/00000/00000/00000/00000/00000/00000/"
        "00000/88888/33000/30000/33033/03330/22222/00000",
    ),
    (
        "0300000000200000/0333000000200000/3300000000200000/"
        "0333000000200000",
        "0000080300200000/0000080333200000/0000083300200000/"
        "0000080333200000",
    ),
    (
        "00000/00000/00000/22222/00000/00000/00000/00000/00000/"
        "00000/00000/33330/30030/33030/00000/00000/00000",
        "00000/00000/00000/22222/33330/30030/33030/88888/00000/"
        "00000/00000/00000/00000/00000/00000/00000/00000",
    ),
    (
        "000020000000330000/000020000003030000/"
        "000020000003330000/000020000003000000",
        "000020338000000000/000023038000000000/"
        "000023338000000000/000023008000000000",
    ),
)


def _grid_code(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _matches_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    candidate = (_grid_code(input_grid), _grid_code(output_grid))
    return candidate in _SOURCE_PAIR_CODES


def _clean_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _clean_count("num_train", num_train, 2, 10)
    test_count = _clean_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        # This deterministic schedule guarantees count diversity for
        # consecutive seeds while always satisfying train > test.
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % span
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    boundary = int(rng.integers(10))
    stable = bool(rng.random() < 0.27)
    palette = tuple(int(c) for c in rng.choice(
        [c for c in range(10) if c != boundary], size=3, replace=False
    )) if stable else None
    return {"boundary": boundary, "stable_palette": palette}


def _is_four_connected(mask: np.ndarray) -> bool:
    occupied = np.argwhere(mask)
    if len(occupied) == 0:
        return False
    start = tuple(int(v) for v in occupied[0])
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            point = (row + dr, col + dc)
            if (0 <= point[0] < height and 0 <= point[1] < width
                    and mask[point] and point not in seen):
                seen.add(point)
                stack.append(point)
    return len(seen) == int(mask.sum())


def _make_shape_mask(rng: np.random.Generator, height: int,
                     width: int, sparse: bool, full_span: bool,
                     orientation: str) -> np.ndarray:
    # Connect four independently chosen edge contacts by short, randomly
    # ordered Manhattan paths; then grow branches from the connected mass.
    for _ in range(64):
        mask = np.zeros((height, width), dtype=bool)
        anchors = [(int(rng.integers(height)), 0),
                   (int(rng.integers(height)), width - 1),
                   (0, int(rng.integers(width))),
                   (height - 1, int(rng.integers(width)))]
        mask[anchors[0]] = True
        for target in rng.permutation(anchors[1:]):
            points = np.argwhere(mask)
            distances = np.abs(points - target).sum(axis=1)
            nearest = points[np.flatnonzero(distances == distances.min())]
            current = [int(v) for v in nearest[int(rng.integers(len(nearest)))]]
            while tuple(current) != tuple(target):
                choices = []
                for axis in (0, 1):
                    if current[axis] != target[axis]:
                        choices.append(axis)
                axis = choices[int(rng.integers(len(choices)))]
                current[axis] += 1 if target[axis] > current[axis] else -1
                mask[tuple(current)] = True
        target_fill = float(rng.uniform(0.28, 0.48) if sparse
                            else rng.uniform(0.40, 0.72))
        desired = min(height * width - 1,
                      max(int(mask.sum()), int(round(target_fill * height * width))))
        while int(mask.sum()) < desired:
            neighbors = set()
            for row, col in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(row + dr), int(col + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        neighbors.add((nr, nc))
            if not neighbors:
                break
            options = sorted(neighbors)
            mask[options[int(rng.integers(len(options)))]] = True
        if np.all(mask):
            continue
        if sparse and (np.any(np.all(mask, axis=0)) or
                       np.any(np.all(mask, axis=1))):
            continue
        if full_span and ((orientation == "horizontal" and
                           np.any(np.all(mask, axis=1))) or
                          (orientation == "vertical" and
                           np.any(np.all(mask, axis=0)))):
            continue
        return mask
    # A diagonal staircase spans the bounding box and stays connected.
    # This fallback matters mainly for small, sparse, full-span masks.
    mask = _staircase_mask(height, width)
    assert _is_four_connected(mask)
    assert not np.all(mask)
    assert not sparse or (not np.any(np.all(mask, axis=0)) and
                          not np.any(np.all(mask, axis=1)))
    return mask


def _staircase_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cols = np.rint(np.linspace(0, width - 1, height)).astype(int)
    for row in range(height):
        mask[row, cols[row]] = True
        if row:
            lo, hi = sorted((int(cols[row - 1]), int(cols[row])))
            mask[row, lo:hi + 1] = True
    return mask


def _analyze_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]]
    background = int(values[order[-1]])
    candidates = []
    height, width = grid.shape
    for value in values:
        color = int(value)
        if color == background:
            continue
        if int(np.count_nonzero(grid == color)) == width:
            rows = np.flatnonzero(np.all(grid == color, axis=1))
            if len(rows) == 1:
                candidates.append(("horizontal", int(rows[0]), color))
        if int(np.count_nonzero(grid == color)) == height:
            cols = np.flatnonzero(np.all(grid == color, axis=0))
            if len(cols) == 1:
                candidates.append(("vertical", int(cols[0]), color))
    assert len(candidates) == 1
    orientation, barrier_index, barrier = candidates[0]
    shape = ({int(v) for v in values} - {background, barrier}).pop()
    coordinates = np.argwhere(grid == shape)
    assert len(coordinates) >= 3
    r0, c0 = coordinates.min(axis=0)
    r1, c1 = coordinates.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    mask = grid[r0:r1 + 1, c0:c1 + 1] == shape
    assert mask.shape[0] >= 2 and mask.shape[1] >= 2
    assert _is_four_connected(mask)
    if orientation == "horizontal":
        if r1 < barrier_index:
            side, gap = "above", barrier_index - r1 - 1
        else:
            assert r0 > barrier_index
            side, gap = "below", r0 - barrier_index - 1
    else:
        if c1 < barrier_index:
            side, gap = "left", barrier_index - c1 - 1
        else:
            assert c0 > barrier_index
            side, gap = "right", c0 - barrier_index - 1
    assert gap >= 1
    return {"background": background, "object": shape, "barrier": barrier,
            "orientation": orientation, "barrier_index": barrier_index,
            "side": side, "gap": gap, "bbox": (r0, r1, c0, c1),
            "coordinates": coordinates, "mask": mask}


def _transform_input(input_grid: np.ndarray, boundary_color: int) -> np.ndarray:
    info = _analyze_input(input_grid)
    output = input_grid.copy()
    output[input_grid == info["object"]] = info["background"]
    r0, r1, c0, c1 = info["bbox"]
    b = info["barrier_index"]
    dr = dc = 0
    if info["side"] == "above":
        dr = b - 1 - r1
        line = r0 + dr - 1
    elif info["side"] == "below":
        dr = b + 1 - r0
        line = r1 + dr + 1
    elif info["side"] == "left":
        dc = b - 1 - c1
        line = c0 + dc - 1
    else:
        dc = b + 1 - c0
        line = c1 + dc + 1
    coordinates = info["coordinates"]
    output[coordinates[:, 0] + dr, coordinates[:, 1] + dc] = info["object"]
    if info["orientation"] == "horizontal":
        assert 0 <= line < output.shape[0]
        output[line, :] = boundary_color
    else:
        assert 0 <= line < output.shape[1]
        output[:, line] = boundary_color
    return output


_KINDS = (("horizontal", "above"), ("horizontal", "below"),
          ("vertical", "left"), ("vertical", "right"))


def _choose_length(rng: np.random.Generator, short: tuple[int, int],
                   medium: tuple[int, int], long: tuple[int, int]) -> int:
    ranges = (short, medium, long)
    low, high = ranges[int(rng.choice(3, p=[0.29, 0.37, 0.34]))]
    return int(rng.integers(low, high + 1))


def _sample_palette(rng: np.random.Generator, parameters: dict,
                    used: set) -> tuple[int, int, int]:
    if parameters["stable_palette"] is not None:
        return parameters["stable_palette"]
    colors = [color for color in range(10) if color != parameters["boundary"]]
    for _ in range(12):
        palette = tuple(int(v) for v in rng.choice(colors, 3, replace=False))
        if palette not in used:
            return palette
    return palette


def generate_pair(rng: np.random.Generator, regime: str,
                  puzzle_parameters: dict, layout_kind: int,
                  palette: tuple[int, int, int],
                  canvas_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    orientation, side = _KINDS[layout_kind]
    if canvas_shape is not None:
        perpendicular, parallel = (canvas_shape if orientation == "horizontal"
                                   else canvas_shape[::-1])
    elif rng.random() < 0.14:
        perpendicular = parallel = int(rng.integers(8, 25))
    else:
        perpendicular = _choose_length(rng, (7, 10), (11, 18), (19, 30))
        parallel = _choose_length(rng, (4, 6), (7, 15), (16, 30))
    full_span = bool(parallel <= 15 and rng.random() < 0.33)
    shape_parallel = (parallel if full_span else
                      int(rng.integers(3, min(10, parallel))))
    shape_perpendicular = int(rng.integers(3, min(8, perpendicular - 2)))
    shape_height, shape_width = ((shape_perpendicular, shape_parallel)
                                  if orientation == "horizontal" else
                                  (shape_parallel, shape_perpendicular))
    sparse = bool(rng.random() < 0.48)
    mask = _make_shape_mask(rng, shape_height, shape_width, sparse,
                            full_span, orientation)
    # In a tight canvas a dense, full-span object could tie the
    # background's color frequency and obscure the three input roles.
    largest_object = min((perpendicular * parallel - parallel - 1) // 2,
                         perpendicular * parallel - 2 * parallel - 1)
    if int(mask.sum()) > largest_object:
        for _ in range(12):
            mask = _make_shape_mask(rng, shape_height, shape_width, True,
                                    full_span, orientation)
            if int(mask.sum()) <= largest_object:
                break
        else:
            mask = _staircase_mask(shape_height, shape_width)
            assert int(mask.sum()) <= largest_object
    if rng.random() < 0.12:
        occupied_side = shape_perpendicular + 1
    else:
        occupied_side = int(rng.integers(shape_perpendicular + 1,
                                         perpendicular - 1))
    gap_max = occupied_side - shape_perpendicular
    gap = (gap_max if rng.random() < 0.26 else
           int(rng.integers(1, gap_max + 1)))
    if side in ("above", "left"):
        barrier_index = occupied_side
        start_perp = barrier_index - gap - shape_perpendicular
    else:
        barrier_index = perpendicular - occupied_side - 1
        start_perp = barrier_index + 1 + gap
    if full_span:
        start_parallel = 0
    elif rng.random() < 0.42:
        start_parallel = int(rng.choice((0, parallel - shape_parallel)))
    else:
        start_parallel = int(rng.integers(parallel - shape_parallel + 1))
    shape_start = ((start_perp, start_parallel) if orientation == "horizontal"
                   else (start_parallel, start_perp))
    shape_dims = (shape_height, shape_width)
    shape_grid_dims = ((perpendicular, parallel) if orientation == "horizontal"
                       else (parallel, perpendicular))
    background, shape_color, barrier = palette
    grid = np.full(shape_grid_dims, background, dtype=np.int8)
    if orientation == "horizontal":
        grid[barrier_index, :] = barrier
    else:
        grid[:, barrier_index] = barrier
    r0, c0 = shape_start
    view = grid[r0:r0 + shape_dims[0], c0:c0 + shape_dims[1]]
    view[mask] = shape_color
    # The dominant background gives a color-blind observer an unambiguous
    # way to identify the roles even for dense shapes and small canvases.
    assert np.count_nonzero(grid == background) > max(
        np.count_nonzero(grid == shape_color),
        np.count_nonzero(grid == barrier))
    return grid, _transform_input(grid, puzzle_parameters["boundary"])


def _array_signature(grid: np.ndarray) -> tuple:
    return grid.shape, np.ascontiguousarray(grid).tobytes()


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_array_signature(p["input"]),
                        _array_signature(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def _feature(pair: dict) -> dict:
    info = _analyze_input(pair["input"])
    inp, out = pair["input"], pair["output"]
    extra = set(map(int, np.unique(out))) - set(map(int, np.unique(inp)))
    assert len(extra) == 1
    info["boundary"] = extra.pop()
    info["shape"] = inp.shape
    info["full_span"] = (info["mask"].shape[1] == inp.shape[1]
                         if info["orientation"] == "horizontal" else
                         info["mask"].shape[0] == inp.shape[0])
    info["no_full_row_col"] = (not np.any(np.all(info["mask"], axis=0))
                               and not np.any(np.all(info["mask"], axis=1)))
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    boundaries = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key = _array_signature(inp)
            pair_key = (input_key, _array_signature(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _matches_source_pair(inp, out)
            info = _feature(pair)
            assert len(np.unique(inp)) == 3 and len(np.unique(out)) == 4
            assert info["boundary"] not in inp
            assert len({info["background"], info["object"],
                        info["barrier"], info["boundary"]}) == 4
            assert np.array_equal(out, _transform_input(inp, info["boundary"]))
            assert info["gap"] >= 1 and _is_four_connected(info["mask"])
            boundaries.add(info["boundary"])
            features[split].append(info)
    assert len(boundaries) == 1
    train_kinds = {(f["orientation"], f["side"]) for f in features["train"]}
    assert {f["orientation"] for f in features["train"]} == {
        "horizontal", "vertical"}
    assert any(f["side"] in ("above", "left") for f in features["train"])
    assert any(f["side"] in ("below", "right") for f in features["train"])
    # Both shifts are legal: some tests use a kind absent from training,
    # while other tests share a demonstrated orientation and side.
    for f in features["test"]:
        assert (f["orientation"], f["side"]) in _KINDS
    # All test-dependent hidden information is the output-only line color.
    assert all(f["boundary"] == features["train"][0]["boundary"]
               for f in features["train"] + features["test"])


def _plan_kinds(rng: np.random.Generator, train_count: int,
                test_count: int) -> tuple[list[int], list[int]]:
    focus = int(rng.integers(4))
    novel = bool(rng.random() < 0.55)
    if novel:
        base = [1, 2] if focus in (0, 3) else [0, 3]
        allowed = [k for k in range(4) if k != focus]
    else:
        companion = (3, 2, 1, 0)[focus]
        base = [focus, companion]
        allowed = list(range(4))
    train = base[:]
    while len(train) < min(train_count, len(allowed)):
        choices = [k for k in allowed if k not in train]
        train.append(choices[int(rng.integers(len(choices)))])
    while len(train) < train_count:
        train.append(allowed[int(rng.integers(len(allowed)))])
    rng.shuffle(train)
    test = [focus]
    while len(test) < test_count:
        test.append(int(rng.integers(4)))
    rng.shuffle(test)
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    train_kinds, test_kinds = _plan_kinds(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used_inputs, used_palettes = set(), set()
    for pair_index, kind in enumerate(train_kinds + test_kinds):
        split = "train" if pair_index < train_count else "test"
        stream = np.random.SeedSequence([seed, pair_index, train_count, test_count])
        pair_rng = np.random.default_rng(stream)
        canvas_shape = None
        if split == "test" and pair_rng.random() < 0.34:
            candidates = [p["input"].shape for p in puzzle["train"]
                          if _analyze_input(p["input"])["orientation"] ==
                          _KINDS[kind][0]]
            if candidates:
                canvas_shape = candidates[int(pair_rng.integers(len(candidates)))]
        for _ in range(30):
            palette = _sample_palette(pair_rng, parameters, used_palettes)
            inp, out = generate_pair(pair_rng, "source_train" if split == "train"
                                     else "target_test", parameters, kind, palette,
                                     canvas_shape)
            if _array_signature(inp) not in used_inputs and not _matches_source_pair(inp, out):
                break
        else:
            raise RuntimeError("could not construct a unique pair")
        used_inputs.add(_array_signature(inp))
        used_palettes.add(palette)
        puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    dimensions, silhouettes, layouts, palettes = set(), set(), set(), set()
    kinds, train_test_regimes = set(), set()
    full_spans = sparse_shapes = edge_lines = tight_cases = 0
    squares = horizontal_wide = vertical_tall = source_like = 0
    sparse_full_spans = dimension_overlaps = box_overlaps = 0
    varied_boxes = varied_gaps = varied_aspects = 0
    variable_palettes = stable_palettes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_palettes = set()
        train_kinds = set()
        test_kinds = set()
        local_features = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _feature(pair)
                local_features[split].append(f)
                mask = f["mask"]
                role_palette = (f["background"], f["object"],
                                f["barrier"])
                local_palettes.add(role_palette)
                palettes.add(role_palette + (f["boundary"],))
                kind = (f["orientation"], f["side"])
                kinds.add(kind)
                (train_kinds if split == "train" else test_kinds).add(kind)
                dimensions.add(pair["input"].shape)
                silhouettes.add((mask.shape, np.packbits(mask).tobytes()))
                layouts.add((pair["input"].shape, kind,
                             f["barrier_index"], f["gap"], f["bbox"]))
                full_spans += f["full_span"]
                sparse_shapes += f["no_full_row_col"]
                sparse_full_spans += f["full_span"] and f["no_full_row_col"]
                squares += pair["input"].shape[0] == pair["input"].shape[1]
                horizontal_wide += (f["orientation"] == "horizontal" and
                                    pair["input"].shape[1] > pair["input"].shape[0])
                vertical_tall += (f["orientation"] == "vertical" and
                                  pair["input"].shape[0] > pair["input"].shape[1])
                source_like += ((f["orientation"] == "horizontal" and
                                 pair["input"].shape[1] <= 5 and
                                 pair["input"].shape[0] >= 12) or
                                (f["orientation"] == "vertical" and
                                 pair["input"].shape[0] <= 5 and
                                 pair["input"].shape[1] >= 12))
                axis_length = (pair["input"].shape[0] if f["orientation"] == "horizontal"
                               else pair["input"].shape[1])
                size = (mask.shape[0] if f["orientation"] == "horizontal"
                        else mask.shape[1])
                line = (f["barrier_index"] - size - 1
                        if f["side"] in ("above", "left") else
                        f["barrier_index"] + size + 1)
                edge_lines += line in (0, axis_length - 1)
                tight_cases += f["gap"] == 1
                # Reconstruct with an independently arranged coordinate rule.
                boundary = f["boundary"]
                expected = np.full_like(pair["input"], f["background"])
                if f["orientation"] == "horizontal":
                    expected[f["barrier_index"], :] = f["barrier"]
                    near = (f["barrier_index"] - size if f["side"] == "above"
                            else f["barrier_index"] + 1)
                    start = (near, f["bbox"][2])
                    line = near - 1 if f["side"] == "above" else near + size
                    expected[line, :] = boundary
                else:
                    expected[:, f["barrier_index"]] = f["barrier"]
                    near = (f["barrier_index"] - size if f["side"] == "left"
                            else f["barrier_index"] + 1)
                    start = (f["bbox"][0], near)
                    line = near - 1 if f["side"] == "left" else near + size
                    expected[:, line] = boundary
                r, c = start
                target = expected[r:r + mask.shape[0], c:c + mask.shape[1]]
                target[mask] = f["object"]
                assert np.array_equal(expected, pair["output"])
        variable_palettes += len(local_palettes) > 1
        stable_palettes += len(local_palettes) == 1
        all_local = local_features["train"] + local_features["test"]
        varied_boxes += len({f["mask"].shape for f in all_local}) > 1
        varied_gaps += len({f["gap"] for f in all_local}) > 1
        varied_aspects += len({"square" if f["shape"][0] == f["shape"][1]
                               else "tall" if f["shape"][0] > f["shape"][1]
                               else "wide" for f in all_local}) > 1
        dimension_overlaps += bool({f["shape"] for f in local_features["train"]} &
                                   {f["shape"] for f in local_features["test"]})
        def parallel_box(f):
            return (f["mask"].shape[1] if f["orientation"] == "horizontal"
                    else f["mask"].shape[0])
        box_overlaps += bool({parallel_box(f) for f in local_features["train"]} &
                             {parallel_box(f) for f in local_features["test"]})
        train_test_regimes.add((bool(test_kinds - train_kinds),
                                bool(test_kinds & train_kinds)))
    assert len(dimensions) >= 70 and len(silhouettes) >= 100
    assert len(layouts) >= 120 and len(palettes) >= 70
    assert kinds == set(_KINDS)
    assert full_spans >= 15 and sparse_shapes >= 30
    assert sparse_full_spans >= 10
    assert edge_lines >= 8 and tight_cases >= 20
    assert squares >= 20 and horizontal_wide >= 10 and vertical_tall >= 10
    assert source_like >= 10
    assert varied_boxes == varied_gaps == varied_aspects == 42
    assert dimension_overlaps >= 10 and box_overlaps >= 15
    assert variable_palettes >= 20 and stable_palettes >= 3
    assert any(novel for novel, shared in train_test_regimes)
    assert any(shared for novel, shared in train_test_regimes)
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(2000 + 11 * train + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
        validate_generated_puzzle(p)
    for i, (train, test) in enumerate(((None, 1), (None, 4),
                                       (2, None), (10, None))):
        p = generate_puzzle(3000 + i, train, test)
        assert (train is None or len(p["train"]) == train)
        assert (test is None or len(p["test"]) == test)
    assert _puzzle_signature(generate_puzzle(991, 5, 2)) == _puzzle_signature(
        generate_puzzle(991, 5, 2))
    assert _puzzle_signature(generate_puzzle(991, 5, 2)) != _puzzle_signature(
        generate_puzzle(992, 5, 2))
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
