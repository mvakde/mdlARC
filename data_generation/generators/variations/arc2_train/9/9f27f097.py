import numpy as np


# Exact source inputs. Rejecting these inputs is sufficient to ensure that no
# generated input/output pair exactly duplicates a source-task pair.
_SOURCE_INPUT_ROWS = frozenset(
    {
        (
            "222220000022",
            "222220000022",
            "222220000022",
            "222220000022",
            "222220000022",
            "222222222222",
            "244414222222",
            "244144222222",
            "241411222222",
            "244141222222",
            "244444222222",
            "222222222222",
        ),
        (
            "222222222222",
            "213312222222",
            "211312222222",
            "213332222222",
            "211112222222",
            "222222222222",
            "222222222222",
            "222200002222",
            "222200002222",
            "222200002222",
            "222200002222",
            "222222222222",
        ),
        (
            "111111111111",
            "111111111111",
            "133333311111",
            "133323311111",
            "122232311111",
            "133323311111",
            "111111111111",
            "111111111111",
            "111000000111",
            "111000000111",
            "111000000111",
            "111000000111",
        ),
        (
            "888888888888",
            "811128888888",
            "811218888888",
            "812228888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "888888888888",
            "880000888888",
            "880000888888",
            "880000888888",
        ),
    }
)


def _checked_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        # The seed schedule makes all legal count regimes reachable.
        seed_value = int(seed)
        num_test = 1 + seed_value % 4
        num_train = num_test + 1 + (seed_value // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _rows_key(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _bbox(mask):
    points = np.argwhere(mask)
    assert points.size, "missing rectangle"
    lo = points.min(axis=0)
    hi = points.max(axis=0) + 1
    r0, c0 = (int(x) for x in lo)
    r1, c1 = (int(x) for x in hi)
    expected = np.zeros(mask.shape, dtype=bool)
    expected[r0:r1, c0:c1] = True
    assert np.array_equal(mask, expected), "region is not a filled rectangle"
    return (r0, r1, c0, c1)


def _shape(rng, square=None):
    if square is True:
        side = int(rng.integers(3, 8))
        return (side, side)
    while True:
        height = int(rng.integers(2, 10))
        width = int(rng.integers(2, 10))
        if height == width == 2:
            continue
        if square is False and height == width:
            continue
        return (height, width)


def _motif_bits(rng, height, width, style):
    bits = np.zeros((height, width), dtype=np.int8)
    if style == 0:  # irregular texture
        density = float(rng.uniform(0.2, 0.8))
        bits[:] = rng.random((height, width)) < density
    elif style == 1:  # sparse marks
        count = int(rng.integers(1, min(7, height * width // 2) + 1))
        positions = rng.choice(height * width, size=count, replace=False)
        bits.flat[positions] = 1
    elif style == 2:  # connected, wandering glyph
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        for _ in range(int(rng.integers(height + width, 2 * height * width + 1))):
            bits[row, col] = 1
            direction = int(rng.integers(0, 4))
            row = max(0, min(height - 1, row + (direction == 0) - (direction == 1)))
            col = max(0, min(width - 1, col + (direction == 2) - (direction == 3)))
    elif style == 3:  # bars with an offset crosspiece
        row = int(rng.integers(0, height))
        start = int(rng.integers(0, width - 1))
        stop = int(rng.integers(start + 2, width + 1))
        bits[row, start:stop] = 1
        col = int(rng.integers(0, width))
        start = int(rng.integers(0, height - 1))
        stop = int(rng.integers(start + 2, height + 1))
        bits[start:stop, col] = 1
    elif style == 4:  # blocky regions
        for _ in range(int(rng.integers(1, 4))):
            r0 = int(rng.integers(0, height))
            c0 = int(rng.integers(0, width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            bits[r0:r1, c0:c1] = 1
    elif style == 5:  # stripes with a local break
        if int(rng.integers(0, 2)):
            bits[:] = np.arange(width)[None, :] % 2
        else:
            bits[:] = np.arange(height)[:, None] % 2
        for _ in range(int(rng.integers(1, 4))):
            bits[int(rng.integers(height)), int(rng.integers(width))] ^= 1
    else:  # repeated rows
        templates = rng.integers(0, 2, size=(int(rng.integers(1, min(4, height + 1))), width), dtype=np.int8)
        for row in range(height):
            bits[row] = templates[int(rng.integers(0, len(templates)))]
    return bits


def _make_motif(rng, height, width, colors):
    style = int(rng.integers(0, 7))
    for attempt in range(120):
        bits = _motif_bits(rng, height, width, style if attempt < 16 else int(rng.integers(0, 7)))
        if not (np.any(bits == 0) and np.any(bits == 1)):
            continue
        reflected = bits[:, ::-1]
        if np.array_equal(bits, reflected):
            continue
        if np.array_equal(reflected, bits[::-1, :]):
            continue
        if np.array_equal(reflected, bits[::-1, ::-1]):
            continue
        if height == width and np.array_equal(reflected, bits.T):
            continue
        if int(rng.integers(0, 2)):
            colors = colors[::-1]
        return np.where(bits, colors[1], colors[0]).astype(np.int8)
    raise ValueError("could not form an informative motif")


def _palette(rng, train_palettes=None):
    if train_palettes and rng.random() < 0.38:
        return train_palettes[int(rng.integers(len(train_palettes)))]
    if train_palettes and rng.random() < 0.58:
        familiar = sorted({c for palette in train_palettes for c in palette})
        if len(familiar) >= 3:
            values = rng.choice(familiar, size=3, replace=False)
            return tuple(int(x) for x in values)
    values = rng.choice(np.arange(1, 10), size=3, replace=False)
    return tuple(int(x) for x in values)


def _make_pair(rng, shape, palette, mode, shift=None):
    height, width = shape
    canvas, first, second = palette
    for _ in range(100):
        if mode == "side":
            gap = int(rng.integers(0, 8))
            distance = width + gap
            dx = distance if (shift is None and int(rng.integers(0, 2))) else -distance
            if shift is not None:
                dx = (1 if shift >= 0 else -1) * distance
            dr = int(rng.integers(-height + 1, height))
        else:
            gap = int(rng.integers(0, 10))
            dr = height + gap if mode == "above" else -(height + gap)
            if shift is None:
                if rng.random() < 0.67:
                    distance = int(rng.integers(0, width))
                else:
                    distance = int(rng.integers(width, width + 8))
                dx = distance * (1 if int(rng.integers(0, 2)) else -1)
            else:
                dx = int(shift)

        union_height = max(height, dr + height) - min(0, dr)
        union_width = max(width, dx + width) - min(0, dx)
        if union_height > 30 or union_width > 30:
            continue

        if union_height <= 12 and union_width <= 12 and rng.random() < 0.20:
            if 144 <= 3 * height * width:
                continue
            top = int(rng.integers(0, 13 - union_height))
            bottom = 12 - union_height - top
            left = int(rng.integers(0, 13 - union_width))
            right = 12 - union_width - left
        else:
            margins = [int(rng.integers(0, 8)) for _ in range(4)]
            # Zero and roomy margins both have explicit probability.
            if rng.random() < 0.25:
                margins[int(rng.integers(0, 4))] = 0
            if rng.random() < 0.20:
                margins[int(rng.integers(0, 4))] = int(rng.integers(8, 13))
            top, bottom, left, right = margins

        grid_height = top + union_height + bottom
        grid_width = left + union_width + right
        if grid_height > 30 or grid_width > 30:
            continue
        if grid_height * grid_width <= 3 * height * width:
            continue

        sr = top - min(0, dr)
        sc = left - min(0, dx)
        dest_r, dest_c = sr + dr, sc + dx
        motif = _make_motif(rng, height, width, (first, second))
        inp = np.full((grid_height, grid_width), canvas, dtype=np.int8)
        inp[sr:sr + height, sc:sc + width] = motif
        inp[dest_r:dest_r + height, dest_c:dest_c + width] = 0
        if _rows_key(inp) in _SOURCE_INPUT_ROWS:
            continue
        out = inp.copy()
        out[dest_r:dest_r + height, dest_c:dest_c + width] = motif[:, ::-1]
        return {"input": inp, "output": out}
    raise ValueError("could not place a pair")


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    largest = np.flatnonzero(counts == counts.max())
    assert len(largest) == 1
    canvas = int(largest[0])
    assert canvas != 0
    dest = _bbox(inp == 0)
    source = _bbox((inp != canvas) & (inp != 0))
    sr0, sr1, sc0, sc1 = source
    dr0, dr1, dc0, dc1 = dest
    height, width = sr1 - sr0, sc1 - sc0
    assert (height, width) == (dr1 - dr0, dc1 - dc0)
    assert sr1 <= dr0 or dr1 <= sr0 or sc1 <= dc0 or dc1 <= sc0
    motif = inp[sr0:sr1, sc0:sc1]
    motif_colors = tuple(int(x) for x in np.unique(motif))
    assert len(motif_colors) == 2
    assert 0 not in motif_colors and canvas not in motif_colors
    assert set(int(x) for x in np.unique(inp)) == {0, canvas, *motif_colors}
    assert not np.array_equal(motif, motif[:, ::-1])
    expected = inp.copy()
    expected[dr0:dr1, dc0:dc1] = motif[:, ::-1]
    assert np.array_equal(out, expected)
    if sr1 <= dr0:
        mode = "above"
    elif dr1 <= sr0:
        mode = "below"
    else:
        mode = "side"
    return {
        "canvas": canvas, "motif_colors": motif_colors,
        "palette": (canvas, *motif_colors),
        "shape": (height, width), "source": source,
        "dest": dest, "mode": mode, "shift": dc0 - sc0,
        "motif": motif,
    }


def _validate_generated_puzzle_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _rows_key(inp) not in _SOURCE_INPUT_ROWS
            info[split].append(_analyze_pair(inp, out))
    train = info["train"]
    test = info["test"]
    assert any(p["shape"][0] == p["shape"][1] for p in train)
    assert any(p["shape"][0] != p["shape"][1] for p in train)
    assert {"above", "below"}.issubset({p["mode"] for p in train})
    # A literal color must change role across training pairs.
    assert any(
        p["canvas"] in q["motif_colors"]
        for p in train for q in train if p is not q
    )
    train_left = any(p["shift"] < 0 for p in train)
    train_side = any(p["mode"] == "side" for p in train)
    for p in test:
        assert p["shift"] >= 0 or train_left
        assert p["mode"] != "side" or train_side
    # Reject training sets where copy, vertical flip, or a half-turn explains
    # every demonstrated output equally well.
    for transform in (
        lambda m: m,
        lambda m: m[::-1, :],
        lambda m: m[::-1, ::-1],
    ):
        assert any(
            not np.array_equal(p["motif"][:, ::-1], transform(p["motif"]))
            for p in train
        )
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed puzzle: {exc}") from exc


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)

    # The puzzle-wide convention is the zero destination and a left-right
    # reflection. Color roles, shape, and placement are visible pair locally.
    base = tuple(int(x) for x in rng.choice(np.arange(1, 10), size=3, replace=False))
    train_palettes = [(base[0], base[1], base[2]), (base[1], base[0], base[2])]
    train_modes = ["above", "below"]
    if int(rng.integers(0, 2)):
        train_modes.reverse()
    for _ in range(2, num_train):
        train_modes.append(("above", "below", "side")[int(rng.choice(3, p=(0.4, 0.4, 0.2)))])

    puzzle = {"train": [], "test": []}
    train_shapes = []
    train_shifts = []
    seen = set()
    for index in range(num_train):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        shape = _shape(pair_rng, True if index == 0 else False if index == 1 else None)
        palette = train_palettes[index] if index < 2 else _palette(pair_rng, train_palettes)
        mode = train_modes[index]
        for _ in range(25):
            pair = _make_pair(pair_rng, shape, palette, mode)
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise ValueError("duplicate training inputs")
        puzzle["train"].append(pair)
        seen.add(key)
        train_shapes.append(shape)
        train_palettes.append(palette)
        analyzed = _analyze_pair(pair["input"], pair["output"])
        train_shifts.append(analyzed["shift"])

    train_has_left = any(shift < 0 for shift in train_shifts)
    train_has_side = "side" in train_modes
    for index in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        draw = pair_rng.random()
        if draw < 0.25:
            shape = (3, 4)
        elif draw < 0.55:
            shape = train_shapes[int(pair_rng.integers(len(train_shapes)))]
        else:
            shape = _shape(pair_rng)
        palette = _palette(pair_rng, train_palettes)
        mode = ("above", "below", "side")[int(pair_rng.choice(3, p=(0.38, 0.42, 0.20)))]
        if mode == "side" and not train_has_side:
            mode = ("above", "below")[int(pair_rng.integers(0, 2))]
        shift = None
        if mode != "side" and pair_rng.random() < 0.32:
            shift = train_shifts[int(pair_rng.integers(len(train_shifts)))]
        for _ in range(30):
            pair = _make_pair(pair_rng, shape, palette, mode, shift)
            analyzed = _analyze_pair(pair["input"], pair["output"])
            if analyzed["shift"] < 0 and not train_has_left:
                shift = abs(analyzed["shift"])
                continue
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise ValueError("could not construct a unique test input")
        puzzle["test"].append(pair)
        seen.add(key)

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(
        tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
        for split in ("train", "test")
    )


def _expect_value_error(seed, train, test):
    try:
        generate_puzzle(seed, train, test)
    except ValueError:
        return
    raise AssertionError("invalid counts were accepted")


def _motif_structure_flags(motif):
    colors, amounts = np.unique(motif, return_counts=True)
    mask = motif == colors[int(np.argmin(amounts))]
    height, width = motif.shape
    flags = set()
    if int(amounts.min()) <= 2:
        flags.add("sparse")
    if len({tuple(int(x) for x in row) for row in motif}) < height:
        flags.add("repeated")
    if any(np.all(mask[r, c:c + 3]) for r in range(height)
           for c in range(width - 2)) or any(
           np.all(mask[r:r + 3, c]) for r in range(height - 2)
           for c in range(width)):
        flags.add("bar")
    if any(np.all(mask[r:r + 2, c:c + 2]) for r in range(height - 1)
           for c in range(width - 1)):
        flags.add("block")
    bits = mask.astype(np.int8)
    stripes = (np.arange(height)[:, None] % 2,
               np.arange(width)[None, :] % 2)
    if min(int(np.count_nonzero(bits != candidate))
           for pattern in stripes for candidate in (pattern, 1 - pattern)) <= 3:
        flags.add("stripe")
    points = set(zip(*np.nonzero(mask)))
    if len(points) >= 3:
        visited = {next(iter(points))}
        stack = list(visited)
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in points and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if visited == points:
            flags.add("connected")
    return flags


def self_test() -> None:
    for seed, train, test in ((13, None, None), (14, 7, 3), (15, None, 1),
                              (16, None, 4), (17, 2, None), (18, 10, None),
                              (19, 2, 1), (20, 10, 1), (21, 10, 4)):
        a = generate_puzzle(seed, train, test)
        b = generate_puzzle(seed, train, test)
        assert _puzzle_key(a) == _puzzle_key(b)
        validate_generated_puzzle(a)
        if train is not None:
            assert len(a["train"]) == train
        if test is not None:
            assert len(a["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        _expect_value_error(22, train, test)

    puzzle_keys = set()
    shapes, grids, layouts, palettes = set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    pairs = []
    review_flags = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        info = _validate_generated_puzzle_impl(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_palettes = {p["palette"] for p in info["train"]}
        train_colors = {color for palette in train_palettes for color in palette}
        train_shapes = {p["shape"] for p in info["train"]}
        train_shifts = {p["shift"] for p in info["train"]}
        if len(train_palettes) < len(info["train"]):
            review_flags.add("repeated_palette")
        if any(a["canvas"] == b["canvas"] and a["mode"] != b["mode"]
               for a in info["train"] for b in info["train"]):
            review_flags.add("same_canvas_different_order")
        for split in ("train", "test"):
            for pair, data in zip(puzzle[split], info[split]):
                inp, out = pair["input"], pair["output"]
                # Infer the candidate rule only from the visible input and
                # compare four competing transformations against the label.
                r0, r1, c0, c1 = data["dest"]
                candidates = (
                    data["motif"][:, ::-1],
                    data["motif"],
                    data["motif"][::-1, :],
                    data["motif"][::-1, ::-1],
                )
                assert np.array_equal(out[r0:r1, c0:c1], candidates[0])
                assert not any(np.array_equal(out[r0:r1, c0:c1], x) for x in candidates[1:])
                shapes.add(data["shape"])
                grids.add(inp.shape)
                palettes.add(data["palette"])
                layouts.add((inp.shape, data["shape"], data["source"], data["dest"]))
                pairs.append((split, inp, data))
                review_flags.update(_motif_structure_flags(data["motif"]))
                sr0, sr1, sc0, sc1 = data["source"]
                dr0, dr1, dc0, dc1 = data["dest"]
                if sr0 == 0 or dr0 == 0:
                    review_flags.add("top_edge")
                if sr1 == inp.shape[0] or dr1 == inp.shape[0]:
                    review_flags.add("bottom_edge")
                if sc0 == 0 or dc0 == 0:
                    review_flags.add("left_edge")
                if sc1 == inp.shape[1] or dc1 == inp.shape[1]:
                    review_flags.add("right_edge")
                gap = (abs(dr0 - sr0) - data["shape"][0]
                       if data["mode"] != "side"
                       else abs(data["shift"]) - data["shape"][1])
                review_flags.add("gap_zero" if gap == 0 else
                                 "gap_large" if gap >= 5 else "gap_middle")
                if data["shift"] < 0:
                    review_flags.add("shift_left")
                elif data["shift"] == 0:
                    review_flags.add("shift_aligned")
                elif data["shift"] < data["shape"][1]:
                    review_flags.add("shift_overlap")
                else:
                    review_flags.add("shift_separated")
                if split == "test":
                    if data["palette"] in train_palettes:
                        review_flags.add("test_familiar_palette")
                    elif set(data["palette"]).issubset(train_colors):
                        review_flags.add("test_recombined_palette")
                    else:
                        review_flags.add("test_new_color")
                    if data["shape"] in train_shapes:
                        review_flags.add("test_familiar_shape")
                    elif data["shape"][0] * data["shape"][1] > max(
                            h * w for h, w in train_shapes):
                        review_flags.add("test_larger_shape")
                    elif data["shape"][0] * data["shape"][1] < min(
                            h * w for h, w in train_shapes):
                        review_flags.add("test_smaller_shape")
                    if data["shift"] in train_shifts:
                        review_flags.add("test_familiar_shift")
    assert len(puzzle_keys) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(grids) >= 30
    assert len(layouts) >= 150 and len(palettes) >= 50
    assert {
        "sparse", "repeated", "bar", "block", "stripe", "connected",
        "repeated_palette", "same_canvas_different_order",
        "top_edge", "bottom_edge", "left_edge", "right_edge",
        "gap_zero", "gap_middle", "gap_large",
        "shift_left", "shift_aligned", "shift_overlap", "shift_separated",
        "test_familiar_palette", "test_recombined_palette", "test_new_color",
        "test_familiar_shape", "test_larger_shape", "test_smaller_shape",
        "test_familiar_shift",
    }.issubset(review_flags)
    assert any(inp.shape == (12, 12) for _, inp, _ in pairs)
    assert any(inp.shape[0] >= 23 for _, inp, _ in pairs)
    assert any(inp.shape[1] >= 23 for _, inp, _ in pairs)
    assert any(data["shift"] < 0 for _, _, data in pairs)
    assert any(data["shift"] == 0 for _, _, data in pairs)
    assert any(data["mode"] == "side" for _, _, data in pairs)
    assert any(data["mode"] == "below" for split, _, data in pairs if split == "test")
    assert any(data["shape"][0] > data["shape"][1] for _, _, data in pairs)
    assert any(data["shape"][0] < data["shape"][1] for _, _, data in pairs)
    assert any(
        data["source"][0] == 0 or data["dest"][0] == 0 or
        data["source"][1] == inp.shape[0] or data["dest"][1] == inp.shape[0]
        for _, inp, data in pairs
    )
    assert any(
        np.unique(data["motif"], return_counts=True)[1].min() <= 2
        for _, _, data in pairs
    )
    assert any(
        len({tuple(int(x) for x in row) for row in data["motif"]}) <
        data["shape"][0]
        for _, _, data in pairs
    )
    corrupted = generate_puzzle(900, 3, 1)
    first = corrupted["train"][0]
    destination = _analyze_pair(first["input"], first["output"])["dest"]
    bad_output = first["output"].copy()
    bad_output[destination[0], destination[2]] = 0
    first["output"] = bad_output
    rejected = False
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        rejected = True
    assert rejected, "validator accepted an incorrect reflected cell"
