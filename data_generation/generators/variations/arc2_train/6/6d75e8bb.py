import numpy as np


# Avoiding every original input is stronger than merely avoiding exact
# original input/output pairs.
_SOURCE_INPUTS = (
    (
        "00000000",
        "08888880",
        "08088080",
        "08080080",
        "00080880",
        "00000000",
        "00000000",
    ),
    (
        "0000000000",
        "0000000000",
        "0888000000",
        "0800000000",
        "0888800000",
        "0880000000",
        "0888000000",
        "0800000000",
        "0888000000",
        "0888000000",
        "0880000000",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "00000000",
        "08888800",
        "00080800",
        "00888800",
        "00088800",
        "00008800",
        "00088800",
        "00000000",
        "00000000",
    ),
    (
        "00000000000",
        "00000000000",
        "00800000000",
        "00800088000",
        "00808008000",
        "00888008000",
        "00888808800",
        "00888888800",
        "00000000000",
    ),
)


def _check_integer(value, name: str, low: int, high: int) -> int | None:
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
    train_count = _check_integer(num_train, "num_train", 2, 10)
    test_count = _check_integer(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % number_of_choices
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + seed % number_of_choices
    else:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for the requested train count")
        test_count = 1 + seed % maximum_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("could not obtain a legal train/test count combination")

    return int(train_count), int(test_count)


# A puzzle has one learned output color.  Input colors may be fixed or may
# change by pair; their roles remain visible from the uniform corner color.
def _palette(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    mode = int(rng.choice(4, p=[0.40, 0.18, 0.18, 0.24]))
    return {
        "completion": colors[0],
        "backgrounds": colors[1:4] if mode in (1, 3) else colors[1:2],
        "foregrounds": colors[4:7] if mode in (2, 3) else colors[4:5],
    }


def _colors_for_pair(
    rng: np.random.Generator, palette: dict, index: int, train_count: int,
    is_train: bool,
) -> tuple[int, int, int]:
    # The first two demonstrations exhibit any active color variation.
    if is_train:
        bi = index % len(palette["backgrounds"])
        fi = index % len(palette["foregrounds"])
    else:
        shown = min(train_count, 3)
        bi = int(rng.integers(0, min(shown, len(palette["backgrounds"]))))
        fi = int(rng.integers(0, min(shown, len(palette["foregrounds"]))))
    return (
        palette["backgrounds"][bi],
        palette["foregrounds"][fi],
        palette["completion"],
    )


def _path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int],
          rng: np.random.Generator) -> None:
    row, col = start
    er, ec = end
    mask[row, col] = True
    while (row, col) != (er, ec):
        choices = []
        if row != er:
            choices.append(0)
        if col != ec:
            choices.append(1)
        if choices[int(rng.integers(len(choices)))] == 0:
            row += 1 if er > row else -1
        else:
            col += 1 if ec > col else -1
        mask[row, col] = True


def _grow(mask: np.ndarray, rng: np.random.Generator, target: int,
          protected: set[tuple[int, int]] = frozenset()) -> None:
    height, width = mask.shape
    target = min(target, height * width - len(protected))
    while int(mask.sum()) < target:
        frontier = []
        for row in range(height):
            for col in range(width):
                if mask[row, col] or (row, col) in protected:
                    continue
                if ((row > 0 and mask[row - 1, col])
                    or (row + 1 < height and mask[row + 1, col])
                    or (col > 0 and mask[row, col - 1])
                    or (col + 1 < width and mask[row, col + 1])):
                    frontier.append((row, col))
        if not frontier:
            break
        row, col = frontier[int(rng.integers(len(frontier)))]
        mask[row, col] = True


def _tips(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    center = (int(rng.integers(1, height - 1)),
              int(rng.integers(1, width - 1)))
    tips = ((0, int(rng.integers(width))),
            (height - 1, int(rng.integers(width))),
            (int(rng.integers(height)), 0),
            (int(rng.integers(height)), width - 1))
    for tip in tips:
        _path(mask, center, tip, rng)
    target = int(round(height * width * rng.uniform(0.23, 0.63)))
    _grow(mask, rng, max(int(mask.sum()), target))
    return mask


def _comb(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    spine = int(rng.choice([0, width - 1, int(rng.integers(1, width - 1))]))
    mask[:, spine] = True
    rows = rng.permutation(height)
    tooth_count = int(rng.integers(2, height + 1))
    for row in rows[:tooth_count]:
        row = int(row)
        side = int(rng.integers(2))
        endpoint = (int(rng.integers(0, spine + 1)) if side == 0
                    else int(rng.integers(spine, width)))
        mask[row, min(spine, endpoint):max(spine, endpoint) + 1] = True
    # At least one tooth establishes each missing horizontal extreme.
    if spine != 0:
        row = int(rows[0])
        mask[row, :spine + 1] = True
    if spine != width - 1:
        row = int(rows[1])
        mask[row, spine:] = True
    target = int(round(height * width * rng.uniform(0.26, 0.61)))
    _grow(mask, rng, max(int(mask.sum()), target))
    return mask


def _outline(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    side = int(rng.integers(4))
    if side < 2:
        row = 0 if side == 0 else height - 1
        gap = (row, int(rng.integers(1, width - 1)))
    else:
        col = 0 if side == 2 else width - 1
        gap = (int(rng.integers(1, height - 1)), col)
    mask[gap] = False
    target = int(round(height * width * rng.uniform(0.32, 0.69)))
    _grow(mask, rng, max(int(mask.sum()), target), {gap})
    return mask


def _ring(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    # A missing edge cell gives an open gap; a separator encloses another
    # background pocket. This directly contrasts both kinds of completion.
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    divider = int(rng.integers(2, width - 2)) if width >= 5 else 1
    mask[:, divider] = True
    gap_col = int(rng.choice([c for c in range(1, width - 1) if c != divider]))
    gap = (0, gap_col)
    mask[gap] = False
    # Preserve an enclosed pocket on the side opposite the gap.
    hole_col = (width - 2 if gap_col < divider else 1)
    hole = (int(rng.integers(1, height - 1)), hole_col)
    protected = {gap, hole}
    target = int(round(height * width * rng.uniform(0.36, 0.58)))
    _grow(mask, rng, max(int(mask.sum()), target), protected)
    return mask


def _dense(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    count = int(rng.integers(1, 3))
    choices = [(r, c) for r in range(height) for c in range(width)
               if (0 < r < height - 1 or 0 < c < width - 1)]
    indices = rng.choice(len(choices), size=count, replace=False)
    for index in indices:
        mask[choices[int(index)]] = False
    return mask


def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(int(x) for x in cells[0])
    seen = {start}
    stack = [start]
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for nr, nc in ((row - 1, col), (row + 1, col),
                       (row, col - 1), (row, col + 1)):
            if (0 <= nr < height and 0 <= nc < width
                    and mask[nr, nc] and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _edge_gaps(mask: np.ndarray) -> int:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    return int((edge & ~mask).sum())


def _has_hole(mask: np.ndarray) -> bool:
    height, width = mask.shape
    seen = set()
    stack = []
    for row in range(height):
        for col in (0, width - 1):
            if not mask[row, col]:
                stack.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if not mask[row, col]:
                stack.append((row, col))
    while stack:
        row, col = stack.pop()
        if (row, col) in seen:
            continue
        seen.add((row, col))
        for nr, nc in ((row - 1, col), (row + 1, col),
                       (row, col - 1), (row, col + 1)):
            if (0 <= nr < height and 0 <= nc < width
                    and not mask[nr, nc] and (nr, nc) not in seen):
                stack.append((nr, nc))
    return int((~mask).sum()) > len(seen)


def _make_mask(
    height: int, width: int, style: str, rng: np.random.Generator,
) -> np.ndarray:
    builders = {
        "tips": _tips, "comb": _comb, "outline": _outline,
        "ring": _ring, "dense": _dense,
    }
    if style not in builders:
        raise ValueError("unknown shape style")
    for _ in range(24):
        transpose = bool(rng.integers(2))
        h, w = (width, height) if transpose else (height, width)
        mask = builders[style](h, w, rng)
        if transpose:
            mask = mask.T
        if bool(rng.integers(2)):
            mask = mask[::-1, :]
        if bool(rng.integers(2)):
            mask = mask[:, ::-1]
        if not (_connected(mask) and 1 <= int((~mask).sum())
                and mask[0, :].any() and mask[-1, :].any()
                and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if style in ("tips", "comb", "outline", "ring") and not _edge_gaps(mask):
            continue
        if style == "ring" and not _has_hole(mask):
            continue
        return mask.copy()
    # A deterministic connected fallback with open gaps and all four extrema.
    mask = np.zeros((height, width), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    if style == "dense":
        mask[:, :] = True
        mask[1, 1] = False
    return mask


def _dimensions(
    rng: np.random.Generator, regime: str,
    train_shapes: list[tuple[int, int]],
) -> tuple[int, int]:
    if regime == "source_train":
        mode = int(rng.choice(3, p=[0.70, 0.15, 0.15]))
        if mode == 0:
            return int(rng.integers(4, 10)), int(rng.integers(4, 10))
        if mode == 1:
            return int(rng.integers(4, 9)), int(rng.integers(9, 13))
        return int(rng.integers(9, 13)), int(rng.integers(4, 9))
    mode = int(rng.choice(5, p=[0.19, 0.21, 0.18, 0.22, 0.20]))
    if mode == 0:
        return (6, 7)
    if mode == 1 and train_shapes:
        return train_shapes[int(rng.integers(len(train_shapes)))]
    if mode == 2:
        return int(rng.integers(9, 15)), int(rng.integers(4, 7))
    if mode == 3:
        return int(rng.integers(4, 11)), int(rng.integers(4, 11))
    return int(rng.integers(8, 15)), int(rng.integers(8, 15))


def _margins(
    rng: np.random.Generator, mask: np.ndarray,
) -> tuple[int, int, int, int]:
    margins = [int(rng.integers(1, 9)) for _ in range(4)]
    if float(rng.random()) < 0.26:
        margins[int(rng.integers(4))] = 0
    height, width = mask.shape
    while height + margins[0] + margins[1] > 30:
        i = 0 if margins[0] >= margins[1] else 1
        margins[i] -= 1
    while width + margins[2] + margins[3] > 30:
        i = 2 if margins[2] >= margins[3] else 3
        margins[i] -= 1
    # Keep the uniform background the dominant input color even for dense
    # rectangles and narrow canvases. A zero margin, when sampled, stays zero.
    minimum_area = int(np.ceil(2.2 * int(mask.sum())))
    while ((height + margins[0] + margins[1])
           * (width + margins[2] + margins[3]) < minimum_area):
        canvas_height = height + margins[0] + margins[1]
        canvas_width = width + margins[2] + margins[3]
        choices = []
        if canvas_height < 30:
            choices.extend((0, 1))
        if canvas_width < 30:
            choices.extend((2, 3))
        choices = [i for i in choices if margins[i] > 0]
        assert choices
        gains = [canvas_width, canvas_width, canvas_height, canvas_height]
        maximum_gain = max(gains[i] for i in choices)
        best = [i for i in choices if gains[i] == maximum_gain]
        margins[best[int(rng.integers(len(best)))]] += 1
    return tuple(margins)


def _pair(
    rng: np.random.Generator, shape: tuple[int, int], style: str,
    roles: tuple[int, int, int],
) -> dict:
    mask = _make_mask(*shape, style, rng)
    top, bottom, left, right = _margins(rng, mask)
    height, width = mask.shape
    background, foreground, completion = roles
    input_grid = np.full((top + height + bottom, left + width + right),
                         background, dtype=np.int8)
    input_box = input_grid[top:top + height, left:left + width]
    input_box[mask] = foreground
    output_grid = input_grid.copy()
    output_box = output_grid[top:top + height, left:left + width]
    output_box[~mask] = completion
    return {"input": input_grid, "output": output_grid}


def _array_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _facts(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    corners = (int(inp[0, 0]), int(inp[0, -1]),
               int(inp[-1, 0]), int(inp[-1, -1]))
    assert len(set(corners)) == 1
    background = corners[0]
    input_colors = set(int(x) for x in np.unique(inp))
    assert len(input_colors) == 2 and background in input_colors
    foreground = next(iter(input_colors - {background}))
    assert int((inp == background).sum()) > int((inp == foreground).sum())
    full_mask = inp == foreground
    assert _connected(full_mask)
    coordinates = np.argwhere(full_mask)
    row0, col0 = [int(x) for x in coordinates.min(axis=0)]
    row1, col1 = [int(x) for x in coordinates.max(axis=0)]
    mask = full_mask[row0:row1 + 1, col0:col1 + 1]
    assert min(mask.shape) >= 4
    assert 1 <= int((~mask).sum())
    changed = inp != out
    assert changed.any() and np.all(inp[changed] == background)
    changed_colors = np.unique(out[changed])
    assert len(changed_colors) == 1
    completion = int(changed_colors[0])
    assert len({background, foreground, completion}) == 3
    expected = inp.copy()
    box = expected[row0:row1 + 1, col0:col1 + 1]
    box[box == background] = completion
    assert np.array_equal(expected, out)
    assert int(changed.sum()) == int((~mask).sum())
    return {
        "background": background, "foreground": foreground,
        "completion": completion, "bbox_shape": mask.shape,
        "canvas_shape": inp.shape, "margins": (
            row0, inp.shape[0] - row1 - 1, col0, inp.shape[1] - col1 - 1),
        "mask": mask, "gaps": int((~mask).sum()),
        "edge_gaps": _edge_gaps(mask), "hole": _has_hole(mask),
        "density": float(mask.mean()),
        "edge_counts": (int(mask[0].sum()), int(mask[-1].sum()),
                        int(mask[:, 0].sum()), int(mask[:, -1].sum())),
    }


def _layout_key(facts: dict) -> tuple:
    mask = facts["mask"]
    return (tuple(mask.shape), tuple(int(x) for x in mask.flat))


def _full_profile(facts: dict) -> tuple[bool, bool, bool, bool]:
    height, width = facts["bbox_shape"]
    return tuple(
        count == length for count, length in
        zip(facts["edge_counts"], (width, width, height, height))
    )


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    completions = set()
    for pair in train:
        changed = pair["input"] != pair["output"]
        assert changed.any()
        completions.update(int(x) for x in np.unique(pair["output"][changed]))
    assert len(completions) == 1
    completion = next(iter(completions))
    background = int(test_input[0, 0])
    foregrounds = set(int(x) for x in np.unique(test_input)) - {background}
    assert len(foregrounds) == 1
    foreground = next(iter(foregrounds))
    coordinates = np.argwhere(test_input == foreground)
    row0, col0 = coordinates.min(axis=0)
    row1, col1 = coordinates.max(axis=0)
    answer = test_input.copy()
    box = answer[row0:row1 + 1, col0:col1 + 1]
    box[box == background] = completion
    return answer


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            ik = _array_key(inp)
            pk = (ik, _array_key(out))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            assert _rows_signature(inp) not in _SOURCE_INPUTS
            detail = _facts(pair)
            key = _layout_key(detail)
            assert key not in seen_layouts
            seen_layouts.add(key)
            facts[split].append(detail)
    completion_colors = {f["completion"] for f in facts["train"] + facts["test"]}
    assert len(completion_colors) == 1
    assert any(f["edge_gaps"] for f in facts["train"])
    assert len({f["bbox_shape"] for f in facts["train"]}) >= 2
    assert len({_full_profile(f) for f in facts["train"]}) >= 2
    for pair in test:
        assert np.array_equal(_predict_from_training(train, pair["input"]),
                              pair["output"])


def _unique_pair(
    rng: np.random.Generator, regime: str, style: str,
    palette: dict, index: int, train_count: int,
    train_shapes: list[tuple[int, int]],
    seen_inputs: set, seen_layouts: set,
    forbidden_train_shape: tuple[int, int] | None = None,
    forbidden_edge_profile: tuple[bool, bool, bool, bool] | None = None,
) -> dict:
    roles = _colors_for_pair(rng, palette, index, train_count,
                            regime == "source_train")
    for _ in range(160):
        shape = _dimensions(rng, regime, train_shapes)
        if shape == forbidden_train_shape:
            continue
        pair = _pair(rng, shape, style, roles)
        inp = pair["input"]
        if _rows_signature(inp) in _SOURCE_INPUTS:
            continue
        ik = _array_key(inp)
        facts = _facts(pair)
        lk = _layout_key(facts)
        if _full_profile(facts) == forbidden_edge_profile:
            continue
        if ik in seen_inputs or lk in seen_layouts:
            continue
        seen_inputs.add(ik)
        seen_layouts.add(lk)
        return pair
    raise AssertionError("bounded pair construction could not find a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    palette = _palette(rng)
    styles = ("tips", "comb", "outline", "ring", "dense")
    first_style = styles[int(rng.integers(4))]
    second_style = styles[int(rng.integers(5))]
    if second_style == first_style:
        second_style = styles[(styles.index(first_style) + 1
                               + int(rng.integers(4))) % 5]
    seen_inputs, seen_layouts = set(), set()
    train_shapes: list[tuple[int, int]] = []
    train = []
    for index in range(train_count):
        style = (first_style if index == 0 else
                 second_style if index == 1 else
                 styles[int(rng.integers(5))])
        forbidden = train_shapes[0] if index == 1 else None
        forbidden_profile = _full_profile(_facts(train[0])) if index == 1 else None
        pair = _unique_pair(rng, "source_train", style, palette, index,
                            train_count, train_shapes, seen_inputs,
                            seen_layouts, forbidden, forbidden_profile)
        train.append(pair)
        train_shapes.append(_facts(pair)["bbox_shape"])
    test = []
    for index in range(test_count):
        style = styles[int(rng.integers(5))]
        pair = _unique_pair(rng, "target_test", style, palette, index,
                            train_count, train_shapes, seen_inputs,
                            seen_layouts)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((split, tuple((_array_key(p["input"]), _array_key(p["output"]))
                               for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    masks = set()
    palettes = set()
    canvas_shapes = set()
    bbox_shapes = set()
    train_full_left = test_full_left = 0
    train_full_top = test_full_top = 0
    sparse = dense = one_gap = holes_and_open = tip_extrema = edge_touch = 0
    same_size = source_like_test = larger_test = tall_test = 0
    variable_background = variable_foreground = 0
    fixed_palette = shared_edge_profiles = smaller_test = 0
    open_train = varied_profiles = varied_margins = 0
    tip_sides = [0, 0, 0, 0]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_facts = [_facts(p) for p in puzzle["train"]]
        test_facts = [_facts(p) for p in puzzle["test"]]
        if len({f["background"] for f in train_facts}) > 1:
            variable_background += 1
        if len({f["foreground"] for f in train_facts}) > 1:
            variable_foreground += 1
        fixed_palette += len({
            (f["background"], f["foreground"])
            for f in train_facts + test_facts
        }) == 1
        train_profiles = {_full_profile(f) for f in train_facts}
        test_profiles = {_full_profile(f) for f in test_facts}
        shared_edge_profiles += bool(train_profiles & test_profiles)
        varied_profiles += len(train_profiles) > 1
        varied_margins += len({
            f["margins"] for f in train_facts + test_facts
        }) > 1
        open_train += any(f["edge_gaps"] for f in train_facts)
        train_shapes = {f["bbox_shape"] for f in train_facts}
        train_max_area = max(f["mask"].size for f in train_facts)
        for f in test_facts:
            same_size += f["bbox_shape"] in train_shapes
            source_like_test += f["bbox_shape"] == (6, 7)
            larger_test += f["mask"].size > train_max_area
            smaller_test += f["mask"].size <= train_max_area
            tall_test += f["bbox_shape"][0] >= 9 and f["bbox_shape"][1] <= 6
        for split, facts_list in (("train", train_facts), ("test", test_facts)):
            for f in facts_list:
                mask = f["mask"]
                masks.add(_layout_key(f))
                palettes.add((f["background"], f["foreground"], f["completion"]))
                canvas_shapes.add(f["canvas_shape"])
                bbox_shapes.add(f["bbox_shape"])
                full_left = bool(mask[:, 0].all())
                full_top = bool(mask[0, :].all())
                if split == "train":
                    train_full_left += full_left
                    train_full_top += full_top
                else:
                    test_full_left += full_left
                    test_full_top += full_top
                sparse += f["density"] < 0.42
                dense += f["density"] > 0.85
                one_gap += f["gaps"] == 1
                holes_and_open += f["hole"] and f["edge_gaps"] > 0
                tip_extrema += any(c <= 2 for c in f["edge_counts"])
                for side, count in enumerate(f["edge_counts"]):
                    tip_sides[side] += count <= 2
                edge_touch += 0 in f["margins"]
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_training(puzzle["train"],
                                  pair["input"]), pair["output"])
    assert len(masks) >= 100 and len(canvas_shapes) >= 30
    assert len(bbox_shapes) >= 25 and len(palettes) >= 30
    assert train_full_left > 10 and test_full_left > 4
    assert train_full_top > 10 and test_full_top > 4
    assert sparse > 10 and dense > 10 and one_gap > 5
    assert holes_and_open > 8 and tip_extrema > 20 and edge_touch > 10
    assert same_size > 5 and source_like_test > 4
    assert larger_test > 5 and tall_test > 4
    assert variable_background > 4 and variable_foreground > 4
    assert fixed_palette > 4 and shared_edge_profiles > 8
    assert smaller_test > 10 and open_train == 42
    assert varied_profiles == 42 and varied_margins == 42
    assert min(tip_sides) > 10
    a = generate_puzzle(20260827, 7, 3)
    b = generate_puzzle(20260827, 7, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    assert _puzzle_key(puzzles[0]) != _puzzle_key(puzzles[1])
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (2001, 3, 2), (2002, 6, 3), (2003, 9, 4),
                         (3001, None, 1), (3002, None, 4),
                         (3003, 2, None), (3004, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4},
                   {"num_train": 2.5, "num_test": 1},
                   {"num_train": 3, "num_test": True}):
        try:
            generate_puzzle(4001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
