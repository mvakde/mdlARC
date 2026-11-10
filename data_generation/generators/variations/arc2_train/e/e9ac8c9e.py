import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(rng, num_train, num_test):
    num_train = _validate_count_value("num_train", num_train, 2, 10)
    num_test = _validate_count_value("num_test", num_test, 1, 4)
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


def _array_signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _pair_signature(pair):
    return (_array_signature(pair["input"]), _array_signature(pair["output"]))


def _components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr,nc] and not visited[nr,nc]:
                        visited[nr,nc] = True
                        stack.append((nr,nc))
            result.append(cells)
    return result


def _separated(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah + 1 <= br or br + bh + 1 <= ar or
            ac + aw + 1 <= bc or bc + bw + 1 <= ac)


def _sample_roles(rng, conventions, split, index):
    base_bg, base_ph = conventions["background"], conventions["placeholder"]
    if conventions["vary_roles"] and (split, index) != ("train", 0):
        choices = [v for v in range(10) if v != base_bg] if (split, index) == ("train", 1) else list(range(10))
        background = int(rng.choice(choices))
        excluded = {background}
        if (split, index) == ("train", 1):
            excluded.add(base_ph)
        placeholder = int(rng.choice([v for v in range(10) if v not in excluded]))
    else:
        background, placeholder = base_bg, base_ph
    return background, placeholder


def _sample_shape(rng, count, regime):
    if count >= 3:
        lengths = (2, 4, 6)
    elif count == 2:
        lengths = (2, 4, 6, 8)
    else:
        lengths = (2, 4, 6, 8, 10)
    height = int(rng.choice(lengths))
    if rng.random() < (0.33 if regime == "source_train" else 0.23):
        width = height
    else:
        width = int(rng.choice(lengths))
    return height, width


def _place(rng, shape, dimensions):
    height, width = shape
    footprints = []
    for object_height, object_width in dimensions:
        extent_height, extent_width = object_height + 2, object_width + 2
        candidates = [(r, c, extent_height, extent_width)
                      for r in range(height - extent_height + 1)
                      for c in range(width - extent_width + 1)]
        if not candidates:
            return None
        order = rng.permutation(len(candidates))
        fit = next((candidates[int(i)] for i in order
                    if all(_separated(candidates[int(i)], prior) for prior in footprints)), None)
        if fit is None:
            return None
        footprints.append(fit)
    return footprints


def _sample_geometry(rng, regime, index, count):
    if regime == "source_train" and index == 0:
        size = int(rng.choice((2, 4, 6, 8)))
        dimensions = [(size, size)]
    elif regime == "source_train" and index == 1:
        horizontal = ((2, 4), (2, 6), (2, 8), (4, 6), (4, 8), (6, 8))
        dimensions = [horizontal[int(rng.integers(len(horizontal)))],
                      tuple(reversed(horizontal[int(rng.integers(len(horizontal)))]))]
        if rng.random() < 0.5:
            dimensions.reverse()
    else:
        dimensions = [_sample_shape(rng, count, regime) for _ in range(count)]
    minimum = 8 if count == 1 else 12 if count == 2 else 16 if count == 3 else 18
    for attempt in range(80):
        if attempt and attempt % 12 == 0 and not (regime == "source_train" and index in (0, 1)):
            dimensions = [_sample_shape(rng, count, regime) for _ in range(count)]
        if attempt == 0 and rng.random() < 0.22:
            height = width = int(rng.choice((10, 15)))
        elif attempt < 48:
            height = int(rng.integers(minimum, 31))
            width = int(rng.integers(minimum, 31))
        else:
            height = width = 30
        for _ in range(4):
            footprints = _place(rng, (height, width), dimensions)
            if footprints is not None:
                return (height, width), dimensions, footprints
    raise ValueError("unable to place separated rectangles")


def _marker_rows(rng, pool, count, scheme, allow_repeat, allow_equal):
    available = [int(v) for v in pool]
    if scheme == "palette_permute" and count >= 2:
        base = tuple(int(v) for v in rng.choice(available, size=4, replace=False))
        rows = [base]
        for _ in range(1, count):
            row = tuple(int(v) for v in rng.permutation(base))
            if row == base:
                row = base[1:] + base[:1]
            rows.append(row)
        return rows
    rows = []
    disjoint = scheme == "disjoint" and len(available) >= 4 * count
    if disjoint:
        available = [int(v) for v in rng.permutation(available)]
    for object_index in range(count):
        if scheme == "teaching_repeat" and object_index == 0:
            distinct_count = int(rng.choice((2, 3)))
        elif allow_equal and rng.random() < 0.07:
            distinct_count = 1
        elif allow_repeat and rng.random() < 0.31:
            distinct_count = int(rng.choice((2, 3)))
        else:
            distinct_count = 4
        choices = available[4*object_index:4*object_index+4] if disjoint else available
        if distinct_count > len(choices):
            choices = available
        if distinct_count == 4:
            row = [int(v) for v in rng.choice(choices, size=4, replace=False)]
        else:
            selected = [int(v) for v in rng.choice(choices, size=distinct_count, replace=False)]
            row = selected + [int(rng.choice(selected)) for _ in range(4-distinct_count)]
            rng.shuffle(row)
        rows.append(tuple(row))
    if scheme == "role_swap" and count >= 2:
        color = rows[0][0]
        second = list(rows[1])
        target = int(rng.choice((1, 2, 3)))
        if color in second and not allow_repeat:
            old = second.index(color)
            second[old], second[target] = second[target], second[old]
        else:
            second[target] = color
        if not allow_equal and len(set(second)) == 1:
            second[(target + 1) % 4] = next(v for v in available if v != color)
        rows[1] = tuple(second)
    return rows


def _render(shape, footprints, dimensions, rows, background, placeholders):
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    for (top, left, _, _), (height, width), markers, placeholder in zip(footprints, dimensions, rows, placeholders):
        r0, c0 = top + 1, left + 1
        r1, c1 = r0 + height, c0 + width
        inp[r0:r1, c0:c1] = placeholder
        for (r,c), color in zip(((top,left), (top,c1), (r1,left), (r1,c1)), markers):
            inp[r,c] = color
        rm, cm = r0 + height//2, c0 + width//2
        for (ra,rb,ca,cb), color in zip(((r0,rm,c0,cm), (r0,rm,cm,c1),
                                        (rm,r1,c0,cm), (rm,r1,cm,c1)), markers):
            out[ra:rb,ca:cb] = color
    return {"input": inp, "output": out}


def _generate_pair(rng, conventions, regime, index, count, repeat_taught, equal_taught):
    shape, dimensions, footprints = _sample_geometry(rng, regime, index, count)
    background, primary = _sample_roles(rng, conventions, "train" if regime == "source_train" else "test", index)
    mixed = conventions["mixed_placeholders"] and count >= 2 and (
        (regime == "source_train" and index == 1) or rng.random() < 0.48)
    if mixed:
        if conventions["vary_roles"]:
            secondary = int(rng.choice([v for v in range(10) if v not in (background, primary)]))
        else:
            secondary = conventions["second_placeholder"]
        placeholders = [primary if i % 2 == 0 else secondary for i in range(count)]
    else:
        placeholders = [primary] * count
    pool = [v for v in range(10) if v not in {background, *placeholders}]
    if regime == "source_train" and index == 0:
        scheme, allow_repeat, allow_equal = "ordinary", False, False
    elif regime == "source_train" and index == 1 and conventions["repeat_markers"]:
        scheme, allow_repeat, allow_equal = "teaching_repeat", True, False
    else:
        scheme = str(rng.choice(("ordinary", "ordinary", "role_swap", "disjoint", "palette_permute")))
        allow_repeat = conventions["repeat_markers"] if regime == "source_train" else repeat_taught
        allow_equal = (regime == "source_train" and index >= 2 and conventions["repeat_markers"])
        if regime == "target_test":
            allow_equal = equal_taught
    rows = _marker_rows(rng, pool, count, scheme, allow_repeat, allow_equal)
    return _render(shape, footprints, dimensions, rows, background, placeholders)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    roles = [int(v) for v in rng.choice(10, size=3, replace=False)]
    conventions = {"background": roles[0], "placeholder": roles[1],
                   "second_placeholder": roles[2],
                   "vary_roles": bool(rng.random() < 0.48),
                   "mixed_placeholders": bool(rng.random() < 0.42),
                   "repeat_markers": bool(rng.random() < 0.68)}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, amount in (("train", train_count), ("test", test_count)):
        for index in range(amount):
            if split == "train":
                count = 1 if index == 0 else 2 if index == 1 else int(rng.choice((1,1,2,2,3,3,4)))
                regime = "source_train"
            else:
                options = (1,2,2,3,3,3,4,4)
                if index:
                    first_count = len(_analyze_pair(puzzle["test"][0]))
                    options = tuple(value for value in options if value != first_count)
                count = int(rng.choice(options))
                regime = "target_test"
            repeat_taught = any(len(set(obj["markers"])) < 4 for pair in puzzle["train"]
                                 for obj in _analyze_pair(pair)) if split == "test" else False
            equal_taught = any(len(set(obj["markers"])) == 1 for pair in puzzle["train"]
                                for obj in _analyze_pair(pair)) if split == "test" else False
            for _ in range(30):
                pair = _generate_pair(rng, conventions, regime, index, count, repeat_taught, equal_taught)
                try:
                    _analyze_pair(pair)
                except AssertionError:
                    continue
                if index and pair["input"].shape == puzzle[split][0]["input"].shape:
                    continue
                signature = _array_signature(pair["input"])
                if signature not in seen and not _is_exact_source_pair(pair):
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("unable to generate a new pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input and output sizes differ")
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _require(int(np.sum(counts == counts.max())) == 1, "background is not identifiable")
    objects = []
    singleton_cells = set()
    for color in colors:
        if color == background:
            continue
        for cells in _components(inp == color):
            if len(cells) == 1:
                singleton_cells.add(cells[0])
                continue
            r0 = min(r for r,c in cells)
            r1 = max(r for r,c in cells) + 1
            c0 = min(c for r,c in cells)
            c1 = max(c for r,c in cells) + 1
            height, width = r1-r0, c1-c0
            _require(height >= 2 and width >= 2 and height%2 == 0 and width%2 == 0,
                     "invalid placeholder dimensions")
            _require(len(cells) == height*width and np.all(inp[r0:r1,c0:c1] == color),
                     "placeholder is not a solid rectangle")
            objects.append({"color": int(color), "bounds": (r0,r1,c0,c1),
                            "shape": (height,width), "footprint": (r0-1,c0-1,height+2,width+2)})
    _require(bool(objects), "pair has no placeholder")
    _require(len(objects) <= 4, "too many rectangles")
    expected = np.full(inp.shape, background, dtype=np.int8)
    used = set()
    for obj in objects:
        r0,r1,c0,c1 = obj["bounds"]
        _require(r0 > 0 and c0 > 0 and r1 < inp.shape[0] and c1 < inp.shape[1],
                 "missing marker margin")
        coords = ((r0-1,c0-1), (r0-1,c1), (r1,c0-1), (r1,c1))
        _require(all(point in singleton_cells and point not in used for point in coords),
                 "missing or shared diagonal marker")
        used.update(coords)
        markers = tuple(int(inp[point]) for point in coords)
        ring = inp[r0-1:r1+1,c0-1:c1+1].copy()
        ring[1:-1,1:-1] = background
        for rr,cc in ((0,0), (0,-1), (-1,0), (-1,-1)):
            ring[rr,cc] = background
        _require(np.all(ring == background), "non-background cell in marker border")
        rm,cm = r0+(r1-r0)//2,c0+(c1-c0)//2
        for (ra,rb,ca,cb), color in zip(((r0,rm,c0,cm), (r0,rm,cm,c1),
                                        (rm,r1,c0,cm), (rm,r1,cm,c1)), markers):
            expected[ra:rb,ca:cb] = color
        obj["markers"] = markers
    _require(singleton_cells == used, "unassigned marker or noise")
    _require(all(_separated(a["footprint"],b["footprint"])
                 for i,a in enumerate(objects) for b in objects[i+1:]),
             "object footprints are not separated")
    _require(np.array_equal(out, expected), "incorrect corner-to-quadrant transfer")
    _require(not np.array_equal(inp,out), "unchanged input")
    return objects


def _grid_from_rows(
    rows: tuple[str, ...],
) -> np.ndarray:
    return np.array(
        [
            [int(character) for character in row]
            for row in rows
        ],
        dtype=np.int8,
    )


_SOURCE_PAIR_ROWS = (
    (
        (
            "0000000000",
            "0000000000",
            "0030000400",
            "0005555000",
            "0005555000",
            "0005555000",
            "0005555000",
            "0080000600",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0003344000",
            "0003344000",
            "0008866000",
            "0008866000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0400200000",
            "0055000000",
            "0055000000",
            "0700100000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0042000000",
            "0071000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "8000000900",
            "0555555000",
            "0555555000",
            "0555555000",
            "0555555000",
            "0555555000",
            "0555555000",
            "7000000600",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0888999000",
            "0888999000",
            "0888999000",
            "0777666000",
            "0777666000",
            "0777666000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "600000090000000",
            "055555500000000",
            "055555500000000",
            "055555500090070",
            "055555500005500",
            "055555500005500",
            "055555500020060",
            "700000080000000",
            "000000000000000",
            "000006000020000",
            "000000555500000",
            "000000555500000",
            "000000555500000",
            "000000555500000",
            "000008000030000",
        ),
        (
            "000000000000000",
            "066699900000000",
            "066699900000000",
            "066699900000000",
            "077788800009700",
            "077788800002600",
            "077788800000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000662200000",
            "000000662200000",
            "000000883300000",
            "000000883300000",
            "000000000000000",
        ),
    ),
)

_SOURCE_PAIRS = tuple(
    (
        _grid_from_rows(input_rows),
        _grid_from_rows(output_rows),
    )
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _is_exact_source_pair(pair: dict) -> bool:
    return any(
        pair["input"].shape == source_input.shape
        and pair["output"].shape == source_output.shape
        and np.array_equal(
            pair["input"],
            source_input,
        )
        and np.array_equal(
            pair["output"],
            source_output,
        )
        for source_input, source_output in _SOURCE_PAIRS
    )



def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test only")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs input and output only")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D NumPy array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape), "invalid grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must use integer dtype")
                _require(np.all((0 <= grid) & (grid <= 9)), "illegal ARC color")
            key = _array_signature(pair["input"])
            pair_key = _pair_signature(pair)
            _require(key not in seen_inputs and pair_key not in seen_pairs, "duplicate grid or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(not _is_exact_source_pair(pair), "exact source pair copy")
            analyses[split].append(_analyze_pair(pair))
    first, second = analyses["train"][:2]
    _require(len(first) == 1 and first[0]["shape"][0] == first[0]["shape"][1]
             and len(set(first[0]["markers"])) == 4,
             "first training example must clearly show all four corner roles")
    _require(len(second) == 2 and
             any(o["shape"][0] < o["shape"][1] for o in second) and
             any(o["shape"][0] > o["shape"][1] for o in second),
             "training must demonstrate both rectangular orientations")
    train_objects = [obj for group in analyses["train"] for obj in group]
    test_objects = [obj for group in analyses["test"] for obj in group]
    training_repeats = any(len(set(obj["markers"])) < 4 for obj in train_objects)
    training_equal = any(len(set(obj["markers"])) == 1 for obj in train_objects)
    training_mixed = any(len(set(obj["color"] for obj in group)) > 1
                         for group in analyses["train"])
    _require(training_repeats or all(len(set(obj["markers"])) == 4 for obj in test_objects),
             "test repeats marker colors without a training demonstration")
    _require(training_equal or all(len(set(obj["markers"])) > 1 for obj in test_objects),
             "test uses all-equal markers without a training demonstration")
    _require(training_mixed or all(len(set(obj["color"] for obj in group)) == 1
                                   for group in analyses["test"]),
             "test mixes placeholder colors without a training demonstration")
    train_backgrounds = {int(np.unique(pair["input"], return_counts=True)[0][
        int(np.argmax(np.unique(pair["input"], return_counts=True)[1]))])
        for pair in puzzle["train"]}
    if len(train_backgrounds) == 1:
        background = next(iter(train_backgrounds))
        _require(all(int(np.unique(pair["input"], return_counts=True)[0][
            int(np.argmax(np.unique(pair["input"], return_counts=True)[1]))]) == background
            for pair in puzzle["test"]), "unshown background role change")
        taught_colors = {obj["color"] for obj in train_objects}
        _require(all(obj["color"] in taught_colors for obj in test_objects),
                 "unshown placeholder role change")
    _require(len({len(group) for group in analyses["train"]}) >= 2,
             "training needs both single and multiple objects")
    _require(puzzle["train"][0]["input"].shape != puzzle["train"][1]["input"].shape,
             "training canvas dimensions do not vary")
    if ns >= 2:
        _require(len(analyses["test"][0]) != len(analyses["test"][1]),
                 "test object counts do not vary")
        _require(puzzle["test"][0]["input"].shape != puzzle["test"][1]["input"].shape,
                 "test canvas dimensions do not vary")


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x,y in zip(a[split],b[split])) for split in ("train", "test"))


def self_test() -> None:
    for seed, args in ((314159, {}), (271828, {"num_train": 6, "num_test": 3}),
                       (10121, {"num_train": 2, "num_test": 1}),
                       (11001, {"num_train": 10, "num_test": 1}),
                       (11004, {"num_train": 10, "num_test": 4}),
                       (20001, {"num_test": 1}), (20004, {"num_test": 4}),
                       (20002, {"num_train": 2}), (20010, {"num_train": 10})):
        puzzle = generate_puzzle(seed, **args)
        validate_generated_puzzle(puzzle)
        _require(_puzzles_equal(puzzle, generate_puzzle(seed, **args)), "nondeterministic puzzle")
        if args.get("num_train") is not None:
            _require(len(puzzle["train"]) == args["num_train"], "train count ignored")
        if args.get("num_test") is not None:
            _require(len(puzzle["test"]) == args["num_test"], "test count ignored")
    _require(not _puzzles_equal(generate_puzzle(271828), generate_puzzle(271829)),
             "different seeds generated identical puzzles")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(42, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request was accepted")
    puzzles = [generate_puzzle(700000 + i) for i in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "training counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    train_objects = [_analyze_pair(pair) for p in puzzles for pair in p["train"]]
    test_objects = [_analyze_pair(pair) for p in puzzles for pair in p["test"]]
    all_objects = [obj for group in train_objects + test_objects for obj in group]
    _require({len(group) for group in train_objects} >= {1,2,3}, "training object counts too narrow")
    _require({len(group) for group in test_objects} >= {1,2,3}, "test object counts too narrow")
    _require(any(o["shape"][0] < o["shape"][1] for o in all_objects) and
             any(o["shape"][0] > o["shape"][1] for o in all_objects),
             "rectangle orientations missing")
    _require(any(len(set(o["markers"])) < 4 for o in all_objects), "repeated local markers missing")
    _require(any(len(group) == 1 for group in test_objects) and
             any(len(group) == 2 for group in test_objects),
             "one- and two-object tests missing")
    _require(any(len(set(o["markers"])) < 4 for group in test_objects for o in group),
             "repeated-marker tests missing")
    groups = train_objects + test_objects
    _require(any(any(o["shape"][0] == o["shape"][1] for o in group) and
                 any(o["shape"][0] != o["shape"][1] for o in group)
                 for group in groups), "square and rectangle combination missing")
    _require(any(len(set(o["color"] for o in group)) > 1 and
                 any(len(set(o["markers"])) < 4 for o in group)
                 for group in groups), "mixed placeholders and repeated markers never combine")
    multi = [group for group in groups if len(group) >= 2]
    _require(any(all(not set(group[i]["markers"]) & set(group[j]["markers"])
                     for i in range(len(group)) for j in range(i+1, len(group)))
                 for group in multi), "disjoint multi-object palettes missing")
    _require(any(any(set(group[i]["markers"]) == set(group[j]["markers"]) and
                     group[i]["markers"] != group[j]["markers"] and
                     len(set(group[i]["markers"])) == 4
                     for i in range(len(group)) for j in range(i+1, len(group)))
                 for group in multi), "permuted full palette missing")
    _require(any(any(group[i]["markers"][a] == group[j]["markers"][b] and a != b
                     for i in range(len(group)) for j in range(i+1, len(group))
                     for a in range(4) for b in range(4))
                 for group in multi), "cross-object corner-role reuse missing")
    _require(len({pair["input"].shape for p in puzzles for split in ("train", "test")
                  for pair in p[split]}) >= 40, "canvas dimensions too narrow")
    _require(any(len({int(np.unique(pair["input"], return_counts=True)[0][
        int(np.argmax(np.unique(pair["input"], return_counts=True)[1]))])
        for pair in p["train"]}) > 1 for p in puzzles), "within-puzzle backgrounds never vary")
    _require(any(len({int(np.unique(pair["input"], return_counts=True)[0][
        int(np.argmax(np.unique(pair["input"], return_counts=True)[1]))])
        for pair in p["train"]}) == 1 for p in puzzles), "stable backgrounds never appear")
    _require(any(len(set(o["color"] for o in group)) > 1 for group in train_objects),
             "mixed placeholders never appear in training")
