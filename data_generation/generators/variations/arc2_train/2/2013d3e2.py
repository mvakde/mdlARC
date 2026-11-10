import numpy as np


def _make_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _make_grid(
            (
                "0000000000",
                "0100001000",
                "0036530000",
                "0052260000",
                "0062250000",
                "0035630000",
                "0100001000",
                "0000000000",
                "0000000000",
                "0000000000",
            )
        ),
        _make_grid(("100", "036", "052")),
    ),
    (
        _make_grid(
            (
                "0000000000",
                "0000000000",
                "0000770000",
                "0006886000",
                "0078448700",
                "0078448700",
                "0006886000",
                "0000770000",
                "0000000000",
                "0000000000",
            )
        ),
        _make_grid(("007", "068", "784")),
    ),
    (
        _make_grid(
            (
                "0000000000",
                "0000000000",
                "0000080000",
                "0004484000",
                "0088334000",
                "0004338800",
                "0004844000",
                "0000800000",
                "0000000000",
                "0000000000",
            )
        ),
        _make_grid(("000", "044", "883")),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _is_integer_argument(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _validate_optional_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if not _is_integer_argument(value):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(num_train, "num_train", 2, 10)
    test_count = _validate_optional_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        # This formula guarantees varied legal counts for consecutive seeds.
        test_count = 1 + seed % 4
        minimum_train = test_count + 1
        number_of_choices = 11 - minimum_train
        train_count = (
            minimum_train
            + (seed // 4) % number_of_choices
        )
    elif train_count is None:
        minimum_train = test_count + 1
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(int(dimension) for dimension in array.shape),
        tuple(int(value) for value in array.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _array_key(pair["input"]),
        _array_key(pair["output"]),
    )


def _is_source_pair(pair: dict) -> bool:
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            np.array_equal(pair["input"], source_input)
            and np.array_equal(pair["output"], source_output)
        ):
            return True
    return False


def _all_rotations_distinct(tile: np.ndarray) -> bool:
    return all(not np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3))


def _compose_pinwheel(tile: np.ndarray) -> np.ndarray:
    return np.block([[tile, np.rot90(tile, -1)],
                     [np.rot90(tile, 1), np.rot90(tile, 2)]])


def _sample_size(rng: np.random.Generator) -> int:
    band = int(rng.choice(4, p=(0.12, 0.34, 0.34, 0.20)))
    lo, hi = ((2, 3), (3, 6), (6, 10), (10, 14))[band]
    return int(rng.integers(lo, hi + 1))


def _blank_mode(rng: np.random.Generator, n: int, regime: str,
                first_train: bool) -> str:
    if first_train:
        return "none"
    modes = ("none", "top", "bottom", "left", "right", "internal_row",
             "internal_col", "top_two", "bottom_two", "left_two", "right_two",
             "cross")
    weights = ([4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1] if regime == "source_train"
               else [3, 3, 2, 3, 2, 1, 1, 2, 1, 2, 1, 1])
    if n < 4:
        for i in range(7, 12):
            weights[i] = 0
    if n < 3:
        weights[5] = weights[6] = 0
    weights = np.array(weights, dtype=float)
    return str(rng.choice(modes, p=weights / weights.sum()))


def _sample_mask(rng: np.random.Generator, n: int, family: str) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if family == "scatter":
        mask = rng.random((n, n)) < rng.uniform(0.28, 0.86)
    elif family == "stair":
        span = int(rng.integers(1, max(2, n // 2 + 1)))
        for row in range(n):
            start = min(n - 1, max(0, row + int(rng.integers(-1, 2))))
            mask[row, start:min(n, start + span)] = True
    elif family == "strokes":
        for _ in range(int(rng.integers(2, 6))):
            a, lo = (int(x) for x in rng.integers(0, n, size=2))
            hi = int(rng.integers(lo + 1, n + 1))
            if rng.random() < 0.5:
                mask[a, lo:hi] = True
            else:
                mask[lo:hi, a] = True
    elif family == "patch":
        r0, r1 = sorted(int(x) for x in rng.integers(0, n, size=2))
        c0, c1 = sorted(int(x) for x in rng.integers(0, n, size=2))
        mask[r0:r1 + 1, c0:c1 + 1] = True
        mask |= rng.random((n, n)) < 0.08
    elif family == "holes":
        mask = rng.random((n, n)) > rng.uniform(0.12, 0.42)
    elif family == "runs":
        for row in range(n):
            if rng.random() < 0.8:
                lo = int(rng.integers(0, n))
                mask[row, lo:min(n, lo + int(rng.integers(1, n + 1)))] = True
    elif family in ("full", "symmetric"):
        mask[:] = True
        if family == "symmetric" and n >= 3 and rng.random() < 0.5:
            mask[1:-1, 1:-1] = False
    return mask


def _clear_lines(mask: np.ndarray, mode: str, rng: np.random.Generator) -> None:
    n = mask.shape[0]
    if mode == "top":
        mask[0] = False
    elif mode == "bottom":
        mask[-1] = False
    elif mode == "left":
        mask[:, 0] = False
    elif mode == "right":
        mask[:, -1] = False
    elif mode == "top_two":
        mask[:2] = False
    elif mode == "bottom_two":
        mask[-2:] = False
    elif mode == "left_two":
        mask[:, :2] = False
    elif mode == "right_two":
        mask[:, -2:] = False
    elif mode == "internal_row":
        mask[int(rng.integers(1, n - 1))] = False
    elif mode == "internal_col":
        mask[:, int(rng.integers(1, n - 1))] = False
    elif mode == "cross":
        mask[0] = False
        mask[:, 0] = False


def _palette_target(rng: np.random.Generator, n: int) -> int:
    return int(rng.integers(1, min(9, n*n) + 1))


def _sample_tile(rng: np.random.Generator, n: int, background: int,
                 regime: str, first_train: bool, palette_target: int,
                 exact_palette: bool) -> np.ndarray:
    families = ("scatter", "stair", "strokes", "patch", "holes", "runs",
                "full", "symmetric")
    weights = (0.26, 0.14, 0.14, 0.12, 0.12, 0.10, 0.06, 0.06)
    colors = np.array([c for c in range(10) if c != background], dtype=np.int8)
    for _ in range(100):
        family = str(rng.choice(families, p=weights))
        mode = _blank_mode(rng, n, regime, first_train)
        if family == "full":
            mode = "none"
        mask = _sample_mask(rng, n, family)
        _clear_lines(mask, mode, rng)
        count = int(mask.sum())
        required_cells = palette_target if exact_palette else (2 if first_train else 1)
        if count < required_cells or (first_train and count == n*n):
            continue
        # At least one tile boundary cell must be visible. Rotating it makes
        # every edge of the complete pinwheel visible and its bounds tight.
        if not np.any(mask[0]) and not np.any(mask[-1]) and not np.any(mask[:, 0]) and not np.any(mask[:, -1]):
            continue
        palette_count = min(palette_target, count)
        if first_train:
            palette_count = max(2, palette_count)
        palette = rng.choice(colors, size=palette_count, replace=False)
        positions = np.flatnonzero(mask.ravel())
        labels = rng.integers(0, palette_count, size=count)
        labels[:palette_count] = np.arange(palette_count)
        rng.shuffle(labels)
        tile = np.full((n, n), background, dtype=np.int8)
        tile.flat[positions] = palette[labels]
        if _all_rotations_distinct(tile):
            return tile
    tile = np.full((n, n), background, dtype=np.int8)
    tile[0, 0], tile[-1, 0], tile[-1, -1] = colors[0], colors[1], colors[0]
    return tile


def _sample_canvas(rng: np.random.Generator, n: int) -> tuple[int, int, int, int]:
    minimum = 2*n + 2
    maximum_extra = 30 - minimum
    dimensions = []
    for _ in range(2):
        band = int(rng.choice(3, p=(0.45, 0.35, 0.20)))
        lo, hi = ((0, min(4, maximum_extra)),
                  (min(5, maximum_extra), min(12, maximum_extra)),
                  (min(13, maximum_extra), maximum_extra))[band]
        dimensions.append(minimum + int(rng.integers(lo, hi + 1)))
    height, width = dimensions
    top = int(rng.integers(1, height - 2*n))
    left = int(rng.integers(1, width - 2*n))
    return height, width, top, left


def _generate_pair(rng: np.random.Generator, n: int, background: int,
                   regime: str, first_train: bool, palette_target: int,
                   exact_palette: bool,
                   layout: tuple[int, int, int, int] | None = None) -> dict:
    tile = _sample_tile(rng, n, background, regime, first_train,
                        palette_target, exact_palette)
    height, width, top, left = layout if layout is not None else _sample_canvas(rng, n)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[top:top+2*n, left:left+2*n] = _compose_pinwheel(tile)
    return {"input": inp, "output": tile}


def _analyze_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    background = int(inp[0, 0])
    border = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    _require(np.all(border == background), "canvas border must be uniform")
    foreground = np.argwhere(inp != background)
    _require(len(foreground) > 0, "input must contain a pinwheel")
    r0, c0 = (int(x) for x in foreground.min(axis=0))
    r1, c1 = (int(x) for x in foreground.max(axis=0))
    side = r1 - r0 + 1
    _require(side == c1-c0+1 and side % 2 == 0 and side >= 4,
             "foreground needs tight even square bounds")
    n = side // 2
    _require(out.shape == (n, n), "output must be half the composite side")
    crop = inp[r0:r1+1, c0:c1+1]
    nw = crop[:n, :n]
    _require(np.array_equal(out, nw), "output must be the complete northwest tile")
    _require(np.array_equal(crop, _compose_pinwheel(nw)),
             "quadrants must be quarter-turn copies")
    _require(_all_rotations_distinct(nw), "four orientations must be distinguishable")
    _require(r0 >= 1 and c0 >= 1 and r1 < height-1 and c1 < width-1,
             "pinwheel needs a background margin")
    _require(np.any(crop[0] != background) and np.any(crop[-1] != background)
             and np.any(crop[:, 0] != background) and np.any(crop[:, -1] != background),
             "all four outer edges need visible evidence")
    colors = set(int(x) for x in np.unique(out)) - {background}
    mask = out != background
    return {"background": background, "tile_size": n,
            "bbox": (r0, r1, c0, c1), "palette_count": len(colors),
            "blank_rows": tuple(int(i) for i in np.flatnonzero(np.all(~mask, axis=1))),
            "blank_cols": tuple(int(i) for i in np.flatnonzero(np.all(~mask, axis=0))),
            "full": bool(np.all(mask)),
            "mask_symmetric": bool(np.array_equal(mask, np.rot90(mask))),
            "canvas": (height, width)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_integer_argument(seed) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # Northwest extraction and rotation order are the only puzzle-wide
    # conventions. Background, scale, and palette are visible in each input.
    constant_size = bool(rng.random() < 0.25)
    constant_background = bool(rng.random() < 0.25)
    size_anchor = _sample_size(rng)
    background_anchor = int(rng.integers(0, 10))
    palette_relation = str(rng.choice(("equal", "up", "down", "mixed")))
    sizes = []
    for index in range(train_count):
        if constant_size:
            n = size_anchor
        elif index == 0:
            n = int(rng.integers(3, 8))
        elif index == 1:
            n = int(rng.choice([v for v in range(2, 15) if v != sizes[0]]))
        else:
            n = _sample_size(rng)
        sizes.append(n)
    for _ in range(test_count):
        if constant_size:
            n = size_anchor
        elif rng.random() < 0.55:
            n = int(rng.choice(sizes[:train_count]))
        else:
            smallest, largest = min(sizes[:train_count]), max(sizes[:train_count])
            held_out = [v for v in range(2, 15) if v < smallest or v > largest]
            n = int(rng.choice(held_out)) if held_out else _sample_size(rng)
        sizes.append(n)
    backgrounds = []
    for index in range(train_count + test_count):
        if constant_background:
            background = background_anchor
        elif index == 0:
            background = background_anchor
        elif index == 1:
            background = int(rng.choice([c for c in range(10) if c != backgrounds[0]]))
        else:
            background = int(rng.integers(0, 10))
        backgrounds.append(background)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    saved_layouts = []
    for index, (n, background) in enumerate(zip(sizes, backgrounds)):
        regime = "source_train" if index < train_count else "target_test"
        split = "train" if index < train_count else "test"
        first_train = index == 0
        if palette_relation == "equal":
            palette_target = 2
        elif palette_relation == "up":
            palette_target = (int(rng.integers(2, 5)) if split == "train"
                              else int(rng.integers(min(5, n*n), min(9, n*n) + 1)))
        elif palette_relation == "down":
            palette_target = (int(rng.integers(5, min(9, n*n) + 1))
                              if split == "train" and n >= 3 else
                              int(rng.integers(1, min(3, n*n) + 1)))
        else:
            palette_target = _palette_target(rng, n)
        reused_layout = None
        if split == "test" and rng.random() < 0.35:
            candidates = [layout for size, layout in saved_layouts if size == n]
            if candidates:
                reused_layout = candidates[int(rng.integers(0, len(candidates)))]
        for attempt in range(16):
            layout = reused_layout if attempt < 8 else None
            pair = _generate_pair(rng, n, background, regime, first_train,
                                  palette_target, palette_relation == "equal", layout)
            key = _array_key(pair["input"])
            if key in seen_inputs or _is_source_pair(pair):
                continue
            try:
                info = _analyze_pair(pair)
            except AssertionError:
                continue
            seen_inputs.add(key)
            saved_layouts.append((n, (info["canvas"][0], info["canvas"][1],
                                      info["bbox"][0], info["bbox"][2])))
            puzzle[split].append(pair)
            break
        else:
            raise RuntimeError("could not construct a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{name} must be a two-dimensional NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape), "grid dimensions out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must use integers")
                _require(grid.min() >= 0 and grid.max() <= 9, "grid colors out of range")
            key, pair_key = _array_key(pair["input"]), _pair_key(pair)
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "input and output must differ")
            _require(not _is_source_pair(pair), "source pair copied")
            analyses[split].append(_analyze_pair(pair))
    _require(len({_array_key(p["output"]) for p in puzzle["train"]}) >= 2,
             "training needs distinct motifs")
    _require(any(np.any(p["output"] == a["background"])
                 for p, a in zip(puzzle["train"], analyses["train"])),
             "training must demonstrate retained background")


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[split]) == len(b[split]) and
               all(np.array_equal(x[name], y[name])
                   for x, y in zip(a[split], b[split]) for name in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    for rows in (("012", "023", "034"),
                 ("0000", "0000", "1203", "2431"),
                 ("123", "204", "341"),
                 ("123", "456", "789")):
        tile = _make_grid(rows)
        n = len(tile)
        inp = np.zeros((2*n+2, 2*n+2), dtype=np.int8)
        inp[1:-1, 1:-1] = _compose_pinwheel(tile)
        _analyze_pair({"input": inp, "output": tile})
    invisible = _make_grid(("000", "012", "034"))
    inp = np.zeros((8, 8), dtype=np.int8)
    inp[1:-1, 1:-1] = _compose_pinwheel(invisible)
    try:
        _analyze_pair({"input": inp, "output": invisible})
    except AssertionError:
        pass
    else:
        raise AssertionError("invisible outer ring was accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                r0, _, c0, _ = info["bbox"]
                n = info["tile_size"]
                _require(np.array_equal(pair["input"][r0:r0+n, c0:c0+n],
                                        pair["output"]), "northwest crop failed")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    all_pairs = [pair for p in puzzles for split in ("train", "test") for pair in p[split]]
    info = [_analyze_pair(pair) for pair in all_pairs]
    _require(len({_array_key(pair["input"]) for pair in all_pairs}) == len(all_pairs),
             "cross-seed duplicate input")
    _require(len({a["canvas"] for a in info}) >= 20, "canvas sizes lack diversity")
    _require(len({a["tile_size"] for a in info}) >= 10, "tile sizes lack diversity")
    _require(len({a["background"] for a in info}) == 10, "backgrounds lack diversity")
    _require(any(a["canvas"][0] != a["canvas"][1] for a in info), "no rectangles")
    _require(any(a["tile_size"] == 2 for a in info), "no compact tiles")
    _require(any(a["tile_size"] >= 11 for a in info), "no large tiles")
    _require(any(a["palette_count"] == 1 for a in info), "no single-color tiles")
    _require(any(a["palette_count"] >= 7 for a in info), "no rich palette")
    _require(any(a["full"] for a in info), "no fully colored tile")
    _require(any(a["mask_symmetric"] for a in info), "no symmetric occupancy mask")
    _require(any(a["blank_rows"] and a["blank_rows"][0] == 0 for a in info),
             "no blank top row")
    _require(any(a["blank_rows"] and a["blank_rows"][-1] == a["tile_size"]-1
                 for a in info), "no blank bottom row")
    _require(any(a["blank_cols"] and a["blank_cols"][0] == 0 for a in info),
             "no blank left column")
    _require(any(a["blank_cols"] and a["blank_cols"][-1] == a["tile_size"]-1
                 for a in info), "no blank right column")
    _require(any(len(a["blank_rows"]) >= 2 or len(a["blank_cols"]) >= 2 for a in info),
             "no multiple blank lines")
    _require(any(any(0 < r < a["tile_size"]-1 for r in a["blank_rows"]) or
                 any(0 < c < a["tile_size"]-1 for c in a["blank_cols"]) for a in info),
             "no internal blank line")
    _require(any(len({_analyze_pair(x)["tile_size"] for x in p["train"]}) > 1
                 for p in puzzles), "no within-puzzle size variation")
    _require(any(len({_analyze_pair(x)["background"] for x in p["train"]}) > 1
                 for p in puzzles), "no within-puzzle background variation")
    analyses = [([_analyze_pair(x) for x in p["train"]],
                 [_analyze_pair(x) for x in p["test"]]) for p in puzzles]
    _require(sum(len({a["background"] for a in train+test}) > 1
                 for train, test in analyses) >= 10, "too few varying-background puzzles")
    _require(sum(len({a["background"] for a in train+test}) == 1
                 for train, test in analyses) >= 3, "no constant-background regime")
    _require(sum(len({a["tile_size"] for a in train+test}) > 1
                 for train, test in analyses) >= 10, "too few varying-size puzzles")
    _require(sum(len({a["tile_size"] for a in train+test}) == 1
                 for train, test in analyses) >= 3, "no constant-size regime")
    _require(sum(len({(a["canvas"], a["bbox"]) for a in train+test}) < len(train+test)
                 for train, test in analyses) >= 3, "no repeated geometry with new motifs")
    _require(sum(any(a["tile_size"] in {b["tile_size"] for b in train}
                     for a in test) for train, test in analyses) >= 5,
             "tests lack familiar sizes")
    _require(sum(any(a["tile_size"] < min(b["tile_size"] for b in train)
                     for a in test) for train, test in analyses) >= 3,
             "tests lack held-out smaller sizes")
    _require(sum(any(a["tile_size"] > max(b["tile_size"] for b in train)
                     for a in test) for train, test in analyses) >= 3,
             "tests lack held-out larger sizes")
    palettes = [([a["palette_count"] for a in train],
                 [a["palette_count"] for a in test]) for train, test in analyses]
    _require(sum(len(set(a+b)) == 1 for a, b in palettes) >= 3,
             "equal-complexity palette regime missing")
    _require(sum(min(b) > max(a) for a, b in palettes) >= 2,
             "increasing-complexity palette regime missing")
    _require(sum(max(b) < min(a) for a, b in palettes) >= 2,
             "decreasing-complexity palette regime missing")
    _require(sum(min(b) < max(a) and max(b) > min(a) for a, b in palettes) >= 3,
             "mixed-complexity palette regime missing")
    _require(any(a["blank_rows"] or a["blank_cols"]
                 for train, _ in analyses for a in train),
             "training does not demonstrate blank lines")
    _require(any(not a["blank_rows"] and not a["blank_cols"]
                 for _, test in analyses for a in test),
             "tests always have blank lines")
    _require(any(a["tile_size"] >= 11 and a["canvas"][0] != a["canvas"][1]
                 and (a["blank_rows"] or a["blank_cols"])
                 for train, test in analyses for a in train+test),
             "size, aspect ratio, and blank-line changes never combine")
    _require(_puzzles_equal(generate_puzzle(117), generate_puzzle(117)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(903, 7, 3), generate_puzzle(903, 7, 3)),
             "explicit generation is not deterministic")
    for seed, train, test in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                              (3001, None, 1), (3002, None, 4),
                              (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count changed")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count changed")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(4001, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
