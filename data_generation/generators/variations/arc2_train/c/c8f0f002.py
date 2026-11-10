"""Generate ARC puzzles whose sole rule is one global color substitution."""

import numpy as np


_ORIGINAL_SOURCE_PAIRS = (
    (((7, 7, 7, 1), (1, 8, 1, 7), (7, 1, 1, 7)),
     ((5, 5, 5, 1), (1, 8, 1, 5), (5, 1, 1, 5))),
    (((1, 8, 8, 7, 7, 8), (1, 1, 7, 7, 1, 8), (7, 1, 1, 7, 7, 8)),
     ((1, 8, 8, 5, 5, 8), (1, 1, 5, 5, 1, 8), (5, 1, 1, 5, 5, 8))),
    (((1, 8, 1, 7, 1), (7, 8, 8, 1, 1), (7, 1, 8, 8, 7)),
     ((1, 8, 1, 5, 1), (5, 8, 8, 1, 1), (5, 1, 8, 8, 5))),
    (((1, 7, 7, 1, 7), (8, 1, 7, 7, 7), (8, 7, 1, 7, 8)),
     ((1, 5, 5, 1, 5), (8, 1, 5, 5, 5), (8, 5, 1, 5, 8))),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _rows(grid: np.ndarray) -> tuple:
    return tuple(tuple(int(x) for x in row) for row in grid)


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    count = 0
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            count += 1
            seen[row, col] = True
            stack = [(row, col)]
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
    return count


def _normalize_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, rng: np.random.Generator,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count("num_train", num_train, 2, 10)
    test = _normalize_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        preferred = (2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 10)
        train = max(test + 1, preferred[(seed // 4) % len(preferred)])
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_convention(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    source, replacement = colors[:2]
    pool_size = int(rng.choice((1, 2, 3, 4, 5), p=(.13, .19, .29, .25, .14)))
    protected = colors[2:2 + pool_size]
    held_out = protected[-1] if pool_size >= 2 and rng.random() < .30 else None
    return {
        "source": source,
        "replacement": replacement,
        "protected": protected,
        "held_out": held_out,
        "merge": bool(rng.random() < .37),
    }


_SIZES = np.array((1, 2, 3, 4, 5, 6, 7, 8, 10, 12))
_SIZE_WEIGHTS = np.array((.07, .09, .14, .15, .15, .13, .10, .08, .05, .04))


def _sample_shape(rng: np.random.Generator, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    for _ in range(48):
        height = int(rng.choice(_SIZES, p=_SIZE_WEIGHTS))
        width = int(rng.choice(_SIZES, p=_SIZE_WEIGHTS))
        if height * width >= 4 and (height, width) not in forbidden:
            return height, width
    for height in _SIZES:
        for width in _SIZES:
            shape = (int(height), int(width))
            if height * width >= 4 and shape not in forbidden:
                return shape
    raise AssertionError("no available shape")


def _choose_protected(rng: np.random.Generator, convention: dict,
                      regime: str, index: int) -> list[int]:
    anchor = convention["protected"][0]
    extras = [color for color in convention["protected"][1:]
              if regime == "target_test" or color != convention["held_out"]]
    if convention["merge"]:
        extras.append(convention["replacement"])
    if regime == "source_train" and index == 0:
        chosen = [anchor]
        if convention["merge"]:
            chosen.append(convention["replacement"])
        return chosen
    if regime == "source_train" and index == 1:
        # A complementary demonstration varies the input palette when possible.
        non_merge = [color for color in extras if color != convention["replacement"]]
        return [anchor] + ([non_merge[0]] if non_merge else [])
    chosen = [anchor]
    for color in extras:
        if rng.random() < .48:
            chosen.append(color)
    if regime == "target_test" and index == 0 and convention["held_out"] is not None:
        if convention["held_out"] not in chosen:
            chosen.append(convention["held_out"])
    return chosen[:5]


def _source_count(rng: np.random.Generator, total: int, first_train: bool) -> int:
    modes = ("single", "sparse", "middle", "dominant")
    probs = (.16, .24, .32, .28)
    mode = str(rng.choice(modes, p=probs))
    if first_train and mode == "single":
        mode = "middle"
    if mode == "single":
        return 1
    if mode == "sparse":
        upper = max(2, int(np.ceil(.20 * total)))
        return min(total - 1, int(rng.integers(2, upper + 1)))
    if mode == "middle":
        low = max(2, int(np.ceil(.25 * total)))
        high = min(total - 1, max(low, int(np.floor(.62 * total))))
        return int(rng.integers(low, high + 1))
    low = max(2, int(np.ceil(.66 * total)))
    high = min(total - 1, max(low, int(np.floor(.91 * total))))
    return int(rng.integers(min(low, total - 1), high + 1))


def _scatter_mask(rng: np.random.Generator, shape: tuple[int, int], count: int) -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    mask[rng.choice(mask.size, size=count, replace=False)] = True
    return mask.reshape(shape)


def _connected_mask(rng: np.random.Generator, shape: tuple[int, int],
                    count: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    start = (int(rng.integers(height)), int(rng.integers(width)))
    mask[start] = True
    cells = [start]
    while len(cells) < count:
        frontier = set()
        for row, col in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        options = sorted(frontier)
        selected = options[int(rng.integers(len(options)))]
        mask[selected] = True
        cells.append(selected)
    return mask


def _source_mask(rng: np.random.Generator, shape: tuple[int, int],
                 count: int, preferred: str | None = None) -> np.ndarray:
    height, width = shape
    mode = preferred or str(rng.choice(
        ("scatter", "connected", "bar", "isolated", "groups", "interior"),
        p=(.18, .20, .17, .17, .16, .12),
    ))
    if mode == "connected":
        return _connected_mask(rng, shape, count)
    if mode == "bar":
        orientations = [axis for axis, extent in enumerate(shape) if count <= extent]
        if orientations:
            axis = orientations[int(rng.integers(len(orientations)))]
            mask = np.zeros(shape, dtype=bool)
            if axis == 0:
                col = int(rng.integers(width))
                row = int(rng.integers(height - count + 1))
                mask[row:row + count, col] = True
            else:
                row = int(rng.integers(height))
                col = int(rng.integers(width - count + 1))
                mask[row, col:col + count] = True
            return mask
        return _connected_mask(rng, shape, count)
    if mode == "isolated":
        parity = int(rng.integers(2))
        candidates = [(r, c) for r in range(height) for c in range(width)
                      if (r + c) % 2 == parity]
        if count <= len(candidates):
            mask = np.zeros(shape, dtype=bool)
            for index in rng.choice(len(candidates), size=count, replace=False):
                mask[candidates[int(index)]] = True
            return mask
    if mode == "interior" and height >= 3 and width >= 3:
        candidates = [(r, c) for r in range(1, height - 1)
                      for c in range(1, width - 1)]
        if count <= len(candidates):
            mask = np.zeros(shape, dtype=bool)
            for index in rng.choice(len(candidates), size=count, replace=False):
                mask[candidates[int(index)]] = True
            return mask
    if mode == "groups" and 2 <= count <= (height * width) // 2:
        for _ in range(40):
            mask = _scatter_mask(rng, shape, count)
            if _component_count(mask) >= 2:
                return mask
    return _scatter_mask(rng, shape, count)


def _protected_field(rng: np.random.Generator, mask: np.ndarray,
                     palette: list[int]) -> np.ndarray:
    height, width = mask.shape
    mode = str(rng.choice(
        ("random", "rows", "columns", "checker", "patch", "singleton"),
        p=(.30, .15, .15, .15, .13, .12),
    ))
    grid = np.full(mask.shape, palette[0], dtype=np.int8)
    if len(palette) == 1:
        return grid
    if mode == "random":
        weights = rng.dirichlet(np.ones(len(palette)))
        grid[:, :] = rng.choice(palette, size=mask.shape, p=weights)
    elif mode == "rows":
        offset = int(rng.integers(len(palette)))
        for row in range(height):
            grid[row, :] = palette[(row + offset) % len(palette)]
    elif mode == "columns":
        offset = int(rng.integers(len(palette)))
        for col in range(width):
            grid[:, col] = palette[(col + offset) % len(palette)]
    elif mode == "checker":
        offset = int(rng.integers(len(palette)))
        for row in range(height):
            for col in range(width):
                grid[row, col] = palette[(row + col + offset) % len(palette)]
    elif mode == "patch":
        row0, row1 = sorted(int(x) for x in rng.integers(0, height + 1, size=2))
        col0, col1 = sorted(int(x) for x in rng.integers(0, width + 1, size=2))
        grid[row0:row1 + 1, col0:col1 + 1] = palette[1]
    free = np.flatnonzero(~mask)
    if mode == "singleton":
        grid[:, :] = palette[0]
        chosen = int(rng.choice(free))
        grid.flat[chosen] = palette[int(rng.integers(1, len(palette)))]
    elif len(free) >= len(palette):
        # Materialize the selected palette, even when a pattern hides a color.
        positions = rng.choice(free, size=len(palette), replace=False)
        for position, color in zip(positions, palette):
            grid.flat[int(position)] = color
    return grid


def _draw_pair(rng: np.random.Generator, shape: tuple[int, int],
               convention: dict, regime: str, index: int,
               used_inputs: set) -> dict:
    count = _source_count(rng, shape[0] * shape[1],
                          regime == "source_train" and index == 0)
    if (regime == "source_train" and index == 0 and convention["merge"]) or (
        regime == "target_test" and index == 0 and convention["held_out"] is not None
    ):
        count = min(count, shape[0] * shape[1] - 2)
    palette = _choose_protected(rng, convention, regime, index)
    preferred = None
    if regime == "source_train" and index == 0:
        preferred = "connected"
    elif regime == "source_train" and index == 1:
        preferred = "isolated"
    for attempt in range(100):
        mask = _source_mask(rng, shape, count, preferred if attempt < 20 else None)
        input_grid = _protected_field(rng, mask, palette)
        mandatory = None
        if regime == "source_train" and index == 0 and convention["merge"]:
            mandatory = convention["replacement"]
        if regime == "target_test" and index == 0 and convention["held_out"] is not None:
            mandatory = convention["held_out"]
        if mandatory is not None:
            free = np.flatnonzero(~mask)
            input_grid.flat[int(rng.choice(free))] = mandatory
        input_grid[mask] = convention["source"]
        key = _grid_key(input_grid)
        if key in used_inputs:
            continue
        if any(_rows(input_grid) == original_input
               for original_input, _ in _ORIGINAL_SOURCE_PAIRS):
            continue
        output_grid = input_grid.copy()
        output_grid[mask] = convention["replacement"]
        used_inputs.add(key)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    convention = _sample_convention(rng)
    shapes = []
    for index in range(train_count):
        forbidden = set(shapes[:1]) if index == 1 else set()
        shapes.append(_sample_shape(rng, forbidden))
    for index in range(test_count):
        if rng.random() < .35:
            shape = shapes[int(rng.integers(train_count))]
        else:
            shape = _sample_shape(rng, set(shapes) if rng.random() < .7 else set())
        shapes.append(shape)
    entropies = rng.integers(0, 2**63 - 1, size=train_count + test_count)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for index, shape in enumerate(shapes):
        regime = "source_train" if index < train_count else "target_test"
        local_index = index if regime == "source_train" else index - train_count
        child_rng = np.random.default_rng(
            np.random.SeedSequence([int(entropies[index]), index])
        )
        pair = _draw_pair(child_rng, shape, convention, regime, local_index, used_inputs)
        puzzle["train" if regime == "source_train" else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_mapping(train: list[dict]) -> tuple[int, int]:
    sources = set()
    replacements = set()
    for pair in train:
        changed = pair["input"] != pair["output"]
        _require(bool(np.any(changed)), "training pair has no changed cell")
        sources.update(int(x) for x in np.unique(pair["input"][changed]))
        replacements.update(int(x) for x in np.unique(pair["output"][changed]))
    _require(len(sources) == 1 and len(replacements) == 1,
             "training must teach one unambiguous mapping")
    source, replacement = next(iter(sources)), next(iter(replacements))
    _require(source != replacement, "source and replacement coincide")
    return source, replacement


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    all_pairs = train + test
    for pair in all_pairs:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must have exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grid must be a two-dimensional NumPy array")
            _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
            _require(all(1 <= size <= 30 for size in grid.shape),
                     "grid dimensions are out of range")
            _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                     "grid color is out of range")
        inp, out = pair["input"], pair["output"]
        _require(inp.shape == out.shape, "grid shape changed")
        input_key = _grid_key(inp)
        pair_key = (input_key, _grid_key(out))
        _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                 "duplicate generated pair")
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        _require(not any(_rows(inp) == original_input
                         for original_input, _ in _ORIGINAL_SOURCE_PAIRS),
                 "copied an original source input")
    source, replacement = _infer_mapping(train)
    _require(any(np.count_nonzero(pair["input"] == source) >= 2 for pair in train),
             "training does not demonstrate a repeated source color")
    _require(len({pair["input"].shape for pair in train}) >= 2,
             "training shapes do not vary")
    for pair in all_pairs:
        inp, out = pair["input"], pair["output"]
        source_mask = inp == source
        _require(bool(np.any(source_mask)), "source is absent from an input")
        _require(bool(np.any(~source_mask)), "input has no protected color")
        expected = inp.copy()
        expected[source_mask] = replacement
        _require(np.array_equal(out, expected),
                 "pair violates exact global substitution")
    # The mapping is inferred exclusively from training, then predicts each test.
    for pair in test:
        prediction = pair["input"].copy()
        prediction[prediction == source] = replacement
        _require(np.array_equal(prediction, pair["output"]),
                 "test cannot be predicted from the training mapping")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "too few automatic training counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "too few automatic test counts")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40,
             "too few distinct puzzles")
    _require(_puzzle_key(generate_puzzle(23)) == _puzzle_key(generate_puzzle(23)),
             "automatic generation is not deterministic")
    _require(_puzzle_key(generate_puzzle(314159, 7, 3)) ==
             _puzzle_key(generate_puzzle(314159, 7, 3)),
             "explicit generation is not deterministic")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + train_count * 10 + test_count,
                                 train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts were ignored")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "partial train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "partial test count ignored")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(99, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")

    shapes, train_densities, test_densities = set(), set(), set()
    palette_sizes, mappings, layouts = set(), set(), set()
    merge_puzzles = novel_test_puzzles = singleton_pairs = 0
    connected = disconnected = interior = bars = 0
    new_test_shapes = familiar_test_shapes = varied_palettes = 0
    isolated_groups = tall_dense_multicolor = square_disconnected = 0
    split_shapes = {"train": set(), "test": set()}
    for puzzle in puzzles:
        source, replacement = _infer_mapping(puzzle["train"])
        mappings.add((source, replacement))
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        test_shapes = {pair["input"].shape for pair in puzzle["test"]}
        new_test_shapes += bool(test_shapes - train_shapes)
        familiar_test_shapes += bool(test_shapes & train_shapes)
        train_palette = set()
        for pair in puzzle["train"]:
            train_palette.update(int(x) for x in np.unique(pair["input"]))
        if replacement in train_palette:
            merge_puzzles += 1
        test_colors = {int(x) for pair in puzzle["test"]
                       for x in np.unique(pair["input"])}
        if test_colors - train_palette:
            novel_test_puzzles += 1
        puzzle_palette_sizes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                mask = inp == source
                n = inp.size
                density = int(np.count_nonzero(mask)) / n
                (train_densities if split == "train" else test_densities).add(
                    "single" if np.count_nonzero(mask) == 1 else
                    "sparse" if density < .25 else
                    "middle" if density < .65 else "dominant")
                shapes.add(inp.shape)
                split_shapes[split].add(inp.shape)
                palette_sizes.add(len(np.unique(inp)) - 1)
                puzzle_palette_sizes.add(len(np.unique(inp)) - 1)
                layout = (inp.shape, tuple(int(x) for x in mask.flat))
                layouts.add(layout)
                components = _component_count(mask)
                connected += components == 1
                disconnected += components >= 2
                isolated_groups += (np.count_nonzero(mask) >= 2 and
                                    components == np.count_nonzero(mask))
                tall_dense_multicolor += (inp.shape[0] > inp.shape[1]
                                          and density >= .65
                                          and len(np.unique(inp)) >= 3)
                square_disconnected += (inp.shape[0] == inp.shape[1]
                                        and components >= 2
                                        and len(np.unique(inp)) >= 3)
                if inp.shape[0] >= 3 and inp.shape[1] >= 3:
                    inner = np.zeros(inp.shape, dtype=bool)
                    inner[1:-1, 1:-1] = True
                    interior += bool(np.all(~mask | inner))
                rows, cols = np.where(mask)
                bars += bool(len(rows) >= 2 and
                             (len(set(int(x) for x in rows)) == 1 or
                              len(set(int(x) for x in cols)) == 1) and
                             components == 1)
                protected = inp[~mask]
                singleton_pairs += any(np.count_nonzero(protected == color) == 1
                                       for color in np.unique(protected))
                expected = inp.copy()
                expected[mask] = replacement
                _require(np.array_equal(expected, pair["output"]),
                         "self-test found a wrong output")
        varied_palettes += len(puzzle_palette_sizes) >= 2
    _require(len(mappings) >= 20, "color mappings lack diversity")
    _require(len(shapes) >= 30 and any(h == 1 for h, _ in shapes)
             and any(w == 1 for _, w in shapes), "shapes lack diversity")
    for split in ("train", "test"):
        observed = split_shapes[split]
        _require(any(h == 1 for h, _ in observed)
                 and any(w == 1 for _, w in observed)
                 and any(h == w for h, w in observed)
                 and any(h > w for h, w in observed)
                 and any(h < w for h, w in observed),
                 f"{split} lacks shape regimes")
    _require(new_test_shapes >= 12 and familiar_test_shapes >= 8,
             "tests lack a mix of familiar and new dimensions")
    _require({"single", "sparse", "middle", "dominant"} <= train_densities
             and {"single", "sparse", "middle", "dominant"} <= test_densities,
             "source density is tied to the split")
    _require({1, 2, 3} <= palette_sizes, "protected palette lacks breadth")
    _require(merge_puzzles >= 5 and novel_test_puzzles >= 4,
             "palette extensions lack breadth")
    _require(varied_palettes >= 20, "protected palette is too fixed within puzzles")
    _require(singleton_pairs >= 8, "protected singletons are too rare")
    _require(connected >= 30 and disconnected >= 30 and
             interior >= 3 and bars >= 10, "source layouts lack breadth")
    _require(isolated_groups >= 10 and tall_dense_multicolor >= 5
             and square_disconnected >= 3,
             "spatial, density, and palette axes do not combine")
    _require(len(layouts) >= 150, "source geometry repeats excessively")
