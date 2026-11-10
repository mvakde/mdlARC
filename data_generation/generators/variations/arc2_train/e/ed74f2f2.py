import numpy as np


# Each symbol is a binary drawing. Reflected drawings are separate symbols.
_GLYPHS = (
    ("111", "010", "010"),  # T
    ("110", "010", "011"),  # zigzag
    ("011", "010", "110"),  # reflected zigzag
    ("100", "100", "111"),
    ("001", "001", "111"),
    ("101", "111", "001"),
    ("101", "111", "100"),
    ("111", "100", "110"),
    ("111", "001", "011"),
    ("010", "111", "010"),
)

_PANEL_SHAPES = ((3, 3), (3, 3), (3, 3), (3, 3), (3, 4), (4, 3), (4, 4))

# Exact seven original input/output pairs, including the original test pair.
_SOURCE_ROWS = (
    (("000000000", "055505050", "005005550", "005005500", "000000000"),
     ("101", "111", "110")),
    (("000000000", "005505050", "005005050", "055005500", "000000000"),
     ("303", "303", "330")),
    (("000000000", "055505050", "005000550", "005005050", "000000000"),
     ("101", "011", "101")),
    (("000000000", "055005500", "005000550", "005500500", "000000000"),
     ("220", "022", "020")),
    (("000000000", "055005550", "005005050", "005505050", "000000000"),
     ("222", "202", "202")),
    (("000000000", "055005000", "005000550", "005505000", "000000000"),
     ("200", "022", "200")),
    (("000000000", "005505500", "005005550", "055005050", "000000000"),
     ("330", "333", "303")),
)
_SOURCE_PAIRS = frozenset(_SOURCE_ROWS)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _expanded_glyph(index, height, width):
    base = np.array([[int(cell) for cell in row] for row in _GLYPHS[index]], dtype=np.int8)
    rows = np.rint(np.linspace(0, 2, height)).astype(int)
    columns = np.rint(np.linspace(0, 2, width)).astype(int)
    return base[np.ix_(rows, columns)]


def _layout(rng):
    height, width = _PANEL_SHAPES[int(rng.integers(len(_PANEL_SHAPES)))]
    orientation = "horizontal" if int(rng.integers(4)) else "vertical"
    top, bottom, left, right, gap = (int(rng.integers(1, 4)) for _ in range(5))
    if orientation == "horizontal":
        shape = (top + height + bottom, left + 2 * width + gap + right)
        first = (top, left)
        second = (top, left + width + gap)
    else:
        shape = (top + 2 * height + gap + bottom, left + width + right)
        first = (top, left)
        second = (top + height + gap, left)
    return (orientation, height, width, top, bottom, left, right, gap, shape, first, second)


def _palette_plan(rng, output_colors, count):
    available = [color for color in range(10) if color not in output_colors]
    vary_background = bool(rng.integers(2))
    vary_ink = bool(rng.integers(2))
    allow_different_panel_inks = bool(rng.integers(2))
    fixed_background = int(rng.choice(available))
    ink_choices = [color for color in available if color != fixed_background]
    fixed_ink = int(rng.choice(ink_choices))
    result = []
    previous_background = None
    previous_ink = None
    for index in range(count):
        if vary_background:
            choices = [color for color in available if color != previous_background]
            background = int(rng.choice(choices))
        else:
            background = fixed_background
        ink_choices = [color for color in available if color != background]
        if vary_ink:
            choices = [color for color in ink_choices if color != previous_ink]
            selector_ink = int(rng.choice(choices))
        else:
            selector_ink = fixed_ink if fixed_ink != background else int(rng.choice(ink_choices))
        if allow_different_panel_inks and (index == 0 or bool(rng.integers(2))):
            payload_ink = int(rng.choice([color for color in ink_choices if color != selector_ink]))
        else:
            payload_ink = selector_ink
        result.append((background, selector_ink, payload_ink))
        previous_background, previous_ink = background, selector_ink
    return result


def sample_puzzle_parameters(rng, num_train, num_test):
    symbol_count = 2 if num_train <= 3 else 3
    glyphs = tuple(int(v) for v in rng.choice(len(_GLYPHS), symbol_count, replace=False))
    colors = tuple(int(v) for v in rng.choice(np.arange(1, 10), symbol_count, replace=False))
    layout = _layout(rng)
    train_symbols = [0, 1]
    if num_train >= 3:
        train_symbols.append(0)
    if symbol_count == 3:
        train_symbols.append(2)
    while len(train_symbols) < num_train:
        train_symbols.append(int(rng.integers(symbol_count)))
    # With three or more examples, the first two isolate the selector by
    # sharing a mask; the third isolates the mask by repeating a selector.
    test_symbols = [int(rng.integers(symbol_count))]
    while len(test_symbols) < num_test:
        if len(test_symbols) == 1:
            choices = [v for v in range(symbol_count) if v != test_symbols[0]]
            test_symbols.append(int(rng.choice(choices)))
        else:
            test_symbols.append(int(rng.integers(symbol_count)))
    palettes = _palette_plan(rng, colors, num_train + num_test)
    if num_train == 2:
        # At the minimum count, hold the input palette steady so glyph and
        # payload changes are the only differences the demonstrations teach.
        palettes = [palettes[0]] * (num_train + num_test)
    return dict(glyphs=glyphs, colors=colors, layout=layout,
                train_symbols=train_symbols, test_symbols=test_symbols,
                palettes=palettes)


def _mask_of_weight(rng, height, width, weight, forbidden, touch_edges=False):
    area = height * width
    def valid(mask):
        signature = tuple(int(v) for v in mask.ravel())
        if signature in forbidden:
            return False
        if touch_edges and not (mask[0].any() and mask[-1].any()
                                and mask[:, 0].any() and mask[:, -1].any()):
            return False
        return True
    for _ in range(64):
        cells = rng.choice(area, weight, replace=False)
        mask = np.zeros(area, dtype=np.int8)
        mask[cells] = 1
        mask = mask.reshape(height, width)
        if valid(mask):
            return mask
    # Complete deterministic fallback for rare collisions or edge constraints.
    for code in range(1, 1 << area):
        if code.bit_count() != weight:
            continue
        mask = np.array([(code >> bit) & 1 for bit in range(area)], dtype=np.int8)
        mask = mask.reshape(height, width)
        if valid(mask):
            return mask
    raise ValueError("no eligible payload mask")


def _fresh_mask(rng, height, width, desired_weight, forbidden):
    area = height * width
    weights = [desired_weight] + [weight for weight in rng.permutation(np.arange(1, area + 1))
                                  if int(weight) != desired_weight]
    for weight in weights:
        try:
            return _mask_of_weight(rng, height, width, int(weight), forbidden)
        except ValueError:
            pass
    raise ValueError("payload space exhausted")


def _make_pair(layout, glyph_index, output_color, mask, palette):
    _, height, width, _, _, _, _, _, shape, first, second = layout
    background, selector_ink, payload_ink = palette
    grid = np.full(shape, background, dtype=np.int8)
    glyph = _expanded_glyph(glyph_index, height, width)
    y, x = first
    grid[y:y + height, x:x + width][glyph != 0] = selector_ink
    y, x = second
    grid[y:y + height, x:x + width][mask != 0] = payload_ink
    output = np.where(mask != 0, output_color, background).astype(np.int8)
    return {"input": grid, "output": output}


def _source_signature(pair):
    return (tuple("".join(str(int(v)) for v in row) for row in pair["input"]),
            tuple("".join(str(int(v)) for v in row) for row in pair["output"]))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    params = sample_puzzle_parameters(rng, num_train, num_test)
    layout = params["layout"]
    height, width = layout[1:3]
    area = height * width
    base_weight = int(rng.integers(4, max(5, min(area, (3 * area) // 4) + 1)))
    base_mask = _mask_of_weight(rng, height, width, base_weight, set(), touch_edges=True)
    base_signature = tuple(int(v) for v in base_mask.ravel())
    masks = [base_mask]
    used_masks = {base_signature}
    if num_train == 2:
        # A minimum-size demonstration must show two different geometries.
        second = _fresh_mask(rng, height, width, base_weight, used_masks)
        masks.append(second)
        used_masks.add(tuple(int(v) for v in second.ravel()))
    else:
        masks.append(base_mask)
    for _ in range(2, num_train):
        weight = int(rng.integers(1, area + 1))
        mask = _fresh_mask(rng, height, width, weight, used_masks)
        used_masks.add(tuple(int(v) for v in mask.ravel()))
        masks.append(mask)
    train_weights = [int(mask.sum()) for mask in masks]
    for _ in range(num_test):
        if bool(rng.integers(5)):
            desired = int(rng.choice(train_weights))
        else:
            desired = int(rng.integers(1, area + 1))
        mask = _fresh_mask(rng, height, width, desired, used_masks)
        used_masks.add(tuple(int(v) for v in mask.ravel()))
        masks.append(mask)
    puzzle = {"train": [], "test": []}
    symbols = params["train_symbols"] + params["test_symbols"]
    for index, (symbol, mask) in enumerate(zip(symbols, masks)):
        split = "train" if index < num_train else "test"
        pair = _make_pair(layout, params["glyphs"][symbol], params["colors"][symbol],
                          mask, params["palettes"][index])
        # Exact source copies are the only source-based exclusion.
        rejected = set()
        while _source_signature(pair) in _SOURCE_PAIRS:
            rejected.add(tuple(int(v) for v in mask.ravel()))
            replacement = _fresh_mask(rng, height, width, int(mask.sum()), used_masks | rejected)
            pair = _make_pair(layout, params["glyphs"][symbol], params["colors"][symbol],
                              replacement, params["palettes"][index])
            mask = replacement
            used_masks.add(tuple(int(v) for v in replacement.ravel()))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.ravel()))


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _require(np.all((grid >= 0) & (grid <= 9)), f"{label} has an illegal color")


def _candidate_layouts(input_shape, output_shape):
    height, width = output_shape
    for orientation in ("horizontal", "vertical"):
        for top in range(1, 4):
            for left in range(1, 4):
                for gap in range(1, 4):
                    if orientation == "horizontal":
                        bottom = input_shape[0] - top - height
                        right = input_shape[1] - left - 2 * width - gap
                        first, second = (top, left), (top, left + width + gap)
                    else:
                        bottom = input_shape[0] - top - 2 * height - gap
                        right = input_shape[1] - left - width
                        first, second = (top, left), (top + height + gap, left)
                    if 1 <= bottom <= 3 and 1 <= right <= 3:
                        yield (orientation, height, width, top, bottom, left, right,
                               gap, input_shape, first, second)


def _decode_pair(pair, layout):
    input_grid, output = pair["input"], pair["output"]
    _, height, width, _, _, _, _, _, shape, first, second = layout
    if input_grid.shape != shape or output.shape != (height, width):
        return None
    background = int(input_grid[0, 0])
    covered = np.zeros(shape, dtype=bool)
    panels = []
    for y, x in (first, second):
        covered[y:y + height, x:x + width] = True
        panel = input_grid[y:y + height, x:x + width]
        inks = set(int(v) for v in np.unique(panel)) - {background}
        if len(inks) != 1:
            return None
        panels.append((panel != background, next(iter(inks))))
    if np.any(input_grid[~covered] != background):
        return None
    selector, selector_ink = panels[0]
    payload, payload_ink = panels[1]
    if not payload.any():
        return None
    valid_glyphs = [tuple(int(v) for v in _expanded_glyph(i, height, width).ravel())
                    for i in range(len(_GLYPHS))]
    selector_signature = tuple(int(v) for v in selector.ravel())
    if selector_signature not in valid_glyphs:
        return None
    output_colors = set(int(v) for v in np.unique(output)) - {background}
    if len(output_colors) != 1:
        return None
    color = next(iter(output_colors))
    if np.any(output[payload] != color) or np.any(output[~payload] != background):
        return None
    return dict(selector=selector_signature, mask=tuple(int(v) for v in payload.ravel()),
                color=color, background=background, selector_ink=selector_ink,
                payload_ink=payload_ink, weight=int(payload.sum()))


def _decode_puzzle(puzzle):
    pairs = puzzle["train"] + puzzle["test"]
    input_shape = pairs[0]["input"].shape
    output_shape = pairs[0]["output"].shape
    for layout in _candidate_layouts(input_shape, output_shape):
        infos = [_decode_pair(pair, layout) for pair in pairs]
        if all(info is not None for info in infos):
            return layout, infos
    raise AssertionError("no shared panel layout gives an exact payload crop")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal train/test counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must have exactly input and output")
            _validate_grid(pair["input"], label + ".input")
            _validate_grid(pair["output"], label + ".output")
            input_signature = _grid_signature(pair["input"])
            pair_signature = (input_signature, _grid_signature(pair["output"]))
            _require(input_signature not in seen_inputs and pair_signature not in seen_pairs,
                     "duplicate input or pair")
            _require(_source_signature(pair) not in _SOURCE_PAIRS, "exact source pair copied")
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
    layout, infos = _decode_puzzle(puzzle)
    train, test = infos[:train_count], infos[train_count:]
    learned = {}
    for info in train:
        old = learned.get(info["selector"])
        _require(old is None or old == info["color"], "contradictory selector colors")
        learned[info["selector"]] = info["color"]
    _require(len(learned) >= 2 and len(set(learned.values())) == len(learned),
             "training must show distinct symbol colors")
    if train_count >= 4:
        _require(len(learned) >= 3, "training lacks a third symbol")
    for info in test:
        _require(info["selector"] in learned and learned[info["selector"]] == info["color"],
                 "test uses an unlearned or changed selector mapping")
    if test_count >= 2:
        _require(len({info["selector"] for info in test}) >= 2,
                 "multi-test puzzle does not exercise two selectors")
    train_masks = {info["mask"] for info in train}
    _require(all(info["mask"] not in train_masks for info in test),
             "test must combine a learned selector with a new payload")
    all_masks = [info["mask"] for info in train]
    _require(len(set(all_masks)) >= min(2, train_count - 1) or train_count == 2,
             "training does not show varying payloads")
    height, width = layout[1:3]
    union = np.zeros((height, width), dtype=bool)
    for info in train:
        union |= np.asarray(info["mask"], dtype=bool).reshape(height, width)
    _require(union[0].any() and union[-1].any() and union[:, 0].any() and union[:, -1].any(),
             "training does not establish the full output crop")


def _puzzle_signature(puzzle):
    return tuple((split, tuple((_grid_signature(p["input"]), _grid_signature(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (8, 2)):
        puzzle = generate_puzzle(31000 + train * 10 + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit count not honored")
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(32000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train, "partial train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "partial test count ignored")
        validate_generated_puzzle(puzzle)
    for arguments in (dict(num_train=1), dict(num_train=11), dict(num_test=0),
                      dict(num_test=5), dict(num_train=2, num_test=2),
                      dict(num_train=3, num_test=4), dict(num_train=True),
                      dict(num_train=4.0)):
        try:
            generate_puzzle(33000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal request accepted: {arguments}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(_puzzle_signature(puzzles[17]) == _puzzle_signature(generate_puzzle(17)),
             "automatic generation is not deterministic")
    _require(_puzzle_signature(generate_puzzle(123, 7, 3))
             == _puzzle_signature(generate_puzzle(123, 7, 3)),
             "explicit generation is not deterministic")
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40,
             "insufficient distinct puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts do not vary")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts do not vary")
    layouts, colors, backgrounds, inks, train_weights, test_weights = set(), set(), set(), set(), set(), set()
    glyphs, masks = set(), set()
    multi_color_tests = shared_train_masks = repeated_symbol_masks = changed_background = 0
    different_panel_inks = overlapping_density = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        layout, infos = _decode_puzzle(puzzle)
        layouts.add(layout[:8])
        train = infos[:len(puzzle["train"])]
        test = infos[len(puzzle["train"]):]
        if len({v["color"] for v in test}) >= 2:
            multi_color_tests += 1
        if len({v["background"] for v in infos}) >= 2:
            changed_background += 1
        if any(v["selector_ink"] != v["payload_ink"] for v in infos):
            different_panel_inks += 1
        if {v["weight"] for v in train} & {v["weight"] for v in test}:
            overlapping_density += 1
        if any(a["mask"] == b["mask"] and a["selector"] != b["selector"]
               for i, a in enumerate(train) for b in train[i + 1:]):
            shared_train_masks += 1
        if any(a["selector"] == b["selector"] and a["mask"] != b["mask"]
               for i, a in enumerate(train) for b in train[i + 1:]):
            repeated_symbol_masks += 1
        for v in train:
            train_weights.add(v["weight"])
        for v in test:
            test_weights.add(v["weight"])
        for v in infos:
            colors.add(v["color"])
            backgrounds.add(v["background"])
            inks.update((v["selector_ink"], v["payload_ink"]))
            glyphs.add(v["selector"])
            masks.add(v["mask"])
    _require(len(layouts) >= 20 and len(glyphs) >= 8 and len(masks) >= 80,
             "layout, glyph, or payload structure lacks diversity")
    _require(len(colors) >= 7 and len(backgrounds) >= 7 and len(inks) >= 7,
             "palette roles lack diversity")
    _require(multi_color_tests >= 8 and shared_train_masks >= 20
             and repeated_symbol_masks >= 15, "selector/mask controls lack demonstrations")
    _require(changed_background >= 10 and different_panel_inks >= 10,
             "input palette roles do not vary")
    _require(overlapping_density >= 15 and len(train_weights) >= 8 and len(test_weights) >= 8,
             "train/test payload densities are too narrow")
