"""Connected scaffolds deformed by a bottom-anchored four-row wave."""

import numpy as np


_SOURCE_INPUTS = (
    "000000000000000/000000000000000/000055555555000/000050055005000/000050055005000/000055555555000/000050055005000/000050055005000/000055555555000/000050055005000/000050055005000/000055555555000/000000000000000/000000000000000/000000000000000",
    "00000000000000/00000000000000/00000000000000/00022222200000/00020000200000/00020000200000/00020000200000/00020000200000/00020000200000/00020000200000/00020000200000/00020000200000/00020000200000/00022222200000/00000000000000",
    "000000000000000/000000000000000/000088888000000/000080808000000/000080808000000/000080808000000/000080808000000/000080808000000/000080808000000/000080808000000/000080808000000/000080808000000/000080808000000/000088888000000/000000000000000",
    "00000000000000/00000000000000/00000000000000/00000000000000/00033333333300/00030003000300/00030003000300/00033333333300/00030003000300/00030003000300/00033333333300/00000000000000/00000000000000",
    "000000000000000/000077777770000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000070070070000/000077777770000/000000000000000",
)


def _check(condition, message):
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


def _offset(distance):
    return (0, -1, 0, 1)[distance % 4]


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_key(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _crop_signature(grid):
    bg = int(grid[0, 0])
    rows, cols = np.where(grid != bg)
    crop = grid[int(rows.min()):int(rows.max()) + 1,
                int(cols.min()):int(cols.max()) + 1]
    return (crop.shape, (crop != bg).tobytes())


def _validate_grid(grid):
    _check(isinstance(grid, np.ndarray), "grid must be an ndarray")
    _check(grid.ndim == 2, "grid must be two-dimensional")
    _check(all(1 <= size <= 30 for size in grid.shape), "grid size out of bounds")
    _check(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
    _check(np.all((grid >= 0) & (grid <= 9)), "grid color out of bounds")


def _analyze(grid):
    """Recover color roles and scaffold geometry from the grid."""
    bg = int(grid[0, 0])
    colors = np.unique(grid)
    _check(len(colors) == 2 and bg in colors, "one foreground and one background required")
    fg = int(colors[colors != bg][0])
    mask = grid == fg
    rows, cols = np.where(mask)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    _check(left >= 1 and right < grid.shape[1] - 1, "wave needs lateral margin")
    _check(bottom - top + 1 >= 5, "scaffold too short")
    _check(np.all(mask[top:bottom + 1].any(axis=1)), "empty scaffold row")
    _check(np.all(mask[top, left:right + 1]), "top bar incomplete")
    _check(np.all(mask[bottom, left:right + 1]), "bottom bar incomplete")
    core = mask[top + 1:bottom, left:right + 1]
    rail_mask = np.all(core, axis=0)
    _check(bool(rail_mask[0]) and bool(rail_mask[-1]), "outer supports missing")
    _check(sum(np.array_equal(row, rail_mask) for row in core) >= 2,
           "at least two open rail rows required")
    starts = np.flatnonzero(rail_mask & ~np.r_[False, rail_mask[:-1]])
    ends = np.flatnonzero(rail_mask & ~np.r_[rail_mask[1:], False])
    supports = tuple((int(a), int(b)) for a, b in zip(starts, ends))
    _check(2 <= len(supports) <= 6, "unsupported support count")
    gaps = tuple(supports[i + 1][0] - supports[i][1] - 1
                 for i in range(len(supports) - 1))
    _check(all(gap >= 1 for gap in gaps), "supports need visible openings")
    bars = []
    partial = 0
    for row_index, row in enumerate(core, start=top + 1):
        _check(np.all(row[rail_mask]), "interrupted vertical support")
        filled = []
        for i in range(len(supports) - 1):
            a, b = supports[i][1] + 1, supports[i + 1][0]
            segment = row[a:b]
            _check(np.all(segment) or not np.any(segment), "broken crossbar")
            filled.append(bool(np.all(segment)))
        if any(filled):
            bars.append(row_index)
            partial += int(not all(filled))
    _check(np.all(~mask[:top]) and np.all(~mask[bottom + 1:]),
           "extra object outside scaffold")
    return dict(bg=bg, fg=fg, top=top, bottom=bottom, left=left, right=right,
                height=bottom - top + 1, width=right - left + 1,
                supports=supports, gaps=gaps, bars=tuple(bars), partial=partial,
                top_margin=top, bottom_margin=grid.shape[0] - bottom - 1,
                left_margin=left, right_margin=grid.shape[1] - right - 1)


def _wave(input_grid):
    info = _analyze(input_grid)
    output = np.full_like(input_grid, info["bg"])
    for row in range(info["top"], info["bottom"] + 1):
        columns = np.flatnonzero(input_grid[row] == info["fg"])
        shifted = columns + _offset(info["bottom"] - row)
        _check(shifted.min() >= 0 and shifted.max() < input_grid.shape[1],
               "clipped wave")
        output[row, shifted] = info["fg"]
    return output


def _sample_supports(rng, forced_count=None, compact=False):
    count = forced_count if forced_count is not None else int(rng.integers(2, 7))
    if rng.random() < 0.28:
        widths = [1] * count
    else:
        ceiling = 3 if compact else 4
        widths = [int(rng.integers(1, ceiling)) for _ in range(count)]
    gaps = []
    available = 28 - sum(widths) - (count - 1)
    for _ in range(count - 1):
        extra = int(rng.integers(0, min(2 if compact else 7, available) + 1))
        gaps.append(1 + extra)
        available -= extra
    supports = []
    cursor = 0
    for i, width in enumerate(widths):
        supports.append((cursor, cursor + width - 1))
        cursor += width + (gaps[i] if i < len(gaps) else 0)
    return supports


def _sample_bars(rng, height, count, force_bar, hollow):
    if hollow:
        return {}
    density = float(rng.uniform(0.12, 0.52))
    chosen = [row for row in range(1, height - 1) if rng.random() < density]
    if force_bar and not chosen:
        chosen = [int(rng.integers(1, height - 1))]
    while len(chosen) > height - 4:
        chosen.pop(int(rng.integers(0, len(chosen))))
    bars = {}
    for row in chosen:
        if count == 2 or rng.random() < 0.55:
            bars[row] = (0, count - 1)
        else:
            first = int(rng.integers(0, count - 1))
            last = int(rng.integers(first + 1, count))
            bars[row] = (first, last)
    return bars


def _sample_color_roles(rng, preferred_fg=None, avoid_fg=None):
    bg = 0 if rng.random() < 0.48 else int(rng.integers(1, 10))
    if preferred_fg is not None and preferred_fg != bg:
        fg = preferred_fg
    else:
        choices = [color for color in range(10) if color != bg and color != avoid_fg]
        fg = int(rng.choice(choices))
    return bg, fg


def _pair(rng, height=None, support_count=None, hollow=False,
          force_bar=False, top_margin=None, bottom_margin=None,
          preferred_fg=None, avoid_fg=None):
    compact = height is None and rng.random() < 0.26
    if height is None:
        if compact:
            height = int(rng.integers(5, 10))
            if support_count is None:
                support_count = int(rng.integers(2, 4))
        else:
            band = int(rng.integers(0, 3))
            height = int(rng.integers(*((5, 11), (11, 19), (19, 28))[band]))
    supports = _sample_supports(rng, support_count, compact)
    width = supports[-1][1] + 1
    bars = _sample_bars(rng, height, len(supports), force_bar, hollow)
    remaining_h = 30 - height
    if top_margin is None:
        top_limit = 2 if compact or rng.random() < 0.30 else 10
        top_margin = int(rng.integers(0, min(remaining_h, top_limit) + 1))
    if bottom_margin is None:
        bottom_limit = 2 if compact or rng.random() < 0.30 else 10
        bottom_margin = int(rng.integers(0, min(remaining_h - top_margin, bottom_limit) + 1))
    _check(top_margin + bottom_margin <= remaining_h, "invalid vertical margins")
    remaining_w = 30 - width - 2
    if compact or rng.random() < 0.38:
        left_extra = int(rng.integers(0, min(2, remaining_w) + 1))
        right_extra = int(rng.integers(0, min(2, remaining_w - left_extra) + 1))
    else:
        left_extra = int(rng.integers(0, remaining_w + 1))
        right_extra = int(rng.integers(0, remaining_w - left_extra + 1))
    left_margin, right_margin = 1 + left_extra, 1 + right_extra
    bg, fg = _sample_color_roles(rng, preferred_fg, avoid_fg)
    grid = np.full((height + top_margin + bottom_margin,
                    width + left_margin + right_margin), bg, dtype=np.int8)
    top, left = top_margin, left_margin
    for a, b in supports:
        grid[top:top + height, left + a:left + b + 1] = fg
    grid[top, left:left + width] = fg
    grid[top + height - 1, left:left + width] = fg
    for row, (first, last) in bars.items():
        grid[top + row, left + supports[first][1]:left + supports[last][0] + 1] = fg
    return {"input": grid, "output": _wave(grid)}


def _observed_offsets(pair, info):
    inp, out = pair["input"], pair["output"]
    _check(inp.shape == out.shape, "shape changed")
    _check(set(int(x) for x in np.unique(out)) == {info["bg"], info["fg"]},
           "colors changed")
    for row in range(info["top"], info["bottom"] + 1):
        a = np.flatnonzero(inp[row] == info["fg"])
        b = np.flatnonzero(out[row] == info["fg"])
        _check(len(a) == len(b), "foreground count changed in row")
        delta = b - a
        _check(np.all(delta == delta[0]), "row did not translate rigidly")
        _check(int(delta[0]) == _offset(info["bottom"] - row), "wrong wave phase")
    _check(np.array_equal(out, _wave(inp)), "output does not match wave")
    _check(not np.array_equal(inp, out), "unchanged pair")


def _predict_from_training(train, test_input):
    """Infer the offset table solely from demonstrations and apply it."""
    learned = {}
    for pair in train:
        info = _analyze(pair["input"])
        inp, out = pair["input"], pair["output"]
        for row in range(info["top"], info["bottom"] + 1):
            a = np.flatnonzero(inp[row] == info["fg"])
            b = np.flatnonzero(out[row] == info["fg"])
            _check(len(a) == len(b) and np.all(b - a == b[0] - a[0]),
                   "training row does not show one offset")
            phase = (info["bottom"] - row) % 4
            delta = int(b[0] - a[0])
            _check(phase not in learned or learned[phase] == delta,
                   "contradictory phase examples")
            learned[phase] = delta
    _check(set(learned) == {0, 1, 2, 3}, "training omits a phase")
    info = _analyze(test_input)
    predicted = np.full_like(test_input, info["bg"])
    for row in range(info["top"], info["bottom"] + 1):
        cols = np.flatnonzero(test_input[row] == info["fg"])
        shifted = cols + learned[(info["bottom"] - row) % 4]
        _check(shifted.min() >= 0 and shifted.max() < test_input.shape[1],
               "prediction clipped")
        predicted[row, shifted] = info["fg"]
    return predicted


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    requested_train = _count(num_train, "num_train", 2, 10)
    requested_test = _count(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(seed)
    if requested_train is None and requested_test is None:
        ntest = 1 + seed % 4
        ntrain = ntest + 1 + (seed // 4) % (10 - ntest)
    elif requested_train is None:
        ntest = requested_test
        ntrain = int(rng.integers(ntest + 1, 11))
    elif requested_test is None:
        ntrain = requested_train
        ntest = int(rng.integers(1, min(4, ntrain - 1) + 1))
    else:
        ntrain, ntest = requested_train, requested_test
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")

    # The phase rule is fixed for the puzzle. Style only biases pair geometry.
    style = int(rng.integers(0, 3))
    shared_top = int(rng.integers(0, 4))
    first_height = int(rng.choice((10, 14, 18, 22, 26)))
    second_height = int(rng.choice((12, 16, 20, 24)))
    train = [
        _pair(rng, height=first_height, support_count=2, hollow=True,
              top_margin=shared_top, bottom_margin=0),
        _pair(rng, height=second_height, support_count=int(rng.integers(3, 6)),
              force_bar=True, top_margin=shared_top, bottom_margin=1),
    ]
    first_fg = _analyze(train[0]["input"])["fg"]
    if _analyze(train[1]["input"])["fg"] == first_fg:
        train[1] = _pair(rng, height=second_height,
                         support_count=int(rng.integers(3, 6)),
                         force_bar=True, top_margin=shared_top, bottom_margin=1,
                         avoid_fg=first_fg)
    seen_inputs = {_grid_key(pair["input"]) for pair in train}
    used_heights = {_analyze(pair["input"])["height"] for pair in train}
    for _ in range(ntrain - 2):
        for attempt in range(17):
            if attempt == 16:
                unused_height = next(h for h in range(5, 28) if h not in used_heights)
                pair = _pair(rng, height=unused_height)
            else:
                pair = _pair(rng, hollow=(style == 0 and rng.random() < 0.55))
            if (_grid_key(pair["input"]) not in seen_inputs
                    and _source_key(pair["input"]) not in _SOURCE_INPUTS):
                break
        train.append(pair)
        seen_inputs.add(_grid_key(pair["input"]))
        used_heights.add(_analyze(pair["input"])["height"])
    rng.shuffle(train)
    train_foregrounds = [_analyze(pair["input"])["fg"] for pair in train]
    test = []
    test_crops = set()
    for index in range(ntest):
        preferred = train_foregrounds[index % len(train)] if seed % 3 == 0 else None
        for attempt in range(17):
            if attempt == 16:
                unused_height = next(h for h in range(5, 28) if h not in used_heights)
                pair = _pair(rng, height=unused_height, preferred_fg=preferred)
            else:
                pair = _pair(rng, hollow=(style == 0 and rng.random() < 0.45),
                             preferred_fg=preferred)
            if (_grid_key(pair["input"]) not in seen_inputs
                    and _crop_signature(pair["input"]) not in test_crops
                    and _source_key(pair["input"]) not in _SOURCE_INPUTS):
                break
        test.append(pair)
        seen_inputs.add(_grid_key(pair["input"]))
        used_heights.add(_analyze(pair["input"])["height"])
        test_crops.add(_crop_signature(pair["input"]))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle schema invalid")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
           "pair counts invalid")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair schema invalid")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            ikey, okey = _grid_key(inp), _grid_key(out)
            _check(ikey not in seen_inputs and (ikey, okey) not in seen_pairs,
                   "duplicate pair/input")
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            _check(_source_key(inp) not in _SOURCE_INPUTS, "source input copied")
            info = _analyze(inp)
            _observed_offsets(pair, info)
            infos[split].append(info)
    train_infos = infos["train"]
    _check(any(x["height"] >= 10 and x["height"] % 4 == 2 for x in train_infos)
           and any(x["height"] >= 12 and x["height"] % 4 == 0 for x in train_infos),
           "training needs contrasting height phases")
    _check(len({x["bottom_margin"] % 4 for x in train_infos}) >= 2,
           "canvas-bottom anchor remains ambiguous")
    _check(len({x["bottom"] % 4 for x in train_infos}) >= 2,
           "absolute bottom row lacks contrast")
    _check(any(len(x["supports"]) == 2 and not x["bars"] for x in train_infos),
           "training needs a hollow frame")
    _check(any(len(x["supports"]) >= 3 and x["bars"] for x in train_infos),
           "training needs a subdivided lattice")
    _check(len({(x["bg"], x["fg"]) for x in train_infos}) >= 2,
           "training needs distinct local palettes")
    _check(len({_crop_signature(pair["input"]) for pair in test}) == len(test),
           "test scaffold geometries repeat")
    top_anchor_matches = all(
        _offset(row - info["top"]) == _offset(info["bottom"] - row)
        for info in train_infos
        for row in range(info["top"], info["bottom"] + 1)
    )
    canvas_anchor_matches = all(
        _offset(pair["input"].shape[0] - 1 - row) == _offset(info["bottom"] - row)
        for pair, info in zip(train, train_infos)
        for row in range(info["top"], info["bottom"] + 1)
    )
    _check(not top_anchor_matches and not canvas_anchor_matches,
           "training permits a competing phase anchor")
    for pair in test:
        _check(np.array_equal(_predict_from_training(train, pair["input"]), pair["output"]),
               "test not recoverable from training")


def self_test() -> None:
    for seed, ntrain, ntest in ((9, None, None), (17, 6, 2), (23, None, 4),
                                (29, 10, None), (101, 2, 1), (102, 10, 1),
                                (103, 10, 4), (104, None, 1), (105, None, 4),
                                (106, 2, None), (107, 10, None)):
        a = generate_puzzle(seed, ntrain, ntest)
        b = generate_puzzle(seed, ntrain, ntest)
        _check(all(np.array_equal(p[k], q[k]) for split in ("train", "test")
                   for p, q in zip(a[split], b[split]) for k in ("input", "output")),
               "nondeterministic generation")
        validate_generated_puzzle(a)
    for bad in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(1, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    all_info = [_analyze(pair["input"]) for p in puzzles
                for split in ("train", "test") for pair in p[split]]
    _check(len({x["height"] for x in all_info}) >= 15, "height range too narrow")
    _check(len({x["width"] for x in all_info}) >= 15, "width range too narrow")
    _check({0, 1, 2, 3}.issubset({x["height"] % 4 for x in all_info}),
           "height phases missing")
    _check(any(x["fg"] == 0 for x in all_info), "zero foreground missing")
    _check(any(x["bg"] != 0 for x in all_info), "nonzero background missing")
    _check(any(any(b - a == 1 for a, b in zip(x["bars"], x["bars"][1:]))
               for x in all_info), "thick crossbars missing")
    _check(any(len(x["bars"]) > 3 for x in all_info), "rich crossbars missing")
    _check(any(x["partial"] for x in all_info), "partial bars missing")
    _check(any(any(b - a + 1 > 1 for a, b in x["supports"]) for x in all_info),
           "thick supports missing")
    _check(any(all(b == a for a, b in x["supports"]) for x in all_info),
           "thin supports missing")
    _check(any(len(set(x["gaps"])) > 1 for x in all_info),
           "unequal support spaces missing")
    _check(any(pair["input"].shape[0] <= 10 and pair["input"].shape[1] <= 10
               for puzzle in puzzles for split in ("train", "test")
               for pair in puzzle[split]), "small canvases missing")
    _check(any(29 <= max(pair["input"].shape) <= 30
               for puzzle in puzzles for split in ("train", "test")
               for pair in puzzle[split]), "large canvases missing")
    test_infos = [_analyze(pair["input"]) for puzzle in puzzles
                  for pair in puzzle["test"]]
    _check({x["height"] % 4 for x in test_infos} == {0, 1, 2, 3},
           "test height residues missing")
    _check(any(x["bars"] for x in test_infos), "test crossbars missing")
    _check(any(len({(r - x["top"]) % 2 for r in x["bars"]}) == 2
               for x in test_infos), "mixed bar parity missing")
    repeated_color = novel_color = False
    for puzzle in puzzles:
        train_colors = {_analyze(pair["input"])["fg"] for pair in puzzle["train"]}
        for pair in puzzle["test"]:
            foreground = _analyze(pair["input"])["fg"]
            repeated_color |= foreground in train_colors
            novel_color |= foreground not in train_colors
    _check(repeated_color and novel_color, "test color regimes missing")
    for encoded in _SOURCE_INPUTS:
        source_grid = np.array([[int(color) for color in row]
                                for row in encoded.split("/")], dtype=np.int8)
        _wave(source_grid)
