import numpy as np


def _rows_to_grid(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIR_ROWS = (
    (
        (
            "000000812",
            "000000841",
            "888888888",
            "003303800",
            "330000800",
            "330303800",
            "000030800",
            "333333800",
            "000030800",
        ),
        (
            "001202",
            "110000",
            "110202",
            "000010",
            "444111",
            "000010",
        ),
    ),
    (
        (
            "248000000",
            "168000000",
            "888888888",
            "008030030",
            "008333333",
            "008030030",
            "008030030",
            "008333333",
            "008030030",
        ),
        (
            "020040",
            "222444",
            "020040",
            "010060",
            "111666",
            "010060",
        ),
    ),
    (
        (
            "008003003",
            "008003003",
            "008330330",
            "008000030",
            "008030300",
            "008030003",
            "888888888",
            "248000000",
            "658000000",
        ),
        (
            "002004",
            "002004",
            "220440",
            "000050",
            "060500",
            "060005",
        ),
    ),
    (
        (
            "000300800",
            "330303800",
            "030303800",
            "033300800",
            "030003800",
            "003000800",
            "888888888",
            "000000821",
            "000000847",
        ),
        (
            "000100",
            "220101",
            "020101",
            "044700",
            "040007",
            "004000",
        ),
    ),
)

_SOURCE_PAIRS = tuple(
    (_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array_token(grid):
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _puzzle_token(puzzle):
    return tuple(
        (
            split,
            tuple(
                (
                    _array_token(pair["input"]),
                    _array_token(pair["output"]),
                )
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _is_source_pair(input_grid, output_grid):
    return any(
        np.array_equal(input_grid, source_input)
        and np.array_equal(output_grid, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


_CORNERS = ("TL", "TR", "BL", "BR")


def _validated_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(code, num_train, num_test):
    train = _validated_count(num_train, "num_train", 2, 10)
    test = _validated_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + code % 4
        low = max(2, test + 1)
        train = low + (code // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + (code ^ 0x9E3779B97F4A7C15) % (11 - low)
    elif test is None:
        high = min(4, train - 1)
        test = 1 + (code ^ 0xD1B54A32D192ED03) % high
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _slices(key_shape, block_shape, layout):
    kr, kc = key_shape
    bh, bw = block_shape
    mh, mw = kr * bh, kc * bw
    if layout[0] == "T":
        key_rows, sep_row, mask_rows = slice(0, kr), kr, slice(kr + 1, kr + 1 + mh)
    else:
        mask_rows, sep_row, key_rows = slice(0, mh), mh, slice(mh + 1, mh + 1 + kr)
    if layout[1] == "L":
        key_cols, sep_col, mask_cols = slice(0, kc), kc, slice(kc + 1, kc + 1 + mw)
    else:
        mask_cols, sep_col, key_cols = slice(0, mw), mw, slice(mw + 1, mw + 1 + kc)
    return key_rows, sep_row, mask_rows, key_cols, sep_col, mask_cols


def _make_mask(rng, key_shape, block_shape, mode, extreme, showcase):
    kr, kc = key_shape
    bh, bw = block_shape
    h, w = kr * bh, kc * bw
    mask = np.zeros((h, w), dtype=bool)
    if mode in ("repeat", "partial"):
        motif = np.zeros((bh, bw), dtype=bool)
        motif_type = int(rng.integers(0, 5))
        if motif_type == 0:
            motif[bh // 2, :] = True
            motif[:, bw // 2] = True
        elif motif_type == 1:
            motif[:, int(rng.integers(0, bw))] = True
            motif[int(rng.integers(0, bh)), :] = True
        elif motif_type == 2:
            motif[0, :] = True
            motif[:, 0] = True
        elif motif_type == 3:
            motif[np.arange(bh), np.minimum(bw - 1, np.arange(bh) * bw // bh)] = True
            motif[bh // 2, :] = True
        else:
            motif[0, :] = True
            motif[-1, :] = True
            motif[:, bw // 2] = True
        if np.all(motif):
            motif[0, 0] = False
        mask[:] = np.tile(motif, (kr, kc))
        if mode == "partial":
            for sr in range(kr):
                for sc in range(kc):
                    if (sr, sc) in ((0, 0), (kr - 1, kc - 1)):
                        continue
                    sector = mask[sr * bh:(sr + 1) * bh, sc * bw:(sc + 1) * bw]
                    count = int(rng.integers(1, max(2, bh * bw // 3 + 1)))
                    positions = rng.choice(bh * bw, size=count, replace=False)
                    sector.flat[positions] = ~sector.flat[positions]
    elif mode == "bars":
        mask[int(rng.integers(0, h)), :] = True
        mask[:, int(rng.integers(0, w))] = True
        for _ in range(int(rng.integers(1, 4))):
            if int(rng.integers(0, 2)):
                row = int(rng.integers(0, h))
                start = int(rng.integers(0, w - 1))
                mask[row, start:min(w, start + int(rng.integers(2, w + 1)))] = True
            else:
                col = int(rng.integers(0, w))
                start = int(rng.integers(0, h - 1))
                mask[start:min(h, start + int(rng.integers(2, h + 1))), col] = True
    elif mode == "connected":
        row = int(rng.integers(0, h))
        col = bw - 1
        mask[row, col:col + 2] = True
        col += 1
        for _ in range(int(rng.integers(h + w, h * w + 1))):
            direction = int(rng.integers(0, 4))
            row = min(h - 1, max(0, row + (direction == 0) - (direction == 1)))
            col = min(w - 1, max(0, col + (direction == 2) - (direction == 3)))
            mask[row, col] = True
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(0, h))
            col = int(rng.integers(0, w))
            mask[row, col:min(w, col + int(rng.integers(1, bw + 1)))] = True
    elif mode == "composite":
        for _ in range(int(rng.integers(2, 6))):
            row = int(rng.integers(0, h))
            col = int(rng.integers(0, w))
            height = int(rng.integers(1, bh + 2))
            width = int(rng.integers(1, bw + 2))
            mask[row:min(h, row + height), col:min(w, col + width)] = True
        for _ in range(int(rng.integers(0, max(2, h * w // 8)))):
            row = int(rng.integers(0, h))
            col = int(rng.integers(0, w))
            mask[row, col] = ~mask[row, col]
    else:
        base_density = float(rng.uniform(0.08, 0.92))
        for sr in range(kr):
            for sc in range(kc):
                density = float(np.clip(base_density + rng.normal(0, 0.14), 0.04, 0.96))
                mask[sr * bh:(sr + 1) * bh, sc * bw:(sc + 1) * bw] = rng.random((bh, bw)) < density

    if extreme:
        sr = int(rng.integers(0, kr))
        sc = int(rng.integers(0, kc))
        mask[sr * bh:(sr + 1) * bh, sc * bw:(sc + 1) * bw] = bool(rng.integers(0, 2))
    if showcase:
        for sr in range(kr):
            for sc in range(kc):
                sector = mask[sr * bh:(sr + 1) * bh, sc * bw:(sc + 1) * bw]
                if not np.any(sector):
                    sector[int(rng.integers(0, bh)), int(rng.integers(0, bw))] = True
                if np.all(sector):
                    sector[int(rng.integers(0, bh)), int(rng.integers(0, bw))] = False
    if not np.any(mask):
        mask[0, 0] = True
    if np.all(mask):
        mask[0, 0] = False
    return mask


def _choose_key(rng, shape, colors, required=(), showcase=False):
    colors = tuple(dict.fromkeys(int(c) for c in colors))
    required = tuple(dict.fromkeys(int(c) for c in required))
    cells = shape[0] * shape[1]
    _require(set(required) <= set(colors), "required key color unavailable")
    if showcase:
        _require(len(colors) >= cells, "insufficient showcase colors")
        selected = [int(v) for v in rng.choice(np.array(colors), size=cells, replace=False)]
    else:
        available = [c for c in colors if c not in required]
        low = max(2, len(required) + 1)
        high = min(cells, len(colors))
        _require(high >= low, "insufficient key colors")
        distinct = int(rng.integers(low, high + 1))
        selected = list(required)
        selected += [int(v) for v in rng.choice(np.array(available), size=distinct - len(required), replace=False)]
    flat = np.array(selected + [int(rng.choice(selected)) for _ in range(cells - len(selected))], dtype=np.int8)
    rng.shuffle(flat)
    return flat.reshape(shape)


def _pair(rng, shape, block, layout, roles, key, mode, extreme, showcase):
    background, separator, placeholder = roles
    binary = _make_mask(rng, shape, block, mode, extreme, showcase)
    h, w = shape[0] * (block[0] + 1) + 1, shape[1] * (block[1] + 1) + 1
    inp = np.full((h, w), background, dtype=np.int8)
    key_rows, sep_row, mask_rows, key_cols, sep_col, mask_cols = _slices(shape, block, layout)
    inp[sep_row, :] = separator
    inp[:, sep_col] = separator
    inp[key_rows, key_cols] = key
    inp[mask_rows, mask_cols] = np.where(binary, placeholder, background)
    expansion = np.repeat(np.repeat(key, block[0], axis=0), block[1], axis=1)
    out = np.where(binary, expansion, background).astype(np.int8)
    return {"input": inp, "output": out}


def _analyze_pair(inp, out):
    h, w = inp.shape
    candidates = []
    for color in range(10):
        if int(np.count_nonzero(inp == color)) != h + w - 1:
            continue
        rows = np.flatnonzero(np.all(inp == color, axis=1))
        cols = np.flatnonzero(np.all(inp == color, axis=0))
        if len(rows) == len(cols) == 1:
            candidates.append((int(rows[0]), int(cols[0]), color))
    _require(len(candidates) == 1, "separator must be unique full row and column")
    sep_row, sep_col, separator = candidates[0]
    row_lengths = (sep_row, h - sep_row - 1)
    col_lengths = (sep_col, w - sep_col - 1)
    _require(row_lengths[0] != row_lengths[1] and col_lengths[0] != col_lengths[1], "key and mask sizes must differ")
    kr, kc = min(row_lengths), min(col_lengths)
    mh, mw = max(row_lengths), max(col_lengths)
    _require((kr, kc) in ((2, 2), (2, 3), (3, 2)), "unsupported key shape")
    _require(mh % kr == 0 and mw % kc == 0, "unequal expansion blocks")
    block = (mh // kr, mw // kc)
    _require(all(2 <= side <= 6 for side in block), "block size out of range")
    layout = ("T" if row_lengths[0] == kr else "B") + ("L" if col_lengths[0] == kc else "R")
    key_rows, _, mask_rows, key_cols, _, mask_cols = _slices((kr, kc), block, layout)
    key = inp[key_rows, key_cols]
    mask = inp[mask_rows, mask_cols]
    off_a, off_b = inp[key_rows, mask_cols], inp[mask_rows, key_cols]
    background = int(off_a[0, 0])
    _require(np.all(off_a == background) and np.all(off_b == background), "off diagonal compartments must be background")
    _require(background != separator, "background and separator must differ")
    _require(not np.any(key == background) and not np.any(key == separator), "key roles overlap")
    values = set(int(x) for x in np.unique(mask))
    _require(len(values) == 2 and background in values, "mask must contain background and placeholder")
    placeholder = next(x for x in values if x != background)
    _require(placeholder != separator and not np.any(key == placeholder), "placeholder overlaps other roles")
    expansion = np.repeat(np.repeat(key, block[0], axis=0), block[1], axis=1)
    expected = np.where(mask == placeholder, expansion, background)
    _require(out.shape == mask.shape and np.array_equal(out, expected), "positional recoloring or output crop is wrong")
    counts = tuple(int(np.count_nonzero(mask[sr * block[0]:(sr + 1) * block[0], sc * block[1]:(sc + 1) * block[1]] == placeholder)) for sr in range(kr) for sc in range(kc))
    motifs = tuple(tuple(int(x == placeholder) for x in mask[sr * block[0]:(sr + 1) * block[0], sc * block[1]:(sc + 1) * block[1]].ravel()) for sr in range(kr) for sc in range(kc))
    return {"shape": (kr, kc), "block": block, "layout": layout, "roles": (background, separator, placeholder), "key": key, "counts": counts, "motifs": motifs, "mask": mask, "placeholder": placeholder, "mask_token": (mask.shape, tuple(int(x == placeholder) for x in mask.ravel()))}


def _different_block(rng, prior=None):
    if prior != (3, 3) and rng.random() < 0.27:
        return (3, 3)
    choices = [(a, b) for a in range(2, 7) for b in range(2, 7) if (a, b) != prior]
    return choices[int(rng.integers(0, len(choices)))]


def _role_draw(rng, palette, minimum=2, novel_candidates=()):
    for _ in range(40):
        roles = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
        if len(set(palette) - set(roles)) >= minimum and set(novel_candidates).isdisjoint(roles):
            return roles
    raise AssertionError("could not allocate local roles")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    code = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(code, num_train, num_test)
    rng = np.random.default_rng(code)
    shape = ((2, 2),) * 8 + ((2, 3), (3, 2))
    shape = shape[int(rng.integers(0, len(shape)))]
    simple_mode = bool(rng.random() < 0.12)
    if simple_mode:
        shape = (2, 2)
    palette_size = shape[0] * shape[1] if shape != (2, 2) else int(rng.integers(4, 6))
    train_palette = tuple(int(x) for x in rng.choice(10, size=palette_size, replace=False))
    local_roles = False if simple_mode else bool(rng.integers(0, 2))
    first_roles = tuple(int(x) for x in rng.choice([x for x in range(10) if x not in train_palette], size=3, replace=False))
    held_out = simple_mode or bool(rng.random() < 0.65)
    held_corner = "BR" if simple_mode else (_CORNERS[int(rng.integers(0, 4))] if held_out else None)
    diagonal = ("TR", "BL") if held_corner in ("TL", "BR") else ("TL", "BR")
    if not held_out:
        diagonal = (("TL", "BR"), ("TR", "BL"))[int(rng.integers(0, 2))]
    train_layouts = list(diagonal)
    if train_count >= 3:
        remaining = [x for x in _CORNERS if x not in train_layouts and x != held_corner]
        train_layouts.append(remaining[int(rng.integers(0, len(remaining)))])
    while len(train_layouts) < train_count:
        options = [x for x in _CORNERS if x != held_corner]
        train_layouts.append(options[int(rng.integers(0, len(options)))])
    if bool(rng.integers(0, 2)):
        train_layouts[0], train_layouts[1] = train_layouts[1], train_layouts[0]
    novelty = 1 if simple_mode else int(rng.integers(0, 3 if shape == (2, 2) else 2))
    pair_rngs = [np.random.default_rng(int(x)) for x in rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count, dtype=np.int64)]
    train, test = [], []
    seen_inputs, seen_masks = set(), set()
    fixed_scale = simple_mode or (shape == (2, 2) and rng.random() < 0.14)
    first_block = (3, 3) if fixed_scale or rng.random() < 0.65 else _different_block(rng)
    modes = ("irregular", "repeat", "partial", "bars", "connected", "composite")

    def draw_pair(index, split, layout, block, roles, key, mode, extreme, showcase):
        pair_rng = pair_rngs[index]
        for _ in range(30):
            pair = _pair(pair_rng, shape, block, layout, roles, key, mode, extreme, showcase)
            info = _analyze_pair(pair["input"], pair["output"])
            token = _array_token(pair["input"])
            if token in seen_inputs or info["mask_token"] in seen_masks or _is_source_pair(pair["input"], pair["output"]):
                continue
            seen_inputs.add(token)
            seen_masks.add(info["mask_token"])
            return pair
        raise AssertionError(f"could not draw unique {split} pair")

    for index in range(train_count):
        pair_rng = pair_rngs[index]
        roles = first_roles if index == 0 or not local_roles else _role_draw(pair_rng, train_palette)
        available = [c for c in train_palette if c not in roles]
        key = _choose_key(pair_rng, shape, available, showcase=(index == 0))
        block = first_block if index == 0 or fixed_scale else _different_block(pair_rng, first_block if index == 1 else None)
        mode = ("repeat", "bars", "connected", "composite")[int(pair_rng.integers(0, 4))] if index == 0 else ("irregular" if index == 1 else modes[int(pair_rng.integers(0, len(modes)))])
        extreme = index > 0 and bool(pair_rng.random() < 0.24)
        train.append(draw_pair(index, "train", train_layouts[index], block, roles, key, mode, extreme, index == 0))

    train_details = [_analyze_pair(pair["input"], pair["output"]) for pair in train]
    seen_train_colors = {int(x) for info in train_details for x in info["key"].ravel()}
    train_blocks = {info["block"] for info in train_details}
    for index in range(test_count):
        pair_rng = pair_rngs[train_count + index]
        if simple_mode or (held_out and index == 0):
            layout = held_corner
        elif held_out and pair_rng.random() < 0.7:
            layout = held_corner
        else:
            layout = train_layouts[int(pair_rng.integers(0, len(train_layouts)))]
        roles = first_roles if not local_roles else _role_draw(pair_rng, seen_train_colors, minimum=2)
        old = [c for c in sorted(seen_train_colors) if c not in roles]
        unseen = [c for c in range(10) if c not in seen_train_colors and c not in roles]
        if len(unseen) < novelty:
            roles = _role_draw(pair_rng, seen_train_colors, minimum=2, novel_candidates=unseen)
            unseen = [c for c in range(10) if c not in seen_train_colors and c not in roles]
        _require(len(unseen) >= novelty, "novel colors unavailable")
        required = [int(x) for x in pair_rng.choice(np.array(unseen), size=novelty, replace=False)] if novelty else []
        key = _choose_key(pair_rng, shape, old + required, required=required)
        block = (3, 3) if fixed_scale else _different_block(pair_rng)
        if index == 0 and not held_out and novelty == 0:
            new_blocks = [(a, b) for a in range(2, 7) for b in range(2, 7) if (a, b) not in train_blocks]
            block = new_blocks[int(pair_rng.integers(0, len(new_blocks)))]
        mode = modes[int(pair_rng.integers(0, len(modes)))]
        extreme = bool(pair_rng.random() < 0.36) or (index == 0 and not held_out and novelty == 0)
        test.append(draw_pair(train_count + index, "test", layout, block, roles, key, mode, extreme, False))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen_inputs, seen_pairs, seen_masks = set(), set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema wrong")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D array")
                _require(all(1 <= side <= 30 for side in grid.shape), "grid size out of range")
                _require(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)), "grid cells invalid")
            token = _array_token(pair["input"])
            pair_token = (token, _array_token(pair["output"]))
            _require(token not in seen_inputs and pair_token not in seen_pairs, "duplicate pair")
            _require(not _is_source_pair(pair["input"], pair["output"]), "source pair copied")
            info = _analyze_pair(pair["input"], pair["output"])
            _require(info["mask_token"] not in seen_masks, "duplicate binary mask")
            seen_inputs.add(token)
            seen_pairs.add(pair_token)
            seen_masks.add(info["mask_token"])
            infos[split].append(info)
    train_infos, test_infos = infos["train"], infos["test"]
    shapes = {x["shape"] for x in train_infos + test_infos}
    _require(len(shapes) == 1, "key shape changes within puzzle")
    _require(len({x["layout"][0] for x in train_infos}) == 2 and len({x["layout"][1] for x in train_infos}) == 2, "training must vary both placement axes")
    _require(any(len(set(int(v) for v in x["key"].ravel())) == x["shape"][0] * x["shape"][1] and all(0 < c < x["block"][0] * x["block"][1] for c in x["counts"]) for x in train_infos), "training must reveal every positional correspondence")
    train_colors = {int(v) for x in train_infos for v in x["key"].ravel()}
    _require(all(len({int(v) for v in x["key"].ravel()} - train_colors) <= 2 for x in test_infos), "test palette shift too large")
    train_layouts = {x["layout"] for x in train_infos}
    train_blocks = {x["block"] for x in train_infos}
    _require(any(x["layout"] not in train_layouts or bool({int(v) for v in x["key"].ravel()} - train_colors) or x["block"] not in train_blocks or 0 in x["counts"] or x["block"][0] * x["block"][1] in x["counts"] for x in test_infos), "test must exercise a visible shift")


def _puzzles_equal(first, second):
    return _puzzle_token(first) == _puzzle_token(second)


def self_test() -> None:
    for seed in (0, 1, 7, 19, 41):
        validate_generated_puzzle(generate_puzzle(seed))
    explicit = generate_puzzle(123456, 7, 3)
    _require(_puzzles_equal(explicit, generate_puzzle(123456, 7, 3)), "determinism failed")
    _require(not _puzzles_equal(generate_puzzle(101), generate_puzzle(102)), "seeds did not differ")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(2000 + train_count * 10 + test_count, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count), "explicit counts ignored")
    for seed, train_count, test_count in ((3001, None, 1), (3002, None, 4), (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((train_count is None or len(puzzle["train"]) == train_count) and (test_count is None or len(puzzle["test"]) == test_count), "partial counts ignored")
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (2, 4)):
        try:
            generate_puzzle(4000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corpus = [generate_puzzle(seed) for seed in range(42)]
    trains = {len(p["train"]) for p in corpus}
    tests = {len(p["test"]) for p in corpus}
    _require(len(trains) >= 4 and len(tests) >= 3, "count diversity insufficient")
    _require(len({_puzzle_token(p) for p in corpus}) >= 40, "puzzle diversity insufficient")
    _require(sum(len({_analyze_pair(pair["input"], pair["output"])["block"] for split in ("train", "test") for pair in p[split]}) > 1 for p in corpus) >= 30, "within-puzzle scale diversity insufficient")
    _require(sum(all(pair["input"].shape == (9, 9) for split in ("train", "test") for pair in p[split]) for p in corpus) >= 1, "source-scale puzzles absent")
    all_infos = [[_analyze_pair(pair["input"], pair["output"]) for split in ("train", "test") for pair in p[split]] for p in corpus]
    flat = [x for infos in all_infos for x in infos]
    _require({x["shape"] for x in flat} == {(2, 2), (2, 3), (3, 2)}, "key shape diversity insufficient")
    _require(len({x["block"] for x in flat}) >= 16 and (4, 2) in {x["block"] for x in flat}, "block diversity insufficient")
    _require({x["layout"] for x in flat} == set(_CORNERS), "corner diversity insufficient")
    _require(any(0 in x["counts"] for x in flat) and any(x["block"][0] * x["block"][1] in x["counts"] for x in flat), "blank/full sector diversity insufficient")
    _require(any(len(set(x["motifs"])) == 1 for x in flat), "repeated motifs absent")
    _require(any(len(set(x["motifs"])) < len(x["motifs"]) and len(set(x["motifs"])) > 1 for x in flat), "partial repeats absent")
    _require(any(len({x["roles"] for x in infos}) > 1 for infos in all_infos) and any(len({x["roles"] for x in infos}) == 1 for infos in all_infos), "local/fixed role diversity absent")
    source_sized = 0
    rectangular = 0
    blank = 0
    solid = 0
    repeats = 0
    partials = 0
    bar_crosses = 0
    boundary_crosses = 0
    asymmetrics = 0
    prior_key_role_reuse = 0
    combined = 0
    source_like = 0
    for puzzle, infos in zip(corpus, all_infos):
        train_infos = infos[:len(puzzle["train"])]
        test_infos = infos[len(puzzle["train"]):]
        train_palette = {int(v) for x in train_infos for v in x["key"].ravel()}
        local = len({x["roles"] for x in infos}) > 1
        earlier = set()
        for x in infos:
            if earlier & set(x["roles"]):
                prior_key_role_reuse += 1
            earlier.update(int(v) for v in x["key"].ravel())
            area = x["block"][0] * x["block"][1]
            has_blank = 0 in x["counts"]
            has_solid = area in x["counts"]
            source_sized += x["shape"] == (2, 2) and x["block"] == (3, 3)
            rectangular += x["block"][0] != x["block"][1]
            blank += has_blank
            solid += has_solid
            repeats += len(set(x["motifs"])) == 1
            partials += 1 < len(set(x["motifs"])) < len(x["motifs"])
            binary = x["mask"] == x["placeholder"]
            bar_crosses += bool(np.any(np.all(binary, axis=0)) and np.any(np.all(binary, axis=1)))
            bh, bw = x["block"]
            boundary_crosses += any(np.any(binary[r - 1] & binary[r]) for r in range(bh, binary.shape[0], bh)) or any(np.any(binary[:, c - 1] & binary[:, c]) for c in range(bw, binary.shape[1], bw))
            asymmetrics += not np.array_equal(binary, binary[:, ::-1]) and not np.array_equal(binary, binary[::-1, :])
        for x in test_infos:
            combined += local and x["block"][0] != x["block"][1] and (0 in x["counts"] or x["block"][0] * x["block"][1] in x["counts"]) and bool({int(v) for v in x["key"].ravel()} - train_palette)
        source_like += (all(x["shape"] == (2, 2) and x["block"] == (3, 3) for x in infos) and len({x["roles"] for x in infos}) == 1 and {x["layout"] for x in train_infos} <= {"TL", "TR", "BL"} and all(x["layout"] == "BR" and len({int(v) for v in x["key"].ravel()} - train_palette) == 1 for x in test_infos))
    _require(source_sized >= 50 and rectangular >= 50 and source_like >= 1, "source and rectangular scale coverage insufficient")
    _require(blank >= 20 and solid >= 10, "blank/full sector coverage insufficient")
    _require(repeats >= 10 and partials >= 20 and bar_crosses >= 20 and boundary_crosses >= 100 and asymmetrics >= 100, "mask geometry coverage insufficient")
    _require(prior_key_role_reuse >= 5 and combined >= 1, "local roles and combined shifts insufficient")
    novelty_counts = set()
    held_flags = set()
    for puzzle, infos in zip(corpus, all_infos):
        train_infos = infos[:len(puzzle["train"])]
        test_infos = infos[len(puzzle["train"]):]
        palette = {int(v) for x in train_infos for v in x["key"].ravel()}
        novelty_counts.update(len({int(v) for v in x["key"].ravel()} - palette) for x in test_infos)
        held_flags.update(x["layout"] not in {y["layout"] for y in train_infos} for x in test_infos)
    _require(novelty_counts == {0, 1, 2} and held_flags == {False, True}, "train/test regime diversity insufficient")
