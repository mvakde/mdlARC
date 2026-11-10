import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        # Favor concise demonstrations without excluding larger puzzles.
        train = int(rng.choice([2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10]))
        test = int(rng.integers(1, min(4, train - 1) + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _connected(mask):
    cells = {tuple(x) for x in np.argwhere(mask)}
    if not cells:
        return False
    found = {next(iter(cells))}
    todo = list(found)
    while todo:
        r, c = todo.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in cells and neighbor not in found:
                found.add(neighbor)
                todo.append(neighbor)
    return len(found) == len(cells)


def _hole_cells(mask):
    h, w = mask.shape
    blank = {(r, c) for r in range(h) for c in range(w) if not mask[r, c]}
    outside = {cell for cell in blank if cell[0] in (0, h - 1) or cell[1] in (0, w - 1)}
    todo = list(outside)
    while todo:
        r, c = todo.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in blank and neighbor not in outside:
                outside.add(neighbor)
                todo.append(neighbor)
    return blank - outside


def _mask(rng, h, w, style):
    mask = np.zeros((h, w), dtype=bool)
    if style == "ring":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif style == "bar":
        mask[:, 0] = mask[:, -1] = True
        mask[int(rng.integers(1, h - 1)), :] = True
    elif style == "solid":
        mask[:, :] = True
    elif style == "hole":
        mask[:, :] = True
        r = int(rng.integers(1, h - 1))
        c = int(rng.integers(1, w - 1))
        mask[r, c] = False
        if h > 4 and w > 4 and rng.random() < 0.45:
            mask[r, min(w - 2, c + 1)] = False
    elif style == "cross":
        r = int(rng.integers(h))
        c = int(rng.integers(w))
        mask[r, :] = True
        mask[:, c] = True
        for _ in range(int(rng.integers(0, max(2, h * w // 4)))):
            rr, cc = map(int, rng.choice(np.argwhere(mask)))
            nr, nc = rr + int(rng.integers(-1, 2)), cc + int(rng.integers(-1, 2))
            if 0 <= nr < h and 0 <= nc < w and abs(nr - rr) + abs(nc - cc) == 1:
                mask[nr, nc] = True
    elif style == "branch":
        # A spanning, randomly offset connected skeleton with optional branches.
        top = int(rng.integers(w))
        bottom = int(rng.integers(w))
        for r in range(h):
            c = int(round(top + (bottom - top) * r / max(1, h - 1)))
            mask[r, c] = True
            if r and not mask[r - 1, c]:
                lo = min(c, int(np.flatnonzero(mask[r - 1])[0]))
                hi = max(c, int(np.flatnonzero(mask[r - 1])[0]))
                mask[r, lo:hi + 1] = True
        bar_r = int(rng.integers(h))
        mask[bar_r, :] = True
        for _ in range(int(rng.integers(0, h * w // 3 + 1))):
            rr, cc = map(int, rng.choice(np.argwhere(mask)))
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            if 0 <= rr + dr < h and 0 <= cc + dc < w:
                mask[rr + dr, cc + dc] = True
    elif style == "groups":
        if w >= 4 and (h < 4 or rng.random() < 0.5):
            gap = int(rng.integers(1, w - 1))
            left_end = int(rng.integers(0, gap))
            right_start = int(rng.integers(gap + 1, w))
            top_end = int(rng.integers(0, h))
            bottom_start = int(rng.integers(top_end, h))
            mask[0:top_end + 1, 0:left_end + 1] = True
            mask[bottom_start:h, right_start:w] = True
            if (top_end + 1) * (left_end + 1) == 1:
                mask[1, 0] = True
            if (h - bottom_start) * (w - right_start) == 1:
                mask[h - 2, w - 1] = True
        else:
            gap = int(rng.integers(1, h - 1))
            top_end = int(rng.integers(0, gap))
            bottom_start = int(rng.integers(gap + 1, h))
            left_end = int(rng.integers(0, w))
            right_start = int(rng.integers(left_end, w))
            mask[0:top_end + 1, 0:left_end + 1] = True
            mask[bottom_start:h, right_start:w] = True
            if (top_end + 1) * (left_end + 1) == 1:
                mask[0, 1] = True
            if (h - bottom_start) * (w - right_start) == 1:
                mask[h - 1, w - 2] = True
    elif style == "fragments":
        middle = int(rng.integers(2, h - 2))
        top_col = int(rng.integers(1, w - 1))
        bottom_col = int(rng.integers(1, w - 1))
        mask[:middle, top_col] = True
        mask[middle + 1:, bottom_col] = True
        mask[middle, 0] = mask[middle, -1] = True
    else:
        raise ValueError("unknown stencil style")
    _require(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(), "nonminimal stencil")
    _require(_connected(mask) == (style not in ("groups", "fragments")), "stencil connectivity error")
    return mask


def _stamp(grid, mask, top, left, color):
    rows, cols = grid.shape
    for dr, dc in np.argwhere(mask):
        r, c = top + int(dr), left + int(dc)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r, c] = color


def _bbox(grid, source):
    cells = np.argwhere(grid == source)
    _require(len(cells) > 0, "empty source stencil")
    low, high = cells.min(axis=0), cells.max(axis=0)
    r, c = map(int, low)
    return r, c, grid[r:int(high[0]) + 1, c:int(high[1]) + 1] == source


def _output(input_grid, bg, src, accent, directions):
    rows, cols = input_grid.shape
    r, c, mask = _bbox(input_grid, src)
    h, w = mask.shape
    horizontal, vertical = directions
    result = np.full(input_grid.shape, bg, dtype=np.int8)
    k = 0
    while True:
        left = c + horizontal * k * (w + 1)
        if left >= cols or left + w <= 0:
            break
        if k % 3 == 2:
            m = 0
            while True:
                top = r + vertical * m * (h + 1)
                if top >= rows or top + h <= 0:
                    break
                _stamp(result, mask, top, left, accent)
                m += 1
        else:
            _stamp(result, mask, r, left, src)
        k += 1
    return result


def _axis_options(length, direction, minimum, maximum, clip):
    options = []
    step = length + 1
    for count in range(minimum, maximum + 1):
        for visible in (range(1, length) if clip else (length,)):
            for near_margin in range(0, 9):
                for far_margin in ((0,) if clip else (0, 1)):
                    if direction == 1:
                        origin = near_margin
                        size = origin + (count - 1) * step + visible + far_margin
                    else:
                        origin = (count - 1) * step - (length - visible) + far_margin
                        size = origin + length + near_margin
                    if origin >= 0 and origin + length <= size and 8 <= size <= 30:
                        options.append((origin, size, count, clip))
    return options


def _mask_choice(rng, connected, demo=False):
    if connected:
        styles = ("ring", "bar", "solid", "hole", "cross", "branch")
        style = styles[int(rng.integers(len(styles)))]
        h = int(rng.integers(3 if style in ("ring", "bar", "hole") else 2, 5 if demo else 7))
        w = int(rng.integers(3 if style in ("ring", "bar", "hole") else 2, 5 if demo else 7))
    else:
        style = "fragments" if not demo and rng.random() < 0.3 else "groups"
        if style == "fragments":
            h = int(rng.integers(5, 7))
            w = int(rng.integers(3, 7))
        else:
            h = int(rng.integers(3, 5 if demo else 7))
            w = int(rng.integers(3, 5 if demo else 7))
            if h < 4 and w < 4:
                w = 4
    return _mask(rng, h, w, style)


def _make_pair(rng, accent, palette, directions, connected, demo, clip_h, clip_v):
    bg, src = palette
    # A complete second cycle fits the demonstration only for compact widths.
    for _ in range(80):
        mask = _mask_choice(rng, connected, demo)
        h, w = mask.shape
        if demo and w > 4:
            continue
        horizontal = _axis_options(w, directions[0], 6 if demo else 3, 8, clip_h)
        vertical = _axis_options(h, directions[1], 3 if demo else 2, 5, clip_v)
        if horizontal and vertical:
            break
    else:
        raise ValueError("could not make a feasible stencil")
    c, cols, _, _ = horizontal[int(rng.integers(len(horizontal)))]
    r, rows, _, _ = vertical[int(rng.integers(len(vertical)))]
    input_grid = np.full((rows, cols), bg, dtype=np.int8)
    _stamp(input_grid, mask, r, c, src)
    return {"input": input_grid, "output": _output(input_grid, bg, src, accent, directions)}


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    a, b = pair["input"], pair["output"]
    for grid in (a, b):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid dimensions")
        _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
        _require(np.all((grid >= 0) & (grid <= 9)), "grid colors")
    _require(a.shape == b.shape, "shape changed")
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 2 and counts.max() > counts.min(), "input roles ambiguous")
    bg = int(colors[np.argmax(counts)])
    src = int(colors[np.argmin(counts)])
    r, c, mask = _bbox(a, src)
    _require(2 <= mask.shape[0] <= 6 and 2 <= mask.shape[1] <= 6, "stencil not compact")
    _require(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any(), "nonminimal bbox")
    extras = set(map(int, np.unique(b))) - {bg, src}
    _require(len(extras) == 1 and set(map(int, np.unique(b))) == {bg, src, *extras}, "accent roles")
    accent = next(iter(extras))
    _require(not np.array_equal(a, b), "degenerate pair")
    return dict(bg=bg, src=src, accent=accent, r=r, c=c, h=mask.shape[0], w=mask.shape[1], mask=mask, rows=a.shape[0], cols=a.shape[1])


def _original(desc):
    if (desc["rows"], desc["cols"], desc["bg"], desc["src"], desc["accent"]) != (23, 28, 0, 7, 6):
        return False
    originals = (
        (5, 3, ("111", "101", "111")),
        (4, 5, ("101", "101", "111", "101", "101")),
        (7, 2, ("010", "010", "101", "010", "010")),
    )
    for r, c, lines in originals:
        if desc["r"] == r and desc["c"] == c and np.array_equal(desc["mask"], np.array([[x == "1" for x in line] for line in lines])):
            return True
    return False


def _axis_features(desc, horizontal, direction):
    origin, size, length = (desc["c"], desc["cols"], desc["w"]) if horizontal else (desc["r"], desc["rows"], desc["h"])
    count = 0
    while True:
        start = origin + direction * count * (length + 1)
        if start >= size or start + length <= 0:
            break
        count += 1
    last = origin + direction * (count - 1) * (length + 1)
    return count, last < 0 or last + length > size


def _directions_matching(train):
    matches = []
    for direction in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        if all(np.array_equal(pair["output"], _output(pair["input"], desc["bg"], desc["src"], desc["accent"], direction)) for pair, desc in train):
            matches.append(direction)
    return matches


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    accent = int(rng.integers(10))
    directions = (int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))
    fixed_palette = bool(rng.random() < 0.28)
    connected_shift = bool(rng.random() < 0.25)
    role_colors = [x for x in range(10) if x != accent]
    palettes = [(bg, src) for bg in role_colors for src in role_colors if bg != src]
    first_palette = palettes[int(rng.integers(len(palettes)))]
    result = {"train": [], "test": []}
    used = set()
    used_masks = set()
    used_profiles = set()
    total = n_train + n_test
    for index in range(total):
        split = "train" if index < n_train else "test"
        local = index if split == "train" else index - n_train
        demo = split == "train" and local == 0
        if demo:
            connected = True
        elif split == "train" and local == 1:
            connected = connected_shift
        elif split == "test" and local == 0 and connected_shift:
            connected = False
        else:
            connected = bool(rng.random() < (0.6 if split == "test" else 0.55))
        if fixed_palette or demo:
            palette = first_palette
        else:
            choices = [p for p in palettes if p != first_palette] if split == "train" and local == 1 else palettes
            palette = choices[int(rng.integers(len(choices)))]
        # Clipping is sampled separately on the two axes and on each side of the split.
        clip_h = bool(rng.random() < (0.49 if split == "test" else 0.42))
        clip_v = bool(rng.random() < (0.49 if split == "test" else 0.42))
        for attempt in range(160):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            pair = _make_pair(pair_rng, accent, palette, directions, connected, demo, clip_h, clip_v)
            desc = _analyze(pair)
            mask_key = (desc["h"], desc["w"], desc["mask"].tobytes())
            profile = (desc["r"], desc["c"], desc["h"], desc["w"], desc["rows"], desc["cols"])
            if _key(pair["input"]) in used or mask_key in used_masks:
                continue
            if split == "train" and local == 1 and profile in used_profiles:
                continue
            break
        else:
            raise ValueError("could not construct distinct pairs")
        used.add(_key(pair["input"]))
        used_masks.add(mask_key)
        used_profiles.add(profile)
        result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyzed = {"train": [], "test": []}
    accents = set()
    stencil_keys = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            desc = _analyze(pair)
            input_key, output_key = _key(pair["input"]), _key(pair["output"])
            _require(input_key not in seen_inputs and (input_key, output_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _require(not _original(desc), "original source pair copied")
            stencil_key = (desc["h"], desc["w"], desc["mask"].tobytes())
            _require(stencil_key not in stencil_keys, "repeated stencil within puzzle")
            stencil_keys.add(stencil_key)
            accents.add(desc["accent"])
            analyzed[split].append((pair, desc))
    _require(len(accents) == 1, "accent differs within puzzle")
    directions = _directions_matching(analyzed["train"])
    _require(len(directions) == 1, "training does not reveal a unique direction")
    direction = directions[0]
    _require(any(_axis_features(d, True, direction[0])[0] >= 6 for _, d in analyzed["train"]), "training hides second color cycle")
    _require(any(_axis_features(d, False, direction[1])[0] >= 3 for _, d in analyzed["train"]), "training hides repeated accent column")
    _require(len({(d["h"], d["w"], d["mask"].tobytes()) for _, d in analyzed["train"]}) >= 2, "training stencil diversity")
    for split in ("train", "test"):
        for pair, desc in analyzed[split]:
            _require(_axis_features(desc, True, direction[0])[0] >= 3, "accent head missing")
            _require(_axis_features(desc, False, direction[1])[0] >= 2, "vertical copy missing")
            predicted = _output(pair["input"], desc["bg"], desc["src"], next(iter(accents)), direction)
            _require(np.array_equal(pair["output"], predicted), "wrong stencil lattice")
    # This prediction uses only solved training pairs and the test input.
    for pair, desc in analyzed["test"]:
        inferred_accent = analyzed["train"][0][1]["accent"]
        prediction = _output(pair["input"], desc["bg"], desc["src"], inferred_accent, direction)
        _require(np.array_equal(pair["output"], prediction), "test not inferable from training")


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for seed in (0, 7, 99):
        validate_generated_puzzle(generate_puzzle(seed))
    for i, (n_train, n_test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(500 + i, n_train, n_test)
        if n_train is not None:
            _require(len(puzzle["train"]) == n_train, "train count ignored")
        if n_test is not None:
            _require(len(puzzle["test"]) == n_test, "test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2, 1.0)):
        try:
            generate_puzzle(700, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    _require(_equal(generate_puzzle(1234, 7, 3), generate_puzzle(1234, 7, 3)), "nondeterministic")
    _require(not _equal(generate_puzzle(1234, 7, 3), generate_puzzle(1235, 7, 3)), "seeds collapsed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    seen_masks, shapes, placements, directions, profiles = set(), set(), set(), set(), set()
    train_disconnected = test_connected = asymmetric = solid = thin = offcenter_holes = 0
    clip_combos = set()
    mixed_palettes = concise = varied_origins = varied_shapes = varied_clips = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        if len(puzzle["train"]) <= 3:
            concise += 1
        train_descs = [_analyze(p) for p in puzzle["train"]]
        if len({(d["bg"], d["src"]) for d in train_descs}) > 1:
            mixed_palettes += 1
        direction = _directions_matching(list(zip(puzzle["train"], train_descs)))[0]
        directions.add(direction)
        puzzle_origins, puzzle_shapes, puzzle_clips = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analyze(pair)
                seen_masks.add((d["h"], d["w"], d["mask"].tobytes()))
                shapes.add((d["rows"], d["cols"]))
                placements.add((d["r"], d["c"]))
                horizontal = _axis_features(d, True, direction[0])
                vertical = _axis_features(d, False, direction[1])
                profiles.add((horizontal[0], vertical[0], horizontal[1], vertical[1]))
                puzzle_origins.add((d["r"], d["c"]))
                puzzle_shapes.add((d["rows"], d["cols"]))
                puzzle_clips.add((horizontal[1], vertical[1]))
                asymmetric += not np.array_equal(d["mask"], d["mask"][:, ::-1])
                solid += bool(d["mask"].all())
                thin += d["mask"].mean() <= 0.4
                offcenter_holes += any(2 * r != d["h"] - 1 or 2 * c != d["w"] - 1 for r, c in _hole_cells(d["mask"]))
                if split == "train" and not _connected(d["mask"]):
                    train_disconnected += 1
                if split == "test":
                    test_connected += int(_connected(d["mask"]))
                    clip_combos.add((horizontal[1], vertical[1]))
        varied_origins += len(puzzle_origins) > 1
        varied_shapes += len(puzzle_shapes) > 1
        varied_clips += len(puzzle_clips) > 1
    _require(len(seen_masks) >= 45 and len(shapes) >= 50 and len(placements) >= 25, "structural diversity")
    _require(len(profiles) >= 18 and len(directions) == 4, "lattice diversity")
    _require(train_disconnected >= 8 and test_connected >= 8, "train/test shape split persists")
    _require(len(clip_combos) == 4 and mixed_palettes >= 10 and concise >= 6, "review regimes lack diversity")
    _require(asymmetric >= 25 and solid >= 8 and thin >= 15 and offcenter_holes >= 8, "stencil styles lack diversity")
    _require(min(varied_origins, varied_shapes, varied_clips) >= 35, "pair properties are fixed within puzzles")
