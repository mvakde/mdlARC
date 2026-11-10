import numpy as np


# Exact source inputs are excluded; their sizes and color schemes remain legal.
ORIGINAL_INPUTS = {
    ("111111022222222", "144441024444222", "141141024224222",
     "141141024224222", "141141024224222", "144441024444222",
     "111111022222222", "111111022222222"),
    ("77777777", "77887777", "77887777", "77777777",
     "00000000", "33333333", "33388883", "33388883",
     "33388883", "33333333", "33333333"),
    ("22222212222222", "22222121222222", "22221222122222",
     "22222121222222", "22222212222222", "22222222222222",
     "00000000000000", "88888888888188", "88888888881818",
     "88888888818881", "88888888881818", "88888888888188",
     "88888888888888"),
    ("999999", "922299", "922299", "999999", "999999",
     "000000", "555555", "522555", "522555", "555555"),
    ("44444444", "44488444", "44488444", "44444444",
     "00000000", "33333333", "88333333", "88333333"),
    ("4444044444", "4444044444", "4444044444", "4334044334",
     "3333044334", "4334044334", "4444044444", "4444044444",
     "4444044444", "4444044444"),
    ("55000055", "55000055", "55000055", "55555555",
     "00000000", "33333333", "33333333", "33000033",
     "33000033", "33000033", "33333333"),
}


def _counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    legal = [
        (a, b) for a in range(2, 11) for b in range(1, 5)
        if a > b
        and (num_train is None or a == num_train)
        and (num_test is None or b == num_test)
    ]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    return legal[int(rng.integers(len(legal)))]


def _parameters(rng):
    separator = 0 if rng.random() < 0.65 else int(rng.integers(1, 10))
    other = [c for c in range(10) if c != separator]
    size = int(rng.integers(3, 10))
    palette = tuple(int(c) for c in rng.choice(other, size, replace=False))
    return {
        "separator": separator,
        "palette": palette,
        "orientation_style": ("mixed", "train_h", "train_v")[
            int(rng.choice(3, p=[0.55, 0.35, 0.10]))
        ],
    }


def _full_box(mask):
    return bool(
        mask[0].any() and mask[-1].any()
        and mask[:, 0].any() and mask[:, -1].any()
    )


def _connected(mask):
    cells = {tuple(map(int, point)) for point in np.argwhere(mask)}
    if not cells:
        return False
    reached = {next(iter(cells))}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (r + dr, c + dc)
                if neighbor in cells and neighbor not in reached:
                    reached.add(neighbor)
                    stack.append(neighbor)
    return len(reached) == len(cells)


def _mask(rng, kind, h, w):
    mask = np.zeros((h, w), dtype=bool)
    if kind == "solid":
        mask[:] = True
    elif kind in ("frame", "partial_frame"):
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if kind == "partial_frame":
            inside = [(r, c) for r in range(1, h - 1)
                      for c in range(1, w - 1)]
            amount = int(rng.integers(1, max(2, len(inside) // 2 + 1)))
            for _ in range(min(amount, len(inside))):
                frontier = [
                    (r, c) for r, c in inside if not mask[r, c]
                    and any(mask[r + dr, c + dc]
                            for dr in (-1, 0, 1)
                            for dc in (-1, 0, 1))
                ]
                mask[frontier[int(rng.integers(len(frontier)))]] = True
    elif kind == "diagonal":
        steps = max(h, w)
        for i in range(steps):
            r = int(round(i * (h - 1) / (steps - 1)))
            c = int(round(i * (w - 1) / (steps - 1)))
            mask[r, c] = True
            mask[r, w - 1 - c] = True
    elif kind == "cross":
        mask[h // 2, :] = True
        mask[:, w // 2] = True
    else:
        anchor = (int(rng.integers(h)), int(rng.integers(w)))
        mask[anchor] = True
        targets = (
            (0, int(rng.integers(w))),
            (h - 1, int(rng.integers(w))),
            (int(rng.integers(h)), 0),
            (int(rng.integers(h)), w - 1),
        )
        for target in targets:
            r, c = anchor
            while (r, c) != target:
                move_r = r != target[0] and (
                    c == target[1] or rng.random() < 0.75
                )
                move_c = c != target[1] and (
                    r == target[0] or rng.random() < 0.75
                )
                if not move_r and not move_c:
                    move_r = r != target[0]
                    move_c = not move_r
                if move_r:
                    r += 1 if target[0] > r else -1
                if move_c:
                    c += 1 if target[1] > c else -1
                mask[r, c] = True
        target_count = int(rng.uniform(0.28, 0.65) * h * w)
        while int(mask.sum()) < target_count:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = int(r) + dr, int(c) + dc
                        if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                            frontier.add((rr, cc))
            if not frontier:
                break
            mask[sorted(frontier)[int(rng.integers(len(frontier)))]] = True
    assert _full_box(mask) and mask.sum() >= 2 and _connected(mask)
    return mask


def _base_mask(rng, kind=None, close=False):
    if close:
        h = int(rng.integers(4, 9))
        w = int(rng.integers(4, 9))
        kind = ("frame", "partial_frame", "random")[
            int(rng.integers(3))
        ]
    else:
        lower = 3 if kind in ("frame", "partial_frame") else 2
        upper = 5 if rng.random() < 0.35 else 9
        h = int(rng.integers(lower, upper))
        w = int(rng.integers(lower, upper))
        if kind is None:
            kind = ("solid", "frame", "partial_frame", "diagonal",
                    "cross", "random")[int(rng.integers(6))]
        if min(h, w) == 2 and kind in ("frame", "partial_frame"):
            kind = "solid" if rng.random() < 0.5 else "random"
    return _mask(rng, kind, h, w)


def _different(rng, a, mode):
    h, w = a.shape
    if mode == "size":
        for _ in range(20):
            hb, wb = int(rng.integers(2, 9)), int(rng.integers(2, 9))
            if (hb, wb) != (h, w):
                kind = "solid" if rng.random() < 0.45 else "random"
                return _mask(rng, kind, hb, wb)
        return np.ones((2 if h != 2 else 3, 2), dtype=bool)

    if mode == "transform":
        candidates = [np.fliplr(a), np.flipud(a), np.rot90(a, 2)]
        if h == w:
            candidates.extend((np.rot90(a), np.rot90(a, 3)))
        candidates = [m for m in candidates if not np.array_equal(m, a)]
        if candidates:
            return candidates[int(rng.integers(len(candidates)))].copy()
        mode = "count"

    if mode == "swap":
        ones = np.argwhere(a)
        zeros = np.argwhere(~a)
        for i in rng.permutation(len(ones)):
            for j in rng.permutation(len(zeros)):
                b = a.copy()
                b[tuple(ones[int(i)])] = False
                b[tuple(zeros[int(j)])] = True
                if _full_box(b) and _connected(b):
                    return b
        mode = "count"

    if mode == "count":
        candidates = np.argwhere(~a)
        for i in rng.permutation(len(candidates)):
            b = a.copy()
            b[tuple(candidates[int(i)])] = True
            if _full_box(b) and _connected(b):
                return b
        candidates = np.argwhere(a)
        for i in rng.permutation(len(candidates)):
            b = a.copy()
            b[tuple(candidates[int(i)])] = False
            if _full_box(b) and _connected(b):
                return b

    # A 2x2 solid cannot have a distinct same-size full-box mask.
    return _different(rng, a, "size")


def _location(rng, maximum):
    if maximum == 0:
        return 0
    roll = rng.random()
    if roll < 0.24:
        return 0
    if roll < 0.48:
        return maximum
    return int(rng.integers(maximum + 1))


def _place(panel, mask, fg, position):
    top, left = position
    panel[top:top + mask.shape[0], left:left + mask.shape[1]][mask] = fg


def _compose(rng, a, b, orientation, backgrounds, fg, separator, aligned,
             placement):
    ha, wa = a.shape
    hb, wb = b.shape
    ba, bb = backgrounds
    if orientation == "H":
        width = max(wa, wb) + int(rng.integers(2, 8))
        margin = 2 if placement == "inset" else 1
        min_a = max(ha + margin, 2 * int(a.sum()) // width + 1)
        min_b = max(hb + margin, 2 * int(b.sum()) // width + 1)
        high_a = min(min_a + 6, 29 - min_b)
        height_a = int(rng.integers(min_a, high_a + 1))
        high_b = min(min_b + 6, 29 - height_a)
        height_b = int(rng.integers(min_b, high_b + 1))
        pa = np.full((height_a, width), ba, dtype=np.int8)
        pb = np.full((height_b, width), bb, dtype=np.int8)
    else:
        height = max(ha, hb) + int(rng.integers(2, 8))
        margin = 2 if placement == "inset" else 1
        min_a = max(wa + margin, 2 * int(a.sum()) // height + 1)
        min_b = max(wb + margin, 2 * int(b.sum()) // height + 1)
        high_a = min(min_a + 6, 29 - min_b)
        width_a = int(rng.integers(min_a, high_a + 1))
        high_b = min(min_b + 6, 29 - width_a)
        width_b = int(rng.integers(min_b, high_b + 1))
        pa = np.full((height, width_a), ba, dtype=np.int8)
        pb = np.full((height, width_b), bb, dtype=np.int8)

    def choose(maximum):
        if placement == "inset":
            assert maximum >= 2
            return int(rng.integers(1, maximum))
        return _location(rng, maximum)

    if aligned:
        top = choose(min(pa.shape[0] - ha, pb.shape[0] - hb))
        left = choose(min(pa.shape[1] - wa, pb.shape[1] - wb))
        posa = posb = (top, left)
    else:
        posa = (choose(pa.shape[0] - ha), choose(pa.shape[1] - wa))
        posb = (choose(pb.shape[0] - hb), choose(pb.shape[1] - wb))
    if placement == "edge":
        side = int(rng.integers(4))
        if side < 2:
            posa = (0 if side == 0 else pa.shape[0] - ha, posa[1])
        else:
            posa = (posa[0], 0 if side == 2 else pa.shape[1] - wa)
    _place(pa, a, fg, posa)
    _place(pb, b, fg, posb)
    if orientation == "H":
        return np.vstack((pa, np.full((1, width), separator, dtype=np.int8), pb))
    return np.hstack((pa, np.full((height, 1), separator, dtype=np.int8), pb))


def _panel_info(panel, separator):
    colors, counts = np.unique(panel, return_counts=True)
    assert len(colors) == 2
    index = int(np.argmax(counts))
    assert 2 * int(counts[index]) > panel.size
    bg = int(colors[index])
    fg = int(colors[1 - index])
    assert bg != separator
    rr, cc = np.where(panel == fg)
    top, bottom = int(rr.min()), int(rr.max()) + 1
    left, right = int(cc.min()), int(cc.max()) + 1
    mask = panel[top:bottom, left:right] == fg
    assert min(mask.shape) >= 2 and _full_box(mask) and _connected(mask)
    return bg, fg, mask, (top, left), (
        top == 0, bottom == panel.shape[0],
        left == 0, right == panel.shape[1],
    )


def _analyze(x, separator):
    rows = np.flatnonzero(np.all(x == separator, axis=1))
    cols = np.flatnonzero(np.all(x == separator, axis=0))
    assert (len(rows), len(cols)) in ((1, 0), (0, 1))
    if len(rows):
        k = int(rows[0])
        assert 0 < k < x.shape[0] - 1
        panels = (x[:k], x[k + 1:])
        orientation = "H"
    else:
        k = int(cols[0])
        assert 0 < k < x.shape[1] - 1
        panels = (x[:, :k], x[:, k + 1:])
        orientation = "V"
    a = _panel_info(panels[0], separator)
    b = _panel_info(panels[1], separator)
    assert a[1] == b[1]
    return orientation, a, b


def _expected(info, separator):
    _, a, b = info
    if not np.array_equal(a[2], b[2]):
        return np.full((1, 1), separator, dtype=np.int8)
    return np.where(a[2], a[1], separator).astype(np.int8)


def _colors(rng, params, regime, force_regular=False):
    palette = params["palette"]
    separator = params["separator"]
    same_rate = 0.25 if regime == "source_train" else 0.34
    foreground_rate = 0.19 if regime == "source_train" else 0.30
    ba = int(rng.choice(palette))
    same = not force_regular and rng.random() < same_rate
    bb = ba if same else int(rng.choice([c for c in palette if c != ba]))
    fg = (separator if not force_regular and rng.random() < foreground_rate else
          int(rng.choice([c for c in palette if c not in (ba, bb)])))
    return (ba, bb), fg


def _orientation(rng, params, regime, index, ntrain):
    style = params["orientation_style"]
    if regime == "source_train":
        if style == "train_h":
            return "H"
        if style == "train_v":
            return "V"
        if index == 0:
            return "H"
        if index == 1:
            return "V"
        return "H" if rng.random() < 0.5 else "V"
    if style == "train_h":
        return "V" if rng.random() < 0.6 else "H"
    if style == "train_v":
        return "H" if rng.random() < 0.6 else "V"
    return "H" if rng.random() < 0.5 else "V"


def _pair(rng, params, regime, job, orientation, placement):
    matching, kind, difference = job
    a = _base_mask(rng, kind, close=difference in ("swap", "count"))
    if difference == "transform":
        for _ in range(20):
            a = _base_mask(rng, "random")
            if any(not np.array_equal(a, m) for m in
                   (np.fliplr(a), np.flipud(a), np.rot90(a, 2))):
                break
    b = a.copy() if matching else _different(rng, a, difference)
    anchor = regime == "source_train" and matching and kind == "frame"
    backgrounds, fg = _colors(rng, params, regime, force_regular=anchor)
    x = _compose(
        rng, a, b, orientation, backgrounds, fg, params["separator"],
        aligned=matching and rng.random() < 0.32,
        placement=placement,
    )
    info = _analyze(x, params["separator"])
    return {"input": x, "output": _expected(info, params["separator"])}


def _grid_key(x):
    return x.shape, x.tobytes()


def _puzzle_key(puzzle):
    return tuple(
        (split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                      for p in puzzle[split]))
        for split in ("train", "test")
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    params = _parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()

    positive = int(rng.integers(1, ntrain))
    jobs = [(True, "frame", None), (False, "random", "count")]
    remaining_positive = positive - 1
    for i in range(ntrain - 2):
        matching = i < remaining_positive
        kind = ("solid", "frame", "partial_frame", "diagonal",
                "cross", "random")[int(rng.integers(6))]
        difference = None if matching else (
            "size", "count", "swap", "transform"
        )[int(rng.integers(4))]
        jobs.append((matching, kind, difference))
    rng.shuffle(jobs)
    placements = ["inset", "edge"] + [
        ("inset", "edge", "free")[int(rng.choice(3, p=[0.27, 0.32, 0.41]))]
        for _ in range(ntrain - 2)
    ]
    rng.shuffle(placements)

    def add(split, regime, job, index, placement):
        orientation = _orientation(rng, params, regime, index, ntrain)
        for _ in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _pair(pair_rng, params, regime, job, orientation, placement)
            key = _grid_key(pair["input"])
            rows = tuple("".join(str(int(c)) for c in row)
                         for row in pair["input"])
            if key not in seen and rows not in ORIGINAL_INPUTS:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise AssertionError("failed to construct a unique source-free pair")

    for i, job in enumerate(jobs):
        add("train", "source_train", job, i, placements[i])
    for i in range(ntest):
        matching = bool(rng.integers(2))
        kind = ("solid", "frame", "partial_frame", "diagonal",
                "cross", "random")[int(rng.integers(6))]
        difference = None if matching else (
            "size", "count", "swap", "transform"
        )[int(rng.integers(4))]
        placement = ("inset", "edge", "free")[
            int(rng.choice(3, p=[0.25, 0.32, 0.43]))
        ]
        add("test", "target_test", (matching, kind, difference), i,
            placement)
    return puzzle


def _separator_from_train(train):
    negatives = [p for p in train if p["output"].shape == (1, 1)]
    assert negatives
    values = {int(p["output"][0, 0]) for p in negatives}
    assert len(values) == 1
    return values.pop()


def _predict_from_demonstrations(train, test_input):
    # This path receives only visible demonstrations and the unsolved input.
    separator = _separator_from_train(train)
    for example in train:
        info = _analyze(example["input"], separator)
        assert np.array_equal(example["output"], _expected(info, separator))
    return _expected(_analyze(test_input, separator), separator)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    separator = _separator_from_train(puzzle["train"])
    assert 0 <= separator <= 9
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            x, y = pair["input"], pair["output"]
            input_key = _grid_key(x)
            pair_key = (input_key, _grid_key(y))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not np.array_equal(x, y)
            rows = tuple("".join(str(int(c)) for c in row) for row in x)
            assert rows not in ORIGINAL_INPUTS
            info = _analyze(x, separator)
            assert np.array_equal(y, _expected(info, separator))
            infos[split].append(info)
    train = infos["train"]
    assert any(np.array_equal(a[2], b[2]) and np.any(~a[2])
               and a[1] != separator and a[0] != b[0]
               for _, a, b in train)
    assert any(not np.array_equal(a[2], b[2]) for _, a, b in train)
    assert any(a[2].shape == b[2].shape
               and not np.array_equal(a[2], b[2])
               for _, a, b in train)
    for pair in puzzle["test"]:
        assert np.array_equal(
            pair["output"],
            _predict_from_demonstrations(
                [{"input": p["input"].copy(), "output": p["output"].copy()}
                 for p in puzzle["train"]],
                pair["input"].copy(),
            ),
        )


def self_test() -> None:
    keys, train_counts, test_counts = set(), set(), set()
    input_sizes, crop_sizes, masks, palettes = set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    negative_types = set()
    edge_pairs = 0
    aligned = translated = 0
    train_special = test_special = 0
    train_orders = set()
    mixed_train = new_orientation = 0
    nonzero_separators = set()
    first_outcomes, palette_sizes = set(), set()
    small_crops = 0
    same_background = {"train": 0, "test": 0}
    foreground_separator = {"train": 0, "test": 0}
    combined_special = {"train": 0, "test": 0}
    shape_classes = set()
    edge_mixed_train = 0
    familiar_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        sep = _separator_from_train(puzzle["train"])
        if sep:
            nonzero_separators.add(sep)
        palette = set()
        train_orientations = set()
        train_edges = set()
        train_orders.add(tuple(p["output"].shape != (1, 1)
                               for p in puzzle["train"]))
        first_outcomes.add(puzzle["train"][0]["output"].shape != (1, 1))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = pair["input"]
                y = pair["output"]
                orientation, a, b = _analyze(x, sep)
                assert np.array_equal(y, _expected((orientation, a, b), sep))
                input_sizes.add(x.shape)
                crop_sizes.add((a[2].shape, b[2].shape))
                masks.add((a[2].shape, a[2].tobytes()))
                small_crops += a[2].shape == (2, 2)
                if a[2].all():
                    shape_classes.add("filled")
                elif (a[2][0].all() and a[2][-1].all()
                      and a[2][:, 0].all() and a[2][:, -1].all()):
                    shape_classes.add("frame")
                elif a[2].mean() < 0.45:
                    shape_classes.add("sparse")
                else:
                    shape_classes.add("other")
                palette.update(int(c) for c in np.unique(x))
                orientations[split].add(orientation)
                if split == "train":
                    train_orientations.add(orientation)
                    train_edges.add(any(a[4]) or any(b[4]))
                same_background[split] += a[0] == b[0]
                foreground_separator[split] += a[1] == sep
                combined_special[split] += a[0] == b[0] and a[1] == sep
                special = a[0] == b[0] or a[1] == sep
                if special:
                    if split == "train":
                        train_special += 1
                    else:
                        test_special += 1
                if any(a[4]) or any(b[4]):
                    edge_pairs += 1
                if np.array_equal(a[2], b[2]):
                    if a[3] == b[3]:
                        aligned += 1
                    else:
                        translated += 1
                else:
                    if a[2].shape != b[2].shape:
                        negative_types.add("size")
                    elif int(a[2].sum()) != int(b[2].sum()):
                        negative_types.add("count")
                    else:
                        negative_types.add("same_count")
        palettes.add(tuple(sorted(palette)))
        palette_sizes.add(len(palette))
        if len(train_orientations) == 2:
            mixed_train += 1
        if len(train_edges) == 2:
            edge_mixed_train += 1
        if any(_analyze(p["input"], sep)[0] not in train_orientations
               for p in puzzle["test"]):
            new_orientation += 1
        if any(_analyze(p["input"], sep)[0] in train_orientations
               for p in puzzle["test"]):
            familiar_test += 1
    assert len(keys) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(input_sizes) >= 35 and len(crop_sizes) >= 25
    assert len(masks) >= 100 and len(palettes) >= 15
    assert orientations["train"] == orientations["test"] == {"H", "V"}
    assert negative_types == {"size", "count", "same_count"}
    assert edge_pairs >= 30 and aligned >= 5 and translated >= 5
    assert train_special >= 10 and test_special >= 5
    assert len(train_orders) >= 10
    assert mixed_train >= 5 and new_orientation >= 3
    assert len(nonzero_separators) >= 3
    assert first_outcomes == {True, False}
    assert edge_mixed_train == 42
    assert familiar_test >= 5
    assert small_crops >= 2
    assert len(palette_sizes) >= 4
    assert shape_classes == {"filled", "frame", "sparse", "other"}
    assert all(same_background[s] >= 5 for s in ("train", "test"))
    assert all(foreground_separator[s] >= 5 for s in ("train", "test"))
    assert all(combined_special[s] >= 1 for s in ("train", "test"))

    for num_train, num_test in (
        (2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
        (2, None), (10, None),
    ):
        puzzle = generate_puzzle(2718, num_train, num_test)
        validate_generated_puzzle(puzzle)
        assert num_train is None or len(puzzle["train"]) == num_train
        assert num_test is None or len(puzzle["test"]) == num_test
        assert _puzzle_key(puzzle) == _puzzle_key(
            generate_puzzle(2718, num_train, num_test)
        )
    for bad in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(1, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {bad}")
