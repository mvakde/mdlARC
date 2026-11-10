import numpy as np


def _construct_output(input_grid: np.ndarray, background: int, frame: int) -> np.ndarray:
    h, w = input_grid.shape
    tile = np.rot90(input_grid)
    result = np.full((2 * w + 3, h + 2), background, dtype=np.int8)
    for row in (0, w + 1, 2 * w + 2):
        result[row, 1:-1] = frame
    for start in (1, w + 2):
        result[start:start + w, 0] = frame
        result[start:start + w, -1] = frame
        result[start:start + w, 1:-1] = tile
    return result


def _rows(*lines: str) -> np.ndarray:
    return np.array([[int(c) for c in line] for line in lines], dtype=np.int8)


_SOURCE_INPUTS = (
    _rows("557777", "557777", "597777", "557777", "777777",
          "777755", "777755", "777759", "777755"),
    _rows("55577777", "55577777", "95577777", "77777777",
          "77777555", "77777555", "77777955"),
    _rows("59777", "55777", "77777", "77759", "77755"),
    _rows("5555777777", "5555775555", "5555775555",
          "9555775555", "7777779555"),
)


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_KEYS = {_key(grid) for grid in _SOURCE_INPUTS}
_LEGAL_COUNTS = tuple(
    (train, test) for test in range(1, 5) for train in range(test + 1, 11)
)


def _counts(rng: np.random.Generator, seed: int,
            num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        return _LEGAL_COUNTS[seed % len(_LEGAL_COUNTS)]
    if num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _canvas(rng: np.random.Generator, split: str, shift: str) -> tuple[int, int]:
    # The source's wide test is one possible shift, not a split-wide rule.
    if shift == "source":
        if split == "train":
            return int(rng.integers(5, 12)), int(rng.integers(5, 10))
        return int(rng.integers(5, 10)), int(rng.integers(10, 14))
    if split == "test" and shift == "smaller" and rng.random() < 0.55:
        return int(rng.integers(4, 8)), int(rng.integers(4, 8))
    h_band = int(rng.choice(3, p=[0.31, 0.43, 0.26]))
    w_band = int(rng.choice(3, p=[0.32, 0.40, 0.28]))
    h_bounds = ((4, 7), (7, 15), (15, 29))[h_band]
    w_bounds = ((4, 7), (7, 11), (11, 14))[w_band]
    return int(rng.integers(*h_bounds)), int(rng.integers(*w_bounds))


def _motif(rng: np.random.Generator, mh: int, mw: int, bg: int,
           colors: list[int], style: str) -> np.ndarray:
    primary, accent, third = colors[:3]
    motif = np.full((mh, mw), primary, dtype=np.int8)
    area = mh * mw
    if style == "simple":
        motif[int(rng.integers(mh)), int(rng.integers(mw))] = accent
    elif style == "marked":
        n = int(rng.integers(2, min(6, area)))
        cells = rng.choice(area, size=n, replace=False)
        motif.flat[cells] = accent
    elif style == "stripe":
        if rng.random() < 0.5:
            motif[int(rng.integers(mh)), :] = accent
        else:
            motif[:, int(rng.integers(mw))] = accent
    elif style == "holes":
        if mh >= 3 and mw >= 3:
            motif[int(rng.integers(1, mh - 1)),
                  int(rng.integers(1, mw - 1))] = bg
        free = np.flatnonzero(motif != bg)
        motif.flat[int(rng.choice(free))] = accent
    elif style == "shape":
        # Two arms of variable thickness make a connected, asymmetric object.
        motif[:] = bg
        thick_r = int(rng.integers(1, max(2, mh)))
        thick_c = int(rng.integers(1, max(2, mw)))
        thick_r = min(thick_r, mh - 1)
        thick_c = min(thick_c, mw - 1)
        motif[:thick_r, :] = primary
        motif[:, :thick_c] = primary
        if rng.random() < 0.5:
            motif = np.flipud(motif)
        if rng.random() < 0.5:
            motif = np.fliplr(motif)
        free = np.flatnonzero(motif != bg)
        motif.flat[int(rng.choice(free))] = accent
    elif style == "multicolor":
        if rng.random() < 0.5:
            motif[:, ::2] = accent
        else:
            motif[::2, :] = accent
        motif[int(rng.integers(motif.shape[0])),
              int(rng.integers(motif.shape[1]))] = third
    return motif


def _layout(mh: int, mw: int, h: int, w: int) -> str:
    row = 2 * mh - h
    col = 2 * mw - w
    if row < 0 and col < 0:
        return "separated"
    if row > 0 and col < 0:
        return "row_overlap"
    if col > 0 and row < 0:
        return "col_overlap"
    return "touching"


def _paired(rng: np.random.Generator, grid: np.ndarray, bg: int,
            colors: list[int], style: str, preferred: str | None) -> bool:
    h, w = grid.shape
    minimum = 3 if style == "holes" else 2
    choices = [
        (mh, mw) for mh in range(minimum, min(7, h))
        for mw in range(minimum, min(7, w))
        if 2 * mh <= h or 2 * mw <= w
    ]
    if not choices:
        return False
    if preferred is not None:
        selected = [x for x in choices if _layout(*x, h, w) == preferred]
        if selected:
            choices = selected
    mh, mw = choices[int(rng.integers(len(choices)))]
    motif = _motif(rng, mh, mw, bg, colors, style)
    if rng.random() < 0.48:
        offsets = [(0, 0, 0, 0)]
    else:
        offsets = [
            (a, b, c, d)
            for a in range(min(3, h - mh + 1))
            for b in range(min(3, w - mw + 1))
            for c in range(min(3, h - mh + 1))
            for d in range(min(3, w - mw + 1))
            if a + c + 2 * mh <= h or b + d + 2 * mw <= w
        ]
    a, b, c, d = offsets[int(rng.integers(len(offsets)))]
    grid[a:a + mh, b:b + mw] = motif
    grid[h - mh - c:h - c, w - mw - d:w - d] = motif
    return bool(grid[0, -1] == bg and grid[-1, 0] == bg)


def _objects(rng: np.random.Generator, grid: np.ndarray, bg: int,
             palette: list[int], mode: str) -> bool:
    h, w = grid.shape
    count = {"single": 1, "unequal": 2, "many": int(rng.integers(3, 5))}[mode]
    if mode == "many" and (h < 7 or w < 7):
        return False
    boxes = []
    for index in range(count):
        placed = False
        for _ in range(90):
            size_limit = 3 if mode == "many" else 6
            mh = int(rng.integers(2, min(size_limit, h) + 1))
            mw = int(rng.integers(2, min(size_limit, w) + 1))
            if mode == "unequal" and index == 1 and (mh, mw) == boxes[0][2:]:
                continue
            r = int(rng.integers(0, h - mh + 1))
            c = int(rng.integers(0, w - mw + 1))
            if (r == 0 and c + mw == w) or (r + mh == h and c == 0):
                continue
            if any(
                r <= br + bh and br <= r + mh
                and c <= bc + bw and bc <= c + mw
                for br, bc, bh, bw in boxes
            ):
                continue
            colors = list(rng.choice(palette, size=3, replace=False))
            style = str(rng.choice(
                ["simple", "marked", "stripe", "shape", "multicolor"]
            ))
            grid[r:r + mh, c:c + mw] = _motif(
                rng, mh, mw, bg, colors, style
            )
            boxes.append((r, c, mh, mw))
            placed = True
            break
        if not placed:
            return False
    return bool(grid[0, -1] == bg and grid[-1, 0] == bg)


def _make_input(rng: np.random.Generator, frame: int, split: str,
                shift: str, forbidden_bg: int | None,
                allowed_modes: tuple[str, ...] | None) -> np.ndarray:
    modes = ("simple", "marked", "stripe", "holes", "shape",
             "multicolor", "single", "unequal", "many")
    weights = (0.16, 0.13, 0.12, 0.09, 0.11, 0.11, 0.06, 0.12, 0.10)
    for _ in range(120):
        h, w = _canvas(rng, split, shift)
        palette = [c for c in range(10) if c != frame and c != forbidden_bg]
        bg = int(rng.choice(palette))
        content = [c for c in range(10) if c not in (frame, bg)]
        colors = list(rng.choice(content, size=3, replace=False))
        grid = np.full((h, w), bg, dtype=np.int8)
        mode = str(rng.choice(modes, p=weights)) if allowed_modes is None else (
            str(rng.choice(allowed_modes))
        )
        if mode in ("single", "unequal", "many"):
            okay = _objects(rng, grid, bg, content, mode)
        else:
            preferred = str(rng.choice(
                ("separated", "row_overlap", "col_overlap", "touching"),
                p=(0.42, 0.23, 0.23, 0.12)
            ))
            if split == "test" and shift == "source" and rng.random() < 0.5:
                preferred = "row_overlap"
            okay = _paired(rng, grid, bg, colors, mode, preferred)
        if okay and len(np.unique(grid)) >= 3 and _key(grid) not in _SOURCE_KEYS:
            return grid
    raise ValueError("could not construct a legal input")


def _symmetries(grid: np.ndarray) -> list[np.ndarray]:
    flipped = np.fliplr(grid)
    return [np.rot90(grid, k) for k in range(4)] + [
        np.rot90(flipped, k) for k in range(4)
    ]


def _direction_unique(train: list[dict]) -> bool:
    for alternative in (0, 2, 3, 4, 5, 6, 7):
        if all(
            _symmetries(pair["input"])[alternative].shape
            == (pair["input"].shape[1], pair["input"].shape[0])
            and np.array_equal(
                _symmetries(pair["input"])[alternative],
                pair["output"][1:pair["input"].shape[1] + 1, 1:-1],
            )
            for pair in train
        ):
            return False
    return True


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
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    frame = int(rng.integers(10))
    shift = str(rng.choice(("source", "similar", "smaller")))
    content = str(rng.choice(("classic", "motifs", "mixed"),
                             p=(0.12, 0.18, 0.70)))
    puzzle = {"train": [], "test": []}
    used = set()
    first_bg = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(120):
                forbidden = first_bg if split == "train" and index == 1 else None
                if content == "classic":
                    allowed = ("simple",)
                elif content == "motifs":
                    allowed = ("simple", "marked", "stripe", "holes",
                               "shape", "multicolor")
                elif split == "train" and index == 0:
                    allowed = ("unequal", "many")
                else:
                    allowed = None
                inp = _make_input(rng, frame, split, shift, forbidden, allowed)
                key = _key(inp)
                if key in used:
                    continue
                bg = int(inp[0, -1])
                out = _construct_output(inp, bg, frame)
                pair = {"input": inp, "output": out}
                if split == "train" and index == count - 1:
                    if not _direction_unique(puzzle["train"] + [pair]):
                        continue
                puzzle[split].append(pair)
                used.add(key)
                if split == "train" and index == 0:
                    first_bg = bg
                break
            else:
                raise ValueError("could not construct distinct, informative pairs")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair: dict) -> tuple[int, int, int, int]:
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert all(1 <= x <= 30 for x in grid.shape)
        assert np.all((grid >= 0) & (grid <= 9))
    h, w = inp.shape
    assert h <= 28 and w <= 13
    assert out.shape == (2 * w + 3, h + 2)
    bg, frame = int(out[0, 0]), int(out[0, 1])
    assert bg != frame and frame not in inp
    assert int(inp[0, -1]) == bg and int(inp[-1, 0]) == bg
    assert len(np.unique(inp)) >= 3
    # Inspect output regions independently from _construct_output.
    for row in (0, w + 1, 2 * w + 2):
        assert int(out[row, 0]) == bg and int(out[row, -1]) == bg
        assert np.all(out[row, 1:-1] == frame)
    for start in (1, w + 2):
        assert np.all(out[start:start + w, 0] == frame)
        assert np.all(out[start:start + w, -1] == frame)
        for r in range(w):
            for c in range(h):
                assert out[start + r, c + 1] == inp[c, w - 1 - r]
    assert _key(inp) not in _SOURCE_KEYS
    return h, w, bg, frame


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        nt, ne = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
        seen = set()
        frames = set()
        backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _, _, bg, frame = _inspect_pair(pair)
                key = _key(pair["input"])
                assert key not in seen
                seen.add(key)
                frames.add(frame)
                if split == "train":
                    backgrounds.add(bg)
        assert len(frames) == 1
        assert len(backgrounds) >= 2
        assert _direction_unique(puzzle["train"])
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _same(a: dict, b: dict) -> bool:
    return all(
        len(a[s]) == len(b[s]) and all(
            np.array_equal(x[t], y[t])
            for x, y in zip(a[s], b[s]) for t in ("input", "output")
        )
        for s in ("train", "test")
    )


def _observed_features(pair: dict) -> dict:
    inp = pair["input"]
    bg = int(pair["output"][0, 0])
    remaining = set(zip(*np.nonzero(inp != bg)))
    groups = []
    while remaining:
        group = {remaining.pop()}
        stack = list(group)
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c),
                          (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    group.add(point)
                    stack.append(point)
        groups.append(group)
    boxes = []
    for group in groups:
        rows = [p[0] for p in group]
        cols = [p[1] for p in group]
        boxes.append((min(rows), max(rows), min(cols), max(cols)))
    matching = False
    layout = None
    if len(groups) == 2:
        a, b = boxes
        crop_a = inp[a[0]:a[1] + 1, a[2]:a[3] + 1]
        crop_b = inp[b[0]:b[1] + 1, b[2]:b[3] + 1]
        matching = np.array_equal(crop_a, crop_b)
        row_overlap = max(a[0], b[0]) <= min(a[1], b[1])
        col_overlap = max(a[2], b[2]) <= min(a[3], b[3])
        layout = ("row_overlap" if row_overlap else
                  "col_overlap" if col_overlap else "separated")
    hole = any(
        inp[r, c] == bg and all(
            inp[rr, cc] != bg for rr, cc in
            ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        )
        for top, bottom, left, right in boxes
        for r in range(top + 1, bottom)
        for c in range(left + 1, right)
    )
    values, counts = np.unique(inp, return_counts=True)
    dominant = int(values[int(np.argmax(counts))])
    return {
        "components": len(groups),
        "matching": matching,
        "layout": layout,
        "hole": hole,
        "off_corner": bool(inp[0, 0] == bg and inp[-1, -1] == bg),
        "background_not_dominant": bg != dominant,
        "colors": len(values),
    }


def self_test() -> None:
    puzzles = []
    train_counts, test_counts = set(), set()
    frames, backgrounds, shapes, structures = set(), set(), set(), set()
    component_counts = set()
    layouts = set()
    observed = {
        "train": {"matching": 0, "unequal": 0, "hole": 0,
                  "off_corner": 0, "multicolor": 0},
        "test": {"matching": 0, "unequal": 0, "hole": 0,
                 "off_corner": 0, "multicolor": 0},
    }
    non_dominant = 0
    smaller_tests = 0
    wider_tests = 0
    train_input_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        smaller_tests += any(
            t["input"].size < a["input"].size
            for t in puzzle["test"] for a in puzzle["train"]
        )
        wider_tests += (
            min(t["input"].shape[1] for t in puzzle["test"]) >
            max(a["input"].shape[1] for a in puzzle["train"])
        )
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        frames.add(int(puzzle["train"][0]["output"][0, 1]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg = int(pair["output"][0, 0])
                backgrounds.add(bg)
                shapes.add((split, inp.shape))
                structure = tuple(int(x) for x in
                                  np.unique(inp, return_inverse=True)[1].flat)
                structures.add((split, inp.shape, structure))
                if split == "train":
                    train_input_keys.add(_key(inp))
                feature = _observed_features(pair)
                component_counts.add(feature["components"])
                layouts.add((split, feature["layout"]))
                non_dominant += feature["background_not_dominant"]
                observed[split]["matching"] += feature["matching"]
                observed[split]["unequal"] += (
                    feature["components"] == 2 and not feature["matching"]
                )
                observed[split]["hole"] += feature["hole"]
                observed[split]["off_corner"] += feature["off_corner"]
                observed[split]["multicolor"] += feature["colors"] >= 4
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(frames) >= 8 and len(backgrounds) >= 8
    assert len(shapes) >= 50 and len(structures) >= 100
    assert {1, 2, 3}.issubset(component_counts)
    assert len(train_input_keys) >= 150
    assert all(
        (split, layout) in layouts
        for split in ("train", "test")
        for layout in ("separated", "row_overlap", "col_overlap")
    )
    assert all(
        all(count > 0 for count in observed[split].values())
        for split in ("train", "test")
    )
    assert non_dominant > 0 and smaller_tests > 0 and wider_tests > 0
    assert any(
        any(pair["input"].shape[0] >= 20 for pair in p["train"] + p["test"])
        for p in puzzles
    )
    for seed, nt, ne in ((500, 2, 1), (501, 10, 1), (502, 10, 4)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
    for seed, nt, ne in ((600, None, 1), (601, None, 4),
                         (602, 2, None), (603, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert (nt is None or len(puzzle["train"]) == nt)
        assert (ne is None or len(puzzle["test"]) == ne)
    first = generate_puzzle(314159, 6, 3)
    assert _same(first, generate_puzzle(314159, 6, 3))
    assert not _same(first, generate_puzzle(314160, 6, 3))
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
