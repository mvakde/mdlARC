"""Two-dimensional periodic pattern repair puzzle generator."""
import numpy as np

SOURCE_TILES = (
    ("133", "315", "353"),
    ("13443", "32235", "42324", "43215", "35453"),
    ("1325523", "3735665", "2351532", "5517271", "5652442", "2637415", "3521253"),
    ("13131313", "37117355", "11111111", "31137557",
     "17171717", "33157751", "15151515", "35177153"),
)


def _expand(tile, shape):
    return tile[np.arange(shape[0])[:, None] % tile.shape[0],
                np.arange(shape[1])[None, :] % tile.shape[1]].astype(np.int64)


def _axis_period(grid, axis):
    length = grid.shape[axis]
    for p in range(1, length):
        if axis == 0 and np.array_equal(grid[p:], grid[:-p]):
            return p
        if axis == 1 and np.array_equal(grid[:, p:], grid[:, :-p]):
            return p
    return length


def _source_outputs():
    return [_expand(np.array([[int(v) for v in row] for row in rows],
                             dtype=np.int64), (22, 22)) for rows in SOURCE_TILES]


def _resolve_counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None:
        options = [2, 3, 3, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10]
        num_train = int(rng.choice([n for n in options if num_test is None or n > num_test]))
    if num_test is None:
        options = [1, 1, 1, 1, 2, 2, 3, 4]
        num_test = int(rng.choice([n for n in options if n < num_train]))
    return int(num_train), int(num_test)


def _sample_period(rng, square=None, low=2, high=9):
    row = int(rng.integers(low, high + 1))
    if square is True:
        return row, row
    if square is False:
        return row, int(rng.choice([n for n in range(low, high + 1) if n != row]))
    return (row, row) if rng.random() < 0.48 else _sample_period(rng, False, low, high)


def _sample_tile(rng, periods, colors, style):
    ph, pw = periods
    colors = np.asarray(colors, dtype=np.int64)
    k = len(colors)
    for attempt in range(160):
        if style == 0:
            tile = rng.choice(colors, size=periods)
        elif style == 1:
            tile = np.broadcast_to(rng.choice(colors, size=(ph, 1)), periods).copy()
        elif style == 2:
            tile = np.broadcast_to(rng.choice(colors, size=(1, pw)), periods).copy()
        elif style == 3:
            rows, cols = rng.integers(k, size=ph), rng.integers(k, size=pw)
            tile = colors[(rows[:, None] + cols[None, :]) % k]
        elif style == 4:
            tile = np.full(periods, colors[0], dtype=np.int64)
            for _ in range(int(rng.integers(2, 5))):
                top, left = int(rng.integers(ph)), int(rng.integers(pw))
                bottom = min(ph, top + int(rng.integers(1, max(2, ph // 2 + 1))))
                right = min(pw, left + int(rng.integers(1, max(2, pw // 2 + 1))))
                tile[top:bottom, left:right] = int(rng.choice(colors))
        elif style == 5:
            tile = np.full(periods, colors[0], dtype=np.int64)
            tile[int(rng.integers(ph)), :] = colors[1 % k]
            tile[:, int(rng.integers(pw))] = colors[2 % k]
            tile[:max(1, ph // 2), :max(1, pw // 2)] = colors[3 % k]
        elif style == 6:
            rr = np.minimum(np.arange(ph), np.arange(ph)[::-1])[:, None]
            cc = np.minimum(np.arange(pw), np.arange(pw)[::-1])[None, :]
            tile = colors[(rr + 2 * cc) % k].copy()
        else:
            rows, cols = rng.permutation(ph), rng.permutation(pw)
            tile = colors[(rows[:, None] + cols[None, :]) % k]
        if style != 0:
            orientation = int(rng.integers(4))
            if orientation == 1:
                tile = tile[::-1, :].copy()
            elif orientation == 2:
                tile = tile[:, ::-1].copy()
            elif orientation == 3:
                tile = tile.T.copy() if ph == pw else tile[::-1, ::-1].copy()
        accents = 0 if style == 0 else int(rng.integers(1, max(3, ph * pw // 6 + 1)))
        for _ in range(accents):
            r, c = int(rng.integers(ph)), int(rng.integers(pw))
            tile[r, c] = int(rng.choice(colors[colors != tile[r, c]]))
        for color in colors:
            if color not in tile:
                r, c = int(rng.integers(ph)), int(rng.integers(pw))
                tile[r, c] = color
        if len(np.unique(tile)) == k and _axis_period(tile, 0) == ph and _axis_period(tile, 1) == pw:
            return tile
        if attempt == 79:
            style = 0
    # The unique corner accent guarantees both primitive axes, while the
    # remaining colors occupy distinct cells. This also bounds construction.
    tile = np.full(periods, colors[0], dtype=np.int64)
    tile[0, 0] = colors[1]
    positions = rng.choice(np.arange(1, ph * pw), size=k - 2, replace=False)
    for index, color in zip(positions, colors[2:]):
        tile.flat[int(index)] = color
    return tile


def _rectangle(rng, mask, placement, thin=False):
    h, w = mask.shape
    if thin:
        if rng.random() < 0.5:
            rh = int(rng.integers(1, 4))
            rw = int(rng.integers(max(4, w // 3), w - (1 if placement == "interior" else 0) + 1))
        else:
            rh = int(rng.integers(max(4, h // 3), h - (1 if placement == "interior" else 0) + 1))
            rw = int(rng.integers(1, 4))
    else:
        rh = int(rng.integers(2, min(8, h - 2) + 1))
        rw = int(rng.integers(2, min(8, w - 2) + 1))
    if placement == "interior":
        rh, rw = min(rh, h - 2), min(rw, w - 2)
        top, left = int(rng.integers(1, h - rh)), int(rng.integers(1, w - rw))
    elif placement == "edge":
        side = int(rng.integers(4))
        top = 0 if side == 0 else h - rh if side == 1 else int(rng.integers(h - rh + 1))
        left = 0 if side == 2 else w - rw if side == 3 else int(rng.integers(w - rw + 1))
    else:
        top, left = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
    mask[top:top + rh, left:left + rw] = True


def _sample_mask(rng, shape, mode, placement, hard):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if mode == "rectangles":
        for i in range(int(rng.integers(1, 5 if hard else 4))):
            _rectangle(rng, mask, placement if i == 0 else "any")
    elif mode == "connected":
        if placement == "interior":
            start = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        elif placement == "edge":
            start = (0 if rng.random() < 0.5 else h - 1), int(rng.integers(w))
        else:
            start = int(rng.integers(h)), int(rng.integers(w))
        mask[start] = True
        target = int(rng.integers(8, min(h * w // (4 if hard else 7), 85) + 1))
        for _ in range(target - 1):
            candidates = []
            for r, c in np.argwhere(mask):
                for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        if placement != "interior" or (0 < nr < h-1 and 0 < nc < w-1):
                            candidates.append((nr, nc))
            if not candidates:
                break
            mask[candidates[int(rng.integers(len(candidates)))]] = True
    elif mode == "thin":
        _rectangle(rng, mask, placement, thin=True)
        if hard and rng.random() < 0.45:
            _rectangle(rng, mask, "any", thin=True)
    elif mode == "sparse":
        count = int(rng.integers(4, 16 if hard else 11))
        points = [(r, c) for r in range(1 if placement == "interior" else 0,
                                       h - (1 if placement == "interior" else 0))
                  for c in range(1 if placement == "interior" else 0,
                                 w - (1 if placement == "interior" else 0))]
        for i in rng.choice(len(points), size=count, replace=False):
            mask[points[int(i)]] = True
        if placement == "edge":
            mask[0, int(rng.integers(w))] = True
    else:
        for i in range(int(rng.integers(2, 6 if hard else 4))):
            _rectangle(rng, mask, placement if i == 0 else "any")
        if rng.random() < 0.45:
            for _ in range(int(rng.integers(1, 4))):
                r, c = int(rng.integers(h)), int(rng.integers(w))
                if placement != "interior" or (0 < r < h-1 and 0 < c < w-1):
                    mask[r, c] = True
    return mask


def _completion_is_unique(inp, out, erasure, periods):
    h, w = inp.shape
    visible = inp != erasure
    if not np.any(~visible):
        return False
    rows, cols = np.nonzero(visible)
    values = inp[rows, cols]
    rp = [p for p in range(1, h // 2 + 1)
          if np.all((~visible[p:] | ~visible[:-p]) | (inp[p:] == inp[:-p]))]
    cp = [q for q in range(1, w // 2 + 1)
          if np.all((~visible[:, q:] | ~visible[:, :-q]) | (inp[:, q:] == inp[:, :-q]))]
    if periods[0] not in rp or periods[1] not in cp:
        return False
    for p in rp:
        for q in cp:
            phase = (rows % p) * q + cols % q
            low = np.full(p * q, 10, dtype=np.int64)
            high = np.full(p * q, -1, dtype=np.int64)
            np.minimum.at(low, phase, values)
            np.maximum.at(high, phase, values)
            if np.any((high >= 0) & (low != high)):
                continue
            if np.any(high < 0):
                return False
            completion = low[(np.arange(h)[:, None] % p) * q + np.arange(w)[None, :] % q]
            if not np.array_equal(completion, out):
                return False
    return True


def _make_pair(rng, erasure, color_pool, periods, style, mode, placement, hard):
    ph, pw = periods
    h = int(rng.integers(max(11, 2 * ph + 1), 31))
    w = int(rng.integers(max(11, 2 * pw + 1), 31))
    if rng.random() < 0.24:
        h = w = int(rng.integers(max(11, 2 * ph + 1, 2 * pw + 1), 31))
    k = int(rng.integers(2, min(8, ph * pw, len(color_pool)) + 1))
    colors = rng.choice(color_pool, size=k, replace=False)
    tile = _sample_tile(rng, periods, colors, style)
    out = _expand(tile, (h, w))
    for _ in range(80):
        mask = _sample_mask(rng, out.shape, mode, placement, hard)
        if hard and rng.random() < 0.38:
            phase = (int(rng.integers(ph)), int(rng.integers(pw)))
            copies = [(r, c) for r in range(phase[0], h, ph)
                      for c in range(phase[1], w, pw)]
            keeper = copies[int(rng.integers(len(copies)))]
            for cell in copies:
                if cell != keeper:
                    mask[cell] = True
        if not 4 <= int(mask.sum()) <= int(out.size * 0.30):
            continue
        counts = np.zeros(periods, dtype=np.int64)
        rr, cc = np.nonzero(~mask)
        np.add.at(counts, (rr % ph, cc % pw), 1)
        if int(counts.min()) < (1 if hard else 2):
            continue
        inp = out.copy()
        inp[mask] = erasure
        if _completion_is_unique(inp, out, erasure, periods):
            return {"input": inp, "output": out}
    mask = np.zeros(out.shape, dtype=bool)
    mask[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1] = True
    inp = out.copy()
    inp[mask] = erasure
    if _completion_is_unique(inp, out, erasure, periods):
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a recoverable pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    erasure = int(rng.integers(10))
    available = np.array([c for c in range(10) if c != erasure], dtype=np.int64)
    shared = rng.random() < 0.32
    pool = rng.choice(available, size=int(rng.integers(5, 10)), replace=False) if shared else available
    geometry = int(rng.integers(3))
    train_periods = []
    for i in range(num_train):
        square = True if geometry == 0 else (i == 0 if geometry == 1 else i == num_train - 1)
        if i >= 2 and geometry != 0:
            square = bool(rng.integers(2))
        train_periods.append(_sample_period(rng, square, 2, 8))
    rng.shuffle(train_periods)
    puzzle = {"train": [], "test": []}
    seen = set()
    originals = _source_outputs()

    def append_pair(section, periods):
        for _ in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            style = int(pair_rng.integers(8))
            mode = str(pair_rng.choice(["rectangles", "connected", "thin", "sparse", "mixed"],
                                       p=[0.32, 0.19, 0.16, 0.10, 0.23]))
            placement = str(pair_rng.choice(["interior", "edge", "any"], p=[0.43, 0.27, 0.30]))
            hard = section == "test" and pair_rng.random() < 0.36
            pair = _make_pair(pair_rng, erasure, pool, periods, style, mode, placement, hard)
            key = pair["input"].shape, pair["input"].tobytes()
            if key not in seen and not any(np.array_equal(pair["output"], x) for x in originals):
                seen.add(key)
                puzzle[section].append(pair)
                return
        raise RuntimeError("could not construct a distinct pair")

    for periods in train_periods:
        append_pair("train", periods)
    for _ in range(num_test):
        choice = int(rng.integers(3))
        if choice == 0:
            periods = train_periods[int(rng.integers(len(train_periods)))]
        elif choice == 1:
            old = train_periods[int(rng.integers(len(train_periods)))]
            periods = (max(2, old[0] - int(rng.integers(1, 4))),
                       max(2, old[1] - int(rng.integers(1, 4))))
        else:
            periods = _sample_period(rng, None, 6, 10)
        if periods[0] != periods[1] and all(a == b for a, b in train_periods):
            periods = (periods[0], periods[0])
        append_pair("test", periods)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, erasures = set(), set()
    originals = _source_outputs()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
        assert inp.shape == out.shape and inp.ndim == 2
        assert all(1 <= n <= 30 for n in inp.shape)
        for grid in (inp, out):
            assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
        key = inp.shape, inp.tobytes()
        assert key not in seen
        seen.add(key)
        changed = inp != out
        assert np.any(changed) and len(np.unique(inp[changed])) == 1
        erasure = int(inp[changed][0])
        erasures.add(erasure)
        assert not np.any(out == erasure)
        periods = _axis_period(out, 0), _axis_period(out, 1)
        assert periods[0] <= inp.shape[0] // 2 and periods[1] <= inp.shape[1] // 2
        assert np.array_equal(out, _expand(out[:periods[0], :periods[1]], inp.shape))
        assert _completion_is_unique(inp, out, erasure, periods)
        assert not any(np.array_equal(out, original) for original in originals)
    assert len(erasures) == 1


def _same_puzzle(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, periods, palette_sizes = set(), set(), set(), set(), set()
    masks, tiles = set(), set()
    rectangular = new_test_color = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set(int(c) for pair in puzzle["train"] for c in np.unique(pair["output"]))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                ph, pw = _axis_period(out, 0), _axis_period(out, 1)
                shapes.add(inp.shape)
                periods.add((ph, pw))
                palette_sizes.add(len(np.unique(out)))
                masks.add((inp.shape, (inp != out).tobytes()))
                tiles.add((ph, pw, out[:ph, :pw].tobytes()))
                rectangular += ph != pw
                if section == "test":
                    new_test_color += bool(set(int(c) for c in np.unique(out)) - train_colors)
    assert len(train_counts) >= 5 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(periods) >= 20 and rectangular >= 30
    assert len(palette_sizes) >= 5 and len(masks) >= 100 and len(tiles) >= 100
    assert new_test_color >= 1
    for args in ((2, 1), (3, 1), (10, 4), (None, 4), (2, None), (None, None)):
        a, b = generate_puzzle(891, *args), generate_puzzle(891, *args)
        assert _same_puzzle(a, b)
        if args[0] is not None:
            assert len(a["train"]) == args[0]
        if args[1] is not None:
            assert len(a["test"]) == args[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2),
                 (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(891, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
