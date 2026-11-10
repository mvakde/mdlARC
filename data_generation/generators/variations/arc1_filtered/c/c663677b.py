import numpy as np


SOURCE_TILES = tuple(
    np.array([[int(ch) for ch in row] for row in rows], dtype=np.int64)
    for rows in (
        ("51351351", "15311531", "33331111", "51351351",
         "11111111", "35131351", "53151531", "11111111"),
        ("37413741", "76544321", "45675671", "14736251",
         "34565671", "73626251", "42757531", "11111111"),
        ("31757531", "11111111", "71353571", "51515151",
         "71353571", "51515151", "31757531", "11111111"),
        ("54324321", "45677891", "36931471", "27384951",
         "47141471", "38494951", "29757531", "11111111"),
    )
)


def _key(array):
    return array.shape, np.asarray(array, dtype=np.int64).tobytes()


def _counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None:
        choices = np.arange(max(2, (num_test or 0) + 1), 11)
        weights = np.array([12, 25, 20, 15, 10, 7, 5, 4, 2], dtype=float)
        weights = weights[choices - 2]
        num_train = int(rng.choice(choices, p=weights / weights.sum()))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(num_train), int(num_test)


def _periods(grid):
    height, width = grid.shape
    rows = [p for p in range(1, min(10, height - 1) + 1)
            if np.array_equal(grid[p:, :], grid[:-p, :])]
    cols = [p for p in range(1, min(10, width - 1) + 1)
            if np.array_equal(grid[:, p:], grid[:, :-p])]
    return (rows[0], cols[0]) if rows and cols else None


def _tile(rng, h, w, palette, flat, flat_color):
    k = len(palette)
    for _ in range(90):
        family = int(rng.integers(7))
        if family == 0:
            base = rng.choice(palette, size=w)
            prototypes = np.stack((base, np.roll(base, 1), base[::-1]))
            tile = prototypes[rng.integers(len(prototypes), size=h)].copy()
        elif family == 1:
            base = rng.choice(palette, size=h)
            prototypes = np.stack((base, np.roll(base, 1), base[::-1]), axis=1)
            tile = prototypes[:, rng.integers(prototypes.shape[1], size=w)].copy()
        elif family == 2:
            row_codes = rng.integers(k, size=h)
            col_codes = rng.integers(k, size=w)
            tile = palette[(row_codes[:, None] + col_codes[None, :]) % k].copy()
        elif family == 3:
            patterns = rng.choice(palette, size=(int(rng.integers(2, 4)),
                                                 (w + 1) // 2))
            half = patterns[rng.integers(len(patterns), size=h)]
            tile = np.concatenate((half, half[:, :w // 2][:, ::-1]), axis=1)
            for r in range(1, h, 2):
                tile[r] = np.roll(tile[r - 1], int(rng.integers(1, w)))
        elif family == 4:
            tile = np.full((h, w), palette[0], dtype=np.int64)
            for color in palette[1:]:
                r = int(rng.integers(h))
                c = int(rng.integers(w))
                tile[r:r + int(rng.integers(1, 3)),
                     c:c + int(rng.integers(1, 3))] = color
            if rng.random() < .5:
                tile[:, int(rng.integers(w))] = palette[int(rng.integers(k))]
        elif family == 5:
            base = rng.choice(palette, size=w)
            tile = np.stack([np.roll(base, (r // 2) + (r % 2) * 2)
                             for r in range(h)])
            tile[1::2] = np.flip(tile[1::2], axis=1)
        else:
            tile = np.empty((h, w), dtype=np.int64)
            for r in range(h):
                if r and rng.random() < .45:
                    tile[r] = np.roll(tile[int(rng.integers(r))],
                                      int(rng.integers(w)))
                else:
                    tile[r] = rng.choice(palette, size=w)

        flat_rows = []
        if flat:
            flat_rows.append(int(rng.integers(h)))
            if h >= 6 and rng.random() < .23:
                candidates = [r for r in range(h) if r not in flat_rows]
                flat_rows.append(int(rng.choice(candidates)))
            for r in flat_rows:
                tile[r] = (int(rng.choice(palette)) if flat_color is None
                           else flat_color)

        eligible = [(r, c) for r in range(h) if r not in flat_rows
                    for c in range(w)]
        rng.shuffle(eligible)
        for color, (r, c) in zip(palette, eligible):
            tile[r, c] = color

        if flat_color is not None:
            for r in range(h):
                if np.all(tile[r] == tile[r, 0]):
                    tile[r] = flat_color

        if len(np.unique(tile)) != k:
            continue
        if not flat and any(np.all(tile[r] == tile[r, 0]) for r in range(h)):
            continue
        if any(np.array_equal(tile, np.roll(tile, d, axis=axis))
               for axis, size in ((0, h), (1, w)) for d in range(1, size)):
            continue
        return tile.astype(np.int64)
    raise RuntimeError("tile sampling failed")


def _paint(mask, r, c, h, w):
    height, width = mask.shape
    mask[max(0, r):min(height, r + h), max(0, c):min(width, c + w)] = True


def _position(rng, height, width, h, w):
    def inner(limit, extent):
        if limit - extent >= 2:
            return int(rng.integers(1, limit - extent))
        return int(rng.integers(0, max(1, limit - extent + 1)))

    if rng.random() < .15:
        side = int(rng.integers(4))
        if side == 0:
            return 0, inner(width, w)
        if side == 1:
            return height - h, inner(width, w)
        if side == 2:
            return inner(height, h), 0
        return inner(height, h), width - w
    if rng.random() < .07:
        return int(rng.integers(-h // 3, height)), int(rng.integers(-w // 3, width))
    return inner(height, h), inner(width, w)


def _draw_shape(mask, rng, broad):
    height, width = mask.shape
    style = int(rng.integers(6))
    if style == 0:
        h, w = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        r, c = _position(rng, height, width, h, w)
        _paint(mask, r, c, h, w)
    elif style == 1:
        horizontal = bool(rng.random() < .5)
        limit = width if horizontal else height
        long = int(rng.integers(8, min(20, limit - 1) + 1))
        short = int(rng.integers(1, 4))
        h, w = (short, long) if horizontal else (long, short)
        r, c = _position(rng, height, width, h, w)
        _paint(mask, r, c, h, w)
    elif style == 2:
        h, w = int(rng.integers(5, 12)), int(rng.integers(5, 12))
        t = int(rng.integers(2, 5))
        r, c = _position(rng, height, width, h, w)
        _paint(mask, r, c, h, t)
        _paint(mask, r + h - t if rng.random() < .5 else r, c, t, w)
    elif style == 3:
        h, w = int(rng.integers(6, 12)), int(rng.integers(6, 12))
        r, c = _position(rng, height, width, h, w)
        t = max(2, h // 3)
        for step in range(3):
            _paint(mask, r + step * t, c + step * max(1, w // 4), t, w // 2 + 1)
    elif style == 4:
        stop = 12 if broad else 8
        h, w = int(rng.integers(5, stop)), int(rng.integers(5, stop))
        r, c = _position(rng, height, width, h, w)
        _paint(mask, r, c, h, w)
        _paint(mask, r + h // 3, c + w - 1, max(2, h // 2),
               int(rng.integers(2, 6)))
    else:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        r, c = _position(rng, height, width, 3 * h + 2, 3 * w + 2)
        for shift in range(3):
            _paint(mask, r + shift * (h + 1), c + shift * (w + 1), h, w)


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                x, y = stack.pop()
                size += 1
                for xx, yy in ((x - 1, y), (x + 1, y),
                               (x, y - 1), (x, y + 1)):
                    if (0 <= xx < height and 0 <= yy < width
                            and mask[xx, yy] and not seen[xx, yy]):
                        seen[xx, yy] = True
                        stack.append((xx, yy))
            sizes.append(size)
    return sizes


def _phase_coverage(mask, h, w):
    return min(np.count_nonzero(~mask[r::h, c::w])
               for r in range(h) for c in range(w))


def _has_square8(mask):
    return any(mask[r:r + 8, c:c + 8].all()
               for r in range(mask.shape[0] - 7)
               for c in range(mask.shape[1] - 7))


def _two_dim_count(mask, h, w):
    rows, cols = np.nonzero(mask)
    return sum(not np.any(~mask[r, c % w::w])
               and not np.any(~mask[r % h::h, c])
               and np.any(~mask[r % h::h, c % w::w])
               for r, c in zip(rows, cols))


def _visible_shorter_periods(inp, marker, h, w):
    visible = inp != marker
    for d in range(1, h):
        common = visible[d:, :] & visible[:-d, :]
        if not np.any(common & (inp[d:, :] != inp[:-d, :])):
            return True
    for d in range(1, w):
        common = visible[:, d:] & visible[:, :-d]
        if not np.any(common & (inp[:, d:] != inp[:, :-d])):
            return True
    return False


def _infer(inp):
    height, width = inp.shape
    solutions = []
    for marker in np.unique(inp):
        visible = inp != marker
        ps = [p for p in range(1, min(10, height - 1) + 1)
              if not np.any((inp[p:, :] != inp[:-p, :])
                            & visible[p:, :] & visible[:-p, :])]
        qs = [q for q in range(1, min(10, width - 1) + 1)
              if not np.any((inp[:, q:] != inp[:, :-q])
                            & visible[:, q:] & visible[:, :-q])]
        if not ps or not qs:
            continue
        h, w = ps[0], qs[0]
        tile = np.empty((h, w), dtype=np.int64)
        good = True
        for r in range(h):
            for c in range(w):
                values = inp[r::h, c::w]
                values = values[values != marker]
                if values.size < 2 or np.any(values != values[0]):
                    good = False
                    break
                tile[r, c] = values[0]
            if not good:
                break
        if good:
            out = np.tile(tile, ((height + h - 1) // h,
                                 (width + w - 1) // w))[:height, :width]
            if np.array_equal(out[visible], inp[visible]):
                solutions.append((int(marker), h, w, out))
    return solutions


def _mask(rng, output, marker, h, w, force_two_dim, broad):
    height, width = output.shape
    for _ in range(90):
        mask = np.zeros((height, width), dtype=bool)
        if force_two_dim:
            r = int(rng.integers(h, height - h))
            c = int(rng.integers(w, width - w))
            mask[r, c % w::w] = True
            mask[r % h::h, c] = True
        fraction = rng.uniform(.20, .34) if broad else rng.uniform(.09, .19)
        if height * width < 360:
            fraction = min(fraction, .25)
        goal = int(height * width * fraction)
        for _ in range(7):
            if np.count_nonzero(mask) >= goal:
                break
            _draw_shape(mask, rng, broad)
        area = int(np.count_nonzero(mask))
        if not (.06 * height * width <= area <= .39 * height * width):
            continue
        if len(_components(mask)) < 2 or _phase_coverage(mask, h, w) < 2:
            continue
        if force_two_dim and not _two_dim_count(mask, h, w):
            continue
        inp = output.copy()
        inp[mask] = marker
        if _visible_shorter_periods(inp, marker, h, w):
            continue
        solutions = _infer(inp)
        if (len(solutions) == 1 and solutions[0][:3] == (marker, h, w)
                and np.array_equal(solutions[0][3], output)):
            return inp
    raise RuntimeError("recoverable mask sampling failed")


def _sample_pair(rng, marker, regime, force_two_dim,
                 flat_color, flat_required):
    for _ in range(25):
        if rng.random() < (.20 if regime == "source_train" else .27):
            h = w = 8
            height = width = 27
        else:
            h, w = rng.integers(4, 10, size=2).tolist()
            height = int(rng.integers(max(16, 2 * h + 3), 31))
            width = int(rng.integers(max(16, 2 * w + 3), 31))
        weights = ([8, 10, 13, 12, 10, 8, 6] if regime == "source_train"
                   else [5, 7, 8, 9, 10, 10, 15])
        weights = np.array(weights, dtype=float)
        k = int(rng.choice(np.arange(3, 10), p=weights / weights.sum()))
        if flat_color is None:
            palette = rng.choice([c for c in range(10) if c != marker],
                                 size=k, replace=False)
        else:
            palette = np.concatenate((np.array([flat_color]),
                                      rng.choice([c for c in range(10)
                                                  if c not in (marker, flat_color)],
                                                 size=k - 1, replace=False)))
        flat = flat_required or bool(rng.random() < .50)
        try:
            tile = _tile(rng, h, w, palette, flat, flat_color)
        except RuntimeError:
            continue
        if rng.random() < .26:
            phase_r = phase_c = 0
        else:
            phase_r = int(rng.integers(h))
            phase_c = int(rng.integers(w))
        tile = np.roll(tile, (-phase_r, -phase_c), axis=(0, 1))
        out = np.tile(tile, ((height + h - 1) // h,
                             (width + w - 1) // w))[:height, :width].copy()
        if _periods(out) != (h, w):
            continue
        if (out.shape == (27, 27) and (h, w) == (8, 8)
                and any(np.array_equal(out[:8, :8], source)
                        for source in SOURCE_TILES)):
            continue
        broad = bool(rng.random() < (.27 if regime == "source_train" else .32))
        try:
            inp = _mask(rng, out, marker, h, w, force_two_dim, broad)
        except RuntimeError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("pair sampling failed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    for _ in range(30):
        shared_marker = bool(rng.random() < .43)
        shared_line = bool(rng.random() < .32)
        fixed_line = int(rng.integers(10)) if shared_line else None
        fixed_marker = int(rng.choice([c for c in range(10)
                                       if c != fixed_line]))
        demonstration = int(rng.integers(num_train))
        puzzle = {"train": [], "test": []}
        used_inputs = set()
        train_markers = []
        train_flat_colors = []
        try:
            for split, count in (("train", num_train), ("test", num_test)):
                for index in range(count):
                    for _ in range(25):
                        if shared_marker:
                            marker = fixed_marker
                        elif split == "train" and index == 1:
                            marker = int(rng.choice([c for c in range(10)
                                                     if c != train_markers[0]
                                                     and c != fixed_line]))
                        else:
                            marker = int(rng.choice([c for c in range(10)
                                                     if c != fixed_line]))
                        if shared_line:
                            flat_color = fixed_line
                            flat_required = True
                        elif split == "train" and index < 2:
                            unavailable = {marker}
                            if train_flat_colors:
                                unavailable.add(train_flat_colors[0])
                            flat_color = int(rng.choice([c for c in range(10)
                                                         if c not in unavailable]))
                            flat_required = True
                        else:
                            flat_color = None
                            flat_required = False
                        force = ((split == "train" and index == demonstration)
                                 or (split == "test" and rng.random() < .38))
                        pair_rng = np.random.default_rng(int(rng.integers(2**63)))
                        pair = _sample_pair(pair_rng, marker,
                                            "source_train" if split == "train"
                                            else "target_test", force,
                                            flat_color, flat_required)
                        key = _key(pair["input"])
                        if key in used_inputs:
                            continue
                        used_inputs.add(key)
                        puzzle[split].append(pair)
                        if split == "train":
                            train_markers.append(marker)
                            if not shared_line and index < 2:
                                train_flat_colors.append(flat_color)
                        break
                    else:
                        raise RuntimeError("duplicate pair sampling failed")
            validate_generated_puzzle(puzzle)
            return puzzle
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("puzzle sampling failed")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_markers, test_markers = [], []
    train_periods, train_shapes, train_palettes = set(), set(), set()
    train_flat_colors, test_flat_colors = set(), set()
    every_train_has_flat = True
    train_two_dim = False
    test_two_dim = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            assert _key(inp) not in inputs
            assert (_key(inp), _key(out)) not in pairs
            inputs.add(_key(inp))
            pairs.add((_key(inp), _key(out)))
            colors_in = set(np.unique(inp).tolist())
            colors_out = set(np.unique(out).tolist())
            extra = colors_in - colors_out
            assert len(extra) == 1 and colors_out == colors_in - extra
            marker = int(next(iter(extra)))
            (train_markers if split == "train" else test_markers).append(marker)
            mask = inp == marker
            assert np.any(mask) and not np.array_equal(inp, out)
            assert np.array_equal(inp[~mask], out[~mask])
            assert np.all(inp[mask] != out[mask])
            period = _periods(out)
            assert period is not None
            h, w = period
            height, width = out.shape
            assert 4 <= h <= 9 and 4 <= w <= 9
            assert height >= max(16, 2 * h + 3)
            assert width >= max(16, 2 * w + 3)
            assert 3 <= len(colors_out) <= 9
            flat_rows = [r for r in range(h)
                         if np.all(out[r, :w] == out[r, 0])]
            if split == "train":
                every_train_has_flat &= bool(flat_rows)
                train_flat_colors.update(int(out[r, 0]) for r in flat_rows)
            else:
                test_flat_colors.update(int(out[r, 0]) for r in flat_rows)
            assert _phase_coverage(mask, h, w) >= 2
            assert not _visible_shorter_periods(inp, marker, h, w)
            assert len(_components(mask)) >= 2
            assert .06 <= np.mean(mask) <= .39
            solutions = _infer(inp)
            assert len(solutions) == 1
            found_marker, found_h, found_w, found_output = solutions[0]
            assert (found_marker, found_h, found_w) == (marker, h, w)
            assert np.array_equal(found_output, out)
            if out.shape == (27, 27) and period == (8, 8):
                assert not any(np.array_equal(out[:8, :8], source)
                               for source in SOURCE_TILES)
            two_dim = _two_dim_count(mask, h, w) > 0
            if split == "train":
                train_periods.add(period)
                train_shapes.add(out.shape)
                train_palettes.add(len(colors_out))
                train_two_dim |= two_dim
            else:
                test_two_dim |= two_dim
    assert len(train_periods) >= 2 and len(train_shapes) >= 2
    assert len(train_palettes) >= 2
    assert train_two_dim
    assert not test_two_dim or train_two_dim
    if len(set(train_markers)) == 1:
        assert all(m == train_markers[0] for m in test_markers)
    if every_train_has_flat and len(train_flat_colors) == 1:
        assert test_flat_colors <= train_flat_colors


def _equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[part], y[part])
        for x, y in zip(a[split], b[split]) for part in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, periods, palettes, markers = set(), set(), set(), set()
    source_size = rectangular = flat = no_flat = 0
    broad_train = compact_test = rich_modest_test = familiar_test = 0
    two_dim_train = two_dim_test = border = 0
    full8_train = no8_test = source_like_test = 0
    small = large = no_border = rich_train = rich_test = 0
    marker_modes = set()
    varied_flat_color_puzzles = shared_flat_color_puzzles = 0
    flat_positions, component_counts, palette_mask_combinations = set(), set(), set()
    structure = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_markers = set()
        flat_colors = set()
        all_train_flat = True
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                marker = int(next(iter(set(np.unique(inp).tolist())
                                       - set(np.unique(out).tolist()))))
                h, w = _periods(out)
                mask = inp == marker
                # Reconstruct from the input alone, without generation parameters.
                assert len(_infer(inp)) == 1
                assert np.array_equal(_infer(inp)[0][3], out)
                assert np.array_equal(inp[~mask], out[~mask])
                assert _phase_coverage(mask, h, w) >= 2
                assert not _visible_shorter_periods(inp, marker, h, w)
                shapes.add(out.shape)
                periods.add((h, w))
                palette_size = len(np.unique(out))
                palettes.add(palette_size)
                markers.add(marker)
                structure.add(_key(out[:h, :w]))
                source_size += out.shape == (27, 27) and (h, w) == (8, 8)
                rectangular += h != w
                rows = [r for r in range(h) if np.all(out[r, :w] == out[r, 0])]
                flat += bool(rows)
                no_flat += not rows
                flat_positions.update(rows)
                touches = bool(np.any(mask[0]) or np.any(mask[-1])
                               or np.any(mask[:, 0]) or np.any(mask[:, -1]))
                border += touches
                no_border += not touches
                area = int(mask.sum())
                components = _components(mask)
                component_counts.add(len(components))
                full8 = _has_square8(mask)
                palette_mask_combinations.add((palette_size >= 8,
                                               area >= .20 * mask.size))
                small += min(out.shape) <= 19
                large += max(out.shape) >= 28
                two_d = _two_dim_count(mask, h, w) > 0
                if split == "train":
                    train_markers.add(marker)
                    flat_colors.update(int(out[r, 0]) for r in rows)
                    all_train_flat &= bool(rows)
                    two_dim_train += two_d
                    broad_train += area >= .20 * mask.size
                    full8_train += full8
                    rich_train += palette_size == 9
                else:
                    two_dim_test += two_d
                    compact_test += area <= .19 * mask.size
                    rich_modest_test += palette_size >= 8 and area <= .19 * mask.size
                    familiar_test += palette_size <= 7
                    rich_test += palette_size == 9
                    no8_test += not full8
                    source_like_test += (out.shape == (27, 27)
                                         and (h, w) == (8, 8)
                                         and palette_size == 9
                                         and 60 <= area <= 110
                                         and max(components) <= 70
                                         and not full8)
        marker_modes.add(len(train_markers) == 1)
        varied_flat_color_puzzles += len(flat_colors) >= 2
        shared_flat_color_puzzles += all_train_flat and len(flat_colors) == 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(periods) >= 15
    assert len(palettes) == 7 and len(markers) >= 8
    assert len(structure) >= 120
    assert source_size >= 8 and rectangular >= 50
    assert flat >= 25 and no_flat >= 25 and border >= 30 and no_border >= 20
    assert len(flat_positions) >= 6 and len(component_counts) >= 4
    assert small >= 20 and large >= 20
    assert broad_train >= 8 and compact_test >= 8
    assert rich_modest_test >= 2 and familiar_test >= 10
    assert two_dim_train >= 42 and two_dim_test >= 8
    assert full8_train >= 3 and no8_test >= 15 and source_like_test >= 1
    assert rich_test > rich_train
    assert marker_modes == {True, False}
    assert varied_flat_color_puzzles >= 5 and shared_flat_color_puzzles >= 5
    assert len(palette_mask_combinations) == 4
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4),
                          (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(901, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
        assert _equal(puzzle, generate_puzzle(901, ntrain, ntest))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert _equal(generate_puzzle(1001, 7, 3), generate_puzzle(1001, 7, 3))
    assert not _equal(generate_puzzle(1001, 7, 3),
                      generate_puzzle(1002, 7, 3))
