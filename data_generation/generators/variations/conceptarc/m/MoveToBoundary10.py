import numpy as np


_SOURCE_INPUTS = {
    tuple(s.split("/"))
    for s in (
        "00000000000/00000000000/00000000000/00222011100/00222011100/"
        "00222011100/00000000000/00000000000/00000000000",
        "0000000000000000000/0030303000404040000/0000000000000000000/"
        "0030003000400040000/0000000000000000000/0030303000404040000/"
        "0000000000000000000",
        "000000000000000000/000003000070000000/000030300707000000/"
        "000300037000700000/000030300707000000/000003000070000000/"
        "000000000000000000",
        "000000000000000000/000000000000000000/005550000555555000/"
        "005450000544445000/005550000544445000/000000000544445000/"
        "000000000555555000/000000000000000000/000000000000000000",
        "9999999003334300000000000/9000909003344300000000000/"
        "9999999003434300000000000/9000909003443300000000000/"
        "9000909003333300000000000/9999999000000000000000000/"
        "0000000000000000000000000/0000000000000000000000000",
    )
}


def _signature(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _counts(rng, num_train, num_test):
    for count, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if count is not None and (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or not low <= count <= high
        ):
            raise ValueError("invalid pair count")
    if num_train is None and num_test is None:
        num_train = int(rng.choice(np.arange(2, 11), p=[.16, .10, .20, .16, .10, .08, .07, .07, .06]))
    if num_test is None:
        upper = min(4, num_train - 1)
        weights = {1: [1], 2: [.4, .6], 3: [.2, .25, .55], 4: [.15, .2, .35, .3]}[upper]
        num_test = int(rng.choice(np.arange(1, upper + 1), p=weights))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _mask(h, w, style):
    r, c = np.indices((h, w))
    if style == "solid":
        mask = np.ones((h, w), dtype=bool)
    elif style == "frame":
        mask = (r == 0) | (r == h - 1) | (c == 0) | (c == w - 1)
    elif style == "checker":
        mask = (r + c) % 2 == 0
    elif style == "lattice":
        mask = ((r % 2 == 0) | (r == h - 1)) & ((c % 2 == 0) | (c == w - 1))
    elif style == "stripes":
        mask = (c % 2 == 0) | (c == w - 1)
    elif style == "bars":
        mask = (r == 0) | (r == h - 1) | (r == h // 2)
    else:
        diagonal = np.rint(c * (h - 1) / (w - 1)).astype(int)
        other = np.rint((w - 1 - c) * (h - 1) / (w - 1)).astype(int)
        mask = (r == diagonal) | (r == other)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = True
    return mask


def _paint(mask, colors, background, rng):
    tile = np.full(mask.shape, background, dtype=np.int8)
    tile[mask] = colors[0]
    if len(colors) == 1:
        return tile
    h, w = mask.shape
    r, c = np.indices((h, w))
    mode = int(rng.integers(0, 4))
    if mode == 0:
        accent = (r > 0) & (r < h - 1) & (c > 0) & (c < w - 1)
    elif mode == 1:
        accent = r >= h // 2
    elif mode == 2:
        accent = c >= w // 2
    else:
        accent = (r + c) % 3 == 0
    if not np.any(mask & accent) or np.all(~mask | accent):
        accent = r >= h // 2
    tile[mask & accent] = colors[1]
    return tile


def _choose_colors(rng, palette, profile, abut):
    if profile == "anchor":
        return (palette[0],), (palette[1],)
    if profile == "contrast":
        return (palette[2] if len(palette) > 2 else palette[1],), (
            palette[3] if len(palette) > 3 else palette[0],
        )
    if abut:
        selected = rng.choice(palette, size=2, replace=False)
        return (int(selected[0]),), (int(selected[1]),)
    if len(palette) >= 3:
        mode = int(rng.choice(9, p=[.25, .11, .09, .09, .08, .08, .10, .10, .10]))
    else:
        mode = int(rng.choice(6, p=[.30, .14, .14, .14, .14, .14]))
    a, b = (int(v) for v in rng.choice(palette, size=2, replace=False))
    if mode >= 6:
        c = int(rng.choice([v for v in palette if v not in (a, b)]))
        if mode == 6:
            return (a,), (b, c)
        if mode == 7:
            return (a, c), (b,)
        return (a, b), (b, c)
    if mode == 0:
        return (a,), (b,)
    if mode == 1:
        return (a,), (a,)
    if mode == 2:
        return (a, b), (a,)
    if mode == 3:
        return (a,), (a, b)
    if mode == 4:
        return (a, b), (b, a)
    return (a, b), (a, b)


def _make_pair(rng, background, palette, profile):
    familiar = profile == "free" and bool(rng.random() < .18)
    abut = profile == "free" and not familiar and bool(rng.random() < .14)
    matched = profile == "anchor" or familiar or (profile == "free" and rng.random() < .28)
    if matched:
        h_l = h_r = int(rng.integers(3, 9))
        w_l = w_r = int(rng.integers(3, 8))
    else:
        h_l, h_r = (int(v) for v in rng.integers(3, 9, size=2))
        w_l, w_r = (int(v) for v in rng.integers(3, 8, size=2))
    styles = ("solid", "frame", "checker", "lattice", "stripes", "bars", "diagonal")
    if abut:
        styles = ("solid", "frame", "bars")
    style_l = str(rng.choice(styles))
    style_r = style_l if matched else str(rng.choice(styles))
    if profile == "contrast" and (h_l, w_l, style_l) == (h_r, w_r, style_r):
        style_r = "frame" if style_l != "frame" else "lattice"
    mask_l = _mask(h_l, w_l, style_l)
    mask_r = mask_l.copy() if matched else _mask(h_r, w_r, style_r)
    if familiar:
        chosen = rng.choice(palette, size=2, replace=False)
        colors_l, colors_r = (int(chosen[0]),), (int(chosen[1]),)
    else:
        colors_l, colors_r = _choose_colors(rng, palette, profile, abut)
    tile_l = _paint(mask_l, colors_l, background, rng)
    tile_r = _paint(mask_r, colors_r, background, rng)

    padding = int(rng.integers(5, 15))
    width = w_l + w_r + padding
    placement = 0 if profile in ("anchor", "contrast") or familiar else int(rng.choice(4, p=[.48, .21, .21, .10]))
    if abut and placement == 3:
        placement = int(rng.integers(0, 3))
    if abut:
        gap = 0
    elif placement == 3:
        gap = padding
    else:
        max_gap = padding - (2 if placement == 0 else 1)
        gap = int(rng.integers(3, max_gap + 1))
    available = padding - gap
    if placement == 0:
        left_margin = int(rng.integers(1, available))
    elif placement == 1:
        left_margin = 0
    else:
        left_margin = available
    right_start = left_margin + w_l + gap

    foreground = int(np.count_nonzero(mask_l) + np.count_nonzero(mask_r))
    height = max(h_l, h_r) + int(rng.integers(2, 11))
    while height * width <= 2 * foreground:
        height += 1
    assert height <= 30 and width <= 30
    if abut:
        placements = [
            (tl, tr)
            for tl in range(height - h_l + 1)
            for tr in range(height - h_r + 1)
            if any(
                mask_l[row - tl, -1] and mask_r[row - tr, 0]
                for row in range(max(tl, tr), min(tl + h_l, tr + h_r))
            )
        ]
        aligned = bool(rng.integers(0, 2))
        subset = [(tl, tr) for tl, tr in placements if (tl == tr) == aligned]
        if subset:
            placements = subset
        top_l, top_r = placements[int(rng.integers(len(placements)))]
    elif matched:
        top_l = top_r = int(rng.integers(0, height - h_l + 1))
    else:
        top_l = int(rng.integers(0, height - h_l + 1))
        top_r = int(rng.integers(0, height - h_r + 1))
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full_like(inp, background)
    inp[top_l:top_l + h_l, left_margin:left_margin + w_l] = tile_l
    inp[top_r:top_r + h_r, right_start:right_start + w_r] = tile_r
    out[top_l:top_l + h_l, :w_l] = tile_l
    out[top_r:top_r + h_r, width - w_r:] = tile_r
    return inp, out


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(rng, num_train, num_test)
    background = 0 if rng.random() < .72 else int(rng.integers(1, 10))
    colors = np.array([c for c in range(10) if c != background])
    palette = tuple(int(v) for v in rng.choice(colors, size=int(rng.integers(2, 5)), replace=False))
    train_profiles = ["free"] * n_train
    first, second = (int(v) for v in rng.choice(n_train, size=2, replace=False))
    train_profiles[first] = "anchor"
    train_profiles[second] = "contrast"
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, profiles in (("train", train_profiles), ("test", ["free"] * n_test)):
        for profile in profiles:
            for _ in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                inp, out = _make_pair(pair_rng, background, palette, profile)
                sig = _signature(inp)
                if sig not in seen and sig not in _SOURCE_INPUTS:
                    seen.add(sig)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp, out, background):
    assert inp.shape == out.shape
    width = inp.shape[1]
    occupied = np.flatnonzero(np.any(inp != background, axis=0))
    assert len(occupied) >= 4
    gaps = np.diff(occupied) - 1
    large = np.flatnonzero(gaps >= 3)
    if len(large) == 1:
        cut = int(large[0])
        column_ranges = ((int(occupied[0]), int(occupied[cut]) + 1),
                         (int(occupied[cut + 1]), int(occupied[-1]) + 1))
        assert all(g <= 1 for i, g in enumerate(gaps) if i != cut)
        separation = column_ranges[1][0] - column_ranges[0][1]
    else:
        assert len(large) == 0
        colors = [int(v) for v in np.unique(inp) if v != background]
        assert len(colors) == 2
        boxes = []
        for color in colors:
            cols = np.flatnonzero(np.any(inp == color, axis=0))
            boxes.append((int(cols[0]), int(cols[-1]) + 1, color))
        boxes.sort()
        assert boxes[0][1] == boxes[1][0]
        boundary = boxes[0][1]
        assert np.any((inp[:, boundary - 1] == boxes[0][2]) &
                      (inp[:, boundary] == boxes[1][2]))
        column_ranges = ((boxes[0][0], boxes[0][1]), (boxes[1][0], boxes[1][1]))
        separation = 0
    objects = []
    expected = np.full_like(inp, background)
    for side, (left, right) in enumerate(column_ranges):
        rows = np.flatnonzero(np.any(inp[:, left:right] != background, axis=1))
        assert len(rows) >= 2
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        assert bottom - top >= 3
        tile = inp[top:bottom, left:right]
        assert np.any(tile[0] != background) and np.any(tile[-1] != background)
        assert np.any(tile[:, 0] != background) and np.any(tile[:, -1] != background)
        destination = 0 if side == 0 else width - (right - left)
        expected[top:bottom, destination:destination + right - left] = tile
        objects.append((top, bottom, left, right, tile))
    assert column_ranges[0][1] <= column_ranges[1][0]
    assert sum(right - left for left, right in column_ranges) <= width
    assert np.array_equal(out, expected)
    assert sorted(int(v) for v in np.unique(out)) == sorted(int(v) for v in np.unique(inp))
    return objects, separation


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    train_colors, test_colors, backgrounds = set(), set(), set()
    moving_train = 0
    independent_train = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            in_sig, out_sig = _signature(inp), _signature(out)
            assert in_sig not in seen_inputs and (in_sig, out_sig) not in seen_pairs
            assert in_sig not in _SOURCE_INPUTS
            seen_inputs.add(in_sig)
            seen_pairs.add((in_sig, out_sig))
            values, counts = np.unique(inp, return_counts=True)
            background = int(values[np.argmax(counts)])
            assert int(np.max(counts)) > inp.size // 2
            backgrounds.add(background)
            objects, separation = _inspect_pair(inp, out, background)
            assert separation == 0 or separation >= 3
            (lt, lb, la, lr, left_tile), (rt, rb, ra, rr, right_tile) = objects
            colors = {int(v) for v in np.unique(inp) if v != background}
            if split == "train":
                train_colors.update(colors)
                moving_train += la > 0 and rr < inp.shape[1]
                independent_train += (
                    left_tile.shape != right_tile.shape
                    or lt != rt
                    or not np.array_equal(left_tile != background, right_tile != background)
                )
            else:
                test_colors.update(colors)
    assert len(backgrounds) == 1
    assert moving_train >= 2 and independent_train >= 1
    assert test_colors <= train_colors


def _equal_puzzles(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(pa[k], pb[k])
            for pa, pb in zip(a[split], b[split]) for k in ("input", "output")
        ) for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, dimensions, palettes, shapes = set(), set(), set(), set(), set()
    backgrounds, gaps, placements, pair_signatures = set(), set(), set(), set()
    measures = {name: 0 for name in (
        "two_train", "many_test", "mixed_placement", "mixed_theme", "nonzero_background",
        "train_unequal", "train_offset", "train_shared", "train_stationary",
        "train_multicolor", "train_abut", "train_blank_row", "train_blank_column",
        "train_abut_offset",
        "train_equal_height_unequal_width", "train_unequal_height_equal_width",
        "test_equal", "test_unequal", "test_offset", "test_familiar",
        "test_equal_height_unequal_width", "test_unequal_height_equal_width",
        "test_left_multicolor", "test_right_multicolor", "test_shared",
        "test_identical_palette", "test_three_colors", "test_abut", "test_blank_row",
        "test_abut_offset",
        "test_blank_column", "test_top", "test_bottom", "test_unchanged",
        "test_three_colors_offset", "test_shared_unequal",
        "test_first_left_stationary", "test_first_left_moving",
    )}
    test_placements = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        measures["two_train"] += len(puzzle["train"]) == 2
        measures["many_test"] += len(puzzle["test"]) >= 3
        local = set()
        local_placements, local_themes = set(), set()
        local_backgrounds, train_colors, test_colors = set(), set(), set()
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                sig = _signature(inp)
                assert sig not in pair_signatures
                pair_signatures.add(sig)
                dimensions.add(inp.shape)
                values, counts = np.unique(inp, return_counts=True)
                bg = int(values[np.argmax(counts)])
                backgrounds.add(bg)
                local_backgrounds.add(bg)
                if split == "train" and pair is puzzle["train"][0]:
                    measures["nonzero_background"] += bg != 0
                objects, gap = _inspect_pair(inp, out, bg)
                (lt, lb, la, lr, left), (rt, rb, ra, rr, right) = objects
                shape = (left.shape, right.shape, tuple((left != bg).flat), tuple((right != bg).flat))
                shapes.add(shape)
                local.add(shape)
                placement = (la == 0, rr == inp.shape[1])
                placements.add((placement, lt == 0 or lb == inp.shape[0], rt == 0 or rb == inp.shape[0]))
                local_placements.add(placement)
                gaps.add(gap)
                lc = {int(v) for v in np.unique(left) if v != bg}
                rc = {int(v) for v in np.unique(right) if v != bg}
                if split == "train":
                    train_colors.update(lc | rc)
                else:
                    test_colors.update(lc | rc)
                predicted = np.full_like(inp, bg)
                for side, (top, bottom, start, stop, tile) in enumerate(objects):
                    target = 0 if side == 0 else inp.shape[1] - (stop - start)
                    for dy, dx in np.argwhere(tile != bg):
                        predicted[top + dy, target + dx] = tile[dy, dx]
                assert np.array_equal(predicted, out)
                local_themes.add((len(lc), len(rc), bool(lc & rc)))
                blank_row = any(not np.any(tile[r] != bg) for tile in (left, right) for r in range(1, tile.shape[0] - 1))
                blank_column = any(not np.any(tile[:, c] != bg) for tile in (left, right) for c in range(1, tile.shape[1] - 1))
                if split == "train":
                    measures["train_unequal"] += left.shape != right.shape
                    measures["train_offset"] += lt != rt
                    measures["train_shared"] += bool(lc & rc)
                    measures["train_stationary"] += any(placement)
                    measures["train_multicolor"] += len(lc) > 1 or len(rc) > 1
                    measures["train_abut"] += gap == 0
                    measures["train_abut_offset"] += gap == 0 and lt != rt
                    measures["train_blank_row"] += blank_row
                    measures["train_blank_column"] += blank_column
                    measures["train_equal_height_unequal_width"] += left.shape[0] == right.shape[0] and left.shape[1] != right.shape[1]
                    measures["train_unequal_height_equal_width"] += left.shape[0] != right.shape[0] and left.shape[1] == right.shape[1]
                else:
                    test_placements.add(placement)
                    if index == 0:
                        measures["test_first_left_stationary"] += la == 0
                        measures["test_first_left_moving"] += la > 0
                    measures["test_equal"] += left.shape == right.shape
                    measures["test_unequal"] += left.shape != right.shape
                    measures["test_equal_height_unequal_width"] += left.shape[0] == right.shape[0] and left.shape[1] != right.shape[1]
                    measures["test_unequal_height_equal_width"] += left.shape[0] != right.shape[0] and left.shape[1] == right.shape[1]
                    measures["test_offset"] += lt != rt
                    measures["test_left_multicolor"] += len(lc) > 1
                    measures["test_right_multicolor"] += len(rc) > 1
                    measures["test_shared"] += bool(lc & rc)
                    measures["test_identical_palette"] += lc == rc
                    measures["test_three_colors"] += len(lc | rc) >= 3
                    measures["test_abut"] += gap == 0
                    measures["test_abut_offset"] += gap == 0 and lt != rt
                    measures["test_blank_row"] += blank_row
                    measures["test_blank_column"] += blank_column
                    measures["test_top"] += lt == 0 or rt == 0
                    measures["test_bottom"] += lb == inp.shape[0] or rb == inp.shape[0]
                    measures["test_unchanged"] += np.array_equal(inp, out)
                    measures["test_three_colors_offset"] += len(lc | rc) >= 3 and lt != rt
                    measures["test_shared_unequal"] += bool(lc & rc) and left.shape != right.shape
                    measures["test_familiar"] += (
                        len(lc) == len(rc) == 1 and lc != rc and gap >= 3
                        and placement == (False, False) and left.shape == right.shape
                    )
                for row in range(inp.shape[0]):
                    assert sorted(int(v) for v in inp[row]) == sorted(int(v) for v in out[row])
        assert len(local) >= 2
        assert len(local_backgrounds) == 1 and test_colors <= train_colors
        measures["mixed_placement"] += len(local_placements) >= 2
        measures["mixed_theme"] += len(local_themes) >= 2
        palettes.add(tuple(sorted({int(v) for pair in puzzle["train"] for v in np.unique(pair["input"]) if v != bg})))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(pair_signatures) > 200 and len(dimensions) >= 35
    assert len(shapes) >= 100 and len(palettes) >= 15 and len(backgrounds) >= 3
    assert len(gaps) >= 8 and len(placements) >= 6
    assert len(test_placements) == 4
    assert measures["two_train"] >= 2 and measures["many_test"] >= 2
    assert measures["mixed_placement"] >= 15 and measures["mixed_theme"] >= 15
    assert measures["nonzero_background"] >= 2
    assert all(value >= 3 for key, value in measures.items() if key.startswith("train_"))
    assert all(value >= 2 for key, value in measures.items() if key.startswith("test_"))
    for n_train, n_test in ((2, 1), (10, 1), (10, 4), (5, 2)):
        seed = 1000 + 10 * n_train + n_test
        puzzle = generate_puzzle(seed, n_train, n_test)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (n_train, n_test)
        assert _equal_puzzles(puzzle, generate_puzzle(seed, n_train, n_test))
    for n_train, n_test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(207, n_train, n_test)
        validate_generated_puzzle(puzzle)
        assert n_train is None or len(puzzle["train"]) == n_train
        assert n_test is None or len(puzzle["test"]) == n_test
    for n_train, n_test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, n_train, n_test)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal pair counts")
