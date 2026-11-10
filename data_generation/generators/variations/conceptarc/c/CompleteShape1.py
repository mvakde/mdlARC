"""Two-axis reflection completion puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ("0000000000", "0000000000", "0112002110", "0112002110",
     "0220000220", "0110000110", "0110000110", "0220000220",
     "0112112110", "0112112110", "0000000000", "0000000000"),
    ("02666620", "06000060", "06000060", "06000060", "06000060",
     "02660020"),
    ("0000000000", "0000000000", "0000020000", "0000404000",
     "0004000400", "0040000040", "0200000000", "0040000040",
     "0004000400", "0000404000", "0000020000", "0000000000"),
    ("000000000", "000000000", "003333300", "003300300",
     "003333300", "003333300", "003333300", "000000000",
     "000000000"),
    ("0000000000", "0000000000", "0002220222", "0002220222",
     "0000000000", "0002220222", "0002220222", "0000000000",
     "0002220000", "0002220000"),
}
_FAMILIES = ("frame", "filled", "diamond", "blocks", "motif")


def _key(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def _bbox(grid, bg):
    points = np.argwhere(grid != bg)
    assert len(points)
    return (int(points[:, 0].min()), int(points[:, 0].max()),
            int(points[:, 1].min()), int(points[:, 1].max()))


def _orbit(r, c, h, w):
    return {(r, c), (h - 1 - r, c), (r, w - 1 - c),
            (h - 1 - r, w - 1 - c)}


def _orbits(figure, bg):
    h, w = figure.shape
    seen = set()
    result = []
    for r in range(h):
        for c in range(w):
            if figure[r, c] == bg or (r, c) in seen:
                continue
            orbit = _orbit(r, c, h, w)
            seen.update(orbit)
            result.append(orbit)
    return result


def _reflect_fill(inp, bg, axes="both"):
    y0, y1, x0, x1 = _bbox(inp, bg)
    result = inp.copy()
    for r in range(y0, y1 + 1):
        for c in range(x0, x1 + 1):
            ys = (r, y0 + y1 - r) if axes in ("both", "horizontal") else (r,)
            xs = (c, x0 + x1 - c) if axes in ("both", "vertical") else (c,)
            colors = {int(inp[yy, xx]) for yy in ys for xx in xs
                      if inp[yy, xx] != bg}
            assert len(colors) <= 1
            if colors:
                result[r, c] = next(iter(colors))
    return result


def _symmetric_color(rng, mask, bg, palette):
    h, w = mask.shape
    figure = np.full((h, w), bg, dtype=np.int64)
    orbits = []
    seen = set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or (r, c) in seen:
                continue
            orbit = _orbit(r, c, h, w)
            seen.update(orbit)
            orbits.append(orbit)
    rng.shuffle(orbits)
    for i, orbit in enumerate(orbits):
        if i < len(palette):
            color = palette[i]
        else:
            color = palette[int(rng.choice(len(palette), p=(
                [0.78] + [0.22 / (len(palette) - 1)] * (len(palette) - 1)
                if len(palette) > 1 else [1.0]
            )))]
        for r, c in orbit:
            figure[r, c] = color
    return figure


def _color_figure(rng, mask, family, bg, palette):
    if family == "blocks":
        row_on = np.any(mask, axis=1)
        col_on = np.any(mask, axis=0)
        row_starts = [i for i, on in enumerate(row_on)
                      if on and (i == 0 or not row_on[i - 1])]
        col_starts = [i for i, on in enumerate(col_on)
                      if on and (i == 0 or not col_on[i - 1])]
        bh = int(np.count_nonzero(row_on) // len(row_starts))
        bw = int(np.count_nonzero(col_on) // len(col_starts))
        tile = _symmetric_color(rng, np.ones((bh, bw), dtype=bool),
                                bg, palette)
        figure = np.full(mask.shape, bg, dtype=np.int64)
        for y in row_starts:
            for x in col_starts:
                figure[y:y + bh, x:x + bw] = tile
        return figure
    if family == "diamond":
        figure = np.full(mask.shape, bg, dtype=np.int64)
        figure[mask] = palette[0]
        middle = mask.shape[0] // 2
        if len(palette) > 1:
            for r, c in ((0, middle), (middle, 0),
                         (middle, mask.shape[1] - 1),
                         (mask.shape[0] - 1, middle)):
                figure[r, c] = palette[1]
        if len(palette) > 2:
            for r, c in _orbit(2, middle - 2, *mask.shape):
                figure[r, c] = palette[2]
        return figure
    return _symmetric_color(rng, mask, bg, palette)


def _make_mask(rng, family):
    if family == "frame":
        h = int(rng.integers(5, 14))
        w = int(rng.integers(5, 15))
        mask = np.zeros((h, w), dtype=bool)
        thickness = int(rng.integers(1, min(3, (h - 1) // 2, (w - 1) // 2) + 1))
        mask[:thickness, :] = mask[-thickness:, :] = True
        mask[:, :thickness] = mask[:, -thickness:] = True
        if h >= 8 and w >= 8 and rng.random() < 0.5:
            mask[2, 2:w - 2] = mask[-3, 2:w - 2] = True
        return mask

    if family == "filled":
        h = int(rng.integers(5, 13))
        w = int(rng.integers(5, 14))
        mask = np.ones((h, w), dtype=bool)
        if h >= 8 and w >= 8 and rng.random() < 0.25:
            mask[h // 2 - 1:h - (h // 2 - 1),
                 w // 2 - 1:w - (w // 2 - 1)] = False
        return mask

    if family == "diamond":
        size = int(rng.choice((7, 9, 11, 13)))
        middle = size // 2
        mask = np.zeros((size, size), dtype=bool)
        for r in range(size):
            step = min(r, size - 1 - r)
            mask[r, middle - step] = True
            mask[r, middle + step] = True
        if rng.random() < 0.22:
            # A deliberate thicker contour; the ordinary contour steps by one.
            for r in range(2, size - 2):
                step = min(r, size - 1 - r) - 1
                mask[r, middle - step] = True
                mask[r, middle + step] = True
        return mask

    if family == "blocks":
        choices = [(nr, nc, bh, bw, gy, gx)
                   for nr in (2, 3, 4) for nc in (2, 3, 4)
                   for bh in (2, 3) for bw in (2, 3, 4)
                   for gy in (1, 2, 3) for gx in (1, 2, 3)
                   if nr * bh + (nr - 1) * gy <= 23
                   and nc * bw + (nc - 1) * gx <= 23]
        nr, nc, bh, bw, gy, gx = choices[int(rng.integers(len(choices)))]
        h = nr * bh + (nr - 1) * gy
        w = nc * bw + (nc - 1) * gx
        mask = np.zeros((h, w), dtype=bool)
        for i in range(nr):
            for j in range(nc):
                y, x = i * (bh + gy), j * (bw + gx)
                mask[y:y + bh, x:x + bw] = True
        return mask

    if family == "motif":
        h = int(rng.integers(7, 16))
        w = int(rng.integers(7, 17))
        mask = np.zeros((h, w), dtype=bool)
        qh, qw = (h + 1) // 2, (w + 1) // 2
        # Visible extrema and an interior four-member orbit are guaranteed.
        anchors = [(0, qw - 1), (qh - 1, 0), (1, 1)]
        for r, c in anchors:
            for rr, cc in _orbit(r, c, h, w):
                mask[rr, cc] = True
        for _ in range(int(rng.integers(2, 6))):
            y = int(rng.integers(qh))
            x = int(rng.integers(qw))
            eh = int(rng.integers(1, min(4, qh - y) + 1))
            ew = int(rng.integers(1, min(4, qw - x) + 1))
            if rng.random() < 0.35:
                eh = 1
            elif rng.random() < 0.35:
                ew = 1
            for r in range(y, y + eh):
                for c in range(x, x + ew):
                    for rr, cc in _orbit(r, c, h, w):
                        mask[rr, cc] = True
        return mask

    raise ValueError("unknown figure family")


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        frontier = [start]
        while frontier:
            r, c = frontier.pop()
            for neighbor in ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    return components


def _observed_kind(local, bg):
    """Classify completed geometry for output-based diversity checks."""
    mask = local != bg
    h, w = mask.shape
    if np.all(mask):
        return "filled", None
    if h == w and h % 2 and h >= 7:
        middle = h // 2
        ordinary = np.zeros_like(mask)
        for r in range(h):
            step = min(r, h - 1 - r)
            ordinary[r, middle - step] = True
            ordinary[r, middle + step] = True
        if np.array_equal(mask, ordinary):
            return "diamond", None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if np.array_equal(mask, np.outer(rows, cols)):
        def runs(flags):
            starts = [i for i, on in enumerate(flags)
                      if on and (i == 0 or not flags[i - 1])]
            lengths = []
            for start in starts:
                length = 0
                while start + length < len(flags) and flags[start + length]:
                    length += 1
                lengths.append(length)
            gaps = [starts[i + 1] - starts[i] - lengths[i]
                    for i in range(len(starts) - 1)]
            return starts, lengths, gaps
        ry, bh, gy = runs(rows)
        rx, bw, gx = runs(cols)
        if (len(ry) >= 2 and len(rx) >= 2
                and len(set(bh)) == len(set(bw)) == 1
                and len(set(gy)) == len(set(gx)) == 1):
            return "blocks", (len(ry), len(rx), bh[0], bw[0], gy[0], gx[0])
    if (np.all(mask[0]) and np.all(mask[-1])
            and np.all(mask[:, 0]) and np.all(mask[:, -1])):
        return "frame", None
    return "motif", None


def _valid_erasure(figure, removed, bg):
    if not removed or len(removed) >= np.count_nonzero(figure != bg) * 0.48:
        return False
    inp = figure.copy()
    for r, c in removed:
        if figure[r, c] == bg:
            return False
        inp[r, c] = bg
    if _bbox(inp, bg) != _bbox(figure, bg):
        return False
    for orbit in _orbits(figure, bg):
        if orbit <= removed:
            return False
    return np.array_equal(_reflect_fill(inp, bg), figure)


def _erasure(rng, figure, bg, mode):
    h, w = figure.shape
    candidates = [orbit for orbit in _orbits(figure, bg) if len(orbit) == 4]
    rng.shuffle(candidates)

    def axis_options(axis):
        options = []
        for orbit in candidates:
            top = min(r for r, _ in orbit)
            left = min(c for _, c in orbit)
            if axis == "horizontal":
                removal = {(top, left), (top, w - 1 - left)}
            else:
                removal = {(top, left), (h - 1 - top, left)}
            if _valid_erasure(figure, removal, bg):
                options.append((orbit, removal))
        return options

    if mode in ("horizontal", "vertical"):
        options = axis_options(mode)
        if options:
            return options[int(rng.integers(len(options)))][1]
    elif mode == "mixed":
        h_opts, v_opts = axis_options("horizontal"), axis_options("vertical")
        for orbit_h, rm_h in h_opts:
            for orbit_v, rm_v in v_opts:
                if orbit_h != orbit_v and _valid_erasure(figure, rm_h | rm_v, bg):
                    return rm_h | rm_v
    elif mode == "component":
        comps = _components(figure != bg)
        rng.shuffle(comps)
        for component in comps:
            if len(component) >= 2 and _valid_erasure(figure, component, bg):
                return component
    elif mode == "edge":
        edges = []
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    for dr, dc in ((1, 0), (0, 1)):
                        run = {(r + k * dr, c + k * dc)
                               for k in range(int(rng.integers(2, 5)))}
                        if all(0 <= y < h and 0 <= x < w for y, x in run):
                            if _valid_erasure(figure, run, bg):
                                edges.append(run)
        if edges:
            return edges[int(rng.integers(len(edges)))]
    elif mode == "single":
        all_cells = list(map(tuple, np.argwhere(figure != bg)))
        rng.shuffle(all_cells)
        for r, c in all_cells:
            orbit = list(_orbit(r, c, h, w))
            rng.shuffle(orbit)
            for count in (3, 2, 1):
                removal = set(orbit[:count])
                if _valid_erasure(figure, removal, bg):
                    return removal
    return None


def _place(rng, figure, removed, bg):
    h, w = figure.shape
    canvas_h = int(rng.integers(h + 2, min(30, h + 7) + 1))
    canvas_w = int(rng.integers(w + 2, min(30, w + 7) + 1))
    y = int(rng.integers(1, canvas_h - h))
    x = int(rng.integers(1, canvas_w - w))
    out = np.full((canvas_h, canvas_w), bg, dtype=np.int64)
    out[y:y + h, x:x + w] = figure
    inp = out.copy()
    for r, c in removed:
        inp[y + r, x + c] = bg
    return {"input": inp, "output": out}


def _make_pair(rng, family, mode, bg):
    for attempt in range(100):
        count = int(rng.choice((1, 1, 2, 2, 3)))
        palette = [int(v) for v in rng.choice(
            [c for c in range(10) if c != bg], size=count, replace=False)]
        mask = _make_mask(rng, family)
        figure = _color_figure(rng, mask, family, bg, palette)
        effective_mode = mode
        if mode == "component" and attempt >= 35:
            effective_mode = "edge" if attempt < 65 else "single"
        elif mode == "edge" and attempt >= 65:
            effective_mode = "single"
        removal = _erasure(rng, figure, bg, effective_mode)
        if removal is not None:
            return _place(rng, figure, removal, bg)
    raise RuntimeError("could not construct a recoverable pair")


def _validate_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _validate_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    # The background is shared within a puzzle; every pair identifies it at
    # its canvas border. Foreground colors are visible local roles, not a map.
    bg = 0 if rng.random() < 0.78 else int(rng.integers(1, 10))
    simple_same_family = rng.random() < 0.14
    if simple_same_family:
        shared_family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
        train_families = [shared_family, shared_family]
    else:
        train_families = list(rng.choice(_FAMILIES, size=2, replace=False))
    for _ in range(num_train - 2):
        train_families.append(
            shared_family if simple_same_family
            else _FAMILIES[int(rng.integers(len(_FAMILIES)))])
    train_modes = ["horizontal", "vertical"]
    extra_modes = ("horizontal", "vertical", "single", "edge", "component")
    train_modes += [extra_modes[int(rng.integers(len(extra_modes)))]
                    for _ in range(num_train - 2)]
    test_families = []
    test_modes = []
    for _ in range(num_test):
        if simple_same_family:
            family = shared_family
        elif rng.random() < 0.55:
            family = train_families[int(rng.integers(len(train_families)))]
        else:
            family = _FAMILIES[int(rng.integers(len(_FAMILIES)))]
        test_families.append(family)
        if simple_same_family:
            test_modes.append(("single", "edge", "horizontal", "vertical")
                              [int(rng.integers(4))])
        else:
            test_modes.append(("mixed", "single", "edge", "component",
                               "horizontal", "vertical")[
                                   int(rng.choice(6, p=(0.37, 0.16, 0.15,
                                                        0.13, 0.095, 0.095)))])

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, families, modes in (("train", train_families, train_modes),
                                   ("test", test_families, test_modes)):
        for family, mode in zip(families, modes):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, family, mode, bg)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair, bg):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    assert np.all((inp == bg) | (inp == out))
    assert np.any(inp != out)
    assert _bbox(inp, bg) == _bbox(out, bg)
    y0, y1, x0, x1 = _bbox(out, bg)
    local = out[y0:y1 + 1, x0:x1 + 1]
    assert np.array_equal(local, local[::-1, :])
    assert np.array_equal(local, local[:, ::-1])
    assert np.array_equal(_reflect_fill(inp, bg), out)
    for orbit in _orbits(local, bg):
        assert any(inp[y0 + r, x0 + c] != bg for r, c in orbit)
    return local


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    backgrounds = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.min(grid) >= 0 and np.max(grid) <= 9
        inp, out = pair["input"], pair["output"]
        bg = int(inp[0, 0])
        backgrounds.add(bg)
        assert np.all(inp[0, :] == bg) and np.all(inp[-1, :] == bg)
        assert np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg)
        assert np.all(out[0, :] == bg) and np.all(out[-1, :] == bg)
        assert np.all(out[:, 0] == bg) and np.all(out[:, -1] == bg)
        key = _key(inp)
        assert key not in seen
        seen.add(key)
        assert tuple("".join(str(v) for v in row) for row in key) not in _SOURCE_INPUTS
        _check_pair(pair, bg)
    assert len(backgrounds) == 1
    bg = next(iter(backgrounds))
    first, second = puzzle["train"][:2]
    assert np.array_equal(_reflect_fill(first["input"], bg, "horizontal"),
                          first["output"])
    assert not np.array_equal(_reflect_fill(first["input"], bg, "vertical"),
                              first["output"])
    assert np.array_equal(_reflect_fill(second["input"], bg, "vertical"),
                          second["output"])
    assert not np.array_equal(_reflect_fill(second["input"], bg, "horizontal"),
                              second["output"])


def _same_puzzle(a, b):
    return all(np.array_equal(pa[name], pb[name])
               for split in ("train", "test")
               for pa, pb in zip(a[split], b[split])
               for name in ("input", "output")) and all(
                   len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    counts_train, counts_test = set(), set()
    shapes, masks, color_sets, backgrounds = set(), set(), set(), set()
    components, missing_sizes, mixed_tests = set(), set(), 0
    within_palette_change, rectangular = 0, 0
    observed = {"train": set(), "test": set()}
    block_params, edge_count, interior_count = set(), 0, 0
    complete_components, simple_puzzles, new_test_families = 0, 0, 0
    varied_train, one_step_diamonds, multicolor_mixed_bg = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        palettes = []
        train_kinds, test_kinds = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = int(inp[0, 0])
                backgrounds.add(bg)
                palette = tuple(int(c) for c in np.unique(out) if c != bg)
                palettes.append(palette)
                color_sets.add(palette)
                y0, y1, x0, x1 = _bbox(out, bg)
                local = out[y0:y1 + 1, x0:x1 + 1]
                kind, params = _observed_kind(local, bg)
                observed[split].add(kind)
                (train_kinds if split == "train" else test_kinds).append(kind)
                if kind == "blocks":
                    block_params.add(params)
                if kind == "diamond":
                    one_step_diamonds += 1
                shapes.add(out.shape)
                masks.add((local.shape, (local != bg).tobytes()))
                components.add(len(_components(local != bg)))
                missing_sizes.add(int(np.count_nonzero(inp != out)))
                rectangular += local.shape[0] != local.shape[1]
                removed = np.argwhere(inp != out)
                edge_count += any(y in (y0, y1) or x in (x0, x1)
                                  for y, x in removed)
                interior_count += any(y0 < y < y1 and x0 < x < x1
                                      for y, x in removed)
                for component in _components(local != bg):
                    if (len(component) >= 2 and all(
                            inp[y0 + r, x0 + c] == bg
                            for r, c in component)):
                        complete_components += 1
                        break
                if split == "test" and (not np.array_equal(
                        _reflect_fill(inp, bg, "horizontal"), out)
                        and not np.array_equal(
                            _reflect_fill(inp, bg, "vertical"), out)):
                    mixed_tests += 1
                    if bg and len(palette) > 1:
                        multicolor_mixed_bg += 1
        within_palette_change += len(set(palettes)) > 1
        varied_train += len(set(train_kinds)) > 1
        simple_puzzles += (len(set(train_kinds)) == 1
                           and set(test_kinds) <= set(train_kinds))
        new_test_families += bool(set(test_kinds) - set(train_kinds))
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(shapes) >= 30 and len(masks) >= 70
    assert len(color_sets) >= 20 and len(backgrounds) >= 3
    assert len(components) >= 6 and len(missing_sizes) >= 10
    assert within_palette_change >= 30 and rectangular >= 50
    assert mixed_tests >= 8
    assert observed["train"] == observed["test"] == set(_FAMILIES)
    assert len(block_params) >= 30 and one_step_diamonds >= 20
    assert edge_count >= 30 and interior_count >= 30
    assert complete_components >= 5
    assert simple_puzzles >= 2 and new_test_families >= 3
    assert varied_train >= 20 and multicolor_mixed_bg >= 1

    for seed, nt, ns in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                         (1003, None, 1), (1004, None, 4),
                         (1005, 2, None), (1006, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
        assert _same_puzzle(puzzle, generate_puzzle(seed, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(55, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    assert not _same_puzzle(generate_puzzle(75), generate_puzzle(76))
