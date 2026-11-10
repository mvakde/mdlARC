"""ARC reflection puzzles with visible rectangular occlusion."""

import numpy as np


_SOURCE_INPUTS = (
    "0000000000000000000000000000000000000000000000000002002000000000"
    "3333222200000000333322222000000033332202000000003333000200000000"
    "0022222200000000022022022000000002200002200000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000066006600000000006601111000000"
    "0000661111000000000006111100000000000660000000000000666600000000"
    "0000600600000000000066660000000000000660000000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000"
    "0000000000000000000000050005000000008888555050000000888800005500"
    "0000888855555500000088880500000000008888555555000000005505055000"
    "0000000500050000000000000000000000000000000000000000000000000000",
)


def _checked_count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
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


def _rect_components(grid, color):
    """All four-connected components must be solid rectangles."""
    cells = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    for start_r, start_c in np.argwhere(cells):
        r, c = int(start_r), int(start_c)
        if seen[r, c]:
            continue
        todo = [(r, c)]
        seen[r, c] = True
        points = []
        while todo:
            y, x = todo.pop()
            points.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]
                        and cells[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        arr = np.asarray(points)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        if len(points) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        rectangles.append((int(r0), int(r1), int(c0), int(c1)))
    return rectangles


def _find_solutions(inp):
    """Enumerate input-only reconstructions in both orientations."""
    colors = [int(x) for x in np.unique(inp)]
    if len(colors) < 3 or len(colors) > 5:
        return []
    solutions = []
    for orientation in ("vertical", "horizontal"):
        view = inp if orientation == "vertical" else inp.T
        height, width = view.shape
        for background in colors:
            for mask_color in colors:
                if mask_color == background:
                    continue
                motif_colors = tuple(x for x in colors if x not in (background, mask_color))
                if not 1 <= len(motif_colors) <= 2:
                    continue
                rectangles = _rect_components(view, mask_color)
                if rectangles is None or not 1 <= len(rectangles) <= 2:
                    continue
                mask = view == mask_color
                motif = (view != background) & ~mask
                if np.count_nonzero(motif) < 8:
                    continue
                mr, mc = np.where(motif)
                rr, cc = np.where(mask)
                for axis2 in range(1, 2 * width - 2):
                    reflected_masks = axis2 - cc
                    if np.any(reflected_masks < 0) or np.any(reflected_masks >= width):
                        continue
                    if np.any(mask[rr, reflected_masks]):
                        continue
                    sides = []
                    for r0, r1, c0, c1 in rectangles:
                        if 2 * c1 < axis2:
                            sides.append("left")
                        elif 2 * c0 > axis2:
                            sides.append("right")
                        else:
                            break
                    if len(sides) != len(rectangles):
                        continue
                    mirrored_motif = axis2 - mc
                    if np.any(mirrored_motif < 0) or np.any(mirrored_motif >= width):
                        continue
                    visible_mirrors = ~mask[mr, mirrored_motif]
                    if not np.array_equal(
                        view[mr[visible_mirrors], mc[visible_mirrors]],
                        view[mr[visible_mirrors], mirrored_motif[visible_mirrors]],
                    ):
                        continue
                    reconstructed = view.copy()
                    reconstructed[rr, cc] = view[rr, reflected_masks]
                    if np.any(reconstructed == mask_color):
                        continue
                    complete_motif = reconstructed != background
                    if np.count_nonzero(complete_motif) < 8:
                        continue
                    if any(not np.any(reconstructed == color) for color in motif_colors):
                        continue
                    yr, xc = np.where(complete_motif)
                    if int(xc.min() + xc.max()) != axis2:
                        continue
                    mx = axis2 - xc
                    if (np.any(mx < 0) or np.any(mx >= width)
                            or not np.array_equal(reconstructed[yr, xc], reconstructed[yr, mx])):
                        continue
                    result = reconstructed.copy() if orientation == "vertical" else reconstructed.T.copy()
                    rects = rectangles if orientation == "vertical" else [
                        (c0, c1, r0, r1) for r0, r1, c0, c1 in rectangles
                    ]
                    solutions.append({
                        "output": result,
                        "orientation": orientation,
                        "axis2": axis2,
                        "background": background,
                        "mask_color": mask_color,
                        "motif_colors": motif_colors,
                        "rects": rects,
                        "sides": tuple(sides),
                    })
    return solutions


def _motif(rng, height, width, family):
    half = width // 2
    left = np.zeros((height, half), dtype=bool)
    if family in ("sparse", "dense"):
        density = float(rng.uniform(0.22, 0.47) if family == "sparse"
                        else rng.uniform(0.58, 0.84))
        left = rng.random((height, half)) < density
        for row in range(height):
            if not np.any(left[row]) and rng.random() < 0.7:
                left[row, int(rng.integers(0, half))] = True
    elif family in ("body", "outline", "holes", "appendage"):
        radius = max(1, half - 2) if family == "appendage" else int(
            rng.integers(max(1, half // 2), half + 1))
        for row in range(height):
            taper = abs(2 * row - height + 1) / max(1, height - 1)
            reach = max(1, min(half, radius - int(taper * rng.uniform(0, 2.2))
                                + int(rng.integers(-1, 2))))
            left[row, half - reach:] = True
        if family == "outline":
            filled = np.concatenate((left, left[:, ::-1]), axis=1)
            if width % 2:
                filled = np.concatenate((left, np.ones((height, 1), bool),
                                         left[:, ::-1]), axis=1)
            inner = filled.copy()
            inner[1:-1, 1:-1] &= (
                filled[:-2, 1:-1] & filled[2:, 1:-1]
                & filled[1:-1, :-2] & filled[1:-1, 2:]
            )
            left = (filled & ~inner)[:, :half]
        elif family == "holes" and height >= 7 and half >= 3:
            for _ in range(int(rng.integers(1, 4))):
                row = int(rng.integers(2, height - 2))
                col = int(rng.integers(max(0, half - 3), half))
                left[row:row + int(rng.integers(1, 3)), col] = False
        elif family == "appendage":
            row = int(rng.integers(1, height - 1))
            left[row, 0] = True
            if half > 2:
                left[row, 1:half] = True
    elif family == "branches":
        left[:, -1] = True
        arms = int(rng.integers(2, max(3, height // 2 + 1)))
        for row in rng.choice(height, size=min(height, arms), replace=False):
            reach = int(rng.integers(1, half + 1))
            left[int(row), half - reach:] = True
            if row + 1 < height and rng.random() < 0.45:
                left[int(row) + 1, half - reach] = True
    else:
        raise ValueError("unknown motif family")
    left[0, -1] = True
    left[-1, -1] = True
    if family != "appendage":
        left[int(rng.integers(0, height)), 0] = True
    if width % 2:
        center = np.any(left[:, -1:], axis=1, keepdims=True)
        if family in ("sparse", "dense"):
            center = rng.random((height, 1)) < (0.4 if family == "sparse" else 0.75)
        return np.concatenate((left, center, left[:, ::-1]), axis=1)
    return np.concatenate((left, left[:, ::-1]), axis=1)


def _choose_geometry(rng, parity, orientation):
    h = int(rng.integers(9, 31))
    w = int(rng.integers(9, 31))
    if orientation == "horizontal":
        h, w = w, h
    widths = [x for x in range(5, min(w, 23) + 1) if x % 2 == parity]
    mw = int(widths[int(rng.integers(len(widths)))])
    mh = int(rng.integers(5, min(h, 25) + 1))
    return h, w, mh, mw


def _candidate_masks(rng, output, background, axis2, side, style, count):
    height, width = output.shape
    motif = output != background
    total = int(np.count_nonzero(motif))
    candidates = []
    for _ in range(650):
        mask_w = int(rng.integers(1, min(9, width) + 1))
        mask_h = int(rng.integers(1, min(11, height) + 1))
        if side == "left":
            choices = [x for x in range(width - mask_w + 1)
                       if 2 * (x + mask_w - 1) < axis2 and axis2 - x < width]
        else:
            choices = [x for x in range(width - mask_w + 1)
                       if 2 * x > axis2 and axis2 - (x + mask_w - 1) >= 0]
        if style == "edge":
            choices = [x for x in choices if x == 0 or x + mask_w == width]
        if not choices:
            continue
        c0 = int(choices[int(rng.integers(len(choices)))])
        r0 = int(rng.integers(0, height - mask_h + 1))
        r1, c1 = r0 + mask_h - 1, c0 + mask_w - 1
        hidden = int(np.count_nonzero(motif[r0:r1 + 1, c0:c1 + 1]))
        if hidden < 1:
            continue
        if style == "easy" and hidden > 3:
            continue
        if style == "substantial" and hidden < max(4, int(0.12 * total)):
            continue
        if style == "protrusion":
            outside = motif.copy()
            outside[r0:r1 + 1, c0:c1 + 1] = False
            xs = np.where(outside)[1]
            if len(xs) < 8 or int(xs.min() + xs.max()) == axis2:
                continue
        candidates.append((r0, r1, c0, c1))
        if len(candidates) >= 45:
            break
    if count == 1:
        return [[x] for x in candidates]
    combinations = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if not (a[1] + 1 < b[0] or b[1] + 1 < a[0]
                    or a[3] + 1 < b[2] or b[3] + 1 < a[2]):
                continue
            combinations.append([a, b])
            if len(combinations) >= 100:
                return combinations
    return combinations


def _make_pair(rng, parity, orientation, colors, style, family, rich):
    background, motif_color, mask_color = colors[:3]
    extra_color = colors[3] if rich == "multi_color" else None
    mask_count = 2 if rich == "multi_mask" else 1
    for _ in range(70):
        height, width, mh, mw = _choose_geometry(rng, parity, orientation)
        side = "left" if int(rng.integers(0, 2)) == 0 else "right"
        if style == "edge":
            edge_start = min(3, width - mw)
            offset = int(rng.integers(0, edge_start + 1))
            x0 = offset if side == "left" else width - mw - offset
        else:
            x0 = int(rng.integers(0, width - mw + 1))
        y0 = int(rng.integers(0, height - mh + 1))
        shape = _motif(rng, mh, mw, family)
        if np.count_nonzero(shape) < 10:
            continue
        output = np.full((height, width), background, dtype=np.int8)
        block = output[y0:y0 + mh, x0:x0 + mw]
        block[shape] = motif_color
        if extra_color is not None:
            rows = np.arange(mh)[:, None]
            band = ((rows // max(1, mh // 4)) % 2 == 1) & shape
            block[band] = extra_color
        axis2 = 2 * x0 + mw - 1
        options = _candidate_masks(rng, output, background, axis2, side, style, mask_count)
        for rects in options[:80]:
            inp = output.copy()
            for r0, r1, c0, c1 in rects:
                inp[r0:r1 + 1, c0:c1 + 1] = mask_color
            if extra_color is not None and not np.any(inp == extra_color):
                continue
            if not np.any(inp == motif_color):
                continue
            solutions = _find_solutions(inp)
            if (len(solutions) == 1 and solutions[0]["axis2"] == axis2
                    and solutions[0]["orientation"] == "vertical"
                    and np.array_equal(solutions[0]["output"], output)):
                if orientation == "horizontal":
                    return inp.T.copy(), output.T.copy()
                return inp, output
    raise ValueError("could not construct a uniquely recoverable pair")


def _source_copy(inp):
    if inp.shape != (16, 16):
        return False
    return "".join(str(int(x)) for x in inp.flat) in _SOURCE_INPUTS


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    solutions = _find_solutions(inp)
    assert len(solutions) == 1, "the input must have one reconstruction"
    info = solutions[0]
    assert np.array_equal(info["output"], out)
    mask_color = info["mask_color"]
    changed = inp == mask_color
    assert np.all(inp[~changed] == out[~changed])
    assert not np.any(out == mask_color)
    assert not _source_copy(inp)
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_infos, test_infos = [], []
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp_key = _grid_key(pair["input"])
            pair_key = (inp_key, _grid_key(pair["output"]))
            assert inp_key not in inputs and pair_key not in pairs
            inputs.add(inp_key)
            pairs.add(pair_key)
            infos.append(_analyze_pair(pair))
    if any(len(i["rects"]) == 2 for i in test_infos):
        assert any(len(i["rects"]) == 2 for i in train_infos)
    if any(len(i["motif_colors"]) == 2 for i in test_infos):
        assert any(len(i["motif_colors"]) == 2 for i in train_infos)
    if any(i["orientation"] == "horizontal" for i in test_infos):
        assert any(i["orientation"] == "horizontal" for i in train_infos)


def _schedule(rng, train_count, test_count):
    parity_mode = int(rng.integers(0, 5))
    orientation_mode = int(rng.integers(0, 3))
    palette_mode = int(rng.integers(0, 2))
    richness = ("plain", "multi_mask", "multi_color")[int(rng.integers(0, 3))]
    n = train_count + test_count
    parity, orient = [], []
    for i in range(n):
        train = i < train_count
        if parity_mode == 0:
            p = 0 if train else 1
        elif parity_mode == 1:
            p = 1 if train else 0
        elif parity_mode == 2:
            p = i % 2 if train else int(rng.integers(0, 2))
        else:
            p = parity_mode - 3
        parity.append(p)
        if orientation_mode == 0:
            o = "vertical"
        elif orientation_mode == 1:
            o = "horizontal"
        else:
            o = ("vertical" if i % 2 == 0 else "horizontal") if train else (
                "vertical" if int(rng.integers(0, 2)) == 0 else "horizontal")
        orient.append(o)
    if "horizontal" in orient[train_count:] and "horizontal" not in orient[:train_count]:
        orient[1] = "horizontal"
    styles = []
    style_names = ("easy", "ordinary", "substantial", "edge", "protrusion")
    for i in range(n):
        probs = (0.22, 0.25, 0.19, 0.18, 0.16) if i < train_count else (
            0.10, 0.21, 0.27, 0.18, 0.24)
        styles.append(style_names[int(rng.choice(5, p=probs))])
    families = ("sparse", "dense", "body", "outline", "holes", "branches", "appendage")
    chosen_families = [families[int(rng.integers(len(families)))] for _ in range(n)]
    for i, style in enumerate(styles):
        if style == "protrusion":
            chosen_families[i] = "appendage"
    rich = ["plain"] * n
    if richness != "plain":
        rich[0] = richness
        if rng.random() < 0.7:
            rich[train_count] = richness
        for i in range(1, n):
            if rng.random() < 0.12:
                rich[i] = richness
    palettes = []
    common = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    for i in range(n):
        palette = common if palette_mode == 0 else tuple(
            int(x) for x in rng.choice(10, size=4, replace=False))
        palettes.append(palette)
    return parity, orient, styles, chosen_families, rich, palettes


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parity, orient, styles, families, rich, palettes = _schedule(
        rng, train_count, test_count)
    seeds = rng.integers(0, np.iinfo(np.uint64).max,
                         size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for i, pair_seed in enumerate(seeds):
        pair_rng = np.random.default_rng(pair_seed)
        for attempt in range(12):
            # Difficult combinations have a deterministic, ordinary-layout
            # fallback while retaining their parity, orientation and rule.
            style = styles[i] if attempt == 0 else "ordinary"
            family = families[i] if attempt == 0 else "body"
            try:
                inp, out = _make_pair(pair_rng, parity[i], orient[i], palettes[i],
                                      style, family, rich[i])
            except ValueError:
                continue
            if _grid_key(inp) not in seen and not _source_copy(inp):
                break
        else:
            raise ValueError("could not construct distinct pairs")
        seen.add(_grid_key(inp))
        puzzle["train" if i < train_count else "test"].append({
            "input": inp, "output": out
        })
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(
        len(a[split]) == len(b[split])
        and all(np.array_equal(x[k], y[k])
                for x, y in zip(a[split], b[split]) for k in ("input", "output"))
        for split in ("train", "test")
    )


def _expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _components8(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for start_y, start_x in np.argwhere(mask):
        start_y, start_x = int(start_y), int(start_x)
        if seen[start_y, start_x]:
            continue
        count += 1
        pending = [(start_y, start_x)]
        seen[start_y, start_x] = True
        while pending:
            y, x = pending.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                            and mask[yy, xx] and not seen[yy, xx]):
                        seen[yy, xx] = True
                        pending.append((yy, xx))
    return count


def self_test() -> None:
    for seed, train, test in (
        (201, 2, 1), (202, 10, 1), (203, 10, 4),
        (301, None, 1), (302, None, 4), (303, 2, None),
        (304, 10, None), (12345, 7, 3),
    ):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _equal_puzzles(generate_puzzle(777, 6, 2), generate_puzzle(777, 6, 2))
    assert not _equal_puzzles(generate_puzzle(778, 6, 2), generate_puzzle(779, 6, 2))
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        _expect_value_error(lambda args=args: generate_puzzle(1, *args))
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in corpus}) >= 4
    assert len({len(p["test"]) for p in corpus}) >= 3
    shapes, signatures, motifs, regimes = set(), set(), set(), set()
    within_shape = within_palette = within_orient = 0
    consistent_palette = role_exchange = 0
    edge = axis_shortcut_fails = easy = substantial = solid = 0
    coherent = fragmented = two_masks = two_colors = 0
    small = large = source_size = 0
    orientations = set()
    parity_by_split = {"train": set(), "test": set()}
    mask_heights_by_split = {"train": set(), "test": set()}
    sides = set()
    parity_modes = set()
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        all_pairs = puzzle["train"] + puzzle["test"]
        infos = [_analyze_pair(pair) for pair in all_pairs]
        local_shapes = {pair["input"].shape for pair in all_pairs}
        local_palettes = {(info["background"], info["motif_colors"], info["mask_color"])
                          for info in infos}
        local_orient = {info["orientation"] for info in infos}
        train_parity = {i["axis2"] % 2 for i in infos[:len(puzzle["train"])]}
        test_parity = {i["axis2"] % 2 for i in infos[len(puzzle["train"]):]}
        if train_parity == {1} and test_parity == {0}:
            parity_modes.add("source_shift")
        elif train_parity == {0} and test_parity == {1}:
            parity_modes.add("reverse_shift")
        elif len(train_parity) == 2:
            parity_modes.add("mixed")
        elif train_parity == test_parity:
            parity_modes.add("consistent")
        within_shape += len(local_shapes) > 1
        within_palette += len(local_palettes) > 1
        consistent_palette += len(local_palettes) == 1
        within_orient += len(local_orient) > 1
        roles = {}
        for info in infos:
            roles.setdefault(info["background"], set()).add("background")
            roles.setdefault(info["mask_color"], set()).add("mask")
            for color in info["motif_colors"]:
                roles.setdefault(color, set()).add("motif")
        role_exchange += any(len(uses) > 1 for uses in roles.values())
        shapes.update(local_shapes)
        for index, (pair, info) in enumerate(zip(all_pairs, infos)):
            split = "train" if index < len(puzzle["train"]) else "test"
            parity_by_split[split].add(info["axis2"] % 2)
            orientations.add(info["orientation"])
            sides.update(info["sides"])
            signatures.add((_grid_key(pair["input"]), _grid_key(pair["output"])))
            inp, out = pair["input"], pair["output"]
            small += min(inp.shape) < 14
            large += max(inp.shape) >= 28
            source_size += inp.shape == (16, 16)
            two_masks += len(info["rects"]) == 2
            two_colors += len(info["motif_colors"]) == 2
            for r0, r1, c0, c1 in info["rects"]:
                axis_height = r1 - r0 + 1 if info["orientation"] == "vertical" else c1 - c0 + 1
                mask_heights_by_split[split].add(axis_height)
                edge += (r0 == 0 or c0 == 0 or r1 == inp.shape[0] - 1
                         or c1 == inp.shape[1] - 1)
            motif_mask = out != info["background"]
            component_count = _components8(motif_mask)
            coherent += component_count == 1
            fragmented += component_count > 1
            covered = inp == info["mask_color"]
            hidden = int(np.count_nonzero(covered & motif_mask))
            easy += hidden <= 3
            substantial += hidden >= max(4, int(0.12 * np.count_nonzero(motif_mask)))
            solid += not np.any(covered & ~motif_mask)
            visible_axis_coords = np.where((inp != info["background"]) & ~covered)[
                1 if info["orientation"] == "vertical" else 0]
            axis_shortcut_fails += int(visible_axis_coords.min() + visible_axis_coords.max()) != info["axis2"]
            ys, xs = np.where(motif_mask)
            crop = motif_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            motifs.add((crop.shape, crop.tobytes()))
            regimes.add((info["orientation"], info["axis2"] % 2,
                         len(info["rects"]), len(info["motif_colors"]),
                         any(r0 == 0 or c0 == 0 or r1 == out.shape[0] - 1
                             or c1 == out.shape[1] - 1 for r0, r1, c0, c1 in info["rects"])))
    assert len(shapes) >= 90 and len(signatures) >= 170 and len(motifs) >= 140
    assert within_shape >= 36 and within_palette >= 10 and within_orient >= 8
    assert len(regimes) >= 10
    assert consistent_palette >= 5 and role_exchange >= 5
    assert orientations == {"vertical", "horizontal"} and sides == {"left", "right"}
    assert parity_by_split == {"train": {0, 1}, "test": {0, 1}}
    assert parity_modes == {"source_shift", "reverse_shift", "mixed", "consistent"}
    assert len(mask_heights_by_split["train"] & mask_heights_by_split["test"]) >= 3
    assert edge >= 25 and axis_shortcut_fails >= 20
    assert easy >= 20 and substantial >= 20 and solid >= 3
    assert coherent >= 30 and fragmented >= 20
    assert two_masks >= 5 and two_colors >= 5
    assert small >= 20 and large >= 20 and source_size >= 1
