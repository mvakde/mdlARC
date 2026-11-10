"""Synthetic instances of the monochromatic eight-connected object tally."""

import numpy as np


_SOURCE_ROWS = (
    ("000000000000", "022200666000", "022200666000", "022200666000",
     "000000000000", "000000333000", "000000333000", "077770333000",
     "077770000000", "077770008880", "077770008880", "000000000000"),
    ("000000000000", "066660000700", "060060007070", "060060070007",
     "066660007070", "000000000700", "005500880000", "005500880000",
     "005500000000", "005500000000", "000000000000"),
    ("00000000", "07777700", "07777700", "00000000", "00444044",
     "00444044"),
    ("000000000000", "000000000000", "000011000000", "000011000000",
     "000000000000", "000000000000"),
    ("00000000", "00006666", "01106006", "01106666", "00000000",
     "00440070", "00400770", "00440000"),
    ("3330333033", "3330333033", "3330333000", "0000000000"),
)
_SOURCE_KEYS = {(len(rows), len(rows[0]), "".join(rows)) for rows in _SOURCE_ROWS}


def _components(grid, background, eight=True):
    seen = np.zeros(grid.shape, dtype=bool)
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (eight or abs(dr) + abs(dc) == 1)]
    height, width = grid.shape
    result = []
    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color == background or seen[row, col]:
                continue
            seen[row, col] = True
            stack, cells = [(row, col)], []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < height and 0 <= cc < width
                            and not seen[rr, cc] and int(grid[rr, cc]) == color):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append((color, cells))
    return result


def _mask_from_cells(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    mask = np.zeros((max(r for r, _ in cells) - min_r + 1,
                     max(c for _, c in cells) - min_c + 1), dtype=bool)
    for r, c in cells:
        mask[r - min_r, c - min_c] = True
    return mask


def _motif(rng, kind):
    if kind == "solid":
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        if rng.random() < .3:
            h = 1 if rng.random() < .5 else h
            w = 1 if h != 1 and rng.random() < .5 else w
        shape = np.ones((h, w), dtype=bool)
    elif kind == "frame":
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        shape = np.ones((h, w), dtype=bool)
        shape[1:-1, 1:-1] = False
    elif kind == "diagonal":
        cells = {(0, 0)}
        target = int(rng.integers(3, 8))
        for _ in range(target - 1):
            choices = sorted({
                (r + dr, c + dc)
                for r, c in cells for dr in (-1, 1) for dc in (-1, 1)
                if -3 <= r + dr <= 3 and -3 <= c + dc <= 3
                and (r + dr, c + dc) not in cells
                and all(abs(r + dr - rr) + abs(c + dc - cc) != 1
                        for rr, cc in cells)
            })
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
        shape = _mask_from_cells(cells)
    elif kind == "ell":
        h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        shape = np.zeros((h, w), dtype=bool)
        shape[:, 0], shape[-1, :] = True, True
    elif kind == "bar":
        shape = np.ones((1, int(rng.integers(2, 8))), dtype=bool)
    elif kind == "cross":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        shape = np.zeros((h, w), dtype=bool)
        shape[h // 2, :], shape[:, w // 2] = True, True
    elif kind == "irregular":
        cells = {(0, 0)}
        target = int(rng.integers(4, 12))
        while len(cells) < target:
            choices = sorted({
                (r + dr, c + dc)
                for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and -3 <= r + dr <= 3 and -3 <= c + dc <= 3
                and (r + dr, c + dc) not in cells
            })
            cells.add(choices[int(rng.integers(len(choices)))])
        shape = _mask_from_cells(cells)
    else:
        raise ValueError("unknown motif")
    return np.rot90(shape, int(rng.integers(4)))


def _place(rng, shapes, colors, background, height, width, compact):
    grid = np.full((height, width), background, dtype=np.int8)
    order = sorted(range(len(shapes)), key=lambda i: shapes[i].size, reverse=True)
    for index in order:
        shape = shapes[index]
        sh, sw = shape.shape
        if sh > height or sw > width:
            return None
        candidates = []
        for row in range(height - sh + 1):
            for col in range(width - sw + 1):
                r0, r1 = max(0, row - 1), min(height, row + sh + 1)
                c0, c1 = max(0, col - 1), min(width, col + sw + 1)
                if np.any(grid[r0:r1, c0:c1] != background):
                    continue
                if any(shape[rr - row, cc - col]
                       for rr, cc in ((0, 0), (0, width - 1),
                                      (height - 1, 0), (height - 1, width - 1))
                       if row <= rr < row + sh and col <= cc < col + sw):
                    continue
                if compact and np.any(grid != background):
                    occupied = np.argwhere(grid != background)
                    center = np.mean(occupied, axis=0)
                    score = abs(row + sh / 2 - center[0]) + abs(col + sw / 2 - center[1])
                else:
                    score = float(rng.random())
                candidates.append((score, row, col))
        if not candidates:
            return None
        if compact and np.any(grid != background):
            candidates.sort()
            candidates = candidates[:max(1, len(candidates) // 6)]
        _, row, col = candidates[int(rng.integers(len(candidates)))]
        patch = grid[row:row + sh, col:col + sw]
        patch[shape] = colors[index]
    return grid


def _aligned(rng, count, color, background):
    height = int(rng.integers(2, 4))
    widths = [int(rng.integers(2, 5)) for _ in range(count)]
    gap = 1
    inner_width = sum(widths) + gap * (count - 1)
    horizontal = bool(rng.integers(2))
    grid = np.full((height + int(rng.integers(1, 4)),
                    inner_width + int(rng.integers(1, 4))),
                   background, dtype=np.int8)
    row = int(rng.integers(grid.shape[0] - height + 1))
    col = int(rng.integers(grid.shape[1] - inner_width + 1))
    for width in widths:
        grid[row:row + height, col:col + width] = color
        col += width + gap
    if not horizontal:
        grid = grid.T.copy()
    return grid


def _background(grid):
    h, w = grid.shape
    corners = [int(grid[0, 0]), int(grid[0, w - 1]),
               int(grid[h - 1, 0]), int(grid[h - 1, w - 1])]
    return max(set(corners), key=corners.count)


def _make_pair(rng, count, background, tally, palette, flags):
    if flags["aligned"] and 2 <= count <= 5:
        color = int(flags["anchor"] if flags["anchor"] is not None else
                    flags["novel"] if flags["novel"] is not None else
                    tally if flags["tally"] else rng.choice(palette))
        grid = _aligned(rng, count, color, background)
        output = np.array([[tally] * count + [background] * (6 - count),
                           ], dtype=np.int8)
        return {"input": grid, "output": output}

    kinds = ["solid" if flags["simple"] else
             str(rng.choice(("solid", "solid", "frame", "diagonal", "ell",
                             "bar", "cross", "irregular"))) for _ in range(count)]
    if flags["diagonal"]:
        kinds[int(rng.integers(count))] = "diagonal"
    colors = [int(rng.choice(palette)) for _ in range(count)]
    if flags["anchor"] is not None:
        colors[0] = int(flags["anchor"])
    if flags["repeat"] and count >= 2:
        repeated = int(flags["novel"] if flags["novel"] is not None
                       else colors[0])
        colors[0] = colors[1] = repeated
    elif flags["novel"] is not None:
        colors[0] = int(flags["novel"])
    if flags["tally"]:
        colors[-1] = tally
    if flags["repeat"] and count == 2 and flags["tally"]:
        colors[:] = [tally, tally]

    for _ in range(80):
        shapes = [_motif(rng, kind) for kind in kinds]
        footprint = sum((s.shape[0] + 1) * (s.shape[1] + 1) for s in shapes)
        base = int(np.ceil(np.sqrt(footprint)))
        low_h = max(4, max(s.shape[0] for s in shapes) + 1)
        low_w = max(4, max(s.shape[1] for s in shapes) + 1)
        compact = rng.random() < .55
        if compact:
            height = int(rng.integers(max(low_h, base - 3),
                                      min(23, max(low_h, base + 2)) + 1))
            width = int(rng.integers(max(low_w, base - 3),
                                     min(23, max(low_w, base + 2)) + 1))
        else:
            height = int(rng.integers(max(low_h, base), 24))
            width = int(rng.integers(max(low_w, base), 24))
        grid = _place(rng, shapes, colors, background, height, width, compact)
        if grid is not None and len(_components(grid, background)) == count:
            output = np.array([[tally] * count + [background] * (6 - count)],
                              dtype=np.int8)
            return {"input": grid, "output": output}
    raise RuntimeError("could not place valid components")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    for value, name, low, high in (
        (num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None:
        num_train = int(rng.integers(max(2, 1 if num_test is None else num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    backgrounds = [0]
    if rng.random() < .3:
        backgrounds = [int(rng.integers(0, 10))]
        if rng.random() < .7:
            backgrounds.append(int(rng.choice([c for c in range(10)
                                                if c not in backgrounds])))
    tally = int(rng.choice([c for c in range(1, 10) if c not in backgrounds]))
    available = [c for c in range(10) if c not in backgrounds]
    train_palette = [tally]
    extras = [c for c in available if c != tally]
    rng.shuffle(extras)
    train_palette += extras[:int(rng.integers(2, min(5, len(extras)) + 1))]
    novel_palette = [c for c in available if c not in train_palette]
    simple = rng.random() < .3
    train_counts = [int(rng.integers(2, 6)) for _ in range(num_train)]
    if num_train >= 3 and rng.random() < .28:
        train_counts[int(rng.integers(num_train))] = 1
    if len(set(train_counts)) == 1:
        train_counts[int(rng.integers(num_train))] = (
            5 if train_counts[0] != 5 else 3)
    test_counts = [int(rng.choice((1, 2, 3, 4, 5, 6),
                                  p=(.16, .16, .23, .24, .15, .06)))
                   for _ in range(num_test)]
    if 6 in test_counts and 5 not in train_counts:
        train_counts[int(rng.integers(num_train))] = 5
    if len(set(train_counts)) == 1:
        train_counts[0] = 3 if train_counts[0] != 3 else 4

    train_backgrounds = [int(rng.choice(backgrounds)) for _ in range(num_train)]
    if len(backgrounds) == 2:
        positions = rng.permutation(num_train)
        train_backgrounds[int(positions[0])] = backgrounds[0]
        train_backgrounds[int(positions[1])] = backgrounds[1]
    test_backgrounds = [int(rng.choice(backgrounds)) for _ in range(num_test)]
    diagonal_index = None if simple else int(rng.integers(num_train))
    repeat_index = int(rng.choice([i for i, count in enumerate(train_counts)
                                   if count >= 2]))
    tally_index = int(rng.integers(num_train))
    other_color_index = int(rng.choice([i for i in range(num_train)
                                        if i != tally_index]))
    train = []
    for i, count in enumerate(train_counts):
        flags = {
            "simple": simple,
            "diagonal": i == diagonal_index,
            "repeat": i == repeat_index,
            "tally": i == tally_index,
            "novel": None,
            "anchor": tally if i == tally_index else
                      train_palette[1] if i == other_color_index else None,
            "aligned": rng.random() < .12 and 2 <= count <= 5 and i != diagonal_index,
        }
        if flags["aligned"] and i == tally_index:
            flags["tally"] = True
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        train.append(_make_pair(pair_rng, count, train_backgrounds[i],
                                tally, train_palette, flags))

    test = []
    for i, count in enumerate(test_counts):
        novel = (int(rng.choice(novel_palette))
                 if novel_palette and rng.random() < .48 else None)
        flags = {
            "simple": simple,
            "diagonal": not simple and rng.random() < .38,
            "repeat": count >= 2 and rng.random() < .47,
            "tally": rng.random() < .38,
            "novel": novel,
            "anchor": None,
            "aligned": count in (2, 3, 4, 5) and rng.random() < .22,
        }
        if flags["aligned"]:
            flags["repeat"], flags["diagonal"] = True, False
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        test.append(_make_pair(pair_rng, count, test_backgrounds[i],
                               tally, train_palette, flags))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    inputs, pairs = set(), set()
    tally = None
    train_colors, backgrounds = set(), set()
    train_counts = []
    train_diagonal = train_repeat = train_tally_object = False
    test_diagonal = test_repeat = test_tally_object = False
    for regime, group in (("train", train), ("test", test)):
        for pair in group:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert out.shape == (1, 6)
            assert inp.shape != out.shape
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.shape, out.tobytes())
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            source_key = (inp.shape[0], inp.shape[1],
                          "".join(str(int(v)) for v in inp.flat))
            assert source_key not in _SOURCE_KEYS
            bg = _background(inp)
            corners = [int(inp[0, 0]), int(inp[0, -1]),
                       int(inp[-1, 0]), int(inp[-1, -1])]
            assert corners.count(bg) >= 3
            assert np.any(inp == bg)
            components = _components(inp, bg)
            count = len(components)
            assert 1 <= count <= 6
            this_tally = int(out[0, 0])
            assert this_tally != bg
            if tally is None:
                tally = this_tally
            assert tally == this_tally
            assert np.array_equal(out, np.array(
                [[tally] * count + [bg] * (6 - count)], dtype=np.int8))
            colors = [color for color, _ in components]
            diagonal = len(_components(inp, bg, eight=False)) > count
            repeat = len(set(colors)) < count
            tally_object = tally in colors
            if regime == "train":
                backgrounds.add(bg)
                train_counts.append(count)
                train_colors.update(colors)
                train_diagonal |= diagonal
                train_repeat |= repeat
                train_tally_object |= tally_object
            else:
                assert bg in backgrounds
                test_diagonal |= diagonal
                test_repeat |= repeat
                test_tally_object |= tally_object
                if count == 6:
                    assert 5 in train_counts
    assert len(set(train_counts)) >= 2
    assert len(train_colors) >= 2
    assert not test_diagonal or train_diagonal
    assert not test_repeat or train_repeat
    assert not test_tally_object or train_tally_object


def _same_puzzle(first, second):
    return all(
        np.array_equal(a[key], b[key])
        for regime in ("train", "test")
        for a, b in zip(first[regime], second[regime])
        for key in ("input", "output")
    ) and all(len(first[r]) == len(second[r]) for r in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(1000, 1042)]
    train_lengths, test_lengths, dimensions, layouts = set(), set(), set(), set()
    signatures, tallies, backgrounds, all_solid, diagonal, novel = set(), set(), set(), 0, 0, 0
    for seed, puzzle in zip(range(1000, 1042), puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        tally = int(puzzle["train"][0]["output"][0, 0])
        tallies.add(tally)
        known_colors = set()
        signature = []
        for pair in puzzle["train"]:
            bg = _background(pair["input"])
            known_colors.update(int(v) for v in np.unique(pair["input"]) if v != bg)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            bg = _background(inp)
            backgrounds.add(bg)
            count = len(_components(inp, bg))
            assert np.array_equal(out, np.array([[tally] * count + [bg] * (6 - count)]))
            dimensions.add(inp.shape)
            layouts.add((inp.shape, (inp != bg).tobytes()))
            signature.append((inp.shape, inp.tobytes(), out.tobytes()))
        signatures.add(tuple(signature))
        for pair in puzzle["test"]:
            inp = pair["input"]
            bg = _background(inp)
            diagonal += len(_components(inp, bg, eight=False)) > len(_components(inp, bg))
            novel += any(int(v) not in known_colors for v in np.unique(inp) if v != bg)
        all_solid += all(
            all(len(cells) == (max(r for r, _ in cells) - min(r for r, _ in cells) + 1)
                * (max(c for _, c in cells) - min(c for _, c in cells) + 1)
                for _, cells in _components(pair["input"], _background(pair["input"])))
            for pair in puzzle["train"] + puzzle["test"])
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert len(dimensions) >= 35 and len(layouts) >= 120
    assert len(signatures) == 42 and len(tallies) >= 4 and len(backgrounds) >= 3
    assert all_solid > 0 and diagonal > 0 and novel > 0
    coverage = set()
    four_positions = set()
    for puzzle in puzzles:
        train = puzzle["train"]
        test = puzzle["test"]
        tally = int(train[0]["output"][0, 0])
        known_colors = {
            int(color) for pair in train
            for color in np.unique(pair["input"])
            if int(color) != _background(pair["input"])
        }
        if len({_background(pair["input"]) for pair in train}) > 1:
            coverage.add("background_changes_in_training")
        if any(a["input"].shape == b["input"].shape for a in train for b in test):
            coverage.add("equal_train_test_size")
        if any(max(b["input"].shape) >= min(max(a["input"].shape) for a in train)
               for b in test):
            coverage.add("test_size_overlap")
        all_rectangles = True
        for split, group in (("train", train), ("test", test)):
            for index, pair in enumerate(group):
                inp = pair["input"]
                bg = _background(inp)
                objects = _components(inp, bg)
                colors = [color for color, _ in objects]
                repeated = len(set(colors)) < len(colors)
                diagonal = len(_components(inp, bg, eight=False)) > len(objects)
                if split == "test":
                    if len(objects) == 4:
                        four_positions.add(index)
                    if len(objects) == 6:
                        coverage.add("full_six_tally")
                    if any(color not in known_colors for color in colors):
                        coverage.add("new_test_color")
                    if (diagonal and repeated and tally in colors
                            and any(color not in known_colors for color in colors)):
                        coverage.add("composite_test")
                density = float(np.mean(inp != bg))
                if density >= .4:
                    coverage.add("dense_canvas")
                if max(inp.shape) <= 10:
                    coverage.add("compact_canvas")
                if min(inp.shape) >= 12:
                    coverage.add("large_canvas")
                boxes = []
                for color, cells in objects:
                    rows = [r for r, _ in cells]
                    cols = [c for _, c in cells]
                    top, bottom = min(rows), max(rows)
                    left, right = min(cols), max(cols)
                    height, width = bottom - top + 1, right - left + 1
                    full = len(cells) == height * width
                    all_rectangles &= full
                    boxes.append((color, top, bottom, left, right))
                    if full and max(height, width) >= 5 and min(height, width) <= 3:
                        coverage.add("long_thin_rectangle")
                    if (height >= 3 and width >= 3 and height != width
                            and len(cells) == 2 * height + 2 * width - 4
                            and all(r in (top, bottom) or c in (left, right)
                                    for r, c in cells)):
                        coverage.add("nonsquare_frame")
                    if (len(cells) >= 3 and len(set(rows)) > 1 and len(set(cols)) > 1
                            and all(not any(abs(r - rr) + abs(c - cc) == 1
                                            for rr, cc in cells if (rr, cc) != (r, c))
                                    for r, c in cells)
                            and len({r - c for r, c in cells}) > 1
                            and len({r + c for r, c in cells}) > 1):
                        coverage.add("bent_diagonal")
                    if not full and len(cells) >= 5:
                        coverage.add("irregular_or_hollow")
                for a in boxes:
                    for b in boxes:
                        if a is b or a[0] != b[0]:
                            continue
                        if ((b[3] - a[4] == 2 and max(a[1], b[1]) <= min(a[2], b[2]))
                                or (b[1] - a[2] == 2
                                    and max(a[3], b[3]) <= min(a[4], b[4]))):
                            coverage.add("near_aligned_same_color")
        if all_rectangles:
            coverage.add("all_rectangles_puzzle")
    assert len(four_positions) == 4
    assert coverage == {
        "background_changes_in_training", "equal_train_test_size",
        "test_size_overlap", "full_six_tally", "new_test_color",
        "composite_test", "dense_canvas", "compact_canvas", "large_canvas",
        "long_thin_rectangle", "nonsquare_frame", "bent_diagonal",
        "irregular_or_hollow", "near_aligned_same_color",
        "all_rectangles_puzzle",
    }
    for nt, ns in ((2, 1), (4, 3), (10, 1), (10, 4),
                   (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(7201, nt, ns)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        assert _same_puzzle(puzzle, generate_puzzle(7201, nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(7, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count was accepted")
    assert not _same_puzzle(generate_puzzle(12, 4, 2),
                            generate_puzzle(13, 4, 2))
