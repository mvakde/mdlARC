"""Synthetic ARC puzzles that compress a rectangle of solid lattice compartments."""

import hashlib

import numpy as np


_SOURCE_INPUTS = frozenset({
    (29, 29, "5a7207e5255d17e2a69a924f47ce9a3e52215bf3798d20ceeb1ba65da25f48cf"),
    (29, 29, "a05b140bd43449d3f129e895fb58e1214f16c82df4648f2dba6d4fb06340d528"),
    (27, 27, "c23cf526b5d73701b509b5a7077cf41c462554c9ce2097f6c13daab5d53b4824"),
})


def _count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train = _count("num_train", num_train)
    test = _count("num_test", num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return seed, train, test


def _orientation_distinct(array):
    transforms = [np.flipud(array), np.fliplr(array), np.rot90(array, 2)]
    if array.shape[0] == array.shape[1]:
        transforms.extend((array.T, np.rot90(array), np.rot90(array, 3)))
    return all(not np.array_equal(array, other) for other in transforms)


def _pick_geometry(rng, regime, strong, source_style, exact_source, index):
    # Some puzzles revisit the source's scale/shape shift. Other pairs are
    # compositional draws, including both reverse and shared-scale shifts.
    if source_style and regime == "source_train" and index < 2:
        side = 5 if index == 0 else 4
        macro_rows = macro_cols = 5 if index == 0 else 6
        payload_rows = payload_cols = 2 if index == 0 else 3
        top = (2 if index == 0 else 1) if exact_source else int(
            rng.integers(1, macro_rows - payload_rows))
        left = 1 if exact_source else int(
            rng.integers(1, macro_cols - payload_cols))
        return side, macro_rows, macro_cols, payload_rows, payload_cols, top, left
    if source_style and regime == "target_test" and index == 0:
        side, macro_rows, macro_cols = 6, 4, 4
        if exact_source:
            return 6, 4, 4, 3, 2, 1, 0
        payload_rows, payload_cols = (3, 2) if int(rng.integers(2)) else (2, 3)
        edge = int(rng.integers(4))
        if edge == 0:
            top, left = 0, int(rng.integers(3))
        elif edge == 1:
            top, left = macro_rows - payload_rows, int(rng.integers(3))
        elif edge == 2:
            top, left = int(rng.integers(macro_rows - payload_rows + 1)), 0
        else:
            top, left = int(rng.integers(macro_rows - payload_rows + 1)), macro_cols - payload_cols
        return side, macro_rows, macro_cols, payload_rows, payload_cols, top, left

    if regime == "target_test" and int(rng.integers(5)) == 0:
        side = int(rng.integers(2, 7))
        maximum_macro = 31 // (side + 1)
        macro_rows = int(rng.integers(4, maximum_macro + 1))
        macro_cols = int(rng.integers(4, maximum_macro + 1))
        payload_rows = payload_cols = int(rng.integers(1, 3))
        top = int(rng.integers(1, macro_rows - payload_rows))
        left = int(rng.integers(1, macro_cols - payload_cols))
        return side, macro_rows, macro_cols, payload_rows, payload_cols, top, left

    sides = np.arange(2, 9)
    weights = np.asarray((2, 3, 4, 4, 3, 2, 1) if regime == "source_train"
                         else (2, 3, 3, 3, 3, 2, 2), dtype=float)
    side = int(rng.choice(sides, p=weights / weights.sum()))
    maximum_macro = 31 // (side + 1)
    macro_rows = int(rng.integers(3, maximum_macro + 1))
    macro_cols = int(rng.integers(3, maximum_macro + 1))
    shapes = [(r, c) for r in range(1, macro_rows + 1)
              for c in range(1, macro_cols + 1)
              if r * c < macro_rows * macro_cols and
              (not strong or (r >= 2 and c >= 2))]
    area = macro_rows * macro_cols
    groups = (
        [s for s in shapes if 4 * s[0] * s[1] < area],
        [s for s in shapes if area <= 4 * s[0] * s[1] and 2 * s[0] * s[1] < area],
        [s for s in shapes if 2 * s[0] * s[1] >= area],
    )
    choices = groups[int(rng.integers(3))] or shapes
    payload_rows, payload_cols = choices[int(rng.integers(len(choices)))]
    locations = [(r, c) for r in range(macro_rows - payload_rows + 1)
                 for c in range(macro_cols - payload_cols + 1)]
    interior = [(r, c) for r, c in locations if r > 0 and c > 0 and
                r + payload_rows < macro_rows and c + payload_cols < macro_cols]
    edge = [(r, c) for r, c in locations if r == 0 or c == 0 or
            r + payload_rows == macro_rows or c + payload_cols == macro_cols]
    corner = [(r, c) for r, c in locations if r in (0, macro_rows - payload_rows)
              and c in (0, macro_cols - payload_cols)]
    preferred = (interior, edge, corner)[int(rng.integers(3))]
    locations = preferred or locations
    top, left = locations[int(rng.integers(len(locations)))]
    return side, macro_rows, macro_cols, payload_rows, payload_cols, top, left


def _wallpaper(rng, divider, side, palette_override=None, family_override=None):
    if palette_override is None:
        palette_size = int(rng.choice(np.arange(2, 9), p=np.asarray(
            (1, 2, 3, 3, 3, 2, 1), dtype=float) / 15))
        colors = [int(v) for v in rng.permutation(
            [c for c in range(10) if c != divider])[:palette_size]]
    else:
        colors = [int(v) for v in rng.permutation(palette_override)]
        palette_size = len(colors)
    py = int(rng.integers(2, min(9, 2 * side + 3)))
    px = int(rng.integers(2, min(9, 2 * side + 3)))
    by = int(rng.integers(1, min(4, py) + 1))
    bx = int(rng.integers(1, min(4, px) + 1))
    y, x = np.indices((py, px))
    kind = int(rng.integers(6)) if family_override is None else family_override
    if kind == 0:
        indices = y // by + 2 * (x // bx)
    elif kind == 1:
        indices = 2 * (y // by) + x // bx
    elif kind == 2:
        indices = (y // by) * (x // bx) + y // by + x // bx
    elif kind == 3:
        indices = y // by + (x // bx) ** 2
    elif kind == 4:
        indices = np.bitwise_xor(y // by, x // bx)
    else:
        indices = rng.integers(0, palette_size, size=(py, px))
    indices = (indices + int(rng.integers(palette_size))) % palette_size
    indices = np.rot90(indices, int(rng.integers(4)))
    if int(rng.integers(2)):
        indices = np.fliplr(indices)
    if int(rng.integers(2)):
        indices = np.flipud(indices)
    return np.asarray(colors, dtype=np.int8)[indices]


def _payload_indices(rng, shape, strong):
    rows, cols = shape
    size = rows * cols
    mode = "rich" if strong else (
        "mono", "two", "stripe", "mirror", "rich", "random")[int(rng.integers(6))]
    if mode == "mono":
        return np.zeros(shape, dtype=np.int8)
    if mode == "stripe" and max(rows, cols) >= 2:
        along_rows = rows >= 2 and (cols == 1 or bool(rng.integers(2)))
        extent = rows if along_rows else cols
        count = int(rng.integers(2, min(5, extent) + 1))
        labels = np.arange(extent) % count
        rng.shuffle(labels)
        return np.broadcast_to(labels[:, None] if along_rows else labels[None, :], shape).copy()
    if mode == "mirror" and size >= 2:
        horizontal = bool(rng.integers(2))
        orbit_count = rows * ((cols + 1) // 2) if horizontal else ((rows + 1) // 2) * cols
        count = int(rng.integers(1, min(5, orbit_count) + 1))
        labels = np.arange(orbit_count) % count
        rng.shuffle(labels)
        base = labels.reshape((rows, (cols + 1) // 2) if horizontal
                              else ((rows + 1) // 2, cols))
        if horizontal:
            return np.concatenate((base, np.fliplr(base[:, :cols // 2])), axis=1)
        return np.concatenate((base, np.flipud(base[:rows // 2])), axis=0)
    if mode == "two" and size >= 2:
        count = 2
    elif mode == "rich":
        count = int(rng.integers(min(3, size), min(9, size) + 1))
    else:
        count = int(rng.integers(1, min(9, size) + 1))
    values = np.concatenate((np.arange(count), rng.integers(count, size=size - count)))
    rng.shuffle(values)
    return values.reshape(shape)


def _make_payload(rng, shape, divider, wallpaper_colors, strong):
    wallpaper = list(wallpaper_colors)
    other = [c for c in range(10) if c != divider and c not in wallpaper]
    for _ in range(32):
        labels = _payload_indices(rng, shape, strong)
        count = len(np.unique(labels))
        options = [v for v in range(count + 1)
                   if v <= len(wallpaper) and count - v <= len(other)]
        if strong:
            options = [v for v in options if v >= 1] or options
        if not options:
            continue
        groups = ([v for v in options if v == 0],
                  [v for v in options if 0 < v < count],
                  [v for v in options if v == count])
        available = [group for group in groups if group]
        chosen = available[int(rng.integers(len(available)))]
        overlap = int(chosen[int(rng.integers(len(chosen)))])
        colors = [int(v) for v in rng.choice(wallpaper, size=overlap, replace=False)]
        colors += [int(v) for v in rng.choice(other, size=count - overlap, replace=False)]
        rng.shuffle(colors)
        payload = np.asarray(colors, dtype=np.int8)[labels]
        if not strong or (_orientation_distinct(payload) and len(np.unique(payload)) >= 3):
            return payload
    raise RuntimeError("could not make orientation evidence")


def _candidate(grid, divider):
    height, width = grid.shape
    row_lines = np.flatnonzero(np.all(grid == divider, axis=1))
    col_lines = np.flatnonzero(np.all(grid == divider, axis=0))
    macro_rows, macro_cols = len(row_lines) + 1, len(col_lines) + 1
    if macro_rows < 2 or macro_cols < 2:
        return None
    if (height - macro_rows + 1) % macro_rows or (width - macro_cols + 1) % macro_cols:
        return None
    side = (height - macro_rows + 1) // macro_rows
    if side < 2 or (width - macro_cols + 1) // macro_cols != side:
        return None
    if not np.array_equal(row_lines, side + np.arange(macro_rows - 1) * (side + 1)):
        return None
    if not np.array_equal(col_lines, side + np.arange(macro_cols - 1) * (side + 1)):
        return None
    mask = np.zeros((macro_rows, macro_cols), dtype=bool)
    values = np.full((macro_rows, macro_cols), -1, dtype=np.int8)
    wallpaper_tiles = []
    wallpaper_colors = set()
    for row in range(macro_rows):
        for col in range(macro_cols):
            tile = grid[row * (side + 1):row * (side + 1) + side,
                        col * (side + 1):col * (side + 1) + side]
            if np.any(tile == divider):
                return None
            if np.all(tile == tile[0, 0]):
                mask[row, col] = True
                values[row, col] = tile[0, 0]
            else:
                wallpaper_tiles.append((row, col, tile))
                wallpaper_colors.update(int(v) for v in np.unique(tile))
    where = np.argwhere(mask)
    if not len(where) or not wallpaper_tiles:
        return None
    top, left = np.min(where, axis=0)
    bottom, right = np.max(where, axis=0) + 1
    if int(mask.sum()) != int((bottom - top) * (right - left)):
        return None
    output = values[top:bottom, left:right].copy()
    return output, {
        "side": int(side), "macro_shape": (macro_rows, macro_cols),
        "payload_shape": tuple(output.shape),
        "bounds": (int(top), int(left), int(bottom), int(right)),
        "divider": int(divider), "wallpaper_palette": tuple(sorted(wallpaper_colors)),
    }


def _decode_input(grid):
    candidates = []
    for divider in range(10):
        result = _candidate(grid, divider)
        if result is not None:
            candidates.append(result)
    assert len(candidates) == 1, "input must have a unique lattice and solid rectangle"
    return candidates[0]


def _make_pair(rng, regime, style, index):
    strong = regime == "source_train" and index == 0
    for _ in range(192):
        side, macro_rows, macro_cols, pr, pc, top, left = _pick_geometry(
            rng, regime, strong, style["source_style"], style["exact_source"], index)
        if style["stable_divider"]:
            divider = style["anchor_divider"]
        elif style["stable_palette"]:
            divider = int(rng.choice(style["divider_choices"]))
        else:
            divider = int(rng.integers(10))
        motif = _wallpaper(
            rng, divider, side,
            style["anchor_palette"] if style["stable_palette"] else None,
            style["anchor_family"] if style["stable_family"] else None,
        )
        py, px = motif.shape
        height = macro_rows * side + macro_rows - 1
        width = macro_cols * side + macro_cols - 1
        grid = np.full((height, width), divider, dtype=np.int8)
        wallpaper_colors = set()
        ordinary_tiles = []
        for row in range(macro_rows):
            for col in range(macro_cols):
                if top <= row < top + pr and left <= col < left + pc:
                    continue
                ys = (row * side + np.arange(side)) % py
                xs = (col * side + np.arange(side)) % px
                tile = motif[np.ix_(ys, xs)]
                if np.all(tile == tile[0, 0]):
                    break
                ordinary_tiles.append((row, col, tile))
                wallpaper_colors.update(int(v) for v in np.unique(tile))
            else:
                continue
            break
        else:
            if len(wallpaper_colors) < 2:
                continue
            if style["stable_palette"] and wallpaper_colors != set(style["anchor_palette"]):
                continue
            payload = _make_payload(rng, (pr, pc), divider, wallpaper_colors, strong)
            for row, col, tile in ordinary_tiles:
                grid[row * (side + 1):row * (side + 1) + side,
                     col * (side + 1):col * (side + 1) + side] = tile
            for row in range(pr):
                for col in range(pc):
                    grid[(top + row) * (side + 1):(top + row) * (side + 1) + side,
                         (left + col) * (side + 1):(left + col) * (side + 1) + side] = payload[row, col]
            try:
                decoded, _ = _decode_input(grid)
            except AssertionError:
                continue
            if np.array_equal(decoded, payload):
                return {"input": grid, "output": payload}
    raise RuntimeError("could not construct a readable lattice pair")


def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed, train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Row-major compression is the sole puzzle-level convention. No color,
    # scale, placement, or wallpaper code is shared secretly across pairs.
    color_order = tuple(int(v) for v in rng.permutation(10))
    palette_size = int(rng.integers(3, 7))
    anchor_palette = color_order[:palette_size]
    divider_choices = color_order[palette_size:]
    style = {
        "source_style": seed % 7 == 0,
        "exact_source": seed % 14 == 0,
        "stable_divider": bool(rng.integers(3) == 0),
        "anchor_divider": divider_choices[0],
        "stable_palette": bool(rng.integers(3) == 0),
        "anchor_palette": anchor_palette,
        "divider_choices": divider_choices,
        "stable_family": bool(rng.integers(3) == 0),
        "anchor_family": int(rng.integers(6)),
    }
    puzzle = {"train": [], "test": []}
    used = set()
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            for _ in range(32):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, regime, style, index)
                key = _grid_key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not avoid a duplicate input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs = set(), set()
    training_evidence = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= extent <= 30 for extent in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_grid, output_grid = pair["input"], pair["output"]
            key = _grid_key(input_grid)
            pair_key = (key, _grid_key(output_grid))
            assert key not in used_inputs and pair_key not in used_pairs
            used_inputs.add(key)
            used_pairs.add(pair_key)
            source_hash = hashlib.sha256(key[1]).hexdigest()
            assert (*input_grid.shape, source_hash) not in _SOURCE_INPUTS
            decoded, analysis = _decode_input(input_grid)
            assert np.array_equal(decoded, output_grid), "output does not decode from input"
            assert input_grid.shape != output_grid.shape or not np.array_equal(input_grid, output_grid)
            assert analysis["side"] >= 2
            if split == "train" and output_grid.shape[0] >= 2 and output_grid.shape[1] >= 2:
                training_evidence |= (len(np.unique(output_grid)) >= 3 and
                                      _orientation_distinct(output_grid))
    assert training_evidence, "training must establish unchanged row-major orientation"


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1004, 3, 2), (1005, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(99, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {request}")
    try:
        generate_puzzle(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative seed")
    first = generate_puzzle(314159, 8, 3)
    second = generate_puzzle(314159, 8, 3)
    assert all(_grid_key(a[role]) == _grid_key(b[role])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for role in ("input", "output"))
    signatures, shapes, sides = set(), set(), set()
    source_layouts = set()
    observed = set()
    for puzzle in puzzles:
        by_split = {split: [] for split in ("train", "test")}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                output, analysis = _decode_input(pair["input"])
                assert np.array_equal(output, pair["output"])
                by_split[split].append(analysis)
                side = analysis["side"]
                rows, cols = analysis["macro_shape"]
                pr, pc = analysis["payload_shape"]
                top, left, bottom, right = analysis["bounds"]
                signatures.add((side, rows, cols, pr, pc, top, left))
                source_layouts.add((side, rows, cols, pr, pc, top, left))
                shapes.add(tuple(pair["input"].shape))
                sides.add(side)
                if split == "train":
                    if pr != pc:
                        observed.add("train_rectangle")
                    if top == 0 or left == 0 or bottom == rows or right == cols:
                        observed.add("train_edge")
                else:
                    if pr == pc:
                        observed.add("test_square")
                    if top > 0 and left > 0 and bottom < rows and right < cols:
                        observed.add("test_interior")
                count = len(np.unique(output))
                output_colors = set(int(v) for v in np.unique(output))
                wallpaper_colors = set(analysis["wallpaper_palette"])
                overlap = len(output_colors & wallpaper_colors)
                if overlap == 0:
                    observed.add("palette_disjoint")
                elif overlap == count:
                    observed.add("palette_full_overlap")
                else:
                    observed.add("palette_partial_overlap")
                if count == 1:
                    observed.add("monochrome")
                if count == 2:
                    observed.add("two_color")
                if not _orientation_distinct(output):
                    observed.add("symmetric")
                if count >= 7:
                    observed.add("rich")
                if len(analysis["wallpaper_palette"]) >= 6:
                    observed.add("six_wallpaper")
                if 2 * pr * pc >= rows * cols:
                    observed.add("majority")
                if (pr, pc) == (1, 1):
                    observed.add("one_tile")
                if max(pr, pc) >= 5 and min(pr, pc) == 1:
                    observed.add("long_axis")
                if split == "train" and pr != pc and (
                    top == 0 or left == 0 or bottom == rows or right == cols
                ) and count == 1:
                    observed.add("combined_train_edge_rectangle_mono")
                if split == "test" and pr == pc and (
                    top > 0 and left > 0 and bottom < rows and right < cols
                ):
                    observed.add("combined_test_interior_square")
        train_sides = {a["side"] for a in by_split["train"]}
        test_sides = {a["side"] for a in by_split["test"]}
        if train_sides == {4, 5} and 6 in test_sides:
            observed.add("source_shift")
        if train_sides & test_sides:
            observed.add("shared_scale")
        if min(test_sides) < max(train_sides):
            observed.add("smaller_test")
        analyses = by_split["train"] + by_split["test"]
        if len({a["divider"] for a in analyses}) == 1:
            observed.add("stable_divider")
        else:
            observed.add("changing_divider")
        if len({a["wallpaper_palette"] for a in analyses}) == 1:
            observed.add("stable_palette")
        else:
            observed.add("changing_palette")
        if any(a["side"] == b["side"] and a["payload_shape"] != b["payload_shape"]
               for i, a in enumerate(analyses) for b in analyses[i + 1:]):
            observed.add("same_side_different_shape")
    assert len(signatures) >= 100 and len(shapes) >= 15
    assert sides == set(range(2, 9))
    required = {"train_rectangle", "train_edge", "test_square", "test_interior",
                "monochrome", "two_color", "symmetric", "rich", "six_wallpaper",
                "majority", "one_tile", "long_axis", "source_shift",
                "shared_scale", "smaller_test", "same_side_different_shape",
                "palette_disjoint", "palette_full_overlap", "palette_partial_overlap",
                "stable_divider", "changing_divider", "stable_palette",
                "changing_palette", "combined_train_edge_rectangle_mono",
                "combined_test_interior_square"}
    assert required <= observed, required - observed
    assert {(5, 5, 5, 2, 2, 2, 1), (4, 6, 6, 3, 3, 1, 1),
            (6, 4, 4, 3, 2, 1, 0)} <= source_layouts
