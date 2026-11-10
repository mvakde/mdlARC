"""Generate a masked crop recovered by reflection across the row centerline."""

import numpy as np


def _pair_counts(seed, num_train, num_test, rng):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _texture(rng, height, width, count):
    """Compose short runs, small patches, mosaics, and irregular motifs."""
    style = int(rng.integers(4))
    if style == 0:
        indices = np.empty((height, width), dtype=np.int8)
        for row in range(height):
            col = 0
            while col < width:
                run = int(rng.integers(1, 5))
                value = (indices[row - 1, col] if row and rng.random() < .24
                         else rng.integers(count))
                indices[row, col:col + run] = value
                col += run
    elif style == 1:
        indices = rng.integers(count, size=(height, width), dtype=np.int8)
        for _ in range(max(2, height * width // 6)):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            indices[row:row + int(rng.integers(1, 4)),
                    col:col + int(rng.integers(1, 5))] = rng.integers(count)
    elif style == 2:
        indices = np.empty((height, width), dtype=np.int8)
        row = 0
        while row < height:
            row_run = int(rng.integers(1, 4))
            col = 0
            while col < width:
                col_run = int(rng.integers(1, 5))
                indices[row:row + row_run, col:col + col_run] = rng.integers(count)
                col += col_run
            row += row_run
        for _ in range(max(1, height * width // 9)):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            indices[row, col] = rng.integers(count)
    else:
        rows, cols = np.indices((height, width))
        a, b, divisor, period = (int(rng.integers(1, count + 2)),
                                  int(rng.integers(1, count + 2)),
                                  int(rng.integers(2, 6)),
                                  int(rng.integers(2, 6)))
        indices = ((a * (rows // int(rng.integers(1, 3)))
                    + b * (cols // int(rng.integers(1, 4)))
                    + (rows * cols) // divisor
                    + (rows + cols) % period) % count).astype(np.int8)
        for _ in range(max(2, height * width // 10)):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            indices[row:row + int(rng.integers(1, 3)),
                    col:col + int(rng.integers(1, 4))] = rng.integers(count)
    return indices


def _make_pattern(rng, height, width, mask_color, bilateral):
    allowed = np.array([c for c in range(10) if c != mask_color], dtype=np.int8)
    palette = rng.choice(allowed, size=int(rng.integers(3, 10)), replace=False)
    top_height = (height + 1) // 2
    half_width = (width + 1) // 2 if bilateral else width
    top = palette[_texture(rng, top_height, half_width, len(palette))]
    if bilateral:
        top = np.concatenate((top, top[:, -2::-1] if width % 2 else top[:, ::-1]), axis=1)
    bottom = top[-2::-1, :] if height % 2 else top[::-1, :]
    return np.concatenate((top, bottom), axis=0).astype(np.int8)


def _size(rng, limit, witness=False):
    if witness:
        return int(rng.integers(3, min(limit, 9) + 1))
    mode = rng.random()
    if mode < .12:
        return 1
    if mode < .76 or limit <= 8:
        return int(rng.integers(2, min(limit, 8) + 1))
    return int(rng.integers(9, limit + 1))


def _locations(width, mask_width, zone):
    middle = width // 2
    right_start = (width + 1) // 2
    positions = range(width - mask_width + 1)
    if zone == "left":
        return [c for c in positions if c + mask_width <= middle]
    if zone == "right":
        return [c for c in positions if c >= right_start]
    return [c for c in positions if c < right_start and c + mask_width > middle]


def _mask_candidates(inp):
    height, width = inp.shape
    candidates = []
    for color in np.unique(inp):
        cells = np.argwhere(inp == color)
        start = cells.min(axis=0)
        end = cells.max(axis=0) + 1
        r0, c0 = map(int, start)
        r1, c1 = map(int, end)
        if len(cells) != (r1 - r0) * (c1 - c0):
            continue
        if not (r1 <= height // 2 or r0 >= (height + 1) // 2):
            continue
        reflected = inp[height - r1:height - r0, c0:c1][::-1, :]
        if np.any(reflected == color):
            continue
        restored = inp.copy()
        restored[r0:r1, c0:c1] = reflected
        if np.array_equal(restored, restored[::-1, :]):
            candidates.append((int(color), r0, c0, r1 - r0, c1 - c0,
                               reflected.copy(), restored))
    return candidates


def _make_pair(rng, mask_color, bilateral, side, zone, witness=False,
               forbidden_shape=None):
    for _ in range(240):
        height = int(rng.integers(12, 31))
        width = int(rng.integers(14, 31))
        half_height = height // 2
        mask_height = _size(rng, half_height, witness)
        width_limit = width if zone == "cross" else width // 2
        mask_width = _size(rng, min(width_limit, 15), witness)
        if forbidden_shape == (mask_height, mask_width):
            continue
        columns = _locations(width, mask_width, zone)
        if not columns:
            continue
        col = int(rng.choice(columns))
        if side == "upper":
            row = int(rng.integers(half_height - mask_height + 1))
        else:
            row = int(rng.integers((height + 1) // 2,
                                   height - mask_height + 1))
        pattern = _make_pattern(rng, height, width, mask_color, bilateral)
        output = pattern[row:row + mask_height, col:col + mask_width].copy()
        if witness and np.array_equal(output, output[::-1, :]):
            continue
        if len(np.unique(output)) < (2 if output.size > 1 else 1):
            continue
        inp = pattern.copy()
        inp[row:row + mask_height, col:col + mask_width] = mask_color
        candidates = _mask_candidates(inp)
        if len(candidates) != 1 or candidates[0][0:5] != (
                mask_color, row, col, mask_height, mask_width):
            continue
        if not np.array_equal(candidates[0][5], output):
            continue
        if not bilateral and np.array_equal(pattern, pattern[:, ::-1]):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("Could not construct a uniquely recoverable mask")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(seed, num_train, num_test, rng)
    bilateral = bool(rng.random() < .78)
    fixed_mask = bool(rng.random() < .64)
    primary_mask = int(rng.integers(10))
    alternate_mask = int(rng.choice([c for c in range(10) if c != primary_mask]))
    first_side = "upper" if rng.random() < .5 else "lower"
    witness_index = int(rng.integers(train_count))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_zones = []
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train" and index == 0:
                side = first_side
            elif split == "train" and index == 1:
                side = "lower" if first_side == "upper" else "upper"
            else:
                side = str(rng.choice(("upper", "lower")))
            witness = split == "train" and index == witness_index
            if witness:
                zone = "cross"
            else:
                zones = ("left", "right", "cross")
                if split == "test" and rng.random() < .55:
                    fewest = min(train_zones.count(z) for z in zones)
                    choices = [z for z in zones if train_zones.count(z) == fewest]
                    zone = str(rng.choice(choices))
                else:
                    zone = str(rng.choice(zones))
            mask_color = (primary_mask if fixed_mask or index == 0 else
                          alternate_mask if split == "train" and index == 1 else
                          int(rng.integers(10)))
            forbidden_shape = (puzzle["train"][0]["output"].shape
                               if split == "train" and index == 1 else None)
            for _ in range(20):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair = _make_pair(np.random.default_rng(child_seed), mask_color,
                                  bilateral, side, zone, witness, forbidden_shape)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not construct a unique input")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_zones.append(zone)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    candidates = _mask_candidates(inp)
    assert len(candidates) == 1
    color, row, col, mask_height, mask_width, recovered, restored = candidates[0]
    assert out.shape == (mask_height, mask_width)
    assert np.array_equal(out, recovered)
    assert not np.any(out == color)
    assert len(np.unique(out)) >= (2 if out.size > 1 else 1)
    height, width = inp.shape
    if col + mask_width <= width // 2:
        zone = "left"
    elif col >= (width + 1) // 2:
        zone = "right"
    else:
        zone = "cross"
    return {"color": color, "row": row, "col": col,
            "height": mask_height, "width": mask_width,
            "side": "upper" if row + mask_height <= height // 2 else "lower",
            "zone": zone, "canvas": inp.shape,
            "bilateral": bool(np.array_equal(restored, restored[:, ::-1])),
            "palette_size": len(np.unique(restored))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen = set()
    details = {}
    for split, pairs in (("train", train), ("test", test)):
        details[split] = []
        for pair in pairs:
            detail = _analyze_pair(pair)
            details[split].append(detail)
            key = (pair["input"].shape, pair["input"].tobytes())
            assert key not in seen
            seen.add(key)
    train_details = details["train"]
    assert {d["side"] for d in train_details} == {"upper", "lower"}
    assert len({(d["height"], d["width"]) for d in train_details}) >= 2
    assert any(d["zone"] == "cross" and d["height"] >= 2
               and not np.array_equal(pair["output"], pair["output"][::-1, :])
               for d, pair in zip(train_details, train))
    assert len({d["bilateral"] for d in train_details + details["test"]}) == 1
    mask_colors = {d["color"] for d in train_details + details["test"]}
    if len(mask_colors) > 1:
        assert len({d["color"] for d in train_details}) > 1


def _same_puzzle(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    canvas_shapes, output_shapes, layouts, signatures = set(), set(), set(), set()
    palettes, mask_colors, sides, zones, test_sides, test_zones = (
        set(), set(), set(), set(), set(), set())
    bilateral_modes, hue_modes = set(), set()
    strips = large = edge = center_adjacent = zero_pattern = 0
    per_puzzle_shape_variety = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pairs = puzzle["train"] + puzzle["test"]
        details = [_analyze_pair(pair) for pair in pairs]
        hue_modes.add(len({d["color"] for d in details}) > 1)
        bilateral_modes.add(details[0]["bilateral"])
        per_puzzle_shape_variety += len({d["canvas"] for d in details}) > 1
        for index, (pair, d) in enumerate(zip(pairs, details)):
            h, w = d["canvas"]
            canvas_shapes.add((h, w))
            output_shapes.add((d["height"], d["width"]))
            layouts.add((h, w, d["row"], d["col"], d["height"], d["width"]))
            signatures.add((pair["input"].shape, pair["input"].tobytes()))
            palettes.add(d["palette_size"])
            mask_colors.add(d["color"])
            sides.add(d["side"])
            zones.add(d["zone"])
            if index >= len(puzzle["train"]):
                test_sides.add(d["side"])
                test_zones.add(d["zone"])
            strips += d["height"] == 1 or d["width"] == 1
            large += d["height"] >= 9 or d["width"] >= 9
            edge += (d["row"] == 0 or d["col"] == 0 or
                     d["row"] + d["height"] == h or d["col"] + d["width"] == w)
            center_adjacent += (d["row"] + d["height"] == h // 2 or
                                d["row"] == (h + 1) // 2)
            zero_pattern += 0 in pair["input"] and d["color"] != 0
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(canvas_shapes) >= 50 and len(output_shapes) >= 30
    assert len(layouts) >= 100 and len(signatures) >= 100
    assert len(palettes) >= 5 and len(mask_colors) == 10
    assert sides == test_sides == {"upper", "lower"}
    assert zones == test_zones == {"left", "right", "cross"}
    assert bilateral_modes == hue_modes == {True, False}
    assert strips >= 5 and large >= 5 and edge >= 5 and center_adjacent >= 5
    assert zero_pattern >= 5 and per_puzzle_shape_variety >= 38
    for counts in ((2, 1), (10, 1), (10, 4), (6, 3),
                   (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(117, *counts)
        second = generate_puzzle(117, *counts)
        assert _same_puzzle(first, second)
    assert _same_puzzle(generate_puzzle(19), generate_puzzle(19))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(17, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
    damaged = generate_puzzle(433, 3, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = (
        int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect recovered crop was accepted")
