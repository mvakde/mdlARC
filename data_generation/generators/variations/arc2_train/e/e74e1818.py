import numpy as np


_SOURCE_TEXT_PAIRS = (
    (
        (
            "00000000000",
            "00111111100",
            "00011011000",
            "00088888000",
            "00050505000",
            "00050005000",
            "00055555000",
            "00002220000",
            "00000200000",
            "00033333000",
            "00000000000",
        ),
        (
            "00000000000",
            "00011011000",
            "00111111100",
            "00088888000",
            "00055555000",
            "00050005000",
            "00050505000",
            "00000200000",
            "00002220000",
            "00033333000",
            "00000000000",
        ),
    ),
    (
        (
            "0000000000000",
            "0002222222000",
            "0000202020000",
            "0000020200000",
            "0003003003000",
            "0000333330000",
            "0000111110000",
            "0000100010000",
            "0000111110000",
            "0000440440000",
            "0000004000000",
            "0000004000000",
            "0000000000000",
        ),
        (
            "0000000000000",
            "0000020200000",
            "0000202020000",
            "0002222222000",
            "0000333330000",
            "0003003003000",
            "0000111110000",
            "0000100010000",
            "0000111110000",
            "0000004000000",
            "0000004000000",
            "0000440440000",
            "0000000000000",
        ),
    ),
    (
        (
            "0000000",
            "0099900",
            "0909090",
            "0044400",
            "0333330",
            "0003000",
            "0000000",
        ),
        (
            "0000000",
            "0909090",
            "0099900",
            "0044400",
            "0003000",
            "0333330",
            "0000000",
        ),
    ),
    (
        (
            "000000000000000",
            "002002000200200",
            "000222020222000",
            "000002000200000",
            "000033333330000",
            "000003000300000",
            "000000333000000",
            "000000030000000",
            "000444404444000",
            "000040404040000",
            "000044404440000",
            "000000888000000",
            "000008808800000",
            "008888888888800",
            "000000000000000",
        ),
        (
            "000000000000000",
            "000002000200000",
            "000222020222000",
            "002002000200200",
            "000000030000000",
            "000000333000000",
            "000003000300000",
            "000033333330000",
            "000044404440000",
            "000040404040000",
            "000444404444000",
            "008888888888800",
            "000008808800000",
            "000000888000000",
            "000000000000000",
        ),
    ),
)


def _text_grid(rows):
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _source_arrays():
    return [
        (_text_grid(input_rows), _text_grid(output_rows))
        for input_rows, output_rows in _SOURCE_TEXT_PAIRS
    ]


def _grid_key(grid):
    return tuple(grid.shape), grid.dtype.str, grid.tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _background(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    return background


def _motif_records(grid, background):
    """Separate each color into occupied-row runs, including repeated colors."""
    records = []
    for value in np.unique(grid):
        color = int(value)
        if color == background:
            continue
        occupied = np.any(grid == color, axis=1)
        starts = np.flatnonzero(occupied & ~np.r_[False, occupied[:-1]])
        ends = np.flatnonzero(occupied & ~np.r_[occupied[1:], False])
        for top, bottom in zip(starts, ends):
            columns = np.flatnonzero(np.any(grid[top:bottom + 1] == color, axis=0))
            left, right = int(columns[0]), int(columns[-1])
            box = grid[top:bottom + 1, left:right + 1]
            assert np.all((box == color) | (box == background))
            mask = box == color
            records.append({
                "color": color,
                "bbox": (int(top), int(bottom), left, right),
                "mask": mask,
            })
    return sorted(records, key=lambda rec: (rec["bbox"][0], rec["bbox"][2]))


def _reflect_grid(grid, background):
    result = np.full_like(grid, background)
    for record in _motif_records(grid, background):
        top, bottom, left, right = record["bbox"]
        block = result[top:bottom + 1, left:right + 1]
        assert not np.any(block[record["mask"][::-1]] != background)
        block[record["mask"][::-1]] = record["color"]
    return result


def _pattern(rng, height, width, symmetric):
    """Compose bars, frames, stems, steps, strokes, and occasional irregular masks."""
    if height == 1:
        mask = rng.random((1, width)) < rng.uniform(.45, .9)
        mask[0, 0] = mask[0, -1] = True
        return mask

    for _ in range(40):
        mask = np.zeros((height, width), dtype=bool)
        style = int(rng.integers(0, 6))
        if style == 0:  # crossbar and one or two stems
            bar = int(rng.integers(0, height))
            mask[bar, :] = True
            stems = rng.choice(width, size=int(rng.integers(1, min(3, width) + 1)), replace=False)
            mask[:, stems] = True
            if width >= 5 and rng.random() < .45:
                mask[bar, int(rng.integers(1, width - 1))] = False
        elif style == 1:  # hollow or broken frame
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            if width >= 4 and height >= 3 and rng.random() < .55:
                row = int(rng.integers(1, height - 1))
                mask[row, 0] = False
                mask[row, int(rng.integers(1, width - 1))] = True
        elif style == 2:  # stepped short bars
            span = int(rng.integers(1, max(2, width // 2 + 1)))
            for row in range(height):
                start = int(round((width - span) * row / max(1, height - 1)))
                start = min(width - span, max(0, start + int(rng.integers(-1, 2))))
                mask[row, start:start + span] = True
        elif style == 3:  # branching stem
            center = int(rng.integers(0, width))
            mask[:, center] = True
            for row in rng.choice(height, size=int(rng.integers(1, height + 1)), replace=False):
                if rng.random() < .5:
                    mask[int(row), :center + 1] = True
                else:
                    mask[int(row), center:] = True
        elif style == 4:  # separated horizontal strokes
            for row in range(height):
                start = int(rng.integers(0, width))
                length = int(rng.integers(1, min(width - start, max(2, width // 2)) + 1))
                mask[row, start:start + length] = True
                if width >= 5 and rng.random() < .5:
                    mask[row, int(rng.integers(0, width))] = True
        else:
            mask = rng.random((height, width)) < rng.uniform(.26, .7)

        for row in range(height):
            if not np.any(mask[row]):
                mask[row, int(rng.integers(0, width))] = True
        if symmetric:
            for row in range(height // 2):
                mask[height - 1 - row] = mask[row]
        mask[int(rng.integers(0, height)), 0] = True
        mask[int(rng.integers(0, height)), -1] = True
        if symmetric:
            for row in range(height // 2):
                mask[height - 1 - row] = mask[row]
        if np.array_equal(mask, mask[::-1]) == symmetric:
            return mask

    # Bounded fallback with guaranteed asymmetry and tight bounds.
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[1:, width // 2] = True
    if symmetric:
        mask[-1, :] = True
    elif np.array_equal(mask, mask[::-1]):
        mask[-1, 0] = False
    return mask


def sample_puzzle_parameters(rng):
    return {
        "background": int(rng.integers(0, 10)),
        "vary_background": bool(rng.random() < .5),
        "allow_side": bool(rng.random() < .55),
        "allow_repeat": bool(rng.random() < .5),
    }


def _pair_background(rng, parameters, section, index):
    base = parameters["background"]
    if not parameters["vary_background"] or (section == "train" and index == 0):
        return base
    if section == "train" and index == 1:
        return (base + int(rng.integers(1, 10))) % 10
    return int(rng.integers(0, 10))


def _generate_pair(rng, parameters, section, index):
    # The first training pair proves both change and invariance. Layout
    # extensions are demonstrated there before they may appear in tests.
    first = section == "train" and index == 0
    weights = (.46, .38, .16) if section == "train" else (.25, .48, .27)
    tier = int(rng.choice(3, p=weights))
    max_height, max_width = ((15, 15), (22, 23), (30, 30))[tier]
    max_motif_height, max_motif_width = ((3, 7), (5, 11), (6, 15))[tier]
    count = int(rng.integers(3 if first or tier == 2 else 2, 6))
    side = parameters["allow_side"] and (first or rng.random() < .5)
    repeat = parameters["allow_repeat"] and (first or rng.random() < .42)
    if side and repeat:
        count = max(count, 3)
    background = _pair_background(rng, parameters, section, index)
    bordered = bool(rng.random() < .48)

    for attempt in range(60):
        if attempt == 20:
            bordered = False
            count = min(count, 4)
        if attempt == 40:
            count = 3
        heights = []
        for motif in range(count):
            if first and motif == 0:
                height = 1 if rng.random() < .35 else int(rng.integers(2, max_motif_height + 1))
            elif first and motif in (1, 2):
                height = int(rng.integers(2, max_motif_height + 1))
            elif tier == 2 and motif < 2:
                height = int(rng.integers(3, max_motif_height + 1))
            else:
                height = int(rng.integers(1, max_motif_height + 1))
            heights.append(height)
        widths = [int(rng.integers(2, max_motif_width + 1)) for _ in range(count)]
        if tier == 2:
            widths[:2] = [max(6, width) for width in widths[:2]]
        symmetric = []
        for motif, height in enumerate(heights):
            if first and motif == 0:
                symmetric.append(True)
            elif first and motif in (1, 2):
                symmetric.append(False)
            else:
                symmetric.append(height == 1 or bool(rng.random() < .23))
        if section == "test":
            needed = min(2, count) - sum(not value for value in symmetric)
            for motif in rng.permutation(count):
                if needed <= 0:
                    break
                if symmetric[int(motif)]:
                    symmetric[int(motif)] = False
                    heights[int(motif)] = max(2, heights[int(motif)])
                    needed -= 1
        if section == "train" and not first and all(symmetric):
            symmetric[0] = False
            heights[0] = max(2, heights[0])

        groups = [[i] for i in range(count)]
        if side:
            groups[0] = [0, 1]
            del groups[1]
        order = rng.permutation(len(groups))
        groups = [groups[int(i)] for i in order]
        gaps = [int(rng.integers(1 if repeat else 0, 4)) for _ in range(len(groups) - 1)]
        content_height = sum(max(heights[i] for i in group) for group in groups) + sum(gaps)
        side_width = widths[0] + widths[1] + int(rng.integers(1, 4)) if side else 0
        needed_width = max(max(widths), side_width)
        minimum_height = (7, 12, 18)[tier]
        minimum_width = (7, 11, 17)[tier]
        if (content_height > max_height or needed_width > max_width
                or (tier == 2 and content_height < 10)):
            continue
        low_h = max(minimum_height, content_height + 2 * bordered)
        low_w = max(minimum_width, needed_width + 2 * bordered)
        if low_h > max_height or low_w > max_width:
            continue
        height = int(rng.integers(low_h, min(max_height, low_h + 5) + 1))
        width = int(rng.integers(low_w, min(max_width, low_w + 5) + 1))
        patterns = [_pattern(rng, h, w, s) for h, w, s in zip(heights, widths, symmetric)]
        colors = [int(c) for c in rng.choice([c for c in range(10) if c != background], size=count, replace=False)]
        if repeat:
            positions = {i: group_index for group_index, group in enumerate(groups) for i in group}
            candidates = [(i, j) for i in range(count) for j in range(i + 1, count)
                          if positions[i] != positions[j]]
            if not candidates:
                continue
            i, j = candidates[int(rng.integers(0, len(candidates)))]
            colors[j] = colors[i]
        grid = np.full((height, width), background, dtype=np.int8)
        vertical_slack = height - content_height
        margin = int(bordered)
        row = int(rng.integers(margin, vertical_slack - margin + 1))
        for group_index, group in enumerate(groups):
            group_height = max(heights[i] for i in group)
            if len(group) == 2:
                i, j = group
                gap = side_width - widths[i] - widths[j]
                start = int(rng.integers(margin, width - side_width - margin + 1))
                columns = {i: start, j: start + widths[i] + gap}
            else:
                i = group[0]
                columns = {i: int(rng.integers(margin, width - widths[i] - margin + 1))}
            for i in group:
                local_row = row + int(rng.integers(0, group_height - heights[i] + 1))
                left = columns[i]
                block = grid[local_row:local_row + heights[i], left:left + widths[i]]
                block[patterns[i]] = colors[i]
            row += group_height
            if group_index < len(gaps):
                row += gaps[group_index]
        if _background(grid) != background:
            continue
        try:
            records = _motif_records(grid, background)
        except AssertionError:
            continue
        if len(records) != count:
            continue
        if first and (sum(not np.array_equal(r["mask"], r["mask"][::-1]) for r in records) < 2):
            continue
        result = _reflect_grid(grid, background)
        if np.array_equal(grid, result):
            continue
        return grid, result
    raise RuntimeError("bounded motif construction failed")


def _validate_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed_key, num_train, num_test):
    train = _validate_count("num_train", num_train, 2, 10)
    test = _validate_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed_key % 4
        train = test + 1 + (seed_key // 4 + seed_key % 7) % (10 - test)
    elif train is None:
        train = test + 1 + seed_key % (10 - test)
    elif test is None:
        test = 1 + seed_key % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return int(train), int(test)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _resolve_counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63, dtype=np.uint64)))
            for _ in range(10):
                input_grid, output_grid = _generate_pair(pair_rng, parameters, section, index)
                if _grid_key(input_grid) not in seen:
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
            seen.add(_grid_key(input_grid))
            puzzle[section].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_has_invariant = False
    train_has_two_changes = False
    train_local_not_global = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert grid.dtype.kind in "iu" and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert source.shape == target.shape and source.dtype == target.dtype
            assert _grid_key(source) not in seen_inputs and _pair_key(pair) not in seen_pairs
            seen_inputs.add(_grid_key(source))
            seen_pairs.add(_pair_key(pair))
            background = _background(source)
            assert _background(target) == background
            input_records = _motif_records(source, background)
            output_records = _motif_records(target, background)
            assert len(input_records) >= 2 and len(input_records) <= 5
            assert [(r["color"], r["bbox"]) for r in input_records] == [
                (r["color"], r["bbox"]) for r in output_records]
            assert np.array_equal(target, _reflect_grid(source, background))
            assert not np.array_equal(source, target)
            changed = sum(not np.array_equal(r["mask"], r["mask"][::-1]) for r in input_records)
            assert changed >= 1
            if section == "train":
                train_has_two_changes |= changed >= 2
                train_has_invariant |= changed < len(input_records)
                train_local_not_global |= not np.array_equal(target, source[::-1])
            for old_input, old_output in _source_arrays():
                assert not (np.array_equal(source, old_input) and np.array_equal(target, old_output))
    assert train_has_two_changes and train_has_invariant and train_local_not_global
    first = puzzle["train"][0]
    first_records = _motif_records(first["input"], _background(first["input"]))
    assert len(first_records) >= 3
    assert sum(not np.array_equal(r["mask"], r["mask"][::-1]) for r in first_records) >= 2
    assert any(np.array_equal(r["mask"], r["mask"][::-1]) for r in first_records)


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    cases = [generate_puzzle(seed) for seed in range(42)]
    cases.extend(generate_puzzle(seed, train, test) for seed, train, test in (
        (51, 2, 1), (52, 10, 1), (53, 10, 4),
        (54, None, 1), (55, None, 4), (56, 2, None), (57, 10, None)))
    for puzzle in cases:
        validate_generated_puzzle(puzzle)
        # An independent row-slice prediction, based only on input and the
        # reflection seen in training; no test label is used to predict.
        for pair in puzzle["train"] + puzzle["test"]:
            inp = pair["input"]
            bg = _background(inp)
            predicted = np.full_like(inp, bg)
            for color in set(int(x) for x in inp.ravel()) - {bg}:
                occupied = np.flatnonzero(np.any(inp == color, axis=1))
                runs = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
                for run in runs:
                    top, bottom = int(run[0]), int(run[-1])
                    predicted[top:bottom + 1][(inp[top:bottom + 1] == color)[::-1]] = color
            assert np.array_equal(predicted, pair["output"])
    assert _puzzles_equal(generate_puzzle(314159, 8, 3), generate_puzzle(314159, 8, 3))
    assert _puzzles_equal(generate_puzzle(2718), generate_puzzle(2718))
    assert not _puzzles_equal(generate_puzzle(111), generate_puzzle(112))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(6000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert len({len(p["train"]) for p in cases[:42]}) >= 4
    assert len({len(p["test"]) for p in cases[:42]}) >= 3
    assert len({tuple(pair["input"].shape for s in ("train", "test") for pair in p[s]) for p in cases[:42]}) >= 30
    assert len({tuple((pair["input"].shape, _grid_key(pair["input"])) for s in ("train", "test") for pair in p[s]) for p in cases[:42]}) == 42

    measured = {name: 0 for name in (
        "all_change_test", "multirow_symmetric_train", "multirow_symmetric_test",
        "complex_train", "complex_test", "side_train", "side_test",
        "repeat_train", "repeat_test", "edge_train", "edge_test",
        "bordered_train", "bordered_test", "new_test_color",
        "changing_background", "varying_motif_count", "smaller_test",
        "larger_test", "bar", "sparse", "dense", "side_repeat_train",
    )}
    geometries = set()
    for puzzle in cases[:42]:
        train_backgrounds = set()
        train_colors = set()
        motif_counts = set()
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                background = _background(grid)
                records = _motif_records(grid, background)
                motif_counts.add(len(records))
                if section == "train":
                    train_backgrounds.add(background)
                    train_colors.update(record["color"] for record in records)
                elif any(record["color"] not in train_colors for record in records):
                    measured["new_test_color"] += 1
                flags = [np.array_equal(r["mask"], r["mask"][::-1]) for r in records]
                if section == "test" and not any(flags):
                    measured["all_change_test"] += 1
                if any(flag and r["mask"].shape[0] >= 2 for flag, r in zip(flags, records)):
                    measured[f"multirow_symmetric_{section}"] += 1
                if any(r["mask"].shape[0] >= 4 and r["mask"].shape[1] >= 9
                       and np.unique(r["mask"], axis=0).shape[0] >= 3 for r in records):
                    measured[f"complex_{section}"] += 1
                side = any(max(a["bbox"][0], b["bbox"][0]) <= min(a["bbox"][1], b["bbox"][1])
                           for index, a in enumerate(records) for b in records[index + 1:])
                repeated = len({r["color"] for r in records}) < len(records)
                if side:
                    measured[f"side_{section}"] += 1
                if repeated:
                    measured[f"repeat_{section}"] += 1
                if section == "train" and side and repeated:
                    measured["side_repeat_train"] += 1
                edge = any(r["bbox"][0] == 0 or r["bbox"][1] == grid.shape[0] - 1
                           or r["bbox"][2] == 0 or r["bbox"][3] == grid.shape[1] - 1
                           for r in records)
                measured[f"{'edge' if edge else 'bordered'}_{section}"] += 1
                if any(r["mask"].shape[1] >= 4 and np.any(np.all(r["mask"], axis=1)) for r in records):
                    measured["bar"] += 1
                if any(r["mask"].size >= 8 and np.mean(r["mask"]) < .4 for r in records):
                    measured["sparse"] += 1
                if any(r["mask"].size >= 8 and np.mean(r["mask"]) > .7 for r in records):
                    measured["dense"] += 1
                for record in records:
                    geometries.add((record["mask"].shape, record["mask"].tobytes()))
                if section == "test":
                    if grid.shape[0] <= max(s[0] for s in train_shapes) and grid.shape[1] <= max(s[1] for s in train_shapes):
                        measured["smaller_test"] += 1
                    if grid.shape[0] > max(s[0] for s in train_shapes) and grid.shape[1] > max(s[1] for s in train_shapes):
                        measured["larger_test"] += 1
        measured["changing_background"] += len(train_backgrounds) > 1
        measured["varying_motif_count"] += len(motif_counts) > 1
    assert len(geometries) >= 300
    assert all(value >= 5 for value in measured.values())
