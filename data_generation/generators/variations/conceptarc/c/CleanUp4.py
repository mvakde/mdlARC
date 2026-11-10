"""Synthetic instances of the hollow-rectangle cleanup task."""

import numpy as np


# Output fingerprints from the six source pairs.  The third frame in the
# first pair ends at column 11, as the source grid shows.
_SOURCE_OUTPUTS = {
    ((16, 16), 0, ((0, 9, 5, 14, 3), (2, 2, 5, 5, 3),
                   (8, 5, 13, 11, 3), (10, 0, 12, 2, 3))),
    ((15, 11), 0, ((1, 6, 5, 10, 8), (2, 1, 6, 4, 2),
                   (7, 6, 14, 10, 4), (9, 2, 13, 5, 6))),
    ((6, 6), 3, ((1, 2, 4, 5, 2),)),
    ((15, 15), 0, ((2, 1, 6, 4, 5), (2, 8, 9, 14, 5),
                   (8, 3, 12, 5, 5))),
    ((18, 20), 0, ((0, 10, 6, 13, 3), (2, 1, 6, 4, 4),
                   (9, 4, 14, 18, 6), (11, 0, 17, 2, 9))),
    ((10, 10), 2, ((1, 2, 5, 8, 4), (7, 2, 9, 8, 3))),
}


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(name)
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("incompatible pair counts")
    return choices[int(rng.integers(len(choices)))]


def _partition(rng, height, width, count):
    """Random recursive partitions leave at least a one-cell gap."""
    regions = [(0, 0, height, width)]
    while len(regions) < count:
        eligible = []
        for i, (r0, c0, r1, c1) in enumerate(regions):
            if r1 - r0 >= 7:
                eligible.append((i, 0))
            if c1 - c0 >= 7:
                eligible.append((i, 1))
        if not eligible:
            return None
        i, axis = eligible[int(rng.integers(len(eligible)))]
        r0, c0, r1, c1 = regions.pop(i)
        if axis == 0:
            minimum = 4 if r1 - r0 >= 9 else 3
            cut = int(rng.integers(r0 + minimum, r1 - minimum))
            regions.extend(((r0, c0, cut, c1),
                            (cut + 1, c0, r1, c1)))
        else:
            minimum = 4 if c1 - c0 >= 9 else 3
            cut = int(rng.integers(c0 + minimum, c1 - minimum))
            regions.extend(((r0, c0, r1, cut),
                            (r0, cut + 1, r1, c1)))
    return regions


def _dimension(rng, available):
    if available <= 3:
        return 3
    if rng.random() < 0.34:
        return available
    if rng.random() < 0.16:
        return 3
    return int(rng.integers(3, available + 1))


def _layout(rng, regime, count):
    for _ in range(80):
        if regime == "source_train":
            minimum = 7 if count == 1 else 10
            height = int(rng.integers(minimum, 17))
            width = int(rng.integers(minimum, 17))
        else:
            size_band = int(rng.choice(3, p=(0.28, 0.43, 0.29)))
            low, high = ((8, 13), (12, 21), (18, 31))[size_band]
            minimum = max(low, 10 if count >= 3 else 8)
            height = int(rng.integers(minimum, high))
            width = int(rng.integers(minimum, high))
        if rng.random() < 0.45:
            margins = (1, 1, 1, 1)
        else:
            margins = tuple(int(rng.integers(0, 3)) for _ in range(4))
        top, left, bottom, right = margins
        if height - top - bottom < 3 or width - left - right < 3:
            continue
        regions = _partition(rng, height - top - bottom,
                             width - left - right, count)
        if regions is None:
            continue
        boxes = []
        for r0, c0, r1, c1 in regions:
            r0 += top
            r1 += top
            c0 += left
            c1 += left
            fh = _dimension(rng, r1 - r0)
            fw = _dimension(rng, c1 - c0)
            row = int(rng.integers(r0, r1 - fh + 1))
            col = int(rng.integers(c0, c1 - fw + 1))
            boxes.append((row, col, row + fh - 1, col + fw - 1))
        return height, width, boxes
    raise RuntimeError("could not partition grid")


def _palette(rng, background, count):
    available = [c for c in range(10) if c != background]
    rng.shuffle(available)
    distinct = int(rng.integers(1, min(count, 4) + 1))
    selected = available[:distinct]
    colors = selected + [selected[int(rng.integers(distinct))]
                         for _ in range(count - distinct)]
    rng.shuffle(colors)
    foreign_count = int(rng.choice((0, 1, 2), p=(0.18, 0.52, 0.30)))
    return colors, available[distinct:distinct + foreign_count]


def _make_output(height, width, background, boxes, colors):
    output = np.full((height, width), background, dtype=np.int8)
    for (r0, c0, r1, c1), color in zip(boxes, colors):
        output[r0, c0:c1 + 1] = color
        output[r1, c0:c1 + 1] = color
        output[r0:r1 + 1, c0] = color
        output[r0:r1 + 1, c1] = color
    return output


def _outline(box):
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)]


def _corrupt(rng, output, boxes, colors, background, foreign,
             density, damage_target, frame_noise, run_noise):
    inp = output.copy()
    all_colors = list(set(colors)) + foreign
    damaged = rng.permutation(len(boxes))[:damage_target]
    for i in damaged:
        box = boxes[int(i)]
        cells = _outline(box)
        r0, c0, r1, c1 = box
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if rng.random() < 0.36:
            locations = [list(corners)[int(rng.integers(4))]]
        else:
            noncorners = [cell for cell in cells if cell not in corners]
            first = noncorners[int(rng.integers(len(noncorners)))]
            locations = [first]
            if (len(cells) >= 16 and rng.random() < 0.30):
                adjacent = [p for p in noncorners
                            if abs(p[0] - first[0]) + abs(p[1] - first[1]) == 1]
                if adjacent:
                    locations.append(adjacent[int(rng.integers(len(adjacent)))])
        for row, col in locations:
            replacements = [background] + [c for c in all_colors
                                           if c != colors[int(i)]]
            inp[row, col] = int(rng.choice(replacements))

    open_cells = np.argwhere(output == background)
    minimum = len(foreign) + int(frame_noise) + (3 if run_noise else 0)
    target = max(minimum, 1, min(len(open_cells) // 4,
                                 int(round(output.size * density))))
    positions = set()
    run_pixels = []
    if run_noise and target >= 4:
        for _ in range(40):
            row, col = map(int, open_cells[int(rng.integers(len(open_cells)))])
            dr, dc = ((0, 1), (1, 0))[int(rng.integers(2))]
            length = min(target - len(foreign), int(rng.integers(3, 7)))
            segment = [(row + dr * j, col + dc * j) for j in range(length)]
            if all(0 <= r < output.shape[0] and 0 <= c < output.shape[1]
                   and output[r, c] == background for r, c in segment):
                positions.update(segment)
                run_pixels = segment
                break
    if frame_noise and target > len(positions):
        for _ in range(20):
            p = tuple(map(int, open_cells[int(rng.integers(len(open_cells)))]))
            if p not in positions:
                positions.add(p)
                break
    while len(positions) < target:
        if rng.random() < 0.42:
            box = boxes[int(rng.integers(len(boxes)))]
            r0, c0, r1, c1 = box
            row = int(rng.integers(max(0, r0 - 2), min(output.shape[0], r1 + 3)))
            col = int(rng.integers(max(0, c0 - 2), min(output.shape[1], c1 + 3)))
            if output[row, col] != background:
                continue
        else:
            row, col = map(int, open_cells[int(rng.integers(len(open_cells)))])
        positions.add((row, col))

    frame_palette = list(set(colors))
    foreign_palette = foreign
    for i, (row, col) in enumerate(sorted(positions)):
        if frame_noise and (i == 0 or rng.random() < 0.48):
            inp[row, col] = int(rng.choice(frame_palette))
        elif foreign_palette:
            inp[row, col] = int(rng.choice(foreign_palette))
        else:
            inp[row, col] = int(rng.choice(frame_palette))
    if foreign_palette:
        free = sorted(positions - set(run_pixels))
        if len(free) < len(foreign_palette):
            free = sorted(positions)
        for i, color in enumerate(foreign_palette):
            inp[free[-1 - i]] = color
    if run_pixels:
        run_color = int(rng.choice(frame_palette if frame_noise or not foreign
                                   else foreign_palette))
        for row, col in run_pixels:
            inp[row, col] = run_color
    return inp


def _visible_candidates(inp, background, raw=False):
    """Enumerate strongly supported outlines using input evidence only."""
    height, width = inp.shape
    found = set()
    for color in (int(c) for c in np.unique(inp) if c != background):
        mask = inp == color
        rows = [set(map(int, np.flatnonzero(mask[r]))) for r in range(height)]
        row_prefix = np.pad(np.cumsum(mask, axis=1), ((0, 0), (1, 0)))
        col_prefix = np.pad(np.cumsum(mask, axis=0), ((1, 0), (0, 0)))
        for r0 in range(height - 2):
            top = rows[r0]
            if len(top) < 2:
                continue
            for r1 in range(r0 + 2, height):
                bottom = rows[r1]
                if len(bottom) < 2:
                    continue
                candidates = sorted(top | bottom)
                for j, c0 in enumerate(candidates[:-1]):
                    for c1 in candidates[j + 1:]:
                        if c1 - c0 < 2:
                            continue
                        corners = ((c0 in top) + (c1 in top)
                                   + (c0 in bottom) + (c1 in bottom))
                        if corners < 3:
                            continue
                        a = int(row_prefix[r0, c1 + 1] - row_prefix[r0, c0])
                        b = int(row_prefix[r1, c1 + 1] - row_prefix[r1, c0])
                        l = int(col_prefix[r1 + 1, c0] - col_prefix[r0, c0])
                        rr = int(col_prefix[r1 + 1, c1] - col_prefix[r0, c1])
                        side_counts = (a, b, l, rr)
                        lengths = (c1 - c0 + 1, c1 - c0 + 1,
                                   r1 - r0 + 1, r1 - r0 + 1)
                        if any(3 * n < 2 * length for n, length
                               in zip(side_counts, lengths)):
                            continue
                        if (10 * max(a, b) < 9 * lengths[0]
                                or 10 * max(l, rr) < 9 * lengths[2]):
                            continue
                        perimeter = 2 * (lengths[0] + lengths[2]) - 4
                        total = sum(side_counts) - corners
                        if 4 * total < 3 * perimeter:
                            continue
                        found.add((r0, c0, r1, c1, color))
    if raw:
        return found
    # A narrow true frame contains many shorter apparent outlines: their
    # end rows contain its two side pixels.  Prefer the complete, best
    # supported outline and suppress intersecting alternatives.
    def evidence(frame):
        r0, c0, r1, c1, color = frame
        cells = _outline(frame[:4])
        visible = sum(inp[r, c] == color for r, c in cells)
        return (4 * visible - 3 * len(cells), visible, len(cells))

    chosen = set()
    occupied = set()
    for frame in sorted(found, key=evidence, reverse=True):
        cells = set(_outline(frame[:4]))
        if cells.isdisjoint(occupied):
            chosen.add(frame)
            occupied.update(cells)
    return chosen


def _decode_output(output):
    background = int(np.bincount(output.ravel(), minlength=10).argmax())
    height, width = output.shape
    seen = np.zeros(output.shape, dtype=bool)
    frames = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or output[row, col] == background:
                continue
            color = int(output[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells = set()
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c),
                               (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr, nc] and output[nr, nc] == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0 = min(r for r, c in cells)
            r1 = max(r for r, c in cells)
            c0 = min(c for r, c in cells)
            c1 = max(c for r, c in cells)
            assert r1 - r0 >= 2 and c1 - c0 >= 2
            assert cells == set(_outline((r0, c0, r1, c1)))
            frames.append((r0, c0, r1, c1, color))
    assert frames
    return background, tuple(sorted(frames))


def _solve_from_input(inp):
    background = int(np.bincount(inp.ravel(), minlength=10).argmax())
    frames = _visible_candidates(inp, background)
    prediction = np.full(inp.shape, background, dtype=np.int8)
    for r0, c0, r1, c1, color in frames:
        prediction[r0, c0:c1 + 1] = color
        prediction[r1, c0:c1 + 1] = color
        prediction[r0:r1 + 1, c0] = color
        prediction[r0:r1 + 1, c1] = color
    return prediction


def _pair_style(rng, regime, index, count, needs):
    density_band = int(rng.choice(3, p=(0.30, 0.39, 0.31)))
    ranges = ((0.012, 0.039), (0.040, 0.090), (0.091, 0.16))
    low, high = ranges[density_band]
    density = float(rng.uniform(low, high))
    damage = int(rng.choice((0, 1, 2), p=(0.57, 0.30, 0.13)))
    damage = min(damage, count)
    frame_noise = bool(rng.integers(2))
    run_noise = bool(rng.random() < (0.35 if regime == "source_train" else 0.50))
    if regime == "source_train" and index == 0:
        if needs[0]:
            frame_noise = True
        if needs[2]:
            damage = max(1, damage)
        if needs[1]:
            run_noise = True
    return density, damage, frame_noise, run_noise


def _generate_pair(rng, regime, count, style, corner_required=False,
                   run_required=False):
    density, damage, frame_noise, run_noise = style
    for _ in range(150):
        height, width, boxes = _layout(rng, regime, count)
        background = int(rng.integers(10))
        colors, foreign = _palette(rng, background, count)
        output = _make_output(height, width, background, boxes, colors)
        if int(np.bincount(output.ravel(), minlength=10).argmax()) != background:
            continue
        inp = _corrupt(rng, output, boxes, colors, background, foreign,
                       density, damage, frame_noise, run_noise)
        frames = {(r0, c0, r1, c1, color)
                  for (r0, c0, r1, c1), color in zip(boxes, colors)}
        if corner_required:
            if not any(inp[r, c] != color for r0, c0, r1, c1, color in frames
                       for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
                continue
        if run_required and not _features({"input": inp, "output": output})["run"]:
            continue
        if (int(np.bincount(inp.ravel(), minlength=10).argmax())
                != background):
            continue
        if _visible_candidates(inp, background) != frames:
            continue
        if (output.shape, background, tuple(sorted(frames))) in _SOURCE_OUTPUTS:
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("could not sample a recoverable pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    # Geometry is the sole puzzle-wide convention.  Other properties can vary
    # between pairs, as in the original training and test examples.
    test_specs = []
    for _ in range(test_count):
        count = int(rng.integers(2, 5))
        style = _pair_style(rng, "target_test", 0, count, (False,) * 3)
        corner = bool(style[1] and rng.random() < 0.30)
        test_specs.append((count, style, corner))
    result = {"train": [], "test": []}
    seen = set()
    for count, style, corner in test_specs:
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            try:
                pair = _generate_pair(pair_rng, "target_test", count, style,
                                      corner)
            except RuntimeError:
                continue
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in seen:
                break
        else:
            raise RuntimeError("test pair resampling failed")
        result["test"].append(pair)
        seen.add(signature)
    test_features = [_features(pair) for pair in result["test"]]
    needs = (any(f["frame_stray"] for f in test_features),
             any(f["run"] for f in test_features),
             any(f["corner"] for f in test_features))
    train_specs = []
    for index in range(train_count):
        count = int(rng.integers(1, 5))
        if index == 0 and needs[2]:
            count = max(count, 2)
        style = _pair_style(rng, "source_train", index, count, needs)
        train_specs.append((count, style, needs[2] and index == 0))
    if all(spec[0] == 1 for spec in train_specs):
        count, style, corner = train_specs[-1]
        train_specs[-1] = (2, style, corner)

    for index, (count, style, corner) in enumerate(train_specs):
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            try:
                pair = _generate_pair(pair_rng, "source_train", count, style,
                                      corner, needs[1] and index == 0)
            except RuntimeError:
                continue
            signature = (pair["input"].shape, pair["input"].tobytes())
            if signature not in seen:
                break
        else:
            raise RuntimeError("duplicate resampling failed")
        seen.add(signature)
        result["train"].append(pair)
    return result


def _features(pair):
    inp, output = pair["input"], pair["output"]
    background, frames = _decode_output(output)
    outline = output != background
    off = ~outline
    frame_colors = {f[4] for f in frames}
    corner_damage = sum(inp[r, c] != color for r0, c0, r1, c1, color in frames
                        for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)))
    runs = 0
    stray = off & (inp != background)
    for axis in (0, 1):
        a = stray if axis == 1 else stray.T
        colors = inp if axis == 1 else inp.T
        runs += sum(bool(np.all(line[j:j + 3])
                         and len(set(map(int, colors[i, j:j + 3]))) == 1)
                    for i, line in enumerate(a)
                    for j in range(len(line) - 2))
    return {"background": background, "frames": frames,
            "damage": int(np.count_nonzero(inp[outline] != output[outline])),
            "corner": int(corner_damage), "stray": int(np.count_nonzero(stray)),
            "frame_stray": bool(np.any(np.isin(inp[off], list(frame_colors)))),
            "foreign": len(set(map(int, np.unique(inp[off])))
                           - {background} - frame_colors),
            "run": runs > 0,
            "mixed": len(frame_colors) > 1}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = pair["input"], pair["output"]
            for grid in (inp, output):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == output.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            f = _features(pair)
            background, frames = f["background"], f["frames"]
            assert int(np.bincount(inp.ravel(), minlength=10).argmax()) == background
            assert _visible_candidates(inp, background) == set(frames)
            assert np.array_equal(_solve_from_input(inp), output)
            assert (output.shape, background, frames) not in _SOURCE_OUTPUTS
            assert f["stray"] >= 1
            for r0, c0, r1, c1, color in frames:
                assert r1 - r0 >= 2 and c1 - c0 >= 2
                assert np.count_nonzero(inp[r0, c0:c1 + 1] == color) >= 2
            if split == "train":
                assert max(inp.shape) <= 16
                assert 1 <= len(frames) <= 4
            else:
                assert 2 <= len(frames) <= 4
            features[split].append(f)
    assert any(len(f["frames"]) >= 2 for f in features["train"])
    for property_name in ("frame_stray", "corner", "run"):
        if any(f[property_name] for f in features["test"]):
            assert any(f[property_name] for f in features["train"])


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(pa[k], pb[k]) for pa, pb in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, grids, layouts, styles = (set() for _ in range(5))
    puzzles = set()
    all_features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                all_features.append(f)
                styles.add((f["corner"] > 0, f["damage"] > 0,
                            f["frame_stray"], f["mixed"], f["foreign"], f["run"]))
                layouts.add((pair["output"].shape,
                             tuple(frame[:4] for frame in f["frames"])))
                for key in ("input", "output"):
                    grid = pair[key]
                    signature = (grid.shape, grid.tobytes())
                    grids.add(signature)
                    signatures.append(signature)
        puzzles.add(tuple(signatures))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(grids) >= 250 and len(layouts) >= 100
    assert len(styles) >= 15 and len(puzzles) >= 38
    assert any(f["corner"] for f in all_features)
    assert any(f["damage"] == 0 for f in all_features)
    assert any(f["foreign"] >= 2 for f in all_features)
    assert any(f["run"] for f in all_features)
    assert any(f["stray"] >= 30 for f in all_features)
    assert any(any(r1 - r0 == 2 or c1 - c0 == 2
                   for r0, c0, r1, c1, color in f["frames"])
               for f in all_features)
    for num_train, num_test in ((2, 1), (10, 1), (10, 4),
                                (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(907, num_train, num_test)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(907, num_train, num_test))
        if num_train is not None:
            assert len(puzzle["train"]) == num_train
        if num_test is not None:
            assert len(puzzle["test"]) == num_test
    for num_train, num_test in ((1, None), (11, None), (None, 0),
                                (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, num_train, num_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(1234, 2, 1)
    f = _features(broken["test"][0])
    r0, c0, r1, c1, color = f["frames"][0]
    broken["test"][0]["output"] = broken["test"][0]["output"].copy()
    broken["test"][0]["output"][r0, c0] = f["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken outline accepted")
