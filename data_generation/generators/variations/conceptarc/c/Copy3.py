import hashlib

import numpy as np


_SOURCE_PAIRS = {
    "e1446a580e8a173db5f906efb096a62976d51617f8d34110bf8e9c7b30c646bc",
    "46283e3673c71a0116dbc305ba9b316dc03e26c37360042d816a7ed64eb402d8",
    "9249493a2de7f0a87ff834084097d0b64f8607df0debee9ac478a35a0d3f1fbf",
    "d02e5322a478b850bc9114197e2f423c90a576ec3de27f79dfae28223f8091d1",
    "8f6fdde7cc8175a4cb4c7e06bc7b7d508d989333b4497fc10102393bf88b9211",
}


def _pair_counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError("Illegal pair count")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("The train count must exceed the test count")
    return num_train, num_test


def _make_pattern(rng, height, width, palette, style):
    p = np.full((height, width), -1, dtype=np.int8)
    k = len(palette)

    if style == "bands":
        rows = np.sort(rng.choice(height, size=min(height, int(rng.integers(4, 8))), replace=False))
        for i, row in enumerate(rows):
            length = int(rng.integers(2, width + 1))
            left = int(rng.integers(0, width - length + 1))
            p[row, left:left + length] = palette[i % k]
            if rng.random() < 0.35:
                p[row, left + int(rng.integers(length))] = palette[(i + 1) % k]
        for _ in range(int(rng.integers(0, 3))):
            x = int(rng.integers(width))
            a = int(rng.integers(height))
            b = min(height, a + int(rng.integers(2, 5)))
            p[a:b, x] = palette[int(rng.integers(k))]

    elif style == "stripes":
        bh = int(rng.integers(3, min(8, height) + 1))
        bw = int(rng.integers(3, width + 1))
        y = int(rng.integers(height - bh + 1))
        x = int(rng.integers(width - bw + 1))
        for i in range(bh):
            p[y + i, x:x + bw] = palette[i % k]
        for _ in range(int(rng.integers(0, 4))):
            row = int(rng.integers(height))
            left = int(rng.integers(width))
            length = min(width - left, int(rng.integers(1, width + 1)))
            p[row, left:left + length] = palette[int(rng.integers(k))]

    elif style == "frame":
        fh = int(rng.integers(4, min(height, 8) + 1))
        fw = int(rng.integers(4, min(width, 8) + 1))
        y = int(rng.integers(height - fh + 1))
        x = int(rng.integers(width - fw + 1))
        color = palette[int(rng.integers(k))]
        p[y, x:x + fw] = color
        p[y + fh - 1, x:x + fw] = color
        p[y:y + fh, x] = color
        p[y:y + fh, x + fw - 1] = color
        for _ in range(int(rng.integers(1, 4))):
            p[y + int(rng.integers(1, fh - 1)), x + int(rng.integers(1, fw - 1))] = palette[int(rng.integers(k))]
        for i in range(int(rng.integers(2, 6))):
            row = int(rng.integers(height))
            left = int(rng.integers(width))
            length = min(width - left, int(rng.integers(1, width + 1)))
            p[row, left:left + length] = palette[(i + 1) % k]

    else:  # Multiple independent bars, blocks, corners, and isolated marks.
        for i in range(int(rng.integers(5, 10))):
            y = int(rng.integers(height))
            x = int(rng.integers(width))
            color = palette[i % k]
            shape = int(rng.integers(4))
            h = min(height - y, int(rng.integers(1, 5)))
            w = min(width - x, int(rng.integers(1, 6)))
            if shape == 0:
                p[y:y + h, x:x + w] = color
            elif shape == 1:
                p[y, x:x + w] = color
            elif shape == 2:
                p[y:y + h, x] = color
            else:
                p[y, x:x + w] = color
                p[y:y + h, x] = color

    # A missing color is inserted into existing geometry, not as a legend.
    occupied = np.argwhere(p != -1)
    missing = [color for color in palette if color not in p]
    if len(occupied) < len(missing):
        return p
    for color, index in zip(missing, rng.choice(len(occupied), size=len(missing), replace=False)):
        y, x = occupied[index]
        p[y, x] = color
    occupied = np.argwhere(p != -1)
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0) + 1
    return p[y0:y1, x0:x1]


def _reconstructions(grid):
    """Enumerate visible-clue-consistent translated copies from either side."""
    height, width = grid.shape
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    results = {}
    for side in (1, -1):
        view = grid if side == 1 else grid[:, ::-1]
        foreground = view != bg
        for cut in range(1, width):
            original = np.argwhere(foreground[:, :cut])
            clues = np.argwhere(foreground[:, cut:])
            if len(original) < 8 or not 2 <= len(clues) < len(original):
                continue
            clues[:, 1] += cut
            if int(clues[:, 1].min()) - int(original[:, 1].max()) < 2:
                continue
            colors = view[original[:, 0], original[:, 1]]
            clue_colors = view[clues[:, 0], clues[:, 1]]
            anchor = clues[0]
            for pos in original[colors == clue_colors[0]]:
                dy, dx = (anchor - pos).astype(int)
                shifted = original + (dy, dx)
                if (dx <= 0 or shifted[:, 0].min() < 0
                        or shifted[:, 0].max() >= height
                        or shifted[:, 1].min() <= original[:, 1].max() + 1
                        or shifted[:, 1].max() >= width):
                    continue
                source_y = clues[:, 0] - dy
                source_x = clues[:, 1] - dx
                if (np.any(source_y < 0) or np.any(source_y >= height)
                        or np.any(source_x < 0) or np.any(source_x >= cut)):
                    continue
                if not np.array_equal(view[source_y, source_x], clue_colors):
                    continue
                candidate = np.full_like(view, bg)
                candidate[original[:, 0], original[:, 1]] = colors
                candidate[shifted[:, 0], shifted[:, 1]] = colors
                if not np.array_equal(candidate[foreground], view[foreground]):
                    continue
                if np.count_nonzero(candidate != bg) == np.count_nonzero(foreground):
                    continue
                output = candidate if side == 1 else candidate[:, ::-1].copy()
                key = output.tobytes()
                if key not in results:
                    results[key] = (output, {
                        "side": side, "dy": int(dy), "dx": int(dx) * side,
                        "original": original.copy(), "clues": clues.copy(),
                        "colors": colors.copy(), "bg": bg,
                        "cut": cut,
                    })
                    if len(results) > 1:
                        return results
    return results


def _clue_order(rng, pattern, profile):
    positions = np.argwhere(pattern != -1)
    colors = pattern[positions[:, 0], positions[:, 1]]
    n = len(positions)
    if profile == "mono":
        counts = [(np.count_nonzero(colors == color), color) for color in np.unique(colors)]
        color = max(counts)[1]
        first = np.flatnonzero(colors == color)
        rest = np.flatnonzero(colors != color)
        rng.shuffle(first)
        rng.shuffle(rest)
        return np.concatenate((first, rest))
    if profile == "cluster":
        center = positions[int(rng.integers(n))]
        distance = np.abs(positions - center).sum(axis=1)
        return np.lexsort((rng.random(n), distance))
    order = np.arange(n)
    rng.shuffle(order)
    return order


def _make_pair(rng, side, direction, bg, palette, style):
    for _ in range(120):
        size = int(rng.integers(3))
        if size == 0:
            ph, pw = int(rng.integers(5, 9)), int(rng.integers(4, 7))
        elif size == 1:
            ph, pw = int(rng.integers(7, 12)), int(rng.integers(4, 9))
        else:
            ph, pw = int(rng.integers(10, 16)), int(rng.integers(6, 10))
        pattern = _make_pattern(rng, ph, pw, palette, style)
        ph, pw = pattern.shape
        positions = np.argwhere(pattern != -1)
        n = len(positions)
        if n < 12:
            continue
        gap = int(rng.integers(2, 8))
        dx = pw + gap
        if 2 * pw + gap > 30:
            continue
        left_margin = int(rng.integers(0, min(4, 31 - 2 * pw - gap)))
        right_margin = int(rng.integers(0, min(4, 31 - 2 * pw - gap - left_margin)))
        width = left_margin + dx + pw + right_margin
        if direction == 0:
            dy = 0
        else:
            dy = direction * int(rng.integers(1, 5))
        top = int(rng.integers(0, 4))
        bottom = int(rng.integers(0, 4))
        y0 = top + max(0, -dy)
        height = y0 + ph + max(0, dy) + bottom
        if height > 30:
            continue
        original_y = positions[:, 0] + y0
        original_x = positions[:, 1] + left_margin + (dx if side == -1 else 0)
        copy_y = original_y + dy
        copy_x = original_x + side * dx
        colors = pattern[positions[:, 0], positions[:, 1]]
        output = np.full((height, width), bg, dtype=np.int8)
        output[original_y, original_x] = colors
        output[copy_y, copy_x] = colors

        profile = ("mono", "cluster", "scattered", "generous")[int(rng.integers(4))]
        order = _clue_order(rng, pattern, profile)
        if profile in ("mono", "cluster"):
            goal = min(n // 2, int(rng.integers(2, 6)))
        elif profile == "scattered":
            goal = min(n // 2, max(3, int(n * rng.uniform(0.08, 0.20))))
        else:
            goal = min(n // 2, max(4, int(n * rng.uniform(0.22, 0.42))))
        limit = min(n // 2, n - 3)
        for count in range(goal, limit + 1):
            chosen = order[:count]
            input_grid = np.full_like(output, bg)
            input_grid[original_y, original_x] = colors
            input_grid[copy_y[chosen], copy_x[chosen]] = colors[chosen]
            solutions = _reconstructions(input_grid)
            if len(solutions) == 1 and np.array_equal(next(iter(solutions.values()))[0], output):
                return input_grid, output
    raise RuntimeError("Could not construct an unambiguous translated copy")


def _signature(grid):
    return grid.shape, grid.tobytes()


def _source_digest(input_grid, output_grid):
    return hashlib.sha256(
        str(input_grid.shape).encode() + input_grid.astype(np.int8).tobytes()
        + output_grid.astype(np.int8).tobytes()
    ).hexdigest()


def _component_count(positions):
    remaining = set(map(tuple, positions))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _geometry_features(pattern, bg):
    occupied = pattern != bg
    dense_stripes = any(
        np.all(block != bg)
        and all(np.all(row == row[0]) for row in block)
        and len(np.unique(block)) > 1
        for y in range(pattern.shape[0] - 2)
        for x in range(pattern.shape[1] - 2)
        for block in (pattern[y:y + 3, x:x + 3],)
    )
    hollow = any(
        np.all(box[0]) and np.all(box[-1])
        and np.all(box[:, 0]) and np.all(box[:, -1])
        and not np.all(box[1:-1, 1:-1])
        for top in range(pattern.shape[0] - 3)
        for bottom in range(top + 4, pattern.shape[0] + 1)
        for left in range(pattern.shape[1] - 3)
        for right in range(left + 4, pattern.shape[1] + 1)
        for box in (occupied[top:bottom, left:right],)
    )
    return dense_stripes, hollow


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _pair_counts(rng, num_train, num_test)
    side = -1 if rng.random() < 0.2 else 1
    base_bg = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    alternate_bg = int(rng.choice([c for c in range(10) if c != base_bg]))
    backgrounds = [base_bg, alternate_bg] + [
        base_bg if rng.random() < 0.65 else int(rng.integers(10))
        for _ in range(num_train + num_test - 2)
    ]
    directions = [0, int(rng.choice((-1, 1)))]
    rng.shuffle(directions)
    directions.extend(int(rng.choice((-1, 0, 1))) for _ in range(num_train + num_test - 2))
    styles = ["bands", "stripes", "frame", "composite"]
    first_styles = rng.choice(styles, size=2, replace=False).tolist()
    styles_for_pairs = first_styles + [styles[int(rng.integers(4))]
                                        for _ in range(num_train + num_test - 2)]
    first_color_counts = rng.choice((2, 3, 4, 5), size=2, replace=False).tolist()
    color_counts = first_color_counts + [int(rng.integers(2, 6))
                                          for _ in range(num_train + num_test - 2)]

    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(num_train + num_test):
        section = "train" if index < num_train else "test"
        bg = backgrounds[index]
        for _ in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            palette = tuple(int(c) for c in pair_rng.choice(
                [c for c in range(10) if c != bg], size=color_counts[index], replace=False
            ))
            inp, out = _make_pair(pair_rng, side, directions[index], bg,
                                  palette, styles_for_pairs[index])
            key = _signature(inp)
            if key not in seen and _source_digest(inp, out) not in _SOURCE_PAIRS:
                seen.add(key)
                puzzle[section].append({"input": inp, "output": out})
                break
        else:
            raise RuntimeError("Could not create a distinct pair")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    sides = set()
    train_directions = set()
    train_backgrounds, train_color_counts = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            bg = int(np.bincount(inp.ravel(), minlength=10).argmax())
            assert np.array_equal(out[inp != bg], inp[inp != bg])
            assert np.count_nonzero(out != bg) > np.count_nonzero(inp != bg)
            input_key = _signature(inp)
            pair_key = (input_key, _signature(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _source_digest(inp, out) not in _SOURCE_PAIRS
            solutions = _reconstructions(inp)
            assert len(solutions) == 1
            predicted, witness = next(iter(solutions.values()))
            assert np.array_equal(predicted, out)
            assert 2 <= len(witness["clues"]) < len(witness["original"])
            assert len(np.unique(witness["colors"])) in (2, 3, 4, 5)
            assert len(witness["original"]) >= 12
            sides.add(witness["side"])
            if section == "train":
                train_directions.add(int(np.sign(witness["dy"])))
                train_backgrounds.add(bg)
                train_color_counts.add(len(np.unique(witness["colors"])))
    assert len(sides) == 1
    assert 0 in train_directions and (train_directions & {-1, 1})
    assert len(train_backgrounds) >= 2 and len(train_color_counts) >= 2


def _same_puzzle(a, b):
    return all(
        np.array_equal(x[name], y[name])
        for section in ("train", "test")
        for x, y in zip(a[section], b[section])
        for name in ("input", "output")
    ) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, patterns, palettes, signatures = set(), set(), set(), set()
    directions = {"train": set(), "test": set()}
    sparse = mono = missing_edge = 0
    backgrounds_within = 0
    missing_colors = disconnected = dense = hollow = 0
    up_one_test = down_four_test = right_puzzles = 0
    compact = {"train": 0, "test": 0}
    large = {"train": 0, "test": 0}
    widths = {"train": set(), "test": set()}
    color_counts, background_colors = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if seed < 6:
            assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        all_pairs = puzzle["train"] + puzzle["test"]
        signatures.add(tuple(_signature(p["input"]) for p in all_pairs))
        local_backgrounds = set()
        first = next(iter(_reconstructions(all_pairs[0]["input"]).values()))[1]
        right_puzzles += first["side"] == -1
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                shapes.add(inp.shape)
                widths[section].add(inp.shape[1])
                sol, witness = next(iter(_reconstructions(inp).values()))
                assert np.array_equal(sol, pair["output"])
                local_backgrounds.add(witness["bg"])
                background_colors.add(witness["bg"])
                directions[section].add(int(np.sign(witness["dy"])))
                original_colors = np.unique(witness["colors"])
                color_counts.add(len(original_colors))
                palettes.add(tuple(sorted(int(x) for x in original_colors)))
                orig = witness["original"]
                clues = witness["clues"]
                crop = orig - orig.min(axis=0)
                ph, pw = orig.max(axis=0) - orig.min(axis=0) + 1
                patterns.add(((int(ph), int(pw)), tuple(map(tuple, crop))))
                compact[section] += ph <= 7 and pw <= 6
                large[section] += ph >= 10 and pw >= 7
                view = inp if witness["side"] == 1 else inp[:, ::-1]
                clue_colors = np.unique(view[clues[:, 0], clues[:, 1]])
                sparse += len(clues) <= 5
                mono += len(clue_colors) == 1
                missing_colors += len(clue_colors) < len(original_colors)
                aligned = clues - (witness["dy"], abs(witness["dx"]))
                missing_edge += (not np.array_equal(aligned.min(axis=0), orig.min(axis=0))
                                 or not np.array_equal(aligned.max(axis=0), orig.max(axis=0)))
                disconnected += _component_count(orig) >= 3
                y0, x0 = orig.min(axis=0)
                pattern = view[y0:y0 + ph, x0:x0 + pw]
                has_dense, has_hollow = _geometry_features(pattern, witness["bg"])
                dense += has_dense
                hollow += has_hollow
                if section == "test":
                    up_one_test += witness["dy"] == -1
                    down_four_test += witness["dy"] == 4
        backgrounds_within += len(local_backgrounds) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(patterns) >= 100
    assert len(palettes) >= 20 and len(signatures) >= 38
    assert directions["train"] == directions["test"] == {-1, 0, 1}
    assert sparse >= 15 and mono >= 5 and missing_edge >= 15
    assert missing_colors >= 15 and disconnected >= 15
    assert dense >= 10 and hollow >= 10
    assert up_one_test >= 2 and down_four_test >= 2
    assert right_puzzles >= 2
    assert all(compact[s] >= 5 and large[s] >= 5 for s in ("train", "test"))
    assert len(widths["train"] & widths["test"]) >= 8
    assert color_counts == {2, 3, 4, 5} and len(background_colors) >= 5
    assert backgrounds_within >= 35
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        seed = 101 + (nt or 0) + (ns or 0)
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed, nt, ns))
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
