"""Congruent transparent-patch ARC puzzles."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = {
    "2bc46ea397281d37a6b7509238b289cfc43bc5dc5161184bb607158b48cef130",
    "f706b569741419afedd11e941d0030fcec9b35ac48c1c40f9f7078b840f51baf",
    "c0e14a20aeb123ac70517a092885f7fac9177efcf2f28fb1a63a95aa0122e1a8",
    "aef52cf485c7796d1a93fd0199d3f4f53947890aebf0269201140c58d0204e7b",
}


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _layout_shape(count, mode):
    if mode == "row":
        return 1, count
    if mode == "column":
        return count, 1
    return 2, 2


def _geometry(rng, count, canvas):
    mode = str(rng.choice(["row", "column", "scattered"], p=[0.28, 0.28, 0.44]))
    rows, cols = _layout_shape(count, mode)
    max_h, max_w = canvas if canvas is not None else (30, 30)
    choices = []
    weights = []
    for h in range(2, 10):
        for w in range(2, 10):
            if h * w < 2 * count or h * w > 72:
                continue
            if 2 + rows * h + rows - 1 > max_h or 2 + cols * w + cols - 1 > max_w:
                continue
            if canvas is not None and 2 * count * h * w >= max_h * max_w:
                continue
            choices.append((h, w))
            weights.append((h * w) ** -0.45)
    if not choices:
        # A fixed compact canvas can rule out a four-patch row or column.
        mode = "scattered"
        rows, cols = _layout_shape(count, mode)
        for h in range(2, 10):
            for w in range(2, 10):
                if (h * w >= 2 * count and 2 + rows * h + rows - 1 <= max_h
                        and 2 + cols * w + cols - 1 <= max_w
                        and 2 * count * h * w < max_h * max_w):
                    choices.append((h, w))
                    weights.append((h * w) ** -0.45)
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    h, w = choices[int(rng.choice(len(choices), p=probabilities))]
    min_h = 2 + rows * h + rows - 1
    min_w = 2 + cols * w + cols - 1
    if canvas is None:
        if min_h <= 16 and min_w <= 16 and rng.random() < 0.19:
            grid_h = grid_w = 16
        else:
            grid_h = min(30, min_h + int(rng.integers(0, 13)))
            grid_w = min(30, min_w + int(rng.integers(0, 13)))
    else:
        grid_h, grid_w = canvas
    if canvas is None:
        while grid_h * grid_w <= 2 * count * h * w:
            if grid_h < 30 and (grid_h <= grid_w or grid_w == 30):
                grid_h += 1
            else:
                grid_w += 1

    # Allocate spare rows/columns to margins and gaps. Every patch retains a
    # full outer-field border and at least one separating row or column.
    row_space = rng.multinomial(grid_h - min_h, [1 / (rows + 1)] * (rows + 1))
    col_space = rng.multinomial(grid_w - min_w, [1 / (cols + 1)] * (cols + 1))
    row_starts = [1 + int(row_space[0])]
    col_starts = [1 + int(col_space[0])]
    for i in range(1, rows):
        row_starts.append(row_starts[-1] + h + 1 + int(row_space[i]))
    for i in range(1, cols):
        col_starts.append(col_starts[-1] + w + 1 + int(col_space[i]))
    if mode == "scattered":
        slots = [(r, c) for r in range(2) for c in range(2)]
        if count == 2:
            slots = [(0, 0), (1, 1)] if rng.random() < 0.5 else [(0, 1), (1, 0)]
        elif count == 3:
            slots.pop(int(rng.integers(4)))
    else:
        slots = [(r, c) for r in range(rows) for c in range(cols)]
    positions = [(row_starts[r], col_starts[c]) for r, c in slots]
    rng.shuffle(positions)
    return h, w, grid_h, grid_w, positions


_MOTIFS = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)


def _labels(rng, h, w, count):
    labels = np.full((h, w), -1, dtype=np.int8)
    full = rng.random() < 0.105
    singleton = rng.random() < 0.055
    for label in range(count):
        placed = False
        for _ in range(90):
            if singleton and label == count - 1:
                motif = ((0, 0),)
            else:
                motif = _MOTIFS[int(rng.integers(len(_MOTIFS)))]
            turn = int(rng.integers(4))
            cells = []
            for r, c in motif:
                if turn == 0:
                    cells.append((r, c))
                elif turn == 1:
                    cells.append((c, -r))
                elif turn == 2:
                    cells.append((-r, -c))
                else:
                    cells.append((-c, r))
            low_r = min(r for r, _ in cells)
            low_c = min(c for _, c in cells)
            cells = [(r - low_r, c - low_c) for r, c in cells]
            shape_h = max(r for r, _ in cells) + 1
            shape_w = max(c for _, c in cells) + 1
            if shape_h > h or shape_w > w:
                continue
            top = int(rng.integers(h - shape_h + 1))
            left = int(rng.integers(w - shape_w + 1))
            cells = [(top + r, left + c) for r, c in cells]
            if (len(cells) <= np.count_nonzero(labels == -1) - (count - label - 1) - int(not full)
                    and all(labels[r, c] == -1 for r, c in cells)):
                for r, c in cells:
                    labels[r, c] = label
                placed = True
                break
        if not placed:
            empty = np.argwhere(labels == -1)
            assert len(empty) > 0
            r, c = empty[int(rng.integers(len(empty)))]
            labels[r, c] = label

    target = h * w if full else min(h * w - 1, max(int(np.count_nonzero(labels >= 0)), int(round(h * w * rng.uniform(0.34, 0.76)))))
    while np.count_nonzero(labels >= 0) < target:
        empty = np.argwhere(labels == -1)
        if rng.random() < 0.86:
            candidates = []
            for r, c in empty:
                neighbors = []
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] >= 0:
                        neighbors.append(int(labels[rr, cc]))
                if neighbors:
                    candidates.append((int(r), int(c), neighbors))
            if candidates:
                r, c, neighbors = candidates[int(rng.integers(len(candidates)))]
                labels[r, c] = neighbors[int(rng.integers(len(neighbors)))]
                continue
        r, c = empty[int(rng.integers(len(empty)))]
        sizes = np.array([np.count_nonzero(labels == i) for i in range(count)])
        labels[r, c] = int(np.argmin(sizes + rng.random(count)))
    return labels


def _colors(rng, pool, count, blank, outer, train_outers, seen_train, regime):
    available = [int(c) for c in pool if c not in (blank, outer)]
    repeated = count >= 3 and rng.random() < 0.23
    distinct = count - int(repeated)
    payloads = [int(c) for c in rng.choice(available, size=distinct, replace=False)]
    if repeated:
        payloads.append(payloads[int(rng.integers(len(payloads)))])
    if regime == "target_test":
        if rng.random() < 0.43:
            unseen = [c for c in range(10) if c not in seen_train and c not in (blank, outer)]
            if unseen:
                payloads[0] = int(rng.choice(unseen))
        elif train_outers and rng.random() < 0.28:
            reused = [c for c in train_outers if c not in (blank, outer)]
            if reused:
                payloads[0] = int(rng.choice(reused))
    # At least one patch must have a different payload from any repeated pair,
    # so the intersection of all patch palettes contains only the blank.
    if len(set(payloads)) == 1:
        payloads[-1] = next(c for c in available if c != payloads[0])
    rng.shuffle(payloads)
    return payloads


def _pair(rng, regime, index, params, seen_train, train_payloads, train_outers):
    if regime == "source_train":
        if index == 0:
            count = int(rng.choice((3, 4)))
        elif index == 1:
            count = 2 if params["first_count"] == 3 else int(rng.choice((2, 3)))
        else:
            count = int(rng.integers(2, 5))
    else:
        count = int(rng.integers(2, 5))
    if params["shared_blank"]:
        blank = params["blank"]
    elif regime == "source_train":
        blank = params["train_blanks"][index % len(params["train_blanks"])]
    else:
        blank = int(rng.choice(params["train_pool"]))

    pool = params["train_pool"] if regime == "source_train" else tuple(range(10))
    outer_choices = [int(c) for c in pool if c != blank]
    if regime == "target_test" and train_payloads and rng.random() < 0.27:
        reused = [c for c in train_payloads if c != blank]
        outer = int(rng.choice(reused)) if reused else int(rng.choice(outer_choices))
    else:
        outer = int(rng.choice(outer_choices))
    payloads = _colors(rng, pool, count, blank, outer, train_outers, seen_train, regime)
    h, w, grid_h, grid_w, positions = _geometry(rng, count, params["canvas"])
    labels = _labels(rng, h, w, count)
    grid = np.full((grid_h, grid_w), outer, dtype=np.int8)
    output = np.full((h, w), blank, dtype=np.int8)
    crops = []
    for label, payload in enumerate(payloads):
        crop = np.full((h, w), blank, dtype=np.int8)
        crop[labels == label] = payload
        crops.append(crop)
    # Repeated payloads may agree at the same registered coordinates.
    for i in range(count):
        for j in range(i + 1, count):
            if payloads[i] == payloads[j] and rng.random() < 0.62:
                candidates = np.argwhere(crops[i] == payloads[i])
                rng.shuffle(candidates)
                for r, c in candidates[:min(3, len(candidates))]:
                    if np.count_nonzero(crops[j] == blank) > 1:
                        crops[j][r, c] = payloads[j]
    for (top, left), crop in zip(positions, crops):
        grid[top:top + h, left:left + w] = crop
        mask = crop != blank
        assert np.all((output[mask] == blank) | (output[mask] == crop[mask]))
        output[mask] = crop[mask]
    return {"input": grid, "output": output}


def _grid_key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col] or seen[row, col]:
                continue
            cells = []
            stack = [(row, col)]
            seen[row, col] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append(cells)
    return result


def _analyze_pair(input_grid, output_grid):
    border = np.concatenate((input_grid[0], input_grid[-1], input_grid[1:-1, 0], input_grid[1:-1, -1]))
    outer = int(border[0])
    assert np.all(border == outer), "nonuniform outer border"
    components = _components(input_grid != outer)
    assert 2 <= len(components) <= 4
    crops = []
    positions = []
    for cells in components:
        top = min(r for r, _ in cells)
        bottom = max(r for r, _ in cells)
        left = min(c for _, c in cells)
        right = max(c for _, c in cells)
        crop = input_grid[top:bottom + 1, left:right + 1]
        assert crop.size == len(cells), "patch is not a complete rectangle"
        assert np.all(crop != outer)
        crops.append(crop)
        positions.append((top, left))
    assert len({crop.shape for crop in crops}) == 1, "patches are not congruent"
    palettes = [set(int(c) for c in np.unique(crop)) for crop in crops]
    assert all(len(p) == 2 for p in palettes), "each patch needs a blank and payload"
    common = set.intersection(*palettes)
    assert len(common) == 1, "blank cannot be uniquely identified"
    blank = next(iter(common))
    assert blank != outer
    h, w = crops[0].shape
    recovered = np.full((h, w), blank, dtype=np.int8)
    overlap = False
    payloads = []
    singleton_count = 0
    disconnected = False
    for crop, palette in zip(crops, palettes):
        payload = next(iter(palette - {blank}))
        payloads.append(payload)
        mask = crop != blank
        assert np.any(mask) and np.any(crop == blank)
        singleton_count += int(np.count_nonzero(mask) == 1)
        disconnected |= len(_components(mask)) > 1
        overlap |= bool(np.any((recovered != blank) & mask))
        assert np.all((recovered[mask] == blank) | (recovered[mask] == crop[mask])), "conflicting overlap"
        recovered[mask] = crop[mask]
    assert output_grid.shape == (h, w)
    assert np.array_equal(output_grid, recovered), "incorrect transparent union"
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    layout = "row" if len(set(rows)) == 1 else "column" if len(set(cols)) == 1 else "scattered"
    return {"outer": outer, "blank": blank, "payloads": tuple(payloads), "count": len(crops),
            "shape": (h, w), "canvas": input_grid.shape, "positions": tuple(positions),
            "layout": layout, "full": not np.any(output_grid == blank), "overlap": overlap,
            "repeated": len(set(payloads)) < len(payloads), "singletons": singleton_count,
            "disconnected": disconnected, "occupancy": float(np.mean(output_grid != blank))}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    n_train, n_test = _counts(rng, seed, num_train, num_test)
    train_pool = tuple(int(c) for c in rng.permutation(10)[:7])
    shared_blank = bool(rng.integers(2))
    canvas = None
    if rng.random() < 0.25:
        canvas = (16, 16) if rng.random() < 0.50 else (int(rng.integers(13, 25)), int(rng.integers(13, 25)))
    params = {"train_pool": train_pool, "shared_blank": shared_blank,
              "blank": train_pool[0], "train_blanks": tuple(rng.permutation(train_pool)),
              "canvas": canvas, "first_count": 0}
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_train = set()
    train_payloads = set()
    train_outers = set()
    for section, count in (("train", n_train), ("test", n_test)):
        regime = "source_train" if section == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(8):
                pair = _pair(pair_rng, regime, index, params, seen_train, train_payloads, train_outers)
                key = _grid_key(pair["input"])
                if key not in seen_inputs:
                    break
            else:
                raise AssertionError("could not generate a unique input")
            seen_inputs.add(key)
            puzzle[section].append(pair)
            if section == "train":
                info = _analyze_pair(pair["input"], pair["output"])
                if index == 0:
                    params["first_count"] = info["count"]
                seen_train.update(int(c) for c in np.unique(pair["input"]))
                train_payloads.update(info["payloads"])
                train_outers.add(info["outer"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            key = _grid_key(input_grid)
            pair_key = (key, _grid_key(output_grid))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            if input_grid.shape == (16, 16):
                assert hashlib.sha256(input_grid.astype(np.uint8).tobytes()).hexdigest() not in _SOURCE_INPUT_DIGESTS
            infos[section].append(_analyze_pair(input_grid, output_grid))
    # Count variation makes the rule visible even in the minimum two-example task.
    assert len({info["count"] for info in infos["train"]}) >= 2
    # Every blank, including a new test blank, is recoverable as the unique
    # common color of all patch palettes. No literal palette map is needed.


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    records = []
    signatures = set()
    layouts = {"train": set(), "test": set()}
    orientations = {"train": set(), "test": set()}
    counts = {"train": set(), "test": set()}
    occupancy = {"train": [], "test": []}
    stats = {name: 0 for name in ("small_canvas", "canvas_16", "large_patch", "slim_patch",
                                   "full_train", "full_test", "overlap_train", "overlap_test",
                                   "novel_test", "familiar_test", "role_reuse", "bar", "L",
                                   "disconnected", "singletons", "fragments", "constant_canvas",
                                   "shared_blank", "varying_blank", "shared_count", "shared_layout",
                                   "shared_orientation")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = set(int(c) for pair in puzzle["train"] for c in np.unique(pair["input"]))
        train_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_infos = train_infos + test_infos
        stats["constant_canvas"] += len({i["canvas"] for i in all_infos}) == 1
        stats["shared_blank"] += len({i["blank"] for i in all_infos}) == 1
        stats["varying_blank"] += len({i["blank"] for i in all_infos}) > 1
        stats["shared_count"] += bool({i["count"] for i in train_infos} & {i["count"] for i in test_infos})
        stats["shared_layout"] += bool({i["layout"] for i in train_infos} & {i["layout"] for i in test_infos})
        stats["shared_orientation"] += bool({np.sign(i["shape"][0] - i["shape"][1]) for i in train_infos} & {np.sign(i["shape"][0] - i["shape"][1]) for i in test_infos})
        train_payloads = {c for i in train_infos for c in i["payloads"]}
        train_outers = {i["outer"] for i in train_infos}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                records.append((section, info, train_colors))
                h, w = info["shape"]
                H, W = info["canvas"]
                orientation = "portrait" if h > w else "landscape" if h < w else "square"
                layouts[section].add(info["layout"])
                orientations[section].add(orientation)
                counts[section].add(info["count"])
                occupancy[section].append(info["occupancy"])
                stats["small_canvas"] += H < 16 and W < 16
                stats["canvas_16"] += (H, W) == (16, 16)
                stats["large_patch"] += h * w >= 36
                stats["slim_patch"] += min(h, w) == 2
                stats["disconnected"] += bool(info["disconnected"])
                stats["full_" + section] += bool(info["full"])
                stats["overlap_" + section] += bool(info["overlap"])
                if section == "test":
                    stats["novel_test"] += bool(set(info["payloads"]) - train_colors)
                    stats["familiar_test"] += bool(set(info["payloads"]) & train_payloads)
                    stats["role_reuse"] += info["outer"] in train_payloads or bool(set(info["payloads"]) & train_outers)
                for top, left in info["positions"]:
                    crop = pair["input"][top:top + h, left:left + w]
                    mask = crop != info["blank"]
                    coordinates = np.argwhere(mask)
                    stats["fragments"] += 1
                    stats["singletons"] += len(coordinates) == 1
                    stats["bar"] += len(coordinates) >= 2 and (len(set(coordinates[:, 0])) == 1 or len(set(coordinates[:, 1])) == 1)
                    stats["L"] += len(coordinates) == 3 and np.ptp(coordinates[:, 0]) == 1 and np.ptp(coordinates[:, 1]) == 1
                # Color-normalized local output plus relative patch positions.
                mapping = {info["blank"]: -1}
                pattern = []
                for value in pair["output"].flat:
                    value = int(value)
                    if value not in mapping:
                        mapping[value] = len(mapping)
                    pattern.append(mapping[value])
                rs, cs = zip(*info["positions"])
                positions = tuple(sorted((r - min(rs), c - min(cs)) for r, c in info["positions"]))
                signatures.add((info["shape"], info["count"], positions, tuple(pattern)))
    assert len(signatures) >= 110
    assert all(counts[s] == {2, 3, 4} for s in ("train", "test"))
    assert all(layouts[s] == {"row", "column", "scattered"} for s in ("train", "test"))
    assert all(orientations[s] == {"portrait", "square", "landscape"} for s in ("train", "test"))
    assert stats["small_canvas"] >= 5 and stats["canvas_16"] >= 10
    assert stats["large_patch"] >= 30 and stats["slim_patch"] >= 30
    assert stats["constant_canvas"] >= 4
    assert stats["shared_blank"] >= 8 and stats["varying_blank"] >= 8
    assert stats["full_train"] >= 5 and stats["full_test"] >= 2
    assert stats["overlap_train"] >= 5 and stats["overlap_test"] >= 5
    assert stats["novel_test"] >= 15 and stats["familiar_test"] >= 15 and stats["role_reuse"] >= 15
    assert stats["bar"] >= 40 and stats["L"] >= 20 and stats["disconnected"] >= 20
    assert 0.01 < stats["singletons"] / stats["fragments"] < 0.12
    assert stats["shared_count"] >= 15 and stats["shared_layout"] >= 15 and stats["shared_orientation"] >= 15
    assert abs(np.mean(occupancy["train"]) - np.mean(occupancy["test"])) < 0.10
    assert {i["count"] for _, i, _ in records} == {2, 3, 4}
    assert {i["layout"] for _, i, _ in records} == {"row", "column", "scattered"}
    assert {"portrait", "square", "landscape"} == {"portrait" if i["shape"][0] > i["shape"][1] else "square" if i["shape"][0] == i["shape"][1] else "landscape" for _, i, _ in records}
    assert any(i["canvas"] == (16, 16) for _, i, _ in records)
    assert any(min(i["canvas"]) < 16 for _, i, _ in records)
    assert any(i["full"] for _, i, _ in records)
    assert any(i["overlap"] and i["repeated"] for _, i, _ in records)
    assert any(i["disconnected"] for _, i, _ in records)
    assert any(section == "test" and set(i["payloads"]) - colors for section, i, colors in records)
    assert any(len({ _analyze_pair(p["input"], p["output"])["blank"] for p in puzzle["train"] + puzzle["test"]}) > 1 for puzzle in puzzles)
    assert any(len({p["input"].shape for p in puzzle["train"] + puzzle["test"]}) == 1 for puzzle in puzzles)
    assert _equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _equal(generate_puzzle(2718, 6, 2), generate_puzzle(2719, 6, 2))
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((200, None, 1), (201, None, 4), (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": 2.0}):
        try:
            generate_puzzle(999, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {request}")
