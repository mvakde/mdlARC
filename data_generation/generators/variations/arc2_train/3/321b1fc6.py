"""Generate translated, multicolored ARC stamps and their blank silhouettes."""

import numpy as np


# Exact source pairs are excluded without excluding their dimensions or palettes.
_SOURCE_PAIRS = {
    (
        "0000000000/0880000000/0888008800/0000008880/0000000000/"
        "0000077000/0000066600/0008800000/0008880000/0000000000",
        "0000000000/0770000000/0666007700/0000006660/0000000000/"
        "0000000000/0000000000/0007700000/0006660000/0000000000",
    ),
    (
        "0000000000/0760000000/0940000000/0000000000/0000088000/"
        "0000088000/0000000000/0088000000/0088000088/0000000088",
        "0000000000/0000000000/0000000000/0000000000/0000076000/"
        "0000094000/0000000000/0076000000/0094000076/0000000094",
    ),
    (
        "0000000000/0880008800/8888088880/0080000800/0000000000/"
        "0440000000/3433008800/0030088880/0000000800/0000000000",
        "0000000000/0440004400/3433034330/0030000300/0000000000/"
        "0000000000/0000004400/0000034330/0000000300/0000000000",
    ),
}


def _text(grid: np.ndarray) -> str:
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, bytes(int(value) for value in grid.flat))


def _resolve_counts(rng, num_train, num_test):
    def checked(value, lo, hi, name):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
        return value

    train = checked(num_train, 2, 10, "num_train")
    test = checked(num_test, 1, 4, "num_test")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _neighbors(row, col, height, width):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _mask_connected(mask):
    cells = np.argwhere(mask)
    start = tuple(int(v) for v in cells[0])
    seen = {start}
    stack = [start]
    while stack:
        row, col = stack.pop()
        for neighbor in _neighbors(row, col, *mask.shape):
            if mask[neighbor] and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(cells)


def _sample_mask(rng, regime, size_mode, style):
    if size_mode == "extrapolate":
        low_h, high_h, low_w, high_w = (
            (2, 3, 2, 4) if regime == "source_train" else (3, 6, 3, 7)
        )
    elif size_mode == "reverse":
        low_h, high_h, low_w, high_w = (
            (3, 7, 3, 7) if regime == "source_train" else (2, 4, 2, 4)
        )
    else:
        low_h, high_h, low_w, high_w = 2, 7, 2, 7
    height = int(rng.integers(low_h, high_h + 1))
    width = int(rng.integers(low_w, high_w + 1))
    if style == "solid":
        return np.ones((height, width), dtype=bool)

    # Include the source test's seven-cell, four-corner-free branch among
    # the many shapes produced by the general connected-growth modes.
    if (height, width) == (3, 4) and rng.random() < 0.30:
        return np.asarray([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0]], dtype=bool)

    if height >= 3 and width >= 3 and rng.random() < 0.24:
        # Two independently located, narrow arms span the bounding box.
        row = int(rng.integers(1, height - 1))
        col = int(rng.integers(1, width - 1))
        mask = np.zeros((height, width), dtype=bool)
        mask[row, :] = True
        mask[:, col] = True
        for _ in range(int(rng.integers(0, 3))):
            frontier = [(r, c) for r in range(height) for c in range(width)
                        if not mask[r, c] and any(mask[nr, nc] for nr, nc in _neighbors(r, c, height, width))]
            mask[frontier[int(rng.integers(len(frontier)))]] = True
        return mask

    if height >= 4 and width >= 4 and rng.random() < 0.12:
        # A slit reaches inward from a boundary without disconnecting the stamp.
        mask = np.ones((height, width), dtype=bool)
        if rng.random() < 0.5:
            col = int(rng.integers(1, width - 1))
            depth = int(rng.integers(2, height))
            mask[:depth, col] = False
        else:
            row = int(rng.integers(1, height - 1))
            depth = int(rng.integers(2, width))
            mask[row, :depth] = False
        return mask

    # A removed interior patch gives an enclosed hole while leaving a
    # connected border. The other modes grow a connected shape freely.
    if height >= 3 and width >= 3 and rng.random() < 0.16:
        mask = np.ones((height, width), dtype=bool)
        interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
        rng.shuffle(interior)
        for row, col in interior[:int(rng.integers(1, max(2, len(interior) // 2 + 1)))]:
            mask[row, col] = False
            if not _mask_connected(mask):
                mask[row, col] = True
        return mask

    no_corners = height >= 3 and width >= 3 and rng.random() < 0.48
    forbidden = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)} if no_corners else set()
    candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in forbidden]
    start = candidates[int(rng.integers(len(candidates)))]
    mask = np.zeros((height, width), dtype=bool)
    mask[start] = True
    capacity = len(candidates)
    density = float(rng.uniform(0.32, 0.91))
    target = min(capacity, max(3, int(round(height * width * density))))
    target = min(target, capacity - 1 if capacity == height * width else capacity)
    while True:
        rows, cols = np.nonzero(mask)
        sides = (0 in rows, height - 1 in rows, 0 in cols, width - 1 in cols)
        if all(sides) and int(mask.sum()) >= target:
            break
        frontier = []
        weights = []
        for row, col in candidates:
            if mask[row, col] or not any(mask[nr, nc] for nr, nc in _neighbors(row, col, height, width)):
                continue
            weight = 1.0
            if not sides[0] and row == 0:
                weight += 4
            if not sides[1] and row == height - 1:
                weight += 4
            if not sides[2] and col == 0:
                weight += 4
            if not sides[3] and col == width - 1:
                weight += 4
            frontier.append((row, col))
            weights.append(weight)
        assert frontier
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        mask[frontier[int(rng.choice(len(frontier), p=probabilities))]] = True
    return mask


def _separate(first, second, height, width):
    r1, c1 = first
    r2, c2 = second
    return (r1 + height < r2 or r2 + height < r1 or
            c1 + width < c2 or c2 + width < c1)


def _sample_layout(rng, mask, target_count):
    height, width = mask.shape
    n = target_count + 1
    gap = int(rng.integers(1, 4))
    layouts = []
    for columns in range(1, n + 1):
        rows = (n + columns - 1) // columns
        base_h = rows * height + (rows - 1) * gap
        base_w = columns * width + (columns - 1) * gap
        if base_h <= 30 and base_w <= 30:
            layouts.append((rows, columns, base_h, base_w))
    # One-cell separation and majority background are always possible
    # for 2..7 objects of at most 7 by 7 cells.
    rows, columns, base_h, base_w = layouts[int(rng.integers(len(layouts)))]
    canvas_h = min(30, base_h + int(rng.integers(0, 11)))
    canvas_w = min(30, base_w + int(rng.integers(0, 11)))
    minimum_area = int(2.15 * n * int(mask.sum())) + 1
    while canvas_h * canvas_w < minimum_area:
        if canvas_h < 30 and (canvas_h <= canvas_w or canvas_w == 30):
            canvas_h += 1
        elif canvas_w < 30:
            canvas_w += 1
        else:
            raise AssertionError("canvas cannot preserve a clear background")
    top = int(rng.integers(canvas_h - base_h + 1))
    left = int(rng.integers(canvas_w - base_w + 1))
    placements = [(top + (i // columns) * (height + gap),
                   left + (i % columns) * (width + gap)) for i in range(n)]
    # Move each object independently in free space. The initial shelf
    # packing is only a guaranteed feasible starting point.
    if rng.random() < 0.82:
        for _ in range(3):
            for index in rng.permutation(n):
                index = int(index)
                for _ in range(35):
                    proposed = (int(rng.integers(canvas_h - height + 1)),
                                int(rng.integers(canvas_w - width + 1)))
                    if all(_separate(proposed, other, height, width)
                           for j, other in enumerate(placements) if j != index):
                        placements[index] = proposed
                        break
    return (canvas_h, canvas_w), placements


def _sample_payload(rng, mask, background, placeholders):
    allowed = [color for color in range(10) if color != background and color not in placeholders]
    coordinates = np.argwhere(mask)
    area = len(coordinates)
    mode = int(rng.integers(6))
    if mode in (4, 5):
        count = min(len(allowed), area, int(rng.integers(2, 5)))
    else:
        count = int(rng.integers(2, min(7, len(allowed), area) + 1))
    if mode in (1, 2):
        axis = 0 if mode == 1 else 1
        count = min(count, len(np.unique(coordinates[:, axis])))
        if count < 2:
            mode = 0
            count = 2
    colors = [int(c) for c in rng.choice(allowed, size=count, replace=False)]
    assigned = np.empty(area, dtype=np.int8)
    if mode == 0:  # mosaic, with every chosen color represented
        order = rng.permutation(area)
        assigned[order[:count]] = colors
        assigned[order[count:]] = rng.choice(colors, size=area - count)
    elif mode in (1, 2):  # horizontal or vertical bands
        axis = 0 if mode == 1 else 1
        ranks = {int(v): i for i, v in enumerate(sorted(set(coordinates[:, axis])))}
        assigned[:] = [colors[ranks[int(pos[axis])] % count] for pos in coordinates]
    elif mode == 3:  # contiguous nearest-seed patches
        centers = coordinates[rng.choice(area, size=count, replace=False)]
        for i, pos in enumerate(coordinates):
            distances = np.abs(centers - pos).sum(axis=1)
            assigned[i] = colors[int(np.argmin(distances))]
    elif mode == 4:  # one dominant field with rare accent cells
        assigned[:] = colors[0]
        order = rng.permutation(area)
        accents = max(1, min(area // 4, count + 1))
        for i, index in enumerate(order[:accents]):
            assigned[int(index)] = colors[1 + i % (count - 1)]
    else:  # repeating diagonal motif
        assigned[:] = [colors[(int(row) + int(col)) % count] for row, col in coordinates]
    if len(set(int(v) for v in assigned)) < 2:
        assigned[int(rng.integers(area))] = colors[1]
    payload = np.full(mask.shape, background, dtype=np.int8)
    for (row, col), color in zip(coordinates, assigned):
        payload[int(row), int(col)] = color
    return payload


def _generate_pair(rng, regime, size_mode, style, target_count, avoid_background=None,
                   avoid_placeholder=None):
    palette = list(range(10))
    background_options = [c for c in palette if c != avoid_background]
    background = int(rng.choice(background_options))
    mask = _sample_mask(rng, regime, size_mode, style)
    placeholder_options = [c for c in palette if c != background and c != avoid_placeholder]
    distinct = 1
    if target_count >= 2 and rng.random() < 0.48:
        distinct = int(rng.integers(2, min(3, target_count) + 1))
    placeholders = [int(c) for c in rng.choice(placeholder_options, size=distinct, replace=False)]
    payload = _sample_payload(rng, mask, background, placeholders)
    shape, placements = _sample_layout(rng, mask, target_count)
    exemplar_index = int(rng.integers(len(placements)))
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = np.full(shape, background, dtype=np.int8)
    occupied = np.argwhere(mask)
    target_index = 0
    for index, (top, left) in enumerate(placements):
        color = placeholders[target_index % len(placeholders)] if index != exemplar_index else None
        if index != exemplar_index:
            target_index += 1
        for row, col in occupied:
            row, col = int(row), int(col)
            if index == exemplar_index:
                input_grid[top + row, left + col] = payload[row, col]
            else:
                input_grid[top + row, left + col] = color
                output_grid[top + row, left + col] = payload[row, col]
    return {"input": input_grid, "output": output_grid}


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in _neighbors(r, c, height, width):
                    if not seen[nr, nc] and int(grid[nr, nc]) != background:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(cells)
    return result


def _describe(grid, cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    height = max(r for r, _ in cells) - top + 1
    width = max(c for _, c in cells) - left + 1
    mask = np.zeros((height, width), dtype=bool)
    colors = set()
    for row, col in cells:
        mask[row - top, col - left] = True
        colors.add(int(grid[row, col]))
    return {"top": top, "left": left, "mask": mask, "colors": colors,
            "cells": cells}


def _has_hole(mask):
    empty = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    stack = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if empty[row, col] and (row in (0, height - 1) or col in (0, width - 1)):
                seen[row, col] = True
                stack.append((row, col))
    while stack:
        row, col = stack.pop()
        for nr, nc in _neighbors(row, col, height, width):
            if empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(empty & ~seen))


def _deep_notch(mask):
    for variant in (mask, mask[::-1], mask.T, mask.T[::-1]):
        if variant.shape[0] >= 3 and variant.shape[1] >= 3:
            for col in range(1, variant.shape[1] - 1):
                if (not variant[0, col] and not variant[1, col] and
                        variant[0, col - 1] and variant[0, col + 1]):
                    return True
    return False


def _layout_kind(tops):
    rows = {r for r, _ in tops}
    cols = {c for _, c in tops}
    if len(rows) == 1:
        return "row"
    if len(cols) == 1:
        return "column"
    if len(rows) > 1 and len(cols) > 1 and len(tops) == len(rows) * len(cols):
        return "grid"
    return "scatter"


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    colors, counts = np.unique(input_grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(counts.max()) > input_grid.size // 2
    objects = [_describe(input_grid, cells) for cells in _components(input_grid, background)]
    exemplars = [obj for obj in objects if len(obj["colors"]) >= 2]
    targets = [obj for obj in objects if len(obj["colors"]) == 1]
    assert len(exemplars) == 1 and len(targets) >= 1
    exemplar = exemplars[0]
    mask = exemplar["mask"]
    placeholder_colors = {next(iter(obj["colors"])) for obj in targets}
    assert background not in placeholder_colors
    assert placeholder_colors.isdisjoint(exemplar["colors"])
    for obj in targets:
        assert np.array_equal(obj["mask"], mask)
    expected = np.full(input_grid.shape, background, dtype=input_grid.dtype)
    for obj in targets:
        for row, col in np.argwhere(mask):
            row, col = int(row), int(col)
            expected[obj["top"] + row, obj["left"] + col] = input_grid[
                exemplar["top"] + row, exemplar["left"] + col]
    assert np.array_equal(expected, output_grid)
    assert not np.array_equal(input_grid, output_grid)
    edge = any(r in (0, input_grid.shape[0] - 1) or c in (0, input_grid.shape[1] - 1)
               for obj in objects for r, c in obj["cells"])
    corners_empty = not any(mask[r, c] for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    top, left = exemplar["top"], exemplar["left"]
    payload = input_grid[top:top + mask.shape[0], left:left + mask.shape[1]]
    occupied = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    values = [int(payload[r, c]) for r, c in occupied]
    bands = (all(len({int(payload[row, col]) for col in range(mask.shape[1]) if mask[row, col]}) == 1
                 for row in range(mask.shape[0])) or
             all(len({int(payload[row, col]) for row in range(mask.shape[0]) if mask[row, col]}) == 1
                 for col in range(mask.shape[1])))
    motifs = False
    for period in (2, 3, 4):
        groups = {}
        for row, col in occupied:
            groups.setdefault((row + col) % period, set()).add(int(payload[row, col]))
        if len(groups) >= 2 and all(len(group) == 1 for group in groups.values()) and len(
                {next(iter(group)) for group in groups.values()}) >= 2:
            motifs = True
    asymmetry = any(np.array_equal(mask, transformed_mask) and
                    not np.array_equal(payload, transformed_payload)
                    for transformed_mask, transformed_payload in
                    ((np.fliplr(mask), np.fliplr(payload)), (np.flipud(mask), np.flipud(payload))))
    branch = any(sum((nr, nc) in occupied for nr, nc in _neighbors(row, col, *mask.shape)) >= 3
                 for row, col in occupied)
    target_positions = [(obj["top"], obj["left"]) for obj in targets]
    return {"background": background, "placeholders": placeholder_colors,
            "mask": mask, "target_count": len(targets), "solid": bool(np.all(mask)),
            "hole": _has_hole(mask), "corners_empty": corners_empty,
            "edge": edge, "payload_count": len(exemplar["colors"]),
            "sparse": int(mask.sum()) / mask.size <= 0.5,
            "dense": int(mask.sum()) / mask.size >= 0.8,
            "asymmetric_mask": not np.array_equal(mask, np.fliplr(mask)) and
                               not np.array_equal(mask, np.flipud(mask)),
            "object_fill": len(objects) * int(mask.sum()) / input_grid.size,
            "branch": branch, "deep_notch": _deep_notch(mask),
            "bands": bands, "motifs": motifs,
            "accent": max(values.count(value) for value in set(values)) / len(values) >= 0.75,
            "asymmetric_payload_on_symmetric_mask": asymmetry,
            "staggered_targets": len({row for row, _ in target_positions}) > 1 and
                                 len({col for _, col in target_positions}) > 1,
            "layout": _layout_kind([(obj["top"], obj["left"]) for obj in objects]),
            "positions": [(obj["top"], obj["left"]) for obj in objects]}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # This mode only changes the size distribution. The semantic rule has
    # no hidden puzzle-wide color or geometric convention.
    size_mode = str(rng.choice(["extrapolate", "overlap", "reverse"], p=[0.36, 0.44, 0.20]))
    train_styles = ["solid", "irregular"] + [str(rng.choice(["solid", "irregular"], p=[0.23, 0.77]))
                                               for _ in range(train_count - 2)]
    rng.shuffle(train_styles)
    train_targets = [1, int(rng.integers(2, 7))] + [int(rng.integers(1, 7))
                                                      for _ in range(train_count - 2)]
    rng.shuffle(train_targets)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            regime = "source_train" if split == "train" else "target_test"
            style = train_styles[index] if split == "train" else str(
                rng.choice(["solid", "irregular"], p=[0.24, 0.76]))
            target_count = train_targets[index] if split == "train" else int(rng.integers(1, 7))
            previous = puzzle["train"][0] if split == "train" and index == 1 else None
            prior = _analyze_pair(previous["input"], previous["output"]) if previous is not None else None
            for _ in range(60):
                pair = _generate_pair(rng, regime, size_mode, style, target_count,
                                      prior["background"] if prior else None,
                                      next(iter(prior["placeholders"])) if prior else None)
                signature = _signature(pair["input"])
                if signature in seen or (_text(pair["input"]), _text(pair["output"])) in _SOURCE_PAIRS:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            input_sig = _signature(pair["input"])
            pair_sig = (input_sig, _signature(pair["output"]))
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            assert (_text(pair["input"]), _text(pair["output"])) not in _SOURCE_PAIRS
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    # Demonstrations expose shape independence, local backgrounds, and
    # variable copy counts. No hidden palette convention affects a test.
    train = analyses["train"]
    assert any(a["solid"] for a in train) and any(not a["solid"] for a in train)
    assert len({a["background"] for a in train}) >= 2
    assert len({a["target_count"] for a in train}) >= 2
    for test in analyses["test"]:
        assert test["target_count"] >= 1
        assert test["payload_count"] >= 2


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[key], b[key])
        for a, b in zip(first[split], second[split]) for key in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(123456), generate_puzzle(123456))
    assert _puzzles_equal(generate_puzzle(789, 10, 4), generate_puzzle(789, 10, 4))
    assert not _puzzles_equal(generate_puzzle(100), generate_puzzle(101))
    features = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                features.append((split, _analyze_pair(pair["input"], pair["output"]),
                                 pair["input"].shape))
    assert len({shape for _, _, shape in features}) >= 35
    assert len({_signature(a["mask"]) for _, a, _ in features}) >= 50
    assert {a["layout"] for _, a, _ in features} >= {"row", "column", "grid", "scatter"}
    assert {a["target_count"] for _, a, _ in features} >= {1, 2, 3, 4, 5, 6}
    assert any(a["corners_empty"] for _, a, _ in features)
    assert any(a["hole"] for _, a, _ in features)
    assert any(a["edge"] for _, a, _ in features)
    assert sum(a["sparse"] for _, a, _ in features) >= 10
    assert sum(a["dense"] for _, a, _ in features) >= 30
    assert sum(a["branch"] for _, a, _ in features) >= 20
    assert sum(a["deep_notch"] for _, a, _ in features) >= 5
    assert sum(a["corners_empty"] for _, a, _ in features) >= 20
    assert sum(a["asymmetric_mask"] for _, a, _ in features) >= 20
    assert sum(a["staggered_targets"] for _, a, _ in features) >= 30
    assert sum(a["object_fill"] < 0.1 for _, a, _ in features) >= 10
    assert sum(a["object_fill"] > 0.3 for _, a, _ in features) >= 10
    assert sum(shape[0] <= 10 and shape[1] <= 10 for _, _, shape in features) >= 5
    assert max(shape[0] for _, _, shape in features) == 30
    assert max(shape[1] for _, _, shape in features) == 30
    assert len({a["background"] for _, a, _ in features}) == 10
    assert sum(a["bands"] for _, a, _ in features) >= 10
    assert sum(a["motifs"] for _, a, _ in features) >= 10
    assert sum(a["accent"] for _, a, _ in features) >= 10
    assert sum(a["asymmetric_payload_on_symmetric_mask"] for _, a, _ in features) >= 10
    assert any(a["mask"].shape == (3, 4) and np.array_equal(
        a["mask"], np.asarray([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0]], dtype=bool))
        for _, a, _ in features)
    assert any(a["payload_count"] >= 5 for _, a, _ in features)
    assert any(a["payload_count"] == 2 for _, a, _ in features)
    assert any(len(a["placeholders"]) >= 2 for _, a, _ in features)
    assert any(split == "train" and a["mask"].shape[0] >= 5 for split, a, _ in features)
    assert any(split == "train" and a["mask"].shape[0] > a["mask"].shape[1]
               for split, a, _ in features)
    assert any(split == "test" and a["solid"] for split, a, _ in features)
    assert any(split == "test" and a["solid"] and max(a["mask"].shape) <= 3
               for split, a, _ in features)
    larger_tests = 0
    smaller_tests = 0
    for puzzle in puzzles:
        train_features = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test_features = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]]
        all_features = train_features + test_features
        assert len({a["background"] for a in all_features}) >= 2
        assert len({a["target_count"] for a in all_features}) >= 2
        assert len({a["mask"].shape for a in all_features}) >= 2
        assert len({tuple(sorted(a["placeholders"])) for a in all_features}) >= 2
        train_max = max(int(a["mask"].sum()) for a in train_features)
        larger_tests += max(int(a["mask"].sum()) for a in test_features) > train_max
        smaller_tests += min(int(a["mask"].sum()) for a in test_features) < train_max
    assert larger_tests >= 5 and smaller_tests >= 5
    for seed, train, test in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                              (1003, 7, 3), (1004, None, 1), (1005, None, 4),
                              (1006, 2, None), (1007, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.5}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {kwargs}")
