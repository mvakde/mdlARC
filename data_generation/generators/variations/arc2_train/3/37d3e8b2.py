from __future__ import annotations

import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "8c664254eff2f26b991237b2ec473ddb8d6ccc947693792e25cd37e03f690c30",
    "63334172ca80c076018fc538d4a0093796d51e53e6e68275e64e6e0669f96d4b",
    "abfb0431725d5ed341ed2ad8bf85db62c977a861f5a51ad5e3dc0b3f666ecec6",
    "5aa960f73ced0cd7b1fa7832478177bf74a52d54f96e3b46111f8ef808dbfbf3",
}


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} is outside its legal integer range")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(test + 1, 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _components(mask, diagonal=False):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        directions += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    answer = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            region = []
            while stack:
                r, c = stack.pop()
                region.append((r, c))
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            answer.append(region)
    return answer


def _holes(mask, diagonal=False):
    height, width = mask.shape
    return [region for region in _components(~mask, diagonal)
            if not any(r in (0, height - 1) or c in (0, width - 1)
                       for r, c in region)]


def _trim(mask):
    rows, cols = np.where(mask)
    return mask[rows.min():rows.max() + 1,
                cols.min():cols.max() + 1].copy()


def _valid_mask(mask, count):
    return (len(_components(mask)) == 1
            and len(_holes(mask)) == count
            and len(_holes(mask, True)) == count)


def _cavity(rng, height, width, area):
    start = (int(rng.integers(1, height - 1)),
             int(rng.integers(1, width - 1)))
    cells = {start}
    for _ in range(area - 1):
        choices = set()
        for row, col in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 1 <= nr < height - 1 and 1 <= nc < width - 1:
                    if (nr, nc) not in cells:
                        choices.add((nr, nc))
        if not choices:
            break
        ordered = sorted(choices)
        cells.add(ordered[int(rng.integers(len(ordered)))])
    return cells


def _place_cavities(rng, height, width, count, large=False, forced_area=None):
    for _ in range(45):
        cavities = []
        for index in range(count):
            if forced_area is not None and index == 0:
                area = forced_area
            elif large and index == 0:
                area = int(rng.integers(5, min(11, (height - 2) * (width - 2)) + 1))
            elif count >= 3 and min(height, width) <= 8:
                area = int(rng.choice((1, 1, 1, 2)))
            else:
                area = int(rng.choice((1, 1, 2, 3, 4, 5, 6)))
            for _ in range(45):
                candidate = _cavity(rng, height, width, area)
                if len(candidate) != area:
                    continue
                if all(max(abs(r - rr), abs(c - cc)) > 1
                       for cavity in cavities for r, c in candidate
                       for rr, cc in cavity):
                    cavities.append(candidate)
                    break
            else:
                break
        if len(cavities) == count:
            mask = np.ones((height, width), dtype=bool)
            for cavity in cavities:
                for row, col in cavity:
                    mask[row, col] = False
            return mask
    return None


def _shape_outline(rng, mask, count, style):
    if style == "plain":
        return mask
    height, width = mask.shape
    work = np.zeros((height + 8, width + 8), dtype=bool)
    work[4:4 + height, 4:4 + width] = mask
    if style in ("stepped", "mixed", "bay"):
        for _ in range(int(rng.integers(1, 4))):
            side = int(rng.integers(4))
            span = int(rng.integers(2, min(6, (width if side < 2 else height)) + 1))
            depth = int(rng.integers(1, 4))
            if side < 2:
                col = int(rng.integers(4, 5 + width - span))
                row = 4 if side == 0 else 4 + height - depth
                work[row:row + depth, col:col + span] = False
            else:
                row = int(rng.integers(4, 5 + height - span))
                col = 4 if side == 2 else 4 + width - depth
                work[row:row + span, col:col + depth] = False
            if not _valid_mask(_trim(work), count):
                work = np.zeros((height + 8, width + 8), dtype=bool)
                work[4:4 + height, 4:4 + width] = mask
    if style in ("lobe", "mixed"):
        side = int(rng.integers(4))
        reach = int(rng.integers(1, 4))
        if side < 2:
            span = int(rng.integers(2, min(6, width) + 1))
            col = int(rng.integers(4, 5 + width - span))
            edge = 4 if side == 0 else 3 + height
            outer = edge - reach if side == 0 else edge + reach
            work[min(edge, outer):max(edge, outer) + 1, col:col + span] = True
            if reach > 1:
                lcol = max(0, col - int(rng.integers(0, 2)))
                work[outer, lcol:min(work.shape[1], col + span + 1)] = True
        else:
            span = int(rng.integers(2, min(6, height) + 1))
            row = int(rng.integers(4, 5 + height - span))
            edge = 4 if side == 2 else 3 + width
            outer = edge - reach if side == 2 else edge + reach
            work[row:row + span, min(edge, outer):max(edge, outer) + 1] = True
            if reach > 1:
                lrow = max(0, row - int(rng.integers(0, 2)))
                work[lrow:min(work.shape[0], row + span + 1), outer] = True
    if style == "bridge":
        side = int(rng.integers(4))
        if side < 2:
            col = 4 + int(rng.integers(1, width - 1))
            if side == 0:
                work[2:4, col] = True
                work[0:2, col - 1:col + 3] = True
            else:
                work[height + 4:height + 6, col] = True
                work[height + 6:height + 8, col - 1:col + 3] = True
        else:
            row = 4 + int(rng.integers(1, height - 1))
            if side == 2:
                work[row, 2:4] = True
                work[row - 1:row + 3, 0:2] = True
            else:
                work[row, width + 4:width + 6] = True
                work[row - 1:row + 3, width + 6:width + 8] = True
    trimmed = _trim(work)
    return trimmed if _valid_mask(trimmed, count) else mask


def _make_object(rng, count, dense=False, anchor=False, holdout=False,
                 forced_area=None, avoid_shape=None):
    for _ in range(120):
        if anchor:
            height = width = 8
        else:
            low = 4 if count == 1 else 5
            high = 9 if dense else 12
            height = int(rng.integers(low, high + 1))
            width = int(rng.integers(low, high + 1))
        large = (count <= 2 and min(height, width) >= 9
                 and rng.random() < (0.48 if holdout else 0.27))
        mask = _place_cavities(rng, height, width, count, large, forced_area)
        if mask is None:
            continue
        styles = ("plain", "plain", "stepped", "bay", "lobe", "mixed", "bridge")
        style = (("plain", "plain", "stepped", "bay")
                 if anchor else styles)[int(rng.integers(4 if anchor else len(styles)))]
        mask = _shape_outline(rng, mask, count, style)
        mask = np.rot90(mask, int(rng.integers(4))).copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask).copy()
        if _valid_mask(mask, count) and mask.shape != avoid_shape:
            return mask
    # A deterministic, connected fallback for crowded requested scenes.
    height, width = (8, 8) if anchor else ((5, 2 * count + 3)
                                               if count <= 3 else (7, 7))
    mask = np.ones((height, width), dtype=bool)
    if height == 7:
        points = ((2, 2), (2, 4), (4, 2), (4, 4))
    else:
        points = tuple((2, 2 + 2 * index) for index in range(count))
    if forced_area == 2 and count == 1:
        mask[2, 2:4] = False
    elif forced_area == 4 and count == 1:
        mask[1:3, 1:3] = False
    else:
        for row, col in points[:count]:
            mask[row, col] = False
    assert _valid_mask(mask, count)
    return mask


def _observations(grid, background):
    answer = []
    for cells in _components(grid != background):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        top, left = min(rows), min(cols)
        mask = np.zeros((max(rows) - top + 1, max(cols) - left + 1), dtype=bool)
        colors = set()
        for row, col in cells:
            mask[row - top, col - left] = True
            colors.add(int(grid[row, col]))
        answer.append({"cells": cells, "bbox": (top, left, mask.shape[0], mask.shape[1]),
                       "mask": mask, "colors": colors, "holes": _holes(mask),
                       "count": len(_holes(mask))})
    return answer


def _background(grid):
    colors, frequencies = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(frequencies))])


def _pack(rng, masks, colors, background):
    order = sorted(range(len(masks)), key=lambda k: -masks[k].size)
    total_box = sum((m.shape[0] + 1) * (m.shape[1] + 1) for m in masks)
    longest = max(max(m.shape) for m in masks)
    minimum = max(longest + 2, int(np.ceil(np.sqrt(total_box * 1.5))))
    minimum = min(minimum, 30)
    for increase in range(4):
        base = min(30, minimum + 2 * increase)
        for _ in range(28):
            h = min(30, max(longest + 2, base + int(rng.integers(-3, 4))))
            w = min(30, max(longest + 2, base + int(rng.integers(-3, 4))))
            if sum(int(m.sum()) for m in masks) >= 0.46 * h * w:
                continue
            boxes = []
            positions = {}
            success = True
            for rank, index in enumerate(order):
                mh, mw = masks[index].shape
                if mh > h or mw > w:
                    success = False
                    break
                candidates = [(int(rng.integers(h - mh + 1)),
                               int(rng.integers(w - mw + 1))) for _ in range(160)]
                if rank == 0 and rng.random() < 0.45:
                    candidates.insert(0, (0 if rng.random() < 0.5 else h - mh,
                                          int(rng.integers(w - mw + 1))))
                for row, col in candidates:
                    if all(row + mh < r or r + bh < row or
                           col + mw < c or c + bw < col
                           for r, c, bh, bw in boxes):
                        boxes.append((row, col, mh, mw))
                        positions[index] = (row, col)
                        break
                else:
                    success = False
                    break
            if not success:
                continue
            grid = np.full((h, w), background, dtype=np.int8)
            for index, mask in enumerate(masks):
                row, col = positions[index]
                view = grid[row:row + mask.shape[0], col:col + mask.shape[1]]
                view[mask] = colors[index]
            if _background(grid) == background:
                return grid
    raise ValueError("could not place separated objects")


def _pair(rng, categories, params, anchor_counts=(), regime="source_train",
          forced_areas=None, size_witness=None):
    dense = len(categories) >= 7
    masks = [_make_object(rng, count, dense=dense,
                          anchor=index in anchor_counts,
                          holdout=regime == "target_test",
                          forced_area=(forced_areas or {}).get(index),
                          avoid_shape=(8, 8) if index == size_witness else None)
             for index, count in enumerate(categories)]
    output_colors = params["mapping"]
    available = params["input_palette"]
    background = int(available[int(rng.integers(len(available)))])
    palette = [color for color in available if color != background]
    multi = rng.random() < 0.43 and len(categories) >= 2
    if multi:
        chosen = [int(palette[int(rng.integers(len(palette)))])
                  for _ in categories]
        if len(set(chosen)) == 1:
            chosen[-1] = next(color for color in palette if color != chosen[0])
        if len(set(categories)) >= 2:
            first = next((j for j in range(1, len(categories))
                          if categories[j] != categories[0]), None)
            if first is not None:
                chosen[first] = chosen[0]
    else:
        source = int(palette[int(rng.integers(len(palette)))])
        chosen = [source] * len(categories)
    grid = _pack(rng, masks, chosen, background)
    output = np.full_like(grid, background)
    observed = _observations(grid, background)
    assert sorted(item["count"] for item in observed) == sorted(categories)
    for item in observed:
        for row, col in item["cells"]:
            output[row, col] = output_colors[item["count"]]
    return {"input": grid, "output": output}


def _signature(grid):
    return grid.shape, np.ascontiguousarray(grid).tobytes()


def _source_digest(pair):
    data = pair["input"].astype(np.uint8).tobytes()
    data += pair["output"].astype(np.uint8).tobytes()
    return hashlib.sha256(data).hexdigest()


def _learn_and_predict(train, test_input):
    mapping = {}
    for pair in train:
        grid, output = pair["input"], pair["output"]
        bg = _background(grid)
        for item in _observations(grid, bg):
            colors = {int(output[r, c]) for r, c in item["cells"]}
            assert len(colors) == 1
            color = next(iter(colors))
            if item["count"] in mapping:
                assert mapping[item["count"]] == color
            mapping[item["count"]] = color
    bg = _background(test_input)
    predicted = np.full_like(test_input, bg)
    for item in _observations(test_input, bg):
        assert item["count"] in mapping
        for row, col in item["cells"]:
            predicted[row, col] = mapping[item["count"]]
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    mapping = {}
    output_colors = set()
    all_backgrounds = set()
    all_sources = set()
    one_hole_areas, one_hole_sizes = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert np.issubdtype(array.dtype, np.integer)
                assert all(1 <= size <= 30 for size in array.shape)
                assert np.all((0 <= array) & (array <= 9))
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            signature = _signature(grid)
            assert signature not in seen
            seen.add(signature)
            assert _source_digest(pair) not in _SOURCE_DIGESTS
            bg = _background(grid)
            all_backgrounds.add(bg)
            assert np.array_equal(grid == bg, output == bg)
            objects = _observations(grid, bg)
            assert objects
            assert 1 <= len(objects) <= 10
            boxes = []
            for item in objects:
                assert 1 <= item["count"] <= 4
                assert len(item["colors"]) == 1
                all_sources.update(item["colors"])
                assert len(_holes(item["mask"], True)) == item["count"]
                if split == "train" and item["count"] == 1:
                    one_hole_areas.add(len(item["holes"][0]))
                    one_hole_sizes.add(item["mask"].shape)
                colors = {int(output[r, c]) for r, c in item["cells"]}
                assert len(colors) == 1
                color = next(iter(colors))
                assert color != bg and color not in item["colors"]
                if item["count"] in mapping:
                    assert mapping[item["count"]] == color
                elif split == "train":
                    mapping[item["count"]] = color
                else:
                    assert False, "test category absent from training"
                output_colors.add(color)
                boxes.append(item["bbox"])
            for i, (r, c, h, w) in enumerate(boxes):
                for rr, cc, hh, ww in boxes[i + 1:]:
                    assert r + h < rr or rr + hh < r or c + w < cc or cc + ww < c
            assert int(np.sum(grid == bg)) > grid.size // 2
    assert set(mapping) == {1, 2, 3, 4}
    assert len(set(mapping.values())) == 4
    assert not (output_colors & (all_backgrounds | all_sources))
    assert {2, 4} <= one_hole_areas
    assert len(one_hole_sizes) >= 2
    for pair in puzzle["test"]:
        assert np.array_equal(_learn_and_predict(puzzle["train"], pair["input"]),
                              pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    params = {"mapping": {k: palette[k - 1] for k in range(1, 5)},
              "input_palette": palette[4:]}
    puzzle = {"train": [], "test": []}
    seen = set()
    category_order = [int(x) for x in rng.permutation([1, 2, 3, 4])]
    first_size = int(rng.integers(1, 4))
    anchor_groups = (category_order[:first_size], category_order[first_size:])
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train" and index < 2:
                categories = list(anchor_groups[index])
                if index == 0 and len(categories) == 1:
                    categories.append(int(rng.choice(category_order[1:])))
                if index == 1 and len(categories) == 1:
                    categories.append(int(rng.choice(category_order[:first_size])))
                if index == 1:
                    categories.append(1)
                anchor_indices = tuple(range(len(anchor_groups[index])))
                forced_areas = {j: 2 for j in anchor_indices if categories[j] == 1}
                if index == 1:
                    forced_areas[len(categories) - 1] = 4
                    size_witness = len(categories) - 1
                else:
                    size_witness = None
            else:
                weights = (0.13, 0.15, 0.17, 0.18, 0.15, 0.10, 0.07, 0.05)
                total = int(rng.choice(np.arange(1, 9), p=weights))
                categories = [int(rng.integers(1, 5)) for _ in range(total)]
                anchor_indices = ()
                forced_areas = {}
                size_witness = None
            for _ in range(16):
                try:
                    pair = _pair(rng, categories, params, anchor_indices, split,
                                 forced_areas, size_witness)
                except ValueError:
                    continue
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not generate a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(first[split], second[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, masks, input_signatures = set(), set(), set(), set()
    backgrounds, source_colors, one_hole_sizes = set(), set(), set()
    counts_seen, cavity_areas = set(), set()
    small_scenes = dense_scenes = edge_scenes = multi_scenes = 0
    same_size_different_count = different_size_same_count = 0
    train_subset = test_subset = test_all_four = 0
    repeated_source_classes = 0
    changed_backgrounds = changed_sources = changed_dimensions = 0
    compact_many_holes = tall_narrow = 0
    outline_by_split = {"train": 0, "test": 0}
    large_by_split = {"train": 0, "test": 0}
    bent_by_split = {"train": 0, "test": 0}
    branched_by_split = {"train": 0, "test": 0}
    long_by_split = {"train": 0, "test": 0}
    deep_bays = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_backgrounds, puzzle_sources, puzzle_shapes = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg = _background(grid)
                observations = _observations(grid, bg)
                backgrounds.add(bg)
                puzzle_backgrounds.add(bg)
                shapes.add(grid.shape)
                puzzle_shapes.add(grid.shape)
                input_signatures.add(_signature(grid))
                small_scenes += len(observations) <= 2
                dense_scenes += len(observations) >= 7
                local_sources = {next(iter(o["colors"])) for o in observations}
                puzzle_sources.add(tuple(sorted(local_sources)))
                multi_scenes += len(local_sources) > 1
                categories = {o["count"] for o in observations}
                train_subset += split == "train" and len(categories) < 4
                test_subset += split == "test" and len(categories) < 4
                test_all_four += split == "test" and len(categories) == 4
                edge_scenes += any(r == 0 or c == 0 or r + h == grid.shape[0]
                                   or c + w == grid.shape[1]
                                   for r, c, h, w in (o["bbox"] for o in observations))
                by_size, by_count, by_source = {}, {}, {}
                layout = []
                for item in observations:
                    count = item["count"]
                    size = item["mask"].shape
                    by_size.setdefault(size, set()).add(count)
                    by_count.setdefault(count, set()).add(size)
                    by_source.setdefault(next(iter(item["colors"])), set()).add(count)
                    source_colors.update(item["colors"])
                    counts_seen.add(count)
                    cavity_areas.update(len(region) for region in item["holes"])
                    if count == 1:
                        one_hole_sizes.add(size)
                    compact_many_holes += count >= 3 and max(size) <= 6
                    tall_narrow += max(size) >= 9 and min(size) <= 5
                    exterior = [region for region in _components(~item["mask"])
                                if any(r in (0, size[0] - 1) or c in (0, size[1] - 1)
                                       for r, c in region)]
                    outline_by_split[split] += bool(exterior)
                    deep_bays += any(any(min(r, c, size[0] - 1 - r,
                                             size[1] - 1 - c) >= 2
                                         for r, c in region) for region in exterior)
                    for region in item["holes"]:
                        area = len(region)
                        rows = [r for r, _ in region]
                        cols = [c for _, c in region]
                        height = max(rows) - min(rows) + 1
                        width = max(cols) - min(cols) + 1
                        large_by_split[split] += area >= 6
                        bent_by_split[split] += area >= 3 and area < height * width
                        long_by_split[split] += max(height, width) >= 4 and min(height, width) <= 2
                        cells = set(region)
                        branched_by_split[split] += any(
                            sum((r + dr, c + dc) in cells
                                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))) >= 3
                            for r, c in region)
                    masks.add((size, item["mask"].tobytes()))
                    layout.append((item["bbox"], count))
                same_size_different_count += any(len(s) > 1 for s in by_size.values())
                different_size_same_count += any(len(s) > 1 for s in by_count.values())
                repeated_source_classes += any(len(s) > 1 for s in by_source.values())
                layouts.add((grid.shape, tuple(sorted(layout))))
        changed_backgrounds += len(puzzle_backgrounds) > 1
        changed_sources += len(puzzle_sources) > 1
        changed_dimensions += len(puzzle_shapes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert counts_seen == {1, 2, 3, 4}
    assert len(shapes) >= 25 and len(layouts) >= 100 and len(masks) >= 100
    assert len(input_signatures) >= 100
    assert len(backgrounds) >= 7 and len(source_colors) >= 7
    assert len(one_hole_sizes) >= 10 and max(cavity_areas) >= 6
    assert small_scenes >= 10 and dense_scenes >= 3
    assert edge_scenes >= 10 and multi_scenes >= 10
    assert same_size_different_count >= 3 and different_size_same_count >= 3
    assert train_subset >= 10 and test_subset >= 10 and test_all_four >= 2
    assert repeated_source_classes >= 20
    assert changed_backgrounds >= 20 and changed_sources >= 20
    assert changed_dimensions >= 20
    assert compact_many_holes >= 2 and tall_narrow >= 5 and deep_bays >= 5
    assert all(outline_by_split[split] >= 10 for split in ("train", "test"))
    assert all(large_by_split[split] >= 3 for split in ("train", "test"))
    assert all(bent_by_split[split] >= 5 for split in ("train", "test"))
    assert all(branched_by_split[split] >= 2 for split in ("train", "test"))
    assert all(long_by_split[split] >= 2 for split in ("train", "test"))
    assert len(shapes & {(17, 17), (19, 17), (17, 16), (17, 18)}) >= 2
    for seed, train, test in ((200, 2, 1), (201, 10, 1), (202, 10, 4),
                              (300, None, 1), (301, None, 4),
                              (302, 2, None), (303, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    first = generate_puzzle(12345, 6, 3)
    assert _equal(first, generate_puzzle(12345, 6, 3))
    assert not _equal(first, generate_puzzle(12346, 6, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {kwargs}")
