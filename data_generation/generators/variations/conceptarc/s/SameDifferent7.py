"""Generate ARC puzzles that retain the two translation-matching accent motifs."""

import numpy as np


def _normalize(cells):
    cells = tuple((int(y), int(x)) for y, x in cells)
    y0 = min(y for y, _ in cells)
    x0 = min(x for _, x in cells)
    return tuple(sorted((y - y0, x - x0) for y, x in cells))


def _connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        y, x = todo.pop()
        for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if point in cells and point not in seen:
                seen.add(point)
                todo.append(point)
    return len(seen) == len(cells)


def _random_shape(rng, height, width, area, excluded=()):
    excluded = set(excluded)
    for _ in range(800):
        start = (int(rng.integers(height)), int(rng.integers(width)))
        cells = {start}
        while len(cells) < area:
            frontier = {
                (ny, nx)
                for y, x in cells
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in cells
            }
            if not frontier:
                break
            cells.add(tuple(sorted(frontier)[int(rng.integers(len(frontier)))]))
        if len(cells) != area:
            continue
        shape = _normalize(cells)
        if (max(y for y, _ in shape) + 1 == height
                and max(x for _, x in shape) + 1 == width
                and shape not in excluded):
            return shape
    raise AssertionError("could not construct a connected motif")


def _motifs(rng, count, near_miss, winner_shape=None):
    sizes = ((3, 3), (3, 4), (4, 3), (4, 4), (3, 5), (5, 3))
    for _ in range(80):
        if winner_shape is None:
            height, width = sizes[int(rng.integers(len(sizes)))]
            area = int(rng.integers(max(height, width) + 1, min(height * width - 1, height + width + 3)))
        else:
            height = max(y for y, _ in winner_shape) + 1
            width = max(x for _, x in winner_shape) + 1
            area = len(winner_shape)
        try:
            winner = winner_shape if winner_shape is not None else _random_shape(rng, height, width, area)
            shapes = [winner, winner]
            if near_miss:
                shapes.append(_random_shape(rng, height, width, area, shapes))
            while len(shapes) < count:
                if rng.random() < 0.55:
                    h, w = height, width
                    a = int(rng.integers(max(h, w) + 1, min(h * w - 1, h + w + 3)))
                else:
                    h, w = sizes[int(rng.integers(len(sizes)))]
                    a = int(rng.integers(max(h, w) + 1, min(h * w - 1, h + w + 3)))
                shapes.append(_random_shape(rng, h, w, a, shapes))
            return shapes
        except AssertionError:
            continue
    raise AssertionError("motif construction exhausted")


def _local_object(rng, shape, preferred=None):
    mh = max(y for y, _ in shape) + 1
    mw = max(x for _, x in shape) + 1
    kinds = ("solid", "outline", "thick", "step", "taper", "asymmetric")
    for attempt in range(100):
        kind = preferred if attempt < 35 and preferred else kinds[int(rng.integers(len(kinds)))]
        height = mh + int(rng.integers(2 if kind in ("outline", "thick") else 1, 5))
        width = mw + int(rng.integers(2 if kind in ("outline", "thick") else 1, 5))
        if height > 9 or width > 9:
            continue
        mask = np.ones((height, width), dtype=bool)
        if kind == "outline":
            mask[1:-1, 1:-1] = False
        elif kind == "thick":
            if height >= 6 and width >= 6:
                mask[2:-2, 2:-2] = False
        elif kind == "step":
            for y in range(height):
                cut = min(2, y // max(2, height // 3))
                if rng.random() < 0.5:
                    mask[y, :cut] = False
                else:
                    mask[y, width - cut:] = False
        elif kind == "taper":
            for y in range(height):
                cut = min(2, y // max(2, height // 3))
                mask[y, :cut] = False
                mask[y, width - cut:] = False
        elif kind == "asymmetric":
            corner = int(rng.integers(4))
            depth = int(rng.integers(1, 3))
            for k in range(depth):
                if corner == 0:
                    mask[k, :depth - k] = False
                elif corner == 1:
                    mask[k, width - (depth - k):] = False
                elif corner == 2:
                    mask[height - 1 - k, :depth - k] = False
                else:
                    mask[height - 1 - k, width - (depth - k):] = False
        if kind in ("outline", "thick"):
            oy = int(rng.integers(1, height - mh))
            ox = int(rng.integers(1, width - mw))
        else:
            oy = int(rng.integers(0, height - mh + 1))
            ox = int(rng.integers(0, width - mw + 1))
        motif = {(oy + y, ox + x) for y, x in shape}
        frame = {(y, x) for y, x in np.argwhere(mask)} - motif
        if len(frame) < len(motif) + 3 or not _connected(frame):
            continue
        if (min(y for y, _ in frame) != 0 or max(y for y, _ in frame) != height - 1
                or min(x for _, x in frame) != 0 or max(x for _, x in frame) != width - 1):
            continue
        if not all(0 <= y < height and 0 <= x < width for y, x in motif):
            continue
        return {"frame": frame, "motif": motif, "height": height, "width": width,
                "kind": kind, "edge": any(y in (0, height - 1) or x in (0, width - 1)
                                           for y, x in motif)}
    raise AssertionError("carrier construction exhausted")


def _touches(a, b):
    ay, ax = a["y"], a["x"]
    by, bx = b["y"], b["x"]
    ah, aw = a["height"], a["width"]
    bh, bw = b["height"], b["width"]
    return ((ax + aw == bx or bx + bw == ax) and ay < by + bh and by < ay + ah) or \
           ((ay + ah == by or by + bh == ay) and ax < bx + bw and bx < ax + aw)


def _place(rng, objects, height, width, want_touch):
    placed = []
    for obj in sorted(objects, key=lambda o: o["height"] * o["width"], reverse=True):
        candidates = []
        for y in range(height - obj["height"] + 1):
            for x in range(width - obj["width"] + 1):
                trial = dict(obj, y=y, x=x)
                if any(y < q["y"] + q["height"] and q["y"] < y + obj["height"]
                       and x < q["x"] + q["width"] and q["x"] < x + obj["width"]
                       for q in placed):
                    continue
                candidates.append(trial)
        if not candidates:
            return None
        touching = [c for c in candidates if any(_touches(c, q) for q in placed)]
        separated = [c for c in candidates if not any(_touches(c, q) for q in placed)]
        pool = touching if want_touch and touching and placed else separated if separated else candidates
        # Prefer multiple spatial arrangements instead of partitioning the canvas into slots.
        placed.append(pool[int(rng.integers(len(pool)))])
    return placed


def _components(grid, background):
    seen = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    result = []
    for sy, sx in np.argwhere(grid != background):
        sy, sx = int(sy), int(sx)
        if seen[sy, sx]:
            continue
        color = int(grid[sy, sx])
        cells = []
        todo = [(sy, sx)]
        seen[sy, sx] = True
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and not seen[ny, nx] and grid[ny, nx] == color:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        result.append({"color": color, "cells": cells,
                       "box": (min(y for y, _ in cells), max(y for y, _ in cells),
                               min(x for _, x in cells), max(x for _, x in cells))})
    return result


def _visible_objects(grid):
    background = int(grid[0, 0])
    components = _components(grid, background)
    objects = []
    used = set()
    for i, parent in enumerate(components):
        py0, py1, px0, px1 = parent["box"]
        children = []
        for j, child in enumerate(components):
            if i == j or len(child["cells"]) >= len(parent["cells"]):
                continue
            cy0, cy1, cx0, cx1 = child["box"]
            if py0 <= cy0 <= cy1 <= py1 and px0 <= cx0 <= cx1 <= px1:
                children.append(j)
        if len(children) == 1:
            j = children[0]
            child = components[j]
            objects.append({"frame": parent["color"], "accent": child["color"],
                            "box": parent["box"], "cells": tuple(child["cells"]),
                            "shape": _normalize(child["cells"])})
            used.update((i, j))
    if len(used) != len(components) or len(objects) * 2 != len(components):
        return []
    return objects


def _solve_visible(grid):
    objects = _visible_objects(grid)
    by_shape = {}
    for obj in objects:
        by_shape.setdefault(obj["shape"], []).append(obj)
    winners = [group for group in by_shape.values() if len(group) == 2]
    if len(winners) != 1 or any(len(group) > 2 for group in by_shape.values()):
        return None
    result = np.full_like(grid, grid[0, 0])
    for obj in winners[0]:
        for y, x in obj["cells"]:
            result[y, x] = obj["accent"]
    return result


def _matching_shape(grid):
    objects = _visible_objects(grid)
    return next(o["shape"] for o in objects
                if sum(q["shape"] == o["shape"] for q in objects) == 2)


def _row_strings(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


_SOURCE_INPUTS = set()
for _rows in (
    "000000000000 004440666600 004340686666 003330888666 004340686600 004440666600 033330077777 032230074477 032230074777 033330074470 000000077770 000000000000",
    "000000099999 000000094449 011110094499 016610094999 016111094449 016611099999 011110003333 000077703113 000773303133 000773703113 000773303133 000077703113",
    "000000000000 000000660000 005555666600 555445633600 555445633600 555445666600 555555000000 000000077770 000000078877 000000078877 000000078877 000000077777",
    "000000000000000 011110004440000 013310004640000 013310004640000 011110004640000 000000333333000 000000333333000 000000331133000 000000331133000 000000333333000 000000333333000 000444000000000 000434000000000 000444000000000",
    "444444433333 004111432223 004144033323 004140033323 004140033333 004440000000 888811111000 666817771000 686817111000 688017110000 880017110000 000011110000",
    "00000000006660 06666666006460 06444446006460 00644460006460 00064600006660 00006000000000 00000000000000 33433000555555 33433000577777 33433000557775 33433000555755 33433000555555 03330000000000 00300000000000",
):
    _SOURCE_INPUTS.add(tuple(_rows.split()))


def _make_pair(rng, regime, index, familiar_shape=None):
    for attempt in range(150):
        count = 4 if regime == "train" and index == 0 else int(rng.integers(3, 7))
        near = regime == "train" and index == 0 or rng.random() < (0.8 if regime == "test" else 0.6)
        shapes = _motifs(rng, count, near, familiar_shape)
        preferred = ("step" if index == 0 else "outline" if index == 1 else None) \
                    if regime == "train" else None
        objects = [_local_object(rng, shape, preferred if i == 0 else None)
                   for i, shape in enumerate(shapes)]
        for obj, shape in zip(objects, shapes):
            obj["shape"] = shape
        min_h = max(max(o["height"] for o in objects), 11 + (count - 3) * 2)
        min_w = max(max(o["width"] for o in objects), 11 + (count - 3) * 2)
        height = int(rng.integers(min_h, 31))
        width = int(rng.integers(min_w, 31))
        want_touch = (index == 0 if index < 2 else rng.random() < 0.4) \
                     if regime == "train" else rng.random() < 0.5
        placed = _place(rng, objects, height, width, want_touch)
        if placed is None:
            continue
        if regime == "train" and index == 1 and any(
            _touches(a, b) for i, a in enumerate(placed) for b in placed[i + 1:]
        ):
            continue
        rng.shuffle(placed)
        winner_indices = [i for i, obj in enumerate(placed) if obj["shape"] == shapes[0]]
        # The corner identifies the background even when structures touch the boundary.
        if any(o["y"] == 0 and o["x"] == 0 for o in placed):
            continue
        background = 0 if regime == "train" and index == 0 else int(rng.integers(10))
        colors = [c for c in range(10) if c != background]
        frame_colors = [int(v) for v in rng.choice(colors, size=count, replace=False)]
        options = [(i, j) for i in range(count) for j in range(i)
                   if not _touches(placed[i], placed[j])]
        if regime == "train" and index == 1 and not options:
            continue
        if regime == "train" and index == 1 or rng.random() < 0.55:
            if options:
                i, j = options[int(rng.integers(len(options)))]
                frame_colors[i] = frame_colors[j]
        accent_colors = []
        same_winners = regime == "train" and index > 0 and rng.random() < 0.35 or \
                       regime == "test" and rng.random() < 0.25
        for i in range(count):
            options = [c for c in colors if c != frame_colors[i]]
            if regime == "train" and index == 0 and i == winner_indices[0]:
                reused = [frame_colors[j] for j in range(count)
                          if j != i and not _touches(placed[i], placed[j])
                          and frame_colors[j] in options]
                accent_colors.append(int(rng.choice(reused if reused else options)))
            elif regime == "train" and index == 1 and i == winner_indices[0]:
                independent = [c for c in options if c not in frame_colors]
                accent_colors.append(int(rng.choice(independent)))
            elif i == winner_indices[1] and same_winners and accent_colors[winner_indices[0]] in options:
                accent_colors.append(accent_colors[winner_indices[0]])
            elif rng.random() < 0.55:
                reused = [c for c in options if c in frame_colors]
                accent_colors.append(int(rng.choice(reused if reused else options)))
            else:
                accent_colors.append(int(rng.choice(options)))
        if regime == "train" and index == 0:
            if accent_colors[winner_indices[0]] == accent_colors[winner_indices[1]]:
                continue
            if not any(accent_colors[i] == frame_colors[j]
                       for i in winner_indices for j in range(count) if i != j):
                continue
        if regime == "train" and index == 1:
            if len(set(frame_colors)) == count:
                continue
            if accent_colors[winner_indices[0]] in frame_colors:
                continue
        grid = np.full((height, width), background, dtype=np.int8)
        for i, obj in enumerate(placed):
            for y, x in obj["frame"]:
                grid[obj["y"] + y, obj["x"] + x] = frame_colors[i]
            for y, x in obj["motif"]:
                grid[obj["y"] + y, obj["x"] + x] = accent_colors[i]
        visible = _visible_objects(grid)
        if len(visible) != count:
            continue
        if sorted(o["shape"] for o in visible) != sorted(shapes):
            continue
        if regime == "train" and index == 0:
            if not any(any(y == o["box"][0] or y == o["box"][1] or
                           x == o["box"][2] or x == o["box"][3]
                           for y, x in o["cells"]) for o in visible):
                continue
            components = _components(grid, background)
            if not any(len(c["cells"]) < (c["box"][1] - c["box"][0] + 1) *
                       (c["box"][3] - c["box"][2] + 1)
                       for c in components if len(c["cells"]) > 12):
                continue
        if regime == "train" and index == 1:
            if not any(obj["kind"] == "outline" for obj in placed):
                continue
        result = _solve_visible(grid)
        if result is None or _row_strings(grid) in _SOURCE_INPUTS:
            continue
        return {"input": grid, "output": result}
    raise AssertionError("pair construction exhausted")


def _check_count(value, name, minimum, maximum):
    if value is not None and (isinstance(value, (bool, np.bool_)) or
                              not isinstance(value, (int, np.integer)) or
                              not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    if num_train is None:
        low = max(2, 1 + (num_test if num_test is not None else 0))
        num_train = int(rng.integers(low, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    puzzle = {"train": [], "test": []}
    seen = set()
    training_shapes = []
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            familiar = None
            if regime == "test" and index > 0 and rng.random() < 0.55:
                familiar = training_shapes[int(rng.integers(len(training_shapes)))]
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, regime, index, familiar)
                key = _row_strings(pair["input"])
                if regime == "test" and familiar is None:
                    objects = _visible_objects(pair["input"])
                    winner = next(o["shape"] for o in objects
                                  if sum(q["shape"] == o["shape"] for q in objects) == 2)
                    if winner in training_shapes:
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    if regime == "train":
                        training_shapes.append(_matching_shape(pair["input"]))
                    break
            else:
                raise AssertionError("duplicate input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for a in (grid, output):
                assert isinstance(a, np.ndarray) and a.ndim == 2
                assert 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
                assert np.issubdtype(a.dtype, np.integer)
                assert np.all((0 <= a) & (a <= 9))
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            assert np.all((output == grid[0, 0]) | (output == grid))
            key = _row_strings(grid)
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            objects = _visible_objects(grid)
            assert 3 <= len(objects) <= 6
            assert len({o["shape"] for o in objects}) == len(objects) - 1
            assert np.array_equal(_solve_visible(grid), output)
    # The first demonstration includes a near miss in a common bounding box.
    first = _visible_objects(puzzle["train"][0]["input"])
    winners = [o for o in first if sum(p["shape"] == o["shape"] for p in first) == 2]
    assert len(winners) == 2
    h = max(y for y, _ in winners[0]["shape"]) + 1
    w = max(x for _, x in winners[0]["shape"]) + 1
    area = len(winners[0]["shape"])
    assert any(o["shape"] != winners[0]["shape"] and
               max(y for y, _ in o["shape"]) + 1 == h and
               max(x for _, x in o["shape"]) + 1 == w and
               len(o["shape"]) == area for o in first)
    def touches(objects):
        boxes = [{"y": o["box"][0], "x": o["box"][2],
                  "height": o["box"][1] - o["box"][0] + 1,
                  "width": o["box"][3] - o["box"][2] + 1} for o in objects]
        return any(_touches(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:])

    assert touches(first)
    assert winners[0]["accent"] != winners[1]["accent"]
    assert any(o["accent"] in {p["frame"] for p in first} for o in winners)
    assert any(any(y in (o["box"][0], o["box"][1]) or
                   x in (o["box"][2], o["box"][3]) for y, x in o["cells"])
               for o in first)

    second_grid = puzzle["train"][1]["input"]
    second = _visible_objects(second_grid)
    assert not touches(second)
    assert len({o["frame"] for o in second}) < len(second)
    second_winner = next(o["shape"] for o in second
                         if sum(p["shape"] == o["shape"] for p in second) == 2)
    assert any(o["accent"] not in {p["frame"] for p in second}
               for o in second if o["shape"] == second_winner)
    background = int(second_grid[0, 0])
    assert any(
        np.all((second_grid[y0:y1 + 1, x0:x1 + 1] != background)[0, :])
        and np.all((second_grid[y0:y1 + 1, x0:x1 + 1] != background)[-1, :])
        and np.all((second_grid[y0:y1 + 1, x0:x1 + 1] != background)[:, 0])
        and np.all((second_grid[y0:y1 + 1, x0:x1 + 1] != background)[:, -1])
        and not np.all(second_grid[y0:y1 + 1, x0:x1 + 1] != background)
        for y0, y1, x0, x1 in (o["box"] for o in second)
    )
    training_winners = {_matching_shape(pair["input"]) for pair in puzzle["train"]}
    assert any(_matching_shape(pair["input"]) not in training_winners
               for pair in puzzle["test"])


def _same_puzzle(a, b):
    return all(np.array_equal(x[k], y[k])
               for regime in ("train", "test")
               for x, y in zip(a[regime], b[regime])
               for k in ("input", "output")) and \
           all(len(a[r]) == len(b[r]) for r in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, shapes, backgrounds = set(), set(), set(), set(), set()
    counts, touches, near_misses, edge_contacts = set(), 0, 0, 0
    familiar_tests, novel_tests = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        learned_shapes = {_matching_shape(pair["input"]) for pair in puzzle["train"]}
        for pair in puzzle["test"]:
            if _matching_shape(pair["input"]) in learned_shapes:
                familiar_tests += 1
            else:
                novel_tests += 1
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid = pair["input"]
                objects = _visible_objects(grid)
                sizes.add(grid.shape)
                backgrounds.add(int(grid[0, 0]))
                counts.add(len(objects))
                shapes.update(o["shape"] for o in objects)
                touches += any(_touches({"y": a["box"][0], "x": a["box"][2],
                                        "height": a["box"][1] - a["box"][0] + 1,
                                        "width": a["box"][3] - a["box"][2] + 1},
                                       {"y": b["box"][0], "x": b["box"][2],
                                        "height": b["box"][1] - b["box"][0] + 1,
                                        "width": b["box"][3] - b["box"][2] + 1})
                               for i, a in enumerate(objects) for b in objects[i + 1:])
                repeated = next(o["shape"] for o in objects
                                if sum(p["shape"] == o["shape"] for p in objects) == 2)
                near_misses += any(o["shape"] != repeated and
                                   max(y for y, _ in o["shape"]) == max(y for y, _ in repeated) and
                                   max(x for _, x in o["shape"]) == max(x for _, x in repeated) and
                                   len(o["shape"]) == len(repeated) for o in objects)
                edge_contacts += any(any(y == o["box"][0] or y == o["box"][1] or
                                         x == o["box"][2] or x == o["box"][3]
                                         for y, x in o["cells"]) for o in objects)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 30 and len(shapes) >= 40 and len(backgrounds) >= 5
    assert counts == {3, 4, 5, 6}
    assert touches >= 10 and near_misses >= 40 and edge_contacts >= 10
    assert familiar_tests >= 5 and novel_tests >= 30
    for nt, ne in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        for seed in (0, 7, 19):
            puzzle = generate_puzzle(seed, nt, ne)
            validate_generated_puzzle(puzzle)
            assert _same_puzzle(puzzle, generate_puzzle(seed, nt, ne))
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(123, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError((nt, ne))
