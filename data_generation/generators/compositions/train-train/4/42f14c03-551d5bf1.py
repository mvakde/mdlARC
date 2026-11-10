"""Shape-locked outlets: oriented fitting gates spatially addressed color rays.

Only standard-library modules and NumPy are used. No parent code or data is loaded.
"""

import json

import numpy as np


DIRECTIONS = ("U", "D", "L", "R")
DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dr, dc in DELTAS.values():
                y, x = row + dr, col + dc
                if 0 <= y < height and 0 <= x < width and mask[y, x] and not seen[y, x]:
                    seen[y, x] = True
                    stack.append((y, x))
        result.append(cells)
    return result


def _box_cells(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _separate(a, b):
    r0, r1, c0, c1 = a
    y0, y1, x0, x1 = b
    return r1 < y0 or y1 < r0 or c1 < x0 or x1 < c0


def _ray(gap, direction, shape):
    dr, dc = DELTAS[direction]
    r, c = gap
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r, c = r + dr, c + dc
    return cells


def _parse(input_grid):
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("expected an integer 2D grid")
    if not all(1 <= n <= 30 for n in grid.shape) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("illegal grid dimensions or colors")
    bg = int(grid[0, 0])
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if np.any(border != bg):
        raise ValueError("input must have a uniform background border")
    objects, frames, tiles, owner = [], [], [], {}
    for cells in _components(grid != bg):
        rows, cols = zip(*cells)
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1
        color = int(grid[cells[0]])
        if any(int(grid[p]) != color for p in cells):
            raise ValueError("objects must be separated and monochrome")
        box = (r0, r1, c0, c1)
        obj = {"box": box, "shape": (r1 - r0, c1 - c0), "color": color,
               "cells": set(cells), "kind": "tile"}
        if len(cells) == (r1 - r0) * (c1 - c0):
            tiles.append(obj)
        else:
            if r1 - r0 < 3 or c1 - c0 < 3:
                raise ValueError("a frame must have a nonempty interior")
            perimeter = {(r, c) for r, c in _box_cells(box)
                         if r in (r0, r1 - 1) or c in (c0, c1 - 1)}
            missing = perimeter - obj["cells"]
            if not obj["cells"].issubset(perimeter) or len(missing) > 1:
                raise ValueError("expected a complete or singly opened frame")
            if np.any(grid[r0 + 1:r1 - 1, c0 + 1:c1 - 1] != bg):
                raise ValueError("frame interiors must be empty")
            obj.update(kind="frame", gap=None, direction=None,
                       interior=(r1 - r0 - 2, c1 - c0 - 2))
            if missing:
                gap = next(iter(missing))
                r, c = gap
                if (r in (r0, r1 - 1)) == (c in (c0, c1 - 1)):
                    raise ValueError("a gap must not be a corner")
                direction = "U" if r == r0 else "D" if r == r1 - 1 else "L" if c == c0 else "R"
                obj.update(gap=gap, direction=direction)
            frames.append(obj)
        objects.append(obj)
        for p in cells:
            owner[p] = obj
    if not frames or not tiles:
        raise ValueError("expected frames and rectangular keys")
    outline_colors = {f["color"] for f in frames}
    if len(outline_colors) != 1 or any(t["color"] in outline_colors for t in tiles):
        raise ValueError("outline and key colors must be distinct roles")
    for i, obj in enumerate(objects):
        if any(not _separate(obj["box"], other["box"]) for other in objects[i + 1:]):
            raise ValueError("object bounding boxes need background clearance")
    for frame in frames:
        ray = [] if frame["gap"] is None else _ray(frame["gap"], frame["direction"], grid.shape)
        hit = next((owner[p] for p in ray[1:] if p in owner), None)
        active = hit is not None and hit["kind"] == "tile" and hit["shape"] == frame["interior"]
        frame.update(ray=ray, hit=hit, active=active)
    return grid, bg, objects, frames, tiles


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every decision from the input, with no generation metadata."""
    grid, bg, _, frames, _ = _parse(input_grid)
    output = grid.copy()
    painted = {}
    for frame in frames:
        if not frame["active"]:
            continue
        key = frame["hit"]
        color = key["color"]
        r0, r1, c0, c1 = frame["box"]
        cells = frame["ray"] + [(r, c) for r in range(r0 + 1, r1 - 1)
                                  for c in range(c0 + 1, c1 - 1)]
        if any(grid[p] != bg and p not in key["cells"] for p in frame["ray"]):
            raise ValueError("successful continuation must be clear beyond its key")
        for p in cells:
            if p in painted and painted[p] != color:
                raise ValueError("differently colored successful regions overlap")
            painted[p] = color
            output[p] = color
    return output


def _shape(rng, status, large=False):
    if large:
        shape = (int(rng.integers(5, 7)), int(rng.integers(1, 4)))
        return shape if int(rng.integers(2)) else shape[::-1]
    options = [(h, w) for h in range(1, 5) for w in range(1, 5)
               if h * w <= 12 and (status != "blocked" or h != w)]
    return options[int(rng.integers(len(options)))]


def _intersects(box, cells):
    r0, r1, c0, c1 = box
    return any(r0 <= r < r1 and c0 <= c < c1 for r, c in cells)


def _valid_box(box, boxes, lanes, shape):
    r0, r1, c0, c1 = box
    return (1 <= r0 < r1 < shape[0] and 1 <= c0 < c1 < shape[1]
            and all(_separate(box, other) for other in boxes)
            and all(not _intersects(box, lane) for lane in lanes))


def _tile_on_line(rng, gap, direction, tile_shape, distance, canvas):
    h, w = tile_shape
    r, c = gap
    if direction in ("U", "D"):
        low, high = max(1, c - w + 1), min(c, canvas[1] - w - 1)
        if low > high:
            return None
        c0 = int(rng.integers(low, high + 1))
        r0 = r + distance if direction == "D" else r - distance - h + 1
    else:
        low, high = max(1, r - h + 1), min(r, canvas[0] - h - 1)
        if low > high:
            return None
        r0 = int(rng.integers(low, high + 1))
        c0 = c + distance if direction == "R" else c - distance - w + 1
    return r0, r0 + h, c0, c0 + w


def _group(rng, item, boxes, lanes, beams, canvas):
    status, color = item["status"], item["color"]
    ih, iw = item["shape"]
    h, w = ih + 2, iw + 2
    for _ in range(450):
        r0 = int(rng.integers(1, canvas[0] - h))
        c0 = int(rng.integers(1, canvas[1] - w))
        frame_box = (r0, r0 + h, c0, c0 + w)
        if not _valid_box(frame_box, boxes, lanes, canvas):
            continue
        direction, gap, lane = item["direction"], None, []
        if status != "closed":
            if direction in ("U", "D"):
                gap = (r0 if direction == "U" else r0 + h - 1,
                       int(rng.integers(c0 + 1, c0 + w - 1)))
            else:
                gap = (int(rng.integers(r0 + 1, r0 + h - 1)),
                       c0 if direction == "L" else c0 + w - 1)
            lane = _ray(gap, direction, canvas)
            if any(_intersects(box, lane) for box in boxes):
                continue
            if status == "active" and any(set(lane) & old for old in beams):
                continue
        own_boxes, donors = [frame_box], []
        if status in ("active", "blocked"):
            shapes = [(ih, iw)] if status == "active" else [item["wrong_shape"], (ih, iw)]
            lengths = [s[0] if direction in ("U", "D") else s[1] for s in shapes]
            max_t = len(lane) - 2  # Last donor cell stays inside the blank border.
            last_start = max_t - sum(lengths) - (len(shapes) - 1) + 1
            if last_start < 2:
                continue
            distance = int(rng.integers(2, last_start + 1))
            for index, (tile_shape, length) in enumerate(zip(shapes, lengths)):
                if index:
                    minimum = distance + lengths[index - 1] + 1
                    distance = int(rng.integers(minimum, max_t - length + 2))
                tile_box = _tile_on_line(rng, gap, direction, tile_shape, distance, canvas)
                if tile_box is None or not _valid_box(tile_box, boxes + own_boxes, lanes, canvas):
                    break
                own_boxes.append(tile_box)
                donors.append((tile_box, color if index == 0 else item["far_color"]))
            if len(donors) != len(shapes):
                continue
        else:
            # A fitting but off-axis key is an explicit counterexample to global matching.
            for _ in range(80):
                tr = int(rng.integers(1, canvas[0] - ih))
                tc = int(rng.integers(1, canvas[1] - iw))
                tile_box = (tr, tr + ih, tc, tc + iw)
                if _valid_box(tile_box, boxes + own_boxes, lanes + [lane], canvas):
                    own_boxes.append(tile_box)
                    donors.append((tile_box, color))
                    break
            if not donors:
                continue
        return {"box": frame_box, "gap": gap, "direction": direction,
                "status": status, "color": color, "donors": donors, "lane": lane}
    return None


def _make_pair(rng, statuses, directions, palette, large=False, force_transpose=False):
    bg, outline, colors = palette
    for attempt in range(50):
        canvas = (int(rng.integers(23, 31)), int(rng.integers(23, 31)))
        items = []
        active_index = 0
        for status, direction in zip(statuses, directions):
            shape = _shape(rng, status, large and status == "active" and active_index == 0)
            color = int(colors[(active_index + attempt) % len(colors)]) if status == "active" else int(rng.choice(colors))
            item = {"status": status, "direction": direction, "shape": shape, "color": color,
                    "far_color": int(rng.choice(colors))}
            if status == "active":
                active_index += 1
            if status == "blocked":
                if force_transpose or rng.random() < 0.65:
                    wrong = shape[::-1]
                else:
                    alternatives = [(a, b) for a in range(1, 5) for b in range(1, 5)
                                    if a * b <= 12 and (a, b) != shape]
                    wrong = alternatives[int(rng.integers(len(alternatives)))]
                item["wrong_shape"] = wrong
            items.append(item)
        # Solve the most space-constrained groups first; positions themselves are free.
        order = sorted(range(len(items)), key=lambda i: (
            items[i]["status"] != "blocked", items[i]["status"] != "active",
            -np.prod(items[i]["shape"]), float(rng.random())))
        boxes, lanes, beams, groups = [], [], [], []
        for i in order:
            group = _group(rng, items[i], boxes, lanes, beams, canvas)
            if group is None:
                break
            groups.append(group)
            boxes.extend([group["box"]] + [box for box, _ in group["donors"]])
            if group["lane"]:
                lanes.append(group["lane"])
            if group["status"] == "active":
                beams.append(set(group["lane"]))
        if len(groups) != len(items):
            continue
        inp = np.full(canvas, bg, dtype=np.int8)
        for group in groups:
            r0, r1, c0, c1 = group["box"]
            inp[r0, c0:c1] = inp[r1 - 1, c0:c1] = outline
            inp[r0:r1, c0] = inp[r0:r1, c1 - 1] = outline
            if group["gap"] is not None:
                inp[group["gap"]] = bg
            for (a, b, x, y), color in group["donors"]:
                inp[a:b, x:y] = color
        # Construct the answer using sampled group semantics, independently of solve().
        out = inp.copy()
        for group in groups:
            if group["status"] == "active":
                r0, r1, c0, c1 = group["box"]
                out[r0 + 1:r1 - 1, c0 + 1:c1 - 1] = group["color"]
                for p in group["lane"]:
                    out[p] = group["color"]
        return {"input": inp, "output": out}
    raise RuntimeError("bounded geometric packing exhausted")


def _signature(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _inspect_pair(pair):
    inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    grid, bg, objects, frames, tiles = _parse(inp)
    assert inp.shape == out.shape
    assert np.array_equal(solve(inp), out)
    assert np.array_equal(inp[inp != bg], out[inp != bg]), "original objects must remain"
    assert set(map(int, np.unique(out))).issubset(set(map(int, np.unique(inp))))
    expected_changed = set()
    directions, active_colors, statuses = set(), set(), set()
    transpose_witness = hidden_match = False
    ignore_a = ignore_b = False
    for frame in frames:
        r0, r1, c0, c1 = frame["box"]
        interior = out[r0 + 1:r1 - 1, c0 + 1:c1 - 1]
        hit = frame["hit"]
        if frame["active"]:
            statuses.add("active")
            directions.add(frame["direction"])
            active_colors.add(hit["color"])
            assert np.all(interior == hit["color"])
            assert all(out[p] == hit["color"] for p in frame["ray"])
            assert frame["ray"][-1][0] in (0, inp.shape[0] - 1) or frame["ray"][-1][1] in (0, inp.shape[1] - 1)
            expected_changed.update((r, c) for r in range(r0 + 1, r1 - 1) for c in range(c0 + 1, c1 - 1))
            expected_changed.update(p for p in frame["ray"] if inp[p] == bg)
        else:
            assert np.all(interior == bg)
            if frame["gap"] is None:
                statuses.add("closed")
            elif hit is None:
                statuses.add("empty")
                assert out[frame["gap"]] == bg
            else:
                statuses.add("blocked")
                assert out[frame["gap"]] == bg
                if hit["kind"] == "tile":
                    ignore_a = True
                    transpose_witness |= (hit["shape"] == frame["interior"][::-1]
                                          and hit["shape"] != frame["interior"])
                    hit_positions = [i for i, p in enumerate(frame["ray"]) if p in hit["cells"]]
                    hidden_match |= any(t["shape"] == frame["interior"]
                                        and any(p in t["cells"] for p in frame["ray"][max(hit_positions) + 1:])
                                        for t in tiles)
            ignore_b |= any(t["shape"] == frame["interior"] for t in tiles)
    assert expected_changed == {tuple(map(int, p)) for p in np.argwhere(inp != out)}
    assert active_colors, "each generated scene must show at least one activation"
    return {"bg": bg, "outline": frames[0]["color"], "frames": frames, "tiles": tiles,
            "statuses": statuses, "directions": directions, "active_colors": active_colors,
            "transpose_witness": transpose_witness, "hidden_match": hidden_match,
            "ignore_a": ignore_a, "ignore_b": ignore_b}


def validate_generated_puzzle(puzzle):
    """Check returned grids and teaching evidence, without generation metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, roles = set(), set()
    teaching_directions, teaching_statuses, teaching_colors = set(), set(), set()
    transposed = hidden = ignore_a = ignore_b = False
    training_infos, test_infos = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                grid = np.asarray(value)
                assert grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _signature(pair["input"])
            assert key not in seen
            seen.add(key)
            info = _inspect_pair(pair)
            (training_infos if split == "train" else test_infos).append(info)
            roles.add((info["bg"], info["outline"]))
            if split == "train":
                teaching_directions.update(info["directions"])
                teaching_statuses.update(info["statuses"])
                teaching_colors.update(info["active_colors"])
                transposed |= info["transpose_witness"]
                hidden |= info["hidden_match"]
                ignore_a |= info["ignore_a"]
                ignore_b |= info["ignore_b"]
    assert len(roles) == 1
    assert teaching_directions == set(DIRECTIONS)
    assert teaching_statuses == {"active", "blocked", "closed", "empty"}
    assert len(teaching_colors) >= 2
    assert transposed and hidden and ignore_a and ignore_b
    all_statuses = {"active", "blocked", "closed", "empty"}
    training_combined = any(x["statuses"] == all_statuses for x in training_infos)
    training_maximum = max(max(f["interior"]) for x in training_infos for f in x["frames"])
    def repeated(info):
        dirs = [f["direction"] for f in info["frames"] if f["active"]]
        return len(dirs) > len(set(dirs))
    training_repeated = any(repeated(x) for x in training_infos)
    for info in test_infos:
        assert ((not training_combined and info["statuses"] == all_statuses)
                or any(f["active"] and max(f["interior"]) > training_maximum for f in info["frames"])
                or (not training_repeated and repeated(info))), "test must contain an evidenced structural shift"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    colors = [int(x) for x in rng.permutation(10)]
    palette = (colors[0], colors[1], colors[2:2 + int(rng.integers(2, 6))])
    mode = int(rng.integers(3))
    directions = [str(x) for x in rng.permutation(DIRECTIONS)]
    plans = [(["active", "active", "blocked"], directions[:2] + [str(rng.choice(DIRECTIONS))], False, True),
             (["active", "active", "closed", "empty"], directions[2:] + [None, str(rng.choice(DIRECTIONS))], False, False)]
    for _ in range(num_train - 2):
        count = int(rng.integers(2, 6))
        statuses = ["active"] + [str(rng.choice(("active", "blocked", "closed", "empty"))) for _ in range(count - 1)]
        if mode == 0 and set(statuses) == {"active", "blocked", "closed", "empty"}:
            statuses = ["active" if s == "empty" else s for s in statuses]
        if mode == 2 and statuses.count("active") > 4:
            statuses[-1] = "empty"
        dirs = [None if s == "closed" else str(rng.choice(DIRECTIONS)) for s in statuses]
        if mode == 2:
            unique = iter(str(x) for x in rng.permutation(DIRECTIONS))
            dirs = [next(unique) if s == "active" else d for s, d in zip(statuses, dirs)]
        plans.append((statuses, dirs, bool(rng.random() < 0.18) and mode != 1, False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                statuses, dirs, large, transpose = plans[index]
            elif mode == 0:
                statuses = ["active", "blocked", "closed", "empty"]
                if rng.random() < 0.35:
                    statuses.append("active")
                dirs = [None if s == "closed" else str(rng.choice(DIRECTIONS)) for s in statuses]
                large = transpose = False
            elif mode == 1:
                statuses = ["active", str(rng.choice(("blocked", "closed", "empty")))]
                if rng.random() < 0.5:
                    statuses.append(str(rng.choice(("active", "closed", "empty"))))
                dirs = [None if s == "closed" else str(rng.choice(DIRECTIONS)) for s in statuses]
                large, transpose = True, False
            else:
                statuses = ["active", "active", str(rng.choice(("blocked", "closed", "empty")))]
                if rng.random() < 0.4:
                    statuses.append(str(rng.choice(("closed", "empty"))))
                repeated = str(rng.choice(DIRECTIONS))
                dirs = [repeated, repeated] + [None if s == "closed" else str(rng.choice(DIRECTIONS)) for s in statuses[2:]]
                large = transpose = False
            for _ in range(10):
                pair = _make_pair(rng, statuses, dirs, palette, large, transpose)
                key = _signature(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not produce distinct examples")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def _regression_tests():
    # Hand-built example: a square fitting key hidden by an unequal-area blocker.
    inp = np.zeros((12, 20), dtype=np.int8)
    inp[3, 2:7] = inp[7, 2:7] = 1
    inp[3:8, 2] = inp[3:8, 6] = 1
    inp[5, 6] = 0  # Interior 3x3; use an unequal-area blocker here.
    inp[5:7, 9:12] = 4
    inp[4:7, 15:18] = 7
    assert np.array_equal(solve(inp), inp), "must not skip the first wrong key"
    correct = inp.copy()
    correct[5:7, 9:12] = 0
    correct[4:7, 9:12] = 4
    correct[4:7, 15:18] = 0
    out = solve(correct)
    assert np.all(out[4:7, 3:6] == 4) and np.all(out[5, 6:] == 4)
    closed = correct.copy()
    closed[5, 6] = 1
    assert np.array_equal(solve(closed), closed)
    # The first tile can be struck at an edge, without a centering condition.
    offset = correct.copy()
    offset[4:7, 9:12] = 0
    offset[3:6, 9:12] = 4
    assert np.all(solve(offset)[4:7, 3:6] == 4)


def self_test():
    """Bounded 42-seed semantic and variation checks; returns compact measurements."""
    _regression_tests()
    shapes, cavities, palettes, structures, frame_counts, active_counts = set(), set(), set(), set(), set(), set()
    statuses, directions, test_directions = set(), set(), set()
    key_colors, ray_lengths, gaps, tile_alignments = set(), set(), set(), set()
    total_pairs = 0
    shift_observed = {"larger_cavity": 0, "repeated_active_direction": 0, "combined_failures": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        train_sizes = {f["interior"] for p in puzzle["train"] for f in _inspect_pair(p)["frames"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                info = _inspect_pair(pair)
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                shapes.add(pair["input"].shape)
                palettes.add((info["bg"], info["outline"]))
                frame_counts.add(len(info["frames"]))
                active_counts.add(sum(f["active"] for f in info["frames"]))
                statuses.update(info["statuses"])
                directions.update(info["directions"])
                normalized = np.zeros(pair["input"].shape, dtype=np.int8)
                normalized[pair["input"] != info["bg"]] = 1
                for f in info["frames"]:
                    for p in f["cells"]:
                        normalized[p] = 2
                    cavities.add(f["interior"])
                    if f["gap"] is not None:
                        r0, r1, c0, c1 = f["box"]
                        gaps.add((f["direction"], f["gap"][0] - r0, f["gap"][1] - c0))
                        ray_lengths.add(len(f["ray"]))
                    if f["active"]:
                        key_colors.add(f["hit"]["color"])
                        a, b, x, y = f["hit"]["box"]
                        tile_alignments.add((f["direction"], f["gap"][0] - a, f["gap"][1] - x))
                structures.add(_signature(normalized))
                if split == "test":
                    test_directions.update(info["directions"])
                    active_dirs = [f["direction"] for f in info["frames"] if f["active"]]
                    shift_observed["repeated_active_direction"] += len(active_dirs) != len(set(active_dirs))
                    shift_observed["larger_cavity"] += any(max(f["interior"]) >= 5 and f["interior"] not in train_sizes
                                                           for f in info["frames"] if f["active"])
                    shift_observed["combined_failures"] += {"blocked", "closed", "empty"}.issubset(info["statuses"])
    assert len(shapes) >= 25 and len(cavities) >= 18 and len(palettes) >= 20
    assert len(structures) == total_pairs and frame_counts.issuperset({2, 3, 4, 5})
    assert active_counts.issuperset({1, 2, 3})
    assert statuses == {"active", "blocked", "closed", "empty"}
    assert directions == test_directions == set(DIRECTIONS)
    assert len(key_colors) >= 8 and len(ray_lengths) >= 15 and len(gaps) >= 20 and len(tile_alignments) >= 20
    assert all(v >= 3 for v in shift_observed.values())
    assert not _same(generate_puzzle(123), generate_puzzle(124))
    legal_count_calls = 0
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(10000 + n * 10 + m, n, m)
            validate_generated_puzzle(p)
            assert len(p["train"]) == n and len(p["test"]) == m
            legal_count_calls += 1
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": total_pairs, "legal_count_calls": legal_count_calls,
            "input_shapes": len(shapes), "interior_shapes": len(cavities), "palettes": len(palettes),
            "normalized_structures": len(structures), "frame_counts": sorted(frame_counts),
            "active_counts": sorted(active_counts), "ray_lengths": [min(ray_lengths), max(ray_lengths)],
            "shift_observed": shift_observed, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
