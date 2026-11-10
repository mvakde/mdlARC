"""Sparse diagonal guides define filled, possibly truncated triangles."""

import hashlib

import numpy as np


_DIRECTIONS = ("D", "U", "L", "R")
_SOURCE_DIGESTS = {
    "1a328a81051861d13125d7507bf734c10702778bf2f31732d8dc9a596bdcf85e",
    "26e4d21f4cca592eaec78fbd48db0f058c8e143d3bcf084d1365836d2d27266f",
    "e7c4ec095dbc0ab504e8fb32b545fccb27b6b353b4d181419c9b6cf722255bcf",
    "da49553b47bd389934442b792362494bfe866b1c0a92254ae457a95cf4da5b44",
}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, train, test):
    train = _checked_count(train, "num_train", 2, 10)
    test = _checked_count(test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _point(apex, direction, distance, sign):
    r, c = apex
    if direction == "D":
        return r + distance, c + sign * distance
    if direction == "U":
        return r - distance, c + sign * distance
    if direction == "R":
        return r + sign * distance, c + distance
    return r + sign * distance, c - distance


def _inside(point, height, width):
    return 0 <= point[0] < height and 0 <= point[1] < width


def _footprint(apex, direction, depth, height, width):
    cells = set()
    for d in range(depth + 1):
        if direction in ("D", "U"):
            r = apex[0] + (d if direction == "D" else -d)
            if 0 <= r < height:
                for c in range(max(0, apex[1] - d), min(width - 1, apex[1] + d) + 1):
                    cells.add((r, c))
        else:
            c = apex[1] + (d if direction == "R" else -d)
            if 0 <= c < width:
                for r in range(max(0, apex[0] - d), min(height - 1, apex[0] + d) + 1):
                    cells.add((r, c))
    return frozenset(cells)


def _visible_depth(apex, direction, depth, height, width):
    if direction == "D":
        return min(depth, height - 1 - apex[0])
    if direction == "U":
        return min(depth, apex[0])
    if direction == "R":
        return min(depth, width - 1 - apex[1])
    return min(depth, apex[1])


def _all_side(apex, direction, depth, height, width):
    return frozenset(
        p
        for d in range(depth + 1)
        for sign in (-1, 1)
        if _inside((p := _point(apex, direction, d, sign)), height, width)
    )


def _guide_components(grid, guide):
    points = set(zip(*np.where(grid == guide)))
    groups = []
    while points:
        start = points.pop()
        group = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            near = {p for p in points if max(abs(p[0] - r), abs(p[1] - c)) <= 3}
            points.difference_update(near)
            group.update(near)
            stack.extend(near)
        groups.append(frozenset(group))
    return groups


def _fit_group(group, height, width):
    """Recover a visible footprint from guide positions alone.

    The terminal visible guide sets the smallest supported extent. A side
    continuing outside the canvas adds no visible cells.
    """
    fits = []
    for direction in _DIRECTIONS:
        if direction == "D":
            boundary = min(p[0] for p in group)
            apices = [p for p in group if p[0] == boundary]
            extent = max(p[0] for p in group) - boundary
        elif direction == "U":
            boundary = max(p[0] for p in group)
            apices = [p for p in group if p[0] == boundary]
            extent = boundary - min(p[0] for p in group)
        elif direction == "R":
            boundary = min(p[1] for p in group)
            apices = [p for p in group if p[1] == boundary]
            extent = max(p[1] for p in group) - boundary
        else:
            boundary = max(p[1] for p in group)
            apices = [p for p in group if p[1] == boundary]
            extent = boundary - min(p[1] for p in group)
        if len(apices) != 1 or not 2 <= extent <= 8:
            continue
        apex = apices[0]
        side = _all_side(apex, direction, extent, height, width)
        if not group <= side:
            continue
        if not any(_point(apex, direction, extent, sign) in group for sign in (-1, 1)):
            continue
        if not all(any(_point(apex, direction, d, sign) in group
                       for d in range(1, extent + 1)) for sign in (-1, 1)):
            continue
        fits.append((direction, apex, extent))
    return fits


def _decode_input(grid, fill):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3 and fill not in colors
    guide = int(colors[int(np.argmin(counts))])
    assert int(counts.min()) < int(np.partition(counts, 1)[1])
    groups = _guide_components(grid, guide)
    assert 1 <= len(groups) <= 4
    output = grid.copy()
    information = []
    occupied = set()
    height, width = grid.shape
    for group in groups:
        fits = _fit_group(group, height, width)
        assert len(fits) == 1
        direction, apex, depth = fits[0]
        cells = _footprint(apex, direction, depth, height, width)
        assert cells.isdisjoint(occupied)
        occupied.update(cells)
        assert group < cells
        for r, c in cells - group:
            output[r, c] = fill
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        truncated = any(not _inside(_point(apex, direction, depth, sign),
                                    height, width) for sign in (-1, 1))
        information.append({
            "direction": direction,
            "apex": apex,
            "depth": depth,
            "box": (min(rows), min(cols), max(rows), max(cols)),
            "truncated": truncated,
            "guides": group,
            "cells": cells,
        })
    return output, information, guide


def _choose_guides(rng, candidate, height, width):
    apex = candidate["apex"]
    direction = candidate["direction"]
    depth = candidate["visible_depth"]
    coverage = int(rng.integers(0, 4))
    guides = {apex}
    for sign in (-1, 1):
        visible = [d for d in range(1, depth + 1)
                   if _inside(_point(apex, direction, d, sign), height, width)]
        if not visible:
            return None
        terminal = visible[-1]
        # Consecutive selected points never lie more than three diagonal
        # steps apart, so a sparse outline remains one visible guide group.
        if coverage == 0 and depth <= 3 and terminal == depth:
            selected = [terminal]
        else:
            step = 1 if (coverage == 3 or
                         (coverage == 2 and sign == -1)) else int(rng.integers(2, 4))
            selected = list(range(step, terminal + 1, step))
            selected.append(terminal)
            if selected and selected[0] > 3:
                selected.insert(0, min(3, terminal))
        for d in selected:
            guides.add(_point(apex, direction, d, sign))
        if coverage in (2, 3):
            for d in visible:
                if rng.random() < (0.47 if coverage == 2 else 0.72):
                    guides.add(_point(apex, direction, d, sign))
    if not any(_point(apex, direction, depth, s) in guides for s in (-1, 1)):
        return None
    if len(guides) >= len(candidate["cells"]):
        return None
    if len(_fit_group(frozenset(guides), height, width)) != 1:
        return None
    return frozenset(guides)


def _candidate(rng, height, width, direction, want_clip):
    for _ in range(400):
        depth = int(rng.integers(2, 8))
        apex = (int(rng.integers(1, height - 1)),
                int(rng.integers(1, width - 1)))
        visible_depth = _visible_depth(apex, direction, depth, height, width)
        if not 2 <= visible_depth <= 7:
            continue
        cells = _footprint(apex, direction, depth, height, width)
        visible_tips = [_point(apex, direction, visible_depth, s) for s in (-1, 1)]
        # Require a visible lateral truncation, not merely an arbitrary
        # continuation past the base-facing canvas edge.
        clipped = any(not _inside(p, height, width) for p in visible_tips)
        if clipped != want_clip or len(cells) < 7:
            continue
        if not any(_inside(p, height, width) for p in visible_tips):
            continue
        candidate = {
            "apex": apex, "direction": direction, "depth": depth,
            "visible_depth": visible_depth, "cells": cells,
            "clip": clipped,
        }
        guides = _choose_guides(rng, candidate, height, width)
        if guides is not None:
            candidate["guides"] = guides
            return candidate
    return None


def _compatible(candidate, placed):
    for other in placed:
        if not candidate["cells"].isdisjoint(other["cells"]):
            return False
        if any(max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 3
               for a in candidate["guides"] for b in other["guides"]):
            return False
    return True


def _dimensions(rng, count):
    low = (9, 11, 14, 17)[count - 1]
    # Height and width are drawn independently; tests are not always larger.
    height = int(rng.integers(low, 31))
    width = int(rng.integers(low, 31))
    return height, width


def _palette(rng, fill, base):
    if rng.random() < 0.48:
        return base
    colors = [c for c in range(10) if c != fill]
    chosen = rng.choice(colors, size=3, replace=False)
    return (int(chosen[0]), int(chosen[1]), int(chosen[2]))


def _substrate(rng, height, width, palette):
    first, second, _ = palette
    proportion = float(rng.uniform(0.24, 0.76))
    mask = rng.random((height, width)) < proportion
    # Small patches produce variable local texture without a new geometric rule.
    for _ in range(int(rng.integers(0, 8))):
        r = int(rng.integers(0, height))
        c = int(rng.integers(0, width))
        rh = int(rng.integers(1, 5))
        cw = int(rng.integers(1, 5))
        mask[r:r + rh, c:c + cw] = bool(rng.integers(0, 2))
    return np.where(mask, first, second).astype(np.int8)


def _construct_pair(rng, directions, clip_flags, fill, base):
    count = len(directions)
    for _ in range(90):
        height, width = _dimensions(rng, count)
        placed = []
        order = rng.permutation(count)
        success = True
        for index in order:
            index = int(index)
            found = None
            for _ in range(150):
                candidate = _candidate(rng, height, width,
                                       directions[index], clip_flags[index])
                if candidate is not None and _compatible(candidate, placed):
                    found = candidate
                    break
            if found is None:
                success = False
                break
            placed.append(found)
        if not success:
            continue
        palette = _palette(rng, fill, base)
        grid = _substrate(rng, height, width, palette)
        footprint = set().union(*(item["cells"] for item in placed))
        guides = set().union(*(item["guides"] for item in placed))
        inside = sorted(footprint - guides)
        outside = sorted(set(np.ndindex(grid.shape)) - footprint)
        if len(inside) < 2 or len(outside) < 2:
            continue
        # Both substrate hues must be evidenced inside and outside.
        for positions in (inside, outside):
            selected = rng.choice(len(positions), size=2, replace=False)
            grid[positions[int(selected[0])]] = palette[0]
            grid[positions[int(selected[1])]] = palette[1]
        for point in guides:
            grid[point] = palette[2]
        try:
            output, information, inferred_guide = _decode_input(grid, fill)
        except AssertionError:
            continue
        if inferred_guide != palette[2] or len(information) != count:
            continue
        expected = grid.copy()
        for point in footprint - guides:
            expected[point] = fill
        if not np.array_equal(output, expected):
            continue
        return {"input": grid, "output": expected}
    raise ValueError("unable to construct a readable triangle layout")


def _plans(rng, num_train, num_test):
    # Pick test configurations first, then ensure every test direction appears
    # in at least one training demonstration.
    progressive = bool(rng.random() < 0.20)
    test_counts = ([4] * num_test if progressive else
                   [int(rng.integers(1, 5)) for _ in range(num_test)])
    test_directions = [
        [str(x) for x in rng.choice(_DIRECTIONS, size=count)]
        for count in test_counts
    ]
    if progressive:
        test_directions[0] = [str(x) for x in rng.permutation(_DIRECTIONS)]
    required = sorted(set().union(*(set(x) for x in test_directions)))
    rng.shuffle(required)
    if progressive:
        train_counts = [2, 3]
        train_counts.extend(int(rng.integers(2, 4)) for _ in range(num_train - 2))
    else:
        train_counts = [int(rng.integers(2, 5)), int(rng.integers(2, 5))]
        train_counts.extend(int(rng.integers(1, 5)) for _ in range(num_train - 2))
    while sum(train_counts) < len(required):
        index = int(rng.integers(0, num_train))
        if train_counts[index] < 4:
            train_counts[index] += 1
    train_directions = [[None] * count for count in train_counts]
    slots = [(i, j) for i, count in enumerate(train_counts) for j in range(count)]
    rng.shuffle(slots)
    for direction, (i, j) in zip(required, slots):
        train_directions[i][j] = direction
    for i, row in enumerate(train_directions):
        for j, direction in enumerate(row):
            if direction is None:
                train_directions[i][j] = str(rng.choice(_DIRECTIONS))
    train_clips = [
        [bool(rng.random() < 0.32) for _ in row]
        for row in train_directions
    ]
    train_clips[0][int(rng.integers(0, len(train_clips[0])))] = True
    train_clips[1][int(rng.integers(0, len(train_clips[1])))] = False
    test_clips = [
        [bool(rng.random() < 0.44) for _ in row]
        for row in test_directions
    ]
    return train_directions, test_directions, train_clips, test_clips


def _same_input(first, second):
    return first.shape == second.shape and np.array_equal(first, second)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    chosen = rng.choice(10, size=4, replace=False)
    base = (int(chosen[0]), int(chosen[1]), int(chosen[2]))
    fill = int(chosen[3])
    train_dirs, test_dirs, train_clips, test_clips = _plans(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    for section, dirs, clips in (
        ("train", train_dirs, train_clips),
        ("test", test_dirs, test_clips),
    ):
        for row, flags in zip(dirs, clips):
            for _ in range(8):
                pair = _construct_pair(rng, row, flags, fill, base)
                if not any(_same_input(pair["input"], old["input"])
                           for old in puzzle["train"] + puzzle["test"]):
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("unable to make unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(pair):
    value = (pair["input"].tolist(), pair["output"].tolist())
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _palette_from_pair(pair):
    grid, output = pair["input"], pair["output"]
    original = set(int(x) for x in np.unique(grid))
    result = set(int(x) for x in np.unique(output))
    assert len(original) == 3 and len(result) == 4
    introduced = result - original
    assert len(introduced) == 1
    fill = next(iter(introduced))
    changed = grid != output
    assert np.any(changed)
    assert np.all(output[changed] == fill)
    assert np.array_equal(output[~changed], grid[~changed])
    guide_candidates = [color for color in original
                        if np.all(~changed[grid == color])]
    assert len(guide_candidates) == 1
    guide = guide_candidates[0]
    backgrounds = original - {guide}
    assert len(backgrounds) == 2
    for color in backgrounds:
        assert np.any(changed & (grid == color))
        assert np.any(~changed & (grid == color))
        assert np.count_nonzero(grid == color) > np.count_nonzero(grid == guide)
    return fill, guide, backgrounds


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    seen = []
    train_directions = set()
    test_directions = set()
    fills = set()
    train_has_clip = False
    for index, pair in enumerate(pairs):
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= side <= 30 for side in grid.shape)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        assert pair["input"].shape == pair["output"].shape
        assert not any(_same_input(pair["input"], old) for old in seen)
        seen.append(pair["input"])
        assert _digest(pair) not in _SOURCE_DIGESTS
        fill, guide, backgrounds = _palette_from_pair(pair)
        fills.add(fill)
        predicted, information, inferred = _decode_input(pair["input"], fill)
        assert inferred == guide and np.array_equal(predicted, pair["output"])
        assert 1 <= len(information) <= 4
        if index < len(puzzle["train"]):
            train_directions.update(item["direction"] for item in information)
            train_has_clip |= any(item["truncated"] for item in information)
        else:
            test_directions.update(item["direction"] for item in information)
    assert len(fills) == 1
    assert test_directions <= train_directions
    assert train_has_clip
    assert any(not item["truncated"]
               for pair in puzzle["train"]
               for item in _decode_input(pair["input"], next(iter(fills)))[1])


def _equal_puzzles(a, b):
    return all(len(a[section]) == len(b[section]) and
               all(np.array_equal(x[key], y[key])
                   for x, y in zip(a[section], b[section])
                   for key in ("input", "output"))
               for section in ("train", "test"))


def self_test() -> None:
    sample = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(x["train"]) for x in sample}) >= 4
    assert len({len(x["test"]) for x in sample}) >= 3
    assert _equal_puzzles(generate_puzzle(17), generate_puzzle(17))
    assert _equal_puzzles(generate_puzzle(9001, 7, 3),
                          generate_puzzle(9001, 7, 3))
    shapes, depths, counts, orientations = set(), set(), set(), set()
    clipped = 0
    contained = 0
    small_test = 0
    sparse = 0
    asymmetric = 0
    near = 0
    overlapping_boxes = 0
    clipped_edges = set()
    repeated_directions = 0
    changing_palettes = 0
    varying_test_counts = 0
    source_shapes = 0
    palettes = set()
    for puzzle in sample:
        validate_generated_puzzle(puzzle)
        max_train_area = max(p["input"].size for p in puzzle["train"])
        pair_palettes = set()
        test_object_counts = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                fill, guide, background = _palette_from_pair(pair)
                palettes.add((tuple(sorted(background)), guide, fill))
                pair_palettes.add((tuple(sorted(background)), guide))
                info = _decode_input(pair["input"], fill)[1]
                shapes.add(pair["input"].shape)
                counts.add(len(info))
                source_shapes += pair["input"].shape in {
                    (15, 12), (13, 12), (15, 16), (15, 18)
                }
                orientations.update(x["direction"] for x in info)
                depths.update(x["depth"] for x in info)
                clipped += sum(x["truncated"] for x in info)
                contained += sum(not x["truncated"] for x in info)
                sparse += sum(len(x["guides"]) <= 3 for x in info)
                repeated_directions += (
                    len({x["direction"] for x in info}) < len(info)
                )
                if section == "test" and pair["input"].size <= max_train_area:
                    small_test += 1
                if section == "test":
                    test_object_counts.add(len(info))
                for i, a in enumerate(info):
                    side_counts = [
                        sum(_point(a["apex"], a["direction"], d, sign)
                            in a["guides"] for d in range(1, a["depth"] + 1))
                        for sign in (-1, 1)
                    ]
                    asymmetric += abs(side_counts[0] - side_counts[1]) >= 2
                    if a["truncated"]:
                        for sign in (-1, 1):
                            r, c = _point(a["apex"], a["direction"],
                                          a["depth"], sign)
                            if r < 0:
                                clipped_edges.add("top")
                            if r >= pair["input"].shape[0]:
                                clipped_edges.add("bottom")
                            if c < 0:
                                clipped_edges.add("left")
                            if c >= pair["input"].shape[1]:
                                clipped_edges.add("right")
                    for b in info[:i]:
                        ar0, ac0, ar1, ac1 = a["box"]
                        br0, bc0, br1, bc1 = b["box"]
                        if not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0):
                            overlapping_boxes += 1
                        if min(max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                               for p in a["cells"] for q in b["cells"]) <= 2:
                            near += 1
        changing_palettes += len(pair_palettes) > 1
        varying_test_counts += len(test_object_counts) > 1
    assert len(shapes) >= 35 and len(palettes) >= 30
    assert counts == {1, 2, 3, 4} and orientations == set(_DIRECTIONS)
    assert {2, 3, 4} <= depths
    assert clipped >= 20 and contained >= 100
    assert sparse >= 20 and small_test >= 10
    assert near >= 5 and overlapping_boxes >= 1
    assert asymmetric >= 30 and repeated_directions >= 20
    assert clipped_edges == {"top", "bottom", "left", "right"}
    assert changing_palettes >= 20 and varying_test_counts >= 5
    assert source_shapes >= 1
    for seed, train, test in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4),
        (201, None, 1), (202, None, 4),
        (203, 2, None), (204, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
