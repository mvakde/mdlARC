"""Clockwise, unrotated copies of the objects rooted at literal grid corners."""

import numpy as np


_SOURCE_INPUTS = (
    ("4477779766", "4477779976", "7777779986", "7777577776", "7777557777", "7577777777", "7557777777", "7778777777", "2788777797", "2777777799"),
    ("57377799", "57377799", "57777777", "77755577", "77773777", "77773775", "72273775", "72277775"),
    ("67337222", "67337776", "77757776", "72775777", "72227777", "77766777", "55777774", "55777774"),
    ("8877777776777755", "8877777776777755", "8877755576799778", "7777779776797887", "7777779776777887", "7999779777777777", "7797777777777777", "7777777888877777", "7777777787773333", "7722777777777777", "7722799777777777", "6777799777777777", "6777777777788777", "1111777777788778", "1111777777777778", "1111667777777778"),
)


def _is_count(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if not _is_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _corners(height, width):
    return ((0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0))


def _neighbors(row, col, height, width):
    if row:
        yield row - 1, col
    if row + 1 < height:
        yield row + 1, col
    if col:
        yield row, col - 1
    if col + 1 < width:
        yield row, col + 1


def _components(grid, background):
    height, width = grid.shape
    seen = set()
    result = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            cells = {(row, col)}
            seen.add((row, col))
            stack = [(row, col)]
            while stack:
                r, c = stack.pop()
                for neighbor in _neighbors(r, c, height, width):
                    if neighbor not in seen and int(grid[neighbor]) == color:
                        seen.add(neighbor)
                        cells.add(neighbor)
                        stack.append(neighbor)
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            result.append({"color": color, "cells": cells,
                           "bbox": (min(rows), max(rows), min(cols), max(cols))})
    return result


def _analyze(grid, background):
    corners = _corners(*grid.shape)
    components = _components(grid, background)
    corner_objects = []
    inert = []
    for component in components:
        memberships = [i for i, cell in enumerate(corners) if cell in component["cells"]]
        if memberships:
            for i in memberships:
                corner_objects.append((i, component))
        else:
            inert.append(component)
    return corner_objects, inert


def _translated(component, corner, height, width):
    lo_r, hi_r, lo_c, hi_c = component["bbox"]
    if corner == 0:
        dr, dc = -lo_r, -lo_c
    elif corner == 1:
        dr, dc = -lo_r, width - 1 - hi_c
    elif corner == 2:
        dr, dc = height - 1 - hi_r, width - 1 - hi_c
    else:
        dr, dc = height - 1 - hi_r, -lo_c
    return {(r + dr, c + dc) for r, c in component["cells"]}


def _infer_background(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    maximum = int(counts.max())
    assert maximum * 2 > grid.size and np.count_nonzero(counts == maximum) == 1
    return int(np.argmax(counts))


def _transform_grid(grid, background):
    sources, _ = _analyze(grid, background)
    copies = [(_translated(component, (corner + 1) % 4, *grid.shape), component["color"])
              for corner, component in sources]
    result = grid.copy()
    for cells, color in copies:
        for cell in cells:
            result[cell] = color
    return result


def _connected(cells, height, width):
    first = next(iter(cells))
    reached = {first}
    stack = [first]
    while stack:
        row, col = stack.pop()
        for neighbor in _neighbors(row, col, height, width):
            if neighbor in cells and neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    return len(reached) == len(cells)


def _mask(rng, height, width, corner, kind):
    """Build in root-outward coordinates, then reflect to the actual corner."""
    if height == 1 or width == 1:
        cells = {(r, c) for r in range(height) for c in range(width)}
    elif kind == "block":
        cells = {(r, c) for r in range(height) for c in range(width)}
    elif kind == "hook":
        cells = {(r, 0) for r in range(height)} | {(0, c) for c in range(width)}
        if rng.random() < 0.5:
            cells = {(r, width - 1 - c) for r, c in cells}
            # Keep the actual root by adding a short connecting top arm.
            cells |= {(0, c) for c in range(width)}
    elif kind == "path":
        steps = [(1, 0)] * (height - 1) + [(0, 1)] * (width - 1)
        rng.shuffle(steps)
        cells = {(0, 0)}
        r = c = 0
        for dr, dc in steps:
            r += dr
            c += dc
            cells.add((r, c))
    elif kind == "branch":
        cells = {(r, 0) for r in range(height)} | {(0, c) for c in range(width)}
        for _ in range(int(rng.integers(1, height + width))):
            frontier = sorted({neighbor for r, c in cells
                               for neighbor in _neighbors(r, c, height, width)
                               if neighbor not in cells})
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    else:  # indented, including concave rectangles
        cells = {(r, c) for r in range(height) for c in range(width)}
        candidates = [(r, c) for r, c in cells if (r, c) != (0, 0)]
        rng.shuffle(candidates)
        target = int(rng.integers(1, max(2, height * width // 3 + 1)))
        removed = 0
        for cell in candidates:
            trial = cells - {cell}
            if (any(r == height - 1 for r, _ in trial)
                    and any(c == width - 1 for _, c in trial)
                    and _connected(trial, height, width)):
                cells = trial
                removed += 1
                if removed >= target:
                    break
    result = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        rr = height - 1 - r if corner in (2, 3) else r
        cc = width - 1 - c if corner in (1, 2) else c
        result[rr, cc] = True
    return result


def _sample_dimensions(rng, regime):
    if regime == "source_train":
        ranges = ((8, 12), (11, 19), (17, 25))
        weights = (0.53, 0.37, 0.10)
    else:
        ranges = ((8, 13), (12, 20), (18, 28))
        weights = (0.20, 0.45, 0.35)
    band = int(rng.choice(3, p=weights))
    low, high = ranges[band]
    height = int(rng.integers(low, high))
    width = int(rng.integers(low, high))
    if rng.random() < 0.30:
        other_band = int(rng.choice(3, p=weights))
        other_low, other_high = ranges[other_band]
        width = int(rng.integers(other_low, other_high))
    return height, width


def _sample_puzzle_parameters(rng):
    colors = list(range(10))
    rng.shuffle(colors)
    constant = rng.random() < 0.28
    reserved = None if constant or rng.random() < 0.18 else colors.pop()
    palette = tuple(colors[:int(rng.integers(4, min(8, len(colors)) + 1))]) if constant else None
    fixed_background = int(rng.choice(colors)) if constant or rng.random() < 0.65 else None
    return {"palette": palette, "reserved": reserved, "fixed_background": fixed_background}


def _pair_colors(rng, params, regime, background, count, force_new):
    if params["palette"] is None:
        available = [c for c in range(10) if c != background
                     and (regime == "target_test" or c != params["reserved"])]
    else:
        available = [c for c in params["palette"] if c != background]
    assert available
    if count <= len(available) and rng.random() < 0.53:
        chosen = [int(x) for x in rng.choice(available, size=count, replace=False)]
    else:
        chosen = [int(rng.choice(available)) for _ in range(count)]
    if force_new:
        chosen[int(rng.integers(count))] = int(params["reserved"])
    return chosen, available


def _place_distractor(grid, rng, background, color, blocked, edge):
    height, width = grid.shape
    for _ in range(85):
        size = int(rng.choice((1, 2, 3, 4, 5, 6), p=(0.12, 0.19, 0.20, 0.20, 0.16, 0.13)))
        if size <= 2 or rng.random() < 0.38:
            if rng.random() < 0.5:
                mh, mw = 1, size
            else:
                mh, mw = size, 1
            kind = "block"
        else:
            mh = int(rng.integers(2, min(4, size) + 1))
            mw = int(rng.integers(2, min(4, size) + 1))
            kind = str(rng.choice(("block", "hook", "path", "branch", "indented")))
        if mh > height - 2 or mw > width - 2:
            continue
        mask = _mask(rng, mh, mw, int(rng.integers(4)), kind)
        if edge:
            side = int(rng.integers(4))
            if side == 0:
                start_r, start_c = 0, int(rng.integers(1, width - mw))
            elif side == 1:
                start_r, start_c = int(rng.integers(1, height - mh)), width - mw
            elif side == 2:
                start_r, start_c = height - mh, int(rng.integers(1, width - mw))
            else:
                start_r, start_c = int(rng.integers(1, height - mh)), 0
        else:
            start_r = int(rng.integers(1, height - mh))
            start_c = int(rng.integers(1, width - mw))
        cells = {(start_r + int(r), start_c + int(c)) for r, c in np.argwhere(mask)}
        if any(cell in blocked or cell in _corners(height, width)
               or int(grid[cell]) != background for cell in cells):
            continue
        if any(int(grid[neighbor]) == color for r, c in cells
               for neighbor in _neighbors(r, c, height, width) if neighbor not in cells):
            continue
        for cell in cells:
            grid[cell] = color
        return True
    return False


def _make_pair(rng, regime, params, occupied, force_edge=False,
               force_asymmetric=False, force_new=False):
    for _ in range(150):
        height, width = _sample_dimensions(rng, regime)
        background = params["fixed_background"]
        if background is None:
            possibilities = [c for c in range(10) if c != params["reserved"]]
            background = int(rng.choice(possibilities))
        grid = np.full((height, width), background, dtype=np.int8)
        colors, available = _pair_colors(rng, params, regime, background,
                                          len(occupied), force_new)
        max_h, max_w = (height - 1) // 2, (width - 1) // 2
        if min(max_h, max_w) < 2:
            continue
        for index, corner in enumerate(occupied):
            if force_asymmetric and index == 0:
                mh, mw = (2, 3) if rng.random() < 0.5 else (3, 2)
                kind = str(rng.choice(("hook", "path")))
            else:
                kind = str(rng.choice(("block", "bar", "hook", "path", "branch", "indented"),
                                      p=(0.20, 0.15, 0.18, 0.12, 0.19, 0.16)))
                if kind == "bar":
                    if rng.random() < 0.5:
                        mh, mw = 1, int(rng.integers(2, min(7, max_w + 1)))
                    else:
                        mh, mw = int(rng.integers(2, min(7, max_h + 1))), 1
                else:
                    mh = int(rng.integers(2, min(7, max_h + 1)))
                    mw = int(rng.integers(2, min(7, max_w + 1)))
            mask = _mask(rng, mh, mw, corner, kind)
            start_r = 0 if corner in (0, 1) else height - mh
            start_c = 0 if corner in (0, 3) else width - mw
            for r, c in np.argwhere(mask):
                grid[start_r + int(r), start_c + int(c)] = colors[index]
        sources, _ = _analyze(grid, background)
        if len(sources) != len(occupied):
            continue
        blocked = set()
        for corner, component in sources:
            cells = _translated(component, (corner + 1) % 4, height, width)
            if blocked.intersection(cells):
                break
            blocked.update(cells)
        else:
            if regime == "source_train":
                count = int(rng.integers(1, min(8, 2 + grid.size // 38)))
            else:
                count = int(rng.integers(2, min(15, 4 + grid.size // 24)))
            if force_edge:
                count = max(1, count)
            for index in range(count):
                color = int(rng.choice(available))
                edge = (force_edge and index == 0) or rng.random() < (0.42 if regime == "source_train" else 0.50)
                if not _place_distractor(grid, rng, background, color, blocked, edge):
                    if force_edge and index == 0:
                        break
            else:
                output = _transform_grid(grid, background)
                if not np.array_equal(grid, output):
                    try:
                        _validate_pair({"input": grid, "output": output})
                    except AssertionError:
                        continue
                    return {"input": grid, "output": output}
    raise ValueError("unable to construct pair")


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_copy(grid):
    rows = tuple("".join(str(int(value)) for value in row) for row in grid)
    return rows in _SOURCE_INPUTS


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = pair["input"], pair["output"]
    for item in (grid, output):
        assert isinstance(item, np.ndarray) and item.ndim == 2
        assert 1 <= item.shape[0] <= 30 and 1 <= item.shape[1] <= 30
        assert np.issubdtype(item.dtype, np.integer)
        assert 0 <= int(item.min()) <= int(item.max()) <= 9
    assert grid.shape == output.shape and not _source_copy(grid)
    background = _infer_background(grid)
    sources, inert = _analyze(grid, background)
    corners = set(_corners(*grid.shape))
    assert 1 <= len(sources) <= 4
    assert len({corner for corner, _ in sources}) == len(sources)
    copied = set()
    for corner, component in sources:
        assert len(component["cells"] & corners) == 1
        translated = _translated(component, (corner + 1) % 4, *grid.shape)
        assert len(translated) == len(component["cells"])
        assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in translated)
        assert copied.isdisjoint(translated)
        assert any(int(grid[cell]) != component["color"] for cell in translated)
        assert any(int(output[cell]) == component["color"] for cell in component["cells"])
        copied.update(translated)
        assert all(int(output[cell]) == component["color"] for cell in translated)
    inert_cells = {cell for component in inert for cell in component["cells"]}
    assert copied.isdisjoint(inert_cells)
    assert all(int(output[cell]) == int(grid[cell]) for cell in inert_cells)
    assert np.array_equal(output, _transform_grid(grid, background))
    assert not np.array_equal(grid, output)
    return {"background": background, "sources": sources, "inert": inert}


def _asymmetric_evidence(pair, analysis):
    for _, component in analysis["sources"]:
        lo_r, hi_r, lo_c, hi_c = component["bbox"]
        height, width = hi_r - lo_r + 1, hi_c - lo_c + 1
        if height > 1 and width > 1 and height != width and len(component["cells"]) < height * width:
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_directions = set()
    train_empty = False
    edge_inert = False
    asymmetric = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _validate_pair(pair)
            grid = pair["input"]
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            if split == "train":
                assert 8 <= grid.shape[0] <= 24 and 8 <= grid.shape[1] <= 24
                train_directions.update(corner for corner, _ in analysis["sources"])
                train_empty |= len(analysis["sources"]) < 4
                edge_inert |= any(any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                                      for r, c in component["cells"]) for component in analysis["inert"])
                asymmetric |= _asymmetric_evidence(pair, analysis)
            else:
                assert 8 <= grid.shape[0] <= 27 and 8 <= grid.shape[1] <= 27
    assert train_directions == {0, 1, 2, 3}
    assert train_empty and edge_inert and asymmetric


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    first_count = int(rng.integers(2, 5))
    first = sorted(int(x) for x in rng.choice(4, size=first_count, replace=False))
    missing = sorted(set(range(4)) - set(first))
    if not missing:
        second = sorted(int(x) for x in rng.choice(4, size=int(rng.integers(2, 4)), replace=False))
    else:
        extras = [i for i in range(4) if i not in missing]
        rng.shuffle(extras)
        second = sorted(missing + extras[:int(rng.integers(0, len(extras) + 1))])
    train_occupations = [first, second]
    for _ in range(train_count - 2):
        count = int(rng.choice((1, 2, 3, 4), p=(0.12, 0.25, 0.32, 0.31)))
        train_occupations.append(sorted(int(x) for x in rng.choice(4, size=count, replace=False)))
    test_occupations = []
    for _ in range(test_count):
        count = int(rng.choice((1, 2, 3, 4), p=(0.10, 0.22, 0.32, 0.36)))
        test_occupations.append(sorted(int(x) for x in rng.choice(4, size=count, replace=False)))
    asym_index = int(rng.integers(train_count))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, occupations in (("train", train_occupations), ("test", test_occupations)):
        regime = "source_train" if split == "train" else "target_test"
        for index, occupied in enumerate(occupations):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _make_pair(pair_rng, regime, params, occupied,
                                  force_edge=split == "train" and index == 0,
                                  force_asymmetric=split == "train" and index == asym_index,
                                  force_new=split == "test" and params["reserved"] is not None
                                  and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("unable to construct distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name]) for a, b in zip(first[split], second[split])
                   for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, signatures, shapes = set(), set(), set()
    solid = hook = bar = empty_test = novel = edge_train = 0
    open_far = large_inert = shared_inert = repeated = small_test = large_train = 0
    varied_background = varied_corners = edge_test = no_edge_test = 0
    first_corner_counts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_grid = puzzle["train"][0]["input"]
        first_corner_counts.add(len(_analyze(first_grid, _infer_background(first_grid))[0]))
        train_colors = {int(c) for pair in puzzle["train"] for c in np.unique(pair["input"])}
        if any(any(int(c) not in train_colors for c in np.unique(pair["input"])) for pair in puzzle["test"]):
            novel += 1
        signature = []
        puzzle_backgrounds = set()
        puzzle_corner_counts = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                dimensions.add(grid.shape)
                signature.append(_key(grid))
                background = _infer_background(grid)
                puzzle_backgrounds.add(background)
                sources, inert = _analyze(grid, background)
                if split == "train":
                    puzzle_corner_counts.add(len(sources))
                if split == "test" and len(sources) < 4:
                    empty_test += 1
                edge = any(any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                               for r, c in obj["cells"]) for obj in inert)
                if split == "train" and edge:
                    edge_train += 1
                if split == "test":
                    edge_test += edge
                    no_edge_test += not edge
                    small_test += max(grid.shape) <= 12
                else:
                    large_train += max(grid.shape) >= 18
                large_inert += any(len(obj["cells"]) >= 4 for obj in inert)
                shared_inert += any(obj["color"] in {source["color"] for _, source in sources}
                                    for obj in inert)
                repeated += len({source["color"] for _, source in sources}) < len(sources)
                for corner, obj in sources:
                    lo_r, hi_r, lo_c, hi_c = obj["bbox"]
                    h, w = hi_r - lo_r + 1, hi_c - lo_c + 1
                    relative = frozenset((r - lo_r, c - lo_c) for r, c in obj["cells"])
                    shapes.add((h, w, relative))
                    solid += h > 1 and w > 1 and len(relative) == h * w
                    bar += h == 1 or w == 1
                    hook += h > 1 and w > 1 and len(relative) == h + w - 1
                    far_r = hi_r if corner in (0, 1) else lo_r
                    far_c = hi_c if corner in (0, 3) else lo_c
                    open_far += (far_r, far_c) not in obj["cells"]
        signatures.add(tuple(signature))
        varied_background += len(puzzle_backgrounds) > 1
        varied_corners += len(puzzle_corner_counts) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 50 and len(signatures) == 42 and len(shapes) >= 80
    assert min(solid, hook, bar, empty_test, novel, edge_train) > 0
    assert first_corner_counts == {2, 3, 4}
    assert varied_corners == 42 and varied_background > 0
    assert min(open_far, large_inert, shared_inert, repeated, small_test,
               large_train, edge_test, no_edge_test) > 0
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    assert _puzzles_equal(generate_puzzle(271828, 6, 3), generate_puzzle(271828, 6, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                           (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (3, 4), (True, 1), (2, False)):
        try:
            generate_puzzle(999, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request succeeded")
