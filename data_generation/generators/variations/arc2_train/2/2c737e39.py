"""Generate marker-registered, transparent motif copies for ARC."""

import hashlib

import numpy as np


# Digests of the four solved source pairs, including shape and both grids.
# Comparing contents permits their canvas sizes and palettes to recur.
_SOURCE_PAIR_DIGESTS = {
    "f6efc66d89d364aee1f88648cf2a270396949eec17855390a2bdae5c87497e00",
    "0897bb772134c3d4500068a2e279b880ca09031000fc4c4565dd33d1fc410eed",
    "0f65a3b5a7a6af2f536e6aa4768b67345df3ac6cd23cfaca6c1903ed4ce53ba5",
    "ffd37c96e616c9acb7e25b230d88515484cfebb2b0d0e7f4545092fbbb052f6a",
}

_DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_DIRECTIONS = _DIAGONALS + ((1, 0), (-1, 0), (0, 1), (0, -1))


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _motif_size(rng):
    kind = int(rng.integers(5))
    if kind == 0:  # Compact, including the original 3-by-2 motif.
        return int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if kind == 1:  # Wide and short, including the original test silhouette.
        return int(rng.integers(2, 5)), int(rng.integers(5, 9))
    if kind == 2:
        return int(rng.integers(5, 9)), int(rng.integers(2, 5))
    if kind == 3:
        if rng.random() < 0.5:
            return 1, int(rng.integers(4, 9))
        return int(rng.integers(4, 9)), 1
    return int(rng.integers(2, 7)), int(rng.integers(2, 7))


def _sample_mask(rng, height, width):
    """Grow one connected motif to the chosen bounding box."""
    if height == 1 or width == 1 or rng.random() < 0.16:
        return np.ones((height, width), dtype=bool)

    mask = np.zeros((height, width), dtype=bool)
    mask[int(rng.integers(height)), int(rng.integers(width))] = True
    wanted = max(4, int(round(height * width * rng.uniform(0.38, 0.88))))
    while (
        int(mask.sum()) < wanted
        or not mask[0].any()
        or not mask[-1].any()
        or not mask[:, 0].any()
        or not mask[:, -1].any()
    ):
        frontier = []
        weights = []
        occupied = np.argwhere(mask)
        top, left = occupied.min(axis=0)
        bottom, right = occupied.max(axis=0)
        for r in range(height):
            for c in range(width):
                if mask[r, c]:
                    continue
                if (
                    (r > 0 and mask[r - 1, c])
                    or (r + 1 < height and mask[r + 1, c])
                    or (c > 0 and mask[r, c - 1])
                    or (c + 1 < width and mask[r, c + 1])
                ):
                    frontier.append((r, c))
                    weights.append(
                        4 if r < top or r > bottom or c < left or c > right else 1
                    )
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        r, c = frontier[int(rng.choice(len(frontier), p=probabilities))]
        mask[r, c] = True
    return mask


def _paint_motif(rng, mask, anchor, colors, background, marker):
    height, width = mask.shape
    motif = np.full((height, width), background, dtype=np.int8)
    motif[anchor] = marker
    cells = [tuple(map(int, p)) for p in np.argwhere(mask) if tuple(p) != anchor]
    order = rng.permutation(len(cells))
    seeds = [cells[int(i)] for i in order[: len(colors)]]
    for cell, color in zip(seeds, colors):
        motif[cell] = color
    if rng.random() < 0.55:
        for cell in cells:
            if cell in seeds:
                continue
            distances = [abs(cell[0] - p[0]) + abs(cell[1] - p[1]) for p in seeds]
            nearest = [i for i, d in enumerate(distances) if d == min(distances)]
            motif[cell] = colors[int(rng.choice(nearest))]
    else:
        for cell in cells:
            if cell not in seeds:
                motif[cell] = colors[int(rng.integers(len(colors)))]
    return motif


def _ordered_positions(rng, limit, span, sign):
    """Two box positions separated by at least one empty row or column."""
    if sign == 0:
        place = int(rng.integers(limit - span + 1))
        return place, place
    first = int(rng.integers(limit - 2 * span))
    second = int(rng.integers(first + span + 1, limit - span + 1))
    return (first, second) if sign > 0 else (second, first)


def _shifted_positions(rng, limit, span, sign):
    if sign == 0:
        place = int(rng.integers(limit - span + 1))
        return place, place
    first = int(rng.integers(limit - span))
    second = int(rng.integers(first + 1, limit - span + 1))
    return (first, second) if sign > 0 else (second, first)


def _place_boxes(rng, motif_height, motif_width, direction):
    row_sign, col_sign = direction
    primary_vertical = col_sign == 0 or (
        row_sign != 0 and rng.random() < 0.5
    )
    if row_sign == 0:
        primary_vertical = False
    minimum_height = (
        2 * motif_height + 1 if primary_vertical else
        motif_height + (row_sign != 0)
    )
    minimum_width = (
        motif_width + (col_sign != 0) if primary_vertical else
        2 * motif_width + 1
    )
    minimum_height = max(3, minimum_height)
    minimum_width = max(3, minimum_width)
    height = minimum_height + int(rng.integers(0, min(31 - minimum_height, 12)))
    width = minimum_width + int(rng.integers(0, min(31 - minimum_width, 12)))
    if primary_vertical:
        source_top, target_top = _ordered_positions(
            rng, height, motif_height, row_sign
        )
        source_left, target_left = _shifted_positions(
            rng, width, motif_width, col_sign
        )
    else:
        source_top, target_top = _shifted_positions(
            rng, height, motif_height, row_sign
        )
        source_left, target_left = _ordered_positions(
            rng, width, motif_width, col_sign
        )
    return (height, width), (source_top, source_left), (target_top, target_left)


def _make_pair(rng, background, marker, colors, direction, forced_color=None):
    motif_height, motif_width = _motif_size(rng)
    mask = _sample_mask(rng, motif_height, motif_width)
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    anchor = cells[int(rng.integers(len(cells)))]
    available = [int(c) for c in colors if c not in (background, marker)]
    count = int(rng.integers(2, min(6, len(cells) - 1, len(available)) + 1))
    selected = tuple(
        available[int(i)] for i in rng.choice(len(available), count, replace=False)
    )
    if forced_color is not None and forced_color not in selected:
        selected = (forced_color,) + selected[1:]
    motif = _paint_motif(rng, mask, anchor, selected, background, marker)
    (height, width), (source_top, source_left), (target_top, target_left) = (
        _place_boxes(rng, motif_height, motif_width, direction)
    )
    input_grid = np.full((height, width), background, dtype=np.int8)
    source_region = input_grid[
        source_top : source_top + motif_height,
        source_left : source_left + motif_width,
    ]
    source_region[mask] = motif[mask]
    destination = (target_top + anchor[0], target_left + anchor[1])
    input_grid[destination] = marker
    output_grid = input_grid.copy()
    output_grid[destination] = background
    for r, c in cells:
        if (r, c) != anchor:
            output_grid[target_top + r, target_left + c] = motif[r, c]
    return {"input": input_grid, "output": output_grid}


def _directions(rng, train_count, test_count):
    mode = int(rng.integers(4))
    if mode == 0:  # An occasional source-like upward holdout.
        train = [(1, 1), (1, -1)]
        test = [(-1, 1)]
    else:
        first = _DIAGONALS[int(rng.integers(4))]
        second = _DIAGONALS[int(rng.integers(4))]
        while second == first:
            second = _DIAGONALS[int(rng.integers(4))]
        train = [first, second]
        if mode == 1:
            test = [train[int(rng.integers(2))]]
        elif mode == 2:
            unseen = [d for d in _DIAGONALS if d not in train]
            test = [unseen[int(rng.integers(len(unseen)))]]
        else:
            test = [_DIRECTIONS[int(rng.integers(len(_DIRECTIONS)))]]
    while len(train) < train_count:
        train.append(_DIRECTIONS[int(rng.integers(len(_DIRECTIONS)))])
    while len(test) < test_count:
        test.append(_DIRECTIONS[int(rng.integers(len(_DIRECTIONS)))])
    return train, test


def _pair_digest(pair):
    left = pair["input"].astype(np.uint8, copy=False)
    right = pair["output"].astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(left.shape) + left.tobytes() + right.tobytes()).hexdigest()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    roles = [int(c) for c in rng.permutation(10)]
    fixed_roles = bool(rng.random() < 0.55)
    reserve_novel_color = fixed_roles and rng.random() < 0.7
    train_dirs, test_dirs = _directions(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()

    for regime, directions in (
        ("source_train", train_dirs), ("target_test", test_dirs)
    ):
        section = "train" if regime == "source_train" else "test"
        for index, direction in enumerate(directions):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if fixed_roles or (section == "train" and index == 0):
                    background, marker = roles[:2]
                elif section == "train" and index == 1:
                    background, marker = roles[2:4]
                else:
                    pair_roles = pair_rng.permutation(10)
                    background, marker = map(int, pair_roles[:2])
                color_pool = tuple(
                    c for c in range(10)
                    if not (reserve_novel_color and section == "train" and c == roles[-1])
                )
                pair = _make_pair(
                    pair_rng, background, marker, color_pool, direction,
                    roles[-1] if reserve_novel_color and section == "test" and index == 0 else None,
                )
                input_grid = pair["input"]
                key = (input_grid.shape, input_grid.tobytes())
                if key not in seen_inputs and _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                    seen_inputs.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct source pair")
    return puzzle


def _components(grid, background):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    input_grid, output_grid = pair["input"], pair["output"]
    for grid in (input_grid, output_grid):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert input_grid.shape == output_grid.shape
    assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert int(counts.max()) > input_grid.size // 2
    components = sorted(_components(input_grid, background), key=len)
    assert len(components) == 2 and len(components[0]) == 1
    assert len(components[1]) >= 4
    destination = next(iter(components[0]))
    source = components[1]
    marker = int(input_grid[destination])
    anchors = [p for p in source if int(input_grid[p]) == marker]
    assert len(anchors) == 1
    anchor = anchors[0]
    colors = {int(input_grid[p]) for p in source if p != anchor}
    assert len(colors) >= 2 and background not in colors and marker not in colors
    top = min(r for r, c in source)
    bottom = max(r for r, c in source)
    left = min(c for r, c in source)
    right = max(c for r, c in source)
    dr, dc = destination[0] - anchor[0], destination[1] - anchor[1]
    assert (dr, dc) != (0, 0)
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            tr, tc = r + dr, c + dc
            assert 0 <= tr < input_grid.shape[0] and 0 <= tc < input_grid.shape[1]
            assert int(input_grid[tr, tc]) == (
                marker if (r, c) == anchor else background
            )
    expected = input_grid.copy()
    expected[destination] = background
    for r, c in source:
        if (r, c) != anchor:
            expected[r + dr, c + dc] = input_grid[r, c]
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    degree = sum(
        (anchor[0] + ar, anchor[1] + ac) in source
        for ar, ac in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    return {
        "background": background, "marker": marker, "colors": colors,
        "source": source, "anchor": anchor, "destination": destination,
        "shape": input_grid.shape, "height": bottom - top + 1,
        "width": right - left + 1, "filled": len(source) == (bottom - top + 1) * (right - left + 1),
        "degree": degree, "dr": dr, "dc": dc, "top": top, "left": left,
        "bottom": bottom, "right": right,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            info = _inspect_pair(pair)
            infos[section].append(info)
            input_grid, output_grid = pair["input"], pair["output"]
            input_key = (input_grid.shape, input_grid.tobytes())
            pair_key = (input_key, output_grid.tobytes())
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
    # A varying color-role convention is demonstrated in training. Constant
    # roles across training must remain constant at test time.
    train_roles = {(p["background"], p["marker"]) for p in infos["train"]}
    if len(train_roles) == 1:
        assert {(p["background"], p["marker"]) for p in infos["test"]} == train_roles
    assert len({(p["dr"], p["dc"]) for p in infos["train"]}) >= 2


def _same_puzzle(left, right):
    return all(
        np.array_equal(a[k], b[k])
        for section in ("train", "test")
        for a, b in zip(left[section], right[section])
        for k in ("input", "output")
    ) and all(len(left[s]) == len(right[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, role_schedules = (
        set(), set(), set(), set(), set()
    )
    directions = set()
    split_directions = {"train": set(), "test": set()}
    anchor_degrees = set()
    palette_sizes = set()
    small = large = filled = role_varying = role_constant = 0
    sparse = bars = small_motifs = tip_small = filled_axis = original_shapes = 0
    varied_canvas = varied_motif = varied_palette = 0
    test_subset = test_wider = test_bottom = 0
    shared_test_direction = novel_test_direction = simpler_test = new_test_color = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {s: [_inspect_pair(p) for p in puzzle[s]] for s in ("train", "test")}
        train_dirs = {(np.sign(p["dr"]), np.sign(p["dc"])) for p in infos["train"]}
        test_dirs = {(np.sign(p["dr"]), np.sign(p["dc"])) for p in infos["test"]}
        shared_test_direction += bool(train_dirs & test_dirs)
        novel_test_direction += bool(test_dirs - train_dirs)
        train_colors = set().union(*(p["colors"] for p in infos["train"]))
        simpler_test += any(
            len(p["colors"]) < max(len(t["colors"]) for t in infos["train"])
            for p in infos["test"]
        )
        new_test_color += any(p["colors"] - train_colors for p in infos["test"])
        test_subset += any(p["colors"] < train_colors for p in infos["test"])
        test_wider += any(
            p["width"] > max(t["width"] for t in infos["train"])
            for p in infos["test"]
        )
        test_bottom += any(
            p["bottom"] >= p["shape"][0] - 2 for p in infos["test"]
        )
        all_infos = infos["train"] + infos["test"]
        varied_canvas += len({p["shape"] for p in all_infos}) > 1
        varied_motif += len({(p["height"], p["width"]) for p in all_infos}) > 1
        varied_palette += len({len(p["colors"]) for p in all_infos}) > 1
        roles = {(p["background"], p["marker"]) for p in infos["train"]}
        role_varying += len(roles) > 1
        role_constant += len(roles) == 1
        role_schedules.add(tuple(sorted(roles)))
        signatures.add(tuple(
            _pair_digest(p) for s in ("train", "test") for p in puzzle[s]
        ))
        for section in ("train", "test"):
            for p in infos[section]:
                shapes.add(p["shape"])
                directions.add((np.sign(p["dr"]), np.sign(p["dc"])))
                split_directions[section].add((np.sign(p["dr"]), np.sign(p["dc"])))
                anchor_degrees.add(p["degree"])
                palette_sizes.add(len(p["colors"]))
                small += p["shape"][0] <= 12 and p["shape"][1] <= 12
                large += p["shape"][0] >= 20 or p["shape"][1] >= 20
                filled += p["filled"]
                sparse += len(p["source"]) / (p["height"] * p["width"]) <= 0.6
                bars += min(p["height"], p["width"]) == 1
                small_motifs += len(p["source"]) <= 6
                tip_small += (
                    p["degree"] == 1 and p["shape"][0] <= 12
                    and p["shape"][1] <= 12
                )
                filled_axis += p["filled"] and (p["dr"] == 0 or p["dc"] == 0)
                original_shapes += p["shape"] in {
                    (12, 12), (12, 10), (9, 10), (11, 10)
                }
                layouts.add((
                    p["shape"], p["height"], p["width"], p["top"], p["left"],
                    p["degree"], p["dr"], p["dc"], len(p["source"]),
                ))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert len(role_schedules) >= 20 and role_varying >= 8 and role_constant >= 8
    assert all(d in directions for d in _DIRECTIONS)
    assert all(d in split_directions["train"] for d in _DIRECTIONS)
    assert all(d in split_directions["test"] for d in _DIRECTIONS)
    assert 1 in anchor_degrees and any(d >= 3 for d in anchor_degrees)
    assert 2 in palette_sizes and any(n >= 5 for n in palette_sizes)
    assert small >= 8 and large >= 15 and filled >= 15
    assert sparse >= 10 and bars >= 10 and small_motifs >= 10
    assert tip_small >= 5 and filled_axis >= 5 and original_shapes >= 1
    assert varied_canvas == varied_motif == varied_palette == 42
    assert shared_test_direction >= 8 and novel_test_direction >= 8
    assert simpler_test >= 8 and new_test_color >= 8
    assert test_subset >= 8 and test_wider >= 1 and test_bottom >= 8
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + 10 * train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(
            puzzle, generate_puzzle(1000 + 10 * train_count + test_count, train_count, test_count)
        )
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + 10 * (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
