"""Generate boundary-filter ARC puzzles with varied visible geometry."""

import numpy as np


_SOURCE_INPUTS = {
    tuple(s.split("/"))
    for s in (
        "000003003000/000000000000/004000000040/004000000040/"
        "004030000040/004000000040/004000000040/004000000043/"
        "004000000040/004000003040/004000000040/004444444440/"
        "300000000000",
        "08000000000/00000000800/60000000006/06008000060/"
        "00600800600/00060006000/00006060000/00000600000/"
        "00000000000/00000008000/00080000000",
        "50005050005000/01000000000100/01050005000105/"
        "01100500001100/00111111111000/50000000000000",
        "4200000002/0200000002/0200040222/0200000200/"
        "0200400200/0200000200/0200002204/0200002000/"
        "0200002040/0200002000/0220020000/0022220000/"
        "0000000000",
        "0000000000600/0000000000000/0060000000600/"
        "0000006000000/3000000600003/3000000000063/"
        "3000000000003/3000000000003/3333333333333/"
        "0000000000000/0000000000000/0000000000000",
    )
}


def _grid_rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _check_count(value, low, high, name):
    if value is not None and (type(value) is not int or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _geometry(rng, kind, regime):
    """Return an upward-opening shape before puzzle-wide rotation."""
    if kind == "v":
        depth = int(rng.integers(3, 11))
        gap = 2 * depth
    else:
        depth = int(rng.integers(3, 15))
        gap = int(rng.integers(5, 24))

    top, bottom, margin_l, margin_r = [int(x) for x in rng.integers(0, 5, size=4)]
    margin_l = min(margin_l, 3)
    margin_r = min(margin_r, 3)
    if top == bottom == margin_l == margin_r == 0:
        bottom = 1
    height = top + depth + 1 + bottom
    width = margin_l + gap + 1 + margin_r
    assert height <= 30 and width <= 30

    lefts = [margin_l] * depth
    rights = [margin_l + gap] * depth
    if kind == "v":
        lefts = [margin_l + k for k in range(depth)]
        rights = [margin_l + gap - k for k in range(depth)]
        lefts.append(margin_l + depth)
        rights.append(margin_l + depth)
    else:
        if kind == "step":
            largest = min(gap - 3, 2 * depth + 2)
            if float(rng.random()) < (0.30 if regime == "target_test" else 0.18):
                shrink = largest
            else:
                shrink = int(rng.integers(1, largest + 1))
            event_count = int(rng.integers(1, min(depth, 5)))
            event_rows = rng.choice(np.arange(1, depth), size=event_count, replace=False)
            for _ in range(shrink):
                row = int(rng.choice(event_rows))
                if int(rng.integers(2)) == 0:
                    for k in range(row, depth):
                        lefts[k] += 1
                else:
                    for k in range(row, depth):
                        rights[k] -= 1
        lefts.append(lefts[-1])
        rights.append(rights[-1])
    return height, width, top, lefts, rights


def _draw_shape(height, width, top, lefts, rights):
    boundary = np.zeros((height, width), dtype=bool)
    inside = np.zeros_like(boundary)
    depth = len(lefts) - 1
    pointed = lefts[-1] == rights[-1]
    for k in range(depth):
        row = top + k
        if k == 0 or pointed:
            boundary[row, lefts[k]] = True
            boundary[row, rights[k]] = True
        else:
            boundary[row, lefts[k - 1]:lefts[k] + 1] = True
            boundary[row, rights[k]:rights[k - 1] + 1] = True
        inside[row, lefts[k] + 1:rights[k]] = True
    row = top + depth
    boundary[row, lefts[-1]:rights[-1] + 1] = True
    return boundary, inside


def _choose_positions(rng, candidates, count):
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in np.atleast_1d(indices)]


def _make_pair(rng, kind, regime, palette, rotation, opening_marker):
    background, wall_color, marker_color = palette
    height, width, top, lefts, rights = _geometry(rng, kind, regime)
    boundary, inside = _draw_shape(height, width, top, lefts, rights)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[boundary] = wall_color
    output = np.full_like(grid, background)

    inside_positions = [tuple(int(x) for x in p) for p in np.argwhere(inside)]
    kept_count = int(rng.integers(2 if opening_marker else 1, min(6, len(inside_positions)) + 1))
    kept = []
    if opening_marker:
        first_row = [p for p in inside_positions if p[0] == top]
        kept = _choose_positions(rng, first_row, 1)
    remaining = [p for p in inside_positions if p not in kept]
    kept += _choose_positions(rng, remaining, kept_count - len(kept)) if kept_count > len(kept) else []

    outside = [tuple(int(x) for x in p) for p in np.argwhere(~inside & ~boundary)]
    assert outside
    outside_count = int(rng.integers(2 if opening_marker else 1, min(7, len(outside)) + 1))
    categories = {
        "above": [p for p in outside if p[0] < top],
        "below": [p for p in outside if p[0] > top + len(lefts) - 1],
        "side": [
            p for p in outside
            if top <= p[0] < top + len(lefts) - 1
            and (p[1] < lefts[p[0] - top] or p[1] > rights[p[0] - top])
        ],
    }
    changed = [
        top + k for k in range(1, len(lefts) - 1)
        if lefts[k] != lefts[k - 1] or rights[k] != rights[k - 1]
    ]
    categories["step"] = [
        p for p in categories["side"]
        if any(abs(p[0] - row) <= 1 for row in changed)
    ]
    categories["far"] = outside
    removed = []
    for _ in range(outside_count):
        available = [name for name, points in categories.items() if any(p not in removed for p in points)]
        name = str(rng.choice(available))
        possibilities = [p for p in categories[name] if p not in removed]
        removed.append(possibilities[int(rng.integers(len(possibilities)))])

    for row, col in kept:
        grid[row, col] = marker_color
        output[row, col] = marker_color
    for row, col in removed:
        grid[row, col] = marker_color
    return {
        "input": np.rot90(grid, rotation).copy(),
        "output": np.rot90(output, rotation).copy(),
    }


def _palette(rng, background=None):
    if background is None:
        background = int(rng.choice(10, p=[0.68] + [0.32 / 9] * 9))
    choices = [x for x in range(10) if x != background]
    wall_color, marker_color = (int(x) for x in rng.choice(choices, size=2, replace=False))
    return background, wall_color, marker_color


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    rotation = int(rng.choice(4, p=[0.76, 0.08, 0.08, 0.08]))
    color_policy = int(rng.choice(3, p=[0.22, 0.58, 0.20]))
    shared_background = int(rng.choice(10, p=[0.72] + [0.28 / 9] * 9))
    fixed_palette = _palette(rng, shared_background)

    train_modes = ["step", str(rng.choice(["flat", "v"]))]
    if num_train >= 3:
        train_modes.append("v" if train_modes[1] == "flat" else "flat")
    while len(train_modes) < num_train:
        train_modes.append(str(rng.choice(["flat", "v", "step"])))
    rng.shuffle(train_modes)
    available_test_modes = sorted(set(train_modes))
    test_weights = np.array(
        [0.6 if mode == "step" else 0.2 for mode in available_test_modes],
        dtype=float,
    )
    test_weights /= test_weights.sum()
    test_modes = [
        str(rng.choice(available_test_modes, p=test_weights))
        for _ in range(num_test)
    ]

    puzzle = {"train": [], "test": []}
    seen = set()
    palettes = []
    for part, modes in (("train", train_modes), ("test", test_modes)):
        for index, mode in enumerate(modes):
            if color_policy == 0:
                palette = fixed_palette
            elif color_policy == 1:
                palette = _palette(rng, shared_background)
                if part == "train" and index == 1 and palette == palettes[0]:
                    palette = (palette[0], palette[2], palette[1])
            else:
                palette = _palette(rng)
                if part == "train" and index == 1 and palette[0] == palettes[0][0]:
                    other_backgrounds = [x for x in range(10) if x != palettes[0][0]]
                    palette = _palette(rng, int(rng.choice(other_backgrounds)))
            palettes.append(palette)
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(
                    pair_rng, mode, "source_train" if part == "train" else "target_test",
                    palette, rotation, part == "train" and index == 0,
                )
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and _grid_rows(pair["input"]) not in _SOURCE_INPUTS:
                    seen.add(key)
                    puzzle[part].append(pair)
                    break
            else:
                # Padding supplies a deterministic distinct layout if an
                # unusually unlucky sequence repeats earlier inputs.
                background = palette[0]
                height, width = pair["input"].shape
                for extra_rows in range(31 - height):
                    for extra_cols in range(31 - width):
                        if extra_rows == extra_cols == 0:
                            continue
                        padded = {
                            name: np.pad(
                                grid, ((0, extra_rows), (0, extra_cols)),
                                constant_values=background,
                            )
                            for name, grid in pair.items()
                        }
                        padded_key = (
                            padded["input"].shape,
                            padded["input"].tobytes(),
                        )
                        if (
                            padded_key not in seen
                            and _grid_rows(padded["input"]) not in _SOURCE_INPUTS
                        ):
                            seen.add(padded_key)
                            puzzle[part].append(padded)
                            break
                    else:
                        continue
                    break
                else:
                    raise RuntimeError("could not construct distinct pair")
    return puzzle


def _runs(columns):
    groups = [[int(columns[0])]]
    for value in columns[1:]:
        value = int(value)
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _trace_up(input_grid, wall_color):
    rows = np.flatnonzero(np.any(input_grid == wall_color, axis=1))
    if len(rows) < 4 or rows.tolist() != list(range(int(rows[0]), int(rows[-1]) + 1)):
        raise AssertionError("boundary rows")
    top, end = int(rows[0]), int(rows[-1])
    lefts, rights = [], []
    for row in range(top, end):
        columns = np.flatnonzero(input_grid[row] == wall_color)
        if len(columns) < 2:
            raise AssertionError("missing sides")
        runs = _runs(columns)
        if len(runs) != 2:
            raise AssertionError("two separate sides required")
        left, right = runs[0][-1], runs[1][0]
        if right - left < 2:
            raise AssertionError("empty interior")
        if row == top:
            if len(runs[0]) != 1 or len(runs[1]) != 1:
                raise AssertionError("opening shape")
        else:
            joined = runs[0][0] == lefts[-1] and runs[1][-1] == rights[-1]
            diagonal = (
                len(runs[0]) == len(runs[1]) == 1
                and left == lefts[-1] + 1 and right == rights[-1] - 1
            )
            if not (joined or diagonal):
                raise AssertionError("broken ledge")
            if left < lefts[-1] or right > rights[-1]:
                raise AssertionError("walls turn outward")
        lefts.append(left)
        rights.append(right)
    base = np.flatnonzero(input_grid[end] == wall_color)
    if len(base) == 1:
        kind = "v"
        if any(lefts[k] != lefts[k - 1] + 1 or rights[k] != rights[k - 1] - 1
               for k in range(1, len(lefts))):
            raise AssertionError("V sides")
        if lefts[-1] + 1 != int(base[0]) or rights[-1] - 1 != int(base[0]):
            raise AssertionError("V point")
        lefts.append(int(base[0]))
        rights.append(int(base[0]))
    else:
        kind = "step" if any(
            lefts[k] != lefts[k - 1] or rights[k] != rights[k - 1]
            for k in range(1, len(lefts))
        ) else "flat"
        if base.tolist() != list(range(lefts[-1], rights[-1] + 1)):
            raise AssertionError("flat base")
        if rights[-1] - lefts[-1] < 3:
            raise AssertionError("base too narrow")
        lefts.append(lefts[-1])
        rights.append(rights[-1])
    expected_boundary, inside = _draw_shape(*input_grid.shape, top, lefts, rights)
    if not np.array_equal(input_grid == wall_color, expected_boundary):
        raise AssertionError("boundary structure")
    return kind, top, lefts, rights, inside


def _inspect_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    colors, counts = np.unique(output_grid, return_counts=True)
    if len(colors) != 2:
        raise AssertionError("output roles")
    background = int(colors[np.argmax(counts)])
    marker = int(colors[np.argmin(counts)])
    input_colors = {int(x) for x in np.unique(input_grid)}
    if len(input_colors) != 3 or not {background, marker} <= input_colors:
        raise AssertionError("input roles")
    wall = (input_colors - {background, marker}).pop()
    for rotation in range(4):
        canonical = np.rot90(input_grid, rotation)
        try:
            kind, top, lefts, rights, inside = _trace_up(canonical, wall)
        except AssertionError:
            continue
        expected = np.full_like(canonical, background)
        expected[(canonical == marker) & inside] = marker
        if not np.array_equal(np.rot90(output_grid, rotation), expected):
            raise AssertionError("incorrect spatial filter")
        kept = int(np.count_nonzero((canonical == marker) & inside))
        removed = int(np.count_nonzero((canonical == marker) & ~inside))
        if not 1 <= kept <= 6 or not 1 <= removed <= 7:
            raise AssertionError("marker counts")
        opening_kept = bool(np.any((canonical[top] == marker) & inside[top]))
        first_below_kept = bool(np.any((canonical[top + 1] == marker) & inside[top + 1]))
        edge = bool(np.any(canonical[0] == wall) or np.any(canonical[-1] == wall)
                    or np.any(canonical[:, 0] == wall) or np.any(canonical[:, -1] == wall))
        steps = [
            (lefts[k] - lefts[k - 1], rights[k - 1] - rights[k])
            for k in range(1, len(lefts) - 1)
        ]
        return {
            "kind": kind, "rotation": rotation, "palette": (background, wall, marker),
            "kept": kept, "removed": removed, "opening_kept": opening_kept,
            "first_below_kept": first_below_kept, "edge": edge, "top": top,
            "bottom_margin": canonical.shape[0] - (top + len(lefts)),
            "base_gap": rights[-1] - lefts[-1], "steps": steps,
            "shape": canonical.shape,
        }
    raise AssertionError("no upward-open canonical boundary")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for part in ("train", "test"):
        for pair in puzzle[part]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            input_key = (pair["input"].shape, pair["input"].tobytes())
            pair_key = (input_key, pair["output"].tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _grid_rows(pair["input"]) not in _SOURCE_INPUTS
            features[part].append(_inspect_pair(pair))
    all_features = features["train"] + features["test"]
    assert len({f["rotation"] for f in all_features}) == 1
    assert {"step"} <= {f["kind"] for f in features["train"]}
    assert any(f["kind"] != "step" for f in features["train"])
    assert {f["kind"] for f in features["test"]} <= {f["kind"] for f in features["train"]}
    assert any(f["opening_kept"] for f in features["train"])
    assert any(f["kept"] >= 2 and f["removed"] >= 2 for f in features["train"])
    train_palettes = {f["palette"] for f in features["train"]}
    if len(train_palettes) == 1:
        assert {f["palette"] for f in all_features} == train_palettes
    train_backgrounds = {f["palette"][0] for f in features["train"]}
    if len(train_backgrounds) == 1:
        assert {f["palette"][0] for f in all_features} == train_backgrounds


def _same_puzzle(first, second):
    return all(
        np.array_equal(a[key], b[key])
        for part in ("train", "test")
        for a, b in zip(first[part], second[part])
        for key in ("input", "output")
    ) and all(len(first[p]) == len(second[p]) for p in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, dimensions, structures = set(), set(), set(), set()
    kinds_train, kinds_test, kept_counts, rotations = set(), set(), set(), set()
    opening_count = no_first_below = edge_count = ledge_count = long_step_count = 0
    narrow_count = no_below_count = varied_palette_puzzles = varied_background_puzzles = 0
    fixed_palette_puzzles = combined_count = 0
    first_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first = puzzle["train"][0]["input"]
        first_inputs.add((first.shape, first.tobytes()))
        feats = {p: [_inspect_pair(x) for x in puzzle[p]] for p in ("train", "test")}
        palettes = {f["palette"] for f in feats["train"] + feats["test"]}
        backgrounds = {p[0] for p in palettes}
        varied_palette_puzzles += len({f["palette"] for f in feats["train"]}) > 1
        varied_background_puzzles += len({f["palette"][0] for f in feats["train"]}) > 1
        fixed_palette_puzzles += len(palettes) == 1
        for part in ("train", "test"):
            for pair, f in zip(puzzle[part], feats[part]):
                dimensions.add(pair["input"].shape)
                structures.add((f["kind"], f["shape"], tuple(f["steps"]), f["base_gap"]))
                kept_counts.add(f["kept"])
                rotations.add(f["rotation"])
                (kinds_train if part == "train" else kinds_test).add(f["kind"])
                opening_count += f["opening_kept"]
                no_first_below += not f["first_below_kept"]
                edge_count += f["edge"]
                ledge_count += f["kind"] == "step"
                long_step_count += any(max(x) >= 2 for x in f["steps"])
                narrow_count += f["kind"] == "step" and f["base_gap"] == 3
                no_below_count += f["bottom_margin"] == 0
                combined_count += f["kind"] == "step" and f["edge"] and f["kept"] >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(first_inputs) >= 40 and len(dimensions) >= 40 and len(structures) >= 75
    assert kinds_train == kinds_test == {"flat", "v", "step"}
    assert {1, 2, 3, 4} <= kept_counts and len(rotations) == 4
    assert opening_count >= 30 and no_first_below >= 30 and edge_count >= 20
    assert ledge_count >= 30 and long_step_count >= 10 and narrow_count >= 3
    assert no_below_count >= 10 and combined_count >= 3
    assert varied_palette_puzzles >= 20 and varied_background_puzzles >= 3
    assert fixed_palette_puzzles >= 2
    for seed, (ntrain, ntest) in enumerate(
        ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)),
        start=100,
    ):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed, ntrain, ntest))
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(11, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    damaged = generate_puzzle(999, 2, 1)
    pair = damaged["train"][0]
    f = _inspect_pair(pair)
    row, col = np.argwhere(pair["output"] == f["palette"][2])[0]
    pair["output"][int(row), int(col)] = f["palette"][0]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
