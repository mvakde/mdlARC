"""Generate cue-anchored two-cell strip puzzles."""

import hashlib

import numpy as np


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
SOURCE_DIGESTS = frozenset({
    "2d6041f1495fac4924704ece5b19cba6320ff64c45cc287d675fa7e3ffe083ee",
    "ac6af256d66d44dfc9b729d2ded25306d88b16c2971dd5e2393933fbfd3cb3c1",
    "b9476d91b7af33e66991e636c6a165cdb17089e2090cb15a5a11c668c5225a95",
    "5dd22d54ff2b8194b3d942312ce6368597e31b1b7eeb55e662ef3be502685be2",
})
SOURCE_SHAPES = {(16, 14), (12, 19), (16, 22), (25, 20)}
HOLE_KINDS = ("equality", "block_zero", "inner", "outer", "cue_strip")


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _count(num_train, "num_train")
    num_test = _count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(array):
    return array.shape, array.tobytes()


def _source_digest(pair):
    a = pair["input"].astype(np.uint8).tobytes()
    b = pair["output"].astype(np.uint8).tobytes()
    return hashlib.sha256(a + b).hexdigest()


def _coordinate(orientation, strip, position, offset):
    if orientation == HORIZONTAL:
        return 1 + 3 * strip + offset, 1 + position
    return 1 + position, 1 + 3 * strip + offset


def _block(position, anchor):
    return (position - anchor) // 2


def _hole_kind(strip, position, cue_strip, anchor):
    if strip == cue_strip:
        return "cue_strip"
    distance = abs(strip - cue_strip)
    block_distance = abs(_block(position, anchor))
    if block_distance == distance:
        return "equality"
    if block_distance < distance:
        return "block_zero" if block_distance == 0 else "inner"
    return "outer"


def _construct_output(grid, info, highlight):
    output = grid.copy()
    for strip in range(info["strip_count"]):
        if strip == info["cue_strip"]:
            continue
        distance = abs(strip - info["cue_strip"])
        for position in range(info["length"]):
            if abs(_block(position, info["cue_anchor"])) < distance:
                continue
            for offset in range(2):
                row, col = _coordinate(info["orientation"], strip, position, offset)
                if output[row, col] == info["fill"]:
                    output[row, col] = highlight
    return output


def _sample_geometry(rng, regime, orientation, rich):
    if regime == "source_train":
        strips = int(rng.integers(4, 9))
        length = int(rng.integers(10 if rich else 9, 23))
    elif regime == "target_test":
        if rng.random() < 0.45:
            strips = int(rng.integers(6, 10))
            length = int(rng.integers(16, 29))
        else:
            strips = int(rng.integers(4, 9))
            length = int(rng.integers(9, 23))
    else:
        raise ValueError("unknown regime")
    if rich:
        cue_strip = int(rng.integers(1, strips - 1))
        anchor = int(rng.integers(4, length - 5))
    else:
        cue_strip = int(rng.integers(0, strips))
        anchor = int(rng.integers(0, length - 1))
    shape = (3 * strips + 1, length + 2) if orientation == HORIZONTAL else (length + 2, 3 * strips + 1)
    return strips, length, cue_strip, anchor, shape


def _add_holes(rng, grid, info, regime):
    strips, length = info["strip_count"], info["length"]
    cue_strip, anchor = info["cue_strip"], info["cue_anchor"]
    orientation = info["orientation"]
    groups = {name: [] for name in HOLE_KINDS}
    witness_groups = {}
    for strip in range(strips):
        distance = abs(strip - cue_strip)
        for position in range(length):
            block_index = _block(position, anchor)
            kind = _hole_kind(strip, position, cue_strip, anchor)
            for offset in range(2):
                row, col = _coordinate(orientation, strip, position, offset)
                if grid[row, col] != info["fill"]:
                    continue
                cell = (row, col, strip, position)
                groups[kind].append(cell)
                if strip != cue_strip and (
                    abs(block_index) in (distance, distance - 1)
                    or (distance == 1 and abs(block_index) == 2)
                ):
                    witness_groups.setdefault((strip, block_index), []).append(cell)
    # Retain visible fill on each present equality and immediately inner block.
    reserved = {cells[0][:2] for cells in witness_groups.values()}
    groups = {name: [cell for cell in cells if cell[:2] not in reserved]
              for name, cells in groups.items()}
    fraction = float(rng.uniform(0.055, 0.17 if regime == "source_train" else 0.18))
    target = max(5, int(round(fraction * (2 * strips * length - 4))))
    used = set()
    per_position = np.zeros(length, dtype=np.int16)

    def place(cell):
        row, col, _strip, position = cell
        if (row, col) in used or per_position[position] >= 2 * strips - 1:
            return False
        grid[row, col] = info["background"]
        used.add((row, col))
        per_position[position] += 1
        return True

    for name in HOLE_KINDS:
        choices = groups[name]
        if not choices:
            raise AssertionError(f"missing {name} hole region")
        for index in rng.permutation(len(choices)):
            if place(choices[int(index)]):
                break
        else:
            raise AssertionError(f"could not place {name} hole")
    if len(used) + 2 <= target and rng.random() < 0.55:
        available = {cell[:2]: cell for cells in groups.values() for cell in cells
                     if cell[:2] not in used}
        adjacent = []
        for cell in available.values():
            row, col, strip, _position = cell
            for neighbour in ((row + 1, col), (row, col + 1)):
                other = available.get(neighbour)
                if other is not None and other[2] == strip:
                    adjacent.append((cell, other))
        if adjacent:
            first, second = adjacent[int(rng.integers(0, len(adjacent)))]
            place(first)
            place(second)
    remaining = [cell for cells in groups.values() for cell in cells]
    for index in rng.permutation(len(remaining)):
        if len(used) >= target:
            break
        place(remaining[int(index)])
    if len(used) != target:
        raise AssertionError("could not place requested holes")


def _generate_pair(rng, regime, orientation, colors, rich):
    strips, length, cue_strip, anchor, shape = _sample_geometry(rng, regime, orientation, rich)
    background, fill, cue, highlight = colors
    grid = np.full(shape, background, dtype=np.int8)
    for strip in range(strips):
        for position in range(length):
            for offset in range(2):
                grid[_coordinate(orientation, strip, position, offset)] = fill
    row, col = _coordinate(orientation, cue_strip, anchor, 0)
    grid[row:row + 2, col:col + 2] = cue
    info = {"orientation": orientation, "strip_count": strips, "length": length,
            "cue_strip": cue_strip, "cue_anchor": anchor,
            "background": background, "fill": fill, "cue": cue}
    _add_holes(rng, grid, info, regime)
    return {"input": grid, "output": _construct_output(grid, info, highlight)}


def _choose_palette(rng, highlight, previous=None):
    available = [color for color in range(10) if color != highlight]
    for _ in range(20):
        chosen = tuple(int(color) for color in rng.choice(available, size=3, replace=False))
        if chosen != previous:
            return (*chosen, highlight)
    raise AssertionError("palette selection failed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    highlight = int(rng.integers(0, 10))
    variable_palette = bool(rng.random() < 0.60)
    base_palette = _choose_palette(rng, highlight)
    orientation_mode = int(rng.choice(3, p=[0.25, 0.25, 0.50]))
    if orientation_mode == 0:
        train_orientations = [HORIZONTAL] * num_train
    elif orientation_mode == 1:
        train_orientations = [VERTICAL] * num_train
    else:
        train_orientations = [HORIZONTAL, VERTICAL]
        train_orientations.extend(HORIZONTAL if rng.random() < 0.5 else VERTICAL
                                  for _ in range(num_train - 2))
        rng.shuffle(train_orientations)
    learned_orientations = set(train_orientations)
    rich_index = int(rng.integers(0, num_train))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_palette = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            orientation = train_orientations[index] if split == "train" else (
                next(iter(learned_orientations)) if len(learned_orientations) == 1
                else (HORIZONTAL if rng.random() < 0.5 else VERTICAL)
            )
            palette = (_choose_palette(rng, highlight, previous_palette)
                       if variable_palette else base_palette)
            previous_palette = palette[:3]
            rich = split == "train" and index == rich_index
            for _ in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _generate_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                      orientation, palette, rich)
                key = _key(pair["input"])
                source_copy = (pair["input"].shape in SOURCE_SHAPES
                               and _source_digest(pair) in SOURCE_DIGESTS)
                if key not in seen and not source_copy:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_input(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    assert np.all(grid[0] == background) and np.all(grid[-1] == background)
    assert np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)
    full_rows = [r for r in range(1, height - 1) if np.all(grid[r] == background)]
    full_cols = [c for c in range(1, width - 1) if np.all(grid[:, c] == background)]
    horizontal = ((height - 1) % 3 == 0
                  and full_rows == list(range(3, height - 1, 3)) and not full_cols)
    vertical = ((width - 1) % 3 == 0
                and full_cols == list(range(3, width - 1, 3)) and not full_rows)
    assert horizontal != vertical
    orientation = HORIZONTAL if horizontal else VERTICAL
    strips = (height - 1) // 3 if horizontal else (width - 1) // 3
    length = width - 2 if horizontal else height - 2
    assert 4 <= strips <= 9 and 9 <= length <= 28
    non_background = [int(c) for c in np.unique(grid) if c != background]
    assert len(non_background) == 2
    cue_choices = [c for c in non_background if np.count_nonzero(grid == c) == 4]
    assert len(cue_choices) == 1
    cue = cue_choices[0]
    fill = next(c for c in non_background if c != cue)
    assert np.count_nonzero(grid == fill) > 4
    cue_positions = np.argwhere(grid == cue)
    rows = sorted({int(p[0]) for p in cue_positions})
    cols = sorted({int(p[1]) for p in cue_positions})
    assert len(rows) == len(cols) == 2
    assert rows[1] == rows[0] + 1 and cols[1] == cols[0] + 1
    assert {(int(r), int(c)) for r, c in cue_positions} == {(r, c) for r in rows for c in cols}
    if horizontal:
        assert (rows[0] - 1) % 3 == 0
        cue_strip = (rows[0] - 1) // 3
        anchor = cols[0] - 1
    else:
        assert (cols[0] - 1) % 3 == 0
        cue_strip = (cols[0] - 1) // 3
        anchor = rows[0] - 1
    assert 0 <= cue_strip < strips and 0 <= anchor <= length - 2
    assert (rows[0], cols[0]) == _coordinate(orientation, cue_strip, anchor, 0)
    hole_regions = {name: 0 for name in HOLE_KINDS}
    fill_regions = {name: 0 for name in HOLE_KINDS}
    holes_by_strip = {strip: set() for strip in range(strips)}
    selected_sides = set()
    selected_reach = {-1: 0, 1: 0}
    equality_sides = set()
    holes = 0
    for strip in range(strips):
        for position in range(length):
            kind = _hole_kind(strip, position, cue_strip, anchor)
            block_index = _block(position, anchor)
            for offset in range(2):
                row, col = _coordinate(orientation, strip, position, offset)
                value = int(grid[row, col])
                assert value in (background, fill, cue)
                if value == background:
                    holes += 1
                    hole_regions[kind] += 1
                    holes_by_strip[strip].add((row, col))
                elif value == fill:
                    fill_regions[kind] += 1
                    if strip != cue_strip and abs(block_index) >= abs(strip - cue_strip):
                        side = -1 if block_index < 0 else 1
                        selected_sides.add(side)
                        selected_reach[side] = max(selected_reach[side], abs(block_index))
                    if kind == "equality":
                        equality_sides.add(-1 if block_index < 0 else 1)
    adjacent_holes = any(
        (row + 1, col) in cells or (row, col + 1) in cells
        for cells in holes_by_strip.values() for row, col in cells
    )
    assert 5 <= holes <= int(np.ceil(0.19 * (2 * strips * length - 4)))
    return {
        "orientation": orientation, "strip_count": strips, "length": length,
        "cue_strip": cue_strip, "cue_anchor": anchor,
        "background": background, "fill": fill, "cue": cue,
        "holes": holes, "hole_regions": hole_regions, "fill_regions": fill_regions,
        "selected_sides": selected_sides, "equality_sides": equality_sides,
        "selected_reach": selected_reach,
        "adjacent_holes": adjacent_holes,
    }


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    assert grid.shape == output.shape
    info = _inspect_input(grid)
    new_colors = set(int(c) for c in np.unique(output)) - set(int(c) for c in np.unique(grid))
    assert len(new_colors) == 1
    highlight = new_colors.pop()
    assert np.array_equal(output, _construct_output(grid, info, highlight))
    assert all(info["hole_regions"][name] > 0 for name in HOLE_KINDS)
    assert all(info["fill_regions"][name] > 0 for name in ("equality", "block_zero", "inner", "outer"))
    info["highlight"] = highlight
    return info


def _predict_from_training(train, test_input):
    # The highlight is learned from training output; geometry and input roles
    # are read from the test input alone.
    highlights = []
    for pair in train:
        new_colors = (set(int(c) for c in np.unique(pair["output"]))
                      - set(int(c) for c in np.unique(pair["input"])))
        assert len(new_colors) == 1
        highlights.append(new_colors.pop())
    assert len(set(highlights)) == 1
    return _construct_output(test_input, _inspect_input(test_input), highlights[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if pair["input"].shape in SOURCE_SHAPES:
                assert _source_digest(pair) not in SOURCE_DIGESTS
            info = _inspect_pair(pair)
            infos[split].append(info)
            if split == "train":
                assert 4 <= info["strip_count"] <= 8 and 9 <= info["length"] <= 22
            else:
                assert 4 <= info["strip_count"] <= 9 and 9 <= info["length"] <= 28
    assert len({i["highlight"] for split in infos for i in infos[split]}) == 1
    assert {i["orientation"] for i in infos["test"]} <= {
        i["orientation"] for i in infos["train"]
    }
    train_palettes = {(i["background"], i["fill"], i["cue"]) for i in infos["train"]}
    all_palettes = {(i["background"], i["fill"], i["cue"])
                    for split in infos for i in infos[split]}
    assert len(train_palettes) > 1 or len(all_palettes) == 1
    assert all(len({i["background"], i["fill"], i["cue"], i["highlight"]}) == 4
               for split in infos for i in infos[split])
    assert any(
        0 < i["cue_strip"] < i["strip_count"] - 1
        and i["selected_sides"] == {-1, 1}
        and i["equality_sides"] == {-1, 1}
        and i["selected_reach"][-1] >= 2
        and i["selected_reach"][1] >= 2
        and i["fill_regions"]["inner"] > 0
        for i in infos["train"]
    )
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(np.array_equal(a[name], b[name])
                for a, b in zip(first[split], second[split]) for name in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    for seed, train_count, test_count in (
        (10, 2, 1), (11, 10, 1), (12, 10, 4),
        (13, None, 1), (14, None, 4), (15, 2, None), (16, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 4, "num_test": 4},
    ):
        try:
            generate_puzzle(30, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    assert _puzzles_equal(generate_puzzle(1234, 6, 3), generate_puzzle(1234, 6, 3))
    assert not _puzzles_equal(generate_puzzle(1234, 6, 3), generate_puzzle(1235, 6, 3))
    train_counts, test_counts = set(), set()
    test_orientations, train_modes = set(), set()
    fixed_palettes, variable_palettes = 0, 0
    cue_positions, phases, sizes, shapes, layouts = set(), set(), set(), set(), set()
    densities = []
    paired_holes = 0
    combined = 0
    edge_tests = near_edge_tests = center_tests = 0
    odd_one_test = 0
    longer_tests = shorter_tests = more_strip_tests = 0
    source_sized_pairs = 0
    novel_test_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_infos = [_inspect_pair(pair) for pair in puzzle["test"]]
        train_modes.add(frozenset(i["orientation"] for i in train_infos))
        palettes = {(i["background"], i["fill"], i["cue"]) for i in train_infos}
        fixed_palettes += len(palettes) == 1
        variable_palettes += len(palettes) > 1
        if len(test_infos) == 1:
            odd_one_test += test_infos[0]["cue_anchor"] % 2 == 1
        for info, pair in zip(test_infos, puzzle["test"]):
            assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])
            test_orientations.add(info["orientation"])
            edge_distance = min(info["cue_strip"], info["strip_count"] - 1 - info["cue_strip"])
            edge_tests += edge_distance == 0
            near_edge_tests += edge_distance == 1
            center_tests += edge_distance >= 2
            longer_tests += info["length"] > max(i["length"] for i in train_infos)
            shorter_tests += info["length"] < max(i["length"] for i in train_infos)
            more_strip_tests += info["strip_count"] > max(i["strip_count"] for i in train_infos)
            novel_test_palettes += len(palettes) > 1 and (
                info["background"], info["fill"], info["cue"]
            ) not in palettes
            if len(palettes) > 1 and info["adjacent_holes"] and info["cue_anchor"] % 2:
                combined += 1
        for split, split_infos in (("train", train_infos), ("test", test_infos)):
            for info, pair in zip(split_infos, puzzle[split]):
                cue_positions.add((info["strip_count"], info["cue_strip"]))
                phases.add(info["cue_anchor"] % 2)
                sizes.add((info["strip_count"], info["length"], info["orientation"]))
                shapes.add(pair["input"].shape)
                source_sized_pairs += pair["input"].shape in SOURCE_SHAPES
                layouts.add(_key(pair["input"]))
                densities.append(info["holes"] / (2 * info["strip_count"] * info["length"] - 4))
                paired_holes += info["adjacent_holes"]
                assert all(info["hole_regions"][name] > 0 for name in HOLE_KINDS)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert test_orientations == {HORIZONTAL, VERTICAL}
    assert train_modes == {frozenset({HORIZONTAL}), frozenset({VERTICAL}),
                           frozenset({HORIZONTAL, VERTICAL})}
    assert fixed_palettes >= 4 and variable_palettes >= 4
    assert phases == {0, 1} and len(cue_positions) >= 20
    assert len(sizes) >= 80 and len(shapes) >= 35 and len(layouts) >= 180
    assert min(densities) < 0.09 and max(densities) > 0.14
    assert paired_holes >= 10 and combined >= 2
    assert min(edge_tests, near_edge_tests, center_tests) >= 10
    assert odd_one_test >= 1
    assert longer_tests >= 5 and shorter_tests >= 10 and more_strip_tests >= 3
    assert novel_test_palettes >= 10 and source_sized_pairs >= 2
