"""Synthetic instances of the left-moving foreground layer ARC task."""

import numpy as np


_SOURCE_INPUTS = {
    "00000000/00000000/22200000/22200000/21110000/01110000/01110000/00000000",
    "00000000/00000000/00000000/00044440/00044440/00044333/00044333/00000333",
    "04440000/04440000/04446666/00066666/00066666/00066666/00000000",
    "00000000/00000000/00555000/00515000/00555000/00000000/00000000/00000000",
    "00000000/00000000/88880000/88990000/88990000/88880000/00000000/00000000",
    "0000000000/0000000000/0000000000/0000000000/0777777770/0700000070/0700066666/0777766666/0000066666/0000000000",
    "0000000000/0040000000/0040000000/0040000000/0545555550/0040000000/0040000000/0040000000/0000000000",
}


def _text(grid):
    return "/".join("".join(str(int(value)) for value in row) for row in grid)


def _bbox(mask):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max() - rows.min() + 1),
        int(cols.max() - cols.min() + 1),
    )


def _local_shape(kind, height, width, corner=0, thickness=1):
    mask = np.ones((height, width), dtype=bool)
    if kind == "frame":
        mask[1:-1, 1:-1] = False
    elif kind == "l":
        mask[:] = False
        if corner in (0, 2):
            mask[:, :thickness] = True
        else:
            mask[:, -thickness:] = True
        if corner in (0, 1):
            mask[:thickness, :] = True
        else:
            mask[-thickness:, :] = True
    return mask


def _shape(size, box, kind, corner=0, thickness=1):
    top, left, height, width = box
    mask = np.zeros(size, dtype=bool)
    mask[top:top + height, left:left + width] = _local_shape(
        kind, height, width, corner, thickness
    )
    return mask


def _shape_candidates(size, box):
    _, _, height, width = box
    choices = [("line" if min(height, width) == 1 else "solid", 0, 1)]
    if height >= 3 and width >= 3:
        choices.append(("frame", 0, 1))
        for thickness in (1, 2):
            if thickness < min(height, width) - 1:
                for corner in range(4):
                    choices.append(("l", corner, thickness))
    for kind, corner, thickness in choices:
        yield kind, corner, thickness, _shape(
            size, box, kind, corner, thickness
        )


def _render(rear, front, background, rear_color, front_color):
    inp = np.full(rear.shape, background, dtype=np.int8)
    inp[rear] = rear_color
    inp[front] = front_color
    moved = np.zeros_like(front)
    moved[:, :-1] = front[:, 1:]
    out = np.full(rear.shape, background, dtype=np.int8)
    out[rear] = rear_color
    out[moved] = front_color
    return inp, out


def _solutions(input_grid):
    """Recover all completions in the exposed solid/frame/line/L grammar."""
    colors, counts = np.unique(input_grid, return_counts=True)
    if len(colors) != 3:
        return []
    background = int(colors[int(np.argmax(counts))])
    if int(counts.max()) <= input_grid.size // 2:
        return []
    results = []
    union_box = _bbox(input_grid != background)
    for front_color in (int(c) for c in colors if c != background):
        rear_color = next(int(c) for c in colors
                          if c != background and c != front_color)
        front_visible = input_grid == front_color
        rear_visible = input_grid == rear_color
        front_box = _bbox(front_visible)
        rear_box = _bbox(rear_visible)
        if front_box is None or rear_box is None or front_box[1] == 0:
            continue
        for front_kind, _, _, front in _shape_candidates(
            input_grid.shape, front_box
        ):
            if not np.array_equal(front, front_visible):
                continue
            rear_bottom = rear_box[0] + rear_box[2]
            rear_right = rear_box[1] + rear_box[3]
            union_bottom = union_box[0] + union_box[2]
            union_right = union_box[1] + union_box[3]
            for top in range(rear_box[0], union_box[0] - 1, -1):
                for left in range(rear_box[1], union_box[1] - 1, -1):
                    for bottom in range(rear_bottom, union_bottom + 1):
                        for right in range(rear_right, union_right + 1):
                            box = (top, left, bottom - top, right - left)
                            for rear_kind, _, _, rear in _shape_candidates(
                                input_grid.shape, box
                            ):
                                if not np.any(rear & front):
                                    continue
                                if not np.array_equal(
                                    rear & ~front, rear_visible
                                ):
                                    continue
                                _, output = _render(
                                    rear, front, background,
                                    rear_color, front_color
                                )
                                if any(
                                    np.array_equal(output, old["output"])
                                    for old in results
                                ):
                                    continue
                                results.append({
                                    "output": output,
                                    "rear": rear,
                                    "front": front,
                                    "rear_box": box,
                                    "front_box": front_box,
                                    "rear_kind": rear_kind,
                                    "front_kind": front_kind,
                                    "rear_color": rear_color,
                                    "front_color": front_color,
                                    "background": background,
                                })
    return results


def _counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _size(rng, kind, previous=None):
    for _ in range(20):
        pick = int(rng.integers(0, 5))
        if pick == 0:
            size = (8, 8)
        elif pick == 1:
            size = (10, 10)
        else:
            size = (int(rng.integers(8, 15)),
                    int(rng.integers(8, 17)))
        if kind == "frame" and size[1] < 9:
            continue
        if size != previous:
            return size
    return (9, 12) if previous != (9, 12) else (10, 13)


def _box_dimensions(rng, kind, size, shallow=False):
    height, width = size
    if kind == "line":
        if int(rng.integers(0, 2)):
            return 1, int(rng.integers(5, min(width, 11) + 1))
        return int(rng.integers(5, min(height, 11) + 1)), 1
    if kind == "frame":
        if shallow:
            return 4, 8
        return (
            int(rng.integers(4, min(height - 1, 8) + 1)),
            int(rng.integers(4, min(width - 1, 9) + 1)),
        )
    if kind == "l":
        return (
            int(rng.integers(4, min(height - 1, 8) + 1)),
            int(rng.integers(4, min(width - 1, 8) + 1)),
        )
    return (
        int(rng.integers(3, min(height - 1, 7) + 1)),
        int(rng.integers(3, min(width - 1, 7) + 1)),
    )


def _front_dimensions(rng, kind, size, rear_box, equal=False,
                      perpendicular=None):
    height, width = size
    if equal:
        return rear_box[2], rear_box[3]
    if kind == "line":
        if perpendicular == "horizontal":
            return int(rng.integers(5, min(height, 11) + 1)), 1
        if perpendicular == "vertical":
            return 1, int(rng.integers(5, min(width, 11) + 1))
        if int(rng.integers(0, 2)):
            return 1, int(rng.integers(4, min(width, 11) + 1))
        return int(rng.integers(4, min(height, 11) + 1)), 1
    if kind == "frame":
        return (
            int(rng.integers(3, min(height - 1, 6) + 1)),
            int(rng.integers(3, min(width - 1, 7) + 1)),
        )
    if kind == "l":
        return (
            int(rng.integers(3, min(height - 1, 6) + 1)),
            int(rng.integers(3, min(width - 1, 6) + 1)),
        )
    return (
        int(rng.integers(2, min(height - 1, 6) + 1)),
        int(rng.integers(2, min(width - 1, 7) + 1)),
    )


def _overlap_class(rear_box, front_box):
    rt, rl, rh, rw = rear_box
    ft, fl, fh, fw = front_box
    if rt <= ft and rl <= fl and ft + fh <= rt + rh and fl + fw <= rl + rw:
        return "inside"
    vertical = ft < rt or ft + fh > rt + rh
    horizontal = fl < rl or fl + fw > rl + rw
    return "corner" if vertical and horizontal else "side"


def _sample_pair(rng, rear_kind, front_kind, geometry, colors,
                 preferred_size=None, shallow=False):
    background, rear_color, front_color = colors
    for attempt in range(900):
        size = preferred_size if attempt < 450 and preferred_size else _size(
            rng, rear_kind
        )
        height, width = size
        rear_height, rear_width = _box_dimensions(
            rng, rear_kind, size, shallow
        )
        rear_box = (
            int(rng.integers(0, height - rear_height + 1)),
            int(rng.integers(0, width - rear_width + 1)),
            rear_height,
            rear_width,
        )
        equal = geometry == "equal" and rear_kind == front_kind == "solid"
        perpendicular = None
        if rear_kind == front_kind == "line":
            perpendicular = "horizontal" if rear_height == 1 else "vertical"
        front_height, front_width = _front_dimensions(
            rng, front_kind, size, rear_box, equal, perpendicular
        )
        if front_width >= width:
            continue
        front_top_min = max(0, rear_box[0] - front_height + 1)
        front_top_max = min(height - front_height,
                            rear_box[0] + rear_height - 1)
        front_left_min = max(1, rear_box[1] - front_width + 1)
        front_left_max = min(width - front_width,
                             rear_box[1] + rear_width - 1)
        if front_top_min > front_top_max or front_left_min > front_left_max:
            continue
        front_box = (
            int(rng.integers(front_top_min, front_top_max + 1)),
            int(rng.integers(front_left_min, front_left_max + 1)),
            front_height,
            front_width,
        )
        overlap_class = _overlap_class(rear_box, front_box)
        requested_geometry = (
            geometry if attempt < 450 or geometry in ("inside", "equal")
            else "any"
        )
        if requested_geometry == "inside" and overlap_class != "inside":
            continue
        if requested_geometry in ("side", "corner") and overlap_class != requested_geometry:
            continue
        if equal and overlap_class == "inside":
            continue
        rear_corner = int(rng.integers(0, 4))
        front_corner = int(rng.integers(0, 4))
        rear_thickness = int(rng.integers(1, 3))
        front_thickness = int(rng.integers(1, 3))
        rear = _shape(size, rear_box, rear_kind, rear_corner, rear_thickness)
        front = _shape(size, front_box, front_kind,
                       front_corner, front_thickness)
        overlap = int(np.count_nonzero(rear & front))
        if overlap == 0 or overlap * 4 > int(np.count_nonzero(rear)) * 3:
            continue
        if not np.array_equal(_bbox(rear & ~front), rear_box):
            continue
        inp, out = _render(
            rear, front, background, rear_color, front_color
        )
        if (
            int(np.count_nonzero(inp == background)) <= inp.size // 2
            or np.array_equal(inp, out)
            or _text(inp) in _SOURCE_INPUTS
            or set(int(c) for c in np.unique(inp))
            != set(int(c) for c in np.unique(out))
        ):
            continue
        solutions = _solutions(inp)
        if len(solutions) != 1:
            continue
        solution = solutions[0]
        if (
            solution["rear_kind"] != rear_kind
            or solution["front_kind"] != front_kind
            or not np.array_equal(solution["rear"], rear)
            or not np.array_equal(solution["front"], front)
            or not np.array_equal(solution["output"], out)
        ):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError(
        f"Could not place {rear_kind}/{front_kind}/{geometry}"
    )


def _palette(rng):
    values = rng.choice(np.arange(10), size=3, replace=False)
    return tuple(int(x) for x in values)


def _same_puzzles(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a[key], b[key])
            for a, b in zip(first[split], second[split])
            for key in ("input", "output")
        )
        for split in ("train", "test")
    )


def _directions_shown(train_pairs):
    candidates = {"left", "right", "up", "down"}
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        colors, counts = np.unique(inp, return_counts=True)
        background = int(colors[int(np.argmax(counts))])
        local = set()
        for color in (int(c) for c in colors if c != background):
            mask = inp == color
            for direction in candidates:
                shifted = np.zeros_like(mask)
                if direction == "left":
                    shifted[:, :-1] = mask[:, 1:]
                elif direction == "right":
                    shifted[:, 1:] = mask[:, :-1]
                elif direction == "up":
                    shifted[:-1, :] = mask[1:, :]
                else:
                    shifted[1:, :] = mask[:-1, :]
                if np.array_equal(shifted, out == color):
                    local.add(direction)
        candidates &= local
    return candidates


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    shared_palette = bool(rng.integers(0, 4) == 0)
    base = _palette(rng)
    homogeneous = bool(rng.integers(0, 5) == 0)
    special = ("solid", "frame", "line", "l")[int(rng.integers(0, 4))]
    other_front = ("solid", "line", "l", "frame")[
        int(rng.integers(0, 4))
    ]
    if homogeneous:
        first_front = "line" if special == "line" else "solid"
        train_specs = [(special, first_front), (special, other_front)]
    else:
        train_specs = [("solid", "solid"), (special, other_front)]
    for _ in range(2, train_count):
        train_specs.append((
            special if homogeneous else
            ("solid", "frame", "line", "l")[int(rng.integers(0, 4))],
            ("solid", "line", "l", "frame")[int(rng.integers(0, 4))],
        ))
    # Each test shape family has appeared in training; combinations can be new.
    rear_families = list(dict.fromkeys(spec[0] for spec in train_specs))
    front_families = list(dict.fromkeys(spec[1] for spec in train_specs))
    rng.shuffle(rear_families)
    test_specs = []
    for index in range(test_count):
        rear_kind = rear_families[index % len(rear_families)]
        front_kind = front_families[
            int(rng.integers(0, len(front_families)))
        ]
        test_specs.append((rear_kind, front_kind))

    puzzle = {"train": [], "test": []}
    seen = set()
    all_specs = train_specs + test_specs
    for index, (rear_kind, front_kind) in enumerate(all_specs):
        if shared_palette:
            background, first, second = base
        elif index == 0:
            background, first, second = base
        elif index == 1:
            background = next(
                c for c in range(10)
                if c not in base and c != base[0]
            )
            first, second = base[1], base[2]
        else:
            background, first, second = _palette(rng)
        if index == 0:
            rear_color, front_color = first, second
        elif index == 1:
            rear_color, front_color = second, first
        elif bool(rng.integers(0, 2)):
            rear_color, front_color = first, second
        else:
            rear_color, front_color = second, first
        if index == 0:
            if rear_kind == front_kind == "solid":
                geometry = "equal" if bool(rng.integers(0, 3) == 0) else "inside"
            else:
                geometry = "any"
        elif index == 1:
            geometry = "side"
        else:
            options = ["any", "side", "corner"]
            if rear_kind == front_kind == "solid":
                options += ["inside", "equal"]
            geometry = options[int(rng.integers(0, len(options)))]
        if index == 0:
            size = (8, 8) if int(rng.integers(0, 2)) else _size(
                rng, rear_kind
            )
        else:
            size = _size(
                rng, rear_kind,
                puzzle["train"][0]["input"].shape if index == 1 else None,
            )
        shallow = rear_kind == "frame" and bool(rng.integers(0, 3) == 0)
        for _ in range(12):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _sample_pair(
                pair_rng, rear_kind, front_kind, geometry,
                (background, rear_color, front_color), size, shallow
            )
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                split = "train" if index < train_count else "test"
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Could not avoid duplicate inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list)
               for split in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count
    inputs, pairs = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {
                "input", "output"
            }
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert all(1 <= value <= 30 for value in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            assert not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            pair_key = (key, out.tobytes())
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert _text(inp) not in _SOURCE_INPUTS
            solutions = _solutions(inp)
            assert len(solutions) == 1
            answer = solutions[0]
            assert np.array_equal(answer["output"], out)
            assert set(int(c) for c in np.unique(inp)) == set(
                int(c) for c in np.unique(out)
            )
            assert answer["front_box"][1] >= 1
            assert np.any(answer["rear"] & answer["front"])
            assert _bbox(answer["rear"] & ~answer["front"]) == answer["rear_box"]
            features[split].append(answer)
    train_rear = {item["rear_kind"] for item in features["train"]}
    train_front = {item["front_kind"] for item in features["train"]}
    assert all(item["rear_kind"] in train_rear
               and item["front_kind"] in train_front
               for item in features["test"])
    assert _directions_shown(puzzle["train"]) == {"left"}
    assert len({pair["input"].shape for pair in puzzle["train"]}) >= 2
    assert (
        features["train"][0]["front_color"]
        == features["train"][1]["rear_color"]
    )
    assert (
        features["train"][0]["rear_color"]
        == features["train"][1]["front_color"]
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(seed))
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    signatures = {
        tuple(
            (pair["input"].shape, pair["input"].tobytes())
            for split in ("train", "test")
            for pair in puzzle[split]
        )
        for puzzle in puzzles
    }
    assert len(signatures) == 42
    all_pairs = [
        pair for puzzle in puzzles
        for split in ("train", "test")
        for pair in puzzle[split]
    ]
    solutions = [_solutions(pair["input"])[0] for pair in all_pairs]
    assert {item["rear_kind"] for item in solutions} == {
        "solid", "frame", "line", "l"
    }
    assert {item["front_kind"] for item in solutions} == {
        "solid", "frame", "line", "l"
    }
    assert any(item["rear_kind"] == "frame"
               and item["rear_box"][2:] == (4, 8)
               for item in solutions)
    assert any(item["rear_kind"] == item["front_kind"] == "solid"
               and item["rear_box"][2:] == item["front_box"][2:]
               for item in solutions)
    assert any(pair["input"].shape[0] != pair["input"].shape[1]
               for pair in all_pairs)
    assert len({pair["input"].shape for pair in all_pairs}) >= 12
    assert len({item["background"] for item in solutions}) >= 5
    assert len({(item["rear_color"], item["front_color"])
                for item in solutions}) >= 12
    assert sum(
        pair["input"].shape[0] != pair["input"].shape[1]
        for pair in all_pairs
    ) >= 80
    overlap_classes = {
        _overlap_class(item["rear_box"], item["front_box"])
        for item in solutions
    }
    assert overlap_classes == {"inside", "side", "corner"}
    assert sum(
        item["front_kind"] == "line" and item["front_box"][2] == 1
        and item["front_box"][3] >= 5 for item in solutions
    ) >= 5
    assert sum(
        item["front_kind"] == "line" and item["front_box"][3] == 1
        and item["front_box"][2] >= 5 for item in solutions
    ) >= 5
    assert sum(
        item["rear_kind"] == "frame" and item["rear_box"][2:] == (4, 8)
        for item in solutions
    ) >= 5
    assert sum(
        item["rear_kind"] == item["front_kind"] == "solid"
        and item["rear_box"][2:] == item["front_box"][2:]
        for item in solutions
    ) >= 5
    assert sum(
        pair["input"].shape == (8, 8)
        and _solutions(pair["input"])[0]["rear_kind"] == "solid"
        for puzzle in puzzles for pair in puzzle["test"]
    ) >= 3
    assert sum(
        any(_solutions(pair["input"])[0]["rear_kind"] == "frame"
            for pair in puzzle["train"])
        for puzzle in puzzles
    ) >= 8
    assert sum(
        any(_solutions(pair["input"])[0]["rear_kind"] == "line"
            for pair in puzzle["train"])
        for puzzle in puzzles
    ) >= 8
    assert sum(
        len({
            _solutions(pair["input"])[0]["rear_kind"]
            for split in ("train", "test") for pair in puzzle[split]
        }) == 1 for puzzle in puzzles
    ) >= 5
    assert sum(
        len({
            tuple(sorted(int(c) for c in np.unique(pair["input"])))
            for split in ("train", "test") for pair in puzzle[split]
        }) > 1 for puzzle in puzzles
    ) >= 20
    assert sum(
        any(
            (
                _solutions(pair["input"])[0]["rear_kind"],
                _solutions(pair["input"])[0]["front_kind"],
            ) not in {
                (
                    _solutions(example["input"])[0]["rear_kind"],
                    _solutions(example["input"])[0]["front_kind"],
                )
                for example in puzzle["train"]
            }
            for pair in puzzle["test"]
        )
        for puzzle in puzzles
    ) >= 10
    rear_revealed = background_revealed = 0
    for item in solutions:
        shifted = np.zeros_like(item["front"])
        shifted[:, :-1] = item["front"][:, 1:]
        vacated = item["front"] & ~shifted
        rear_revealed += bool(np.any(vacated & item["rear"]))
        background_revealed += bool(np.any(vacated & ~item["rear"]))
    assert rear_revealed >= 50 and background_revealed >= 50
    for train_count, test_count in (
        (2, 1), (10, 1), (10, 4), (7, 3),
        (None, 1), (None, 4), (2, None), (10, None),
    ):
        seed = 1000 + (train_count or 0) * 10 + (test_count or 0)
        first = generate_puzzle(seed, train_count, test_count)
        second = generate_puzzle(seed, train_count, test_count)
        assert _same_puzzles(first, second)
        validate_generated_puzzle(first)
        if train_count is not None:
            assert len(first["train"]) == train_count
        if test_count is not None:
            assert len(first["test"]) == test_count
    for train_count, test_count in (
        (1, None), (11, None), (None, 0),
        (None, 5), (2, 2), (3, 4),
    ):
        try:
            generate_puzzle(0, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
