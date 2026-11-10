import numpy as np


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _literal_grid_signature(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_PAIR_ROWS = (
    (
        (
            "0000000000",
            "0000000000",
            "0110110000",
            "0102010000",
            "0100010000",
            "0111110000",
            "0000000000",
            "0000110110",
            "0000103010",
            "0000111110",
        ),
        (
            "0000000000",
            "0222220000",
            "0112110000",
            "0122210000",
            "0122210000",
            "0111110000",
            "0000333330",
            "0000113110",
            "0000133310",
            "0000111110",
        ),
    ),
    (
        (
            "0000000000",
            "0110110000",
            "0100010000",
            "0102010000",
            "0100010000",
            "0111110000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0222220000",
            "0112110000",
            "0122210000",
            "0122210000",
            "0122210000",
            "0111110000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0110110000",
            "0106010000",
            "0100010000",
            "0111110000",
            "0000000000",
            "0000000000",
            "0000110110",
            "0000108010",
            "0000111110",
        ),
        (
            "0666660000",
            "0116110000",
            "0166610000",
            "0166610000",
            "0111110000",
            "0000000000",
            "0000888880",
            "0000118110",
            "0000188810",
            "0000111110",
        ),
    ),
    (
        (
            "0000000000",
            "1101100000",
            "1040100000",
            "1000100000",
            "1111100000",
            "0000000000",
            "0000110110",
            "0000107010",
            "0000100010",
            "0000111110",
        ),
        (
            "4444400000",
            "1141100000",
            "1444100000",
            "1444100000",
            "1111100000",
            "0000777770",
            "0000117110",
            "0000177710",
            "0000177710",
            "0000111110",
        ),
    ),
)

_SOURCE_PAIR_SIGNATURES = frozenset(
    (
        _literal_grid_signature(input_rows),
        _literal_grid_signature(output_rows),
    )
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_pair_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This balanced seed mapping guarantees count diversity while
        # remaining completely deterministic.
        test_count = seed % 4 + 1
        number_of_legal_train_counts = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % number_of_legal_train_counts
        )
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    if maximum_test < 1:
        raise ValueError("no legal test count exists")
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


DIRECTIONS = ("up", "down", "left", "right")


def _bounds(spec: dict) -> tuple[int, int, int, int]:
    top, left = spec["top"], spec["left"]
    return top, top + spec["height"] - 1, left, left + spec["width"] - 1


def _footprint(spec: dict) -> tuple[int, int, int, int]:
    top, bottom, left, right = _bounds(spec)
    direction = spec["direction"]
    return (
        top - (direction == "up"),
        bottom + (direction == "down"),
        left - (direction == "left"),
        right + (direction == "right"),
    )


def _separate(a: tuple, b: tuple, gap: int = 0) -> bool:
    return (
        a[1] + gap < b[0] or b[1] + gap < a[0]
        or a[3] + gap < b[2] or b[3] + gap < a[2]
    )


def _opening(spec: dict) -> tuple[int, int]:
    top, bottom, left, right = _bounds(spec)
    offset = spec["opening"]
    direction = spec["direction"]
    if direction == "up":
        return top, left + offset
    if direction == "down":
        return bottom, left + offset
    if direction == "left":
        return top + offset, left
    return top + offset, right


def _bar(spec: dict) -> tuple[slice, slice]:
    top, bottom, left, right = _bounds(spec)
    direction = spec["direction"]
    if direction == "up":
        return slice(top - 1, top), slice(left, right + 1)
    if direction == "down":
        return slice(bottom + 1, bottom + 2), slice(left, right + 1)
    if direction == "left":
        return slice(top, bottom + 1), slice(left - 1, left)
    return slice(top, bottom + 1), slice(right + 1, right + 2)


def _draw_pair(shape: tuple[int, int], background: int,
               structure: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, background, dtype=np.int8)
    for spec in specs:
        top, bottom, left, right = _bounds(spec)
        inp[top, left:right + 1] = structure
        inp[bottom, left:right + 1] = structure
        inp[top:bottom + 1, left] = structure
        inp[top:bottom + 1, right] = structure
        opening_row, opening_col = _opening(spec)
        inp[opening_row, opening_col] = background
        inp[top + spec["seed_row"], left + spec["seed_col"]] = spec["accent"]
    out = inp.copy()
    for spec in specs:
        top, bottom, left, right = _bounds(spec)
        bar_rows, bar_cols = _bar(spec)
        out[bar_rows, bar_cols] = spec["accent"]
        opening_row, opening_col = _opening(spec)
        out[opening_row, opening_col] = spec["accent"]
        out[top + 1:bottom, left + 1:right] = spec["accent"]
    return inp, out


def _sample_spec(rng: np.random.Generator, direction: str,
                 count: int, source_like: bool, force_off_axis: bool) -> dict:
    if source_like:
        height = int(rng.integers(4, 7))
        width = 5
    elif force_off_axis:
        height = int(rng.integers(5, 9))
        width = int(rng.integers(5, 9))
    else:
        maximum = 10 if count == 1 else (9 if count == 2 else 8)
        height = int(rng.integers(3, maximum + 1))
        width = int(rng.integers(3, maximum + 2))
    side_length = width if direction in ("up", "down") else height
    center = (side_length - 1) // 2
    if source_like:
        opening = center
    elif force_off_axis:
        choices = [x for x in range(1, side_length - 1) if x != center]
        opening = int(rng.choice(choices))
    elif rng.random() < 0.35:
        opening = center
    else:
        opening = int(rng.integers(1, side_length - 1))
    if source_like:
        if direction in ("up", "down"):
            seed_row = int(rng.integers(1, height - 1))
            seed_col = opening
        else:
            seed_row = opening
            seed_col = int(rng.integers(1, width - 1))
    else:
        seed_row = int(rng.integers(1, height - 1))
        seed_col = int(rng.integers(1, width - 1))
        if force_off_axis:
            if direction in ("up", "down"):
                choices = [x for x in range(1, width - 1) if x != opening]
                seed_col = int(rng.choice(choices))
            else:
                choices = [x for x in range(1, height - 1) if x != opening]
                seed_row = int(rng.choice(choices))
    return {
        "top": 0, "left": 0, "height": height, "width": width,
        "direction": direction, "opening": opening,
        "seed_row": seed_row, "seed_col": seed_col, "accent": 0,
    }


def _choose_position(rng: np.random.Generator, minimum: int,
                     maximum: int) -> int:
    if minimum == maximum:
        return minimum
    roll = rng.random()
    if roll < 0.23:
        return minimum
    if roll < 0.46:
        return maximum
    return int(rng.integers(minimum, maximum + 1))


def _place_specs(rng: np.random.Generator, specs: list[dict],
                 shape: tuple[int, int]) -> bool:
    placed: list[dict] = []
    grid_height, grid_width = shape
    for spec in specs:
        height, width = spec["height"], spec["width"]
        direction = spec["direction"]
        min_top = int(direction == "up")
        max_top = grid_height - height - int(direction == "down")
        min_left = int(direction == "left")
        max_left = grid_width - width - int(direction == "right")
        if max_top < min_top or max_left < min_left:
            return False
        positions = [
            (_choose_position(rng, min_top, max_top),
             _choose_position(rng, min_left, max_left))
            for _ in range(90)
        ]
        all_positions = [
            (row, col)
            for row in range(min_top, max_top + 1)
            for col in range(min_left, max_left + 1)
        ]
        rng.shuffle(all_positions)
        positions.extend(all_positions)
        found = False
        for row, col in positions:
            spec["top"], spec["left"] = row, col
            if all(
                _separate(_footprint(spec), _footprint(other))
                and _separate(_bounds(spec), _bounds(other), gap=1)
                for other in placed
            ):
                found = True
                placed.append(spec)
                break
        if not found:
            return False
    return True


def _generate_pair(rng: np.random.Generator, regime: str,
                   pair_index: int, previous_background: int | None,
                   simple_direction: str | None) -> tuple[np.ndarray, np.ndarray]:
    if regime == "source_train" and pair_index == 0:
        count = 1
        directions = [simple_direction or "up"]
        source_like = True
        force_cues = False
    elif regime == "source_train" and pair_index == 1:
        if simple_direction is None:
            count = 4
            directions = list(DIRECTIONS)
            rng.shuffle(directions)
        else:
            count = 2
            directions = [simple_direction] * count
        source_like = False
        force_cues = True
    else:
        probabilities = ([0.24, 0.36, 0.27, 0.13]
                         if regime == "source_train" else
                         [0.15, 0.27, 0.36, 0.22])
        count = int(rng.choice(np.arange(1, 5), p=probabilities))
        directions = ([simple_direction] * count if simple_direction is not None
                      else [str(rng.choice(DIRECTIONS)) for _ in range(count)])
        source_like = False
        force_cues = False
    colors = [int(x) for x in rng.permutation(10)]
    if previous_background is not None and colors[0] == previous_background:
        replacement = next(i for i in range(1, 10) if colors[i] != previous_background)
        colors[0], colors[replacement] = colors[replacement], colors[0]
    background, structure = colors[:2]
    accent_palette = colors[2:]
    accents = [int(rng.choice(accent_palette)) for _ in range(count)]
    if count >= 2 and (force_cues or rng.random() < 0.4):
        accents[-1] = accents[0]
    for attempt in range(80):
        specs = [
            _sample_spec(rng, direction, count, source_like, force_cues)
            for direction in directions
        ]
        if force_cues and len({(s["height"], s["width"]) for s in specs}) < 2:
            continue
        for spec, accent in zip(specs, accents):
            spec["accent"] = accent
        if count == 1:
            min_side, max_side = 8, 22
        elif count == 2:
            min_side, max_side = 12, 28
        elif count == 3:
            min_side, max_side = 18, 30
        else:
            min_side, max_side = 24, 30
        if attempt >= 50:
            shape = (30, 30)
        else:
            shape = (
                int(rng.integers(min_side, max_side + 1)),
                int(rng.integers(min_side, max_side + 1)),
            )
        if not _place_specs(rng, specs, shape):
            continue
        inp, out = _draw_pair(shape, background, structure, specs)
        if (inp == background).sum() <= inp.size // 2:
            continue
        if (_grid_signature(inp), _grid_signature(out)) in _SOURCE_PAIR_SIGNATURES:
            continue
        return inp, out
    raise RuntimeError("could not place a valid frame arrangement")


def _structural_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for start in np.argwhere(mask):
        start_row, start_col = map(int, start)
        if visited[start_row, start_col]:
            continue
        component = set()
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        while stack:
            row, col = stack.pop()
            component.add((row, col))
            for rr, cc in ((row-1,col), (row+1,col), (row,col-1), (row,col+1)):
                if (0 <= rr < height and 0 <= cc < width
                        and mask[rr, cc] and not visited[rr, cc]):
                    visited[rr, cc] = True
                    stack.append((rr, cc))
        components.append(component)
    return components


def _analyze_input(inp: np.ndarray) -> dict:
    height, width = inp.shape
    values, counts = np.unique(inp, return_counts=True)
    majority = np.flatnonzero(counts == counts.max())
    assert len(majority) == 1
    background = int(values[majority[0]])
    assert int(counts[majority[0]]) > inp.size // 2
    non_background = inp[inp != background]
    values, counts = np.unique(non_background, return_counts=True)
    dominant = np.flatnonzero(counts == counts.max())
    assert len(dominant) == 1
    structure = int(values[dominant[0]])
    assert structure != background
    components = _structural_components(inp == structure)
    assert 1 <= len(components) <= 4
    objects = []
    recognized = inp == structure
    expected = inp.copy()
    for component in components:
        rows, cols = zip(*component)
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        frame_height, frame_width = bottom-top+1, right-left+1
        assert 3 <= frame_height <= 10 and 3 <= frame_width <= 11
        border = {
            (row, col)
            for row in range(top, bottom+1)
            for col in range(left, right+1)
            if row in (top, bottom) or col in (left, right)
        }
        missing = border - component
        assert len(missing) == 1
        assert component == border - missing
        opening_row, opening_col = next(iter(missing))
        assert inp[opening_row, opening_col] == background
        assert opening_row not in (top, bottom) or opening_col not in (left, right)
        if opening_row == top:
            direction, offset = "up", opening_col-left
            bar = (slice(top-1, top), slice(left, right+1))
            assert top >= 1
        elif opening_row == bottom:
            direction, offset = "down", opening_col-left
            bar = (slice(bottom+1, bottom+2), slice(left, right+1))
            assert bottom+1 < height
        elif opening_col == left:
            direction, offset = "left", opening_row-top
            bar = (slice(top, bottom+1), slice(left-1, left))
            assert left >= 1
        else:
            direction, offset = "right", opening_row-top
            bar = (slice(top, bottom+1), slice(right+1, right+2))
            assert right+1 < width
        assert np.all(inp[bar] == background)
        interior = inp[top+1:bottom, left+1:right]
        seeds = np.argwhere(interior != background)
        assert len(seeds) == 1
        seed_row = top+1+int(seeds[0, 0])
        seed_col = left+1+int(seeds[0, 1])
        accent = int(inp[seed_row, seed_col])
        assert accent not in (background, structure)
        recognized[seed_row, seed_col] = True
        expected[bar] = accent
        expected[opening_row, opening_col] = accent
        expected[top+1:bottom, left+1:right] = accent
        objects.append({
            "top": top, "bottom": bottom, "left": left, "right": right,
            "height": frame_height, "width": frame_width,
            "direction": direction, "opening": offset,
            "seed_row": seed_row, "seed_col": seed_col,
            "accent": accent,
            "footprint": (
                top-int(direction == "up"), bottom+int(direction == "down"),
                left-int(direction == "left"), right+int(direction == "right"),
            ),
        })
    assert np.all((inp == background) | recognized)
    for i, first in enumerate(objects):
        for second in objects[i+1:]:
            assert _separate(first["footprint"], second["footprint"])
            assert _separate(
                (first["top"], first["bottom"], first["left"], first["right"]),
                (second["top"], second["bottom"], second["left"], second["right"]),
                gap=1,
            )
    objects.sort(key=lambda obj: (obj["top"], obj["left"]))
    return {
        "background": background, "structure": structure,
        "objects": objects, "expected_output": expected,
        "shape": (height, width),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen_inputs, seen_pairs = set(), set()
    train_infos, test_infos = [], []
    for split, infos in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert inp.shape == out.shape
            input_signature = _grid_signature(inp)
            pair_signature = (input_signature, _grid_signature(out))
            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            assert pair_signature not in _SOURCE_PAIR_SIGNATURES
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            info = _analyze_input(inp)
            assert np.array_equal(info["expected_output"], out)
            assert not np.array_equal(inp, out)
            infos.append(info)
    assert len(train_infos[0]["objects"]) == 1
    assert len(train_infos[1]["objects"]) >= 2
    assert train_infos[0]["background"] != train_infos[1]["background"]
    train_objects = [obj for info in train_infos for obj in info["objects"]]
    train_directions = {obj["direction"] for obj in train_objects}
    assert len(train_directions) == 1 or train_directions == set(DIRECTIONS)
    assert all(
        obj["direction"] in train_directions
        for info in test_infos for obj in info["objects"]
    )
    assert len({(obj["height"], obj["width"]) for obj in train_objects}) >= 2
    assert any(_off_center(obj) for obj in train_objects)
    assert any(_off_axis(obj) for obj in train_objects)
    # No puzzle-level literal palette, dimensions, or placement rule exists.
    # Every test direction and cue geometry is already shown in training.
    train_layouts = {_layout_signature(info) for info in train_infos}
    assert all(_layout_signature(info) not in train_layouts for info in test_infos)


def _off_center(obj: dict) -> bool:
    side = obj["width"] if obj["direction"] in ("up", "down") else obj["height"]
    return obj["opening"] != (side - 1) // 2


def _off_axis(obj: dict) -> bool:
    if obj["direction"] in ("up", "down"):
        return obj["seed_col"] != obj["left"] + obj["opening"]
    return obj["seed_row"] != obj["top"] + obj["opening"]


def _layout_signature(info: dict) -> tuple:
    return (info["shape"], tuple(
        (obj["top"], obj["left"], obj["height"], obj["width"],
         obj["direction"], obj["opening"], obj["seed_row"], obj["seed_col"])
        for obj in info["objects"]
    ))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_pair_counts(rng, seed_value, num_train, num_test)
    simple_direction = (str(rng.choice(DIRECTIONS)) if rng.random() < 0.2
                        else None)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_background = None
    for split, count in (("train", train_count), ("test", test_count)):
        for pair_index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = _generate_pair(pair_rng,
                                          "source_train" if split == "train" else "target_test",
                                          pair_index, previous_background,
                                          simple_direction)
                signature = _grid_signature(inp)
                if signature not in seen:
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            seen.add(signature)
            previous_background = int(_analyze_input(inp)["background"])
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (split, tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
                      for pair in puzzle[split]))
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, fingerprints = set(), set(), set()
    shapes, layouts, directions, dimensions, counts = set(), set(), set(), set(), set()
    backgrounds, structures = set(), set()
    seen_single = seen_two = seen_several = False
    seen_mixed_directions = seen_mixed_dimensions = False
    seen_off_center = seen_off_axis = seen_edge = seen_width_ten = False
    seen_shared_accent = seen_lower_taller = seen_compact = False
    simple_puzzles = mixed_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        all_directions = {
            obj["direction"]
            for split in ("train", "test") for pair in puzzle[split]
            for obj in _analyze_input(pair["input"])["objects"]
        }
        simple_puzzles += len(all_directions) == 1
        mixed_puzzles += len(all_directions) > 1
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        palette_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                assert np.array_equal(pair["output"], info["expected_output"])
                palette_roles.add((info["background"], info["structure"]))
                backgrounds.add(info["background"])
                structures.add(info["structure"])
                shapes.add(info["shape"])
                layouts.add(_layout_signature(info))
                objects = info["objects"]
                counts.add(len(objects))
                seen_single |= len(objects) == 1
                seen_two |= len(objects) == 2
                seen_several |= len(objects) >= 3
                seen_mixed_directions |= len({obj["direction"] for obj in objects}) >= 2
                seen_mixed_dimensions |= len({(obj["height"], obj["width"]) for obj in objects}) >= 2
                seen_shared_accent |= len(objects) >= 2 and len({obj["accent"] for obj in objects}) < len(objects)
                for obj in objects:
                    directions.add(obj["direction"])
                    dimensions.add((obj["height"], obj["width"]))
                    seen_off_center |= _off_center(obj)
                    seen_off_axis |= _off_axis(obj)
                    seen_edge |= (obj["top"] == 0 or obj["bottom"] == info["shape"][0]-1
                                  or obj["left"] == 0 or obj["right"] == info["shape"][1]-1)
                    seen_width_ten |= info["shape"][1] == 10
                if len(objects) >= 2:
                    for a in objects:
                        for b in objects:
                            if a["top"] < b["top"] and a["bottom"] < b["top"]:
                                seen_lower_taller |= b["height"] > a["height"]
                                seen_compact |= b["top"] - a["bottom"] <= 3
        assert len(palette_roles) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(shapes) >= 50 and len(layouts) >= 150
    assert directions == set(DIRECTIONS) and len(dimensions) >= 25
    assert len(backgrounds) >= 8 and len(structures) >= 8
    assert simple_puzzles >= 3 and mixed_puzzles >= 20
    assert counts == {1, 2, 3, 4}
    assert all((seen_single, seen_two, seen_several, seen_mixed_directions,
                seen_mixed_dimensions, seen_off_center, seen_off_axis, seen_edge,
                seen_width_ten, seen_shared_accent, seen_lower_taller, seen_compact))
    assert _puzzle_fingerprint(generate_puzzle(314159)) == _puzzle_fingerprint(generate_puzzle(314159))
    assert _puzzle_fingerprint(generate_puzzle(271828, 7, 3)) == _puzzle_fingerprint(generate_puzzle(271828, 7, 3))
    assert _puzzle_fingerprint(generate_puzzle(81173, 6, 2)) != _puzzle_fingerprint(generate_puzzle(81174, 6, 2))
    for seed, train, test in ((5001,2,1), (5002,10,1), (5003,10,4)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((6001,None,1), (6002,None,4), (6003,2,None), (6004,10,None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train":1}, {"num_train":11}, {"num_test":0},
                   {"num_test":5}, {"num_train":2,"num_test":2},
                   {"num_train":3,"num_test":4}, {"num_train":2.5,"num_test":1},
                   {"num_train":True,"num_test":1}):
        try:
            generate_puzzle(7001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid count: {kwargs}")
