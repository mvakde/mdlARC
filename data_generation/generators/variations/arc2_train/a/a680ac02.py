import numpy as np


# Populated below with the exact source inputs, not their canvas sizes.
_SOURCE_INPUTS = (
    ((21, 16), bytes.fromhex('000000000000000000000000000000000000000000000000080808080000000000000000000000000808080800000000000000000000000008080808000000000000000000000000080808080000000000000000000000000000000000000000000101010100000000000000000000000001000001000000000000000000000000010000010000000000000000000000000101010100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000020202020000000000000000000000000200000200000000000000000000000002000002000000000000000000000000020202020000000404040400000000000000000000000004040404000000000000000000000000040404040000000000000000000000000404040400000000000000000000000000000000000000000000000000000000000000000000')),
    ((18, 17), bytes.fromhex('000000000000000000000000000000000000020202020000000000000000000000000002000002000000000404040400000000000200000200000000040000040000000000020202020000000004000004000000000000000000000000000404040400000000000000000000000000000000000000000000000003030303000000000000000000000000000303030300000000000000000000000000030303030000000000000000000000000003030303000000000000000000000000000000000000000000000000000000000000000000000000000808080800000000000000000000000000080808080000000000000000000000000008080808000000000000000000000000000808080800000000000000000000000000000000000000000000000000000000000000000000000000')),
    ((24, 22), bytes.fromhex('000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000080808080000000101010100000000000000000000000808080800000001010101000000000000000000000008080808000000010101010000000000000000000000080808080000000101010100000003030303000000000000000000000000000000000000030000030000000000000000000000000000000000000300000300000000000000000000000000000000000003030303000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006060606000000000000000004040404000000000000060606060000020202020000040000040000000000000606060600000200000200000400000400000000000006060606000002000002000004040404000000000000000000000000020202020000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')),
    ((22, 21), bytes.fromhex('000000000000000000000000000000000000000000000000030303030000000000000000000000000000000000030000030000000000000000000808080800000000030000030000000000000000000808080800000000030303030000000000000000000808080800000000000000000000000000000000000808080800000000000000020202020000000000000000000000000000000000020000020000000000000000000000000000000000020000020000000000000000000000000000000000020202020000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040404040000000000000000000000000000000000040404040000000000000000000000000000000000040404040000000000000000000000000000000000040404040000000000000000000000000000000000000000000000000000010101010000000000000000000000000000000000010000010000000000000000000000000000000000010000010000000000000000000000000000000000010101010000000000000000000000000000000000000000000000000000')),
)


def _count_value(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count_value("num_train", num_train)
    test = _count_value("num_test", num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + abs(int(seed)) % 4
        low = max(2, test + 1)
        train = low + (abs(int(seed)) // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _clear(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _pair_spec(rng, section, index, first_test_axis):
    if section == "train" and index == 0:
        return "vertical", 2, int(rng.choice([1, 3]))
    if section == "train" and index == 1:
        return "horizontal", 3, int(rng.integers(1, 5))
    if section == "test":
        axis = first_test_axis if index % 2 == 0 else (
            "horizontal" if first_test_axis == "vertical" else "vertical"
        )
        n_frames = int(rng.choice([2, 3, 3, 4, 4, 5]))
        n_solids = int(rng.choice([0, 1, 2, 3, 4, 5]))
        return axis, n_frames, n_solids
    axis = "vertical" if rng.integers(2) else "horizontal"
    return axis, int(rng.choice([2, 3, 3, 4, 5])), int(rng.integers(0, 6))


def _layout(rng, size, n_frames, n_solids, axis):
    # Every gap is drawn separately; the bound keeps the full strip in 30 cells.
    minimum = n_frames * size + n_frames - 1
    if minimum > 30:
        raise ValueError("frames do not fit")
    spare = 30 - minimum
    gaps = np.ones(n_frames - 1, dtype=int)
    for i in rng.permutation(n_frames - 1):
        extra = int(rng.integers(0, min(9, spare) + 1))
        gaps[i] += extra
        spare -= extra
    length = n_frames * size + int(sum(gaps))
    for _ in range(60):
        stagger_max = min(length - size - 1, 11)
        if rng.random() < 0.23:
            stagger = 0
        elif rng.random() < 0.5:
            stagger = int(rng.integers(1, min(size, stagger_max + 1)))
        else:
            stagger = int(rng.integers(min(size, stagger_max), stagger_max + 1))
        offsets = rng.integers(0, stagger + 1, size=n_frames)
        offsets[0], offsets[-1] = 0, stagger
        rng.shuffle(offsets)

        axis_min = max(length, 9 + 2 * n_solids)
        cross_min = max(size + stagger + 2, 9 + 2 * n_solids)
        if axis_min > 30 or cross_min > 30:
            continue
        # Nearby and roomy canvases both occur, independently of object scale.
        def dimension(low):
            high = min(30, low + 4) if rng.random() < 0.42 else 30
            return int(rng.integers(low, high + 1))

        axis_dim = dimension(axis_min)
        cross_dim = dimension(cross_min)
        axis_start = int(rng.integers(0, axis_dim - length + 1))
        cross_start = int(rng.integers(1, cross_dim - size - stagger))
        along = axis_start
        frames = []
        for i in range(n_frames):
            across = cross_start + int(offsets[i])
            frames.append((along, across, size, size) if axis == "vertical"
                          else (across, along, size, size))
            along += size + (int(gaps[i]) if i < n_frames - 1 else 0)
        height, width = (axis_dim, cross_dim) if axis == "vertical" else (cross_dim, axis_dim)
        used = list(frames)
        solids = []
        for _solid in range(n_solids):
            sh = int(rng.integers(2, min(8, size + 3)))
            sw = sh if rng.random() < 0.58 else int(rng.integers(2, min(8, size + 3)))
            candidates = []
            for row in range(height - sh + 1):
                for col in range(width - sw + 1):
                    box = (row, col, sh, sw)
                    if (row, col) in ((0, 0), (0, width - sw),
                                      (height - sh, 0), (height - sh, width - sw)):
                        continue
                    if all(_clear(box, old) for old in used):
                        candidates.append(box)
            if not candidates:
                break
            # Sometimes place a solid in the interval spanned by the frames.
            between = [box for box in candidates if
                       min(item[0 if axis == "vertical" else 1] for item in frames)
                       < box[0 if axis == "vertical" else 1]
                       < max(item[0 if axis == "vertical" else 1] for item in frames)]
            pool = between if between and rng.random() < 0.55 else candidates
            box = pool[int(rng.integers(len(pool)))]
            solids.append(box)
            used.append(box)
        if len(solids) == n_solids:
            return (height, width), frames, solids
    raise RuntimeError("could not place objects")


def _make_pair(rng, section, index, first_test_axis, base_background):
    axis, n_frames, n_solids = _pair_spec(rng, section, index, first_test_axis)
    size_limit = min(7, (30 - n_frames + 1) // n_frames)
    size = int(rng.integers(3, size_limit + 1))
    shape, frames, solids = _layout(rng, size, n_frames, n_solids, axis)
    background = base_background if rng.random() < 0.65 else int(rng.integers(10))
    palette = np.array([c for c in range(10) if c != background], dtype=np.int8)
    colors = rng.choice(palette, size=n_frames + n_solids, replace=True)
    if n_solids and rng.random() < 0.42:
        colors[n_frames + int(rng.integers(n_solids))] = colors[int(rng.integers(n_frames))]
    grid = np.full(shape, background, dtype=np.int8)
    for box, color in zip(frames, colors[:n_frames]):
        r, c, h, w = box
        grid[r:r+h, c:c+w] = color
        grid[r+1:r+h-1, c+1:c+w-1] = background
    for box, color in zip(solids, colors[n_frames:]):
        r, c, h, w = box
        grid[r:r+h, c:c+w] = color
    coordinate = 0 if axis == "vertical" else 1
    ordered = sorted(frames, key=lambda box: box[coordinate])
    output = np.concatenate([grid[r:r+h, c:c+w].copy() for r, c, h, w in ordered],
                            axis=coordinate).astype(np.int8, copy=False)
    return {"input": grid, "output": output}


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or grid[row, col] == background:
                continue
            color = int(grid[row, col])
            pending = [(row, col)]
            seen[row, col] = True
            pixels = []
            while pending:
                r, c = pending.pop()
                pixels.append((r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= nr < height and 0 <= nc < width and
                            not seen[nr, nc] and grid[nr, nc] == color):
                        seen[nr, nc] = True
                        pending.append((nr, nc))
            r0 = min(r for r, _ in pixels)
            c0 = min(c for _, c in pixels)
            h = max(r for r, _ in pixels) - r0 + 1
            w = max(c for _, c in pixels) - c0 + 1
            patch = grid[r0:r0+h, c0:c0+w]
            border = np.ones((h, w), dtype=bool)
            border[1:-1, 1:-1] = False
            frame = (h == w and h >= 3 and np.all(patch[border] == color)
                     and np.all(patch[~border] == background))
            solid = bool(np.all(patch == color))
            assert frame != solid
            objects.append({"box": (r0, c0, h, w), "color": color,
                            "frame": bool(frame)})
    return objects


def _analyze(pair):
    grid, output = pair["input"], pair["output"]
    background = int(grid[0, 0])
    assert grid[0, -1] == grid[-1, 0] == grid[-1, -1] == background
    objects = _components(grid, background)
    for i, first in enumerate(objects):
        assert all(_clear(first["box"], second["box"]) for second in objects[i+1:])
    frames = [item for item in objects if item["frame"]]
    solids = [item for item in objects if not item["frame"]]
    assert 2 <= len(frames) <= 5 and len(solids) <= 5
    sizes = {item["box"][2:] for item in frames}
    assert len(sizes) == 1
    size = next(iter(sizes))[0]
    assert size >= 3
    row_span = max(x["box"][0] for x in frames) - min(x["box"][0] for x in frames)
    col_span = max(x["box"][1] for x in frames) - min(x["box"][1] for x in frames)
    assert row_span != col_span
    axis = 0 if row_span > col_span else 1
    ordered = sorted(frames, key=lambda item: item["box"][axis])
    coordinates = [item["box"][axis] for item in ordered]
    gaps = tuple(coordinates[i+1] - coordinates[i] - size for i in range(len(frames)-1))
    assert all(gap >= 1 for gap in gaps)
    expected = np.concatenate([grid[r:r+h, c:c+w] for item in ordered
                               for r, c, h, w in [item["box"]]], axis=axis)
    assert np.array_equal(output, expected)
    assert output.shape == ((size * len(frames), size) if axis == 0
                            else (size, size * len(frames)))
    all_row_span = max(x["box"][0] for x in objects) - min(x["box"][0] for x in objects)
    all_col_span = max(x["box"][1] for x in objects) - min(x["box"][1] for x in objects)
    return {"axis": "vertical" if axis == 0 else "horizontal", "frames": len(frames),
            "solids": len(solids), "size": size, "background": background,
            "gaps": gaps, "stagger": col_span if axis == 0 else row_span,
            "edge": any(r == 0 or c == 0 or r+h == grid.shape[0] or c+w == grid.shape[1]
                        for item in objects for r, c, h, w in [item["box"]]),
            "frame_edge": any(r == 0 or c == 0 or r+h == grid.shape[0] or c+w == grid.shape[1]
                              for item in frames for r, c, h, w in [item["box"]]),
            "whole_axis_misleading": (all_row_span != all_col_span and
                                      (all_row_span > all_col_span) != (axis == 0)),
            "rect_solid": any(x["box"][2] != x["box"][3] for x in solids),
            "varied_solid_size": any(x["box"][2:] != (size, size) for x in solids),
            "shared_color": len({item["color"] for item in objects}) < len(objects),
            "cross_kind_color": bool({x["color"] for x in frames} &
                                     {x["color"] for x in solids}),
            "between_solid": any(min(coordinates) < x["box"][axis] < max(coordinates)
                                 for x in solids)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert grid.size and 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in _SOURCE_INPUTS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not np.array_equal(pair["input"], pair["output"])
            features[section].append(_analyze(pair))
    # The first two demonstrations establish both packing directions and show
    # that solids are discarded even when their count differs from the frames.
    assert features["train"][0]["axis"] == "vertical"
    assert features["train"][0]["frames"] == 2
    assert features["train"][0]["solids"] >= 1
    assert features["train"][0]["solids"] != 2
    assert features["train"][1]["axis"] == "horizontal"
    assert features["train"][1]["frames"] == 3
    assert features["train"][1]["solids"] >= 1
    assert {f["axis"] for f in features["train"]} == {"vertical", "horizontal"}
    if len(test) > 1:
        assert {f["axis"] for f in features["test"]} == {"vertical", "horizontal"}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(seed, rng, num_train, num_test)
    base_background = int(rng.integers(10))
    first_test_axis = "vertical" if rng.integers(2) else "horizontal"
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", n_train), ("test", n_test)):
        for index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                try:
                    pair = _make_pair(pair_rng, section, index, first_test_axis,
                                      base_background)
                except RuntimeError:
                    continue
                key = _key(pair["input"])
                if key in seen or key in _SOURCE_INPUTS:
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("unable to construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    assert _puzzle_key(generate_puzzle(314159)) == _puzzle_key(generate_puzzle(314159))
    assert _puzzle_key(generate_puzzle(271828, 8, 3)) == _puzzle_key(generate_puzzle(271828, 8, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(10000 + 100 * train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(11001 + i, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(12000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {args}")
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    shapes, layouts, sizes, backgrounds, combos = set(), set(), set(), set(), set()
    all_features = []
    train_features, test_features = [], []
    variable_size_puzzles = 0
    variable_background_puzzles = 0
    mixed_test_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        local = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                feature = _analyze(pair)
                local.append(feature)
                all_features.append(feature)
                (train_features if section == "train" else test_features).append(feature)
                sizes.add(feature["size"])
                backgrounds.add(feature["background"])
                combos.add((feature["axis"], feature["frames"], feature["solids"]))
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, feature["axis"], feature["frames"],
                             feature["solids"], feature["gaps"], feature["stagger"]))
                assert _key(pair["input"]) not in _SOURCE_INPUTS
        variable_size_puzzles += len({x["size"] for x in local}) > 1
        variable_background_puzzles += len({x["background"] for x in local}) > 1
        mixed_test_puzzles += len({
            _analyze(pair)["axis"] for pair in puzzle["test"]
        }) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(shapes) >= 30 and len(layouts) >= 100
    assert len(sizes) >= 4 and variable_size_puzzles >= 25
    assert len(backgrounds) >= 7 and variable_background_puzzles >= 15
    assert len(combos) >= 20
    assert mixed_test_puzzles >= 15
    assert sum(x["axis"] == "vertical" and x["frames"] == 3
               for x in train_features) >= 10
    assert sum(x["axis"] == "horizontal" for x in test_features) >= 20
    assert sum(x["frames"] == 2 for x in test_features) >= 5
    assert sum(x["solids"] == 0 for x in test_features) >= 5
    assert sum(x["frames"] >= 4 for x in all_features) >= 40
    assert sum(x["solids"] > x["frames"] for x in all_features) >= 30
    assert sum(x["frame_edge"] for x in all_features) >= 50
    assert sum(x["whole_axis_misleading"] for x in all_features) >= 15
    assert sum(x["rect_solid"] for x in all_features) >= 40
    assert sum(x["varied_solid_size"] for x in all_features) >= 50
    assert sum(x["shared_color"] for x in all_features) >= 70
    assert sum(x["cross_kind_color"] for x in all_features) >= 40
    assert sum(x["between_solid"] for x in all_features) >= 60
    assert sum(len(set(x["gaps"])) > 1 for x in all_features) >= 60
    assert sum(max(x["gaps"]) >= 5 for x in all_features) >= 60
    assert sum(x["stagger"] == 0 for x in all_features) >= 20
    assert sum(x["stagger"] >= x["size"] for x in all_features) >= 30
