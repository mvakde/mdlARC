import numpy as np


_SOURCE_PAIRS_TEXT = (
    (
        (
            "0000000000",
            "9000050000",
            "9000050000",
            "6000550000",
            "6005550000",
            "6005050000",
            "4000050000",
            "4000050000",
            "4000550000",
            "0000000000",
        ),
        (
            "0000000000",
            "9000090000",
            "9000090000",
            "6000660000",
            "6006660000",
            "6006060000",
            "4000040000",
            "4000040000",
            "4000440000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "8000050000",
            "8000050000",
            "8005550000",
            "2005000000",
            "2005000000",
            "2005555000",
            "2000005000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "8000080000",
            "8000080000",
            "8008880000",
            "2002000000",
            "2002000000",
            "2002222000",
            "2000002000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "2000550500",
            "2005555500",
            "3005000000",
            "3005550000",
            "3000050000",
            "4005555000",
            "7005550000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "2000220200",
            "2002222200",
            "3003000000",
            "3003330000",
            "3000030000",
            "4004444000",
            "7007770000",
            "0000000000",
        ),
    ),
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed_value


def _normalize_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count("num_train", num_train)
    test_count = _normalize_count("num_test", num_test)

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        legal = [
            (train, test)
            for test in range(1, 5)
            for train in range(max(2, test + 1), 11)
        ]
        index = (seed * 7 + 3) % len(legal)
        return legal[index]

    if train_count is None:
        candidates = list(range(max(2, test_count + 1), 11))
        index = (seed * 11 + 5) % len(candidates)
        return candidates[index], test_count

    candidates = list(range(1, min(4, train_count - 1) + 1))
    index = (seed * 11 + 5) % len(candidates)
    return train_count, candidates[index]


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Choose color vocabularies; their spatial roles remain visible per pair."""
    colors = [int(c) for c in rng.permutation(10)]
    varying_background = bool(rng.random() < 0.5)
    varying_neutral = bool(rng.random() < 0.5)
    return {
        "backgrounds": tuple(colors[:2] if varying_background else colors[:1]),
        "neutrals": tuple(colors[2:4] if varying_neutral else colors[2:3]),
        "markers": tuple(colors[4:]),
    }


def _mask(rng: np.random.Generator, rows: int, width: int, style: str) -> np.ndarray:
    assert rows >= 4 and width >= 3
    mask = np.zeros((rows, width), dtype=bool)
    if style == "scatter":
        mask = rng.random((rows, width)) < float(rng.uniform(0.18, 0.55))
        for row in range(rows):
            if not mask[row].any():
                mask[row, int(rng.integers(width))] = True
        return mask

    spine = [int(rng.integers(width))]
    for _ in range(1, rows):
        spine.append(int(np.clip(spine[-1] + int(rng.integers(-1, 2)), 0, width - 1)))
    for row, center in enumerate(spine):
        left = int(rng.integers(0, min(5, center + 1)))
        right = int(rng.integers(0, min(5, width - center)))
        mask[row, center - left:center + right + 1] = True

    if style == "bridge":
        row = int(rng.integers(1, rows - 1))
        candidates = [c for c in range(width) if abs(c - spine[row]) >= 2]
        if candidates:
            a, b = sorted((spine[row], int(rng.choice(candidates))))
        else:
            a, b = 0, width - 1
        mask[row] = False
        mask[row, a] = mask[row, b] = True
        mask[row - 1, a:b + 1] = True
        mask[row + 1, a:b + 1] = True
    return mask


def _run_colors(rng: np.random.Generator, rows: int, regime: str, markers: tuple[int, ...]) -> list[int]:
    maximum = min(rows, 5 if regime == "source_train" else 6)
    if regime == "source_train":
        count = int(rng.integers(2, maximum + 1))
    else:
        # Overlap the training range while favoring the source test's richer sequence.
        count = int(rng.integers(2, maximum + 1))
        if maximum >= 4 and rng.random() < 0.45:
            count = int(rng.integers(4, maximum + 1))
    cuts = sorted(int(c) for c in rng.choice(np.arange(1, rows), size=count - 1, replace=False))
    lengths = np.diff([0, *cuts, rows])
    palette = [int(c) for c in rng.permutation(markers)]
    sequence = [palette[0]]
    for _ in range(1, count):
        old = [c for c in sequence[:-1] if c != sequence[-1]]
        if old and rng.random() < 0.34:
            sequence.append(int(rng.choice(old)))
        else:
            choices = [c for c in palette if c != sequence[-1]]
            sequence.append(int(rng.choice(choices)))
    return [color for color, length in zip(sequence, lengths) for _ in range(int(length))]


def _layout(rng: np.random.Generator, width: int, mode: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return marker column and inclusive payload intervals."""
    if mode == "right":
        marker = int(rng.integers(0, min(5, width - 3)))
        gap = int(rng.integers(0, min(5, width - marker - 3)))
        start = marker + gap + 1
        end = int(rng.integers(start + 2, min(width - 1, start + 13) + 1))
        return marker, ((start, end),)
    if mode == "left":
        marker = width - 1 - int(rng.integers(0, min(5, width - 3)))
        gap = int(rng.integers(0, min(5, marker - 2)))
        end = marker - gap - 1
        start = int(rng.integers(max(0, end - 13), end - 1))
        return marker, ((start, end),)
    if mode == "both":
        marker = int(rng.integers(3, width - 3))
        left_gap = int(rng.integers(0, min(3, marker - 2)))
        right_gap = int(rng.integers(0, min(3, width - marker - 3)))
        left_end = marker - left_gap - 1
        right_start = marker + right_gap + 1
        left_start = int(rng.integers(0, left_end - 1))
        right_end = int(rng.integers(right_start + 2, width))
        return marker, ((left_start, left_end), (right_start, right_end))
    raise ValueError("unknown layout mode")


def _make_pair(
    rng: np.random.Generator,
    regime: str,
    parameters: dict,
    role: tuple[int, int],
    mode: str,
    force_bridge: bool = False,
) -> dict:
    if rng.random() < 0.13:
        height = width = 10
    else:
        height = int(rng.integers(7, 25))
        width = int(rng.integers(7, 24))
    low_rows = max(4, height // 2)
    high_rows = min(18, height - 2)
    rows = int(rng.integers(low_rows, high_rows + 1))
    top = int(rng.integers(1, height - rows))
    marker, intervals = _layout(rng, width, mode)
    colors = _run_colors(rng, rows, regime, parameters["markers"])
    background, neutral = role
    input_grid = np.full((height, width), background, dtype=np.int8)
    for offset, color in enumerate(colors):
        input_grid[top + offset, marker] = color
    for start, end in intervals:
        if force_bridge:
            style = "bridge"
        else:
            weights = (0.48, 0.25, 0.27) if regime == "source_train" else (0.35, 0.27, 0.38)
            style = str(rng.choice(("connected", "bridge", "scatter"), p=weights))
        mask = _mask(rng, rows, end - start + 1, style)
        for offset in range(rows):
            input_grid[top + offset, start:end + 1][mask[offset]] = neutral
    output_grid = input_grid.copy()
    for offset, color in enumerate(colors):
        row = top + offset
        output_grid[row, input_grid[row] == neutral] = color
    return {"input": input_grid, "output": output_grid}


def _grid_signature(grid: np.ndarray) -> tuple:
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _pair_signature(pair: dict) -> tuple:
    return _grid_signature(pair["input"]), _grid_signature(pair["output"])


def _matches_source_pair(pair: dict) -> bool:
    input_grid, output_grid = pair["input"], pair["output"]
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return False
    candidate = (
        tuple("".join(map(str, row)) for row in input_grid),
        tuple("".join(map(str, row)) for row in output_grid),
    )
    return candidate in _SOURCE_PAIRS_TEXT


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    roles = [(b, n) for b in parameters["backgrounds"] for n in parameters["neutrals"]]
    rng.shuffle(roles)
    training_roles = [roles[0], roles[1 if len(roles) > 1 else 0]]
    training_roles += [roles[int(rng.integers(len(roles)))] for _ in range(train_count - 2)]
    observed_roles = list(dict.fromkeys(training_roles))
    test_roles = [observed_roles[int(rng.integers(len(observed_roles)))] for _ in range(test_count)]

    first_side = str(rng.choice(("right", "left")))
    second_mode = "both" if rng.random() < 0.55 else ("left" if first_side == "right" else "right")
    train_modes = [first_side, second_mode]
    train_modes += [str(rng.choice(("right", "left", "both"), p=(0.42, 0.42, 0.16))) for _ in range(train_count - 2)]
    test_modes = [str(rng.choice(("right", "left", "both"), p=(0.39, 0.39, 0.22))) for _ in range(test_count)]
    if "both" in test_modes and "both" not in train_modes:
        train_modes[1] = "both"

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, count, role_list, mode_list in (
        ("train", "source_train", train_count, training_roles, train_modes),
        ("test", "target_test", test_count, test_roles, test_modes),
    ):
        for index in range(count):
            for _ in range(32):
                pair = _make_pair(rng, regime, parameters, role_list[index], mode_list[index],
                                  force_bridge=(split == "train" and index == 0))
                signature = _grid_signature(pair["input"])
                if signature not in seen and not _matches_source_pair(pair):
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                background = int(pair["input"][0, 0])
                height, width = pair["input"].shape
                for extra_rows in range(31 - height):
                    for extra_cols in range(31 - width):
                        padded = {
                            key: np.pad(grid, ((0, extra_rows), (0, extra_cols)),
                                        constant_values=background)
                            for key, grid in pair.items()
                        }
                        signature = _grid_signature(padded["input"])
                        if signature not in seen and not _matches_source_pair(padded):
                            seen.add(signature)
                            puzzle[split].append(padded)
                            break
                    else:
                        continue
                    break
                else:
                    raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _segments(values: np.ndarray) -> int:
    return int(np.count_nonzero(values & ~np.r_[False, values[:-1]]))


def _components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        count += 1
        stack = [(row, col)]
        seen[row, col] = True
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _analyze_pair(pair: dict) -> dict:
    source, target = pair["input"], pair["output"]
    assert source.shape == target.shape
    height, width = source.shape
    background = int(source[0, 0])
    assert np.all(source[0] == background) and np.all(source[-1] == background)
    neutral_positions = source != target
    assert neutral_positions.any()
    neutral_colors = np.unique(source[neutral_positions])
    assert len(neutral_colors) == 1
    neutral = int(neutral_colors[0])
    assert neutral != background and not np.any(target == neutral)
    assert np.array_equal(neutral_positions, source == neutral)
    marker_columns = []
    marker_colors = []
    active_rows = []
    split_rows = 0
    for row in range(height):
        payload = source[row] == neutral
        markers = np.flatnonzero((source[row] != background) & ~payload)
        if not payload.any():
            assert len(markers) == 0
            assert np.all(target[row] == background)
            continue
        assert len(markers) == 1
        column = int(markers[0])
        color = int(source[row, column])
        assert color not in (background, neutral)
        assert np.array_equal(target[row, ~payload], source[row, ~payload])
        assert np.all(target[row, payload] == color)
        marker_columns.append(column)
        marker_colors.append(color)
        active_rows.append(row)
        split_rows += _segments(payload) >= 2
    assert active_rows == list(range(active_rows[0], active_rows[-1] + 1))
    assert active_rows[0] > 0 and active_rows[-1] < height - 1
    assert len(set(marker_columns)) == 1
    marker = marker_columns[0]
    payload_cols = np.flatnonzero(source == neutral) % width
    left = bool(np.any(payload_cols < marker))
    right = bool(np.any(payload_cols > marker))
    assert left or right
    mode = "both" if left and right else "left" if left else "right"
    runs = 1 + sum(a != b for a, b in zip(marker_colors, marker_colors[1:]))
    lengths = []
    for color in marker_colors:
        if not lengths or color != lengths[-1][0]:
            lengths.append([color, 1])
        else:
            lengths[-1][1] += 1
    return {
        "background": background, "neutral": neutral, "mode": mode,
        "marker": marker, "runs": runs, "run_lengths": tuple(x[1] for x in lengths),
        "reappearance": len({x[0] for x in lengths}) < len(lengths),
        "split_rows": int(split_rows), "components": _components(source == neutral),
        "shape": (height, width), "active_rows": tuple(active_rows),
        "payload_width": int(np.ptp(payload_cols)) + 1,
        "gap": int(min(abs(payload_cols - marker)) - 1),
        "role": (background, neutral),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            assert np.any(pair["input"] != pair["output"])
            input_sig, pair_sig = _grid_signature(pair["input"]), _pair_signature(pair)
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            assert not _matches_source_pair(pair)
            info = _analyze_pair(pair)
            assert 2 <= info["runs"] <= (5 if split == "train" else 6)
            assert len(info["active_rows"]) >= 4
            infos[split].append(info)
    train_roles = {i["role"] for i in infos["train"]}
    assert all(i["role"] in train_roles for i in infos["test"])
    train_modes = {i["mode"] for i in infos["train"]}
    assert "right" in train_modes or "both" in train_modes
    assert "left" in train_modes or "both" in train_modes
    assert all(i["mode"] != "both" or "both" in train_modes for i in infos["test"])
    assert any(i["split_rows"] > 0 for i in infos["train"])
    assert len({int(c) for p in puzzle["train"] for c in np.unique(p["output"][p["input"] != p["output"]])}) >= 2


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def _expect_value_error(**kwargs) -> None:
    try:
        generate_puzzle(0, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {kwargs}")


def self_test() -> None:
    train_counts, test_counts, shapes, roles, modes, markers, gaps = (set() for _ in range(7))
    run_counts = {"train": set(), "test": set()}
    opening_patterns = {"train": set(), "test": set()}
    connected = {"train": 0, "test": 0}
    connected_split = {"train": 0, "test": 0}
    fragmented = {"train": 0, "test": 0}
    singletons = {"train": 0, "test": 0}
    split = {"train": 0, "test": 0}
    repeated = 0
    changed_roles = 0
    varying = {key: 0 for key in ("shape", "marker", "gap", "payload_width", "mode")}
    exact_ten = 0
    familiar_tests = novel_tests = short_tests = 0
    connected_split_singletons = bilateral_singletons = 0
    connected_novel_tests = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        signatures.add(tuple(_pair_signature(p) for s in ("train", "test") for p in puzzle[s]))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_roles = set()
        local_infos = []
        known_colors = {
            int(color)
            for pair in puzzle["train"]
            for color in np.unique(pair["output"][pair["input"] != pair["output"]])
        }
        for s in ("train", "test"):
            for pair in puzzle[s]:
                info = _analyze_pair(pair)
                local_infos.append(info)
                grid = pair["input"]
                answer = grid.copy()
                for row in info["active_rows"]:
                    answer[row, grid[row] == info["neutral"]] = grid[row, info["marker"]]
                assert np.array_equal(answer, pair["output"])
                assert not _matches_source_pair(pair)
                shapes.add(info["shape"])
                roles.add(info["role"])
                local_roles.add(info["role"])
                modes.add(info["mode"])
                markers.add(info["marker"])
                gaps.add(info["gap"])
                run_counts[s].add(info["runs"])
                first_row = info["active_rows"][0]
                opening_patterns[s].add(tuple(
                    int(col - info["marker"])
                    for col in np.flatnonzero(grid[first_row] == info["neutral"])
                ))
                connected[s] += info["components"] == 1
                connected_split[s] += info["components"] == 1 and info["split_rows"] > 0
                fragmented[s] += info["components"] > 1
                singletons[s] += 1 in info["run_lengths"]
                split[s] += info["split_rows"] > 0
                repeated += info["reappearance"]
                exact_ten += info["shape"] == (10, 10)
                connected_split_singletons += (
                    info["components"] == 1 and info["split_rows"] > 0
                    and 1 in info["run_lengths"]
                )
                bilateral_singletons += info["mode"] == "both" and 1 in info["run_lengths"]
                if s == "test":
                    used = {int(color) for color in grid[info["active_rows"], info["marker"]]}
                    familiar_tests += used <= known_colors
                    novel_tests += bool(used - known_colors)
                    short_tests += info["runs"] in (2, 3)
                    connected_novel_tests += info["components"] == 1 and bool(used - known_colors)
        changed_roles += len(local_roles) > 1
        for key in varying:
            varying[key] += len({info[key] for info in local_infos}) > 1
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(roles) >= 12
    assert modes == {"left", "right", "both"} and len(markers) >= 5 and len(gaps) >= 4
    assert exact_ten > 0 and changed_roles >= 8 and repeated > 0
    assert familiar_tests > 0 and novel_tests > 0 and short_tests > 0
    assert connected_split_singletons > 0 and bilateral_singletons > 0
    assert connected_novel_tests > 0
    assert all(count >= 25 for count in varying.values())
    for s in ("train", "test"):
        assert connected[s] > 0 and fragmented[s] > 0
        assert connected_split[s] > 0 and len(opening_patterns[s]) >= 30
        assert singletons[s] > 0 and split[s] > 0
        assert 2 in run_counts[s] and max(run_counts[s]) >= 4
    for train, test in ((2, 1), (10, 1), (10, 4), (3, 2)):
        p = generate_puzzle(109, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(107, train, test))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        _expect_value_error(**kwargs)
