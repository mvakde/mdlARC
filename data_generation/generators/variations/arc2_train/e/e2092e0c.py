import numpy as np


_SOURCE_INPUT_KEYS = frozenset(
    "".join(block.split())
    for block in (
        """
        340500305807000
        004588000073300
        083500500102009
        555561090033060
        370005000040000
        040055600000150
        021000004909000
        000080007220098
        100010370007003
        001209340010029
        090080004006084
        776000083000827
        090020400000016
        000000030040980
        400090117900080
        """,
        """
        076500001456008
        700500003000306
        090505000000052
        555540004090900
        000000000000200
        236000760094004
        000007000003000
        090000908700000
        061070000000700
        105400800002206
        306020000400060
        410000107000040
        020070097600530
        404100818009477
        080000600000516
        """,
        """
        402500026900500
        070508580700088
        066570350004700
        555580190000500
        800000103980000
        020006640900170
        806000830000009
        300200000008000
        000020040203200
        001000007000508
        094400406607000
        700090080005000
        060010077000400
        000405000070500
        809850000030400
        """,
        """
        073500000003540
        103520100008000
        100560099050009
        555500210030000
        300318505200500
        409200102000000
        002000954080055
        070005570010001
        000307370000700
        000000090005000
        000030003407302
        022000000010300
        005222000010020
        003005470000035
        800171080820004
        """,
    )
)



def _value_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.ravel())


def _source_key(grid: np.ndarray) -> str:
    return "".join(str(int(x)) for x in grid.ravel())


def _frame_coordinates(row: int, col: int, height: int, width: int) -> tuple:
    return tuple(
        [(row - 1, c) for c in range(col - 1, col + width + 1)]
        + [(row + height, c) for c in range(col - 1, col + width + 1)]
        + [(r, col - 1) for r in range(row, row + height)]
        + [(r, col + width) for r in range(row, row + height)]
    )


def _find_template_matches(grid: np.ndarray, template: np.ndarray) -> list:
    h, w = template.shape
    windows = np.lib.stride_tricks.sliding_window_view(grid, (h, w))
    rows, cols = np.nonzero(np.all(windows == template, axis=(-2, -1)))
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _read_exemplar(grid: np.ndarray) -> tuple:
    """Infer the exemplar dimensions and local marker from its clipped L."""
    candidates = []
    for h in range(2, 5):
        for w in range(2, 5):
            if h + 1 >= grid.shape[0] or w + 1 >= grid.shape[1]:
                continue
            marker = int(grid[0, w])
            if marker in grid[:h, :w]:
                continue
            if np.all(grid[:h, w] == marker) and np.all(grid[h, :w + 1] == marker):
                candidates.append((h, w, marker))
    assert len(candidates) == 1, f"exemplar L is ambiguous: {candidates}"
    return candidates[0]


def _make_output(grid: np.ndarray, row: int, col: int,
                 height: int, width: int, marker: int) -> np.ndarray:
    result = grid.copy()
    for r, c in _frame_coordinates(row, col, height, width):
        result[r, c] = marker
    return result


def _check_optional_count(value: int | None, name: str,
                          low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _check_optional_count(num_train, "num_train", 2, 10)
    test = _check_optional_count(num_test, "num_test", 1, 4)
    seed = int(seed)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _sample_target(rng: np.random.Generator, height: int, width: int,
                   h: int, w: int, shift: str, regime: str) -> tuple:
    positions = []
    for r in range(1, height - h):
        for c in range(1, width - w):
            # The completed frame must be separate from the corner exemplar.
            if r < h + 2 and c < w + 2:
                continue
            fraction = (c - 1) / max(1, width - w - 2)
            if shift == "right" and regime == "source_train" and fraction > .64:
                continue
            if shift == "right" and regime == "target_test" and fraction < .50:
                continue
            if shift == "reverse" and regime == "source_train" and fraction < .36:
                continue
            if shift == "reverse" and regime == "target_test" and fraction > .50:
                continue
            positions.append((r, c))
    assert positions
    edge = [p for p in positions if p[0] in (1, height - h - 1)
            or p[1] in (1, width - w - 1)]
    pool = edge if edge and rng.random() < .28 else positions
    return pool[int(rng.integers(len(pool)))]


def _decoy_sites(rng: np.random.Generator, grid: np.ndarray,
                 h: int, w: int, row: int, col: int, count: int) -> list:
    forbidden = np.zeros(grid.shape, dtype=bool)
    forbidden[:h + 1, :w + 1] = True
    forbidden[row - 1:row + h + 1, col - 1:col + w + 1] = True
    sites = []
    for _ in range(count):
        choices = [(r, c) for r in range(grid.shape[0] - h + 1)
                   for c in range(grid.shape[1] - w + 1)
                   if not np.any(forbidden[r:r + h, c:c + w])]
        if not choices:
            return []
        r, c = choices[int(rng.integers(len(choices)))]
        sites.append((r, c))
        forbidden[max(0, r - 1):min(grid.shape[0], r + h + 1),
                  max(0, c - 1):min(grid.shape[1], c + w + 1)] = True
    return sites


def _install_decoy(rng: np.random.Generator, grid: np.ndarray,
                   template: np.ndarray, site: tuple,
                   palette: np.ndarray) -> None:
    patch = template.copy()
    occupied = patch.size
    change_count = 1 if rng.random() < .65 else 2
    indices = rng.choice(occupied, size=change_count, replace=False)
    # A swap is a color rearrangement when the selected values differ.
    if change_count == 2 and rng.random() < .4 and patch.flat[indices[0]] != patch.flat[indices[1]]:
        patch.flat[indices[0]], patch.flat[indices[1]] = (
            int(patch.flat[indices[1]]), int(patch.flat[indices[0]])
        )
    else:
        for index in indices:
            old = int(patch.flat[index])
            alternatives = palette[palette != old]
            patch.flat[index] = int(rng.choice(alternatives))
    r, c = site
    grid[r:r + template.shape[0], c:c + template.shape[1]] = patch


def _near_matches(grid: np.ndarray, template: np.ndarray,
                  target: tuple) -> list:
    h, w = template.shape
    windows = np.lib.stride_tricks.sliding_window_view(grid, (h, w))
    distances = np.count_nonzero(windows != template, axis=(-2, -1))
    sites = []
    for r, c in np.argwhere((distances >= 1) & (distances <= 2)):
        r, c = int(r), int(c)
        if r <= h and c <= w:
            continue
        if abs(r - target[0]) <= h and abs(c - target[1]) <= w:
            continue
        sites.append((r, c))
    return sites


def _generate_pair(rng: np.random.Generator, regime: str, shift: str,
                   shape: tuple, marker: int, background: int,
                   frame_markers: int, decoy_count: int) -> tuple:
    h, w = shape
    for _ in range(120):
        if rng.random() < .17:
            height = width = 15
        else:
            lower = 14 if decoy_count == 2 else 11
            height = int(rng.integers(lower, 24))
            width = int(rng.integers(lower, 24))
        row, col = _sample_target(rng, height, width, h, w, shift, regime)
        palette_size = int(rng.integers(3, 11))
        other_colors = np.array([x for x in range(10) if x not in (marker, background)],
                                dtype=np.int8)
        chosen = rng.choice(other_colors, size=palette_size - 2, replace=False)
        palette = np.array([background, marker, *chosen], dtype=np.int8)
        content_colors = np.array(chosen, dtype=np.int8)
        density = float(rng.uniform(.045, .43 if regime == "source_train" else .49))
        grid = np.full((height, width), background, dtype=np.int8)
        noise = rng.random(grid.shape) < density
        grid[noise] = rng.choice(palette[palette != background],
                                 size=int(np.count_nonzero(noise)))

        template = np.full((h, w), background, dtype=np.int8)
        occupancy = int(rng.integers(3, h * w + 1))
        occupied = rng.choice(h * w, size=occupancy, replace=False)
        if len(content_colors) >= occupancy and rng.random() < .45:
            values = rng.choice(content_colors, size=occupancy, replace=False)
        else:
            values = rng.choice(content_colors, size=occupancy, replace=True)
            if occupancy >= 2 and rng.random() < .55:
                values[1] = values[0]
        template.flat[occupied] = values
        grid[:h, :w] = template
        grid[:h, w] = marker
        grid[h, :w + 1] = marker
        grid[row:row + h, col:col + w] = template

        frame = _frame_coordinates(row, col, h, w)
        for fr, fc in frame:
            if grid[fr, fc] == marker:
                grid[fr, fc] = background
        if frame_markers:
            for index in rng.choice(len(frame), size=frame_markers, replace=False):
                fr, fc = frame[int(index)]
                grid[fr, fc] = marker

        sites = _decoy_sites(rng, grid, h, w, row, col, decoy_count)
        if len(sites) != decoy_count:
            continue
        for site in sites:
            _install_decoy(rng, grid, template, site, palette)
        try:
            if _read_exemplar(grid) != (h, w, marker):
                continue
        except AssertionError:
            continue
        if set(_find_template_matches(grid, template)) != {(0, 0), (row, col)}:
            continue
        if decoy_count and len(_near_matches(grid, template, (row, col))) < decoy_count:
            continue
        if grid.shape == (15, 15) and _source_key(grid) in _SOURCE_INPUT_KEYS:
            continue
        output = _make_output(grid, row, col, h, w, marker)
        if np.array_equal(grid, output):
            continue
        return grid, output
    raise RuntimeError("failed to construct a unique copy with the requested decoys")


def _analyze_task_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    h, w, marker = _read_exemplar(input_grid)
    template = input_grid[:h, :w]
    matches = _find_template_matches(input_grid, template)
    assert len(matches) == 2 and (0, 0) in matches
    target = next(p for p in matches if p != (0, 0))
    row, col = target
    assert 1 <= row < input_grid.shape[0] - h
    assert 1 <= col < input_grid.shape[1] - w
    assert row >= h + 2 or col >= w + 2
    frame = _frame_coordinates(row, col, h, w)
    assert all(0 <= r < input_grid.shape[0] and 0 <= c < input_grid.shape[1]
               for r, c in frame)
    expected = _make_output(input_grid, row, col, h, w, marker)
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    assert np.array_equal(output_grid[:h + 1, :w + 1], input_grid[:h + 1, :w + 1])
    assert np.array_equal(output_grid[row:row + h, col:col + w], template)
    frame_before = np.array([input_grid[r, c] for r, c in frame])
    return {"shape": (h, w), "marker": marker, "target": target,
            "frame_markers": int(np.count_nonzero(frame_before == marker)),
            "near_matches": len(_near_matches(input_grid, template, target)),
            "palette_size": len(np.unique(input_grid)),
            "height_width": input_grid.shape,
            "template": tuple(int(x) for x in template.ravel())}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _value_key(pair["input"])
            assert key not in keys
            keys.add(key)
            if pair["input"].shape == (15, 15):
                assert _source_key(pair["input"]) not in _SOURCE_INPUT_KEYS
            infos[split].append(_analyze_task_pair(pair["input"], pair["output"]))
    train = infos["train"]
    test = infos["test"]
    # Both a bare destination and a partial frame appear in demonstrations.
    assert any(info["frame_markers"] == 0 for info in train)
    assert any(info["frame_markers"] > 0 for info in train)
    assert any(info["near_matches"] >= 1 for info in train)
    assert all(info["near_matches"] >= 1 for info in test)
    assert (3, 3) in {info["shape"] for info in train}
    assert len({info["shape"] for info in train}) >= 2
    assert {info["shape"] for info in test} <= {info["shape"] for info in train}
    assert len({info["marker"] for info in train}) >= 2
    assert len({info["target"] for info in train + test}) == len(train + test)


def _sample_pair_colors(rng: np.random.Generator, first_marker: int | None,
                        first_background: int | None) -> tuple:
    if rng.random() < .43 and first_background != 0:
        background = 0
    else:
        options = [x for x in range(10) if x != first_background]
        background = int(rng.choice(options))
    markers = [x for x in range(10) if x != background and x != first_marker]
    marker = int(rng.choice(markers))
    return marker, background


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # This is only a placement regime. Every output rule is visible locally.
    shift = ("right", "reverse", "overlap")[int(rng.integers(3))]
    alternatives = ((2, 3), (3, 4), (4, 4))
    train_shapes = [(3, 3), alternatives[int(rng.integers(len(alternatives)))]]
    all_shapes = ((2, 3), (3, 3), (3, 4), (4, 4))
    train_shapes += [all_shapes[int(rng.integers(len(all_shapes)))]
                     for _ in range(train_count - 2)]
    test_shapes = [train_shapes[int(rng.integers(len(train_shapes)))]
                   for _ in range(test_count)]
    frame_plan = [0, int(rng.integers(1, 4))]
    decoy_plan = [0, 1]
    rng.shuffle(frame_plan)
    rng.shuffle(decoy_plan)
    puzzle = {"train": [], "test": []}
    first_marker = first_background = None
    seen_targets = set()
    for split, shapes in (("train", train_shapes), ("test", test_shapes)):
        for index, shape in enumerate(shapes):
            pair_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
            marker, background = _sample_pair_colors(
                pair_rng, first_marker if split == "train" and index == 1 else None,
                first_background if split == "train" and index == 1 else None)
            if split == "train" and index == 0:
                first_marker, first_background = marker, background
            if split == "train" and index < 2:
                frame_markers, decoy_count = frame_plan[index], decoy_plan[index]
            else:
                frame_markers = (0, 0, 0, 1, 1, 2, 3)[int(pair_rng.integers(7))]
                decoy_count = (1 if split == "test" else 0) + int(pair_rng.random() < .25)
            for _ in range(32):
                grid, output = _generate_pair(pair_rng,
                    "source_train" if split == "train" else "target_test",
                    shift, shape, marker, background, frame_markers, decoy_count)
                info = _analyze_task_pair(grid, output)
                if info["target"] not in seen_targets:
                    seen_targets.add(info["target"])
                    puzzle[split].append({"input": grid, "output": output})
                    break
            else:
                raise RuntimeError("failed to vary destinations within a puzzle")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, _value_key(pair["input"]), _value_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    records = []
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_task_pair(pair["input"], pair["output"])
                records.append((split, pair["input"], info))
    assert len({info["height_width"] for _, _, info in records}) >= 25
    assert len({info["shape"] for _, _, info in records}) == 4
    assert len({info["target"] for _, _, info in records}) >= 45
    assert len({info["marker"] for _, _, info in records}) == 10
    assert len({info["palette_size"] for _, _, info in records}) >= 5
    assert any(info["frame_markers"] == 0 for _, _, info in records)
    assert any(info["frame_markers"] >= 2 for _, _, info in records)
    assert any(info["near_matches"] >= 2 for _, _, info in records)
    assert any(g.shape == (15, 15) for _, g, _ in records)
    assert any(info["target"][0] == 1 for _, _, info in records)
    assert any(info["target"][1] == 1 for _, _, info in records)
    assert any(info["target"][1] >= g.shape[1] - info["shape"][1] - 1
               for _, g, info in records)
    for args in ((2, 1), (10, 1), (10, 4), (3, 2), (7, 3)):
        puzzle = generate_puzzle(1000 + args[0] * 10 + args[1], *args)
        assert len(puzzle["train"]) == args[0] and len(puzzle["test"]) == args[1]
        validate_generated_puzzle(puzzle)
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2000 + int(args[0] or args[1]), *args))
    assert _puzzle_signature(generate_puzzle(314159, 8, 3)) == _puzzle_signature(
        generate_puzzle(314159, 8, 3))
    assert len({_puzzle_signature(generate_puzzle(4000 + x, 6, 2))
                for x in range(8)}) == 8
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(5000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
