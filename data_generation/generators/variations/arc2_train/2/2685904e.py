"""Frequency-filtered line puzzles with a unary count marker."""

import numpy as np


# The seven original input signatures, expressed in their original layout.
_SOURCE_ROWS = {
    (2, (2, 3, 3, 2, 3, 1, 1, 3, 1, 1)),
    (1, (6, 6, 4, 6, 2, 1, 9, 2, 9, 4)),
    (3, (4, 1, 4, 4, 6, 3, 1, 6, 3, 6)),
    (4, (2, 1, 2, 1, 2, 1, 1, 2, 2, 2)),
    (3, (8, 6, 4, 3, 4, 7, 3, 8, 3, 7)),
    (1, (1, 3, 1, 1, 1, 1, 4, 1, 1, 1)),
    (2, (2, 3, 6, 4, 6, 2, 4, 4, 3, 9)),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _palette(rng):
    return tuple(int(x) for x in rng.permutation(10)[:3])


def _profile(rng, n, q, mode):
    """Return qualifying and rejected frequencies, respecting the ARC bounds."""
    if mode == "all":
        return [n] * q, []
    for _ in range(100):
        if mode == "zero":
            d = int(rng.integers(2, 6))
            rejected = [int(rng.integers(1, 14)) for _ in range(d)]
        elif mode == "simple":
            rejected = [int(rng.integers(5, 17))]
        elif mode == "dominant":
            rejected = [int(rng.integers(8, 19))]
            if rng.random() < 0.4:
                rejected.append(int(rng.integers(1, 5)))
        elif mode == "most" and n > 1:
            d = int(rng.integers(1, 4))
            rejected = [int(rng.integers(1, n)) for _ in range(d)]
        elif mode == "least":
            d = int(rng.integers(1, 4))
            rejected = [int(rng.integers(n + 1, min(15, n + 7)))
                        for _ in range(d)]
        elif mode == "intermediate" and n > 1:
            rejected = [
                int(rng.integers(1, n)),
                int(rng.integers(n + 1, min(16, n + 9))),
            ]
            for _ in range(int(rng.integers(0, 3))):
                rejected.append(int(rng.integers(1, 13)))
        elif mode == "equal":
            count = int(rng.integers(1, 13))
            rejected = [count, count]
            if rng.random() < 0.5:
                rejected.append(int(rng.integers(1, 13)))
        else:
            d = int(rng.integers(1, 6))
            rejected = [int(rng.integers(1, 15)) for _ in range(d)]
        if (
            all(c != n for c in rejected)
            and q + len(rejected) <= 9
            and max(6, n + 2) <= n * q + sum(rejected) <= 29
        ):
            return [n] * q, rejected
    # A legal deterministic fallback for extreme count/color combinations.
    rejected = [1 if n != 1 else 2]
    while n * q + sum(rejected) < max(6, n + 2):
        rejected[0] += 1
        if rejected[0] == n:
            rejected[0] += 1
    if n * q + sum(rejected) > 29:
        rejected = [1 if n != 1 else 2]
    return [n] * q, rejected


def _source(rng, background, marker, divider, q_counts, d_counts,
            force_qualifier=None, force_rejected=None):
    pool = [c for c in range(10) if c != background]
    qualifiers = []
    rejected = []
    if force_qualifier is not None:
        qualifiers.append(force_qualifier)
    if force_rejected is not None:
        rejected.append(force_rejected)
    for color in rng.permutation(pool):
        color = int(color)
        if color in qualifiers or color in rejected:
            continue
        if len(qualifiers) < len(q_counts):
            qualifiers.append(color)
        elif len(rejected) < len(d_counts):
            rejected.append(color)
    assert len(qualifiers) == len(q_counts)
    assert len(rejected) == len(d_counts)
    pieces = [
        np.full(count, color, dtype=np.int8)
        for color, count in zip(qualifiers + rejected, q_counts + d_counts)
    ]
    source = np.concatenate(pieces)
    rng.shuffle(source)
    return source


def _layout(rng, width, minimum_lead=0):
    lead = int(rng.integers(max(3, minimum_lead), 10))
    gap = int(rng.integers(0, 6))
    tail = int(rng.integers(0, 8))
    marker_row = int(rng.integers(0, lead))
    return lead, gap, tail, marker_row


def _build(source, n, colors, layout, marker_start, rotation):
    background, marker, divider = colors
    lead, gap, tail, marker_row = layout
    width = int(source.size)
    divider_row = lead + n
    source_row = divider_row + 1 + gap
    height = source_row + 1 + tail
    grid = np.full((height, width), background, dtype=np.int8)
    grid[marker_row, marker_start:marker_start + n] = marker
    grid[divider_row, :] = divider
    grid[source_row, :] = source
    values, counts = np.unique(source, return_counts=True)
    keep = values[counts == n]
    filtered = np.where(np.isin(source, keep), source, background)
    answer = grid.copy()
    answer[divider_row - n:divider_row, :] = filtered
    return np.rot90(grid, rotation).copy(), np.rot90(answer, rotation).copy()


def _sample_pair(rng, n, q, mode, colors, rotation, *,
                 force_qualifier=None, force_rejected=None):
    qc, dc = _profile(rng, n, q, mode)
    source = _source(rng, colors[0], colors[1], colors[2], qc, dc,
                     force_qualifier, force_rejected)
    layout = _layout(rng, len(source))
    start = int(rng.integers(0, len(source) - n + 1))
    return _build(source, n, colors, layout, start, rotation)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _is_original(grid):
    if grid.shape != (10, 10):
        return False
    if not (
        np.all(grid[6] == 5)
        and np.all(grid[1:6] == 0)
        and np.all(grid[7] == 0)
        and np.all(grid[9] == 0)
    ):
        return False
    n = int(np.count_nonzero(grid[0] == 8))
    return (n, tuple(int(x) for x in grid[8])) in _SOURCE_ROWS


def _inspect_pair(input_grid, output_grid):
    """Reconstruct a canonical scene from grids, without generator metadata."""
    assert input_grid.shape == output_grid.shape
    background = int(np.bincount(input_grid.ravel(), minlength=10).argmax())
    matches = []
    for rotation in range(4):
        grid = np.rot90(input_grid, rotation)
        answer = np.rot90(output_grid, rotation)
        height, width = grid.shape
        if width < 6 or height < 6:
            continue
        divider_rows = [
            row for row in range(height)
            if int(grid[row, 0]) != background
            and np.all(grid[row] == grid[row, 0])
        ]
        if len(divider_rows) != 1:
            continue
        divider_row = divider_rows[0]
        divider = int(grid[divider_row, 0])
        if divider_row < 2 or divider_row >= height - 1:
            continue
        nonempty_after = [
            row for row in range(divider_row + 1, height)
            if np.any(grid[row] != background)
        ]
        if len(nonempty_after) != 1:
            continue
        source_row = nonempty_after[0]
        source = grid[source_row]
        if np.any(source == background) or len(np.unique(source)) < 2:
            continue
        nonempty_before = [
            row for row in range(divider_row)
            if np.any(grid[row] != background)
        ]
        if len(nonempty_before) != 1:
            continue
        marker_row = nonempty_before[0]
        cells = np.flatnonzero(grid[marker_row] != background)
        n = len(cells)
        if not 1 <= n < width or divider_row - marker_row <= n:
            continue
        marker = int(grid[marker_row, cells[0]])
        if marker == divider or np.any(grid[marker_row, cells] != marker):
            continue
        if not np.array_equal(cells, np.arange(cells[0], cells[0] + n)):
            continue
        values, counts = np.unique(source, return_counts=True)
        kept = values[counts == n]
        filtered = np.where(np.isin(source, kept), source, background)
        expected = grid.copy()
        expected[divider_row - n:divider_row, :] = filtered
        if not np.array_equal(answer, expected):
            continue
        rejected = [int(c) for c in counts if c != n]
        relations = (
            "none" if len(kept) == 0 else
            "all" if len(rejected) == 0 else
            "intermediate" if min(rejected) < n < max(rejected) else
            "most" if max(rejected) < n else
            "least" if min(rejected) > n else "mixed"
        )
        matches.append({
            "rotation": rotation,
            "height": height,
            "width": width,
            "background": background,
            "marker": marker,
            "divider": divider,
            "n": n,
            "qualifier_count": len(kept),
            "rejected_count": len(rejected),
            "rejected_frequencies": tuple(rejected),
            "frequencies": tuple(int(x) for x in counts),
            "profile": relations,
            "marker_row": marker_row,
            "marker_start": int(cells[0]),
            "divider_row": divider_row,
            "source_row": source_row,
            "gap": source_row - divider_row - 1,
            "tail": height - source_row - 1,
            "marker_as_token": marker in values,
            "divider_as_token": divider in values,
            "marker_kept": marker in kept,
            "divider_kept": divider in kept,
        })
    assert len(matches) == 1, "scene must have exactly one readable orientation"
    return matches[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            signature = _signature(pair["input"])
            assert signature not in seen
            seen.add(signature)
            assert not _is_original(pair["input"])
            infos[split].append(_inspect_pair(pair["input"], pair["output"]))
    train = infos["train"]
    assert len({info["n"] for info in train}) >= 2
    assert any(info["qualifier_count"] >= 2 for info in train)
    assert any(0 < info["qualifier_count"] < len(info["frequencies"])
               for info in train)
    # The two structural colors are shown as countable source tokens in
    # training, including an accepted and a rejected occurrence.
    assert any(info["marker_kept"] for info in train)
    assert any(info["divider_kept"] for info in train)
    assert any(info["marker_as_token"] and not info["marker_kept"]
               for info in train)
    assert any(info["divider_as_token"] and not info["divider_kept"]
               for info in train)
    train_palettes = {
        (info["background"], info["marker"], info["divider"])
        for info in train
    }
    test_palettes = {
        (info["background"], info["marker"], info["divider"])
        for info in infos["test"]
    }
    assert len(train_palettes) > 1 or test_palettes <= train_palettes
    for info in train + infos["test"]:
        assert 1 <= info["n"] <= 6
        assert info["width"] <= 29
        assert info["marker"] != info["background"]
        assert info["divider"] != info["background"]
        assert info["divider"] != info["marker"]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(seed, rng, num_train, num_test)
    palette_mode = int(rng.integers(0, 3))  # 0: stable, 1/2: variable roles
    common_colors = _palette(rng)
    orientation_mode = int(rng.integers(0, 3))  # consistent, mixed, held out
    base_rotation = int(rng.integers(0, 4))
    contrast = seed % 3 == 0
    if contrast and train_count == 2:
        # A same-source contrast needs matching colors. With only two
        # demonstrations, keep those roles fixed in the test as well.
        palette_mode = 0
    counts = [int(x) for x in rng.permutation([2, 3, 4])[:2]]
    puzzle = {"train": [], "test": []}
    seen = set()
    contrast_colors = None

    if contrast:
        n_a, n_b = counts
        colors = common_colors if palette_mode == 0 else _palette(rng)
        contrast_colors = colors
        # Marker and divider trade accepted/rejected status as n changes.
        other = [c for c in rng.permutation(10)
                 if c not in colors and c != colors[0]]
        color_counts = [
            (colors[1], n_a), (int(other[0]), n_a),
            (colors[2], n_b), (int(other[1]), n_b),
            (int(other[2]), 1), (int(other[3]), 5),
        ]
        source = np.concatenate([
            np.full(count, color, dtype=np.int8)
            for color, count in color_counts
        ])
        rng.shuffle(source)
        lead = int(rng.integers(2, 10))
        gap = int(rng.integers(0, 6))
        tail = int(rng.integers(0, 8))
        marker_row = int(rng.integers(0, lead))
        marker_start = int(rng.integers(0, len(source) - max(counts) + 1))
        # Both scenes have identical divider and source coordinates.
        divider_position = lead + max(counts)
        for n in counts:
            local_lead = divider_position - n
            layout = (local_lead, gap, tail, marker_row)
            pair = _build(source, n, colors, layout, marker_start,
                          base_rotation)
            puzzle["train"].append({"input": pair[0], "output": pair[1]})
            seen.add(_signature(pair[0]))

    for index in range(len(puzzle["train"]), train_count + test_count):
        split = "train" if index < train_count else "test"
        local_index = index if split == "train" else index - train_count
        if split == "train" and index < 2:
            n = counts[index]
            q = index + 1
            mode = "intermediate" if index == 0 else "least"
        else:
            n = int(rng.integers(1, 7))
            q = int(rng.integers(1, min(4, 25 // n) + 1))
            if contrast and train_count == 2 and split == "test":
                choices = [value for value in (1, 3, 4) if value <= 25 // n]
                q = int(rng.choice(choices))
            if split == "train" and index == 2:
                q = min(3, 25 // n)
            modes = ("simple", "dominant", "most", "least",
                     "intermediate", "equal", "random")
            mode = modes[int(rng.integers(0, len(modes)))]
            if split == "train" and index >= 3 and seed % 5 == 0 and index == 3:
                q, mode = 0, "zero"
            elif split == "train" and index >= 4 and seed % 7 == 0 and index == 4:
                q = int(rng.integers(max(2, (6 + n - 1) // n),
                                     min(7, 29 // n) + 1))
                mode = "all"
            elif split == "test" and local_index == 0 and seed % 11 == 0:
                q, mode = 0, "zero"
            elif split == "test" and local_index == 0 and seed % 13 == 0:
                q = int(rng.integers(max(2, (6 + n - 1) // n),
                                     min(7, 29 // n) + 1))
                mode = "all"

        if palette_mode == 0:
            colors = common_colors
        elif split == "train" and index == 0:
            colors = common_colors
        else:
            colors = _palette(rng)
            if split == "train" and index == 1:
                while colors == common_colors:
                    colors = _palette(rng)
            if split == "train" and contrast and index == 2:
                while colors == contrast_colors:
                    colors = _palette(rng)
        if orientation_mode == 0:
            rotation = base_rotation
        elif orientation_mode == 1:
            rotation = int(rng.integers(0, 4))
            if split == "train" and index == 1 and not contrast:
                rotation = (base_rotation + 1) % 4
            if split == "train" and index == 0:
                rotation = base_rotation
            if contrast and split == "test" and local_index == 0:
                rotation = (base_rotation + 1) % 4
        else:
            rotation = (
                base_rotation if split == "train" else
                (base_rotation + int(rng.integers(1, 4))) % 4
            )
        force_qualifier = None
        force_rejected = None
        if split == "train" and index == 0:
            force_qualifier, force_rejected = colors[1], colors[2]
        elif split == "train" and index == 1:
            force_qualifier, force_rejected = colors[2], colors[1]
        for _ in range(60):
            pair = _sample_pair(
                rng, n, q, mode, colors, rotation,
                force_qualifier=force_qualifier,
                force_rejected=force_rejected,
            )
            signature = _signature(pair[0])
            if signature not in seen and not _is_original(pair[0]):
                seen.add(signature)
                puzzle[split].append({"input": pair[0], "output": pair[1]})
                break
        else:
            raise ValueError("could not create distinct scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(
        len(left[split]) == len(right[split])
        and all(
            np.array_equal(a[key], b[key])
            for a, b in zip(left[split], right[split])
            for key in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    first = generate_puzzle(123456)
    assert _puzzles_equal(first, generate_puzzle(123456))
    explicit = generate_puzzle(987654, 7, 3)
    assert _puzzles_equal(explicit, generate_puzzle(987654, 7, 3))
    assert not _puzzles_equal(generate_puzzle(1001), generate_puzzle(1002))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(
            2000 + train_count * 10 + test_count, train_count, test_count
        ))
    for train_count, test_count in ((None, 1), (None, 4),
                                    (2, None), (10, None)):
        puzzle = generate_puzzle(3000 + (train_count or 0) +
                                 (test_count or 0), train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    train_counts, test_counts, shapes, orientations = set(), set(), set(), set()
    profiles, test_n, test_q, gaps, starts, backgrounds = (
        set(), set(), set(), set(), set(), set()
    )
    boundary = set()
    variable_palette = 0
    combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = [_inspect_pair(pair["input"], pair["output"])
                 for split in ("train", "test") for pair in puzzle[split]]
        if len({(i["background"], i["marker"], i["divider"])
                for i in infos}) > 1:
            variable_palette += 1
        for info in infos:
            shapes.add((info["height"], info["width"]))
            orientations.add(info["rotation"])
            profiles.add(info["profile"])
            gaps.add(info["gap"])
            starts.add(info["marker_start"])
            backgrounds.add(info["background"])
            if info["profile"] in ("none", "all"):
                boundary.add(info["profile"])
            if (
                info["rotation"] != 0 and info["marker_start"] > 0
                and info["qualifier_count"] >= 2
                and info["background"] != 0
            ):
                combined += 1
        for info in infos[len(puzzle["train"]):]:
            test_n.add(info["n"])
            test_q.add(info["qualifier_count"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and orientations == {0, 1, 2, 3}
    assert {"least", "most", "intermediate"} <= profiles
    assert boundary == {"none", "all"}
    assert len(test_n) >= 5 and len(test_q) >= 4
    assert len(gaps) >= 5 and len(starts) >= 6
    assert len(backgrounds) >= 8 and variable_palette >= 12
    assert combined >= 5
