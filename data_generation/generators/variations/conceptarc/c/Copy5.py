import numpy as np


_SOURCE_INPUTS = {
    ("44441", "44440", "44440", "44440"),
    ("33311", "33300", "33300"),
    ("6666", "6446", "6666", "1000", "1000", "1000"),
    ("777", "737", "777", "100", "100"),
    ("99991", "99990", "92290", "99990", "99990"),
    ("111", "161", "111", "100"),
}


def _signature(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _repeat(motif, orientation, count, blank):
    axis = 1 if orientation == "horizontal" else 0
    gap_shape = (motif.shape[0], 1) if axis else (1, motif.shape[1])
    gap = np.full(gap_shape, blank, dtype=np.int8)
    parts = []
    for index in range(count + 1):
        if index:
            parts.append(gap)
        parts.append(motif)
    return np.concatenate(parts, axis=axis)


def _decode(grid):
    """Recover the unique full rectangle and one-color counting band."""
    height, width = grid.shape
    found = []
    for orientation in ("horizontal", "vertical"):
        for count in range(1, 4):
            for side in ("before", "after"):
                if orientation == "horizontal":
                    if width - count < 3:
                        continue
                    band = grid[:, :count] if side == "before" else grid[:, -count:]
                    motif = grid[:, count:] if side == "before" else grid[:, :-count]
                else:
                    if height - count < 3:
                        continue
                    band = grid[:count, :] if side == "before" else grid[-count:, :]
                    motif = grid[count:, :] if side == "before" else grid[:-count, :]
                if min(motif.shape) < 3 or not np.all(motif > 0):
                    continue
                base = int(motif[0, 0])
                if not (np.all(motif[0] == base) and np.all(motif[-1] == base)
                        and np.all(motif[:, 0] == base) and np.all(motif[:, -1] == base)):
                    continue
                for edge in ("first", "last"):
                    if orientation == "horizontal":
                        marks = band[0] if edge == "first" else band[-1]
                        filler = band[1:] if edge == "first" else band[:-1]
                    else:
                        marks = band[:, 0] if edge == "first" else band[:, -1]
                        filler = band[:, 1:] if edge == "first" else band[:, :-1]
                    marker, blank = int(marks[0]), int(filler.flat[0])
                    if (marker > 0 and marker != blank and blank not in motif
                            and np.all(marks == marker) and np.all(filler == blank)):
                        found.append((orientation, motif, count, marker, blank, side, edge))
    assert len(found) == 1, "Input must have one motif/band decomposition"
    return found[0]


def _choose_counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _placement(rng, quiet):
    if quiet or rng.random() < 0.48:
        return ("after", "first")
    return ("before" if rng.random() < 0.5 else "after",
            "last" if rng.random() < 0.5 else "first")


def _motif(rng, height, width, marker, blank, reuse, style):
    main_choices = [c for c in range(1, 10)
                    if c != blank and (reuse == "base" or c != marker)]
    main = marker if reuse == "base" else int(rng.choice(main_choices))
    motif = np.full((height, width), main, dtype=np.int8)
    interior = [(r, c) for r in range(1, height - 1)
                for c in range(1, width - 1)]
    if style == "solid":
        return motif
    if len(interior) == 1:
        cells = [interior[0]]
    elif style == "dot":
        cells = [interior[int(rng.integers(len(interior)))]]
    elif style == "block":
        ih, iw = height - 2, width - 2
        sizes = [(h, w) for h in range(1, ih + 1)
                 for w in range(1, iw + 1) if 2 <= h * w <= max(2, int(0.75 * ih * iw))]
        bh, bw = sizes[int(rng.integers(len(sizes)))] if sizes else (ih, iw)
        top = int(rng.integers(1, height - bh))
        left = int(rng.integers(1, width - bw))
        cells = [(r, c) for r in range(top, top + bh)
                 for c in range(left, left + bw)]
    elif style == "bar":
        horizontal = (width > 3 and (height == 3 or rng.random() < 0.5))
        if horizontal:
            length = int(rng.integers(2, width - 1))
            row = int(rng.integers(1, height - 1))
            left = int(rng.integers(1, width - length))
            cells = [(row, c) for c in range(left, left + length)]
        elif height > 3:
            length = int(rng.integers(2, height - 1))
            col = int(rng.integers(1, width - 1))
            top = int(rng.integers(1, height - length))
            cells = [(r, col) for r in range(top, top + length)]
        else:
            cells = interior[:2]
    else:
        target = int(rng.integers(2, min(len(interior), 7) + 1))
        cells = [interior[int(rng.integers(len(interior)))]]
        while len(cells) < target:
            neighbors = [(r, c) for r, c in interior if (r, c) not in cells
                         and any(abs(r - a) + abs(c - b) == 1 for a, b in cells)]
            choices = neighbors if neighbors else [p for p in interior if p not in cells]
            cells.append(choices[int(rng.integers(len(choices)))])
    options = [c for c in range(1, 10) if c not in (main, blank)
               and (reuse == "accent" or c != marker)]
    accent = marker if reuse == "accent" else int(rng.choice(options))
    for r, c in cells:
        motif[r, c] = accent
    if style == "multicolor" and len(cells) >= 2:
        other = int(rng.choice([c for c in options if c != accent and c != marker]))
        for r, c in cells[len(cells) // 2:]:
            motif[r, c] = other
    return motif


def _generate_pair(rng, spec, quiet, fixed_marker, tinted):
    orientation, count, side, edge, reuse = spec
    max_axis = (30 - count) // (count + 1)
    axis_size = int(rng.integers(3, min(10, max_axis) + 1))
    cross_size = int(rng.integers(3, 10))
    height, width = ((cross_size, axis_size) if orientation == "horizontal"
                     else (axis_size, cross_size))
    blank_choices = [c for c in range(2, 10) if c != fixed_marker]
    blank = int(rng.choice(blank_choices)) if tinted else 0
    marker_options = [c for c in range(1, 10) if c != blank]
    marker = fixed_marker if fixed_marker is not None else int(rng.choice(marker_options))
    styles = ("solid", "dot", "block", "bar", "cluster", "multicolor")
    weights = ([0.35, 0.35, 0.18, 0.08, 0.04, 0.0] if quiet
               else [0.18, 0.13, 0.29, 0.17, 0.13, 0.10])
    if reuse == "accent":
        weights[0] = 0
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    style = styles[int(rng.choice(len(styles), p=weights))]
    motif = _motif(rng, height, width, marker, blank, reuse, style)
    if orientation == "horizontal":
        band = np.full((height, count), blank, dtype=np.int8)
        band[0 if edge == "first" else -1, :] = marker
        input_grid = np.concatenate((band, motif) if side == "before" else (motif, band), axis=1)
    else:
        band = np.full((count, width), blank, dtype=np.int8)
        band[:, 0 if edge == "first" else -1] = marker
        input_grid = np.concatenate((band, motif) if side == "before" else (motif, band), axis=0)
    return {"input": input_grid, "output": _repeat(motif, orientation, count, blank)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    quiet = bool(rng.random() < 0.18)
    fixed_marker = int(rng.integers(1, 10)) if quiet or rng.random() < 0.3 else None
    allow_tint = not quiet and bool(rng.random() < 0.45)
    if num_train == 2:
        axis = "horizontal" if rng.random() < 0.5 else "vertical"
        counts = list(rng.choice([1, 2, 3], size=2, replace=False))
        orientations = [axis, axis]
    else:
        counts = [1, 2, 3] + [int(rng.integers(1, 4)) for _ in range(num_train - 3)]
        orientations = ["horizontal", "vertical"] + [
            "horizontal" if rng.random() < 0.5 else "vertical" for _ in range(num_train - 2)]
    rng.shuffle(counts)
    rng.shuffle(orientations)
    train_specs = []
    for axis, count in zip(orientations, counts):
        side, edge = _placement(rng, quiet)
        reuse = str(rng.choice(["distinct", "base", "accent"], p=[0.53, 0.27, 0.20]))
        train_specs.append((axis, count, side, edge, reuse))
    if quiet:
        train_specs = [(a, c, s, e, "distinct") for a, c, s, e, _ in train_specs]
    train_places = [(a, s, e) for a, _, s, e, _ in train_specs]
    train_roles = sorted({spec[4] for spec in train_specs} | {"distinct"})
    max_count = max(counts)
    test_specs = []
    for _ in range(num_test):
        axis, side, edge = train_places[int(rng.integers(len(train_places)))]
        count = (int(rng.choice(counts)) if num_train == 2
                 else int(rng.integers(1, max_count + 1)))
        reuse = train_roles[int(rng.integers(len(train_roles)))]
        test_specs.append((axis, count, side, edge, reuse))
    train_tints = [False] * num_train
    test_tints = [False] * num_test
    if allow_tint:
        train_tints[int(rng.integers(num_train))] = True
        train_tints = [x or rng.random() < 0.35 for x in train_tints]
        test_tints = [bool(rng.random() < 0.45) for _ in test_tints]
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, specs, tints in (("train", train_specs, train_tints),
                                  ("test", test_specs, test_tints)):
        for spec, tinted in zip(specs, tints):
            for _ in range(128):
                pair = _generate_pair(rng, spec, quiet, fixed_marker, tinted)
                key = _signature(pair["input"])
                if key in seen or key in _SOURCE_INPUTS:
                    continue
                try:
                    _decode(pair["input"])
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[regime].append(pair)
                break
            else:
                raise RuntimeError("Could not construct an unambiguous distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
    seen = set()
    records = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _signature(pair["input"])
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            axis, motif, count, marker, blank, side, edge = _decode(pair["input"])
            assert 1 <= count <= 3 and 3 <= min(motif.shape)
            assert np.array_equal(pair["output"], _repeat(motif, axis, count, blank))
            assert not np.array_equal(pair["input"], pair["output"])
            role = ("base" if marker == motif[0, 0] else
                    "accent" if marker in motif else "distinct")
            records[regime].append((axis, count, side, edge, role, blank))
    train, test = records["train"], records["test"]
    assert len({x[1] for x in train}) >= 2
    if nt == 2:
        assert len({x[0] for x in train}) == 1
    else:
        assert {x[0] for x in train} == {"horizontal", "vertical"}
        assert {x[1] for x in train} == {1, 2, 3}
    placements = {(a, s, e) for a, _, s, e, _, _ in train}
    roles = {x[4] for x in train} | {"distinct"}
    for axis, count, side, edge, role, blank in test:
        assert (axis, side, edge) in placements
        assert count <= max(x[1] for x in train)
        assert role in roles
        if blank != 0:
            assert any(x[5] != 0 for x in train)


def _same_puzzle(left, right):
    return all(np.array_equal(a[field], b[field])
               for regime in ("train", "test")
               for a, b in zip(left[regime], right[regime])
               for field in ("input", "output")) and all(
                   len(left[r]) == len(right[r]) for r in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, layouts, shapes, roles, placements = (set() for _ in range(6))
    motifs, blanks, styles = set(), set(), set()
    fingerprints = set()
    coverage = {name: 0 for name in (
        "test_solid", "test_count3", "horizontal3_test", "vertical3_test",
        "train_base_overlap", "train_accent_overlap", "test_overlap",
        "train_block2", "test_block2", "train_multicolor", "test_multicolor",
        "train_mirror", "test_mirror", "train_tint", "test_tint",
        "rich_overlap_mirror", "varied_marker_puzzles", "varied_blank_puzzles",
        "test_larger", "test_equal", "test_smaller")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_signature(p["input"]) for r in ("train", "test") for p in puzzle[r]))
        train_markers, train_blanks = set(), set()
        train_axis_sizes = {}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                axis, motif, count, marker, blank, side, edge = _decode(pair["input"])
                assert np.array_equal(pair["output"], _repeat(motif, axis, count, blank))
                layouts.add((axis, count, *motif.shape, side, edge))
                shapes.add((pair["input"].shape, pair["output"].shape))
                roles.add("base" if marker == motif[0, 0] else "accent" if marker in motif else "distinct")
                placements.add((axis, side, edge))
                blanks.add(blank)
                motifs.add(_signature(motif))
                color_count = len(np.unique(motif))
                accent_count = int(np.count_nonzero(motif != motif[0, 0]))
                styles.add((color_count, accent_count))
                if regime == "train":
                    train_markers.add(marker)
                    train_blanks.add(blank)
                    train_axis_sizes.setdefault(axis, []).append(
                        motif.shape[1] if axis == "horizontal" else motif.shape[0])
                if regime == "test":
                    coverage["test_solid"] += accent_count == 0
                    coverage["test_count3"] += count == 3
                    coverage[f"{axis}3_test"] += count == 3
                    size = motif.shape[1] if axis == "horizontal" else motif.shape[0]
                    largest = max(train_axis_sizes[axis])
                    coverage["test_larger"] += size > largest
                    coverage["test_equal"] += size == largest
                    coverage["test_smaller"] += size < largest
                    coverage["test_overlap"] += marker in motif
                else:
                    coverage["train_base_overlap"] += marker == motif[0, 0]
                    coverage["train_accent_overlap"] += marker in motif and marker != motif[0, 0]
                coverage[f"{regime}_multicolor"] += color_count >= 3
                coverage[f"{regime}_mirror"] += (side, edge) != ("after", "first")
                coverage[f"{regime}_tint"] += blank != 0
                coverage[f"{regime}_block2"] += any(
                    motif[r, c] != motif[0, 0]
                    and np.all(motif[r:r + 2, c:c + 2] == motif[r, c])
                    for r in range(1, motif.shape[0] - 2)
                    for c in range(1, motif.shape[1] - 2))
                coverage["rich_overlap_mirror"] += (
                    accent_count >= 4 and marker in motif
                    and (side, edge) != ("after", "first"))
        coverage["varied_marker_puzzles"] += len(train_markers) > 1
        coverage["varied_blank_puzzles"] += len(train_blanks) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(layouts) >= 80 and len(shapes) >= 40
    assert len(motifs) >= 150 and len(styles) >= 12
    assert roles == {"base", "accent", "distinct"} and len(placements) == 8
    assert 0 in blanks and len(blanks) >= 3
    assert all(coverage[name] > 0 for name in coverage)
    for index, (nt, nv) in enumerate(((2, 1), (10, 1), (10, 4), (3, 2), (5, 3))):
        puzzle = generate_puzzle(200 + index, nt, nv)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, nv)
        assert _same_puzzle(puzzle, generate_puzzle(200 + index, nt, nv))
    for index, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                    {"num_train": 2}, {"num_train": 10})):
        puzzle = generate_puzzle(300 + index, **kwargs)
        validate_generated_puzzle(puzzle)
        for key, value in kwargs.items():
            assert len(puzzle["train" if key == "num_train" else "test"]) == value
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
