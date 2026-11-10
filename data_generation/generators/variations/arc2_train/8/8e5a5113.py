"""Solved ARC strips made from successive clockwise turns of a source square."""

import numpy as np


_SOURCE_ROWS = (
    (("63350005000", "63350005000", "63250005000"),
     ("63356665236", "63353335336", "63252335336")),
    (("11250005000", "41150005000", "44150005000"),
     ("11254415144", "41154115114", "44151125211")),
    (("27850005000", "77850005000", "88850005000"),
     ("27858725888", "77858775877", "88858885872")),
    (("33950005000", "99950005000", "29950005000"),
     ("33952935992", "99959935999", "29959995933")),
)


def _source_array(rows):
    return np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = tuple(
    (_source_array(inp), _source_array(out)) for inp, out in _SOURCE_ROWS
)


def _key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _original_pair(inp, out):
    return any(np.array_equal(inp, old_in) and np.array_equal(out, old_out)
               for old_in, old_out in _SOURCE_PAIRS)


def _count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _mode_roles(rng, n, mode):
    """Build geometry in color-independent roles before assigning ARC colors."""
    area = n * n
    if mode == "two":
        roles = rng.integers(0, 2, size=(n, n), dtype=np.int8)
        if min(np.bincount(roles.flat, minlength=2)) == 0:
            roles[0, 0], roles[-1, -1] = 0, 1
    elif mode == "rich":
        count = int(rng.integers(4, min(6, area) + 1))
        flat = np.arange(area, dtype=np.int8) % count
        rng.shuffle(flat)
        roles = flat.reshape(n, n)
    elif mode in ("corner", "edge", "center"):
        base_count = int(rng.integers(1, min(3, area - 1) + 1))
        roles = rng.integers(0, base_count, size=(n, n), dtype=np.int8)
        if mode == "corner":
            location = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)][
                int(rng.integers(0, 4))]
        elif mode == "edge":
            side = int(rng.integers(0, 4))
            offset = int(rng.integers(1, n - 1))
            location = ((0, offset), (n - 1, offset),
                        (offset, 0), (offset, n - 1))[side]
        else:
            location = (int(rng.integers(1, n - 1)),
                        int(rng.integers(1, n - 1)))
        roles[location] = base_count
    elif mode == "no_singleton":
        count = int(rng.integers(2, min(5, area // 2) + 1))
        flat = np.concatenate((
            np.repeat(np.arange(count, dtype=np.int8), 2),
            rng.integers(0, count, size=area - 2 * count, dtype=np.int8),
        ))
        rng.shuffle(flat)
        roles = flat.reshape(n, n)
    elif mode == "bar":
        roles = np.zeros((n, n), dtype=np.int8)
        axis = int(rng.integers(0, 2))
        location = int(rng.integers(0, n))
        if axis:
            roles[:, location] = 1
        else:
            roles[location, :] = 1
        if n >= 3 and rng.integers(0, 2):
            roles[int(rng.integers(0, n)), int(rng.integers(0, n))] = 2
    elif mode == "blocks":
        roles = np.zeros((n, n), dtype=np.int8)
        for role in (1, 2):
            r0 = int(rng.integers(0, n))
            c0 = int(rng.integers(0, n))
            r1 = int(rng.integers(r0 + 1, n + 1))
            c1 = int(rng.integers(c0 + 1, n + 1))
            roles[r0:r1, c0:c1] = role
    elif mode == "diagonal":
        roles = np.zeros((n, n), dtype=np.int8)
        offset = int(rng.integers(0, n))
        for row in range(n):
            roles[row, (row + offset) % n] = 1
        if rng.integers(0, 2):
            roles[int(rng.integers(0, n)), :] = 2
    elif mode == "symmetric":
        roles = rng.integers(0, 3, size=(n, n), dtype=np.int8)
        for row in range(n):
            for col in range(n):
                roles[n - 1 - row, n - 1 - col] = roles[row, col]
    else:
        raise ValueError(f"unknown motif mode: {mode}")
    return roles


def _make_motif(rng, n, background, divider, mode, strong):
    for _ in range(120):
        roles = _mode_roles(rng, n, mode)
        unique = np.unique(roles)
        if len(unique) < 2:
            continue
        # Reindex actual roles so overwritten shapes need no special cases.
        compact = np.searchsorted(unique, roles)
        palette = [int(c) for c in rng.permutation(10) if c != divider]
        if rng.random() < 0.43:
            palette.remove(background)
            palette = [background] + palette
        else:
            palette.remove(background)
        chosen = np.asarray(palette[:len(unique)], dtype=np.int8)
        motif = chosen[compact]
        if strong and len({_key(np.rot90(motif, k=k)) for k in range(4)}) != 4:
            continue
        return motif
    # A color-independent asymmetric fallback makes the construction bounded.
    roles = np.zeros((n, n), dtype=np.int8)
    roles[0, 0], roles[0, -1], roles[-1, 0] = 1, 2, 3
    colors = [int(c) for c in rng.permutation(10) if c != divider]
    return np.asarray(colors[:4], dtype=np.int8)[roles]


def _windows(n, panels, thickness, orientation):
    if orientation == "H":
        return [(slice(None), slice(i * (n + thickness),
                                     i * (n + thickness) + n))
                for i in range(panels)]
    return [(slice(i * (n + thickness), i * (n + thickness) + n),
             slice(None)) for i in range(panels)]


def _bars(n, panels, thickness, orientation):
    if orientation == "H":
        return [(slice(None), slice(i * (n + thickness) + n,
                                     (i + 1) * (n + thickness)))
                for i in range(panels - 1)]
    return [(slice(i * (n + thickness) + n,
                   (i + 1) * (n + thickness)), slice(None))
            for i in range(panels - 1)]


def _construct(motif, panels, thickness, orientation, background, divider):
    n = motif.shape[0]
    long_side = panels * n + (panels - 1) * thickness
    shape = (n, long_side) if orientation == "H" else (long_side, n)
    inp = np.full(shape, background, dtype=np.int8)
    for bar in _bars(n, panels, thickness, orientation):
        inp[bar] = divider
    windows = _windows(n, panels, thickness, orientation)
    inp[windows[0]] = motif
    out = inp.copy()
    for index in range(1, panels):
        out[windows[index]] = np.rot90(motif, k=-index)
    return inp, out


def _parse_input(inp):
    height, width = inp.shape
    assert height != width
    orientation = "H" if width > height else "V"
    n = min(height, width)
    long_side = max(height, width)
    assert n >= 2
    candidates = []
    for panels in (2, 3, 4):
        for thickness in (1, 2, 3):
            if panels * n + (panels - 1) * thickness != long_side:
                continue
            windows = _windows(n, panels, thickness, orientation)
            bars = _bars(n, panels, thickness, orientation)
            background = int(inp[windows[1]][0, 0])
            divider = int(inp[bars[0]][0, 0])
            source = inp[windows[0]]
            if background == divider or np.any(source == divider):
                continue
            if len(np.unique(source)) < 2:
                continue
            if not all(np.all(inp[bar] == divider) for bar in bars):
                continue
            if not all(np.all(inp[window] == background) for window in windows[1:]):
                continue
            candidates.append((n, panels, thickness, orientation,
                               background, divider, windows, bars))
    assert len(candidates) == 1, "ambiguous or malformed panel segmentation"
    return candidates[0]


def _learn_rotations(train):
    possible = {index: set(range(4)) for index in (1, 2, 3)}
    seen = set()
    for pair in train:
        inp, out = pair["input"], pair["output"]
        _, panels, _, _, _, _, windows, _ = _parse_input(inp)
        source = inp[windows[0]]
        for index in range(1, panels):
            seen.add(index)
            possible[index] &= {
                turn for turn in range(4)
                if np.array_equal(out[windows[index]], np.rot90(source, k=-turn))
            }
    assert {1, 2}.issubset(seen), "training must demonstrate two turns"
    assert all(len(possible[index]) == 1 for index in seen), (
        "training leaves the direction or sequence ambiguous"
    )
    return {index: next(iter(possible[index])) for index in seen}


def _predict_from_training(train, test_input):
    learned = _learn_rotations(train)
    _, panels, _, _, _, _, windows, _ = _parse_input(test_input)
    assert all(index in learned for index in range(1, panels))
    result = test_input.copy()
    source = test_input[windows[0]]
    for index in range(1, panels):
        result[windows[index]] = np.rot90(source, k=-learned[index])
    return result


def _pair_rng(rng):
    return np.random.default_rng(int(rng.integers(0, 2**63 - 1)))


def _sample_pair(rng, panels, orientation, n, thickness, mode, strong,
                 forbidden_inputs):
    for _ in range(160):
        colors = [int(c) for c in rng.permutation(10)]
        background, divider = colors[:2]
        motif = _make_motif(rng, n, background, divider, mode, strong)
        inp, out = _construct(motif, panels, thickness, orientation,
                              background, divider)
        if _key(inp) in forbidden_inputs or _original_pair(inp, out):
            continue
        # Input geometry must identify the intended panel boundaries.
        try:
            _parse_input(inp)
        except AssertionError:
            continue
        forbidden_inputs.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("unable to construct a distinct, clearly segmented pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    total = train_count + test_count

    # The operation and panel order are the only conventions shared by a puzzle.
    # Some puzzles demonstrate both strip orientations; others transfer the
    # learned clockwise turn to an orientation held out until test time.
    orientation_regime = int(rng.integers(0, 4))
    first_orientation = "H" if rng.integers(0, 2) else "V"
    include_four = bool(rng.integers(0, 2))
    modes = ("two", "rich", "corner", "edge", "center",
             "no_singleton", "bar", "blocks", "diagonal", "symmetric")
    mode_offset = int(rng.integers(0, len(modes)))
    used_inputs = set()
    train, test = [], []
    first_size = None
    first_thickness = None

    for index in range(total):
        pair_rng = _pair_rng(rng)
        training = index < train_count
        local_index = index if training else index - train_count

        if orientation_regime < 2:
            orientation = first_orientation if training else (
                "V" if first_orientation == "H" else "H")
        elif training and index < 2:
            orientation = first_orientation if index == 0 else (
                "V" if first_orientation == "H" else "H")
        else:
            orientation = "H" if pair_rng.integers(0, 2) else "V"

        if training and index == 0:
            panels = 3
        elif training and index == 1 and include_four:
            panels = 4
        else:
            upper = 5 if include_four else 4
            panels = int(pair_rng.integers(2, upper))

        maximum_size = (30 - (panels - 1)) // panels
        if training and index < 2:
            candidates = list(range(3, min(6, maximum_size) + 1))
            if index == 1:
                candidates = [size for size in candidates if size != first_size]
            n = int(pair_rng.choice(candidates))
        else:
            category = int(pair_rng.integers(0, 3))
            if category == 0:
                candidates = range(2, min(3, maximum_size) + 1)
            elif category == 1:
                candidates = range(3, min(6, maximum_size) + 1)
            else:
                candidates = range(7, maximum_size + 1)
                if not candidates:
                    candidates = range(max(2, maximum_size - 1), maximum_size + 1)
            if category == 2 and len(candidates) >= 3 and pair_rng.random() < 0.45:
                n = int(pair_rng.choice((maximum_size - 1, maximum_size)))
            else:
                n = int(pair_rng.choice(list(candidates)))

        maximum_thickness = min(3, (30 - panels * n) // (panels - 1))
        choices = list(range(1, maximum_thickness + 1))
        if training and index == 1 and len(choices) > 1:
            choices = [value for value in choices if value != first_thickness]
        thickness = int(pair_rng.choice(choices))
        if training and index == 0:
            first_size, first_thickness = n, thickness

        mode = modes[(mode_offset + index * 3) % len(modes)]
        if n == 2 and mode in ("edge", "center"):
            mode = "no_singleton"
        if training and index < 2 and mode == "symmetric":
            mode = "rich"
        strong = bool(training and index < 2)
        pair = _sample_pair(pair_rng, panels, orientation, n, thickness,
                            mode, strong, used_inputs)
        (train if training else test).append(pair)

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_key = _key(inp)
            pair_key = (input_key, _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _original_pair(inp, out)
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            n, panels, thickness, orientation, background, divider, windows, bars = (
                _parse_input(inp))
            source = inp[windows[0]]
            assert np.array_equal(out[windows[0]], source)
            assert all(np.array_equal(inp[bar], out[bar]) and
                       np.all(out[bar] == divider) for bar in bars)
            for index in range(1, panels):
                assert np.array_equal(out[windows[index]],
                                      np.rot90(source, k=-index))
            assert not np.array_equal(inp, out)
            metadata[split].append((n, panels, thickness, orientation,
                                    background, divider))
    learned = _learn_rotations(train)
    assert learned[1] == 1 and learned[2] == 2
    if any(item[1] == 4 for item in metadata["test"]):
        assert learned[3] == 3
    # Every expected answer is reconstructed from returned training pairs and
    # its input, without using generation parameters or the stored test output.
    for pair in test:
        assert np.array_equal(_predict_from_training(train, pair["input"]),
                              pair["output"])
    assert len({item[:4] for item in metadata["train"] + metadata["test"]}) >= 2
    assert len({item[4:6] for item in metadata["train"] + metadata["test"]}) >= 2


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple((_key(q["input"]), _key(q["output"]))
                      for split in ("train", "test") for q in p[split])
                for p in puzzles}) >= 40
    for seed, train, test in ((101, 2, 1), (102, 10, 1),
                              (103, 10, 4), (104, 6, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for seed, train, test in ((201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(300, *request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {request}")
    for seed, request in ((17, (None, None)), (23, (7, 3)),
                          (29, (None, 4))):
        first = generate_puzzle(seed, *request)
        second = generate_puzzle(seed, *request)
        for split in ("train", "test"):
            assert len(first[split]) == len(second[split])
            for a, b in zip(first[split], second[split]):
                assert np.array_equal(a["input"], b["input"])
                assert np.array_equal(a["output"], b["output"])

    sizes, lengths, thicknesses, orientations = set(), set(), set(), set()
    color_counts, singleton_locations = set(), set()
    blank_in_source = no_singleton = symmetric = near_limit = 0
    mixed_orientation = held_orientation = 0
    normalized_layouts = set()
    for puzzle in puzzles:
        train_orientations = set()
        test_orientations = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                n, panels, thickness, orientation, background, _, windows, _ = (
                    _parse_input(inp))
                source = inp[windows[0]]
                assert np.array_equal(out[windows[0]], source)
                for index in range(1, panels):
                    assert np.array_equal(out[windows[index]],
                                          np.rot90(source, k=-index))
                sizes.add(n)
                lengths.add(panels)
                thicknesses.add(thickness)
                orientations.add(orientation)
                (train_orientations if split == "train" else test_orientations).add(
                    orientation)
                colors, counts = np.unique(source, return_counts=True)
                color_counts.add(len(colors))
                blank_in_source += bool(np.any(source == background))
                no_singleton += bool(np.all(counts >= 2))
                symmetric += bool(np.array_equal(source, np.rot90(source, k=2)))
                near_limit += bool(max(inp.shape) >= 28)
                for color, count in zip(colors, counts):
                    if count == 1:
                        row, col = np.argwhere(source == color)[0]
                        location = ("corner" if row in (0, n - 1) and
                                    col in (0, n - 1) else
                                    "edge" if row in (0, n - 1) or
                                    col in (0, n - 1) else "center")
                        singleton_locations.add(location)
                roles = {int(color): index for index, color in enumerate(colors)}
                normalized_layouts.add((n, panels, thickness, orientation,
                                        tuple(roles[int(value)] for value in source.flat)))
        mixed_orientation += len(train_orientations) == 2
        held_orientation += not test_orientations.issubset(train_orientations)
    assert {2, 3, 4}.issubset(lengths)
    assert {1, 2, 3}.issubset(thicknesses)
    assert orientations == {"H", "V"}
    assert min(sizes) == 2 and max(sizes) >= 9 and len(sizes) >= 8
    assert {2, 3, 4}.issubset(color_counts)
    assert singleton_locations == {"corner", "edge", "center"}
    assert blank_in_source >= 15 and no_singleton >= 15
    assert symmetric >= 1 and near_limit >= 10
    assert mixed_orientation >= 5 and held_orientation >= 5
    assert len(normalized_layouts) >= 100
