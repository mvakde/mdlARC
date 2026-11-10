"""Four orientation-coded L markers expand to the four half-border corners."""

import numpy as np


_ORIENTATIONS = ("tl", "tr", "bl", "br")
_OFFSETS = {
    "tl": frozenset(((0, 0), (0, 1), (1, 0))),
    "tr": frozenset(((0, 0), (0, 1), (1, 1))),
    "bl": frozenset(((0, 0), (1, 0), (1, 1))),
    "br": frozenset(((0, 1), (1, 0), (1, 1))),
}
_TRAIN_SIZES = (5, 7, 9, 11, 13, 15, 17, 19, 21)
_ALL_SIZES = tuple(range(5, 30, 2))
_SOURCE_ROWS = (
    ("88228888855", "88288888885", "88888888888", "88188888885",
     "88118888855", "88888888888", "88888888888", "88888888888",
     "88888888888", "88888888888", "88888888888"),
    ("888888888", "888888888", "883388228", "883888828", "888888888",
     "887888808", "887788008", "888888888", "888888888"),
    ("88808", "88800", "22888", "82744", "87748"),
    ("888888888888888", "888888888888888", "888888888888888",
     "888888888888888", "888888888888888", "888888888888888",
     "888888888888888", "888888888888888", "888888888888888",
     "888888888888888", "888888888888888", "888888888828888",
     "888888888228888", "888886888338998", "888886688838988"),
)


def _signature(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.ravel()))


_SOURCE_SIGNATURES = {
    ((len(rows), len(rows[0])), tuple(int(ch) for row in rows for ch in row))
    for rows in _SOURCE_ROWS
}


def _counts(rng, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} is outside its legal range")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def sample_puzzle_parameters(rng):
    # Orientation-to-corner meaning is the only learned convention. Background
    # and marker colors are local roles visible in each input.
    return {
        "base_background": int(rng.integers(0, 10)),
        "vary_background": bool(rng.random() < 0.55),
    }


def _sample_payloads(rng, background, mode):
    available = [color for color in range(10) if color != background]
    if mode == "distinct":
        counts = (1, 1, 1, 1)
    elif mode == "one_pair":
        counts = (2, 1, 1)
    elif mode == "two_pairs":
        counts = (2, 2)
    elif mode == "triple":
        counts = (3, 1)
    else:
        raise ValueError("unknown color mode")
    selected = [int(x) for x in rng.choice(available, size=len(counts), replace=False)]
    values = [color for color, count in zip(selected, counts) for _ in range(count)]
    rng.shuffle(values)
    return tuple(values)


def _sample_size(rng, regime, train_sizes):
    if regime == "source_train":
        weights = np.array((0.22, 0.15, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05, 0.04))
        return int(rng.choice(_TRAIN_SIZES, p=weights))
    choice = int(rng.integers(0, 10))
    if choice < 4:
        larger = [n for n in _ALL_SIZES if n > max(train_sizes)]
        if larger:
            return int(rng.choice(larger))
    if choice < 7:
        return int(rng.choice(train_sizes))
    return int(rng.choice(_ALL_SIZES))


def _window(rng, size, compact, bottom):
    maximum = size - 1
    spans = []
    for _ in range(2):
        if compact:
            high = min(maximum, max(4, int(rng.integers(4, 9))))
        else:
            high = maximum
        spans.append(int(rng.integers(min(4, high), high + 1)))
    row_span, col_span = spans
    row_start = maximum - row_span if bottom else int(rng.integers(0, maximum - row_span + 1))
    col_start = int(rng.integers(0, maximum - col_span + 1))
    return [(r, c) for r in range(row_start, row_start + row_span)
            for c in range(col_start, col_start + col_span)]


def _marker_cells(origin, orientation):
    row, col = origin
    return frozenset((row + dr, col + dc) for dr, dc in _OFFSETS[orientation])


def _touch(first, second):
    return any(max(abs(r - s), abs(c - d)) <= 1
               for r, c in first for s, d in second)


def _place_in_window(rng, orientations, colors, candidates, contact_mode):
    orders = [rng.permutation(len(candidates)) for _ in range(4)]
    placed = []
    visits = [0]

    def search(depth, has_contact):
        if depth == 4:
            return contact_mode != 1 or has_contact
        orientation = orientations[depth]
        for index in orders[depth]:
            visits[0] += 1
            if visits[0] > 25000:
                return False
            origin = candidates[int(index)]
            cells = _marker_cells(origin, orientation)
            if any(cells & old_cells for _, old_cells, _ in placed):
                continue
            if any(colors[orientation] == old_color and _touch(cells, old_cells)
                   for _, old_cells, old_color in placed):
                continue
            new_contact = has_contact or any(_touch(cells, old_cells)
                                             for _, old_cells, _ in placed)
            if contact_mode == -1 and new_contact:
                continue
            placed.append((orientation, cells, colors[orientation]))
            if search(depth + 1, new_contact):
                return True
            placed.pop()
        return False

    if not search(0, False):
        return None
    return {orientation: cells for orientation, cells, _ in placed}


def _sample_markers(rng, size, colors, regime, force_compact=False):
    compact = force_compact or bool(rng.random() < (0.50 if regime == "source_train" else 0.56))
    bottom = bool(rng.random() < (0.20 if regime == "source_train" else 0.38))
    draw = rng.random()
    if compact:
        contact_mode = 1 if draw < 0.65 else (-1 if draw < 0.83 else 0)
    else:
        contact_mode = 1 if draw < 0.30 else (-1 if draw < 0.72 else 0)
    orientations = tuple(_ORIENTATIONS[int(i)] for i in rng.permutation(4))
    for attempt in range(12):
        candidates = _window(rng, size, compact if attempt < 8 else False, bottom)
        placed = _place_in_window(rng, orientations, colors, candidates, contact_mode)
        if placed is not None:
            return placed
    candidates = [(r, c) for r in range(size - 1) for c in range(size - 1)]
    placed = _place_in_window(rng, orientations, colors, candidates, contact_mode)
    if placed is None:
        raise RuntimeError("could not place four readable markers")
    return placed


def _output(size, background, colors):
    result = np.full((size, size), background, dtype=np.int8)
    middle = size // 2
    result[0, :middle] = colors["tl"]
    result[:middle, 0] = colors["tl"]
    result[0, middle + 1:] = colors["tr"]
    result[:middle, -1] = colors["tr"]
    result[-1, :middle] = colors["bl"]
    result[middle + 1:, 0] = colors["bl"]
    result[-1, middle + 1:] = colors["br"]
    result[middle + 1:, -1] = colors["br"]
    return result


def generate_pair(rng, regime, size, background, payloads, force_compact=False):
    colors = dict(zip(_ORIENTATIONS, payloads))
    markers = _sample_markers(rng, size, colors, regime, force_compact)
    input_grid = np.full((size, size), background, dtype=np.int8)
    for orientation, cells in markers.items():
        for row, col in cells:
            input_grid[row, col] = colors[orientation]
    return input_grid, _output(size, background, colors)


def _analyze_input(grid):
    size, width = grid.shape
    assert size == width and size in _ALL_SIZES
    values, counts = np.unique(grid, return_counts=True)
    background_index = int(np.argmax(counts))
    background = int(values[background_index])
    assert int(counts[background_index]) == size * size - 12
    colors = {}
    origins = {}
    components = {}
    for value in values:
        color = int(value)
        if color == background:
            continue
        cells = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
        while cells:
            start = cells.pop()
            component = {start}
            frontier = [start]
            while frontier:
                row, col = frontier.pop()
                adjacent = {(r, c) for r, c in cells
                            if max(abs(r - row), abs(c - col)) <= 1}
                cells.difference_update(adjacent)
                component.update(adjacent)
                frontier.extend(adjacent)
            assert len(component) == 3
            row0 = min(r for r, _ in component)
            col0 = min(c for _, c in component)
            offsets = frozenset((r - row0, c - col0) for r, c in component)
            matches = [orientation for orientation, shape in _OFFSETS.items()
                       if offsets == shape]
            assert len(matches) == 1
            orientation = matches[0]
            assert orientation not in colors
            colors[orientation] = color
            origins[orientation] = (row0, col0)
            components[orientation] = frozenset(component)
    assert set(colors) == set(_ORIENTATIONS)
    return {"size": size, "background": background, "colors": colors,
            "origins": origins, "components": components}


def _mapping(info):
    return tuple(info["colors"][orientation] for orientation in _ORIENTATIONS)


def _contact_count(info):
    cells = info["components"]
    return sum(_touch(cells[a], cells[b]) for i, a in enumerate(_ORIENTATIONS)
               for b in _ORIENTATIONS[i + 1:])


def _misplaced(info):
    middle = info["size"] // 2
    return sum((orientation[0] == "t") != (info["origins"][orientation][0] < middle)
               or (orientation[1] == "l") != (info["origins"][orientation][1] < middle)
               for orientation in _ORIENTATIONS)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    train_infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_grid, output_grid = pair["input"], pair["output"]
            signature = _signature(input_grid)
            pair_signature = (signature, _signature(output_grid))
            assert signature not in seen_inputs and signature not in _SOURCE_SIGNATURES
            assert pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            info = _analyze_input(input_grid)
            assert output_grid.shape == input_grid.shape
            assert np.array_equal(output_grid, _output(info["size"], info["background"], info["colors"]))
            assert not np.array_equal(input_grid, output_grid)
            if split == "train":
                train_infos.append(info)
    assert any(len(set(_mapping(info))) == 4 and _misplaced(info) > 0
               for info in train_infos)
    assert any(len(set(_mapping(info))) < 4 for info in train_infos)


def _choose_background(rng, params, forbidden=()):
    legal = [color for color in range(10) if color not in forbidden]
    base = params["base_background"]
    if not params["vary_background"] and base in legal:
        return base
    if base in legal and rng.random() < 0.35:
        return base
    return int(rng.choice(legal))


def _test_payloads(rng, background, train_mappings):
    choice = int(rng.integers(0, 10))
    compatible = [mapping for mapping in train_mappings if background not in mapping]
    if compatible and choice < 3:
        return tuple(compatible[int(rng.integers(0, len(compatible)))])
    if compatible and choice < 6:
        values = list(compatible[int(rng.integers(0, len(compatible)))])
        index = int(rng.integers(0, 4))
        values[index] = int(rng.choice([c for c in range(10)
                                        if c != background and c != values[index]]))
        return tuple(values)
    mode = ("distinct", "one_pair", "two_pairs", "triple")[int(rng.integers(0, 4))]
    return _sample_payloads(rng, background, mode)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes = []
    train_mappings = []
    modes = ["distinct", ("one_pair", "two_pairs", "triple")[int(rng.integers(0, 3))]]
    modes.extend(("distinct", "one_pair", "two_pairs", "triple")[int(rng.integers(0, 4))]
                 for _ in range(train_count - 2))
    rng.shuffle(modes)
    small_crowded_index = int(rng.integers(0, train_count)) if rng.random() < 0.55 else -1
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        regime = "source_train" if split == "train" else "target_test"
        for attempt in range(24):
            if split == "train":
                size = 5 if index == small_crowded_index and attempt < 12 else _sample_size(rng, regime, train_sizes)
                if index == 0:
                    background = params["base_background"]
                elif index == 1 and params["vary_background"]:
                    background = int(rng.choice(sorted(set(train_mappings[0]))))
                else:
                    background = _choose_background(rng, params)
                payloads = _sample_payloads(rng, background, modes[index])
            else:
                size = _sample_size(rng, regime, train_sizes)
                background = _choose_background(rng, params)
                payloads = _test_payloads(rng, background, train_mappings)
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            input_grid, output_grid = generate_pair(pair_rng, regime, size, background,
                                                    payloads, index == small_crowded_index)
            signature = _signature(input_grid)
            if (signature in seen or signature in _SOURCE_SIGNATURES
                    or np.array_equal(input_grid, output_grid)):
                continue
            if split == "train" and modes[index] == "distinct":
                if _misplaced(_analyze_input(input_grid)) == 0:
                    continue
            puzzle[split].append({"input": input_grid, "output": output_grid})
            seen.add(signature)
            if split == "train":
                train_sizes.append(size)
                train_mappings.append(payloads)
            break
        else:
            raise RuntimeError("could not make a unique, instructive pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(np.array_equal(a[name], b[name])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for name in ("input", "output")) and all(
                   len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    for seed, counts in ((1001, (None, None)), (2002, (6, 3)),
                         (4000, (2, 1)), (4001, (10, 1)), (4002, (10, 4)),
                         (5000, (None, 1)), (5001, (None, 4)),
                         (5002, (2, None)), (5003, (10, None))):
        puzzle = generate_puzzle(seed, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    first = generate_puzzle(3003, 7, 3)
    assert _puzzles_equal(first, generate_puzzle(3003, 7, 3))
    assert not _puzzles_equal(first, generate_puzzle(3004, 7, 3))
    for index, counts in enumerate(((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4))):
        try:
            generate_puzzle(6000 + index, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")

    train_counts, test_counts, sizes, layouts, backgrounds = set(), set(), set(), set(), set()
    contacts = small_contacts = repeat_pairs = mixed_backgrounds = 0
    separated = edge_contacts = diagonal_contacts = irregular = 0
    bottom_train = center_crossings = corner_clusters = background_swaps = 0
    contact_and_separated = combined_axes = 0
    distinct_counts = {2: 0, 3: 0, 4: 0}
    multiplicities = {(1, 1, 1, 1): 0, (1, 1, 2): 0, (2, 2): 0, (1, 3): 0}
    same_size_tests = larger_tests = upper_tests = reused = one_change = 0
    size_only = packing_only = 0
    for seed in range(7000, 7042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        test_infos = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        all_infos = train_infos + test_infos
        mixed_backgrounds += len({info["background"] for info in all_infos}) > 1
        background_swaps += any(a["background"] in _mapping(b)
                                for a in all_infos for b in all_infos if a is not b)
        contact_and_separated += (any(_contact_count(info) for info in all_infos)
                                  and any(not _contact_count(info) for info in all_infos))
        train_sizes = {info["size"] for info in train_infos}
        train_mappings = [_mapping(info) for info in train_infos]
        for info in all_infos:
            sizes.add(info["size"])
            backgrounds.add(info["background"])
            layouts.add((info["size"], tuple(info["origins"][o] for o in _ORIENTATIONS)))
            contacts += _contact_count(info) > 0
            separated += _contact_count(info) == 0
            small_contacts += info["size"] == 5 and _contact_count(info) > 0
            repeat_pairs += len(set(_mapping(info))) < 4
            distinct_counts[len(set(_mapping(info)))] += 1
            pattern = tuple(sorted(_mapping(info).count(color)
                                   for color in set(_mapping(info))))
            multiplicities[pattern] += 1
            rows = [row for row, _ in info["origins"].values()]
            columns = [column for _, column in info["origins"].values()]
            irregular += (len(set(rows)) > 1 and len(set(columns)) > 1
                          and set(info["origins"].values()) != {
                              (row, column) for row in set(rows)
                              for column in set(columns)})
            center_crossings += (min(rows) < info["size"] // 2 <= max(rows)
                                 or min(columns) < info["size"] // 2 <= max(columns))
            corner_clusters += info["size"] >= 9 and any(
                sum((row <= 4 if vertical == "t" else row >= info["size"] - 6)
                    and (column <= 4 if horizontal == "l" else column >= info["size"] - 6)
                    for row, column in info["origins"].values()) >= 3
                for vertical in ("t", "b") for horizontal in ("l", "r"))
            for index, first in enumerate(_ORIENTATIONS):
                for second in _ORIENTATIONS[index + 1:]:
                    first_cells = info["components"][first]
                    second_cells = info["components"][second]
                    if not _touch(first_cells, second_cells):
                        continue
                    edge = any(abs(r - s) + abs(c - d) == 1
                               for r, c in first_cells for s, d in second_cells)
                    edge_contacts += edge
                    diagonal_contacts += not edge
            combined_axes += (_contact_count(info) > 0
                              and len(set(_mapping(info))) < 4
                              and len({item["background"] for item in all_infos}) > 1)
        bottom_train += any(info["size"] >= 9 and min(
            row for row, _ in info["origins"].values()) >= info["size"] - 5
                            for info in train_infos)
        for info in test_infos:
            same_size_tests += info["size"] in train_sizes
            larger_tests += info["size"] > max(train_sizes)
            upper_tests += min(r for r, _ in info["origins"].values()) < info["size"] // 2
            mapping = _mapping(info)
            reused += mapping in train_mappings
            one_change += any(sum(a != b for a, b in zip(mapping, old)) == 1
                              for old in train_mappings)
            size_only += info["size"] > max(train_sizes) and _contact_count(info) == 0
            packing_only += info["size"] in train_sizes and _contact_count(info) > 0
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 10 and len(layouts) >= 100 and len(backgrounds) >= 8
    assert contacts >= 40 and small_contacts >= 5 and repeat_pairs >= 30
    assert separated >= 20 and edge_contacts >= 20 and diagonal_contacts >= 10
    assert irregular >= 50 and center_crossings >= 30 and corner_clusters >= 5
    assert bottom_train >= 3 and background_swaps >= 10
    assert contact_and_separated >= 10 and combined_axes >= 5
    assert all(distinct_counts[count] >= 10 for count in (2, 3, 4))
    assert all(multiplicities[pattern] >= 5 for pattern in multiplicities)
    assert mixed_backgrounds >= 10 and same_size_tests >= 10 and larger_tests >= 10
    assert upper_tests >= 10 and reused >= 3 and one_change >= 3
    assert size_only >= 3 and packing_only >= 3
