import numpy as np


_TILE_MASKS = {
    "square": (
        (0, 0), (0, 1),
        (1, 0), (1, 1),
    ),
    "missing_upper_left": (
        (0, 1),
        (1, 0), (1, 1),
    ),
    "missing_lower_left": (
        (0, 0), (0, 1),
        (1, 1),
    ),
    "missing_upper_right": (
        (0, 0),
        (1, 0), (1, 1),
    ),
}

_TILE_ORDER = (
    "square",
    "missing_upper_left",
    "missing_lower_left",
    "missing_upper_right",
)

_SOURCE_INPUT_ROWS = (
    (
        "0800800",
        "8800880",
        "0880880",
        "0880880",
        "0088000",
        "0008000",
        "0000000",
    ),
    (
        "0088000",
        "8888800",
        "0880880",
        "0888800",
        "0008800",
        "0000000",
        "0000000",
    ),
    (
        "8880000",
        "8888000",
        "8808000",
        "0888000",
        "0000000",
        "0000000",
        "0000000",
    ),
    (
        "0880880",
        "8888880",
        "0880800",
        "0088800",
        "0000000",
        "0000000",
        "0000000",
    ),
)


def _validate_requested_count(
    value: int,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a Boolean")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _grid_value_key(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _is_source_layout(foreground: np.ndarray) -> bool:
    if foreground.shape != (7, 7):
        return False

    rows = tuple(
        "".join(
            "8" if bool(foreground[row, col]) else "0"
            for col in range(7)
        )
        for row in range(7)
    )
    return rows in _SOURCE_INPUT_ROWS


def _find_exact_covers(
    foreground: np.ndarray,
    limit: int = 2,
    masks: dict | None = None,
    output_grid: np.ndarray | None = None,
) -> list:
    """Return at most `limit` exact covers of a Boolean foreground mask."""
    if masks is None:
        masks = _TILE_MASKS
    positions = [
        (int(row), int(col))
        for row, col in np.argwhere(foreground)
    ]
    if not positions:
        return []

    position_index = {
        position: index for index, position in enumerate(positions)
    }
    height, width = foreground.shape

    # Candidate form:
    # (kind, anchor_row, anchor_col, cells, cell_bit_mask)
    candidates = []
    by_cell = [[] for _ in positions]

    for anchor_row in range(height - 1):
        for anchor_col in range(width - 1):
            for kind in _TILE_ORDER:
                cells = tuple(
                    (anchor_row + delta_row, anchor_col + delta_col)
                    for delta_row, delta_col in masks[kind]
                )

                # The unused corner of an L may be occupied by a touching
                # neighboring tile. Only the tile's own cells are required.
                if not all(
                    bool(foreground[row, col])
                    for row, col in cells
                ):
                    continue
                if output_grid is not None and len({
                    int(output_grid[row, col]) for row, col in cells
                }) != 1:
                    continue

                bit_mask = 0
                for cell in cells:
                    bit_mask |= 1 << position_index[cell]

                candidate_index = len(candidates)
                candidates.append(
                    (
                        kind,
                        anchor_row,
                        anchor_col,
                        cells,
                        bit_mask,
                    )
                )
                for cell in cells:
                    by_cell[position_index[cell]].append(candidate_index)

    full_mask = (1 << len(positions)) - 1
    solutions = []

    def search(used_mask: int, chosen: list) -> None:
        if len(solutions) >= limit:
            return

        if used_mask == full_mask:
            solutions.append(
                [candidates[index][:4] for index in chosen]
            )
            return

        best_options = None

        # Choose the uncovered cell with the fewest currently legal tiles.
        for cell_index in range(len(positions)):
            cell_bit = 1 << cell_index
            if used_mask & cell_bit:
                continue

            options = [
                candidate_index
                for candidate_index in by_cell[cell_index]
                if not (
                    candidates[candidate_index][4] & used_mask
                )
            ]
            if not options:
                return

            if (
                best_options is None
                or len(options) < len(best_options)
            ):
                best_options = options

        if best_options is None:
            return

        for candidate_index in best_options:
            search(
                used_mask | candidates[candidate_index][4],
                chosen + [candidate_index],
            )
            if len(solutions) >= limit:
                return

    search(0, [])
    return solutions


def _tile_counts(cover: list) -> dict:
    counts = {kind: 0 for kind in _TILE_ORDER}
    for kind, _, _, _ in cover:
        counts[kind] += 1
    return counts


def _tile_contact_edges(cover: list) -> int:
    """Count orthogonal cell adjacencies between different selected tiles."""
    owner = {}

    for tile_index, (_, _, _, cells) in enumerate(cover):
        for cell in cells:
            assert cell not in owner
            owner[cell] = tile_index

    contacts = 0
    for (row, col), tile_index in owner.items():
        for neighbor in ((row + 1, col), (row, col + 1)):
            if (
                neighbor in owner
                and owner[neighbor] != tile_index
            ):
                contacts += 1

    return contacts


def _foreground_components(foreground: np.ndarray) -> int:
    remaining = {
        (int(row), int(col))
        for row, col in np.argwhere(foreground)
    }
    components = 0

    while remaining:
        components += 1
        start = remaining.pop()
        stack = [start]

        while stack:
            row, col = stack.pop()
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)

    return components


def _trimmed_layout_key(foreground: np.ndarray) -> tuple:
    coordinates = np.argwhere(foreground)
    assert len(coordinates) > 0

    row_min = int(coordinates[:, 0].min())
    row_max = int(coordinates[:, 0].max())
    col_min = int(coordinates[:, 1].min())
    col_max = int(coordinates[:, 1].max())

    trimmed = foreground[
        row_min:row_max + 1,
        col_min:col_max + 1,
    ]
    return (
        int(trimmed.shape[0]),
        int(trimmed.shape[1]),
        tuple(int(value) for value in trimmed.reshape(-1)),
    )



_SOURCE_OUTPUT_ROWS = (
    ("0200400", "2200440", "0110110", "0110110", "0033000", "0003000", "0000000"),
    ("0011000", "3311400", "0320440", "0221100", "0001100", "0000000", "0000000"),
    ("1140000", "1144000", "3302000", "0322000", "0000000", "0000000", "0000000"),
    ("0240110", "2244110", "0330200", "0032200", "0000000", "0000000", "0000000"),
)
_SOURCE_PAIRS = frozenset(
    (
        (7, 7, tuple(int(c) for row in input_rows for c in row)),
        (7, 7, tuple(int(c) for row in output_rows for c in row)),
    )
    for input_rows, output_rows in zip(_SOURCE_INPUT_ROWS, _SOURCE_OUTPUT_ROWS)
)


def _transform_grid(grid: np.ndarray, symmetry: int) -> np.ndarray:
    transformed = np.rot90(grid, symmetry % 4)
    if symmetry >= 4:
        transformed = np.fliplr(transformed)
    return np.array(transformed, copy=True)


def _oriented_masks(symmetry: int) -> dict:
    masks = {}
    for kind, coordinates in _TILE_MASKS.items():
        tile = np.zeros((2, 2), dtype=np.int8)
        for row, col in coordinates:
            tile[row, col] = 1
        oriented = _transform_grid(tile, symmetry)
        masks[kind] = tuple((int(row), int(col)) for row, col in np.argwhere(oriented))
    return masks


def _inventory(rng: np.random.Generator, minimum: int = 1) -> list:
    count = int(rng.integers(minimum, 9))
    class_count = int(rng.integers(1, min(4, count) + 1))
    kinds = [str(item) for item in rng.choice(_TILE_ORDER, size=class_count, replace=False)]
    result = list(kinds)
    result.extend(str(item) for item in rng.choice(kinds, size=count - class_count))
    rng.shuffle(result)
    return result


def _layout_dimensions(rng: np.random.Generator, count: int, mode: str) -> tuple:
    if mode == "compact":
        side = int(np.ceil(np.sqrt(4 * count)))
        return min(17, side + int(rng.integers(0, 3))), min(17, side + int(rng.integers(0, 3)))
    if mode == "chain_h":
        return int(rng.integers(3, 6)), min(22, 2 * count + int(rng.integers(2, 6)))
    if mode == "chain_v":
        return min(22, 2 * count + int(rng.integers(2, 6))), int(rng.integers(3, 6))
    if mode == "separated":
        side = int(np.ceil(np.sqrt(count))) * 4 + 2
        return min(23, side + int(rng.integers(0, 5))), min(23, side + int(rng.integers(0, 5)))
    if mode == "diagonal":
        side = int(np.ceil(np.sqrt(count))) * 3 + 3
        return min(21, side + int(rng.integers(0, 5))), min(21, side + int(rng.integers(0, 5)))
    return int(rng.integers(5, 17)), int(rng.integers(5, 17))


def _place_tiles(rng: np.random.Generator, tile_types: list, mode: str) -> tuple | None:
    height, width = _layout_dimensions(rng, len(tile_types), mode)
    occupied = set()
    order = list(tile_types)
    rng.shuffle(order)
    for index, kind in enumerate(order):
        choices = []
        for anchor_row in range(height - 1):
            for anchor_col in range(width - 1):
                cells = tuple((anchor_row + dr, anchor_col + dc) for dr, dc in _TILE_MASKS[kind])
                if any(cell in occupied for cell in cells):
                    continue
                contact = 0
                diagonal = 0
                for row, col in cells:
                    contact += sum((row + dr, col + dc) in occupied for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
                    diagonal += sum((row + dr, col + dc) in occupied for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)))
                if occupied:
                    all_rows = [row for row, _ in occupied] + [row for row, _ in cells]
                    all_cols = [col for _, col in occupied] + [col for _, col in cells]
                    row_span = max(all_rows) - min(all_rows) + 1
                    col_span = max(all_cols) - min(all_cols) + 1
                else:
                    row_span = col_span = 2
                choices.append((cells, contact, diagonal, row_span, col_span))
        if not choices:
            return None
        if index:
            if mode in ("compact", "chain_h", "chain_v"):
                choices = [c for c in choices if c[1] > 0]
            elif mode == "separated":
                choices = [c for c in choices if c[1] == 0 and c[2] == 0]
            elif mode == "diagonal":
                choices = [c for c in choices if c[1] == 0 and c[2] > 0]
            elif mode == "mixed":
                want_contact = index % 2 == 1
                choices = [c for c in choices if (c[1] > 0) == want_contact]
            elif mode == "light":
                choices = [c for c in choices if c[1] <= 2]
        if not choices:
            return None
        if index and mode == "compact":
            choices.sort(key=lambda c: (c[3] * c[4] - c[1] * 2, -c[1]))
            choices = choices[:max(2, len(choices) // 5)]
        elif index and mode == "chain_h":
            choices.sort(key=lambda c: (c[3], -c[4], -c[1]))
            choices = choices[:max(2, len(choices) // 5)]
        elif index and mode == "chain_v":
            choices.sort(key=lambda c: (c[4], -c[3], -c[1]))
            choices = choices[:max(2, len(choices) // 5)]
        elif index and mode == "separated":
            choices.sort(key=lambda c: -(c[3] * c[4]))
            choices = choices[:max(2, len(choices) // 3)]
        elif index and mode == "diagonal":
            choices.sort(key=lambda c: -c[2])
            choices = choices[:max(2, len(choices) // 3)]
        selected = choices[int(rng.integers(len(choices)))]
        occupied.update(selected[0])
    foreground = np.zeros((height, width), dtype=bool)
    for row, col in occupied:
        foreground[row, col] = True
    return foreground


def _sample_layout(rng: np.random.Generator, tile_types: list, mode: str) -> tuple | None:
    expected = {kind: tile_types.count(kind) for kind in _TILE_ORDER}
    for _ in range(110):
        foreground = _place_tiles(rng, tile_types, mode)
        if foreground is None:
            continue
        covers = _find_exact_covers(foreground, limit=2)
        if len(covers) != 1 or _tile_counts(covers[0]) != expected:
            continue
        contacts = _tile_contact_edges(covers[0])
        components = _foreground_components(foreground)
        count = len(tile_types)
        if mode in ("compact", "chain_h", "chain_v") and components != 1:
            continue
        if mode == "separated" and components != count:
            continue
        if mode == "diagonal" and contacts != 0:
            continue
        if mode == "mixed" and count >= 3 and not (1 < components < count):
            continue
        if mode == "light" and contacts > max(4, count):
            continue
        return foreground, covers[0]
    return None


def _crop_and_pad(rng: np.random.Generator, foreground: np.ndarray, mode: str) -> np.ndarray:
    coords = np.argwhere(foreground)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    cropped = foreground[r0:r1, c0:c1]
    if mode == "compact" and float(rng.random()) < 0.55:
        pads = [int(rng.integers(0, 2)) for _ in range(4)]
    else:
        pads = [int(rng.integers(0, 5)) for _ in range(4)]
    if not any(pads) and bool(np.all(cropped)):
        pads[int(rng.integers(4))] = 1
    top, bottom, left, right = pads
    return np.pad(cropped, ((top, bottom), (left, right)), constant_values=False)


def _build_pair(rng: np.random.Generator, tile_types: list, mode: str,
                background: int, source: int, tile_colors: dict, symmetry: int) -> dict:
    for candidate_mode in (mode, mode, mode, "light", "mixed", "compact", "separated"):
        sampled = _sample_layout(rng, tile_types, candidate_mode)
        if sampled is None:
            continue
        foreground, _ = sampled
        foreground = _crop_and_pad(rng, foreground, candidate_mode)
        covers = _find_exact_covers(foreground, limit=2)
        if len(covers) != 1:
            continue
        h, w = foreground.shape
        if h > 30 or w > 30:
            continue
        input_grid = np.full((h, w), background, dtype=np.int8)
        output_grid = np.full((h, w), background, dtype=np.int8)
        input_grid[foreground] = source
        for kind, _, _, cells in covers[0]:
            for row, col in cells:
                output_grid[row, col] = tile_colors[kind]
        pair = {"input": _transform_grid(input_grid, symmetry),
                "output": _transform_grid(output_grid, symmetry)}
        if (_grid_value_key(pair["input"]), _grid_value_key(pair["output"])) not in _SOURCE_PAIRS:
            return pair
    raise RuntimeError("unable to construct a unique exact-cover layout")


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray, masks: dict) -> dict:
    assert input_grid.shape == output_grid.shape
    colors = [int(c) for c in np.unique(input_grid)]
    assert len(colors) == 2
    backgrounds = [c for c in colors if bool(np.all(output_grid[input_grid == c] == c))]
    assert len(backgrounds) == 1
    background = backgrounds[0]
    source = next(c for c in colors if c != background)
    foreground = input_grid == source
    assert bool(np.any(foreground)) and bool(np.any(~foreground))
    assert bool(np.all(output_grid[~foreground] == background))
    assert bool(np.all(output_grid[foreground] != background))
    assert bool(np.all(output_grid[foreground] != source))
    covers = _find_exact_covers(foreground, limit=2, masks=masks)
    assert len(covers) == 1
    cover = covers[0]
    mapping = {}
    for kind, _, _, cells in cover:
        tile_colors = {int(output_grid[row, col]) for row, col in cells}
        assert len(tile_colors) == 1
        color = next(iter(tile_colors))
        if kind in mapping:
            assert mapping[kind] == color
        mapping[kind] = color
    assert len(set(mapping.values())) == len(mapping)
    return {"background": background, "source": source, "mapping": mapping,
            "cover": cover, "counts": _tile_counts(cover),
            "contact_edges": _tile_contact_edges(cover),
            "components": _foreground_components(foreground),
            "foreground": foreground}


def _analyze_puzzle(puzzle: dict) -> tuple:
    for symmetry in range(8):
        masks = _oriented_masks(symmetry)
        try:
            analyses = {split: [_analyze_pair(p["input"], p["output"], masks)
                                for p in puzzle[split]] for split in ("train", "test")}
            mapping = {}
            for analysis in analyses["train"] + analyses["test"]:
                for kind, color in analysis["mapping"].items():
                    if kind in mapping:
                        assert mapping[kind] == color
                    mapping[kind] = color
            assert len(set(mapping.values())) == len(mapping)
            seen_train = set().union(*(set(a["mapping"]) for a in analyses["train"]))
            seen_test = set().union(*(set(a["mapping"]) for a in analyses["test"]))
            assert seen_test <= seen_train
            assert set(_TILE_ORDER) <= seen_train
            all_inputs = {a["background"] for split in analyses.values() for a in split}
            all_inputs.update(a["source"] for split in analyses.values() for a in split)
            assert not (all_inputs & set(mapping.values()))
            return symmetry, analyses, mapping
        except AssertionError:
            continue
    raise AssertionError("no consistent, uniquely recoverable tile convention")


def _check_train_only_predictions(puzzle: dict) -> None:
    """Enumerate geometry/color rules consistent with training, then predict tests."""
    backgrounds = set()
    sources = set()
    for pair in puzzle["train"]:
        values = [int(c) for c in np.unique(pair["input"])]
        assert len(values) == 2
        unchanged = [c for c in values if bool(np.all(
            pair["output"][pair["input"] == c] == c
        ))]
        assert len(unchanged) == 1
        backgrounds.add(unchanged[0])
        sources.add(next(c for c in values if c != unchanged[0]))
    assert not (backgrounds & sources)

    predictions = [set() for _ in puzzle["test"]]
    for symmetry in range(8):
        masks = _oriented_masks(symmetry)
        states = [dict()]
        for pair in puzzle["train"]:
            bg = next(c for c in np.unique(pair["input"]) if int(c) in backgrounds)
            foreground = pair["input"] != bg
            covers = _find_exact_covers(
                foreground, limit=65, masks=masks, output_grid=pair["output"]
            )
            assert len(covers) < 65
            new_states = []
            for cover in covers:
                local = {}
                for kind, _, _, cells in cover:
                    color = int(pair["output"][cells[0]])
                    if kind in local and local[kind] != color:
                        break
                    local[kind] = color
                else:
                    if len(set(local.values())) != len(local):
                        continue
                    for state in states:
                        if any(kind in state and state[kind] != color
                               for kind, color in local.items()):
                            continue
                        merged = dict(state)
                        merged.update(local)
                        if len(set(merged.values())) == len(merged):
                            new_states.append(merged)
            states = [dict(items) for items in {
                tuple(sorted(state.items())) for state in new_states
            }]
            if not states:
                break
        for state in states:
            if set(state.values()) & (backgrounds | sources):
                continue
            for test_index, pair in enumerate(puzzle["test"]):
                values = [int(c) for c in np.unique(pair["input"])]
                possible_bg = [c for c in values if c in backgrounds]
                assert len(possible_bg) == 1
                bg = possible_bg[0]
                source = next(c for c in values if c != bg)
                assert source in sources
                covers = _find_exact_covers(
                    pair["input"] != bg, limit=65, masks=masks
                )
                assert len(covers) < 65
                for cover in covers:
                    if any(kind not in state for kind, _, _, _ in cover):
                        continue
                    output = np.full(pair["input"].shape, bg, dtype=np.int8)
                    for kind, _, _, cells in cover:
                        for row, col in cells:
                            output[row, col] = state[kind]
                    predictions[test_index].add(_grid_value_key(output))
    for pair, possible in zip(puzzle["test"], predictions):
        assert possible == {_grid_value_key(pair["output"])}


def _choose_counts(rng: np.random.Generator, num_train, num_test) -> tuple:
    train = None if num_train is None else _validate_requested_count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _validate_requested_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    colors = [int(c) for c in rng.permutation(10)]
    tile_colors = dict(zip(_TILE_ORDER, colors[:4]))
    background_pool = colors[4:7]
    source_pool = colors[7:10]
    symmetry = int(rng.integers(8))
    palette_mode = int(rng.integers(4))
    vary_background = bool(palette_mode & 1)
    vary_source = bool(palette_mode & 2)
    train_roles = []
    for i in range(train_count):
        bg = background_pool[i % 3] if vary_background else background_pool[0]
        src = source_pool[i % 3] if vary_source else source_pool[0]
        train_roles.append((bg, src))
    test_roles = [(train_roles[int(rng.integers(train_count))][0],
                   train_roles[int(rng.integers(train_count))][1])
                  for _ in range(test_count)]
    inventories = []
    if float(rng.random()) < 0.65:
        inventories.append(["square"] * int(rng.integers(1, 3)) + list(_TILE_ORDER[1:]))
    else:
        inventories.append(list(_TILE_ORDER[1:]))
    for _ in range(1, train_count):
        inventories.append(_inventory(rng))
    if "square" not in inventories[0]:
        inventories[1].append("square")
    test_inventories = [_inventory(rng, minimum=2) for _ in range(test_count)]
    modes = ("compact", "chain_h", "chain_v", "separated", "diagonal", "mixed", "light")
    seen_inputs = set()
    puzzle = {"train": [], "test": []}
    for split, split_inventories, roles in (("train", inventories, train_roles),
                                            ("test", test_inventories, test_roles)):
        for kinds, (background, source) in zip(split_inventories, roles):
            for _ in range(20):
                mode = modes[int(rng.integers(len(modes)))]
                pair = _build_pair(rng, kinds, mode, background, source, tile_colors, symmetry)
                key = _grid_value_key(pair["input"])
                if key not in seen_inputs:
                    seen_inputs.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert pair["input"].shape == pair["output"].shape
            assert not np.array_equal(pair["input"], pair["output"])
            key = _grid_value_key(pair["input"])
            pair_key = (key, _grid_value_key(pair["output"]))
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIRS
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
    _analyze_puzzle(puzzle)
    _check_train_only_predictions(puzzle)


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_value_key(p["input"]), _grid_value_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_fingerprint(p) for p in automatic}) >= 40
    assert _puzzle_fingerprint(generate_puzzle(8675309, 7, 3)) == _puzzle_fingerprint(generate_puzzle(8675309, 7, 3))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                       (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(4000 + i, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(6000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid count request")
    inventories = set()
    layouts = set()
    dimensions = set()
    contacts = set()
    components = set()
    no_square = 0
    repeated_other_l = 0
    repeated_lower_left = 0
    repeated_upper_right = 0
    repeated_multiple = 0
    several_squares = 0
    separated = 0
    diagonal_only = 0
    gapped = 0
    long_chains = 0
    tight_crops = 0
    wide_extents = 0
    tall_extents = 0
    changed_background = 0
    changed_source = 0
    orientations = set()
    train_inventory_varied = 0
    sparse_test = 0
    dense_train = 0
    dense_train_sparse_test = 0
    for puzzle in automatic:
        symmetry, analyses, mapping = _analyze_puzzle(puzzle)
        orientations.add(symmetry)
        assert set(mapping) == set(_TILE_ORDER)
        train_counts = set()
        train_backgrounds = set()
        train_sources = set()
        train_has_dense = False
        test_has_sparse = False
        for split in ("train", "test"):
            for pair, analysis in zip(puzzle[split], analyses[split]):
                counts = tuple(analysis["counts"][kind] for kind in _TILE_ORDER)
                count = sum(counts)
                inventories.add(counts)
                layouts.add(_trimmed_layout_key(analysis["foreground"]))
                dimensions.add(pair["input"].shape)
                contacts.add(analysis["contact_edges"])
                components.add(analysis["components"])
                no_square += counts[0] == 0
                several_squares += counts[0] >= 2
                repeated_lower_left += counts[2] >= 2
                repeated_upper_right += counts[3] >= 2
                repeated_multiple += sum(value >= 2 for value in counts) >= 2
                repeated_other_l += counts[2] >= 2 or counts[3] >= 2
                coordinates = np.argwhere(analysis["foreground"])
                bbox_height, bbox_width = (
                    coordinates.max(axis=0) - coordinates.min(axis=0) + 1
                )
                wide_extents += bbox_width >= 2 * bbox_height
                tall_extents += bbox_height >= 2 * bbox_width
                tight_crops += pair["input"].size <= 1.5 * bbox_height * bbox_width
                long_chains += (count >= 4 and analysis["components"] == 1
                                and max(bbox_height, bbox_width) >= 2 * min(bbox_height, bbox_width))
                owner = {cell: tile_index for tile_index, (_, _, _, cells)
                         in enumerate(analysis["cover"]) for cell in cells}
                diagonal = any((row + dr, col + dc) in owner
                               and owner[row + dr, col + dc] != tile_index
                               for (row, col), tile_index in owner.items()
                               for dr, dc in ((1, 1), (1, -1)))
                separated += count >= 2 and analysis["components"] == count
                diagonal_only += count >= 2 and analysis["contact_edges"] == 0 and diagonal
                gapped += count >= 2 and analysis["components"] == count and not diagonal
                if split == "train":
                    train_counts.add(counts)
                    train_backgrounds.add(analysis["background"])
                    train_sources.add(analysis["source"])
                    dense_train += analysis["components"] == 1 and analysis["contact_edges"] >= 5
                    train_has_dense |= analysis["components"] == 1 and analysis["contact_edges"] >= 5
                else:
                    sparse_test += analysis["components"] > 1
                    test_has_sparse |= analysis["components"] > 1
        train_inventory_varied += len(train_counts) > 1
        changed_background += len(train_backgrounds) > 1
        changed_source += len(train_sources) > 1
        dense_train_sparse_test += train_has_dense and test_has_sparse
    assert len(inventories) >= 35 and len(layouts) >= 100 and len(dimensions) >= 35
    assert len(contacts) >= 8 and len(components) >= 4
    assert no_square >= 10 and repeated_other_l >= 10
    assert train_inventory_varied >= 35 and sparse_test >= 10 and dense_train >= 10
    assert several_squares >= 10 and repeated_lower_left >= 10 and repeated_upper_right >= 10
    assert repeated_multiple >= 10 and separated >= 10 and diagonal_only >= 5 and gapped >= 5
    assert long_chains >= 5 and tight_crops >= 5 and wide_extents >= 5 and tall_extents >= 5
    assert changed_background >= 5 and changed_source >= 5
    assert dense_train_sparse_test >= 5 and len(orientations) == 4
