from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


_SOURCE_PAIRS = (
    (("240420086", "440440088", "000000000", "110210055", "220210044", "000000000", "370830033", "330880037"),
     ("68", "88", "00", "45", "45", "00", "88", "83")),
    (("240580024", "440880024", "000000000", "550950033", "990950022", "000000000", "240440083", "240220083", "000000000", "110110099", "120210079"),
     ("58", "58", "00", "23", "23", "00", "33", "88", "00", "99", "97")),
    (("110110044", "210120034", "000000000", "220520033", "550520011", "000000000", "660680077", "880680044"),
     ("44", "43", "00", "13", "13", "00", "74", "74")),
)
_SOURCE_INPUTS = {pair[0] for pair in _SOURCE_PAIRS}
_PALETTES = tuple(itertools.combinations(range(10), 2))
_ROLES = tuple(itertools.permutations(range(3)))
_SHAPES = ((2, 2), (2, 3), (3, 3))


def _resolve_counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _patterns(height, width, checkerboard):
    masks = []

    def add(mask):
        mask = np.asarray(mask, dtype=np.int8)
        mask = mask ^ mask[0, 0]
        key = tuple(int(v) for v in mask.ravel())
        if len(set(key)) == 2 and key not in masks:
            masks.append(key)

    for row in (0, height - 1):
        for col in (0, width - 1):
            mask = np.zeros((height, width), dtype=np.int8)
            mask[row, col] = 1
            add(mask)
    for cut in range(1, height):
        add(np.indices((height, width))[0] >= cut)
    for cut in range(1, width):
        add(np.indices((height, width))[1] >= cut)
    if checkerboard:
        row, col = np.indices((height, width))
        add((row + col) % 2)
    if height == width == 3:
        row, col = np.indices((height, width))
        add(row == col)
        add(row + col == 2)
        add((row == 1) & (col == 1))
        add(row == 1)
        add(col == 1)
    return tuple(masks)


def _checker_mask(shape):
    row, col = np.indices(shape)
    return tuple(int(v) for v in ((row + col) % 2).ravel())


def _tile(mask, palette, flip, shape):
    bits = np.asarray(mask, dtype=np.int8).reshape(shape)
    first, second = palette if not flip else palette[::-1]
    return np.where(bits == 0, first, second).astype(np.int8)


def _palette_pair(rng, available, overlap, local_used=(), preferred_first=()):
    candidates = [(a, b) for a in available for b in available
                  if a != b and len(set(a) & set(b)) == overlap]
    if not candidates:
        candidates = [(a, b) for a in available for b in available if a != b]
    assert candidates
    preferred = [p for p in candidates if p[0] in preferred_first]
    if preferred:
        candidates = preferred
    elif local_used and rng.random() < 0.38:
        repeated = [p for p in candidates if p[0] in local_used or p[1] in local_used]
        if repeated:
            candidates = repeated
    return candidates[int(rng.integers(len(candidates)))]


def _mixed_palette_pair(rng, seen, fresh, overlap):
    candidates = [(a, b) for a in seen for b in fresh
                  if len(set(a) & set(b)) == overlap]
    if not candidates:
        candidates = [(a, b) for a in seen for b in fresh if a != b]
    assert candidates
    pair = candidates[int(rng.integers(len(candidates)))]
    return pair if rng.random() < 0.5 else pair[::-1]


def _overlap_plan(rng, count):
    kind = int(rng.integers(3))
    if kind < 2:
        return [kind] * count
    values = [int(rng.integers(2)) for _ in range(count)]
    if count >= 2:
        values[0], values[1] = 0, 1
        rng.shuffle(values)
    return values


def _draw_count(rng, tile_height, first_train=False):
    maximum = 31 // (tile_height + 1)
    low = 2 if first_train else 1
    return int(rng.integers(low, maximum + 1))


def _layout(rng, width, tile_width):
    spare = width - (3 * tile_width + 2)
    cuts = sorted(int(v) for v in rng.integers(0, spare + 1, size=3))
    left, gap1, gap2 = cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1]
    return (left, left + tile_width + 1 + gap1,
            left + 2 * tile_width + 2 + gap1 + gap2)


def _make_pair(rng, shape, background, record_count, roles, edges, palette_pairs,
               width_override=None):
    tile_height, tile_width = shape
    height = record_count * tile_height + record_count - 1
    width = (3 * tile_width + 2 + int(rng.integers(0, 11))
             if width_override is None else width_override)
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, tile_width), background, dtype=np.int8)
    for index in range(record_count):
        pivot_pattern, alternative_pattern = edges[index]
        pivot_palette, other_palette = palette_pairs[index]
        pivot = _tile(pivot_pattern, pivot_palette, int(rng.integers(2)), shape)
        alternative = _tile(alternative_pattern, pivot_palette,
                            int(rng.integers(2)), shape)
        analog = _tile(pivot_pattern, other_palette, int(rng.integers(2)), shape)
        mapping = {int(a): int(b) for a, b in zip(pivot.ravel(), analog.ravel())}
        assert len(mapping) == 2 and len(set(mapping.values())) == 2
        answer = np.array([mapping[int(v)] for v in alternative.ravel()],
                          dtype=np.int8).reshape(shape)
        slots = [None, None, None]
        pivot_slot, alternative_slot, analog_slot = roles[index]
        slots[pivot_slot], slots[alternative_slot], slots[analog_slot] = (
            pivot, alternative, analog)
        row = index * (tile_height + 1)
        for col, tile in zip(_layout(rng, width, tile_width), slots):
            input_grid[row:row + tile_height, col:col + tile_width] = tile
        output_grid[row:row + tile_height] = answer
    return {"input": input_grid, "output": output_grid}


def _grid_signature(grid):
    return grid.shape, grid.tobytes()


def _rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _normalize(tile):
    values = set(int(v) for v in tile.ravel())
    assert len(values) == 2
    first = int(tile[0, 0])
    return tuple(int(v != first) for v in tile.ravel())


def _parse_pair(pair, shape):
    input_grid, output_grid = pair["input"], pair["output"]
    tile_height, tile_width = shape
    height, width = input_grid.shape
    assert output_grid.shape == (height, tile_width)
    assert (height + 1) % (tile_height + 1) == 0
    count = (height + 1) // (tile_height + 1)
    assert 1 <= count <= 31 // (tile_height + 1)
    assert width >= 3 * tile_width + 2
    interpretations = []
    for background in range(10):
        try:
            records = []
            for index in range(count):
                row = index * (tile_height + 1)
                if index < count - 1:
                    assert np.all(input_grid[row + tile_height] == background)
                    assert np.all(output_grid[row + tile_height] == background)
                block = input_grid[row:row + tile_height]
                occupied = np.any(block != background, axis=0)
                starts = [col for col in range(width)
                          if occupied[col] and (col == 0 or not occupied[col - 1])]
                assert len(starts) == 3
                assert all(start + tile_width <= width for start in starts)
                expected_occupied = np.zeros(width, dtype=bool)
                for start in starts:
                    expected_occupied[start:start + tile_width] = True
                assert np.array_equal(occupied, expected_occupied)
                assert np.all(block[:, ~occupied] == background)
                tiles = [block[:, start:start + tile_width] for start in starts]
                assert all(np.all(tile != background) for tile in tiles)
                palettes = [frozenset(int(v) for v in tile.ravel()) for tile in tiles]
                assert all(len(palette) == 2 for palette in palettes)
                patterns = [_normalize(tile) for tile in tiles]
                candidates = []
                for pivot in range(3):
                    same_palette = [j for j in range(3) if j != pivot and palettes[j] == palettes[pivot]]
                    same_pattern = [j for j in range(3) if j != pivot and patterns[j] == patterns[pivot]]
                    if len(same_palette) == len(same_pattern) == 1 and same_palette[0] != same_pattern[0]:
                        candidates.append((pivot, same_palette[0], same_pattern[0]))
                assert len(candidates) == 1
                pivot, alternative, analog = candidates[0]
                mapping = {}
                for source, target in zip(tiles[pivot].ravel(), tiles[analog].ravel()):
                    source, target = int(source), int(target)
                    assert source not in mapping or mapping[source] == target
                    mapping[source] = target
                assert len(mapping) == len(set(mapping.values())) == 2
                answer = np.array([mapping[int(v)] for v in tiles[alternative].ravel()],
                                  dtype=np.int8).reshape(shape)
                assert np.array_equal(answer, output_grid[row:row + tile_height])
                records.append({"roles": (pivot, alternative, analog),
                                "patterns": (patterns[pivot], patterns[alternative]),
                                "palettes": (tuple(sorted(palettes[pivot])),
                                             tuple(sorted(palettes[analog]))),
                                "overlap": len(palettes[pivot] & palettes[analog]),
                                "columns": tuple(starts)})
            interpretations.append((background, records))
        except AssertionError:
            continue
    assert len(interpretations) == 1
    background, records = interpretations[0]
    return {"background": background, "records": records,
            "shape": shape, "input_width": width}


def _valid_shapes(puzzle):
    shapes = []
    for shape in _SHAPES:
        try:
            for split in ("train", "test"):
                for pair in puzzle[split]:
                    _parse_pair(pair, shape)
            shapes.append(shape)
        except AssertionError:
            pass
    assert len(shapes) == 1
    return shapes[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    try:
        rng = np.random.default_rng(int(seed))
    except ValueError as error:
        raise ValueError("seed must be a nonnegative integer") from error
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    shape = _SHAPES[int(rng.choice(3, p=(0.72, 0.16, 0.12)))]
    source_style = shape == (2, 2) and rng.random() < 0.28
    checkerboard = bool(rng.random() < 0.45)
    patterns = _patterns(*shape, checkerboard)
    all_edges = tuple(itertools.combinations(patterns, 2))
    novelty_mode = int(rng.integers(3))
    reserved = set()
    if novelty_mode:
        order = rng.permutation(len(_PALETTES))
        reserved = {_PALETTES[int(i)] for i in order[:12]}
    train_pool = tuple(p for p in _PALETTES if p not in reserved)
    fresh_pool = tuple(p for p in _PALETTES if p in reserved)
    fixed_role = _ROLES[int(rng.integers(6))] if rng.random() < 0.19 else None
    edge_mode = int(rng.integers(3))
    train_seen_palettes = set()
    train_seen_edges = set()
    train_seen_patterns = set()
    train_seen_roles = set()
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    root_nonce = int(rng.integers(0, 2**63 - 1))

    for split, total in (("train", train_count), ("test", test_count)):
        for pair_index in range(total):
            pair_rng = np.random.default_rng(np.random.SeedSequence(
                [root_nonce, 0 if split == "train" else 1, pair_index]))
            for attempt in range(30):
                count = _draw_count(pair_rng, shape[0], split == "train" and pair_index == 0)
                background = 0 if pair_rng.random() < 0.57 else int(pair_rng.integers(1, 10))
                use_source_style = source_style and split == "train" and pair_index == 0
                if use_source_style:
                    count = 3 + int(pair_rng.integers(2))
                    background = 0
                available_train = tuple(p for p in train_pool if background not in p)
                available_seen = tuple(p for p in train_seen_palettes if background not in p)
                available_fresh = tuple(p for p in fresh_pool if background not in p)
                roles = []
                edges = []
                palette_pairs = []
                local_palettes = set()
                overlaps = _overlap_plan(pair_rng, count)
                for record_index in range(count):
                    if split == "train":
                        role = fixed_role or _ROLES[int(pair_rng.integers(6))]
                        edge = all_edges[int(pair_rng.integers(len(all_edges)))]
                        if pair_index == record_index == 0 and checkerboard:
                            checker = _checker_mask(shape)
                            options = [e for e in all_edges if checker in e]
                            edge = options[int(pair_rng.integers(len(options)))]
                        palettes = _palette_pair(pair_rng, available_train,
                                                 overlaps[record_index], local_palettes)
                    else:
                        role_choices = tuple(sorted(train_seen_roles))
                        role = fixed_role or role_choices[int(pair_rng.integers(len(role_choices)))]
                        possible = list(itertools.combinations(sorted(train_seen_patterns), 2))
                        known = [e for e in possible if frozenset(e) in train_seen_edges]
                        novel = [e for e in possible if frozenset(e) not in train_seen_edges]
                        want_new = edge_mode == 1 or (edge_mode == 2 and pair_rng.random() < 0.5)
                        choices = (novel if want_new else known) or known or possible
                        edge = choices[int(pair_rng.integers(len(choices)))]
                        if novelty_mode == 0 or not available_fresh:
                            palettes = _palette_pair(pair_rng, available_seen,
                                                     overlaps[record_index], local_palettes)
                        elif novelty_mode == 1:
                            palettes = _palette_pair(pair_rng, available_fresh,
                                                     overlaps[record_index], local_palettes)
                        else:
                            palettes = _mixed_palette_pair(pair_rng, available_seen,
                                                           available_fresh, overlaps[record_index])
                    roles.append(role)
                    edges.append(edge)
                    palette_pairs.append(palettes)
                    local_palettes.update(palettes)
                pair = _make_pair(pair_rng, shape, background, count,
                                  roles, edges, palette_pairs,
                                  width_override=9 if use_source_style else None)
                signature = _grid_signature(pair["input"])
                if signature in seen_inputs or _rows(pair["input"]) in _SOURCE_INPUTS:
                    continue
                seen_inputs.add(signature)
                puzzle[split].append(pair)
                if split == "train":
                    train_seen_roles.update(roles)
                    train_seen_patterns.update(p for edge in edges for p in edge)
                    train_seen_edges.update(frozenset(edge) for edge in edges)
                    train_seen_palettes.update(p for item in palette_pairs for p in item)
                break
            else:
                raise AssertionError("unique pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            signature = _grid_signature(pair["input"])
            assert signature not in seen
            seen.add(signature)
            assert _rows(pair["input"]) not in _SOURCE_INPUTS
            assert (_rows(pair["input"]), _rows(pair["output"])) not in _SOURCE_PAIRS
    shape = _valid_shapes(puzzle)
    analyses = {split: [_parse_pair(pair, shape) for pair in puzzle[split]]
                for split in ("train", "test")}
    train_records = [record for analysis in analyses["train"] for record in analysis["records"]]
    test_records = [record for analysis in analyses["test"] for record in analysis["records"]]
    train_roles = {record["roles"] for record in train_records}
    train_patterns = {pattern for record in train_records for pattern in record["patterns"]}
    assert {record["roles"] for record in test_records} <= train_roles
    assert {pattern for record in test_records for pattern in record["patterns"]} <= train_patterns
    assert len(analyses["train"][0]["records"]) >= 2


def _puzzle_signature(puzzle):
    return tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_signature(generate_puzzle(17)) == _puzzle_signature(puzzles[17])
    signatures, shapes, counts, backgrounds = set(), set(), set(), set()
    overlap_kinds, role_orders = set(), set()
    repeated_patterns = repeated_palettes = 0
    reused_palette_tests = novel_palette_tests = 0
    reused_edge_tests = novel_edge_tests = 0
    review = Counter()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        shape = _valid_shapes(puzzle)
        shapes.add(shape)
        train_records, test_records = [], []
        analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                signatures.add(_grid_signature(pair["input"]))
                analysis = _parse_pair(pair, shape)
                analyses[split].append(analysis)
                backgrounds.add(analysis["background"])
                counts.add(len(analysis["records"]))
                records = analysis["records"]
                role_orders.update(record["roles"] for record in records)
                overlap_kinds.add(tuple(sorted({r["overlap"] for r in records})))
                repeated_patterns += len({frozenset(r["patterns"]) for r in records}) < len(records)
                repeated_palettes += len({p for r in records for p in r["palettes"]}) < 2 * len(records)
                review["variable_columns_in_pair"] += len({r["columns"] for r in records}) > 1
                review["repeated_role_in_pair"] += len({r["roles"] for r in records}) < len(records)
                review["omitted_2x2_patterns"] += shape == (2, 2) and len(
                    {pattern for r in records for pattern in r["patterns"]}) < 6
                review["combined_variation"] += (
                    len({frozenset(r["patterns"]) for r in records}) < len(records)
                    and len({palette for r in records for palette in r["palettes"]}) < 2 * len(records)
                    and {r["overlap"] for r in records} == {0, 1})
                review["source_like"] += (shape == (2, 2) and len(records) in (3, 4)
                                          and analysis["input_width"] == 9
                                          and analysis["background"] == 0)
                (train_records if split == "train" else test_records).extend(records)
        train_palettes = {p for r in train_records for p in r["palettes"]}
        train_edges = {frozenset(r["patterns"]) for r in train_records}
        reused_palette_tests += any(p in train_palettes for r in test_records for p in r["palettes"])
        novel_palette_tests += any(p not in train_palettes for r in test_records for p in r["palettes"])
        reused_edge_tests += any(frozenset(r["patterns"]) in train_edges for r in test_records)
        novel_edge_tests += any(frozenset(r["patterns"]) not in train_edges for r in test_records)
        review["all_one_role_puzzles"] += len({r["roles"] for r in train_records + test_records}) == 1
        review["nonfirst_test_pivot"] += any(r["roles"][0] != 0 for r in test_records)
        review["checkerboard_train"] += _checker_mask(shape) in {
            pattern for r in train_records for pattern in r["patterns"]}
        review["checkerboard_test"] += _checker_mask(shape) in {
            pattern for r in test_records for pattern in r["patterns"]}
        review["background_change_in_puzzle"] += len({
            a["background"] for split in analyses for a in analyses[split]}) > 1
        train_counts = {len(a["records"]) for a in analyses["train"]}
        test_counts = {len(a["records"]) for a in analyses["test"]}
        review["test_longer"] += max(test_counts) > min(train_counts)
        review["train_longer"] += max(train_counts) > min(test_counts)
        review["same_record_count"] += bool(train_counts & test_counts)
        test_palettes = {p for r in test_records for p in r["palettes"]}
        test_edges = {frozenset(r["patterns"]) for r in test_records}
        review[f"palette_relation_{bool(train_palettes & test_palettes)}_{bool(test_palettes - train_palettes)}"] += 1
        review[f"edge_relation_{bool(train_edges & test_edges)}_{bool(test_edges - train_edges)}"] += 1
    assert len(signatures) >= 200
    assert shapes == set(_SHAPES)
    assert {1, 2, 3, 4, 5, 6} <= counts
    assert len(backgrounds) >= 7
    assert len(role_orders) == 6
    assert {(0,), (1,), (0, 1)} <= overlap_kinds
    assert repeated_patterns > 0 and repeated_palettes > 0
    assert min(reused_palette_tests, novel_palette_tests,
               reused_edge_tests, novel_edge_tests) > 0
    assert all(review[key] > 0 for key in (
        "variable_columns_in_pair", "repeated_role_in_pair", "omitted_2x2_patterns",
        "combined_variation", "source_like", "all_one_role_puzzles",
        "nonfirst_test_pivot", "checkerboard_train", "checkerboard_test",
        "background_change_in_puzzle", "test_longer", "train_longer",
        "same_record_count", "palette_relation_True_False",
        "palette_relation_False_True", "palette_relation_True_True",
        "edge_relation_True_False", "edge_relation_False_True",
        "edge_relation_True_True"))
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + i, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for i, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                {"num_train": 2}, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(3000 + i, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
