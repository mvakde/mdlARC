"""Generate count-and-crop puzzles from the reviewed ARC tile family."""

import numpy as np


_SOURCE_RAW = (
    (("000088800", "888088800", "888018800", "888000000",
      "000000818", "818000118", "881000181", "188000000",
      "000000000"), ("818", "118", "181")),
    (("088100000", "081808180", "088801880", "000008810",
      "008180000", "001180000", "008810888", "000000888",
      "000000188"), ("818", "118", "881")),
    (("001180000", "008810811", "001180111", "000000818",
      "888000000", "881081800", "188018800", "000088100",
      "000000000"), ("811", "111", "818")),
    (("888000000", "188081800", "888011800", "000088800",
      "088100000", "088800818", "081800181", "000000181",
      "000000000"), ("818", "181", "181")),
    (("888000818", "888000181", "188000818", "000818000",
      "000881000", "000188000", "000000118", "000000811",
      "000000118"), ("118", "811", "118")),
)

def _rows_to_array(rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = tuple(
    (_rows_to_array(source), _rows_to_array(answer))
    for source, answer in _SOURCE_RAW
)
_BASELINE = (9, 9, 3, 3, 4)
_OTHER_SIZES = ((2, 3), (3, 2), (3, 4), (4, 3), (4, 4))


def _validated_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _choose_counts(rng, num_train, num_test):
    num_train = _validated_count(num_train, "num_train", 2, 10)
    num_test = _validated_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _new_geometry(rng, excluded=()):
    for _ in range(80):
        tile_h, tile_w = _OTHER_SIZES[int(rng.integers(len(_OTHER_SIZES)))]
        count = int(rng.integers(3, 7))
        side = int(np.ceil(np.sqrt(2.9 * count * tile_h * tile_w)))
        height = max(8, side + int(rng.integers(-2, 3)))
        width = max(8, side + int(rng.integers(-2, 3)))
        geometry = (height, width, tile_h, tile_w, count)
        if height * width > 2 * count * tile_h * tile_w and geometry not in excluded:
            return geometry
    raise RuntimeError("could not sample tile geometry")


def _geometry_plan(rng):
    mode = int(rng.integers(10))
    if mode < 3:
        return (_BASELINE,)
    if mode < 8:
        return (_BASELINE, _new_geometry(rng))
    first = _new_geometry(rng)
    return (first, _new_geometry(rng, (first,)))


def _compatible(first, second, tile_h, tile_w):
    r1, c1 = first
    r2, c2 = second
    rows_overlap = max(r1, r2) < min(r1 + tile_h, r2 + tile_h)
    cols_overlap = max(c1, c2) < min(c1 + tile_w, c2 + tile_w)
    if rows_overlap and cols_overlap:
        return False
    if cols_overlap and (r1 + tile_h == r2 or r2 + tile_h == r1):
        return False
    if rows_overlap and (c1 + tile_w == c2 or c2 + tile_w == c1):
        return False
    return True


def _sample_positions(rng, geometry, required=None):
    height, width, tile_h, tile_w, count = geometry
    positions = tuple(
        (r, c)
        for r in range(height - tile_h + 1)
        for c in range(width - tile_w + 1)
    )
    for _ in range(300):
        selected = [] if required is None else [required]
        for index in rng.permutation(len(positions)):
            position = positions[int(index)]
            if position in selected:
                continue
            if all(_compatible(position, other, tile_h, tile_w)
                   for other in selected):
                selected.append(position)
                if len(selected) == count:
                    return tuple(selected)
    raise RuntimeError("could not place separated tiles")


def _sample_winner_count(rng, area, excluded=()):
    choices = [n for n in range(2, area + 1) if n not in excluded]
    if not choices:
        choices = list(range(2, area + 1))
    return int(choices[int(rng.integers(len(choices)))])


def _sample_accent_counts(rng, winner_count, tile_count, area):
    for _ in range(100):
        regime = int(rng.integers(3))
        if regime == 0:
            runner_up = winner_count - 1
        elif regime == 1:
            runner_up = max(0, winner_count - 2)
        else:
            runner_up = int(rng.integers(max(0, winner_count - 5),
                                         winner_count))
        losers = [runner_up]
        for _ in range(tile_count - 2):
            if rng.random() < 0.28:
                losers.append(runner_up)
            else:
                losers.append(int(rng.integers(0, runner_up + 1)))
        if winner_count + sum(losers) < tile_count * area / 2:
            rng.shuffle(losers)
            return losers
    return [0] * (tile_count - 1)


def _motif_order(rng, tile_h, tile_w, mode):
    area = tile_h * tile_w
    if mode == 0:
        return [int(i) for i in rng.permutation(area)]
    if mode == 1:
        # A full row or column is filled before the next, with a random
        # orientation. Counts, not motif identity, decide the winner.
        order = np.arange(area).reshape(tile_h, tile_w)
        if rng.random() < 0.5:
            order = order.T
        if rng.random() < 0.5:
            order = order[::-1]
        if rng.random() < 0.5:
            order = order[:, ::-1]
        return [int(i) for i in order.flat]
    if mode == 2:
        start = int(rng.integers(area))
        chosen = [start]
        remaining = set(range(area)) - {start}
        while remaining:
            frontier = [
                cell for cell in remaining
                if any(abs(cell // tile_w - old // tile_w)
                       + abs(cell % tile_w - old % tile_w) == 1
                       for old in chosen)
            ]
            next_cell = int(frontier[int(rng.integers(len(frontier)))])
            chosen.append(next_cell)
            remaining.remove(next_cell)
        return chosen
    raise ValueError("unknown motif mode")


def _symmetric_indices(rng, tile_h, tile_w, accent_count):
    vertical = bool(rng.integers(2))
    orbits = []
    seen = set()
    for r in range(tile_h):
        for c in range(tile_w):
            first = r * tile_w + c
            partner = (r * tile_w + tile_w - 1 - c) if vertical else (
                (tile_h - 1 - r) * tile_w + c
            )
            if first not in seen:
                orbit = {first, partner}
                seen.update(orbit)
                orbits.append(tuple(orbit))
    singles = [orbit for orbit in orbits if len(orbit) == 1]
    doubles = [orbit for orbit in orbits if len(orbit) == 2]
    possible_single_counts = [
        count for count in range(len(singles) + 1)
        if count <= accent_count
        and (accent_count - count) % 2 == 0
        and (accent_count - count) // 2 <= len(doubles)
    ]
    exact = bool(possible_single_counts)
    target = accent_count if exact else accent_count - 1
    if exact:
        single_count = possible_single_counts[
            int(rng.integers(len(possible_single_counts)))
        ]
    else:
        single_count = 0
    double_count = (target - single_count) // 2
    selected = [
        cell for index in rng.permutation(len(singles))[:single_count]
        for cell in singles[int(index)]
    ]
    selected.extend(
        cell for index in rng.permutation(len(doubles))[:double_count]
        for cell in doubles[int(index)]
    )
    if not exact:
        remaining = [index for index in range(tile_h * tile_w)
                     if index not in selected]
        selected.append(remaining[int(rng.integers(len(remaining)))])
    return selected


def _sample_tile(rng, tile_h, tile_w, accent_count, colors):
    background, dominant, accent = colors
    del background
    tile = np.full((tile_h, tile_w), dominant, dtype=np.int8)
    mode = int(rng.integers(4))
    order = (_symmetric_indices(rng, tile_h, tile_w, accent_count)
             if mode == 3 else
             _motif_order(rng, tile_h, tile_w, mode)[:accent_count])
    tile.reshape(-1)[order] = accent
    return tile


def _generate_pair(rng, geometry, colors, winner_count, regime,
                   previous_positions=()):
    height, width, tile_h, tile_w, tile_count = geometry
    bottom_right = (height - tile_h, width - tile_w)
    for _ in range(120):
        # Test layouts encounter the canvas boundary more often, while
        # either split can put a winner or loser there.
        layout_regime = int(rng.integers(
            6 if regime == "source_train" else 4
        ))
        required = bottom_right if layout_regime in (0, 1) else None
        positions = _sample_positions(rng, geometry, required)
        if layout_regime == 0:
            winner = bottom_right
        elif layout_regime == 1:
            winner = positions[int(rng.integers(1, len(positions)))]
        else:
            winner = positions[int(rng.integers(len(positions)))]
        if winner not in previous_positions:
            break
    else:
        raise RuntimeError("could not vary winner position")
    losers = _sample_accent_counts(rng, winner_count, tile_count,
                                  tile_h * tile_w)
    grid = np.full((height, width), colors[0], dtype=np.int8)
    answer = None
    loser_index = 0
    for row, col in positions:
        if (row, col) == winner:
            accent_count = winner_count
        else:
            accent_count = losers[loser_index]
            loser_index += 1
        tile = _sample_tile(rng, tile_h, tile_w, accent_count, colors)
        grid[row:row + tile_h, col:col + tile_w] = tile
        if (row, col) == winner:
            answer = tile.copy()
    assert answer is not None
    return {"input": grid, "output": answer}


def _fingerprint(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _matches_source_pair(pair):
    return any(
        np.array_equal(pair["input"], source)
        and np.array_equal(pair["output"], answer)
        for source, answer in _SOURCE_PAIRS
    )


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _analyze_pair(pair):
    """Recover the entire answer from visible cells, without generation data."""
    grid, output = pair["input"], pair["output"]
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    assert int(np.count_nonzero(grid == background)) > grid.size / 2
    foreground = [int(color) for color in colors if color != background]
    frequencies = {color: int(np.count_nonzero(grid == color))
                   for color in foreground}
    assert frequencies[foreground[0]] != frequencies[foreground[1]]
    accent = min(foreground, key=lambda color: frequencies[color])
    dominant = max(foreground, key=lambda color: frequencies[color])
    components = _components(grid != background)
    assert 3 <= len(components) <= 6
    tiles = []
    positions = []
    shapes = []
    for component in components:
        rows, cols = zip(*component)
        r, c = min(rows), min(cols)
        h, w = max(rows) - r + 1, max(cols) - c + 1
        assert h * w == len(component)
        assert np.all(grid[r:r + h, c:c + w] != background)
        positions.append((r, c))
        shapes.append((h, w))
        tiles.append(grid[r:r + h, c:c + w])
    assert len(set(shapes)) == 1
    tile_h, tile_w = shapes[0]
    assert tile_h >= 2 and tile_w >= 2
    assert output.shape == (tile_h, tile_w)
    accent_counts = [int(np.count_nonzero(tile == accent)) for tile in tiles]
    maximum = max(accent_counts)
    assert accent_counts.count(maximum) == 1
    winner_index = accent_counts.index(maximum)
    assert np.array_equal(output, tiles[winner_index])
    assert 2 <= maximum <= tile_h * tile_w
    return {
        "colors": (background, dominant, accent),
        "geometry": (grid.shape[0], grid.shape[1], tile_h, tile_w,
                     len(tiles)),
        "positions": tuple(positions),
        "winner_position": positions[winner_index],
        "winner_count": maximum,
        "loser_counts": tuple(sorted((n for i, n in enumerate(accent_counts)
                                       if i != winner_index), reverse=True)),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    geometries = _geometry_plan(rng)
    variable_palette = bool(rng.integers(2))
    first_colors = tuple(int(x) for x in rng.choice(10, 3, replace=False))
    rotated_colors = (first_colors[1], first_colors[2], first_colors[0])
    train_geometry = [geometries[i % len(geometries)]
                      for i in range(num_train)]
    rng.shuffle(train_geometry)
    test_geometry = [
        (geometries[1] if _BASELINE in geometries
         and len(geometries) == 2 and rng.random() < 0.7
         else geometries[int(rng.integers(len(geometries)))])
        for _ in range(num_test)
    ]
    train_counts = []
    for index, geometry in enumerate(train_geometry):
        area = geometry[2] * geometry[3]
        excluded = (train_counts[0],) if index == 1 else ()
        train_counts.append(_sample_winner_count(rng, area, excluded))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_positions = set()
    test_positions = set()
    for split, geometry_list in (("train", train_geometry),
                                 ("test", test_geometry)):
        for index, geometry in enumerate(geometry_list):
            if split == "train":
                winner_count = train_counts[index]
            else:
                trained_max = max(
                    count for train_geom, count in zip(train_geometry,
                                                       train_counts)
                    if train_geom == geometry
                )
                winner_count = _sample_winner_count(
                    rng, min(geometry[2] * geometry[3], trained_max)
                )
            if not variable_palette:
                colors = first_colors
            elif split == "train" and index == 0:
                colors = first_colors
            elif split == "train" and index == 1:
                colors = rotated_colors
            else:
                colors = tuple(
                    int(x) for x in rng.choice(10, 3, replace=False)
                )
            previous = train_positions if split == "train" else test_positions
            pair_rng = np.random.default_rng(
                int(rng.integers(0, 2**63, dtype=np.int64))
            )
            for _ in range(100):
                pair = _generate_pair(
                    pair_rng, geometry, colors, winner_count,
                    "source_train" if split == "train" else "target_test",
                    previous,
                )
                fingerprint = _fingerprint(pair["input"])
                if fingerprint not in used_inputs and not _matches_source_pair(pair):
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
            info = _analyze_pair(pair)
            previous.add(info["winner_position"])
            used_inputs.add(fingerprint)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    info_by_split = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= length <= 30 for length in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _fingerprint(pair["input"])
            pair_key = (input_key, _fingerprint(pair["output"]))
            assert input_key not in inputs and pair_key not in pairs
            assert not _matches_source_pair(pair)
            inputs.add(input_key)
            pairs.add(pair_key)
            info_by_split[split].append(_analyze_pair(pair))
    train = info_by_split["train"]
    test = info_by_split["test"]
    train_geometries = {info["geometry"] for info in train}
    train_roles = {info["colors"] for info in train}
    assert len({info["winner_position"] for info in train}) >= 2
    assert len({info["winner_count"] for info in train}) >= 2
    assert len({info["winner_position"] for info in test}) == len(test)
    for info in test:
        geometry = info["geometry"]
        assert geometry in train_geometries
        assert info["winner_count"] <= max(
            item["winner_count"] for item in train
            if item["geometry"] == geometry
        )
        if info["colors"] not in train_roles:
            assert len(train_roles) >= 2


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(np.array_equal(a[role], b[role])
                for a, b in zip(first[split], second[split])
                for role in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    calls = ((11, 2, 1), (12, 10, 1), (13, 10, 4),
             (14, None, 1), (15, None, 4),
             (16, 2, None), (17, 10, None))
    for seed, train_count, test_count in calls:
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train_count,
                                                       test_count))
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {kwargs}")
    train_counts, test_counts = set(), set()
    geometries, roles, layouts, outputs = set(), set(), set(), set()
    puzzle_keys = set()
    margins, winner_counts, test_winner_counts = set(), set(), set()
    bottom_right_train_winners = 0
    bottom_right_test_losers = 0
    variable_role_puzzles = 0
    stable_role_puzzles = 0
    varied_geometry_puzzles = 0
    baseline_pairs = nonbaseline_pairs = 0
    tied_loser_pairs = sparse_loser_pairs = all_accent_pairs = 0
    combined_pairs = 0
    winner_regions = {"train": set(), "test": set()}
    motif_features = {"winner": set(), "loser": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(
            (_fingerprint(p["input"]), _fingerprint(p["output"]))
            for split in ("train", "test") for p in puzzle[split]
        ))
        train_roles = set()
        palette_varies = len({
            _analyze_pair(pair)["colors"] for pair in puzzle["train"]
        }) > 1
        train_geometries = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                geometries.add(info["geometry"])
                roles.add(info["colors"])
                layouts.add((info["geometry"],
                             tuple(sorted(info["positions"]))))
                outputs.add((pair["output"].shape,
                             tuple(int(x == info["colors"][2])
                                   for x in pair["output"].flat)))
                margins.add(info["winner_count"] - info["loser_counts"][0])
                winner_counts.add(info["winner_count"])
                if split == "test":
                    test_winner_counts.add(info["winner_count"])
                if len(set(info["loser_counts"])) < len(info["loser_counts"]):
                    tied_loser_pairs += 1
                if 0 in info["loser_counts"]:
                    sparse_loser_pairs += 1
                height, width, tile_h, tile_w, _ = info["geometry"]
                if info["winner_count"] == tile_h * tile_w:
                    all_accent_pairs += 1
                if info["geometry"] == _BASELINE:
                    baseline_pairs += 1
                else:
                    nonbaseline_pairs += 1
                row, col = info["winner_position"]
                on_row_edge = row in (0, height - tile_h)
                on_col_edge = col in (0, width - tile_w)
                winner_regions[split].add(
                    "corner" if on_row_edge and on_col_edge else
                    "edge" if on_row_edge or on_col_edge else "interior"
                )
                bottom_right = (
                    height - tile_h,
                    width - tile_w,
                )
                if split == "train" and info["winner_position"] == bottom_right:
                    bottom_right_train_winners += 1
                if (split == "test" and bottom_right in info["positions"]
                        and info["winner_position"] != bottom_right):
                    bottom_right_test_losers += 1
                for position in info["positions"]:
                    r, c = position
                    tile = pair["input"][r:r + tile_h, c:c + tile_w]
                    mask = tile == info["colors"][2]
                    status = ("winner" if position == info["winner_position"]
                              else "loser")
                    if np.array_equal(mask, mask[:, ::-1]) or np.array_equal(
                            mask, mask[::-1, :]):
                        motif_features[status].add("symmetric")
                    else:
                        motif_features[status].add("asymmetric")
                    if np.any(np.all(mask, axis=0)) or np.any(
                            np.all(mask, axis=1)):
                        motif_features[status].add("stripe")
                    if len(_components(mask)) == 1 and np.count_nonzero(mask) >= 2:
                        motif_features[status].add("cluster")
                if split == "train":
                    train_roles.add(info["colors"])
                    train_geometries.add(info["geometry"])
                if (palette_varies and info["geometry"] != _BASELINE
                        and info["winner_count"] - info["loser_counts"][0] >= 2):
                    combined_pairs += 1
        if len(train_roles) > 1:
            variable_role_puzzles += 1
        else:
            stable_role_puzzles += 1
        if len(train_geometries) > 1:
            varied_geometry_puzzles += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42
    assert len(geometries) >= 12 and len(layouts) >= 90
    assert len(roles) >= 25 and len(outputs) >= 50
    assert {1, 2}.issubset(margins) and max(margins) >= 3
    assert {4, 5, 6}.issubset(winner_counts) and max(winner_counts) > 6
    assert {4, 5}.issubset(test_winner_counts) and max(test_winner_counts) > 6
    assert bottom_right_train_winners > 0 and bottom_right_test_losers > 0
    assert variable_role_puzzles >= 8 and stable_role_puzzles >= 8
    assert varied_geometry_puzzles >= 8
    assert baseline_pairs > 0 and nonbaseline_pairs > 0
    assert tied_loser_pairs > 0 and sparse_loser_pairs > 0
    assert all_accent_pairs > 0 and combined_pairs > 0
    assert winner_regions["train"] == {"corner", "edge", "interior"}
    assert winner_regions["test"] == {"corner", "edge", "interior"}
    for status in ("winner", "loser"):
        assert {"symmetric", "asymmetric", "stripe", "cluster"}.issubset(
            motif_features[status]
        )
