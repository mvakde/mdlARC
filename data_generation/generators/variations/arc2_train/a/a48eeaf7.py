import numpy as np


_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_CARDINAL = frozenset(("N", "E", "S", "W"))
_DIAGONAL = frozenset(("NE", "SE", "SW", "NW"))

_SOURCE_ROWS = (
    (
        (
            "0000000050",
            "0000000000",
            "0000022000",
            "0500022000",
            "0000000000",
            "0000000000",
            "0000000005",
            "0000000000",
            "0000050000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000500",
            "0000022000",
            "0000522000",
            "0000050500",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0005000000",
            "0000000000",
            "0000000000",
            "0002200050",
            "0002200000",
            "0000000000",
            "0000000000",
            "0000000500",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0005000000",
            "0002250000",
            "0002200000",
            "0000050000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0050000000",
            "0000000050",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0022000500",
            "0022000000",
            "0000000000",
            "0050000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0050500000",
            "0022500000",
            "0022000000",
            "0050000000",
            "0000000000",
        ),
    ),
)


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int16)
    return tuple(normalized.shape), normalized.tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _configuration_key(pair: dict, background: int) -> tuple:
    occupied = (pair["input"] != background).astype(np.int8)
    return _grid_key(occupied)


def _source_pair_keys() -> set:
    keys = set()
    for input_rows, output_rows in _SOURCE_ROWS:
        inp = np.array(
            [[int(ch) for ch in row] for row in input_rows],
            dtype=np.int8,
        )
        out = np.array(
            [[int(ch) for ch in row] for row in output_rows],
            dtype=np.int8,
        )
        keys.add((_grid_key(inp), _grid_key(out)))
    return keys


def _coerce_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _coerce_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _coerce_optional_count(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        # Small source-like puzzles occur often, with longer teaching
        # sequences available for denser and multicolor configurations.
        schedule = (2, 3, 4, 5, 2, 6, 7, 3, 8, 9, 10)
        num_train = schedule[seed % len(schedule)]
        num_test = 1 + (seed // len(schedule) + seed) % min(4, num_train - 1)
    elif num_train is None:
        first_legal_train = max(2, num_test + 1)
        number_of_choices = 11 - first_legal_train
        num_train = (
            first_legal_train
            + seed % number_of_choices
        )
    elif num_test is None:
        largest_legal_test = min(4, num_train - 1)
        num_test = 1 + seed % largest_legal_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _relative_geometry(
    direction: str,
    distance: int,
    track: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Coordinates are relative to a block occupying rows 0..1 and
    columns 0..1. Distance is the number of marker moves needed to
    reach its destination.
    """
    if direction == "N":
        return (-1 - distance, track), (-1, track)
    if direction == "S":
        return (2 + distance, track), (2, track)
    if direction == "W":
        return (track, -1 - distance), (track, -1)
    if direction == "E":
        return (track, 2 + distance), (track, 2)
    if direction == "NW":
        return (
            (-1 - distance, -1 - distance),
            (-1, -1),
        )
    if direction == "NE":
        return (
            (-1 - distance, 2 + distance),
            (-1, 2),
        )
    if direction == "SW":
        return (
            (2 + distance, -1 - distance),
            (2, -1),
        )
    if direction == "SE":
        return (
            (2 + distance, 2 + distance),
            (2, 2),
        )
    raise AssertionError("unknown direction")


_TRACKS = tuple(f"{direction}{track}" for direction in ("N", "E", "S", "W") for track in (0, 1)) + ("NE", "SE", "SW", "NW")


def _track_parts(name: str) -> tuple[str, int]:
    return (name[:-1], int(name[-1])) if name[-1].isdigit() else (name, 0)


def _inward_geometry(name: str, distance: int, block: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    direction, track = _track_parts(name)
    source, destination = _relative_geometry(direction, distance, track)
    return ((source[0] + block[0], source[1] + block[1]),
            (destination[0] + block[0], destination[1] + block[1]))


def _isolated(cells: list[tuple[int, int]], block: tuple[int, int]) -> bool:
    block_cells = [(block[0] + r, block[1] + c) for r in (0, 1) for c in (0, 1)]
    for index, cell in enumerate(cells):
        if any(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) <= 1
               for other in block_cells + cells[index + 1:]):
            return False
    return True


def _make_pair(
    rng: np.random.Generator,
    names: tuple[str, ...],
    distances: tuple[int, ...],
    colors: tuple[int, int, tuple[int, ...]],
    canvas: str,
    edge: str | None,
    require_multicolor: bool = False,
) -> dict | None:
    relatives = []
    for name, distance in zip(names, distances):
        direction, track = _track_parts(name)
        relatives.append(_relative_geometry(direction, distance, track))
    sources = [item[0] for item in relatives]
    north = max(( -row for row, _ in sources if row < 0), default=0)
    south = max((row - 1 for row, _ in sources if row > 1), default=0)
    west = max((-column for _, column in sources if column < 0), default=0)
    east = max((column - 1 for _, column in sources if column > 1), default=0)
    requirements = [north, south, west, east]
    edge_index = {"N": 0, "S": 1, "W": 2, "E": 3}.get(edge)
    if edge_index is not None and requirements[edge_index] != 0:
        return None

    if canvas in ("source", "compact"):
        size = 10 if canvas == "source" else int(rng.integers(6, 10))
        if north + south + 2 > size or west + east + 2 > size:
            return None
        top = int(rng.integers(north, size - south - 1))
        left = int(rng.integers(west, size - east - 1))
        if edge == "N": top = 0
        if edge == "S": top = size - 2
        if edge == "W": left = 0
        if edge == "E": left = size - 2
        height = width = size
    else:
        pads = [int(rng.integers(0, 4)) for _ in range(4)]
        if edge_index is not None:
            pads[edge_index] = 0
        top, bottom, left, right = [need + pad for need, pad in zip(requirements, pads)]
        if top + bottom + 2 < 5:
            if edge == "S": top = 3 - bottom
            else: bottom = 3 - top
        if left + right + 2 < 5:
            if edge == "E": left = 3 - right
            else: right = 3 - left
        height, width = top + bottom + 2, left + right + 2
        if height > 28 or width > 28:
            return None

    block = (top, left)
    markers = [(top + row, left + column) for row, column in sources]
    destinations = [(top + item[1][0], left + item[1][1]) for item in relatives]
    if len(set(markers)) != len(markers) or len(set(destinations)) != len(destinations):
        return None
    if not _isolated(markers, block):
        return None
    if not all(0 <= row < height and 0 <= column < width for row, column in markers + destinations):
        return None
    background, block_color, marker_palette = colors
    inp = np.full((height, width), background, dtype=np.int8)
    inp[top:top + 2, left:left + 2] = block_color
    shades = [int(marker_palette[int(rng.integers(len(marker_palette)))]) for _ in names]
    if require_multicolor:
        if len(shades) < 2 or len(marker_palette) < 2:
            return None
        shades[:2] = marker_palette[:2]
    for cell, shade in zip(markers, shades):
        inp[cell] = shade
    out = inp.copy()
    for cell in markers:
        out[cell] = background
    for cell, shade in zip(destinations, shades):
        out[cell] = shade
    return {"input": inp, "output": out}


def _choose_tracks(rng: np.random.Generator, count: int, edge: str | None,
                   need_both: bool, parallel: bool, allow_parallel: bool) -> tuple[str, ...] | None:
    forbidden = {
        "N": {"N", "NW", "NE"}, "S": {"S", "SW", "SE"},
        "W": {"W", "NW", "SW"}, "E": {"E", "NE", "SE"},
    }.get(edge, set())
    pool = [name for name in _TRACKS if _track_parts(name)[0] not in forbidden]
    if not allow_parallel:
        directions = list(dict.fromkeys(_track_parts(name)[0] for name in pool))
        if count > len(directions):
            return None
        chosen_directions = rng.choice(directions, size=count, replace=False)
        names = tuple(str(rng.choice([name for name in pool if _track_parts(name)[0] == direction]))
                      for direction in chosen_directions)
    else:
        if count > len(pool):
            return None
        names = tuple(str(name) for name in rng.choice(pool, size=count, replace=False))
    directions = [_track_parts(name)[0] for name in names]
    if need_both and (not set(directions) & _CARDINAL or not set(directions) & _DIAGONAL):
        return None
    if parallel and len(set(directions)) == len(directions):
        return None
    return names


def _sample_colors(rng: np.random.Generator, mode: str,
                   fixed: tuple[int, int, int], count: int,
                   force_multicolor: bool = False,
                   avoid_roles: tuple[int, int, int] | None = None) -> tuple[int, int, tuple[int, ...]]:
    if mode == "fixed":
        return fixed[0], fixed[1], (fixed[2],)
    for _ in range(20):
        background, block, marker = (int(x) for x in rng.choice(10, size=3, replace=False))
        if avoid_roles is None or (background, block) != avoid_roles[:2]:
            break
    if avoid_roles is not None and (background, block) == avoid_roles[:2]:
        background, marker = marker, background
    if mode == "variable":
        return background, block, (marker,)
    available = [color for color in range(10) if color not in (background, block, marker)]
    number = int(rng.integers(2 if force_multicolor else 1, min(3, count) + 1))
    extras = tuple(int(x) for x in rng.choice(available, size=number - 1, replace=False))
    return background, block, (marker,) + extras


def _sample_pair(rng: np.random.Generator, count: int, maximum: int,
                 colors: tuple[int, int, tuple[int, ...]],
                 need_both: bool = False, parallel: bool = False,
                 allow_parallel: bool = True,
                 fixed_tracks: tuple[str, ...] | None = None,
                 require_multicolor: bool = False) -> dict:
    for attempt in range(240):
        if maximum <= 3 and count <= 5 and int(rng.integers(3)) == 0:
            canvas = "compact"
        elif maximum <= 5 and int(rng.integers(4)) == 0:
            canvas = "source"
        else:
            canvas = "free"
        edge = str(rng.choice(("N", "S", "W", "E"))) if int(rng.integers(4)) == 0 else None
        if fixed_tracks is not None:
            names = fixed_tracks
            if edge is not None:
                blocked = {"N": ("N", "NE", "NW"), "S": ("S", "SE", "SW"),
                           "E": ("E", "NE", "SE"), "W": ("W", "NW", "SW")}[edge]
                if any(_track_parts(name)[0] in blocked for name in names):
                    edge = None
        else:
            names = _choose_tracks(rng, count, edge, need_both, parallel, allow_parallel)
        if names is None:
            continue
        distances = [int(rng.integers(1, maximum + 1)) for _ in names]
        distances[int(rng.integers(len(names)))] = maximum
        pair = _make_pair(rng, names, tuple(distances), colors, canvas, edge,
                          require_multicolor)
        if pair is not None:
            return pair
    raise AssertionError("could not construct a valid pair")


def _classify_marker(row: int, column: int, block_row: int,
                     block_column: int) -> tuple[str, tuple[int, int], int]:
    for name in _TRACKS:
        direction, track = _track_parts(name)
        for distance in range(0, 30):
            source, destination = _relative_geometry(direction, distance, track)
            if (row, column) == (block_row + source[0], block_column + source[1]):
                return name, (block_row + destination[0], block_column + destination[1]), distance
    raise AssertionError("marker is off the permitted rays")


def _analyze_pair(inp: np.ndarray, out: np.ndarray) -> dict:
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    candidates = []
    for color in values:
        if int(color) == background:
            continue
        cells = np.argwhere(inp == color)
        if len(cells) == 4:
            rows, columns = np.unique(cells[:, 0]), np.unique(cells[:, 1])
            if len(rows) == len(columns) == 2 and rows[1] == rows[0] + 1 and columns[1] == columns[0] + 1:
                candidates.append((int(color), int(rows[0]), int(columns[0])))
    assert len(candidates) == 1
    block_color, top, left = candidates[0]
    marker_cells = [(int(row), int(column)) for row, column in np.argwhere((inp != background) & (inp != block_color))]
    assert 1 <= len(marker_cells) <= 8
    assert _isolated(marker_cells, (top, left))
    expected = inp.copy()
    tracks, distances, destinations = [], [], []
    for row, column in marker_cells:
        name, dest, distance = _classify_marker(row, column, top, left)
        assert distance >= 1
        assert 0 <= dest[0] < inp.shape[0] and 0 <= dest[1] < inp.shape[1]
        assert int(inp[dest]) == background
        tracks.append(name)
        distances.append(distance)
        destinations.append(dest)
        expected[row, column] = background
    assert len(set(tracks)) == len(tracks)
    assert len(set(destinations)) == len(destinations)
    for cell, dest in zip(marker_cells, destinations):
        expected[dest] = inp[cell]
    assert np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    return {
        "background": background, "block_color": block_color,
        "marker_colors": frozenset(int(inp[cell]) for cell in marker_cells),
        "block": (top, left), "tracks": frozenset(tracks),
        "directions": frozenset(_track_parts(name)[0] for name in tracks),
        "distances": tuple(distances), "marker_count": len(marker_cells),
        "destinations": frozenset(destinations),
        "edge": top == 0 or left == 0 or top + 2 == inp.shape[0] or left + 2 == inp.shape[1],
    }


def _distance_variant(pair: dict) -> dict | None:
    inp = pair["input"]
    info = _analyze_pair(inp, pair["output"])
    background, top, left = info["background"], *info["block"]
    markers = [(int(row), int(column)) for row, column in np.argwhere((inp != background) & (inp != info["block_color"]))]
    for index, cell in enumerate(markers):
        name, _, distance = _classify_marker(*cell, top, left)
        for new_distance in (distance + 1, distance - 1, distance + 2):
            if new_distance < 1:
                continue
            new_cell, _ = _inward_geometry(name, new_distance, (top, left))
            if not (0 <= new_cell[0] < inp.shape[0] and 0 <= new_cell[1] < inp.shape[1]):
                continue
            if inp[new_cell] != background:
                continue
            revised = markers.copy()
            revised[index] = new_cell
            if not _isolated(revised, (top, left)):
                continue
            new_input = inp.copy()
            new_input[cell] = background
            new_input[new_cell] = inp[cell]
            new_output = pair["output"].copy()
            return {"input": new_input, "output": new_output}
    return None


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    mode = ("fixed", "variable", "multicolor")[seed % 3]
    fixed = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    train, test = [], []
    used = set()
    source = _source_pair_keys()

    def append_unique(split: list, pair: dict) -> bool:
        key = _grid_key(pair["input"])
        if key in used or _pair_key(pair) in source:
            return False
        used.add(key)
        split.append(pair)
        return True

    # The first demonstration shows both straight and corner-diagonal motion.
    first_count = int(rng.integers(3, 7))
    first_max = int(rng.integers(3, 8))
    first_colors = _sample_colors(rng, mode, fixed, first_count, mode == "multicolor")
    append_unique(train, _sample_pair(rng, first_count, first_max, first_colors,
                                      need_both=True, require_multicolor=mode == "multicolor"))
    paired = num_train >= 3 and seed % 4 == 0
    if paired:
        variant = _distance_variant(train[0])
        if variant is not None:
            append_unique(train, variant)
    while len(train) < num_train:
        count = int(rng.integers(1, 9))
        maximum = int(rng.integers(2, 9))
        if count >= 7:
            maximum = max(maximum, 4)
        avoid = None
        existing_roles = {
            (info["background"], info["block_color"])
            for info in (_analyze_pair(pair["input"], pair["output"]) for pair in train)
        }
        if mode != "fixed" and len(existing_roles) == 1:
            first = _analyze_pair(train[0]["input"], train[0]["output"])
            avoid = (first["background"], first["block_color"], min(first["marker_colors"]))
        colors = _sample_colors(rng, mode, fixed, count, avoid_roles=avoid)
        pair = _sample_pair(rng, count, maximum, colors)
        append_unique(train, pair)

    train_info = [_analyze_pair(p["input"], p["output"]) for p in train]
    train_max = max(max(a["distances"]) for a in train_info)
    train_dense = max(a["marker_count"] for a in train_info)
    has_parallel = any(len(a["directions"]) < a["marker_count"] for a in train_info)
    train_multicolor = any(len(a["marker_colors"]) > 1 for a in train_info)
    train_configs = {_configuration_key(pair, info["background"])
                     for pair, info in zip(train, train_info)}
    for index in range(num_test):
        relation = (seed + index) % 3
        if relation == 0:
            maximum = max(1, train_max - int(rng.integers(1, min(4, train_max))))
        elif relation == 1:
            maximum = train_max
        else:
            maximum = min(12, train_max + int(rng.integers(1, 4)))
        if maximum == 1:
            count = int(rng.integers(1, min(4, train_dense) + 1))
        else:
            count = int(rng.integers(1, train_dense + 1))
        if maximum <= 2:
            count = min(count, 4)
        reuse = bool(int(rng.integers(3)) == 0)
        fixed_tracks = None
        if reuse:
            eligible = [tuple(sorted(a["tracks"])) for a in train_info
                        if a["marker_count"] == count and
                        (maximum >= 3 or len(a["directions"]) == a["marker_count"])]
            if eligible:
                fixed_tracks = eligible[int(rng.integers(len(eligible)))]
        colors = _sample_colors(rng, mode, fixed, count)
        if mode == "multicolor" and not train_multicolor:
            colors = (colors[0], colors[1], (colors[2][0],))
        for _ in range(12):
            pair = _sample_pair(rng, count, maximum, colors,
                                allow_parallel=has_parallel and maximum >= 3,
                                fixed_tracks=fixed_tracks)
            info = _analyze_pair(pair["input"], pair["output"])
            if _configuration_key(pair, info["background"]) not in train_configs and append_unique(test, pair):
                break
        else:
            raise AssertionError("could not construct a novel test configuration")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _source_pair_keys()
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    trains, tests = analyses["train"], analyses["test"]
    directions = frozenset().union(*(a["directions"] for a in trains))
    assert len(directions) >= 2 and directions & _CARDINAL and directions & _DIAGONAL
    train_dense = max(a["marker_count"] for a in trains)
    train_parallel = any(len(a["directions"]) < a["marker_count"] for a in trains)
    train_multicolor = any(len(a["marker_colors"]) > 1 for a in trains)
    train_role_pairs = {(a["background"], a["block_color"]) for a in trains}
    fixed_train_role = all((a["background"], a["block_color"], a["marker_colors"]) ==
                           (trains[0]["background"], trains[0]["block_color"], trains[0]["marker_colors"])
                           for a in trains) and len(trains[0]["marker_colors"]) == 1
    train_configs = {_configuration_key(pair, analysis["background"])
                     for pair, analysis in zip(puzzle["train"], trains)}
    for pair, analysis in zip(puzzle["test"], tests):
        assert analysis["marker_count"] <= train_dense
        assert _configuration_key(pair, analysis["background"]) not in train_configs
        if len(analysis["directions"]) < analysis["marker_count"]:
            assert train_parallel
        if len(analysis["marker_colors"]) > 1:
            assert train_multicolor
        if (analysis["background"], analysis["block_color"]) not in train_role_pairs:
            assert len(train_role_pairs) >= 2
        if fixed_train_role:
            assert (analysis["background"], analysis["block_color"], analysis["marker_colors"]) == (
                trains[0]["background"], trains[0]["block_color"], trains[0]["marker_colors"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    return (len(first["train"]) == len(second["train"])
            and len(first["test"]) == len(second["test"])
            and all(np.array_equal(a[role], b[role])
                    for split in ("train", "test")
                    for a, b in zip(first[split], second[split])
                    for role in ("input", "output")))


def _predict_visible(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    """Reconstruct the taught motion from visible grids alone."""
    def transform(grid: np.ndarray) -> np.ndarray:
        values, counts = np.unique(grid, return_counts=True)
        background = int(values[int(np.argmax(counts))])
        blocks = []
        for row in range(grid.shape[0] - 1):
            for column in range(grid.shape[1] - 1):
                color = int(grid[row, column])
                if color != background and np.all(grid[row:row + 2, column:column + 2] == color) and np.count_nonzero(grid == color) == 4:
                    blocks.append((row, column))
        assert len(blocks) == 1
        top, left = blocks[0]
        block_color = int(grid[top, left])
        answer = grid.copy()
        moves = []
        for row, column in np.argwhere((grid != background) & (grid != block_color)):
            row, column = int(row), int(column)
            if top <= row <= top + 1:
                destination = (row, left - 1 if column < left else left + 2)
            elif left <= column <= left + 1:
                destination = (top - 1 if row < top else top + 2, column)
            else:
                corner_row = top - 1 if row < top else top + 2
                corner_column = left - 1 if column < left else left + 2
                assert abs(row - corner_row) == abs(column - corner_column)
                destination = (corner_row, corner_column)
            moves.append((destination, int(grid[row, column])))
            answer[row, column] = background
        for destination, color in moves:
            answer[destination] = color
        return answer

    for pair in train_pairs:
        assert np.array_equal(transform(pair["input"]), pair["output"])
    return transform(test_input)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    shapes, structures, marker_counts = set(), set(), set()
    edge_count = source_size_count = multicolor_count = parallel_count = 0
    compact_count = long_count = parallel_diagonal_count = 0
    square_count = rectangular_count = three_color_count = 0
    background_variation = block_variation = marker_variation = 0
    parallel_sides, reused_combinations, unseen_directions = set(), 0, 0
    within_count_variation = within_shape_variation = 0
    relation_counts = {-1: 0, 0: 0, 1: 0}
    variable_puzzles = paired_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        trains = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        tests = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_pairs = puzzle["train"] + puzzle["test"]
        if len({a["marker_count"] for a in trains + tests}) > 1:
            within_count_variation += 1
        if len({pair["input"].shape for pair in all_pairs}) > 1:
            within_shape_variation += 1
        background_variation += len({a["background"] for a in trains}) > 1
        block_variation += len({a["block_color"] for a in trains}) > 1
        marker_variation += len({a["marker_colors"] for a in trains}) > 1
        if len({(a["background"], a["block_color"]) for a in trains}) > 1:
            variable_puzzles += 1
        if any(a["block"] == b["block"] and a["tracks"] == b["tracks"] and
               a["destinations"] == b["destinations"] and a["distances"] != b["distances"] and
               puzzle["train"][i]["input"].shape == puzzle["train"][j]["input"].shape
               for i, a in enumerate(trains) for j, b in enumerate(trains) if i < j):
            paired_puzzles += 1
        maximum = max(max(a["distances"]) for a in trains)
        train_combinations = {a["directions"] for a in trains}
        train_vocabulary = frozenset().union(*(a["directions"] for a in trains))
        for pair, analysis in zip(puzzle["test"], tests):
            assert np.array_equal(_predict_visible(puzzle["train"], pair["input"]), pair["output"])
            difference = max(analysis["distances"]) - maximum
            relation_counts[(difference > 0) - (difference < 0)] += 1
            reused_combinations += analysis["directions"] in train_combinations
            unseen_directions += not analysis["directions"] <= train_vocabulary
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair["input"], pair["output"])
                shape = pair["input"].shape
                shapes.add(shape)
                structures.add((shape, analysis["block"], analysis["tracks"], tuple(sorted(analysis["distances"]))))
                marker_counts.add(analysis["marker_count"])
                edge_count += analysis["edge"]
                source_size_count += shape == (10, 10)
                multicolor_count += len(analysis["marker_colors"]) > 1
                parallel_count += len(analysis["directions"]) < analysis["marker_count"]
                compact_count += shape[0] <= 9 and shape[1] <= 9
                long_count += max(analysis["distances"]) >= 8
                square_count += shape[0] == shape[1]
                rectangular_count += shape[0] != shape[1]
                three_color_count += len(np.unique(pair["input"])) == 3
                for side in ("N", "E", "S", "W"):
                    if side + "0" in analysis["tracks"] and side + "1" in analysis["tracks"]:
                        parallel_sides.add(side)
                        parallel_diagonal_count += bool(analysis["directions"] & _DIAGONAL)
    assert marker_counts == set(range(1, 9))
    assert len(shapes) >= 25 and len(structures) >= 100
    assert edge_count and source_size_count and multicolor_count and parallel_count
    assert compact_count and long_count and parallel_diagonal_count
    assert square_count and rectangular_count and three_color_count
    assert parallel_sides == {"N", "E", "S", "W"}
    assert reused_combinations and unseen_directions
    assert within_count_variation >= 30 and within_shape_variation >= 30
    assert background_variation and block_variation and marker_variation
    assert all(relation_counts.values()) and variable_puzzles and paired_puzzles
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert not _puzzles_equal(generate_puzzle(17), generate_puzzle(18))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (6, 3),
                          (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(91, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None: assert len(puzzle["train"]) == ntrain
        if ntest is not None: assert len(puzzle["test"]) == ntest
        assert _puzzles_equal(puzzle, generate_puzzle(91, ntrain, ntest))
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
