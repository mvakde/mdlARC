"""Generate panel-selection and panel-expansion ARC puzzles."""

import numpy as np


_SOURCE_INPUT_KEYS = frozenset({
    ("03050635062", "60452805008", "02850405304", "55555555555",
     "02054035340", "40852065002", "36050805860", "55555555555",
     "63050305003", "00250645280", "84052005406"),
    ("20354605060", "00850025403", "46053805208", "55555555555",
     "40850025064", "00250305300", "30654065802", "55555555555",
     "36050845200", "08452005803", "20050365640"),
    ("38454605208", "00058035603", "62050205400", "55555555555",
     "04258035040", "08650045026", "03052605030", "55555555555",
     "06056205360", "30850835004", "42050045208"),
    ("20050625004", "04354085306", "60053005802", "55555555555",
     "38056205048", "00450045600", "62053805032", "55555555555",
     "03650205060", "20054085008", "80456305234"),
    ("64050305040", "00352865802", "20854005630", "55555555555",
     "20050305362", "34658425004", "08050065800", "55555555555",
     "02450645028", "06350035406", "00052085300"),
})

_GEOMETRIES = (
    (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4),
    (4, 3), (3, 5), (5, 3), (4, 4), (4, 5), (5, 4), (5, 5),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count_argument(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    train = _count_argument(num_train, "num_train", 2, 10)
    test = _count_argument(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _geometry(rng, fixed=False):
    if fixed:
        return 3, 3, 1
    rows, cols = _GEOMETRIES[int(rng.integers(len(_GEOMETRIES)))]
    possible = [t for t in (1, 2) if rows * rows + (rows - 1) * t <= 30
                and cols * cols + (cols - 1) * t <= 30]
    return rows, cols, int(rng.choice(possible))


def _palette(rng, base, mode, index):
    if index == 0 or mode == 0:
        return tuple(int(x) for x in base)
    if mode == 1:
        other = [int(x) for x in base if int(x) != int(base[2])]
        rng.shuffle(other)
        return tuple(other[:2] + [int(base[2])] + other[2:])
    return tuple(int(x) for x in rng.permutation(10))


def _choose_position(rng, rows, cols, avoid=None):
    choices = [(r, c) for r in range(rows) for c in range(cols)]
    if avoid in choices and len(choices) > 1:
        choices.remove(avoid)
    return choices[int(rng.integers(len(choices)))]


def _cells_for_pattern(rng, rows, cols, count, mode):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    if mode == "upper" and rows == cols == 3 and count <= 6:
        cells = [(r, c) for r in range(2) for c in range(3)]
    elif mode == "blank_row" and count <= (rows - 1) * cols:
        blank = int(rng.integers(rows))
        cells = [(r, c) for r, c in cells if r != blank]
    elif mode == "blank_col" and count <= rows * (cols - 1):
        blank = int(rng.integers(cols))
        cells = [(r, c) for r, c in cells if c != blank]
    elif mode == "cluster":
        ar, ac = int(rng.integers(rows)), int(rng.integers(cols))
        jitter = rng.random(len(cells))
        cells = [cell for _, cell in sorted(zip(
            [(abs(r-ar) + abs(c-ac) + float(jitter[i]) * .4)
             for i, (r, c) in enumerate(cells)], cells))]
        return cells[:count]
    elif mode == "dispersed":
        chosen = [cells.pop(int(rng.integers(len(cells))))]
        while len(chosen) < count:
            scores = [min(abs(r-a) + abs(c-b) for a, b in chosen)
                      + float(rng.random()) * .3 for r, c in cells]
            chosen.append(cells.pop(int(np.argmax(scores))))
        return chosen
    selected = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in selected]


def _blueprint(rng, rows, cols, neutral, payload, style):
    area = rows * cols
    k = len(payload)
    if style == "anchor":
        count = 4
        pattern = "upper" if rng.random() < .5 else "random"
    elif style == "rich":
        count = max(k + 2, int(np.ceil(area * float(rng.uniform(.6, .85)))))
        count = min(area - 1, count)
        pattern = str(rng.choice(("cluster", "dispersed", "random")))
    else:
        count = int(rng.integers(k, area))
        pattern = str(rng.choice(("random", "random", "blank_row",
                                  "blank_col", "cluster", "dispersed",
                                  "upper")))
        if pattern == "blank_row" and count > (rows - 1) * cols:
            count = (rows - 1) * cols
        if pattern == "blank_col" and count > rows * (cols - 1):
            count = rows * (cols - 1)
        count = max(k, count)
    cells = _cells_for_pattern(rng, rows, cols, count, pattern)
    result = np.full((rows, cols), neutral, dtype=np.int8)
    colors = list(payload) + [int(rng.choice(payload)) for _ in range(count-k)]
    rng.shuffle(colors)
    for (r, c), color in zip(cells, colors):
        result[r, c] = color
    return result


def _decoy(rng, rows, cols, neutral, marker, payload, style):
    area, k = rows * cols, len(payload)
    if style == "anchor":
        marker_count, occupied = 1, k + 1
    elif style == "rich":
        marker_count, occupied = 2, k + 2
    else:
        marker_count = int(rng.integers(1, min(3, area-k-1) + 1))
        base = k + marker_count
        occupied = base + int(rng.integers(0, min(3, area-base)))
    positions = rng.choice(area, size=occupied, replace=False)
    symbols = (list(payload) + [marker] * marker_count
               + [int(rng.choice(payload)) for _ in range(occupied-k-marker_count)])
    rng.shuffle(symbols)
    panel = np.full((rows, cols), neutral, dtype=np.int8)
    for flat, color in zip(positions, symbols):
        panel.flat[int(flat)] = color
    return panel


def _make_pair(rng, geometry, palette, position, style, repeated=None):
    rows, cols, thick = geometry
    neutral, separator, marker = palette[:3]
    area = rows * cols
    if repeated is None:
        if style == "anchor":
            payload = tuple(palette[3:7])
        elif style == "rich":
            maximum = min(4, area - 3)
            k = int(rng.integers(2, maximum + 1))
            payload = tuple(int(x) for x in rng.choice(palette[3:], size=k, replace=False))
        else:
            k = int(rng.integers(1, min(5, area - 2) + 1))
            payload = tuple(int(x) for x in rng.choice(palette[3:], size=k, replace=False))
        blueprint = _blueprint(rng, rows, cols, neutral, payload, style)
    else:
        blueprint = repeated.copy()
        payload = tuple(int(x) for x in np.unique(blueprint) if int(x) != neutral)
    height = rows * rows + (rows - 1) * thick
    width = cols * cols + (cols - 1) * thick
    inp = np.full((height, width), separator, dtype=np.int8)
    out = np.full((height, width), separator, dtype=np.int8)
    for pr in range(rows):
        rs = pr * (rows + thick)
        for pc in range(cols):
            cs = pc * (cols + thick)
            panel = (blueprint if (pr, pc) == position else
                     _decoy(rng, rows, cols, neutral, marker, payload, style))
            inp[rs:rs+rows, cs:cs+cols] = panel
            out[rs:rs+rows, cs:cs+cols] = blueprint[pr, pc]
    return {"input": inp, "output": out}, blueprint


def _runs(non_separator):
    runs = []
    start = None
    for i, included in enumerate(list(non_separator) + [False]):
        if included and start is None:
            start = i
        elif not included and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _analyze_input(grid):
    height, width = grid.shape
    candidates = []
    for separator in range(10):
        full_rows = np.all(grid == separator, axis=1)
        full_cols = np.all(grid == separator, axis=0)
        if not np.any(full_rows) or not np.any(full_cols):
            continue
        row_runs, col_runs = _runs(~full_rows), _runs(~full_cols)
        rows, cols = len(row_runs), len(col_runs)
        if rows < 2 or cols < 2:
            continue
        if any(b-a != rows for a, b in row_runs):
            continue
        if any(b-a != cols for a, b in col_runs):
            continue
        if row_runs[0][0] != 0 or row_runs[-1][1] != height:
            continue
        if col_runs[0][0] != 0 or col_runs[-1][1] != width:
            continue
        row_gaps = [row_runs[i+1][0]-row_runs[i][1] for i in range(rows-1)]
        col_gaps = [col_runs[i+1][0]-col_runs[i][1] for i in range(cols-1)]
        if not row_gaps or not col_gaps or len(set(row_gaps + col_gaps)) != 1:
            continue
        thick = row_gaps[0]
        if thick < 1 or thick > 2:
            continue
        panels = {(r, c): grid[a:b, x:y].copy()
                  for r, (a, b) in enumerate(row_runs)
                  for c, (x, y) in enumerate(col_runs)}
        if any(np.any(panel == separator) for panel in panels.values()):
            continue
        incidence = {}
        for panel in panels.values():
            for color in np.unique(panel):
                color = int(color)
                incidence[color] = incidence.get(color, 0) + 1
        n = rows * cols
        markers = [color for color, count in incidence.items() if count == n-1]
        if len(markers) != 1:
            continue
        marker = markers[0]
        if any(count != n for color, count in incidence.items() if color != marker):
            continue
        missing = [pos for pos, panel in panels.items() if not np.any(panel == marker)]
        if len(missing) != 1:
            continue
        candidates.append((separator, marker, row_runs, col_runs, thick,
                           panels, missing[0]))
    _require(len(candidates) == 1, "input must expose one separator geometry and marker")
    sep, marker, row_runs, col_runs, thick, panels, position = candidates[0]
    return {"separator": sep, "marker": marker, "row_runs": row_runs,
            "col_runs": col_runs, "thickness": thick, "panels": panels,
            "position": position, "blueprint": panels[position]}


def _expand_from_input(grid, analysis):
    out = np.full(grid.shape, analysis["separator"], dtype=np.int8)
    blueprint = analysis["blueprint"]
    for r, (a, b) in enumerate(analysis["row_runs"]):
        for c, (x, y) in enumerate(analysis["col_runs"]):
            out[a:b, x:y] = blueprint[r, c]
    return out


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input and output dimensions differ")
    analysis = _analyze_input(inp)
    _require(np.array_equal(out, _expand_from_input(inp, analysis)),
             "output does not expand the unique marker-free panel")
    _require(not np.array_equal(inp, out), "input equals output")
    return analysis


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional NumPy arrays")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid dimensions out of bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integral")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors out of bounds")
            key = _key(pair["input"])
            pair_key = (key, _key(pair["output"]))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            _require(key not in _SOURCE_INPUT_KEYS, "source input copied")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair))
    train_positions = {a["position"] for a in analyses["train"]}
    _require(len(train_positions) >= 2, "training must show position independence")
    _require(any(any(np.count_nonzero(panel == a["marker"]) > 1
                     for pos, panel in a["panels"].items() if pos != a["position"])
                 for a in analyses["train"]),
             "training must show that marked panels can contain multiple markers")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(int(seed), rng, num_train, num_test)
    base = tuple(int(x) for x in rng.permutation(10))
    palette_mode = int(rng.integers(3))
    geometry_fixed = bool(rng.random() < .24)
    repeat_witness = bool(rng.random() < .3)
    puzzle = {"train": [], "test": []}
    used = set()
    first_position = None
    rich_record = None
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        local_index = index if split == "train" else index - train_count
        style = "anchor" if index == 0 else "rich" if index == 1 else "general"
        repeat = (split == "test" and local_index == test_count - 1
                  and repeat_witness and rich_record is not None)
        for attempt in range(60):
            if repeat:
                geometry, palette, prior_position, prior_blueprint = rich_record
                position = _choose_position(rng, geometry[0], geometry[1], prior_position)
                pair, blueprint = _make_pair(rng, geometry, palette, position,
                                             "general", prior_blueprint)
            else:
                if index == 0:
                    geometry = (3, 3, 1)
                elif geometry_fixed:
                    geometry = (3, 3, 1)
                elif index == 1:
                    # The second training example always exposes richer decoys.
                    geometry = _geometry(rng)
                elif split == "test" and rng.random() < .45:
                    selected = puzzle["train"][int(rng.integers(train_count))]
                    prior = _analyze_input(selected["input"])
                    geometry = (len(prior["row_runs"]), len(prior["col_runs"]),
                                prior["thickness"])
                else:
                    geometry = _geometry(rng)
                palette = _palette(rng, base, palette_mode, index)
                avoid = first_position if index == 1 else None
                position = _choose_position(rng, geometry[0], geometry[1], avoid)
                if split == "test" and rng.random() < .3:
                    familiar = []
                    for demonstration in puzzle["train"]:
                        observed = _analyze_input(demonstration["input"])
                        observed_geometry = (len(observed["row_runs"]),
                                             len(observed["col_runs"]),
                                             observed["thickness"])
                        if observed_geometry == geometry:
                            familiar.append(observed["position"])
                    if familiar:
                        position = familiar[int(rng.integers(len(familiar)))]
                pair, blueprint = _make_pair(rng, geometry, palette, position, style)
            key = _key(pair["input"])
            if key in used or key in _SOURCE_INPUT_KEYS:
                continue
            used.add(key)
            puzzle[split].append(pair)
            if index == 0:
                first_position = position
            if index == 1:
                rich_record = (geometry, palette, position, blueprint.copy())
            break
        else:
            raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _expect_value_error(seed, **kwargs):
    try:
        generate_puzzle(seed, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {kwargs}")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "automatic train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic test counts lack diversity")
    signatures = set()
    geometries, positions, palettes = set(), set(), set()
    marker_multiplicities, blueprint_color_counts = set(), set()
    repeated_layouts = 0
    variable_palettes = 0
    density_witnesses = 0
    blank_rows = blank_cols = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((tuple(_key(x["input"]) for x in puzzle[split]))
                             for split in ("train", "test")))
        analyses = []
        local_palettes = set()
        layout_positions = {}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                analyses.append(a)
                shape = (len(a["row_runs"]), len(a["col_runs"]), a["thickness"])
                geometries.add(shape)
                positions.add((shape[:2], a["position"]))
                palette_key = (a["separator"], a["marker"],
                               tuple(sorted(int(v) for v in np.unique(pair["input"]))))
                palettes.add(palette_key)
                local_palettes.add(palette_key)
                blueprint = a["blueprint"]
                blueprint_color_counts.add(len(np.unique(blueprint)))
                layout_key = (shape, tuple(int(v) for v in blueprint.ravel()))
                if layout_key in layout_positions and layout_positions[layout_key] != a["position"]:
                    repeated_layouts += 1
                layout_positions[layout_key] = a["position"]
                for pos, panel in a["panels"].items():
                    if pos != a["position"]:
                        marker_multiplicities.add(int(np.count_nonzero(panel == a["marker"])))
                common = set(int(v) for v in np.unique(blueprint))
                counts = {color: sum(int(np.count_nonzero(panel == color))
                                     for panel in a["panels"].values()) for color in common}
                neutral = max(counts, key=counts.get)
                density = int(np.count_nonzero(blueprint != neutral))
                decoy_densities = [int(np.count_nonzero(panel != neutral))
                                   for pos, panel in a["panels"].items() if pos != a["position"]]
                if any(x <= density for x in decoy_densities):
                    density_witnesses += 1
                blank_rows += int(any(np.all(row == neutral) for row in blueprint))
                blank_cols += int(any(np.all(col == neutral) for col in blueprint.T))
        variable_palettes += int(len(local_palettes) > 1)
    _require(len(signatures) == 42, "seed puzzles repeat")
    _require(len(geometries) >= 12 and len({(r, c) for r, c, _ in geometries}) >= 10,
             "geometry range too narrow")
    _require(len(positions) >= 30, "blueprint positions lack diversity")
    _require(len(palettes) >= 80 and variable_palettes >= 20,
             "color roles lack within-puzzle variation")
    _require(max(marker_multiplicities) >= 3 and len(blueprint_color_counts) >= 5,
             "panel contents lack diversity")
    _require(repeated_layouts >= 4 and density_witnesses >= 40,
             "position or density witnesses missing")
    _require(blank_rows >= 15 and blank_cols >= 15,
             "blank row and column patterns missing")
    for seed, train, test in ((1001, 3, 2), (1002, 6, 3), (1003, 9, 4),
                              (2001, 2, 1), (2002, 10, 1), (2003, 10, 4)):
        p = generate_puzzle(seed, train, test)
        _require((len(p["train"]), len(p["test"])) == (train, test),
                 "explicit counts not preserved")
    for seed, train, test in ((3001, None, 1), (3002, None, 4),
                              (3003, 2, None), (3004, 10, None)):
        p = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(p["train"]) == train, "partial train count not preserved")
        if test is not None:
            _require(len(p["test"]) == test, "partial test count not preserved")
    first = generate_puzzle(24680, 6, 3)
    second = generate_puzzle(24680, 6, 3)
    _require(all(np.array_equal(a[name], b[name])
                 for split in ("train", "test")
                 for a, b in zip(first[split], second[split])
                 for name in ("input", "output")), "generation is not deterministic")
    for seed, kwargs in ((4001, {"num_train": 1}), (4002, {"num_train": 11}),
                         (4003, {"num_test": 0}), (4004, {"num_test": 5}),
                         (4005, {"num_train": 2, "num_test": 2}),
                         (4006, {"num_train": 3, "num_test": 4}),
                         (4007, {"num_train": True}), (4008, {"num_test": 1.5})):
        _expect_value_error(seed, **kwargs)
    tampered = {split: [{name: pair[name].copy() for name in ("input", "output")}
                        for pair in first[split]] for split in ("train", "test")}
    tampered["test"][0]["output"][0, 0] = (int(tampered["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(tampered)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a wrong output")
