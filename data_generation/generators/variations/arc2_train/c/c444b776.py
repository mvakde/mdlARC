"""Generate divider-lattice motif-copying ARC puzzles."""

import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = {
    "2a7f4a92bb48038ecd808261d9b882ccd6d27f99458448ba0691c4bcb65ca040",
    "067d2d21ad30f5355cfd7e9b5c82087743799de02a66e4b2996baaa5e7ccf21c",
    "41aeaea4c7d1ac3067e34b70396a9a3fc1064298b1609862512daae57686e05e",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Choose only conventions that matter across pairs."""
    roles = rng.choice(10, size=2, replace=False)
    return {
        "base_roles": (int(roles[0]), int(roles[1])),
        "vary_roles": bool(rng.random() < 0.67),
        "classic_progression": bool(rng.random() < 0.20),
    }


def _layout(rng):
    kind = int(rng.choice(4, p=[0.23, 0.23, 0.30, 0.24]))
    if kind == 0:
        return int(rng.integers(2, 6)), 1
    if kind == 1:
        return 1, int(rng.integers(2, 6))
    if kind == 2:
        return int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if rng.random() < 0.5:
        return int(rng.integers(2, 4)), int(rng.integers(3, 6))
    return int(rng.integers(3, 6)), int(rng.integers(2, 4))


def _train_layouts(rng, count, params):
    if params["classic_progression"]:
        layouts = [(2, 1), (2, 2)]
    else:
        layouts = [(int(rng.integers(2, 5)), int(rng.integers(2, 5))), _layout(rng)]
        rng.shuffle(layouts)
    layouts.extend(_layout(rng) for _ in range(count - 2))
    return layouts


def _test_layout(rng, train_layouts):
    mode = int(rng.integers(0, 6))
    reference = train_layouts[int(rng.integers(len(train_layouts)))]
    if mode == 0:
        return reference
    if mode == 1:
        return reference[1], reference[0]
    if mode == 2:
        next_row = max(layout[0] for layout in train_layouts) + 1
        return int(rng.integers(next_row, 7)), int(rng.integers(2, 5))
    if mode == 3:
        next_column = max(layout[1] for layout in train_layouts) + 1
        return int(rng.integers(2, 5)), int(rng.integers(next_column, 7))
    if mode == 4:
        return int(rng.integers(2, 6)), 1
    return _layout(rng)


def _roles(rng, params, split, index):
    base = params["base_roles"]
    if not params["vary_roles"] or (split == "train" and index == 0):
        return base
    if split == "train" and index == 1:
        if rng.random() < 0.5:
            return base[1], base[0]
        while True:
            roles = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
            if roles != base:
                return roles
    if rng.random() < 0.17:
        return base[1], base[0]
    return tuple(int(v) for v in rng.choice(10, size=2, replace=False))


def _compartment_size(rng, rows, columns):
    max_height = (31 - rows) // rows
    max_width = (31 - columns) // columns

    def choose(limit):
        mode = int(rng.integers(0, 10))
        if mode == 0:
            return limit
        if mode == 1 and limit >= 9:
            return 9
        if mode == 2:
            return int(rng.integers(3, min(7, limit) + 1))
        return int(rng.integers(3, limit + 1))

    return choose(max_height), choose(max_width)


def _neighbors(row, column, height, width):
    return [
        (r, c)
        for r, c in ((row - 1, column), (row + 1, column),
                     (row, column - 1), (row, column + 1))
        if 0 <= r < height and 0 <= c < width
    ]


def _sample_mask(rng, height, width, style, min_cells):
    area = height * width
    mask = np.zeros((height, width), dtype=bool)
    if style == 0:  # scattered, as in the source
        count = int(rng.integers(max(4, min_cells), min(20, max(4, area // 3)) + 1))
        mask.ravel()[rng.choice(area, size=count, replace=False)] = True
    elif style == 1:  # one compact connected cluster
        target = int(rng.integers(max(4, min_cells), min(30, max(4, area // 2)) + 1))
        start = (int(rng.integers(height)), int(rng.integers(width)))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in zip(*np.nonzero(mask)):
                frontier.update(_neighbors(int(r), int(c), height, width))
            frontier = [point for point in frontier if not mask[point]]
            if not frontier:
                break
            mask[frontier[int(rng.integers(len(frontier)))]] = True
    elif style == 2:  # several small objects
        for _ in range(int(rng.integers(2, 5))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            mask[r, c] = True
            for _ in range(int(rng.integers(1, 6))):
                r, c = _neighbors(r, c, height, width)[
                    int(rng.integers(len(_neighbors(r, c, height, width))))
                ]
                mask[r, c] = True
    elif style == 3:  # a partial straight or bent stroke
        r, c = int(rng.integers(height)), int(rng.integers(width))
        vertical = bool(rng.integers(2))
        length = int(rng.integers(2, max(3, min(height if vertical else width, 8)) + 1))
        for k in range(length):
            mask[min(height - 1, r + k) if vertical else r,
                 c if vertical else min(width - 1, c + k)] = True
        if rng.random() < 0.7:
            for k in range(int(rng.integers(1, 4))):
                mask[min(height - 1, r + k), min(width - 1, c + k)] = True
    elif style == 4:  # spread to edges
        edge = [(r, c) for r in range(height) for c in range(width)
                if r in (0, height - 1) or c in (0, width - 1)]
        for index in rng.choice(len(edge), size=min(len(edge), int(rng.integers(2, 6))), replace=False):
            mask[edge[int(index)]] = True
        extra = int(rng.integers(0, min(12, area // 4) + 1))
        if extra:
            mask.ravel()[rng.choice(area, size=extra, replace=False)] = True
    elif style == 5:  # occasional very small template
        count = int(rng.integers(min_cells, 4))
        mask.ravel()[rng.choice(area, size=count, replace=False)] = True
    elif style == 6:  # denser but still visibly incomplete
        density = float(rng.uniform(0.24, 0.56))
        mask = rng.random((height, width)) < density
    else:  # concentrated in a subregion
        sub_height = int(rng.integers(2, max(3, height // 2 + 1)))
        sub_width = int(rng.integers(2, max(3, width // 2 + 1)))
        r0 = int(rng.integers(0, height - sub_height + 1))
        c0 = int(rng.integers(0, width - sub_width + 1))
        count = int(rng.integers(max(4, min_cells), min(sub_height * sub_width, 12) + 1))
        sub = np.zeros(sub_height * sub_width, dtype=bool)
        sub[rng.choice(sub.size, size=count, replace=False)] = True
        mask[r0:r0 + sub_height, c0:c0 + sub_width] = sub.reshape(sub_height, sub_width)
    return mask


def _sample_template(rng, height, width, background, divider, multicolor):
    minimum = 2 if multicolor else 1
    for _ in range(80):
        style = int(rng.integers(8))
        mask = _sample_mask(rng, height, width, style, minimum)
        cells = int(mask.sum())
        required = minimum if style in (3, 5) else max(4, minimum)
        if cells < required or np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)):
            continue
        available = [color for color in range(10) if color not in (background, divider)]
        if multicolor:
            color_count = int(rng.integers(2, min(7, cells, len(available)) + 1))
        else:
            maximum = min(7, cells, len(available))
            weights = np.array(([4, 3] + [2] * max(0, maximum - 2))[:maximum],
                               dtype=float)
            weights /= weights.sum()
            color_count = int(rng.choice(np.arange(1, maximum + 1), p=weights))
        colors = rng.choice(available, size=color_count, replace=False)
        probabilities = rng.dirichlet(np.full(color_count, 0.55))
        values = np.array(
            [*colors, *rng.choice(colors, size=cells - color_count, p=probabilities)],
            dtype=np.int8,
        )
        rng.shuffle(values)
        template = np.full((height, width), background, dtype=np.int8)
        template[mask] = values
        return template
    raise RuntimeError("could not construct motif")


def _make_pair(rng, layout, source, roles, multicolor):
    rows, columns = layout
    height, width = _compartment_size(rng, rows, columns)
    background, divider = roles
    grid = np.full((rows * height + rows - 1, columns * width + columns - 1),
                   background, dtype=np.int8)
    for row in range(1, rows):
        grid[row * (height + 1) - 1, :] = divider
    for column in range(1, columns):
        grid[:, column * (width + 1) - 1] = divider
    template = _sample_template(rng, height, width, background, divider, multicolor)
    sr, sc = source
    grid[sr * (height + 1):sr * (height + 1) + height,
         sc * (width + 1):sc * (width + 1) + width] = template
    output = grid.copy()
    for row in range(rows):
        for column in range(columns):
            output[row * (height + 1):row * (height + 1) + height,
                   column * (width + 1):column * (width + 1) + width] = template
    return {"input": grid, "output": output}


def _grid_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.uint8).tobytes()


def _pair_hash(pair):
    digest = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name]
        digest.update(bytes(grid.shape))
        digest.update(np.ascontiguousarray(grid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _axis_slices(length, lines):
    bounds = (-1, *lines, length)
    slices = [slice(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [piece.stop - piece.start for piece in slices]
    _require(min(sizes) >= 3 and len(set(sizes)) == 1, "noncongruent or tiny compartments")
    return slices


def _analyze_input(grid):
    candidates = []
    for value in np.unique(grid):
        color = int(value)
        mask = grid == color
        rows = tuple(int(i) for i in np.flatnonzero(mask.all(axis=1)))
        columns = tuple(int(i) for i in np.flatnonzero(mask.all(axis=0)))
        if not rows and not columns:
            continue
        lines = np.zeros_like(mask)
        if rows:
            lines[list(rows), :] = True
        if columns:
            lines[:, list(columns)] = True
        if np.array_equal(mask, lines):
            candidates.append((color, rows, columns, lines))
    _require(len(candidates) == 1, "divider color or lines are ambiguous")
    divider, rows, columns, line_mask = candidates[0]
    row_slices = _axis_slices(grid.shape[0], rows)
    column_slices = _axis_slices(grid.shape[1], columns)
    _require(len(row_slices) * len(column_slices) >= 2, "missing partition")
    interior = grid[~line_mask]
    values, counts = np.unique(interior, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    _require(background != divider, "background and divider coincide")
    populated = []
    template = None
    for ri, rs in enumerate(row_slices):
        for ci, cs in enumerate(column_slices):
            block = grid[rs, cs]
            if np.any(block != background):
                populated.append((ri, ci))
                template = block.copy()
    _require(len(populated) == 1, "exactly one input compartment must be populated")
    motif_mask = template != background
    _require(not np.any(motif_mask.all(axis=0)) and not np.any(motif_mask.all(axis=1)),
             "motif imitates a full line")
    return {
        "background": background, "divider": divider,
        "row_slices": row_slices, "column_slices": column_slices,
        "layout": (len(row_slices), len(column_slices)),
        "size": (row_slices[0].stop - row_slices[0].start,
                 column_slices[0].stop - column_slices[0].start),
        "source": populated[0], "template": template,
        "cells": int(motif_mask.sum()),
        "colors": tuple(int(v) for v in np.unique(template[motif_mask])),
    }


def _expected_output(input_grid, analysis):
    expected = input_grid.copy()
    for rs in analysis["row_slices"]:
        for cs in analysis["column_slices"]:
            expected[rs, cs] = analysis["template"]
    return expected


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input/output dimensions differ")
    analysis = _analyze_input(inp)
    _require(np.array_equal(out, _expected_output(inp, analysis)),
             "output is not exact local-coordinate copying")
    _require(not np.array_equal(inp, out), "degenerate pair")
    _require(int(np.count_nonzero(inp != out)) ==
             analysis["cells"] * (analysis["layout"][0] * analysis["layout"][1] - 1),
             "incorrect changed-cell count")
    return analysis


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    train_layouts = _train_layouts(rng, train_count, params)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_positions = set()
    train_configs = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 1000 if split == "train" else 2000, index])
            )
            for _ in range(100):
                layout = train_layouts[index] if split == "train" else _test_layout(pair_rng, train_layouts)
                source = (int(pair_rng.integers(layout[0])),
                          int(pair_rng.integers(layout[1])))
                if split == "train" and index == 1 and source in train_positions:
                    continue
                roles = _roles(pair_rng, params, split, index)
                pair = _make_pair(pair_rng, layout, source, roles,
                                  multicolor=(split == "train" and index == 0))
                key = _grid_key(pair["input"])
                config = (layout, source, _analyze_input(pair["input"])["size"])
                if key in seen or (split == "test" and config in train_configs):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train":
                    train_positions.add(source)
                    train_configs.add(config)
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(type(puzzle) is dict and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(type(puzzle["train"]) is list and type(puzzle["test"]) is list,
             "splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(type(pair) is dict and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(all(1 <= n <= 30 for n in grid.shape),
                         "grid size outside 1..30")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         bool(np.all((grid >= 0) & (grid <= 9))),
                         "grid must contain integer ARC colors")
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            _require(_pair_hash(pair) not in _SOURCE_PAIR_HASHES,
                     "exact original source pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analyses[split].append(_inspect_pair(pair))
    train = analyses["train"]
    test = analyses["test"]
    _require(any(a["layout"][0] > 1 and a["layout"][1] > 1 for a in train),
             "training must show copying in both directions")
    _require(any(len(a["colors"]) >= 2 for a in train),
             "training must show literal multicolor copying")
    _require(len({a["source"] for a in train}) >= 2,
             "training must vary the source position")
    train_roles = {(a["background"], a["divider"]) for a in train}
    all_roles = train_roles | {(a["background"], a["divider"]) for a in test}
    _require(len(all_roles) == 1 or len(train_roles) >= 2,
             "test palette changes lack training evidence")
    train_configs = {(a["layout"], a["source"], a["size"]) for a in train}
    _require(all((a["layout"], a["source"], a["size"]) not in train_configs for a in test),
             "test repeats a training configuration")


def _puzzles_equal(first, second):
    if len(first["train"]) != len(second["train"]) or len(first["test"]) != len(second["test"]):
        return False
    return all(np.array_equal(a[name], b[name])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for name in ("input", "output"))


def _motif_metrics(analysis):
    template = analysis["template"]
    mask = template != analysis["background"]
    points = set(zip(*np.nonzero(mask)))
    remaining = set(points)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            row, column = stack.pop()
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    adjacency = sum((row + 1, column) in points or (row, column + 1) in points
                    for row, column in points)
    rows = [row for row, _ in points]
    columns = [column for _, column in points]
    _, counts = np.unique(template[mask], return_counts=True)
    line = (
        any(bool(np.any(mask[row, :-2] & mask[row, 1:-1] & mask[row, 2:]))
            for row in range(mask.shape[0]))
        or any(bool(np.any(mask[:-2, column] & mask[1:-1, column] &
                           mask[2:, column]))
               for column in range(mask.shape[1]))
    )
    edges = sum((min(rows) == 0, max(rows) == mask.shape[0] - 1,
                 min(columns) == 0, max(columns) == mask.shape[1] - 1))
    return {
        "density": len(points) / mask.size,
        "components": components,
        "adjacency": adjacency,
        "edge": (min(rows) == 0 or max(rows) == mask.shape[0] - 1 or
                 min(columns) == 0 or max(columns) == mask.shape[1] - 1),
        "bounding_fraction": ((max(rows) - min(rows) + 1) *
                              (max(columns) - min(columns) + 1)) / mask.size,
        "majority_fraction": int(max(counts)) / len(points),
        "line": line,
        "asymmetric": (not np.array_equal(mask, mask[::-1, :]) and
                       not np.array_equal(mask, mask[:, ::-1])),
        "multiple_edges": edges >= 2,
    }


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    _require(all(not _puzzles_equal(puzzles[0], p) for p in puzzles[1:]),
             "seeds repeat puzzles")
    analyses = [_analyze_input(pair["input"])
                for puzzle in puzzles for split in ("train", "test")
                for pair in puzzle[split]]
    layouts = {a["layout"] for a in analyses}
    sizes = {a["size"] for a in analyses}
    sources = {a["source"] for a in analyses}
    roles = {(a["background"], a["divider"]) for a in analyses}
    colors = {len(a["colors"]) for a in analyses}
    _require(len(layouts) >= 12 and len(sizes) >= 40 and len(sources) >= 10,
             "insufficient geometric diversity")
    _require(len(roles) >= 20 and {1, 2, 3}.issubset(colors),
             "insufficient motif/palette diversity")
    _require(any(a["size"] == (9, 9) for a in analyses),
             "9x9 compartments were accidentally excluded")
    _require(any(a["layout"][0] == 1 for a in analyses) and
             any(a["layout"][1] == 1 for a in analyses) and
             any(a["layout"][0] >= 3 for a in analyses),
             "one-axis and tall lattices are missing")
    all_pairs = [pair for puzzle in puzzles for split in ("train", "test")
                 for pair in puzzle[split]]
    motifs = [_motif_metrics(a) for a in analyses]
    _require(any(a["size"][0] >= 15 for a in analyses) and
             any(a["size"][1] >= 15 for a in analyses) and
             any(30 in pair["input"].shape for pair in all_pairs),
             "wide, tall, or maximum-sized grids are missing")
    _require(sum(a["cells"] <= 3 for a in analyses) >= 10 and
             sum(m["density"] <= .1 for m in motifs) >= 20 and
             sum(m["density"] >= .25 for m in motifs) >= 10,
             "motif size/density range is too narrow")
    _require(sum(m["components"] == 1 for m in motifs) >= 20 and
             sum(m["components"] >= 3 for m in motifs) >= 20 and
             sum(m["adjacency"] == 0 for m in motifs) >= 20,
             "motif connectivity range is too narrow")
    _require(sum(m["edge"] for m in motifs) >= 20 and
             sum(m["bounding_fraction"] <= .25 for m in motifs) >= 10 and
             sum(m["multiple_edges"] for m in motifs) >= 10,
             "motif placement range is too narrow")
    _require(sum(m["line"] for m in motifs) >= 10 and
             sum(m["asymmetric"] for m in motifs) >= 40,
             "line and asymmetric motifs are missing")
    _require(sum(len(a["colors"]) == 1 for a in analyses) >= 15 and
             sum(len(a["colors"]) >= 5 for a in analyses) >= 8 and
             sum(len(a["colors"]) >= 2 and m["majority_fraction"] >= .7
                 for a, m in zip(analyses, motifs)) >= 5,
             "motif color distributions are too narrow")
    within_layout = 0
    within_roles = 0
    train_swaps = 0
    same_layout_tests = 0
    transposed_tests = 0
    more_row_tests = 0
    more_column_tests = 0
    vertical_tests = 0
    for puzzle in puzzles:
        train = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        test = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        combined = train + test
        within_layout += len({a["layout"] for a in combined}) >= 2
        within_roles += len({(a["background"], a["divider"])
                             for a in combined}) >= 2
        train_swaps += any(
            (a["background"], a["divider"]) ==
            (b["divider"], b["background"])
            for a in train for b in train if a is not b
        )
        for target in test:
            same_layout_tests += any(target["layout"] == a["layout"] for a in train)
            transposed_tests += any(target["layout"] == a["layout"][::-1]
                                    for a in train)
            more_row_tests += target["layout"][0] > max(a["layout"][0] for a in train)
            more_column_tests += target["layout"][1] > max(a["layout"][1] for a in train)
            vertical_tests += target["layout"][1] == 1
    _require(within_layout >= 35 and within_roles >= 8 and train_swaps >= 4,
             "within-puzzle layout/palette diversity is too narrow")
    _require(min(same_layout_tests, transposed_tests, more_row_tests,
                 more_column_tests, vertical_tests) >= 2,
             "train/test relationship range is too narrow")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts changed")
    for index, (train, test) in enumerate(((None, 1), (None, 4),
                                           (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, train, test)
        _require((train is None or len(puzzle["train"]) == train) and
                 (test is None or len(puzzle["test"]) == test),
                 "partial count changed")
    for index, (train, test) in enumerate(((1, None), (11, None),
                                           (None, 0), (None, 5),
                                           (2, 2), (3, 4), (4, 4))):
        try:
            generate_puzzle(4000 + index, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzles_equal(generate_puzzle(123456, 8, 3),
                           generate_puzzle(123456, 8, 3)),
             "generation is nondeterministic")
