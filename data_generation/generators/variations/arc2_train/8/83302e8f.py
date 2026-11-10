"""Damaged regular lattice puzzles with component based recoloring."""

import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "2d9fdb587e8a06e5625f5d0d3d5d141a41861f8ce69714c4c6ba70e74d56921b",
    "cdb81ff68fe655bc4808e947a56752f91e292900aaf7a477bb06c6999dd0271a",
    "ecb10d9478b0e419577b8953c41198fbf53079c668c1ec283964edf4add5a6e3",
    "0e8206dff82e7868ceac3ee4b02276526e9789630466a7ccc57f9fd3b00c1644",
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test, rng):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        # Count variation is assured even for consecutive integer seeds.
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _pair_digest(pair):
    inp = np.asarray(pair["input"], dtype=np.int8)
    out = np.asarray(pair["output"], dtype=np.int8)
    return hashlib.sha256(bytes(inp.shape) + inp.tobytes() + out.tobytes()).hexdigest()


def _spec(height, width, cell_h, cell_w):
    row_lines = np.zeros(height, dtype=bool)
    col_lines = np.zeros(width, dtype=bool)
    row_lines[np.arange(cell_h, height, cell_h + 1)] = True
    col_lines[np.arange(cell_w, width, cell_w + 1)] = True
    return {
        "cell_h": cell_h,
        "cell_w": cell_w,
        "rows": (height + 1) // (cell_h + 1),
        "cols": (width + 1) // (cell_w + 1),
        "row_lines": row_lines,
        "col_lines": col_lines,
        "line_mask": row_lines[:, None] | col_lines[None, :],
    }


def _infer_lattice(grid):
    height, width = grid.shape
    colors = np.unique(grid)
    _require(colors.size == 2, "input needs one background and one divider color")
    candidates = []
    for cell_h in range(2, 7):
        if (height + 1) % (cell_h + 1):
            continue
        rows = (height + 1) // (cell_h + 1)
        if not 3 <= rows <= 9:
            continue
        for cell_w in range(2, 7):
            if (width + 1) % (cell_w + 1):
                continue
            cols = (width + 1) // (cell_w + 1)
            if not 3 <= cols <= 9:
                continue
            spec = _spec(height, width, cell_h, cell_w)
            line_mask = spec["line_mask"]
            for structure in colors:
                structure = int(structure)
                background = int(colors[0] if colors[1] == structure else colors[1])
                if np.any(grid[~line_mask] != background):
                    continue
                if np.count_nonzero(grid[line_mask] == structure) < 0.63 * np.count_nonzero(line_mask):
                    continue
                if any(
                    np.count_nonzero(grid[r] == structure) < 0.45 * width
                    for r in np.flatnonzero(spec["row_lines"])
                ):
                    continue
                if any(
                    np.count_nonzero(grid[:, c] == structure) < 0.45 * height
                    for c in np.flatnonzero(spec["col_lines"])
                ):
                    continue
                candidates.append({**spec, "structure": structure, "background": background})
    _require(len(candidates) == 1, "intended lattice is not uniquely visible")
    return candidates[0]


def _components(grid, spec):
    height, width = grid.shape
    background = spec["background"]
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or grid[start_r, start_c] != background:
                continue
            cells = []
            chambers = set()
            queue = [(start_r, start_c)]
            seen[start_r, start_c] = True
            head = 0
            while head < len(queue):
                r, c = queue[head]
                head += 1
                cells.append((r, c))
                if not spec["row_lines"][r] and not spec["col_lines"][c]:
                    chambers.add((r // (spec["cell_h"] + 1), c // (spec["cell_w"] + 1)))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and not seen[rr, cc] and grid[rr, cc] == background:
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            _require(chambers, "a missing divider component reaches no chamber")
            result.append((cells, chambers))
    return result


def _categories(grid, spec):
    result = np.zeros(grid.shape, dtype=np.int8)
    groups = _components(grid, spec)
    for cells, chambers in groups:
        category = 1 if len(chambers) == 1 else 2
        for r, c in cells:
            result[r, c] = category
    return result, groups


def _edges(rows, cols):
    result = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if r + 1 < rows:
                result.append((u, u + cols))
            if c + 1 < cols:
                result.append((u, u + 1))
    return result


def _grow(rng, selected, vertices, allowed_edges, target):
    vertices = set(vertices)
    while len(vertices) < target:
        frontier = [e for e in allowed_edges if (e[0] in vertices) != (e[1] in vertices)]
        if not frontier:
            break
        edge = frontier[int(rng.integers(len(frontier)))]
        selected.add(edge)
        vertices.update(edge)
    return vertices


def _choose_edges(rng, rows, cols, regime, required_axis=None):
    all_edges = _edges(rows, cols)
    protected = int(rng.integers(rows * cols))
    allowed = [e for e in all_edges if protected not in e]
    modes = ("sparse", "chain", "branch", "loop", "split", "diffuse")
    weights = ([.24, .22, .19, .12, .09, .14] if regime == "source_train"
               else [.12, .13, .17, .14, .22, .22])
    mode = str(rng.choice(modes, p=weights))
    selected = set()
    if mode == "sparse":
        count = int(rng.integers(1, min(6, len(allowed)) + 1))
        for i in rng.choice(len(allowed), size=count, replace=False):
            selected.add(allowed[int(i)])
    elif mode == "chain":
        # Start may be a corner, edge, or interior chamber.
        start = int(rng.choice([v for v in range(rows * cols) if v != protected]))
        target = int(rng.integers(3, min(12, rows * cols - 1) + 1))
        path = [start]
        for _ in range(target - 1):
            options = [e for e in allowed if path[-1] in e and (e[0] if e[1] == path[-1] else e[1]) not in path]
            if not options:
                break
            edge = options[int(rng.integers(len(options)))]
            selected.add(edge)
            path.append(edge[0] if edge[1] == path[-1] else edge[1])
    elif mode == "branch":
        centers = [v for v in range(rows * cols) if sum(v in e for e in allowed) >= 3]
        if centers:
            center = int(rng.choice(centers))
            incident = [e for e in allowed if center in e]
            for i in rng.choice(len(incident), size=3, replace=False):
                selected.add(incident[int(i)])
            group = {v for e in selected for v in e}
            target = int(rng.integers(len(group), min(rows * cols - 1, 15) + 1))
            _grow(rng, selected, group, allowed, target)
        else:
            mode = "diffuse"
    elif mode == "loop":
        plaques = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                u = r * cols + c
                plaque = {(u, u + 1), (u, u + cols), (u + 1, u + cols + 1), (u + cols, u + cols + 1)}
                if plaque.issubset(allowed):
                    plaques.append(plaque)
        if plaques:
            selected.update(plaques[int(rng.integers(len(plaques)))])
            group = {v for e in selected for v in e}
            target = int(rng.integers(len(group), min(rows * cols - 1, 13) + 1))
            _grow(rng, selected, group, allowed, target)
        else:
            mode = "diffuse"
    elif mode == "split":
        # A closed cut keeps two large damaged regions independent.
        cuts = []
        for cut in range(1, rows):
            if min(cut * cols, (rows - cut) * cols) >= 7:
                cuts.append((0, cut))
        for cut in range(1, cols):
            if min(cut * rows, (cols - cut) * rows) >= 7:
                cuts.append((1, cut))
        if cuts:
            axis, cut = cuts[int(rng.integers(len(cuts)))]
            if axis == 0:
                first = {v for v in range(rows * cols) if v // cols < cut}
            else:
                first = {v for v in range(rows * cols) if v % cols < cut}
            parts = (first, set(range(rows * cols)) - first)
            for part in parts:
                internal = [e for e in allowed if e[0] in part and e[1] in part]
                starts = [v for v in part if v != protected and any(v in e for e in internal)]
                start = int(rng.choice(starts))
                limit = min(len(part) - (protected in part), 15)
                size = int(rng.integers(6, limit + 1))
                grown = _grow(rng, selected, {start}, internal, size)
                # Some closed cycles occur without joining the two groups.
                internal_extra = [e for e in internal if e[0] in grown and e[1] in grown and e not in selected]
                if internal_extra and rng.random() < .45:
                    selected.add(internal_extra[int(rng.integers(len(internal_extra)))])
        else:
            mode = "diffuse"
    if mode == "diffuse":
        low = max(2, int(.13 * len(allowed)))
        high = max(low + 1, int(.43 * len(allowed)))
        count = int(rng.integers(low, high + 1))
        for i in rng.choice(len(allowed), size=count, replace=False):
            selected.add(allowed[int(i)])
    if required_axis is not None:
        candidates = [e for e in allowed if ((e[1] - e[0] == cols) == (required_axis == "horizontal"))]
        if not any(e in selected for e in candidates):
            selected.add(candidates[int(rng.integers(len(candidates)))])
    _require(selected, "no opened chamber boundary")
    return sorted(selected), mode


def _geometry(rng, regime):
    size_weights = ([.04, .19, .30, .31, .16] if regime == "source_train"
                    else [.08, .31, .33, .22, .06])
    count_weights = ([.12, .19, .23, .21, .16, .09] if regime == "source_train"
                     else [.06, .12, .20, .23, .23, .16])
    for _ in range(80):
        cell_h = int(rng.choice(np.arange(2, 7), p=size_weights))
        cell_w = int(rng.choice(np.arange(2, 7), p=size_weights))
        rows = int(rng.choice(np.arange(3, 9), p=count_weights))
        cols = int(rng.choice(np.arange(3, 9), p=count_weights))
        height = rows * (cell_h + 1) - 1
        width = cols * (cell_w + 1) - 1
        if height <= 30 and width <= 30:
            return cell_h, cell_w, rows, cols
    return 3, 3, 5, 5


def _construct(rng, geometry, background, structure, edges):
    cell_h, cell_w, rows, cols = geometry
    height = rows * (cell_h + 1) - 1
    width = cols * (cell_w + 1) - 1
    spec = _spec(height, width, cell_h, cell_w)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[spec["line_mask"]] = structure
    for u, v in edges:
        if v - u == cols:
            chamber_r, chamber_c = divmod(u, cols)
            line_r = (chamber_r + 1) * (cell_h + 1) - 1
            length = int(rng.choice(np.arange(1, min(3, cell_w) + 1),
                                    p=np.array([.52, .33, .15][:min(3, cell_w)]) / sum([.52, .33, .15][:min(3, cell_w)])))
            offset = int(rng.integers(cell_w - length + 1))
            start_c = chamber_c * (cell_w + 1) + offset
            grid[line_r, start_c:start_c + length] = background
        else:
            chamber_r, chamber_c = divmod(u, cols)
            line_c = (chamber_c + 1) * (cell_w + 1) - 1
            length = int(rng.choice(np.arange(1, min(3, cell_h) + 1),
                                    p=np.array([.52, .33, .15][:min(3, cell_h)]) / sum([.52, .33, .15][:min(3, cell_h)])))
            offset = int(rng.integers(cell_h - length + 1))
            start_r = chamber_r * (cell_h + 1) + offset
            grid[start_r:start_r + length, line_c] = background
    junctions = []
    for r in np.flatnonzero(spec["row_lines"]):
        for c in np.flatnonzero(spec["col_lines"]):
            if any(grid[rr, cc] == background for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))):
                junctions.append((r, c))
    if junctions and rng.random() < .36:
        order = rng.permutation(len(junctions))
        amount = min(len(junctions), int(rng.integers(1, 4)))
        for i in order[:amount]:
            r, c = junctions[int(i)]
            grid[r, c] = background
    return grid


def _solve(grid, spec, intact, breached):
    categories, _ = _categories(grid, spec)
    result = grid.copy()
    result[categories == 1] = intact
    result[categories == 2] = breached
    return result


def _generate_pair(rng, regime, intact, breached, background, structure, required_axis):
    for _ in range(60):
        geometry = _geometry(rng, regime)
        edges, _ = _choose_edges(rng, geometry[2], geometry[3], regime, required_axis)
        grid = _construct(rng, geometry, background, structure, edges)
        try:
            spec = _infer_lattice(grid)
            categories, _ = _categories(grid, spec)
            if np.any(categories == 1) and np.any(categories == 2):
                return {"input": grid, "output": _solve(grid, spec, intact, breached)}
        except AssertionError:
            continue
    raise ValueError("could not construct an unambiguous damaged lattice")


def _inspect(pair):
    inp, out = pair["input"], pair["output"]
    spec = _infer_lattice(inp)
    categories, groups = _categories(inp, spec)
    intact_mask, breached_mask = categories == 1, categories == 2
    _require(np.any(intact_mask) and np.any(breached_mask), "both region labels must appear")
    intact_values = np.unique(out[intact_mask])
    breached_values = np.unique(out[breached_mask])
    _require(intact_values.size == 1 and breached_values.size == 1, "inconsistent component labels")
    intact, breached = int(intact_values[0]), int(breached_values[0])
    _require(len({intact, breached, spec["structure"], spec["background"]}) == 4,
             "color roles must be distinct")
    _require(np.array_equal(out, _solve(inp, spec, intact, breached)), "incorrect component coloring")
    holes = spec["line_mask"] & (inp == spec["background"])
    _require(np.any(holes), "no divider opening")
    _require(np.all(categories[holes] == 2), "missing divider pixel is not breached")
    horizontal = 0
    vertical = 0
    aperture_sizes = []
    graph_edges = []
    for r in np.flatnonzero(spec["row_lines"]):
        upper_row = r // (spec["cell_h"] + 1)
        for chamber_c in range(spec["cols"]):
            start = chamber_c * (spec["cell_w"] + 1)
            length = int(np.count_nonzero(holes[r, start:start + spec["cell_w"]]))
            if length:
                horizontal += 1
                aperture_sizes.append(length)
                u = upper_row * spec["cols"] + chamber_c
                graph_edges.append((u, u + spec["cols"]))
    for c in np.flatnonzero(spec["col_lines"]):
        left_col = c // (spec["cell_w"] + 1)
        for chamber_r in range(spec["rows"]):
            start = chamber_r * (spec["cell_h"] + 1)
            length = int(np.count_nonzero(holes[start:start + spec["cell_h"], c]))
            if length:
                vertical += 1
                aperture_sizes.append(length)
                u = chamber_r * spec["cols"] + left_col
                graph_edges.append((u, u + 1))
    junctions = int(np.count_nonzero(holes & spec["row_lines"][:, None] & spec["col_lines"][None, :]))
    sizes = sorted((len(chambers) for _, chambers in groups if len(chambers) > 1), reverse=True)
    degrees = [0] * (spec["rows"] * spec["cols"])
    for u, v in graph_edges:
        degrees[u] += 1
        degrees[v] += 1
    # The grid, rather than the sampled graph, decides the component labels.
    breached_chambers = [chambers for _, chambers in groups if len(chambers) > 1]
    corner_chambers = {(0, 0), (0, spec["cols"] - 1),
                       (spec["rows"] - 1, 0), (spec["rows"] - 1, spec["cols"] - 1)}
    edge_chambers = {(r, c) for r in range(spec["rows"]) for c in range(spec["cols"])
                     if r in (0, spec["rows"] - 1) or c in (0, spec["cols"] - 1)}
    edge_group = any(chambers & edge_chambers for chambers in breached_chambers)
    corner_group = any(chambers & corner_chambers for chambers in breached_chambers)
    cycle_count = len(graph_edges) - sum(len(chambers) - 1 for chambers in breached_chambers)
    return {**spec, "intact": intact, "breached": breached, "horizontal": horizontal,
            "vertical": vertical, "apertures": aperture_sizes, "junctions": junctions,
            "breached_sizes": sizes, "opening_pixels": int(np.count_nonzero(holes)),
            "maximum_degree": max(degrees), "cycle_count": cycle_count,
            "edge_group": edge_group, "corner_group": corner_group}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array and rank")
                _require(np.issubdtype(grid.dtype, np.integer), "grid integer type")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid dimensions")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid colors")
            _require(pair["input"].shape == pair["output"].shape, "shape changed")
            key = _grid_key(pair["input"])
            pair_key = (key, _grid_key(pair["output"]))
            _require(key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, "source pair copied")
            information[split].append(_inspect(pair))
    all_info = information["train"] + information["test"]
    _require(len({(x["intact"], x["breached"]) for x in all_info}) == 1,
             "semantic labels changed within puzzle")
    _require(all(x["horizontal"] + x["vertical"] > 0 for x in all_info), "no chamber breach")
    _require(any(x["horizontal"] for x in information["train"]), "training lacks horizontal passage")
    _require(any(x["vertical"] for x in information["train"]), "training lacks vertical passage")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test, rng)
    intact, breached = [int(v) for v in rng.choice(10, size=2, replace=False)]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_structures = []
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                available = [c for c in range(10) if c not in (intact, breached)]
                if split == "test" and train_structures and pair_rng.random() < .48:
                    structure = int(pair_rng.choice(train_structures))
                elif split == "train" and index == 1:
                    structure = int(pair_rng.choice([c for c in available if c != train_structures[0]]))
                else:
                    structure = int(pair_rng.choice(available))
                backgrounds = [c for c in available if c != structure]
                if 0 in backgrounds and pair_rng.random() < .38:
                    background = 0
                else:
                    background = int(pair_rng.choice(backgrounds))
                required_axis = ("horizontal" if index == 0 else "vertical" if index == 1 else None) if split == "train" else None
                pair = _generate_pair(pair_rng, regime, intact, breached, background, structure, required_axis)
                key = _grid_key(pair["input"])
                if key not in seen_inputs and _pair_digest(pair) not in _SOURCE_DIGESTS:
                    seen_inputs.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_structures.append(structure)
                    break
            else:
                raise ValueError("failed to create distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, lattices, backgrounds, mappings = set(), set(), set(), set()
    multi, junction, split_large, square, rectangles, sparse, dense = 0, 0, 0, 0, 0, 0, 0
    source_like, one_axis, branch, chain, cycle, edge_group, corner_group = 0, 0, 0, 0, 0, 0, 0
    reuse, new_color, overlap, varied_background = 0, 0, 0, 0
    train_edges, test_edges, train_pairs, test_pairs = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair)
                local[split].append(info)
                shapes.add(pair["input"].shape)
                lattices.add((info["rows"], info["cols"], info["cell_h"], info["cell_w"]))
                backgrounds.add(info["background"])
                mappings.add((info["intact"], info["breached"]))
                multi += any(x > 1 for x in info["apertures"])
                junction += info["junctions"] > 0
                split_large += len(info["breached_sizes"]) > 1 and info["breached_sizes"][1] >= 8
                square += info["rows"] == info["cols"]
                rectangles += info["cell_h"] != info["cell_w"]
                edges = info["horizontal"] + info["vertical"]
                sparse += edges <= 5
                dense += edges >= 12
                source_like += (info["rows"], info["cols"], info["cell_h"], info["cell_w"]) in (
                    (5, 5, 5, 5), (5, 5, 4, 4), (4, 4, 4, 4), (7, 7, 3, 3))
                one_axis += info["horizontal"] == 0 or info["vertical"] == 0
                branch += info["maximum_degree"] >= 3
                chain += info["maximum_degree"] <= 2 and 3 <= info["breached_sizes"][0] <= 10
                cycle += info["cycle_count"] > 0
                edge_group += info["edge_group"]
                corner_group += info["corner_group"]
                if split == "train":
                    train_edges += edges
                    train_pairs += 1
                else:
                    test_edges += edges
                    test_pairs += 1
        train_colors = {x["structure"] for x in local["train"]}
        reuse += any(x["structure"] in train_colors for x in local["test"])
        new_color += any(x["structure"] not in train_colors for x in local["test"])
        overlap += bool({x["horizontal"] + x["vertical"] for x in local["train"]}
                        & {x["horizontal"] + x["vertical"] for x in local["test"]})
        varied_background += len({x["background"] for x in local["train"] + local["test"]}) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(shapes) >= 25 and len(lattices) >= 35, "lattice diversity")
    _require(len(backgrounds) >= 7 and len(mappings) >= 20, "palette diversity")
    _require(multi >= 20 and junction >= 5 and split_large >= 3, "damage variation")
    _require(square >= 10 and rectangles >= 10 and source_like >= 3, "geometry variation")
    _require(sparse >= 10 and dense >= 5 and one_axis >= 5, "opening density variation")
    _require(min(branch, chain, cycle, edge_group, corner_group) >= 5, "graph variation")
    _require(reuse >= 5 and new_color >= 5 and overlap >= 5 and varied_background >= 20,
             "within puzzle variation")
    _require(test_edges / test_pairs > train_edges / train_pairs,
             "test regime should be denser on average")
    # Two diagonal background pixels alone are two distinct four-connected groups.
    diagonal = np.array([[0, 1], [1, 0]], dtype=np.int8)
    demo_spec = {"background": 0, "row_lines": np.zeros(2, dtype=bool),
                 "col_lines": np.zeros(2, dtype=bool), "cell_h": 2, "cell_w": 2}
    _require(len(_components(diagonal, demo_spec)) == 2, "diagonal contact joined components")
    diagonal[0, 1] = 0
    _require(len(_components(diagonal, demo_spec)) == 1, "orthogonal path did not join components")
    _require(_puzzles_equal(generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3)),
             "explicit generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(654321), generate_puzzle(654321)),
             "automatic generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(901001, 6, 2), generate_puzzle(901002, 6, 2)),
             "different seeds gave the same puzzle")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                        (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(200000 + i, train, test)
        _require(train is None or len(puzzle["train"]) == train, "explicit train count")
        _require(test is None or len(puzzle["test"]) == test, "explicit test count")
    for i, (train, test) in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                        (2, 2), (3, 4), (4, 4))):
        try:
            generate_puzzle(300000 + i, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
