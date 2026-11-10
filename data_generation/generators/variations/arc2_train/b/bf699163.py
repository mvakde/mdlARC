"""Generate hollow-ring selection puzzles."""

import numpy as np


_SOURCE_INPUT_TEXTS = {
    "\n".join(rows) for rows in (
        (
            "555555555555555555", "551115555555555555", "551515555555555555",
            "551115555555555555", "555555533355555555", "555555535355555555",
            "555555533355552225", "577757775555552525", "575555575555552225",
            "575444575555555555", "555454555555555555", "575444575555555555",
            "575555555555555555", "575555555555555555", "575555575558885555",
            "555555575558585555", "575555575558885555",
        ),
        (
            "5555555555555555", "5888555555555555", "5858555555522255",
            "5888555555525255", "5555555555522255", "5555555555555555",
            "5555333555555555", "5555353555555555", "5555333577775577",
            "5555555575555555", "5555555555111555", "5555555575151555",
            "5555555575111555", "5666555575555555", "5656555555555555",
            "5666555555555555", "5555555575555555",
        ),
        (
            "55555555752225", "55666555752525", "55656555752225",
            "55666555555555", "55555555775577", "55555555555555",
            "55555555588855", "55555555585855", "55555555588855",
            "55555555555555", "55511155555555", "55515155555555",
            "55511155555555",
        ),
    )
}


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, seed, num_train, num_test):
    num_train, num_test = _count(num_train, "num_train"), _count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _cells(pos):
    r, c = pos
    return {(r + dr, c + dc) for dr in range(3) for dc in range(3)}


def _separated(a, b):
    ar, ac = a
    br, bc = b
    return ar + 3 < br or br + 3 < ar or ac + 3 < bc or bc + 3 < ac


def _contained(pos, box):
    r, c = pos
    r0, c0, r1, c1 = box
    return r0 <= r and r + 2 <= r1 and c0 <= c and c + 2 <= c1


def _category(pos, box):
    r, c = pos
    r0, c0, r1, c1 = box
    ri = r0 <= r and r + 2 <= r1
    ci = c0 <= c and c + 2 <= c1
    if ri and ci:
        return "selected"
    intersects = r <= r1 and r + 2 >= r0 and c <= c1 and c + 2 >= c0
    if intersects:
        return "straddle"
    if ri:
        return "row_only"
    if ci:
        return "col_only"
    return "outside"


def _marker(rng, box, occupied):
    r0, c0, r1, c1 = box
    if rng.random() < 0.27:
        for corner in rng.permutation(4):
            rr = r0 if corner in (0, 1) else r1
            cc = c0 if corner in (0, 2) else c1
            elbow = {(rr, c) for c in range(c0, c1 + 1)} | \
                    {(r, cc) for r in range(r0, r1 + 1)}
            if not elbow & occupied:
                return elbow
    sides = [
        [(r0, c) for c in range(c0, c1 + 1)],
        [(r, c1) for r in range(r0, r1 + 1)],
        [(r1, c) for c in range(c0, c1 + 1)],
        [(r, c0) for r in range(r0, r1 + 1)],
    ]
    sides = [[p for p in side if p not in occupied] for side in sides]
    if any(not side for side in sides):
        return None
    perimeter = set().union(*map(set, sides))
    if len(perimeter) < 7:
        return None
    # Each side supplies an extremum. Side weights and local runs create both
    # long posts and sparse, disconnected fragments without internal dots.
    weights = rng.dirichlet(np.ones(4) * 0.8)
    anchors = [side[int(rng.integers(len(side)))] for side in sides]
    chosen = set(anchors)
    maximum = min(25, len(perimeter))
    if maximum < 9 or rng.random() < 0.28:
        count = int(rng.integers(7, min(9, maximum) + 1))
    else:
        count = int(rng.integers(9, maximum + 1))
    options = list(perimeter - chosen)
    while len(chosen) < count and options:
        scores = []
        for p in options:
            on = [i for i, side in enumerate(sides) if p in side]
            score = sum(weights[i] for i in on)
            distance = min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in chosen)
            score *= 1.7 if distance == 1 else 1.0
            scores.append(score + 0.02)
        probability = np.array(scores, dtype=float)
        probability /= probability.sum()
        index = int(rng.choice(len(options), p=probability))
        chosen.add(options.pop(index))
    return chosen


def _sample_layout(rng, mode, required):
    if mode == "source_like":
        height = int(rng.integers(15, 20))
        width = int(rng.integers(15, 21))
    elif mode == "small_edge":
        height = int(rng.integers(11, 16))
        width = int(rng.integers(12, 17))
    elif mode == "large_edge":
        height = int(rng.integers(18, 27))
        width = int(rng.integers(18, 27))
    else:
        height = int(rng.integers(12, 26))
        width = int(rng.integers(12, 26))

    if rng.random() < 0.14:
        bh, bw = 5, 6
    else:
        bh = int(rng.integers(5, min(12, height) + 1))
        bw = int(rng.integers(5, min(13, width) + 1))
    if mode in ("small_edge", "large_edge"):
        side = int(rng.integers(4))
        r0 = 0 if side == 0 else height - bh if side == 2 else int(rng.integers(height - bh + 1))
        c0 = width - bw if side == 1 else 0 if side == 3 else int(rng.integers(width - bw + 1))
    elif mode == "source_like":
        r0 = int(rng.integers(7, height - bh + 1)) if height - bh >= 7 else height - bh
        c0 = int(rng.integers(width - bw + 1))
    else:
        r0 = int(rng.integers(height - bh + 1))
        c0 = int(rng.integers(width - bw + 1))
    box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
    selected_options = [(r, c) for r in range(r0, r0 + bh - 2)
                        for c in range(c0, c0 + bw - 2)]
    if mode == "source_like":
        interior = [(r, c) for r, c in selected_options
                    if 0 < r and r + 2 < height - 1 and 0 < c and c + 2 < width - 1]
        if interior:
            selected_options = interior
    elif mode in ("small_edge", "large_edge") and rng.random() < 0.58:
        edge_options = [(r, c) for r, c in selected_options
                        if r == 0 or c == 0 or r + 2 == height - 1 or c + 2 == width - 1]
        if edge_options:
            selected_options = edge_options
    selected = selected_options[int(rng.integers(len(selected_options)))]
    positions = [selected]
    count = int(rng.integers(3, min(7, max(3, height * width // 27)) + 1))
    count = max(count, 1 + len(required))
    if count > 7:
        return None
    candidates = [(r, c) for r in range(height - 2) for c in range(width - 2)
                  if not _contained((r, c), box)]
    order = rng.permutation(len(candidates))
    candidates = [candidates[int(i)] for i in order]

    def add(kind):
        for p in candidates:
            if _category(p, box) != kind:
                continue
            if all(_separated(p, q) for q in positions):
                positions.append(p)
                return True
        return False

    for kind in required:
        if not add(kind):
            return None
    sr, sc = selected
    selected_edges = (sr == 0, sc == width - 3, sr == height - 3, sc == 0)
    if any(selected_edges) and len(positions) < count and rng.random() < 0.62:
        same_edge = [p for p in candidates if
                     ((selected_edges[0] and p[0] == 0) or
                      (selected_edges[1] and p[1] == width - 3) or
                      (selected_edges[2] and p[0] == height - 3) or
                      (selected_edges[3] and p[1] == 0)) and
                     all(_separated(p, q) for q in positions)]
        if same_edge:
            positions.append(same_edge[int(rng.integers(len(same_edge)))])
    while len(positions) < count:
        viable = [p for p in candidates if p not in positions
                  and all(_separated(p, q) for q in positions)]
        if not viable:
            return None
        position = viable[int(rng.integers(len(viable)))]
        positions.append(position)
    occupied = set().union(*(_cells(p) for p in positions))
    marker = _marker(rng, box, occupied)
    if marker is None:
        return None
    return height, width, box, positions, marker


def _generate_pair(rng, mode, required, shared_roles, preferred_color=None, avoid_color=None):
    for _ in range(140):
        layout = _sample_layout(rng, mode, required)
        if layout is None:
            continue
        height, width, _box, positions, marker = layout
        if shared_roles is None:
            background, marker_color = [int(x) for x in rng.choice(10, size=2, replace=False)]
        else:
            background, marker_color = shared_roles
        accents = [x for x in range(10) if x not in (background, marker_color)]
        if preferred_color in accents and preferred_color != avoid_color and rng.random() < 0.35:
            selected_color = preferred_color
        else:
            choices = [x for x in accents if x != avoid_color] or accents
            selected_color = int(rng.choice(choices))
        colors = [selected_color]
        for i in range(len(positions) - 1):
            if preferred_color in accents and rng.random() < 0.35:
                colors.append(preferred_color)
            else:
                colors.append(int(rng.choice(accents)))
        grid = np.full((height, width), background, dtype=np.int8)
        for r, c in marker:
            grid[r, c] = marker_color
        for (r, c), color in zip(positions, colors):
            for dr in range(3):
                for dc in range(3):
                    if (dr, dc) != (1, 1):
                        grid[r + dr, c + dc] = color
        sr, sc = positions[0]
        return grid, grid[sr:sr + 3, sc:sc + 3].copy()
    raise ValueError("could not construct a layout")


def _array_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(rng, seed, num_train, num_test)
    # The learned convention is geometric. Some puzzles also retain literal
    # background/marker roles; others teach those roles separately in each pair.
    shared_roles = None
    if rng.random() < 0.6:
        shared_roles = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    used = set()
    train_answer_colors = []
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if split == "train" and index == 0:
                mode, required = "source_like", ("row_only", "col_only")
            elif split == "train" and index == 1:
                mode, required = str(rng.choice(["small_edge", "large_edge", "free"])), ("straddle",)
            else:
                mode = str(rng.choice(["source_like", "small_edge", "large_edge", "free"],
                                      p=[0.22, 0.27, 0.20, 0.31]))
                kinds = ["row_only", "col_only", "straddle", "outside"]
                required = (str(rng.choice(kinds)),) if rng.random() < 0.55 else ()
            preferred = None
            if split == "test" and train_answer_colors and rng.random() < 0.5:
                preferred = int(rng.choice(train_answer_colors))
            elif split == "train" and index > 1 and train_answer_colors and rng.random() < 0.65:
                preferred = int(rng.choice(train_answer_colors))
            avoid = train_answer_colors[0] if split == "train" and index == 1 else None
            for _ in range(70):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = _generate_pair(child, mode, required, shared_roles, preferred, avoid)
                if _array_key(inp) not in used:
                    break
            else:
                raise ValueError("could not construct unique input")
            used.add(_array_key(inp))
            puzzle[split].append({"input": inp, "output": out})
            if split == "train":
                train_answer_colors.append(int(out[0, 0]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_text(grid):
    return "\n".join("".join(str(int(x)) for x in row) for row in grid)


def _analyze_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(values[int(np.argmax(counts))])
    height, width = grid.shape
    rings = []
    ring_cells = set()
    for r in range(height - 2):
        for c in range(width - 2):
            crop = grid[r:r + 3, c:c + 3]
            color = int(crop[0, 0])
            if color == background or int(crop[1, 1]) != background:
                continue
            if all(int(crop[dr, dc]) == color for dr in range(3)
                   for dc in range(3) if (dr, dc) != (1, 1)):
                cells = _cells((r, c)) - {(r + 1, c + 1)}
                assert not ring_cells.intersection(cells)
                rings.append({"box": (r, c, r + 2, c + 2), "color": color})
                ring_cells.update(cells)
    assert 3 <= len(rings) <= 7
    for i, first in enumerate(rings):
        for second in rings[i + 1:]:
            assert _separated(first["box"][:2], second["box"][:2])
    residual = [(r, c) for r in range(height) for c in range(width)
                if int(grid[r, c]) != background and (r, c) not in ring_cells]
    assert len(residual) >= 7
    marker_colors = {int(grid[r, c]) for r, c in residual}
    assert len(marker_colors) == 1
    marker_color = marker_colors.pop()
    assert marker_color not in {ring["color"] for ring in rings}
    marker_box = (min(r for r, c in residual), min(c for r, c in residual),
                  max(r for r, c in residual), max(c for r, c in residual))
    r0, c0, r1, c1 = marker_box
    assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5
    assert all(r in (r0, r1) or c in (c0, c1) for r, c in residual)
    selected = [ring for ring in rings if _contained(ring["box"][:2], marker_box)]
    assert len(selected) == 1
    return {"background": background, "marker_color": marker_color,
            "marker_cells": residual, "marker_box": marker_box,
            "tokens": rings, "selected": selected[0]}


def _validate_generated_puzzle_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[x], list) for x in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _array_key(inp)
            assert key not in seen
            seen.add(key)
            assert _grid_text(inp) not in _SOURCE_INPUT_TEXTS
            info = _analyze_input(inp)
            r0, c0, r1, c1 = info["selected"]["box"]
            assert out.shape == (3, 3) and np.array_equal(out, inp[r0:r1 + 1, c0:c1 + 1])
            assert int(out[1, 1]) == info["background"]
            infos[split].append(info)
    # These demonstrations distinguish full two-axis containment from color,
    # single-axis alignment, and mere overlap. In particular a training
    # straddler is present before any test straddler can matter.
    train_kinds = set()
    for info in infos["train"]:
        train_kinds.update(_category(ring["box"][:2], info["marker_box"])
                           for ring in info["tokens"] if ring is not info["selected"])
    assert {"row_only", "col_only", "straddle"} <= train_kinds
    train_colors = {info["selected"]["color"] for info in infos["train"]}
    assert len(train_colors) >= 2


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, windows, marker_counts, ring_counts = set(), set(), set(), set()
    role_pairs, layouts, categories = set(), set(), set()
    edge_train = edge_test = interior_test = sparse = repeated = 0
    reuse_test = novel_test = 0
    all_inputs = set()
    total_pairs = 0
    metrics = {}

    def hit(name, condition):
        metrics[name] = metrics.get(name, 0) + int(condition)

    def marker_parts(cells):
        remaining = set(cells)
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                r, c = stack.pop()
                neighbors = {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)} & remaining
                remaining -= neighbors
                stack.extend(neighbors)
        return count

    def longest_run(values):
        values = set(values)
        return max((sum(x + i in values for i in range(13)) for x in values), default=0)

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        answer_colors = {int(p["output"][0, 0]) for p in puzzle["train"]}
        puzzle_features = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                grid, out = pair["input"], pair["output"]
                info = _analyze_input(grid)
                r0, c0, r1, c1 = info["selected"]["box"]
                assert np.array_equal(out, grid[r0:r1 + 1, c0:c1 + 1])
                all_inputs.add(_array_key(grid))
                shapes.add(grid.shape)
                windows.add((info["marker_box"][2] - info["marker_box"][0] + 1,
                             info["marker_box"][3] - info["marker_box"][1] + 1))
                marker_counts.add(len(info["marker_cells"]))
                ring_counts.add(len(info["tokens"]))
                role_pairs.add((info["background"], info["marker_color"]))
                layouts.add((grid.shape, info["marker_box"],
                             tuple(sorted(x["box"] for x in info["tokens"]))))
                categories.update(_category(x["box"][:2], info["marker_box"])
                                  for x in info["tokens"] if x is not info["selected"])
                edge = r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1
                mr0, mc0, mr1, mc1 = info["marker_box"]
                marker = info["marker_cells"]
                bounds = (mr1 - mr0 + 1, mc1 - mc0 + 1)
                ring_colors = [x["color"] for x in info["tokens"]]
                kinds = {_category(x["box"][:2], info["marker_box"])
                         for x in info["tokens"] if x is not info["selected"]}
                sides = [
                    {c for r, c in marker if r == mr0},
                    {r for r, c in marker if c == mc1},
                    {c for r, c in marker if r == mr1},
                    {r for r, c in marker if c == mc0},
                ]
                runs = [longest_run(side) for side in sides]
                for name, run in zip(("top", "right", "bottom", "left"), runs):
                    hit("run_" + name, run == max(runs))
                hit("connected", marker_parts(marker) == 1)
                hit("fragmented", marker_parts(marker) > 1)
                hit("window_5x6", bounds == (5, 6))
                hit("sparse_7_8", len(marker) in (7, 8))
                hit("marker_top_7_8", mr0 in (7, 8))
                hit("marker_bottom_edge", mr1 == grid.shape[0] - 1)
                hit("repeated_accents", len(set(ring_colors)) < len(ring_colors))
                hit("straddle", "straddle" in kinds)
                hit("edge_window", mr0 == 0 or mc0 == 0 or
                    mr1 == grid.shape[0] - 1 or mc1 == grid.shape[1] - 1)
                hit("same_edge_distractor", edge and any(
                    (r0 == 0 and x["box"][0] == 0) or
                    (c0 == 0 and x["box"][1] == 0) or
                    (r1 == grid.shape[0] - 1 and x["box"][2] == grid.shape[0] - 1) or
                    (c1 == grid.shape[1] - 1 and x["box"][3] == grid.shape[1] - 1)
                    for x in info["tokens"] if x is not info["selected"]))
                puzzle_features[split].append((grid.shape, bounds, len(marker),
                                               len(info["tokens"]), edge,
                                               info["background"], info["marker_color"]))
                if split == "train":
                    edge_train += edge
                else:
                    edge_test += edge
                    interior_test += not edge
                    reuse_test += info["selected"]["color"] in answer_colors
                    novel_test += info["selected"]["color"] not in answer_colors
                sparse += len(info["marker_cells"]) <= 8
                repeated += len({x["color"] for x in info["tokens"]}) < len(info["tokens"])
        features = puzzle_features["train"] + puzzle_features["test"]
        for i, name in enumerate(("shape", "window", "marker_count", "ring_count")):
            hit("within_" + name, len({x[i] for x in features}) > 1)
        hit("within_color_roles", len({x[5:7] for x in features}) > 1)
        hit("test_smaller", any(t[0][0] < a[0][0] for a in puzzle_features["train"]
                                for t in puzzle_features["test"]))
        hit("test_larger", any(t[0][0] > a[0][0] for a in puzzle_features["train"]
                               for t in puzzle_features["test"]))
        hit("reversed_edge", any(a[4] and not t[4] for a in puzzle_features["train"]
                                for t in puzzle_features["test"]))
        hit("source_edge", any(not a[4] and t[4] for a in puzzle_features["train"]
                              for t in puzzle_features["test"]))
        hit("shared_edge", any(a[4] == t[4] for a in puzzle_features["train"]
                              for t in puzzle_features["test"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(windows) >= 20
    assert len(marker_counts) >= 10 and len(ring_counts) >= 4
    assert len(role_pairs) >= 15 and len(layouts) >= 100
    assert {"row_only", "col_only", "straddle", "outside"} <= categories
    assert edge_train > 0 and edge_test > 0 and interior_test > 0
    assert sparse > 0 and repeated > 0 and reuse_test > 0 and novel_test > 0
    assert len(all_inputs) >= 0.95 * total_pairs
    assert all(metrics["within_" + name] == 42 for name in
               ("shape", "window", "marker_count", "ring_count"))
    assert metrics["within_color_roles"] > 0
    assert all(metrics[name] > 0 for name in (
        "connected", "fragmented", "window_5x6", "sparse_7_8",
        "marker_top_7_8", "marker_bottom_edge", "repeated_accents",
        "straddle", "edge_window", "same_edge_distractor",
        "run_top", "run_right", "run_bottom", "run_left",
        "test_smaller", "test_larger", "reversed_edge", "source_edge", "shared_edge"))
    a = generate_puzzle(7001, 6, 3)
    assert _puzzles_equal(a, generate_puzzle(7001, 6, 3))
    assert _puzzles_equal(generate_puzzle(19), generate_puzzle(19))
    assert not _puzzles_equal(a, generate_puzzle(7002, 6, 3))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                  (None, 4), (2, None), (10, None))):
        p = generate_puzzle(8000 + i, nt, ns)
        validate_generated_puzzle(p)
        if nt is not None:
            assert len(p["train"]) == nt
        if ns is not None:
            assert len(p["test"]) == ns
    for i, kw in enumerate(({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                            {"num_test": 5}, {"num_train": 2, "num_test": 2},
                            {"num_train": 3, "num_test": 4},
                            {"num_train": 2.5, "num_test": 1},
                            {"num_train": 3, "num_test": True})):
        try:
            generate_puzzle(9000 + i, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kw}")
