"""Periodic color pulses on corresponding, geometrically deformed room trees.

Only the standard library and NumPy are used. Ground truth is painted from a
sampled abstract tree; solve reconstructs rooms, edges and correspondence from
input cells. No sampled convention, cache, seed or example is used by solve.
"""
from collections import Counter, deque

import numpy as np


# The order also labels paths: north, east, south, west.
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = DIRS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else DIRS
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    part.append(p)
        parts.append(np.asarray(part, dtype=int))
    return parts


def _bbox(points):
    r, c = points.min(axis=0)
    R, C = points.max(axis=0) + 1
    return int(r), int(c), int(R), int(C)


def _enclosed(walls):
    free = np.pad(~walls, 1, constant_values=True)
    return [p - 1 for p in _components(free)
            if np.all(p > 0) and np.all(p < np.array(free.shape) - 1)]


def _edges(ids, walls):
    """Neighbors share a wall pixel with interiors on its opposite sides."""
    nodes = sorted(set(map(int, ids.flat)) - {-1})
    graph = {i: {} for i in nodes}
    h, w = ids.shape
    for r, c in np.argwhere(walls):
        for d in (0, 1):
            dr, dc = DIRS[d]
            a, b = (r + dr, c + dc), (r - dr, c - dc)
            if not (0 <= a[0] < h and 0 <= b[0] < h
                    and 0 <= a[1] < w and 0 <= b[1] < w):
                continue
            i, j = int(ids[a]), int(ids[b])
            if i < 0 or j < 0 or i == j:
                continue
            for u, direction, v in ((i, (d + 2) % 4, j), (j, d, i)):
                if direction in graph[u] and graph[u][direction] != v:
                    raise ValueError("two neighbors use the same compass port")
                graph[u][direction] = v
    return graph


def _paths(graph, root):
    paths = {root: ()}
    queue = deque([root])
    while queue:
        i = queue.popleft()
        for d, j in sorted(graph[i].items()):
            if j not in paths:
                paths[j] = paths[i] + (d,)
                queue.append(j)
    if len(paths) != len(graph) or sum(map(len, graph.values())) != 2 * (len(graph) - 1):
        raise ValueError("chamber adjacency must be a connected tree")
    return paths


def _read_figure(crop, bg, wall, tab):
    rooms = _enclosed(crop == wall)
    if len(rooms) < 3:
        raise ValueError("too few enclosed chambers")
    ids = np.full(crop.shape, -1, dtype=int)
    values = {}
    for i, points in enumerate(rooms):
        ids[tuple(points.T)] = i
        colors = np.unique(crop[tuple(points.T)])
        if len(colors) != 1 or int(colors[0]) in (wall, tab):
            raise ValueError("each chamber must have a uniform payload or background")
        values[i] = int(colors[0])
    outside_marks = np.argwhere((crop != bg) & (crop != wall) & (ids < 0))
    if len(outside_marks) != 1 or crop[tuple(outside_marks[0])] != tab:
        raise ValueError("one entrance tab must be outside the chambers")
    r, c = map(int, outside_marks[0])
    root_options = set()
    for dr, dc in DIRS:
        q, p = (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)
        if (0 <= p[0] < crop.shape[0] and 0 <= p[1] < crop.shape[1]
                and crop[q] == wall and ids[p] >= 0):
            root_options.add(int(ids[p]))
    if len(root_options) != 1:
        raise ValueError("the entrance tab must identify one chamber")
    graph = _edges(ids, crop == wall)
    root = next(iter(root_options))
    paths = _paths(graph, root)
    by_path = {path: i for i, path in paths.items()}
    return {"grid": crop, "rooms": rooms, "ids": ids, "values": values,
            "graph": graph, "root": root, "paths": paths, "by_path": by_path}


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if np.any(border != bg):
        raise ValueError("the canvas must have a uniform background border")
    components = _components(grid != bg, diagonal=True)
    if len(components) != 2:
        raise ValueError("expected two separate chamber figures")
    crops = []
    for points in components:
        r, c, R, C = _bbox(points)
        crop = np.full((R - r, C - c), bg, dtype=np.int8)
        crop[points[:, 0] - r, points[:, 1] - c] = grid[tuple(points.T)]
        crops.append(crop)
    palettes = [set(map(int, np.unique(c))) - {bg} for c in crops]
    targets = [i for i, p in enumerate(palettes) if len(p) == 2]
    if len(targets) != 1:
        raise ValueError("the blank target has just walls and one tab")
    target_index = targets[0]
    target_crop, donor_crop = crops[target_index], crops[1 - target_index]
    counts = Counter(map(int, target_crop[target_crop != bg]))
    wall = max(counts, key=counts.get)
    tab = next(c for c in counts if c != wall)
    if counts[tab] != 1 or counts[wall] <= 1:
        raise ValueError("entrance is a singleton outside the wall framework")
    donor = _read_figure(donor_crop, bg, wall, tab)
    target = _read_figure(target_crop, bg, wall, tab)
    if any(v != bg for v in target["values"].values()):
        raise ValueError("target chambers must be blank")
    if donor["by_path"].keys() != target["by_path"].keys():
        raise ValueError("the two rooted compass trees must correspond exactly")
    samples = {path: donor["values"][i] for path, i in donor["by_path"].items()
               if donor["values"][i] != bg}
    color_counts = Counter(samples.values())
    repeated = [color for color, n in color_counts.items() if n > 1]
    if len(repeated) != 1 or color_counts[repeated[0]] != 2:
        raise ValueError("one payload hue must occur in exactly two reference chambers")
    reference = repeated[0]
    reference_paths = sorted((k for k in samples if samples[k] == reference), key=len)
    a, b = map(len, reference_paths)
    if a == b or reference_paths[1][:a] != reference_paths[0]:
        raise ValueError("references must be at different depths on one root path")
    period = b - a
    aligned = [k for k in samples if samples[k] != reference and (len(k) - a) % period == 0]
    if len(aligned) > 1 or (aligned and a <= len(aligned[0]) <= b):
        raise ValueError("at most one on-phase terminal may lie beyond the reference interval")
    terminal = aligned[0] if aligned else None
    signal = {"a": a, "b": b, "period": period, "reference": reference,
              "references": reference_paths, "terminal": terminal, "samples": samples}
    return bg, wall, tab, donor, target, signal


def _paint_from_input(figure, signal, bg):
    result = figure["grid"].copy()
    terminal = signal["terminal"]
    low, high = (0, max(map(len, figure["by_path"]))) if terminal is None else (
        min(signal["a"], len(terminal)), max(signal["b"], len(terminal)))
    color = signal["reference"] if terminal is None else signal["samples"][terminal]
    for path, i in figure["by_path"].items():
        depth = len(path)
        value = signal["samples"].get(path, bg)
        if low <= depth <= high and (depth - signal["a"]) % signal["period"] == 0:
            value = color
        result[tuple(figure["rooms"][i].T)] = value
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible chambers, entrance tabs and donor samples."""
    bg, wall, tab, donor, target, signal = _decode(input_grid)
    return _paint_from_input(target, signal, bg)


def _plan_graph(plan):
    graph = {}
    for r, c in np.argwhere(plan):
        i = int(r * plan.shape[1] + c)
        graph[i] = {}
        for d, (dr, dc) in enumerate(DIRS):
            R, C = r + dr, c + dc
            if 0 <= R < plan.shape[0] and 0 <= C < plan.shape[1] and plan[R, C]:
                graph[i][d] = int(R * plan.shape[1] + C)
    return graph


def _random_plan(rng, count, minimum_depth, branching):
    rows, cols = ((4, 5), (5, 4), (4, 4), (3, 5), (5, 3))[int(rng.integers(5))]
    plan = np.zeros((rows, cols), dtype=bool)
    plan[int(rng.integers(rows)), int(rng.integers(cols))] = True
    for _ in range(count - 1):
        candidates = []
        for r, c in np.argwhere(~plan):
            neighbors = sum(0 <= r + dr < rows and 0 <= c + dc < cols
                            and plan[r + dr, c + dc] for dr, dc in DIRS)
            if neighbors == 1:
                candidates.append((int(r), int(c)))
        if not candidates:
            raise ValueError("tree growth ran out of sites")
        plan[candidates[int(rng.integers(len(candidates)))]] = True
    graph = _plan_graph(plan)
    if branching and max(map(len, graph.values())) < 3:
        raise ValueError("need a branching witness")
    roots = [i for i in graph if len(graph[i]) == 1]
    rng.shuffle(roots)
    for root in roots:
        paths = _paths(graph, root)
        if max(map(len, paths.values())) >= minimum_depth:
            return plan, root, paths
    raise ValueError("tree is too shallow")


def _geometry(rng, plan, fallback=False, target=False):
    rows, cols = plan.shape
    if fallback:
        heights = np.full(rows, 1 if target else 2)
        widths = np.array([1 + (c % 2 if target else 0) for c in range(cols)])
    else:
        heights = rng.choice((1, 1, 2, 2, 3), size=rows)
        widths = rng.choice((1, 1, 2, 2, 3), size=cols)
    ys, xs = np.r_[0, np.cumsum(heights + 1)], np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.argwhere(plan):
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = int(r * cols + c)
    occupied = labels >= 0
    near = np.zeros(labels.shape, dtype=bool)
    padded = np.pad(occupied, 1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    if fallback and not target:
        # The first top-row room after the entrance starts one pixel lower.
        # Its horizontal ports remain visible on the second interior row.
        # This gives the finite fallback a correspondence-vs-scan-order witness.
        labels[int(ys[0]) + 1, int(xs[1]) + 1:int(xs[2])] = -2
    if not fallback:
        graph = _plan_graph(plan)
        for i in graph:
            points = np.argwhere(labels == i)
            r, c, R, C = _bbox(points)
            # Inset a free side. This can reorder room discovery without
            # changing any above/below or left/right graph port.
            options = [d for d in range(4) if d not in graph[i]
                       and (R - r if d % 2 == 0 else C - c) >= 2]
            if options and rng.random() < .5:
                d = options[int(rng.integers(len(options)))]
                cut = [(r, x) for x in range(c, C)] if d == 0 else (
                    [(R - 1, x) for x in range(c, C)] if d == 2 else (
                    [(x, C - 1) for x in range(r, R)] if d == 1 else [(x, c) for x in range(r, R)]))
                for p in cut:
                    labels[p] = -2
            points = np.argwhere(labels == i)
            r, c, R, C = _bbox(points)
            if R - r >= 2 and C - c >= 2 and rng.random() < .65:
                p = ((r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1))[int(rng.integers(4))]
                trial = labels == i
                trial[p] = False
                if len(_components(trial)) == 1:
                    labels[p] = -2
    ids = np.where(labels >= 0, labels, -1)
    if _edges(ids, labels == -2) != _plan_graph(plan):
        raise ValueError("local inset lost a graph edge")
    # No unlabeled accidental cavity is allowed to become an extra room.
    rooms = _enclosed(labels == -2)
    if len(rooms) != int(plan.sum()) or any(len(np.unique(labels[tuple(p.T)])) != 1
                                         or labels[tuple(p[0])] < 0 for p in rooms):
        raise ValueError("geometry has an accidental cavity")
    return np.pad(labels, 2, constant_values=-1)


def _figure(rng, labels, root, bg, wall, tab, colors=None):
    options = []
    h, w = labels.shape
    for r, c in np.argwhere(labels == root):
        for dr, dc in DIRS:
            q, m = (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)
            if not (0 <= m[0] < h and 0 <= m[1] < w):
                continue
            if labels[q] != -2 or labels[m] != -1:
                continue
            candidates = set()
            for er, ec in DIRS:
                v, u = (m[0] + er, m[1] + ec), (m[0] + 2 * er, m[1] + 2 * ec)
                if (0 <= u[0] < h and 0 <= u[1] < w
                        and labels[v] == -2 and labels[u] >= 0):
                    candidates.add(int(labels[u]))
            if candidates == {root}:
                options.append(m)
    if not options:
        raise ValueError("entrance has no unambiguous external tab site")
    grid = np.full(labels.shape, bg, dtype=np.int8)
    grid[labels == -2] = wall
    if colors:
        for i, color in colors.items():
            grid[labels == i] = color
    grid[options[int(rng.integers(len(options)))]] = tab
    r, c, R, C = _bbox(np.argwhere(grid != bg))
    grid, labels = grid[r:R, c:C].copy(), labels[r:R, c:C].copy()
    # Tabs that would lie in an exterior-looking but enclosed pocket fail here.
    _read_figure(grid, bg, wall, tab)
    return grid, labels


def _signal(rng, plan, paths, period, mode, direction, witness=False, noise=False):
    depth = max(map(len, paths.values()))
    geometries = []
    for a in range(depth - period + 1):
        b = a + period
        phase = list(range(a % period, depth + 1, period))
        if len(phase) < 3:
            continue
        if mode == "boundary":
            if witness and not (a % period != 0 and a >= period and b + period <= depth):
                continue
            geometries.append((a, b, None))
        else:
            for t in phase:
                if ((direction > 0 and t > b) or (direction < 0 and t < a)):
                    lo, hi = min(a, t), max(b, t)
                    if any(x < lo or x > hi for x in phase):
                        geometries.append((a, b, t))
    candidates = []
    cols = plan.shape[1]
    shell_counts = Counter(map(len, paths.values()))
    for a, b, t in geometries:
        for j in paths:
            if len(paths[j]) != b:
                continue
            ancestors = [i for i in paths if len(paths[i]) == a and paths[j][:a] == paths[i]]
            for i in ancestors:
                if witness:
                    ir, ic = divmod(i, cols)
                    jr, jc = divmod(j, cols)
                    if ir == jr or ic == jc:
                        continue
                    if not any(n > 1 and (d - a) % period == 0 for d, n in shell_counts.items()):
                        continue
                candidates.append((i, j, t))
    if not candidates:
        raise ValueError("no informative reference placement fits this tree")
    i, j, terminal_depth = candidates[int(rng.integers(len(candidates)))]
    a, b = len(paths[i]), len(paths[j])
    terminal = None
    if terminal_depth is not None:
        options = [k for k in paths if len(paths[k]) == terminal_depth]
        terminal = options[int(rng.integers(len(options)))]
    offphase = [k for k in paths if (len(paths[k]) - a) % period]
    noise_nodes = []
    if offphase and noise:
        n = int(rng.integers(1, min(3, len(offphase)) + 1))
        noise_nodes = list(map(int, rng.choice(offphase, n, replace=False)))
    return i, j, terminal, noise_nodes


def _embed(rng, donor, target, bg):
    # Layout sampling is independent of signal direction and graph orientation.
    dh, dw = donor.shape
    th, tw = target.shape
    options = []
    for gap in range(1, 4):
        if dw + tw + gap + 2 <= 30 and max(dh, th) + 2 <= 30:
            options.append(("side", gap))
        if dh + th + gap + 2 <= 30 and max(dw, tw) + 2 <= 30:
            options.append(("stack", gap))
    if not options:
        raise ValueError("figures do not fit together")
    layout, gap = options[int(rng.integers(len(options)))]
    first, second = (donor, target) if rng.random() < .5 else (target, donor)
    if layout == "side":
        height = max(first.shape[0], second.shape[0]) + 2
        width = first.shape[1] + second.shape[1] + gap + 2
    else:
        height = first.shape[0] + second.shape[0] + gap + 2
        width = max(first.shape[1], second.shape[1]) + 2
    height += int(rng.integers(min(4, 30 - height) + 1))
    width += int(rng.integers(min(4, 30 - width) + 1))
    canvas = np.full((height, width), bg, dtype=np.int8)
    if layout == "side":
        extra = width - first.shape[1] - second.shape[1] - gap - 2
        left = 1 + int(rng.integers(extra + 1))
        placements = ((first, int(rng.integers(1, height - first.shape[0])), left),
                      (second, int(rng.integers(1, height - second.shape[0])), left + first.shape[1] + gap))
    else:
        extra = height - first.shape[0] - second.shape[0] - gap - 2
        top = 1 + int(rng.integers(extra + 1))
        placements = ((first, top, int(rng.integers(1, width - first.shape[1]))),
                      (second, top + first.shape[0] + gap, int(rng.integers(1, width - second.shape[1]))))
    for g, r, c in placements:
        canvas[r:r + g.shape[0], c:c + g.shape[1]] = g
    return canvas


def _make_pair(rng, period, mode, direction, witness=False, noise=False, force_fallback=False):
    # Attempts are bounded; the final tree is a bent path with a side branch.
    fallback_plan = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=bool)
    for attempt in ((99,) if force_fallback else range(100)):
        fallback = attempt == 99
        palette = list(map(int, rng.permutation(10)))
        bg, wall, tab, reference = palette[:4]
        try:
            if fallback:
                plan, root = fallback_plan, 0
                paths = _paths(_plan_graph(plan), root)
            else:
                minimum_depth = max(7 if witness else 6, 3 * period)
                count = int(rng.integers(max(8, minimum_depth + 2), 15))
                plan, root, paths = _random_plan(rng, count, minimum_depth, witness)
                rotation = int(rng.integers(4))
                root_position = np.zeros(plan.shape, dtype=bool)
                root_position[divmod(root, plan.shape[1])] = True
                plan, root_position = np.rot90(plan, rotation), np.rot90(root_position, rotation)
                if rng.random() < .5:
                    plan, root_position = plan[:, ::-1], root_position[:, ::-1]
                rr, cc = np.argwhere(root_position)[0]
                root = int(rr * plan.shape[1] + cc)
                paths = _paths(_plan_graph(plan), root)
            i, j, terminal, extras = _signal(rng, plan, paths, period, mode, direction, witness, noise)
            colors = {i: reference, j: reference}
            next_color = 4
            if terminal is not None:
                colors[terminal] = palette[next_color]
                next_color += 1
            for k in extras:
                colors[k] = palette[next_color]
                next_color += 1
            dlabels = _geometry(rng, plan, fallback)
            tlabels = _geometry(rng, plan, fallback, target=True)
            donor, dlabels = _figure(rng, dlabels, root, bg, wall, tab, colors)
            target, tlabels = _figure(rng, tlabels, root, bg, wall, tab)
            expected = target.copy()
            a, b = len(paths[i]), len(paths[j])
            lo, hi = (0, max(map(len, paths.values()))) if terminal is None else (
                min(a, len(paths[terminal])), max(b, len(paths[terminal])))
            paint = reference if terminal is None else colors[terminal]
            for k, path in paths.items():
                color = colors.get(k, bg)
                if lo <= len(path) <= hi and (len(path) - a) % period == 0:
                    color = paint
                expected[tlabels == k] = color
            inp = _embed(rng, donor, target, bg)
            pair = {"input": inp, "output": expected}
            measures = _measure_pair(pair)
            if not measures["deformed"] or not measures["omit_b_changes"]:
                raise ValueError("need a visibly different target geometry")
            if witness and not (measures["branch_paint"] and measures["left_extension"]
                                and measures["right_extension"] and measures["nonaligned_references"]
                                and measures["scan_shortcut_changes"]):
                raise ValueError("missing a branching, two-sided reference witness")
            return pair
        except (ValueError, AssertionError):
            if fallback:
                raise
    raise RuntimeError("bounded construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    curriculum = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index == 0
            if witness:
                period, mode, direction, noise = 2, "boundary", 1, True
            elif split == "train" and index == 1:
                period, mode, direction, noise = 1, "terminal", 1, False
            elif split == "train":
                mode = "boundary" if rng.random() < .5 else "terminal"
                period = 1 if curriculum == 2 and mode == "terminal" else int(rng.integers(1, 3))
                direction, noise = 1, bool(rng.random() < .7)
            else:
                period = 3 if curriculum == 0 else 2
                mode = ("boundary" if rng.random() < .5 else "terminal") if curriculum == 0 else "terminal"
                direction = -1 if curriculum == 1 else (-1 if rng.random() < .5 else 1)
                noise = True
            for retry in range(16):
                pair = _make_pair(rng, period, mode, direction, witness, noise)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure_pair(pair):
    inp, output = _grid(pair["input"]), _grid(pair["output"])
    bg, wall, tab, donor, target, signal = _decode(inp)
    expected = _paint_from_input(target, signal, bg)
    assert np.array_equal(output, expected), "incorrect graph pulse or correspondence"
    assert output.shape == target["grid"].shape, "output must be the tight target crop"
    mask = target["ids"] >= 0
    assert np.array_equal(output[~mask], target["grid"][~mask]), "walls, tab and exterior must be preserved"
    assert np.all(target["grid"][mask] == bg)
    for points in target["rooms"]:
        assert len(np.unique(output[tuple(points.T)])) == 1, "a chamber must be filled uniformly"
    direct_transfer = target["grid"].copy()
    for path, i in target["by_path"].items():
        direct_transfer[tuple(target["rooms"][i].T)] = signal["samples"].get(path, bg)
    omit_a = not np.array_equal(output, direct_transfer)
    omit_b = not np.array_equal(output, _paint_from_input(donor, signal, bg))
    donor_completed = _paint_from_input(donor, signal, bg)
    scan_transfer = target["grid"].copy()
    for source, destination in zip(donor["rooms"], target["rooms"]):
        scan_transfer[tuple(destination.T)] = donor_completed[tuple(source[0])]
    assert omit_a, "each generated pair must expose periodic completion/recoloring"
    terminal = signal["terminal"]
    D = max(map(len, target["by_path"]))
    phases = [d for d in range(D + 1) if (d - signal["a"]) % signal["period"] == 0]
    span = (0, D) if terminal is None else (min(signal["a"], len(terminal)), max(signal["b"], len(terminal)))
    stopped = terminal is not None and any(d < span[0] or d > span[1] for d in phases)
    if stopped:
        for path, i in target["by_path"].items():
            if len(path) in phases and not span[0] <= len(path) <= span[1]:
                assert np.all(output[tuple(target["rooms"][i].T)] == bg), "terminal must stop the pulse"
    extras = [p for p in signal["samples"] if p not in signal["references"] and p != terminal]
    for path in extras:
        assert (len(path) - signal["a"]) % signal["period"]
        points = target["rooms"][target["by_path"][path]]
        assert np.all(output[tuple(points.T)] == signal["samples"][path]), "off-phase colors copy literally"
    shells = Counter(map(len, target["by_path"]))
    refs = [donor["rooms"][donor["by_path"][p]] for p in signal["references"]]
    boxes = [_bbox(p) for p in refs]
    nonaligned = ((boxes[0][2] <= boxes[1][0] or boxes[1][2] <= boxes[0][0])
                  and (boxes[0][3] <= boxes[1][1] or boxes[1][3] <= boxes[0][1]))
    changed_areas = [len(target["rooms"][target["by_path"][p]]) - len(donor["rooms"][i])
                     for p, i in donor["by_path"].items()]
    return {"rooms": len(target["rooms"]), "depth": D, "period": signal["period"],
            "phase": signal["a"] % signal["period"],
            "mode": "boundary" if terminal is None else "terminal", "stopped": stopped,
            "direction": 0 if terminal is None else (1 if len(terminal) > signal["b"] else -1),
            "extras": len(extras), "background": bg, "wall": wall, "tab": tab,
            "branch_paint": any(n > 1 and d in phases and span[0] <= d <= span[1]
                                for d, n in shells.items()),
            "left_extension": terminal is None and min(phases) < signal["a"],
            "right_extension": terminal is None and max(phases) > signal["b"],
            "nonaligned_references": nonaligned, "omit_a_changes": omit_a, "omit_b_changes": omit_b,
            "scan_shortcut_changes": not np.array_equal(scan_transfer, output),
            "deformed": not np.array_equal(donor["grid"] == wall, target["grid"] == wall),
            "scan_order_changed": any(i != target["by_path"][p] for p, i in donor["by_path"].items()),
            "area_signs": tuple(sorted(set(int(np.sign(x)) for x in changed_areas))),
            "nonrectangular": sum(len(p) != (_bbox(p)[2] - _bbox(p)[0]) * (_bbox(p)[3] - _bbox(p)[1])
                                  for p in donor["rooms"] + target["rooms"]),
            "topology": tuple(sorted(target["by_path"]))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, measures = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "inputs must be distinct within a puzzle"
            seen.add(key)
            m = _measure_pair(pair)
            assert m["deformed"] and m["omit_a_changes"] and m["omit_b_changes"]
            measures[split].append(m)
    train, tests = measures["train"], measures["test"]
    assert any(m["period"] > 1 and m["left_extension"] and m["right_extension"]
               and m["branch_paint"] and m["extras"] and m["nonaligned_references"]
               and m["phase"] != 0 and m["scan_shortcut_changes"] for m in train)
    assert any(m["period"] == 1 and m["stopped"] for m in train)
    # Check the curriculum from visible features, rather than its hidden label.
    assert (min(m["period"] for m in tests) > max(m["period"] for m in train)
            or (all(m["direction"] >= 0 for m in train) and all(m["direction"] < 0 for m in tests))
            or (all(m["mode"] != "terminal" or m["period"] == 1 for m in train)
                and all(m["mode"] == "terminal" and m["period"] > 1 and m["extras"] for m in tests)))


def self_test():
    """Bounded semantic fixtures, 42-seed variation and API validation."""
    # Hand-assigned answers: a nine-room bent spine has a two-room side branch.
    # Fixture 1 copies an off-phase room and paints BOTH depth-five chambers.
    # Fixture 2 has its depth-six terminal on the side branch, yet also paints
    # the other depth-six chamber, while the on-phase entrance stays blank.
    fixtures = (
        ("00000000000000000000000/00500000000050000000000/03333333000333333330000/03032303000303003030000/03333333000333333330000/00000323000000003030000/00033333330003333333330/00030363030003003030030/00033333330003003030030/00030303030003333333330/00033303330003003030030/00000000000003333033330/00000000000000000000000",
         "05000000000/33333333000/30322303000/33333333000/00000323000/00333333333/00322363223/00322363223/00333333333/00300303003/00333303333"),
        ("00000000000000000000000/00500000000050000000000/03333333000333333330000/03030323000303003030000/03333333000333333330000/00000363000000003030000/00033333330003333333330/00030323030003003030030/00033333330003003030030/00037303030003333333330/00033303330003003030030/00000000000003333033330/00000000000000000000000",
         "05000000000/33333333000/30300373000/33333333000/00000363000/00333333333/00300373003/00300373003/00333333333/00377303773/00333303333"))
    for inp, out in fixtures:
        inp = np.array([list(map(int, row)) for row in inp.split("/")])
        out = np.array([list(map(int, row)) for row in out.split("/")])
        assert np.array_equal(solve(inp), out), "hand-assigned shell or stopping fixture failed"
    fallback_rng = np.random.default_rng(42)
    for args in ((2, "boundary", 1, True, True), (1, "terminal", 1, False, False),
                 (2, "terminal", -1, False, True), (2, "terminal", 1, False, True),
                 (3, "boundary", 1, False, True), (3, "terminal", -1, False, True),
                 (3, "terminal", 1, False, True)):
        _make_pair(fallback_rng, *args, force_fallback=True)
    puzzles, features, shapes, normalized = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                features.append(_measure_pair(pair))
                shapes.add((pair["input"].shape, pair["output"].shape))
                roles = {}
                normalized.add((pair["input"].shape, tuple(roles.setdefault(int(c), len(roles))
                                                          for c in pair["input"].flat)))
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 50 and len(normalized) >= 150
    assert len({m["topology"] for m in features}) >= 40
    assert len({m["rooms"] for m in features}) >= 5
    assert {m["period"] for m in features} == {1, 2, 3}
    assert {m["direction"] for m in features} == {-1, 0, 1}
    assert {m["background"] for m in features} == set(range(10))
    assert all(any(m[role] == c for m in features) for role in ("wall", "tab") for c in range(10))
    assert any(m["scan_order_changed"] for m in features)
    assert any(m["nonrectangular"] for m in features)
    assert any(-1 in m["area_signs"] and 1 in m["area_signs"] for m in features)
    assert all(m["omit_a_changes"] and m["omit_b_changes"] for m in features)
    # Metamorphic checks exercise the input parser without regeneration.
    example = generate_puzzle(123, 2, 1)
    for pair in example["train"] + example["test"]:
        for turns in range(4):
            inp, out = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(solve(9 - inp), 9 - out)
            assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
    repeat = generate_puzzle(0)
    assert puzzles[0] == tuple(_key(p["input"]) for s in ("train", "test") for p in repeat[s])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(400 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupted = generate_puzzle(123, 2, 1)
    corrupted["test"][0]["output"] = corrupted["test"][0]["output"].copy()
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("a corrupted target was accepted")
