"""Counted terminal routing: unary stack transfer selects a maze connection.

All solving conventions are input-local or fixed here. No parent generator,
example bank, file access, latent generation parameter, or learned state is used.
"""
from collections import deque
import hashlib
import json

import numpy as np


SOURCE, DESTINATION, ROUTE, PROCESSED, PORT = 2, 3, 4, 5, 6
PALETTE = (0, 1, 7, 8, 9)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a


def _neighbors(cell, shape):
    y, x = cell
    h, w = shape
    for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= yy < h and 0 <= xx < w:
            yield yy, xx


def _bfs(adjacency, start, goal, blocked=None):
    """Shortest route and number of minima, saturated at two (never tie-break)."""
    distance, ways, parent = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adjacency.get(u, ()):
            if v == blocked:
                continue
            if v not in distance:
                distance[v] = distance[u] + 1
                ways[v] = ways[u]
                parent[v] = u
                queue.append(v)
            elif distance[v] == distance[u] + 1:
                ways[v] = min(2, ways[v] + ways[u])
    if goal not in distance:
        return [], 0
    path, p = [goal], goal
    while p != start:
        p = parent[p]
        path.append(p)
    return path[::-1], ways[goal]


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells, queue = [], [first]
        while queue:
            u = queue.pop()
            cells.append(u)
            for v in _neighbors(u, mask.shape):
                if v in remaining:
                    remaining.remove(v)
                    queue.append(v)
        result.append(cells)
    return result


def _parse(input_grid):
    a = _grid(input_grid)
    bg, wall = int(a[0, 0]), int(a[-1, 0])
    _require(bg in PALETTE and wall in PALETTE and bg != wall, "bad background/wall")
    _require(np.all(a[0] == bg) and np.all(a[-1] == wall), "bad boundary")
    _require(not np.any((a == ROUTE) | (a == PROCESSED)), "output colors in input")
    ports = [tuple(map(int, p)) for p in np.argwhere(a == PORT)]
    _require(len(ports) >= 3 and len({p[0] for p in ports}) == 1, "bad ports")
    level = ports[0][0]
    _require(6 <= level < a.shape[0] - 3, "bad port level")
    xs = sorted(p[1] for p in ports)
    _require(all(b - c >= 4 for c, b in zip(xs, xs[1:])), "overlapping lanes")
    _require(np.all(a[level - 1] == wall), "missing solid collar")
    _require(set(map(int, a[level])) == {wall, PORT}, "bad port boundary")
    _require(np.all(a[level + 1:, 0] == wall) and np.all(a[level + 1:, -1] == wall), "open maze edge")
    _require(set(map(int, np.unique(a[level + 1:]))) <= {bg, wall}, "non-maze content below ports")
    control_y = level - 3
    _require(np.all(a[control_y + 1] == bg), "controls need separation")
    _require(np.all(a[control_y - 1] == bg), "tokens need separation")
    controls = {}
    for role in (SOURCE, DESTINATION):
        cells = [tuple(map(int, p)) for p in np.argwhere(a == role)]
        _require(len(cells) in (2, 3) and {y for y, _ in cells} == {control_y}, "bad control bar")
        columns = sorted(x for _, x in cells)
        _require(columns == list(range(columns[0], columns[-1] + 1)), "broken control bar")
        center = columns[0] + len(columns) // 2
        _require(center in xs, "control has no aligned port")
        controls[role] = (center, columns)
    _require(controls[SOURCE][0] != controls[DESTINATION][0], "same source and destination")
    _require(np.all(np.isin(a[control_y], (bg, SOURCE, DESTINATION))), "stray control content")
    payload = (a[:control_y - 1] != bg)
    _require(not np.any(a[:control_y - 1] == wall), "wall in stack region")
    stacks = {x: [] for x in xs}
    for cells in _components(payload):
        yy, xx = zip(*cells)
        y, x, h, w = min(yy), min(xx), max(yy) - min(yy) + 1, max(xx) - min(xx) + 1
        center = x + w // 2
        _require(h in (1, 2) and w in (2, 3) and center in stacks, "bad token")
        block = a[y:y + h, x:x + w]
        mask = block != bg
        colors = set(map(int, block[mask]))
        _require(len(colors) == 1 and colors <= set(PALETTE) - {bg, wall}, "bad token color")
        stacks[center].append({"y": y, "x": x, "mask": mask.copy(), "color": colors.pop()})
    common = None
    for center, tokens in stacks.items():
        _require(bool(tokens), "port without stack")
        tokens.sort(key=lambda t: t["y"])
        mask = tokens[0]["mask"]
        h, w = mask.shape
        _require(all(t["color"] == tokens[0]["color"] and np.array_equal(t["mask"], mask)
                     and t["x"] == tokens[0]["x"] for t in tokens), "nonrepeated stack")
        _require(tokens[-1]["y"] + h == level - 4, "unequal baseline")
        _require(all(b["y"] - c["y"] == h + 1 for c, b in zip(tokens, tokens[1:])), "bad pitch")
        _require(common is None or np.array_equal(common, mask), "pair must share token shape")
        common = mask
        for role, (cx, columns) in controls.items():
            if cx == center:
                _require(columns == list(range(tokens[0]["x"], tokens[0]["x"] + w)), "bar/token span mismatch")
    source = controls[SOURCE][0]
    destination = controls[DESTINATION][0]
    return {"grid": a, "bg": bg, "wall": wall, "level": level, "stacks": stacks,
            "source": source, "destination": destination, "control_y": control_y,
            "mask": common, "ports": [(level, x) for x in xs]}


def _route_data(parsed, source=None):
    a, level = parsed["grid"], parsed["level"]
    start = (level, parsed["source"] if source is None else source)
    goal = (level, parsed["destination"])
    allowed = {tuple(map(int, p)) for p in np.argwhere(a == parsed["bg"])}
    allowed.update((start, goal))
    adjacency = {p: [q for q in _neighbors(p, a.shape) if q in allowed] for p in allowed}
    path, ways = _bfs(adjacency, start, goal)
    _require(ways == 1 and len(path) > 2, "route must exist and be uniquely shortest")
    return path, adjacency


def _added_cells(parsed):
    source = parsed["stacks"][parsed["source"]]
    destination = parsed["stacks"][parsed["destination"]]
    token = destination[0]
    pitch = token["mask"].shape[0] + 1
    cells = []
    for k in range(1, len(source) + 1):
        y = token["y"] - k * pitch
        _require(y >= 1, "insufficient append headroom")
        for dy, dx in np.argwhere(token["mask"]):
            point = y + int(dy), token["x"] + int(dx)
            _require(parsed["grid"][point] == parsed["bg"], "append collision")
            cells.append(point)
    return cells, token["color"]


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    path, _ = _route_data(p)  # Always the original input graph.
    added, color = _added_cells(p)
    _require(not set(added).intersection(path), "route and append overlap")
    output = p["grid"].copy()
    for cell in path[1:-1]:
        output[cell] = ROUTE
    for cell in added:
        output[cell] = color
    for token in p["stacks"][p["source"]]:
        h, w = token["mask"].shape
        output[token["y"]:token["y"] + h, token["x"]:token["x"] + w][token["mask"]] = PROCESSED
    output[p["control_y"], np.isin(output[p["control_y"]], (SOURCE, DESTINATION))] = p["bg"]
    return output


def _longer_alternative(path, adjacency):
    # Remove a route cell; a surviving route is necessarily strictly longer
    # when the original minimum is unique. Ports' one-cell leads are bridges.
    for cell in path[1:-1]:
        other, _ = _bfs(adjacency, path[0], path[-1], blocked=cell)
        if other:
            return len(other) > len(path)
    return False


def _tree_maze(rng, rows, columns):
    adjacency = {i: [] for i in range(rows * columns)}
    seen, stack = {int(rng.integers(rows * columns))}, []
    stack.append(next(iter(seen)))
    while stack:
        u = stack[-1]
        neighbors = [y * columns + x for y, x in _neighbors(divmod(u, columns), (rows, columns))
                     if y * columns + x not in seen]
        if not neighbors:
            stack.pop()
            continue
        v = int(rng.choice(neighbors))
        adjacency[u].append(v)
        adjacency[v].append(u)
        seen.add(v)
        stack.append(v)
    return adjacency


def _make_maze(rng, rows, columns, source, destination, unmarked):
    """Build fresh graph geometry; no parent generator is called or chained."""
    for _ in range(48):
        adjacency = _tree_maze(rng, rows, columns)
        closed = [(u, v) for u in adjacency for y, x in _neighbors(divmod(u, columns), (rows, columns))
                  if (v := y * columns + x) > u and v not in adjacency[u]]
        rng.shuffle(closed)
        found = False
        for u, v in closed:
            adjacency[u].append(v)
            adjacency[v].append(u)
            path, ways = _bfs(adjacency, source, destination)
            if ways == 1 and len(path) >= abs(source - destination) + 3 and _longer_alternative(path, adjacency):
                found = True
                break
            adjacency[u].remove(v)
            adjacency[v].remove(u)
        if not found:
            continue
        # A second route endpoint must be a valid counterfactual source, too.
        if not any(_bfs(adjacency, u, destination)[1] == 1 for u in unmarked):
            continue
        # Optional further cycles change connectivity; never admit a shortest tie.
        for u, v in closed[:int(rng.integers(0, 4))]:
            if v in adjacency[u]:
                continue
            adjacency[u].append(v)
            adjacency[v].append(u)
            changed_path, changed_ways = _bfs(adjacency, source, destination)
            if changed_ways != 1 or len(changed_path) < abs(source - destination) + 3:
                adjacency[u].remove(v)
                adjacency[v].remove(u)
        if not any(_bfs(adjacency, u, destination)[1] == 1 for u in unmarked):
            continue
        wall = np.ones((2 * rows + 1, 2 * columns + 1), dtype=bool)
        for u, neighbors in adjacency.items():
            y, x = divmod(u, columns)
            wall[2 * y + 1, 2 * x + 1] = False
            for v in neighbors:
                yy, xx = divmod(v, columns)
                wall[y + yy + 1, x + xx + 1] = False
        return wall
    raise RuntimeError("bounded maze construction exhausted")


def _mask(rng):
    h, w = int(rng.integers(1, 3)), int(rng.integers(2, 4))
    mask = np.ones((h, w), dtype=bool)
    if h == 2 and rng.random() < 0.7:
        # Remove part of one row while keeping a connected, full-span silhouette.
        row = int(rng.integers(2))
        remove = int(rng.integers(1, w))
        if rng.integers(2):
            mask[row, :remove] = False
        else:
            mask[row, w - remove:] = False
    return mask


def _draw(grid, y, x, color, mask):
    h, w = mask.shape
    grid[y:y + h, x:x + w][mask] = color


def _build_pair(rng, colors, operands, test=False, anchor=None):
    bg, wall_color = colors[:2]
    payload_colors = colors[2:]
    source_count, dest_count = operands
    mask = _mask(rng)
    if source_count + dest_count > 6:
        mask = np.ones((1, int(rng.integers(2, 4))), dtype=bool)
    h, w = mask.shape
    pitch = h + 1
    count = int(rng.integers(3, 6))
    columns = int(rng.integers(max(6, 2 * count - 1), 15))
    # Select separated logical top-row cells for stack ports, with variable gaps.
    spare = columns - (2 * count - 1)
    gaps = rng.multinomial(spare, [1 / (count + 1)] * (count + 1))
    nodes, x = [], int(gaps[0])
    for i in range(count):
        nodes.append(x)
        x += 2 + int(gaps[i + 1])
    # Source ranks alternate across the two arithmetic anchors, breaking a
    # fixed leftmost/rightmost terminal rule even when only two pairs are asked.
    if anchor == 0:
        source_index = 0
    elif anchor == 1:
        source_index = count - 1
    else:
        source_index = int(rng.integers(count))
    destination_index = int(rng.choice([i for i in range(count) if i != source_index]))
    unmarked = [i for i in range(count) if i not in (source_index, destination_index)]
    counts = rng.integers(1, max(4, source_count, dest_count) + 1, size=count).tolist()
    counts[source_index], counts[destination_index] = source_count, dest_count
    # An unmarked twin isolates the causal effect of moving the source bar:
    # same count and color, different routing identity, unchanged append size.
    counts[unmarked[0]] = source_count
    tallest = max(max(counts), source_count + dest_count)
    minimum_level = 5 + h + (tallest - 1) * pitch
    max_rows = min(5, (30 - minimum_level - 1) // 2)
    _require(max_rows >= 3, "operand geometry does not fit")
    # Queries shift to the greatest feasible maze depth. Training already
    # includes cycles, dead ends, detours and all marker meanings.
    rows = max_rows if test else int(rng.integers(3, max_rows + 1))
    maze = _make_maze(rng, rows, columns, nodes[source_index], nodes[destination_index],
                      [nodes[unmarked[0]]])
    level = int(rng.integers(minimum_level, 30 - maze.shape[0] + 1))
    a = np.full((level + maze.shape[0], maze.shape[1]), bg, dtype=np.int8)
    a[level:][maze] = wall_color
    a[level - 1] = wall_color
    bottom_y = level - 4 - h
    token_colors = rng.choice(payload_colors, count).tolist()
    # Matching source/unmarked payloads force spatial, bar-based selection.
    token_colors[unmarked[0]] = token_colors[source_index]
    for i, node in enumerate(nodes):
        center = 2 * node + 1
        left = center - w // 2
        for k in range(counts[i]):
            _draw(a, bottom_y - k * pitch, left, int(token_colors[i]), mask)
        a[level, center] = PORT
        if i in (source_index, destination_index):
            a[level - 3, left:left + w] = SOURCE if i == source_index else DESTINATION
    # Draw ground truth from construction records, separately from solve().
    source, destination = (level, 2 * nodes[source_index] + 1), (level, 2 * nodes[destination_index] + 1)
    allowed = {tuple(map(int, p)) for p in np.argwhere(a == bg)} | {source, destination}
    adjacency = {p: [q for q in _neighbors(p, a.shape) if q in allowed] for p in allowed}
    path, ways = _bfs(adjacency, source, destination)
    _require(ways == 1, "constructed tie")
    b = a.copy()
    for p in path[1:-1]:
        b[p] = ROUTE
    src_left = source[1] - w // 2
    for k in range(source_count):
        _draw(b, bottom_y - k * pitch, src_left, PROCESSED, mask)
    for k in range(dest_count, source_count + dest_count):
        _draw(b, bottom_y - k * pitch, destination[1] - w // 2,
              int(token_colors[destination_index]), mask)
    b[level - 3] = bg
    return {"input": a, "output": b}


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(PALETTE)))
    # Independently varying operands, unequal ratios and unequal results refute
    # doubling, replacement, multiplication, fixed increments and fixed totals.
    options = [(s, d) for s in range(1, 5) for d in range(1, 5) if s + d <= 5]
    comparisons = [(a, b) for a in options for b in options
                   if (a[0] - b[0]) * (a[1] - b[1]) < 0 and sum(a) != sum(b)]
    first, second = comparisons[int(rng.integers(len(comparisons)))]
    query = (max(first[0], second[0]), max(first[1], second[1]))
    later_options = [o for o in options if o != query]
    operands = [first, second]
    while len(operands) < num_train:
        operands.append(later_options[int(rng.integers(len(later_options)))])
    # The first query always combines the largest demonstrated contribution
    # with the largest demonstrated initial destination; each ingredient has
    # already appeared, but this operand combination has not.
    operands.extend([query] + [options[int(rng.integers(len(options)))] for _ in range(num_test - 1)])
    puzzle, seen = {"train": [], "test": []}, set()
    for index, operand in enumerate(operands):
        split = "train" if index < num_train else "test"
        for _ in range(24):
            pair = _build_pair(rng, colors, operand, test=split == "test", anchor=index if index < 2 else None)
            key = _signature(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
        puzzle[split].append(pair)
        seen.add(key)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    p = _parse(pair["input"])
    path, adjacency = _route_data(p)
    source = p["stacks"][p["source"]]
    destination = p["stacks"][p["destination"]]
    return {"shape": p["grid"].shape, "mask": (p["mask"].shape, p["mask"].tobytes()),
            "source": len(source), "destination": len(destination), "stacks": len(p["stacks"]),
            "length": len(path) - 1, "depth": max(y for y, _ in path) - p["level"],
            "background": p["bg"], "wall": p["wall"], "path": path, "adjacency": adjacency,
            "parsed": p}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles, train_features = set(), None, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "shape changed")
            key = _signature(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _features(pair)
            p, path, adjacency = f["parsed"], f["path"], f["adjacency"]
            role = p["bg"], p["wall"]
            _require(roles is None or roles == role, "palette conventions changed")
            roles = role
            _require(np.array_equal(solve(a.copy()), b), "incorrect output")
            _require(np.all(b[a == p["wall"]] == p["wall"]), "wall changed")
            _require(np.all(b[a == PORT] == PORT), "port changed")
            _require(set(map(tuple, np.argwhere(b == ROUTE))) == set(path[1:-1]), "wrong route cells")
            _require(f["depth"] >= 3 and _longer_alternative(path, adjacency), "routing needs a detour and a longer choice")
            # Counterfactual rerouting: move the source bar to an unmarked lane.
            alternatives = []
            for x in p["stacks"]:
                if x in (p["source"], p["destination"]):
                    continue
                try:
                    q, _ = _route_data(p, source=x)
                    _added_cells(dict(p, source=x))
                except ValueError:
                    continue
                alternatives.append(q)
            _require(any(set(q[1:-1]) != set(path[1:-1]) for q in alternatives), "source selection has no route witness")
            if split == "train":
                train_features.append(f)
    ss = [f["source"] for f in train_features]
    ds = [f["destination"] for f in train_features]
    _require(len(set(ss)) >= 2 and len(set(ds)) >= 2, "uninformative arithmetic")
    _require(len({s + d for s, d in zip(ss, ds)}) >= 2, "constant total")
    _require(any(s * ds[0] != ss[0] * d for s, d in zip(ss, ds)), "constant operand ratio")
    rivals = [lambda s, d: 2 * d, lambda s, d: 2 * s, lambda s, d: s * d,
              lambda s, d: max(s, d), lambda s, d: s, lambda s, d: d + 1]
    _require(all(any(rival(s, d) != s + d for s, d in zip(ss, ds)) for rival in rivals), "arithmetic shortcut survives")
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.int8).tobytes())
    return h.hexdigest()


def self_test():
    """42 sequential seeds; retain only small feature sets, never puzzle banks."""
    hashes, shapes, masks, lengths, palettes, stack_counts, operands = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        for pair in puzzle["train"] + puzzle["test"]:
            f = _features(pair)
            shapes.add(f["shape"])
            masks.add(f["mask"])
            lengths.add(f["length"])
            palettes.add((f["background"], f["wall"]))
            stack_counts.add(f["stacks"])
            operands.add((f["source"], f["destination"]))
    _require(len(hashes) == 42 and len(shapes) >= 20 and len(masks) >= 8, "insufficient structural variation")
    _require(len(lengths) >= 20 and len(palettes) >= 10 and stack_counts == {3, 4, 5}, "insufficient maze/palette/count variation")
    _require(len(operands) >= 10, "insufficient arithmetic variation")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(100 + a * 5 + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Pixel-level mutations must be rejected, including an off-route fill.
    puzzle = generate_puzzle(31, 2, 1)
    pair = puzzle["train"][0]
    f = _features(pair)
    p = f["parsed"]
    added, _ = _added_cells(p)
    candidates = [added[0], f["path"][2], p["ports"][0],
                  tuple(map(int, np.argwhere(pair["input"] == p["wall"])[0])),
                  (p["control_y"], p["source"])]
    off = next(cell for cell in f["adjacency"] if cell[0] > p["level"] and cell not in f["path"])
    candidates.append(off)
    token = p["stacks"][p["source"]][0]
    dy, dx = np.argwhere(token["mask"])[0]
    candidates.append((token["y"] + int(dy), token["x"] + int(dx)))
    for cell in candidates:
        previous = int(pair["output"][cell])
        pair["output"][cell] = (previous + 1) % 10
        try:
            validate_generated_puzzle(puzzle)
        except ValueError:
            pass
        else:
            raise AssertionError("output mutation accepted")
        finally:
            pair["output"][cell] = previous
    return {"seeds": 42, "distinct_puzzles": len(hashes), "shapes": len(shapes),
            "token_masks": len(masks), "route_lengths": len(lengths), "palettes": len(palettes),
            "stack_counts": sorted(stack_counts), "operand_pairs": len(operands)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
