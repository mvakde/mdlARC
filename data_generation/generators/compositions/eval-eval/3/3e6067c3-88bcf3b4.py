"""Exact-fit dye thread: ordered checkpoints and unary length jointly route a strand.

Only Python's standard library and NumPy are used. The solver reads the input;
it has no access to the seed, sampled routes, examples, or generator state.
"""
from collections import deque
import hashlib

import numpy as np


DIR4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be 2D integer arrays, sizes 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _component(cells, start, directions=DIR4):
    reached, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in reached:
                reached.add(p)
                stack.append(p)
    return reached


def _paths(cells, root):
    """Root an induced four-neighbor tree, returning input-derived paths."""
    parent, depth, queue = {root: None}, {root: 0}, deque([root])
    edge_count = 0
    while queue:
        p = queue.popleft()
        for dr, dc in DIR4:
            q = p[0] + dr, p[1] + dc
            if q not in cells:
                continue
            edge_count += 1
            if q not in parent:
                parent[q] = p
                depth[q] = depth[p] + 1
                queue.append(q)
    assert len(parent) == len(cells), "passage graph is disconnected"
    assert edge_count == 2 * (len(cells) - 1), "passage graph must be a tree"

    def path_to(p):
        path = []
        while p is not None:
            path.append(p)
            p = parent[p]
        return path[::-1]

    return parent, depth, path_to


def _parse(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    assert h >= 10 and w >= 9
    wall, floor = int(g[0, 0]), int(g[1, 1])
    assert wall != floor
    assert np.all(g[0] == wall) and np.all(g[2] == wall)
    assert np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall)
    assert np.all(g[-1] == wall)
    token_x = [c for c in range(1, w - 1) if int(g[1, c]) != floor]
    assert 3 <= len(token_x) <= 6 and token_x[0] >= 2
    assert all(b - a == 2 for a, b in zip(token_x, token_x[1:]))
    key = [int(g[1, c]) for c in token_x]
    assert wall not in key and floor not in key and len(set(key)) >= 2
    remaining_colors = set(map(int, np.unique(g[3:]))) - {wall, floor} - set(key)
    assert len(remaining_colors) == 1, "support must have a distinct geometric role"
    support_color = remaining_colors.pop()
    support = {tuple(map(int, p)) for p in np.argwhere(g == support_color)}
    assert 3 <= len(support) <= 7 and len({c for r, c in support}) == 1
    ys = sorted(r for r, c in support)
    assert ys == list(range(ys[0], ys[-1] + 1)) and ys[0] >= 4
    endpoint = max(support)
    anchors = [(endpoint[0], endpoint[1] + dc) for dc in (-1, 1)
               if 0 <= endpoint[1] + dc < w
               and int(g[endpoint[0], endpoint[1] + dc]) == key[0]]
    assert len(anchors) == 1
    anchor = anchors[0]
    side = 1 if anchor[1] > endpoint[1] else -1
    slack = {tuple(map(int, p)) for p in np.argwhere(g == key[0])
             if 3 <= p[0] <= endpoint[0]}
    assert anchor in slack and len(slack) >= 3
    assert _component(slack, anchor, DIR8) == slack, "slack must be eight-connected"
    assert all((c - endpoint[1]) * side >= 1 for r, c in slack)
    assert all(ys[0] <= r <= endpoint[0] for r, c in slack)
    assert all(int(g[r, c]) in (wall, floor, support_color, key[0])
               for r in range(3, endpoint[0] + 1) for c in range(w))
    # The support's forward half-plane visibly separates resource from route.
    cells = {tuple(map(int, p)) for p in np.argwhere(g != wall)
             if p[0] > endpoint[0]}
    cells.add(anchor)
    assert (anchor[0] + 1, anchor[1]) in cells
    parent, depth, path_to = _paths(cells, anchor)
    markers = {p: int(g[p]) for p in cells - {anchor} if int(g[p]) != floor}
    assert markers and all(c in key for c in markers.values())
    matches, exact, solutions = [], [], []
    candidates = []
    for end in sorted(markers):
        path = path_to(end)
        sequence = [markers[p] for p in path if p in markers]
        candidates.append((end, path, sequence))
        order_ok = sequence == key[1:]
        length_ok = len(path) == len(slack)
        if order_ok:
            matches.append(path)
        if length_ok:
            exact.append(path)
        if order_ok and length_ok:
            solutions.append(path)
    assert len(solutions) == 1, "order and length must give one exact-fit route"
    path = solutions[0]
    marker_indices = [j for j, p in enumerate(path) if p in markers]
    assert len(marker_indices) == len(key) - 1
    assert all(b - a >= 2 for a, b in zip(marker_indices, marker_indices[1:]))
    return dict(grid=g, wall=wall, floor=floor, key=key, support=support,
                support_color=support_color, endpoint=endpoint, anchor=anchor,
                side=side, slack=slack, length=len(slack), cells=cells,
                markers=markers, path=path, matches=matches, exact=exact,
                candidates=candidates, depth=depth, marker_indices=marker_indices)


def _paint(info, *, mode="source"):
    out = info["grid"].copy()
    for p in info["slack"]:
        out[p] = info["floor"]
    current = info["key"][0]
    path = info["path"]
    for j, p in enumerate(path):
        if p in info["markers"]:
            current = info["markers"][p]
            continue  # Checkpoint cell is preserved, and also consumes length.
        color = current
        if mode == "constant":
            color = info["key"][0]
        elif mode == "destination" and j:
            next_markers = [info["markers"][q] for q in path[j:] if q in info["markers"]]
            if next_markers:
                color = next_markers[0]
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the exact-fit route using only its visible input."""
    return _paint(_parse(input_grid))


def _coordinate_axis(rng, count, first, max_gap=3):
    values = [first]
    for _ in range(count - 1):
        values.append(values[-1] + int(rng.integers(2, max_gap + 1)))
    return values


def _carve_tree(rng, rows, cols, root_col):
    """Randomized growing-tree construction; no complete-layout bank."""
    root = (0, root_col)
    reached = {root}
    frontier = []

    def add_edges(p):
        r, c = p
        for dr, dc in DIR4:
            q = r + dr, c + dc
            if 0 <= q[0] < rows and 0 <= q[1] < cols and q not in reached:
                frontier.append((p, q))

    add_edges(root)
    edges = []
    while frontier:
        # Prim-like growth and depth-biased growth combine without changing rule.
        index = len(frontier) - 1 if rng.random() < .22 else int(rng.integers(len(frontier)))
        a, b = frontier.pop(index)
        if b in reached:
            continue
        reached.add(b)
        edges.append((a, b))
        add_edges(b)
    assert len(edges) == rows * cols - 1
    return root, edges


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _choose_routes(rng, cells, anchor, capacity, checkpoint_count, prefer_long):
    _, depth, path_to = _paths(cells, anchor)
    # A terminal can be in the middle of a corridor: the last dye checkpoint
    # is a visible stopping cue, even if unmarked passages continue beyond it.
    groups = {}
    for p in sorted(cells):
        d = depth[p]
        if 12 <= d < capacity:
            groups.setdefault(d, []).append(p)
    levels = [d for d, points in groups.items() if len(points) >= 2]
    rng.shuffle(levels)
    for d in levels:
        points = groups[d]
        pairs = [(a, b) for i, a in enumerate(points) for b in points[i + 1:]]
        rng.shuffle(pairs)
        for a, b in pairs[:60]:
            pa, pb = path_to(a), path_to(b)
            if _common_prefix(pa, pb) > d - 2 * checkpoint_count:
                continue
            other_levels = [t for t in groups if abs(t - d) >= 2
                            and (t < d if prefer_long else t > d)]
            rng.shuffle(other_levels)
            for t in other_levels:
                others = list(groups[t])
                rng.shuffle(others)
                for c in others[:20]:
                    pc = path_to(c)
                    cut = max(_common_prefix(pa, pb), _common_prefix(pa, pc),
                              _common_prefix(pb, pc))
                    if min(len(pa), len(pb), len(pc)) - cut >= 2 * checkpoint_count:
                        # The two equal-length candidates play different roles.
                        if rng.random() < .5:
                            pa, pb = pb, pa
                        if len(pa) == (abs(pa[-1][0] - anchor[0])
                                       + abs(pa[-1][1] - anchor[1]) + 1):
                            pa, pb = pb, pa
                        if len(pa) == (abs(pa[-1][0] - anchor[0])
                                       + abs(pa[-1][1] - anchor[1]) + 1):
                            continue
                        return [pa, pb, pc], cut
    return None


def _slack(rng, anchor, support_length, width, side, length, diagonal):
    # Coordinates r are backward along support; v is distance from attachment.
    allowed = {(r, v) for r in range(-(support_length - 1), 1) for v in range(width)}
    local = {(0, 0)}
    if diagonal:
        allowed = {(r, v) for r in range(-(support_length - 1), 0)
                   for v in range(1, width)}
        local.add((-1, 1))
    style = int(rng.integers(3))
    while len(local) < length:
        frontier = sorted({(r + dr, v + dv) for r, v in local for dr, dv in DIR8
                           if (r + dr, v + dv) in allowed} - local)
        assert frontier, "insufficient slack capacity"
        if style == 0:
            weights = np.array([1 + sum((p[0] + dr, p[1] + dv) in local
                                       for dr, dv in DIR8) ** 2 for p in frontier], dtype=float)
        elif style == 1:
            weights = np.array([1 / (1 + sum((p[0] + dr, p[1] + dv) in local
                                           for dr, dv in DIR8)) ** 2 for p in frontier], dtype=float)
        else:
            weights = np.ones(len(frontier))
        p = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
        local.add(p)
    return {(anchor[0] + r, anchor[1] + side * v) for r, v in local}


def _build_pair(rng, roles, m, pattern, diagonal, prefer_long, intensity):
    wall, floor, support_color = roles
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    # At least two differing tail labels are needed for the order-only foil.
    if pattern == "return":
        tail = [palette[1], palette[2], palette[0]]
    elif pattern == "adjacent":
        tail = [palette[1], palette[1], palette[2]]
    elif pattern == "distinct":
        tail = palette[1:m + 1]
    else:
        tail = [palette[1], palette[2]] + [int(rng.choice(palette[:4])) for _ in range(m - 2)]
    assert len(tail) == m and len(set(tail)) >= 2
    key = [palette[0]] + tail
    wrong_tail = tail.copy()
    differing = [(i, j) for i in range(m) for j in range(i + 1, m) if tail[i] != tail[j]]
    i, j = differing[int(rng.integers(len(differing)))]
    wrong_tail[i], wrong_tail[j] = wrong_tail[j], wrong_tail[i]
    for _ in range(120):
        cols = int(rng.integers(6, 10 if intensity else 9))
        rows = int(rng.integers(4, 7))
        xs = _coordinate_axis(rng, cols, 2)
        width = xs[-1] + 3
        if width > 30:
            continue
        support_length = int(rng.integers(5 if diagonal else 4, 8))
        end_y = 3 + support_length
        ys = _coordinate_axis(rng, rows, end_y + 2)
        height = ys[-1] + 2
        if height > 30:
            continue
        side = int(rng.choice((-1, 1)))
        root_col = 0 if side == 1 else cols - 1
        root_x = xs[root_col]
        anchor = end_y, root_x
        support_x = root_x - side
        chamber_width = int(rng.integers(8, min(13, width - 3) + 1))
        capacity = ((support_length - 1) * (chamber_width - 1) + 1
                    if diagonal else support_length * chamber_width)
        root, edges = _carve_tree(rng, rows, cols, root_col)
        cells = {anchor, (end_y + 1, root_x)}
        for a, b in edges:
            ar, ac, br, bc = ys[a[0]], xs[a[1]], ys[b[0]], xs[b[1]]
            if ar == br:
                cells.update((ar, c) for c in range(min(ac, bc), max(ac, bc) + 1))
            else:
                cells.update((r, ac) for r in range(min(ar, br), max(ar, br) + 1))
        routes = _choose_routes(rng, cells, anchor, capacity, m, prefer_long)
        if routes is None:
            continue
        chosen, cut = routes
        grid = np.full((height, width), wall, dtype=np.int8)
        grid[1, 1:-1] = floor
        start_x = int(rng.integers(2, width - (2 * len(key) - 1)))
        grid[1, start_x:start_x + 2 * len(key):2] = key
        chamber_cols = [root_x + side * v for v in range(chamber_width)]
        assert min(chamber_cols) >= 1 and max(chamber_cols) <= width - 2
        for r in range(4, end_y + 1):
            grid[r, chamber_cols] = floor
            grid[r, support_x] = support_color
        for p in cells:
            grid[p] = floor
        markers = {}
        for path, labels in zip(chosen, (tail, wrong_tail, tail)):
            # Sample positions without adjacency. Keep the final token at end.
            available = list(range(cut, len(path) - 2))
            rng.shuffle(available)
            indices = [len(path) - 1]
            for k in available:
                if all(abs(k - old) >= 2 for old in indices):
                    indices.append(k)
                    if len(indices) == m:
                        break
            if len(indices) != m:
                break
            for k, color in zip(sorted(indices), labels):
                p = path[k]
                assert p not in markers
                markers[p] = color
                grid[p] = color
        else:
            length = len(chosen[0])
            slack = _slack(rng, anchor, support_length, chamber_width, side, length, diagonal)
            for p in slack:
                grid[p] = key[0]
            # Construction uses sampled geometry, independent of input parsing.
            output = grid.copy()
            for p in slack:
                output[p] = floor
            current = key[0]
            for p in chosen[0]:
                if p in markers:
                    current = markers[p]
                else:
                    output[p] = current
            info = _parse(grid)
            assert info["path"] == chosen[0]
            assert np.array_equal(output, _paint(info))
            return {"input": grid, "output": output}
    # A constructive tree guarantees a legal result after the bounded random
    # search, using the same three competing-route witnesses and all lessons.
    return _fallback_pair(rng, roles, key, tail, wrong_tail, diagonal, prefer_long)


def _fallback_pair(rng, roles, key, tail, wrong_tail, diagonal, prefer_long):
    wall, floor, support_color = roles
    support_length = int(rng.integers(6, 8))
    end_y, width = 3 + support_length, 27
    arm = max(8, 2 * len(tail)) + (2 if prefer_long else 0)
    third_arm = arm - 2 if prefer_long else arm + 2
    height = end_y + 8 + third_arm
    # A trunk turns from the left into a three-way junction. The correct arm
    # turns back toward the left, creating an unavoidable obstacle detour.
    anchor = end_y, 2
    trunk = [anchor, (end_y + 1, 2)]
    trunk += [(end_y + 2, c) for c in range(2, 14)]
    trunk += [(r, 13) for r in range(end_y + 3, end_y + 7)]
    fork = trunk[-1]
    correct = trunk + [(fork[0], 13 - k) for k in range(1, arm + 1)]
    wrong_order = trunk + [(fork[0], 13 + k) for k in range(1, arm + 1)]
    wrong_length = trunk + [(fork[0] + k, 13) for k in range(1, third_arm + 1)]
    cells = set(correct + wrong_order + wrong_length)
    g = np.full((height, width), wall, dtype=np.int8)
    g[1, 1:-1] = floor
    start_x = int(rng.integers(2, width - (2 * len(key) - 1)))
    g[1, start_x:start_x + 2 * len(key):2] = key
    chamber_width = int(rng.integers(9, 13))
    g[4:end_y + 1, 2:2 + chamber_width] = floor
    g[4:end_y + 1, 1] = support_color
    for p in cells:
        g[p] = floor
    markers = {}
    for path, labels in zip((correct, wrong_order, wrong_length), (tail, wrong_tail, tail)):
        indices = list(range(len(path) - 1 - 2 * (len(labels) - 1), len(path), 2))
        for j, color in zip(indices, labels):
            markers[path[j]] = color
            g[path[j]] = color
    slack = _slack(rng, anchor, support_length, chamber_width, 1, len(correct), diagonal)
    for p in slack:
        g[p] = key[0]
    out = g.copy()
    for p in slack:
        out[p] = floor
    current = key[0]
    for p in correct:
        if p in markers:
            current = markers[p]
        else:
            out[p] = current
    if rng.integers(2):
        g, out = np.fliplr(g).copy(), np.fliplr(out).copy()
        # Reflect the arena while preserving the fixed reading convention.
        for a in (g, out):
            a[1, 1:-1] = floor
            a[1, start_x:start_x + 2 * len(key):2] = key
    pair = {"input": g, "output": out}
    _validate_pair(pair)
    return pair


def _signature(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError("training count must be 2..10; test count must be 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.integers(2))
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    intensity = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_preference = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            current_roles = roles if stable else tuple(map(int, rng.choice(10, 3, replace=False)))
            if split == "train" and index == 0:
                m, pattern, diagonal = 3, "return", True
            elif split == "train" and index == 1:
                m, pattern, diagonal = 3, "adjacent", False
            else:
                m = 4 if split == "test" else int(rng.integers(2, 4))
                pattern = "mixed" if rng.random() < .7 else "distinct"
                diagonal = bool(rng.integers(2))
            prefer_long = bool(rng.integers(2))
            if split == "train" and index == 0:
                first_preference = prefer_long
            elif split == "train" and index == 1:
                prefer_long = not first_preference
            pair = _build_pair(rng, current_roles, m, pattern, diagonal, prefer_long, intensity)
            if _signature(pair["input"]) in seen:
                # Forty affine color bijections give distinct versions when
                # at least six roles/labels occur; at most 13 inputs are used.
                found = None
                for multiplier in (1, 3, 7, 9):
                    for offset in range(10):
                        mapping = (multiplier * np.arange(10) + offset) % 10
                        changed = {k: mapping[g].astype(np.int8) for k, g in pair.items()}
                        if _signature(changed["input"]) not in seen:
                            found = changed
                            break
                    if found is not None:
                        break
                assert found is not None
                pair = found
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _parse(inp)
    assert np.array_equal(_paint(info), out), "incorrect reconstructed thread"
    assert not np.array_equal(inp, out)
    assert len(info["matches"]) >= 2, "length clue has no competing key route"
    assert len(info["exact"]) >= 2, "key clue has no competing same-length endpoint"
    assert len(info["path"]) == info["length"]
    assert len(set(info["path"])) == info["length"]
    assert info["path"][0] == info["anchor"]
    assert info["path"][-1] in info["markers"]
    assert len(_component(set(info["path"]), info["anchor"])) == info["length"]
    assert all(out[p] == inp[p] for p in info["support"] | set(info["markers"]))
    assert np.array_equal(out[inp == info["wall"]], inp[inp == info["wall"]])
    assert np.array_equal(out[:3], inp[:3])
    assert not np.array_equal(out, _paint(info, mode="constant"))
    assert not np.array_equal(out, _paint(info, mode="destination"))
    # Each selected checkpoint changes the visible outgoing stretch where
    # labels differ; every non-final checkpoint has a following floor cell.
    assert all(info["path"][j + 1] not in info["markers"]
               for j in info["marker_indices"][:-1])
    a, b = info["path"][0], info["path"][-1]
    assert info["length"] > abs(a[0] - b[0]) + abs(a[1] - b[1]) + 1, "no detour witness"
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            signature = _signature(info["grid"])
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            infos[split].append(info)
    training = infos["train"]
    assert any(len(_component(i["slack"], i["anchor"])) < i["length"] for i in training)
    assert any(i["key"][0] in i["key"][1:] for i in training)
    assert any(any(a == b for a, b in zip(i["key"], i["key"][1:])) for i in training)
    assert any(any(len(p) < i["length"] for p in i["matches"]) for i in training)
    assert any(any(len(p) > i["length"] for p in i["matches"]) for i in training)
    assert max(len(i["key"]) for i in training) < min(len(i["key"]) for i in infos["test"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed semantic, variation, metamorphic and corruption checks."""
    shapes, structures, palettes, lengths, topologies, sides = set(), set(), set(), set(), set(), set()
    order_foils, length_foils, diagonal_cases = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _validate_pair(pair)
                shapes.add(i["grid"].shape)
                palettes.add((i["wall"], i["floor"], i["support_color"]))
                lengths.add(i["length"])
                sides.add(i["side"])
                branching = tuple(sorted(sum((p[0] + dr, p[1] + dc) in i["cells"]
                                             for dr, dc in DIR4) for p in i["cells"]))
                topologies.add(branching)
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in i["grid"].ravel())
                structures.add((i["grid"].shape, hashlib.sha256(normalized).digest()))
                order_foils += len(i["exact"]) - 1
                length_foils += len(i["matches"]) - 1
                diagonal_cases += len(_component(i["slack"], i["anchor"])) < i["length"]
        if seed < 3:
            assert _equal(puzzle, generate_puzzle(seed))
            p = puzzle["test"][0]
            # Both lateral reflection and arbitrary color permutation preserve rule.
            reflected_in = np.fliplr(p["input"]).copy()
            # Reflection moves the arena, while the instruction is still read
            # left-to-right in a fresh, valid header placement.
            original = _parse(p["input"])
            key = original["key"]
            x = list(range(2, 2 + 2 * len(key), 2))
            reflected_in[1, 1:-1] = original["floor"]
            reflected_in[1, x] = key
            expected = np.fliplr(p["output"]).copy()
            expected[1, 1:-1] = original["floor"]
            expected[1, x] = key
            assert np.array_equal(solve(reflected_in), expected)
            permutation = np.array([9, 2, 7, 0, 4, 8, 1, 6, 5, 3], dtype=np.int8)
            assert np.array_equal(solve(permutation[p["input"]]), permutation[p["output"]])
            # Same arena and key, different resource count: choose the other
            # complete key-matching route. This is a direct B-clue ablation.
            width = sum(int(original["grid"][original["endpoint"][0], c])
                        not in (original["wall"], original["support_color"])
                        for c in range(original["grid"].shape[1]))
            alternative = next(path for path in original["matches"] if path != original["path"])
            changed_length = original["grid"].copy()
            for q in original["slack"]:
                changed_length[q] = original["floor"]
            replacement = _slack(np.random.default_rng(777 + seed), original["anchor"],
                                 len(original["support"]), width, original["side"],
                                 len(alternative), False)
            for q in replacement:
                changed_length[q] = original["key"][0]
            assert _parse(changed_length)["path"] == alternative
            assert not np.array_equal(solve(changed_length), p["output"])
            # Same arena and resource count, different order: the equal-length
            # foil becomes the solution. This directly exercises the A clue.
            foil = next((path, sequence) for end, path, sequence in original["candidates"]
                        if len(path) == original["length"] and path != original["path"]
                        and len(sequence) == len(original["key"]) - 1)
            changed_key = original["grid"].copy()
            token_x = [c for c in range(1, changed_key.shape[1] - 1)
                       if int(changed_key[1, c]) != original["floor"]]
            changed_key[1, token_x] = [original["key"][0]] + foil[1]
            assert _parse(changed_key)["path"] == foil[0]
            assert not np.array_equal(solve(changed_key), p["output"])
            # Slack arrangement is not a shape cue: preserve its cardinality.
            rearranged = original["grid"].copy()
            for q in original["slack"]:
                rearranged[q] = original["floor"]
            replacement = _slack(np.random.default_rng(888 + seed), original["anchor"],
                                 len(original["support"]), width, original["side"],
                                 original["length"], False)
            for q in replacement:
                rearranged[q] = original["key"][0]
            assert np.array_equal(solve(rearranged), p["output"])
    assert len(shapes) >= 30 and len(structures) >= 180
    assert len(topologies) >= 30 and len(palettes) >= 20 and len(lengths) >= 15
    assert sides == {-1, 1} and diagonal_cases >= 42
    assert order_foils >= 210 and length_foils >= 210
    for m in (2, 3, 4, 5):
        tail = [4, 5] + [3, 4, 5][:m - 2]
        wrong_tail = [5, 4] + tail[2:]
        for diagonal in (False, True):
            for longer in (False, True):
                _validate_pair(_fallback_pair(np.random.default_rng(100 + m), (0, 1, 2),
                                              [3] + tail, tail, wrong_tail, diagonal, longer))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(3100 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(909, 2, 1)
    info = _parse(damaged["test"][0]["input"])
    damaged["test"][0]["output"][next(iter(info["support"]))] = info["floor"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return {"seeds": 42, "pairs": 210, "grid_shapes": len(shapes),
                "normalized_structures": len(structures), "tree_degree_profiles": len(topologies),
                "role_palettes": len(palettes), "strand_lengths": len(lengths),
                "order_foils": order_foils, "length_foils": length_foils,
                "diagonal_slack_pairs": diagonal_cases}
    raise AssertionError("validator accepted corrupted support")


if __name__ == "__main__":
    print(self_test())
