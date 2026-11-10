"""Symmetry completes the maze before its unique shortest route is painted.

All roles needed by solve are visible or fixed: 0 background, 2 terminals,
6 symmetry markers, 4 route; every other color is an impassable wall.
No files, parent generators, training pairs, or generation state are read.
"""
from collections import deque
import hashlib
import json
import numpy as np


WALL_COLORS = (1, 3, 5, 7, 8, 9)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    """Eight-connected components, with deterministic traversal order."""
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in todo:
                        todo.remove(q)
                        stack.append(q)
        groups.append(sorted(cells))
    return groups


def _kind(marker):
    if len(marker) == 1:
        return "P"
    rows, cols = {r for r, c in marker}, {c for r, c in marker}
    if len(rows) == 1 and len(marker) == max(cols) - min(cols) + 1:
        return "H"
    if len(cols) == 1 and len(marker) == max(rows) - min(rows) + 1:
        return "V"
    raise ValueError("Markers must be isolated singletons or straight segments")


def _mapping(marker, payload, grid):
    kind = _kind(marker)
    ar, ac = marker[0]
    r0, r1 = min(r for r, c in payload), max(r for r, c in payload)
    c0, c1 = min(c for r, c in payload), max(c for r, c in payload)
    if min(max(abs(r-a), abs(c-b)) for r, c in payload for a, b in marker) != 1:
        return None
    if kind == "H":
        if not (r1 < ar or r0 > ar) or c1 < marker[0][1] or c0 > marker[-1][1]:
            return None
        dest = [(2*ar-r, c) for r, c in payload]
    elif kind == "V":
        if not (c1 < ac or c0 > ac) or r1 < marker[0][0] or r0 > marker[-1][0]:
            return None
        dest = [(r, 2*ac-c) for r, c in payload]
    else:
        if not ((r1 <= ar or r0 >= ar) and (c1 <= ac or c0 >= ac)):
            return None
        dest = [(2*ar-r, 2*ac-c) for r, c in payload]
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) or grid[r, c] != 0 for r, c in dest):
        return None
    return dest


def _complete(grid):
    """Associate markers from input geometry; unassociated walls remain static."""
    if np.any(grid == 4):
        raise ValueError("Route color 4 is absent from inputs")
    payloads = [p for color in WALL_COLORS for p in _components(grid == color)]
    markers = _components(grid == 6)
    if not markers:
        raise ValueError("At least one symmetry marker is required")
    groups, used, targets = [], set(), set()
    completed = grid.copy()
    for marker in markers:
        kind = _kind(marker)
        candidates = []
        for i, payload in enumerate(payloads):
            dest = _mapping(marker, payload, grid)
            if dest is not None:
                candidates.append((i, dest))
        if len(candidates) != 1:
            raise ValueError("A marker must have exactly one eligible wall component")
        i, dest = candidates[0]
        if i in used or targets.intersection(dest):
            raise ValueError("Ambiguous ownership or overlapping copies")
        used.add(i)
        targets.update(dest)
        color = int(grid[payloads[i][0]])
        for q in dest:
            completed[q] = color
        groups.append({"kind": kind, "marker": marker, "source": payloads[i],
                       "dest": dest, "color": color})
    return completed, groups


def _bfs(blocked, start):
    """Distances, shortest-path multiplicities capped at two, and predecessors."""
    h, w = blocked.shape
    walls = blocked.ravel().tolist()
    dist, ways, parent = [-1]*(h*w), [0]*(h*w), [-1]*(h*w)
    root = start[0]*w + start[1]
    dist[root], ways[root] = 0, 1
    queue = deque([root])
    while queue:
        p = queue.popleft()
        r, c = divmod(p, w)
        neighbors = []
        if r:
            neighbors.append(p-w)
        if r+1 < h:
            neighbors.append(p+w)
        if c:
            neighbors.append(p-1)
        if c+1 < w:
            neighbors.append(p+1)
        for q in neighbors:
            if walls[q]:
                continue
            if dist[q] < 0:
                dist[q], ways[q], parent[q] = dist[p]+1, ways[p], p
                queue.append(q)
            elif dist[q] == dist[p]+1:
                ways[q] = min(2, ways[q]+ways[p])
    return dist, ways, parent


def _path(blocked, start, goal):
    dist, ways, parent = _bfs(blocked, start)
    w = blocked.shape[1]
    p, root = goal[0]*w+goal[1], start[0]*w+start[1]
    if dist[p] < 0:
        return [], 0
    count, path = ways[p], []
    while p != root:
        path.append(divmod(p, w))
        p = parent[p]
    return [start] + path[::-1], count


def _blocked(grid):
    return (grid != 0) & (grid != 2)


def _terminals(grid):
    cells = [tuple(map(int, q)) for q in np.argwhere(grid == 2)]
    if len(cells) != 2:
        raise ValueError("Exactly two visible terminals are required")
    return cells


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    completed, _ = _complete(grid)
    start, goal = _terminals(completed)
    path, count = _path(_blocked(completed), start, goal)
    if count != 1 or len(path) < 3:
        raise ValueError("The completed maze must have one nontrivial shortest route")
    for q in path[1:-1]:
        completed[q] = 4
    return completed


def _witness(group):
    source, dest = group["source"], group["dest"]
    dr = min(r for r, c in dest) - min(r for r, c in source)
    dc = min(c for r, c in dest) - min(c for r, c in source)
    return {(r+dr, c+dc) for r, c in source} != set(dest)


def _turns(path):
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _longer_route(blocked, path):
    # A missing edge of a shortest route forces any remaining route to be longer.
    for q in path[1:-1]:
        other = blocked.copy()
        other[q] = True
        candidate, _ = _path(other, path[0], path[-1])
        if candidate:
            assert len(candidate) > len(path)
            return True
    return False


def _prototype(rng, operation, witness, attempt=0):
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    style = int(rng.integers(5))
    cells = set()
    if style == 0:
        cells = {(r, c) for r in range(h) for c in range(w)} - {(0, 0)}
    elif style == 1:
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif style == 2:
        cells = {(r, min(r, w-1)) for r in range(h)}
        cells |= {(h-1, c) for c in range(min(h-1, w-1), w)}
    elif style == 3:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
        cells.discard((0, w-1))
    else:
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, h*w+1))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    cells = {(r-r0, c-c0) for r, c in cells}
    if operation == "P":
        # Corner contact makes ownership visible; offsets can reach center axes.
        cells.add((0, 0))
        dr, dc = [(1, 1), (0, 1), (1, 0)][int(rng.integers(3))]
        sr, sc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
        source = [(sr*(r+dr), sc*(c+dc)) for r, c in sorted(cells)]
        marker = [(0, 0)]
        dest = [(-r, -c) for r, c in source]
        kind = "P"
    else:
        sign = int(rng.choice([-1, 1]))
        source = [(sign*(r+1), c) for r, c in sorted(cells)]
        contact = min(c for r, c in cells if r == 0)
        length = int(rng.integers(2, max(3, w+2)))
        left = contact - int(rng.integers(length))
        marker = [(0, c) for c in range(left, left+length)]
        dest = [(-r, c) for r, c in source]
        kind = "H"
        if rng.integers(2):
            source, marker, dest = [[(c, r) for r, c in part]
                                    for part in (source, marker, dest)]
            kind = "V"
    result = {"kind": kind, "source": source, "marker": marker, "dest": dest}
    if witness and not _witness(result):
        if attempt < 16:
            return _prototype(rng, operation, witness, attempt+1)
        # A bounded orientation witness, rather than an unbounded rejection loop.
        if operation == "P":
            result = {"kind": "P", "marker": [(0, 0)],
                      "source": [(1, 1), (1, 2), (2, 1)],
                      "dest": [(-1, -1), (-1, -2), (-2, -1)]}
        else:
            result = {"kind": "H", "marker": [(0, 0), (0, 1)],
                      "source": [(1, 0), (1, 1), (2, 0)],
                      "dest": [(-1, 0), (-1, 1), (-2, 0)]}
        source, marker, dest = result["source"], result["marker"], result["dest"]
    footprint = source + marker + dest
    r0, c0 = min(r for r, c in footprint), min(c for r, c in footprint)
    for key in ("source", "marker", "dest"):
        result[key] = sorted((r-r0, c-c0) for r, c in result[key])
    return result


def _layout(rng, specs, palette):
    h, w = int(rng.integers(11, 20)), int(rng.integers(11, 20))
    grid = np.zeros((h, w), dtype=np.int8)
    reserved = np.zeros((h, w), dtype=bool)
    for operation, witness in specs:
        group = _prototype(rng, operation, witness)
        footprint = group["source"] + group["marker"] + group["dest"]
        gh, gw = max(r for r, c in footprint)+1, max(c for r, c in footprint)+1
        origins = [(r, c) for r in range(h-gh+1) for c in range(w-gw+1)]
        rng.shuffle(origins)
        for top, left in origins:
            if all(not reserved[r+top, c+left] for r, c in footprint):
                break
        else:
            return None
        for r, c in footprint:
            a, b = r+top, c+left
            reserved[max(0,a-1):min(h,a+2), max(0,b-1):min(w,b+2)] = True
        color = int(rng.choice(palette[1:]))
        for key, value in (("source", color), ("marker", 6)):
            for r, c in group[key]:
                grid[r+top, c+left] = value
    # Static walls are whole sampled segments/rectangles, cut only at the
    # reserved symmetry neighborhoods. Their distinct color preserves ownership.
    static = np.zeros((h, w), bool)
    density = float(rng.uniform(.18, .32))
    for _ in range(60):
        if static.sum() >= density*h*w:
            break
        vertical = bool(rng.integers(2))
        length, thick = int(rng.integers(2, 8)), int(rng.choice([1, 1, 1, 2]))
        a, b = (length, thick) if vertical else (thick, length)
        r, c = int(rng.integers(h-a+1)), int(rng.integers(w-b+1))
        static[r:r+a, c:c+b] = True
    # No static cell may hide the association or occupy a copy destination.
    static[reserved] = False
    grid[static] = palette[0]
    try:
        completed, groups = _complete(grid)
    except ValueError:
        return None
    if len(groups) != len(specs):
        return None
    return grid, completed, groups


def _choose_endpoints(rng, grid, completed, groups, test):
    h, w = grid.shape
    blocked = _blocked(completed)
    cells = [tuple(map(int, q)) for q in np.argwhere(completed == 0)]
    boundary = [q for q in cells if q[0] in (0, h-1) or q[1] in (0, w-1)]
    rng.shuffle(cells)
    rng.shuffle(boundary)
    starts = list(dict.fromkeys(boundary[:5]+cells[:9]))
    for start in starts:
        dist, ways, parent = _bfs(blocked, start)
        raw_dist = _bfs(_blocked(grid), start)[0]
        candidates = []
        for goal in cells:
            p = goal[0]*w+goal[1]
            manhattan = abs(start[0]-goal[0])+abs(start[1]-goal[1])
            if ways[p] == 1 and dist[p] >= max(7, manhattan+2) and 0 <= raw_dist[p] < dist[p]:
                candidates.append(goal)
        rng.shuffle(candidates)
        for goal in candidates[:12]:
            path, count = _path(blocked, start, goal)
            if _turns(path) < 3 or not _longer_route(blocked, path):
                continue
            critical = _critical_groups(blocked, groups, path)
            if critical < (min(2, len(groups)) if test else 1):
                continue
            inp = grid.copy()
            inp[start] = inp[goal] = 2
            # This extra input-only read also rules out terminals in a copy.
            return {"input": inp, "output": solve(inp)}
    return None


def _pair(rng, specs, palette, test):
    for _ in range(220):
        layout = _layout(rng, specs, palette)
        if layout is not None:
            pair = _choose_endpoints(rng, *layout, test)
            if pair is not None:
                return pair
    raise RuntimeError("Bounded construction failed to find a symmetry-dependent route")


def _counts(num_train, num_test):
    for name, x, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not lo <= x <= hi:
            raise ValueError(name + " has an illegal count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.choice(WALL_COLORS, size=int(rng.integers(3, 6)), replace=False)]
    operations = [("L",), ("P",), ("L", "P")][int(rng.choice(3, p=[.22, .18, .60]))]
    plans = [[] for _ in range(num_train)]
    order = rng.permutation(num_train)
    for i, kind in enumerate(operations):
        plans[int(order[i])].append((kind, True))
    single_indices = {int(order[0]), int(order[1])}
    for index, plan in enumerate(plans):
        if not plan:
            plan.append((str(rng.choice(operations)), True))
        if index not in single_indices and rng.random() < .40:
            plan.append((str(rng.choice(operations)), False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                specs = plans[i]
            else:
                n = int(rng.integers(2, 4))
                specs = [(kind, False) for kind in operations]
                while len(specs) < n:
                    specs.append((str(rng.choice(operations)), False))
                rng.shuffle(specs)
            for _ in range(12):
                pair = _pair(rng, specs, palette, split == "test")
                sig = _signature(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(inp):
    completed, groups = _complete(inp)
    start, goal = _terminals(inp)
    blocked = _blocked(completed)
    path, count = _path(blocked, start, goal)
    raw_path, raw_count = _path(_blocked(inp), start, goal)
    return completed, groups, path, count, raw_path, raw_count


def _critical_groups(blocked, groups, path):
    """How many individual copies change distance or uniqueness if omitted?"""
    w, critical = blocked.shape[1], 0
    goal = path[-1][0]*w+path[-1][1]
    for group in groups:
        without = blocked.copy()
        for q in group["dest"]:
            without[q] = False
        dist, ways, _ = _bfs(without, path[0])
        critical += int(dist[goal] < len(path)-1 or
                        (dist[goal] == len(path)-1 and ways[goal] > 1))
    return critical


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, trained, witnessed = set(), set(), set()
    single_controls = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            completed, groups, path, count, raw, _ = _facts(inp)
            assert count == 1 and len(path) >= 8 and _turns(path) >= 3
            assert raw and len(raw) < len(path), "Symmetry must alter shortest routing"
            assert len(path)-1 > abs(path[0][0]-path[-1][0])+abs(path[0][1]-path[-1][1])
            assert _longer_route(_blocked(completed), path), "Shortest selection needs a longer alternative"
            assert np.any((completed == 0) & (out == 0)), "Off-route background must survive"
            assert np.array_equal(inp[inp != 0], out[inp != 0])
            assert np.array_equal(solve(inp), out)
            assert np.count_nonzero(out == 4) == len(path)-2
            assert np.count_nonzero((inp == 0) & (out != 0)) == sum(len(g["dest"]) for g in groups)+len(path)-2
            kinds = {"P" if g["kind"] == "P" else "L" for g in groups}
            if split == "train":
                single_controls += len(groups) == 1
                trained.update(kinds)
                witnessed.update("P" if g["kind"] == "P" else "L" for g in groups if _witness(g))
            else:
                assert kinds <= trained
                assert len(groups) >= 2
                assert _critical_groups(_blocked(completed), groups, path) >= 2
    assert trained <= witnessed, "Training must distinguish symmetry from translation"
    assert single_controls >= 2, "Include two uncluttered single-group demonstrations"


def self_test():
    shapes, structures, silhouettes, palettes, counts, kinds = (set() for _ in range(6))
    group_sizes, route_lengths, critical_counts = set(), set(), set()
    digest = hashlib.sha256()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                shapes.add(inp.shape)
                palettes.add(tuple(map(int, np.unique(inp))))
                mapping = {}
                normalized = [mapping.setdefault(int(c), len(mapping)) for c in inp.flat]
                structures.add((inp.shape, bytes(normalized)))
                completed, groups = _complete(inp)
                counts.add(len(groups))
                path, count = _path(_blocked(completed), *_terminals(inp))
                route_lengths.add(len(path)-1)
                critical_counts.add(_critical_groups(_blocked(completed), groups, path))
                digest.update(inp.tobytes())
                digest.update(pair["output"].tobytes())
                for g in groups:
                    group_sizes.add(len(g["source"]))
                    kinds.add(g["kind"])
                    r0 = min(r for r, c in g["source"])
                    c0 = min(c for r, c in g["source"])
                    silhouettes.add(tuple((r-r0, c-c0) for r, c in g["source"]))
                # Transform equivariance checks coordinates and preserves route semantics.
                assert np.array_equal(solve(np.rot90(inp)), np.rot90(pair["output"]))
                assert np.array_equal(solve(inp[:, ::-1]), pair["output"][:, ::-1])
                before = inp.copy()
                solve(inp)
                assert np.array_equal(inp, before), "solve must not mutate its input"
    assert len(shapes) >= 30 and len(structures) >= 190
    assert len(silhouettes) >= 40 and len(palettes) >= 15
    assert {1, 2, 3} <= counts and kinds == {"H", "V", "P"}
    assert len(group_sizes) >= 8 and len(route_lengths) >= 15
    first, again = generate_puzzle(98765, 2, 1), generate_puzzle(98765, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], again[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Asymmetric, diagonal, center-aligned point payload: inspect the actual
    # coordinate image, independently of the procedural silhouette sampler.
    fixture = np.zeros((11, 11), dtype=np.int8)
    fixture[5, 5] = 6
    fixture[5, 6] = fixture[6, 7] = fixture[7, 7] = 3
    completed, groups = _complete(fixture)
    assert set(groups[0]["dest"]) == {(5, 4), (4, 3), (3, 3)}
    # A finite marker permits payload overhang; it does not crop the copy.
    fixture = np.zeros((11, 11), dtype=np.int8)
    fixture[5, 4:6] = 6
    fixture[4, 3:7] = 7
    fixture[3, 3] = 7
    completed, groups = _complete(fixture)
    assert set(groups[0]["dest"]) == {(6, 3), (6, 4), (6, 5), (6, 6), (7, 3)}
    # There is no tie-breaking convention: two equally short routes are invalid.
    mask = np.zeros((3, 3), dtype=bool)
    assert _path(mask, (0, 0), (2, 2))[1] == 2
    mask[1, :] = True
    assert _path(mask, (0, 0), (2, 2))[1] == 0
    # Validate labels as well as solver round-tripping: omission of either stage
    # must be rejected even with valid array structure and preserved input cells.
    for omit in ("symmetry", "route"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
        pair = bad["train"][0]
        completed, groups = _complete(pair["input"])
        if omit == "symmetry":
            for group in groups:
                for q in group["dest"]:
                    pair["output"][q] = 0
        else:
            pair["output"] = completed
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Accepted omitted " + omit)
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "normalized_inputs": len(structures), "source_silhouettes": len(silhouettes),
            "input_palettes": len(palettes), "group_counts": sorted(counts),
            "source_cell_counts": sorted(group_sizes), "marker_kinds": sorted(kinds),
            "route_length_range": [min(route_lengths), max(route_lengths)],
            "critical_group_counts": sorted(critical_counts), "grid_payload_sha256": digest.hexdigest()}


if __name__ == "__main__":
    print(json.dumps({"self_test": "passed", "variation": self_test()}, separators=(",", ":")))
