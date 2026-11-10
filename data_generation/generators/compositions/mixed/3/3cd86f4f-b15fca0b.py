"""Complete a unique shortest maze route, then shear the entire completed scene.

Only NumPy and the standard library are used. Generation and self-tests are
bounded, run in memory, and never write samples or execution traces.
"""
from collections import deque
import hashlib
import json

import numpy as np


BACKGROUND = 0
ROUTE = 4
ROLE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid


def _roles(grid):
    """Zero is ground; exactly two equal nonzero cells are the terminals."""
    colors, counts = np.unique(grid, return_counts=True)
    inventory = dict(zip(map(int, colors), map(int, counts)))
    if len(inventory) != 3 or 0 not in inventory or ROUTE in inventory:
        raise ValueError("input must have ground and two nonzero roles, without route color 4")
    terminals = [c for c, count in inventory.items() if c != 0 and count == 2]
    if len(terminals) != 1:
        raise ValueError("one nonzero color must occur exactly twice")
    terminal = terminals[0]
    wall = next(c for c in inventory if c not in (0, terminal))
    if inventory[wall] < 3:
        raise ValueError("walls must occupy at least three cells")
    ends = [tuple(map(int, p)) for p in np.argwhere(grid == terminal)]
    return wall, terminal, ends


def _search(blocked, start, goal):
    """BFS with shortest-path multiplicity saturated at two; no tie breaker."""
    h, w = blocked.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    ways = np.zeros((h, w), dtype=np.int8)
    predecessor = {}
    queue = deque([start])
    distance[start] = 0
    ways[start] = 1
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            cell = (r + dr, c + dc)
            rr, cc = cell
            if not (0 <= rr < h and 0 <= cc < w) or blocked[cell]:
                continue
            nd = int(distance[r, c]) + 1
            if distance[cell] < 0:
                distance[cell] = nd
                ways[cell] = ways[r, c]
                predecessor[cell] = (r, c)
                queue.append(cell)
            elif distance[cell] == nd:
                ways[cell] = min(2, int(ways[cell]) + int(ways[r, c]))
    if distance[goal] < 0:
        return None, 0, distance
    path = [goal]
    while path[-1] != start:
        path.append(predecessor[path[-1]])
    path.reverse()
    return path, int(ways[goal]), distance


def _shear(scene):
    h, w = scene.shape
    output = np.zeros((h, w + h - 1), dtype=np.int8)
    for r in range(h):
        offset = h - 1 - r
        output[r, offset:offset + w] = scene[r]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use the visible unsheared maze, then map (r,c) to (r,c+h-1-r)."""
    grid = _grid(input_grid)
    if sum(grid.shape) - 1 > 30:
        raise ValueError("sheared output would exceed 30 columns")
    wall, _, ends = _roles(grid)
    path, count, _ = _search(grid == wall, *ends)
    if path is None or count != 1 or len(path) < 3:
        raise ValueError("maze must have a unique shortest route with an intermediate cell")
    completed = grid.astype(np.int8, copy=True)
    for cell in path[1:-1]:
        completed[cell] = ROUTE
    return _shear(completed)


def _turns(path):
    directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _island(rng, h, w, hollow):
    """An obstacle with two unequal detours, optionally with a bay cut in it."""
    height = int(rng.integers(3, h - 1))
    width = int(rng.integers(3 if hollow else 1, w - 1))
    top = int(rng.integers(1, h - height))
    left = int(rng.integers(1, w - width))
    bottom, right = top + height - 1, left + width - 1
    # Choose either end of the obstacle's vertical span, avoiding a distance tie.
    candidates = [r for r in range(top, bottom + 1)
                  if 2 * r != top + bottom]
    row = int(rng.choice(candidates))
    blocked = np.zeros((h, w), dtype=bool)
    blocked[top:bottom + 1, left:right + 1] = True
    if hollow:
        # Keep the nearer cap solid and cut a connected bay from the farther cap.
        bay_left = int(rng.integers(left + 1, right))
        bay_right = int(rng.integers(bay_left, right))
        depth = int(rng.integers(1, height))
        if row - top < bottom - row:
            blocked[bottom - depth + 1:bottom + 1, bay_left:bay_right + 1] = False
        else:
            blocked[top:top + depth, bay_left:bay_right + 1] = False
    return blocked, (row, left - 1), (row, right + 1)


def _gates(rng, h, w):
    """Walls with one-cell openings; openings, thicknesses, and order vary."""
    n = int(rng.integers(2, min(5, (w - 1) // 2) + 1))
    gate_rows = [int(rng.integers(0, h)) for _ in range(n)]
    if all(r in (0, h - 1) for r in gate_rows):
        gate_rows[int(rng.integers(n))] = int(rng.integers(1, h - 1))
    thickness = [1] * n
    clear = [1] * (n + 1)
    # Wider spaces are safe when successive openings align. Otherwise the one
    # intervening column forces the vertical leg and prevents shortest-path ties.
    options = [("wall", i) for i in range(n)]
    options += [("space", i + 1) for i in range(n - 1)
                if gate_rows[i] == gate_rows[i + 1]]
    for _ in range(w - (2 * n + 1)):
        kind, index = options[int(rng.integers(len(options)))]
        (thickness if kind == "wall" else clear)[index] += 1
    blocked = np.zeros((h, w), dtype=bool)
    col = clear[0]
    for i in range(n):
        blocked[:, col:col + thickness[i]] = True
        blocked[gate_rows[i], col:col + thickness[i]] = False
        col += thickness[i] + clear[i + 1]
    return blocked, (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)


def _orient(mask, start, goal, rng):
    turns = int(rng.integers(4))
    reflect = bool(rng.integers(2))
    h, w = mask.shape
    def point(cell):
        r, c = cell
        hh, ww = h, w
        for _ in range(turns):
            r, c = ww - 1 - c, r
            hh, ww = ww, hh
        return (r, ww - 1 - c) if reflect else (r, c)
    transformed = np.rot90(mask, turns)
    if reflect:
        transformed = transformed[:, ::-1]
    return transformed.copy(), point(start), point(goal)


def _alternate_length(blocked, path):
    """Find one strictly longer route by forbidding a chosen shortest-path cell."""
    for cell in path[1:-1]:
        modified = blocked.copy()
        modified[cell] = True
        other, _, _ = _search(modified, path[0], path[-1])
        if other is not None:
            return len(other) - 1
    return None


def _signature(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def _make_pair(rng, mode, test, wall, terminal, seen, different_height=None):
    for _ in range(180):
        if test:
            # Strict size extrapolation, without changing the solving convention.
            h = int(rng.integers(13, 25))
            w = int(rng.integers(5, min(12, 31 - h) + 1))
        else:
            h = int(rng.integers(5, 13))
            w = int(rng.integers(5, 13))
        built = (_gates(rng, h, w) if mode == "gates"
                 else _island(rng, h, w, mode == "bay"))
        blocked, start, goal = _orient(*built, rng)
        if different_height is not None and blocked.shape[0] == different_height:
            continue
        path, count, _ = _search(blocked, start, goal)
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if (path is None or count != 1 or len(path) - 1 <= manhattan
                or _turns(path) < 2 or np.count_nonzero(~blocked) <= len(path)):
            continue
        grid = np.zeros(blocked.shape, dtype=np.int8)
        grid[blocked] = wall
        grid[start] = grid[goal] = terminal
        key = _signature(grid)
        if key in seen:
            continue
        scene = grid.copy()
        for cell in path[1:-1]:
            scene[cell] = ROUTE
        seen.add(key)
        return {"input": grid, "output": _shear(scene)}
    raise RuntimeError(f"bounded sampler exhausted for {mode}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    wall, terminal = map(int, rng.choice(ROLE_COLORS, size=2, replace=False))
    # The two required examples separate global optimization from forced passage.
    modes = ["island", "gates"]
    if num_train >= 3:
        modes.append("bay")
    while len(modes) < num_train:
        modes.append(str(rng.choice(("island", "bay", "gates"))))
    seen = set()
    train = []
    for mode in modes:
        prior_height = train[0]["input"].shape[0] if len(train) == 1 else None
        train.append(_make_pair(rng, mode, False, wall, terminal, seen, prior_height))
    rng.shuffle(train)
    taught_modes = ("island", "gates", "bay") if num_train >= 3 else ("island", "gates")
    test = [_make_pair(rng, str(rng.choice(taught_modes)), True, wall, terminal, seen)
            for _ in range(num_test)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while remaining:
        stack = [remaining.pop()]
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in STEPS:
                cell = (r + dr, c + dc)
                if cell in remaining:
                    remaining.remove(cell)
                    stack.append(cell)
        components.append(component)
    return components


def _rectangle(component):
    rows, cols = zip(*component)
    return len(component) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    h, w = grid.shape
    assert h + w - 1 <= 30 and output.shape == (h, h + w - 1)
    wall, terminal, ends = _roles(grid)
    blocked = grid == wall
    path, count, forward = _search(blocked, *ends)
    assert path is not None and count == 1 and len(path) >= 3
    _, _, backward = _search(blocked, ends[1], ends[0])
    length = len(path) - 1
    # Distance-sum membership checks the whole shortest-route set, not just the
    # predecessor chain used to construct the output.
    membership = (forward >= 0) & (backward >= 0) & (forward + backward == length)
    assert np.count_nonzero(membership) == len(path)
    assert all(np.count_nonzero(membership & (forward == d)) == 1
               for d in range(length + 1))
    route_mask = membership.copy()
    for cell in ends:
        route_mask[cell] = False
    recovered = np.empty_like(grid)
    for r in range(h):
        offset = h - 1 - r
        assert np.all(output[r, :offset] == BACKGROUND)
        assert np.all(output[r, offset + w:] == BACKGROUND)
        recovered[r] = output[r, offset:offset + w]
    assert np.all(grid[route_mask] == BACKGROUND)
    assert np.all(recovered[route_mask] == ROUTE)
    assert np.array_equal(recovered[~route_mask], grid[~route_mask])
    assert np.count_nonzero(output == terminal) == 2
    assert np.count_nonzero(output == wall) == np.count_nonzero(blocked)
    assert np.count_nonzero(output == ROUTE) == len(path) - 2
    assert np.array_equal(solve(grid), output)
    # Both concepts are observable in every generated pair.
    assert length > abs(ends[0][0] - ends[1][0]) + abs(ends[0][1] - ends[1][1])
    assert _turns(path) >= 2
    assert np.any((grid == BACKGROUND) & ~membership)
    assert any(a[0] != b[0] for a, b in zip(path, path[1:]))
    return wall, terminal, path


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, shared_roles, heights = set(), set(), set()
    has_longer_alternative = False
    has_split_wall = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            wall, terminal, path = _check_pair(pair)
            grid = np.asarray(pair["input"])
            key = _signature(grid)
            assert key not in seen
            seen.add(key)
            shared_roles.add((wall, terminal))
            if split == "train":
                assert max(grid.shape) <= 12
                heights.add(grid.shape[0])
                has_split_wall |= len(_components(grid == wall)) >= 2
                if not has_longer_alternative:
                    alternative = _alternate_length(grid == wall, path)
                    if alternative is not None:
                        assert alternative > len(path) - 1
                        has_longer_alternative = True
            else:
                assert max(grid.shape) >= 13
    assert len(shared_roles) == 1 and len(heights) >= 2
    assert has_longer_alternative and has_split_wall


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds plus all 30 legal count pairs; no saved payloads."""
    shapes, roles, structures, lengths, turns, components, signatures = (set() for _ in range(7))
    compound = boundary_outputs = interior_terminals = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                wall, terminal, ends = _roles(grid)
                path, count, _ = _search(grid == wall, *ends)
                assert count == 1
                roles.add((wall, terminal))
                shapes.add(grid.shape)
                normalized = np.zeros_like(grid)
                normalized[grid == wall] = 1
                normalized[grid == terminal] = 2
                structures.add(hashlib.sha256(bytes(grid.shape) + normalized.tobytes()).hexdigest())
                lengths.add(len(path) - 1)
                turns.add(_turns(path))
                groups = _components(grid == wall)
                components.add(len(groups))
                compound += any(not _rectangle(group) for group in groups)
                boundary_outputs += sum(grid.shape) - 1 == 30
                interior_terminals += any(0 < r < grid.shape[0] - 1 and 0 < c < grid.shape[1] - 1
                                          for r, c in ends)
                # Shear turns vertical source steps into diagonals, proving that
                # treating the output as a new four-neighbor maze is a different rule.
                mapped = [(r, c + grid.shape[0] - 1 - r) for r, c in path]
                assert any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2
                           for a, b in zip(mapped, mapped[1:]))
    assert len(signatures) == 42 and len(roles) >= 20
    assert len(shapes) >= 35 and len(structures) >= 180
    assert len(lengths) >= 20 and len(turns) >= 4 and len(components) >= 4
    assert compound >= 20 and boundary_outputs >= 1 and interior_terminals >= 25
    first = generate_puzzle(453009930, 10, 4)
    assert _digest(first) == _digest(generate_puzzle(453009930, 10, 4))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            puzzle = generate_puzzle(1000 + ntrain * 10 + ntest, ntrain, ntest)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None), (None, 1), (4, "1")):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Invalid graphs must fail closed rather than select one of several paths.
    for rows in (("20000", "00000", "00200", "00000", "11100"),
                 ("212", "111", "000")):
        ambiguous = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        try:
            solve(ambiguous)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid maze accepted")
    # Exercise non-sampled layouts and list inputs without generation state.
    manual = [[0, 0, 0, 1, 2], [0, 0, 0, 1, 0], [0, 1, 0, 1, 0],
              [0, 1, 0, 0, 0], [2, 1, 0, 0, 0]]
    original = np.array(manual)
    assert solve(manual).shape == (5, 9)
    assert np.array_equal(solve(manual), solve(original))
    assert np.array_equal(original, np.array(manual))
    fixture = ("00000", "02120", "00100", "00100", "00000")
    answer = ("000004440", "000021200", "000010000", "000100000", "000000000")
    digits = lambda rows: np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    assert np.array_equal(solve(digits(fixture)), digits(answer))
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "role_assignments": len(roles), "normalized_inputs": len(structures),
            "route_lengths": len(lengths), "turn_counts": sorted(turns),
            "wall_component_counts": sorted(components), "compound_pairs": int(compound),
            "width_30_outputs": int(boundary_outputs), "interior_terminal_pairs": int(interior_terminals),
            "legal_count_combinations": 30}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
