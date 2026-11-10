"""Contour gears: cyclic, identity-matched symmetries acting on erosion layers.

Only visible guide strips and the input scaffold determine the answer. No parent
code, fixtures, files, or generation state are consulted by the solver.
"""

from collections import deque
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _transform(grid, operation):
    """0..3 are CCW turns; 4..7 first reflect left/right, then turn."""
    base = grid if operation < 4 else np.fliplr(grid)
    return np.rot90(base, operation % 4).copy()


def _side_maps():
    cross = np.zeros((3, 3), dtype=int)
    for i, side in enumerate(SIDES, 1):
        dr, dc = NORMALS[side]
        cross[1 + dr, 1 + dc] = i
    result = []
    for op in range(8):
        turned = _transform(cross, op)
        result.append({side: next(t for t in SIDES
                                 if turned[1 + NORMALS[t][0], 1 + NORMALS[t][1]] == i)
                       for i, side in enumerate(SIDES, 1)})
    return tuple(result)


SIDE_MAPS = _side_maps()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int16, copy=True)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _side(grid, side):
    return {"T": grid[0, :], "R": grid[:, -1],
            "B": grid[-1, :], "L": grid[:, 0]}[side]


def _adjacent(sides):
    return len(sides) == 2 and any(set(sides) == {SIDES[i], SIDES[(i + 1) % 4]}
                                 for i in range(4))


def _depths(open_mask, diagonal=True):
    """Erosion of all open regions, with the board exterior treated as closed."""
    current = open_mask.copy()
    depth = np.full(open_mask.shape, -1, dtype=int)
    h, w = open_mask.shape
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in DIR8 if diagonal else DIR4:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _bfs_depths(open_mask):
    """Separate boundary-distance calculation used by validation."""
    h, w = open_mask.shape
    depth = np.full((h, w), -1, dtype=int)
    q = deque()
    for r, c in np.argwhere(open_mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and open_mask[r + dr, c + dc]) for dr, dc in DIR8):
            depth[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and open_mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                q.append((rr, cc))
    return depth


def _runs(values, allowed, markers):
    """Maximal interior runs of scaffold/marker cells containing a marker."""
    runs = []
    start = None
    for i in range(1, len(values)):
        valid = i < len(values) - 1 and int(values[i]) in allowed
        if valid and start is None:
            start = i
        if not valid and start is not None:
            end = i - 1
            if 9 <= end - start + 1 <= 24 and any(int(x) in markers for x in values[start:i]):
                runs.append((start, end))
            start = None
    return runs


def _guide_line(grid, bounds, side):
    r, c, n = bounds
    return {"T": grid[r - 1, c:c + n], "B": grid[r + n, c:c + n],
            "L": grid[r:r + n, c - 1], "R": grid[r:r + n, c + n]}[side]


def _read_program(line, wall, markers):
    stop = next((i for i, x in enumerate(line) if int(x) == wall), len(line))
    if not 1 <= stop <= 5 or len(line) - stop < 4:
        return None
    if not all(int(x) in markers for x in line[:stop]) or not np.all(line[stop:] == wall):
        return None
    return tuple(int(x) for x in line[:stop])


def _decode(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    bg = int(grid[0, 0])
    if min(h, w) < 14 or any(int(grid[r, c]) != bg for r, c in
                             ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        raise ValueError("the four background corners must be intact")
    outer = {}
    for side in SIDES:
        core = _side(grid, side)[1:-1]
        if len(np.unique(core)) != 1:
            raise ValueError("outer sides must be solid between the corners")
        color = int(core[0])
        if color != bg:
            outer[side] = color
    if not _adjacent(outer) or len(set(outer.values())) != 2:
        raise ValueError("exactly two distinct adjacent outer controls are required")
    markers = set(outer.values())
    interior_markers = np.isin(grid, list(markers))
    interior_markers[[0, -1], :] = False
    interior_markers[:, [0, -1]] = False
    rows, cols = np.where(interior_markers)
    if not len(rows):
        raise ValueError("missing cyclic guides")
    candidates = []
    for wall in sorted(set(int(x) for x in np.unique(grid)) - markers - {bg}):
        allowed = markers | {wall}
        horizontal = [(int(row), a, b) for row in np.unique(rows)
                      for a, b in _runs(grid[row, :], allowed, markers)]
        vertical = [(int(col), a, b) for col in np.unique(cols)
                    for a, b in _runs(grid[:, col], allowed, markers)]
        for hr, hs, he in horizontal:
            for vc, vs, ve in vertical:
                n = he - hs + 1
                if ve - vs + 1 != n:
                    continue
                arrangements = []
                if vs == hr + 1:
                    arrangements.append((vs, "T"))
                if ve == hr - 1:
                    arrangements.append((vs, "B"))
                for r, hside in arrangements:
                    if vc == hs - 1:
                        vside = "L"
                    elif vc == he + 1:
                        vside = "R"
                    else:
                        continue
                    c = hs
                    if r < 3 or c < 3 or r + n > h - 3 or c + n > w - 3:
                        continue
                    bounds = (r, c, n)
                    hline = _guide_line(grid, bounds, hside)
                    vline = _guide_line(grid, bounds, vside)
                    # Walk away from the missing shared-corner cell.
                    hp = _read_program(hline if vside == "L" else hline[::-1], wall, markers)
                    vp = _read_program(vline if hside == "T" else vline[::-1], wall, markers)
                    if hp is None or vp is None or len(hp) != len(vp):
                        continue
                    if any(a == b for a, b in zip(hp, vp)):
                        continue
                    permitted = np.zeros(grid.shape, dtype=bool)
                    permitted[r:r + n, c:c + n] = True
                    if hside == "T":
                        permitted[r - 1, c:c + n] = True
                    else:
                        permitted[r + n, c:c + n] = True
                    if vside == "L":
                        permitted[r:r + n, c - 1] = True
                    else:
                        permitted[r:r + n, c + n] = True
                    permitted[[0, -1], :] = True
                    permitted[:, [0, -1]] = True
                    if np.any(grid[~permitted] != bg):
                        continue
                    board = grid[r:r + n, c:c + n].copy()
                    if np.any(np.isin(board, list(markers))):
                        continue
                    walls = board == wall
                    if walls.all() or any(not np.array_equal(walls, _transform(walls, op)) for op in range(8)):
                        continue
                    if not any(int(x) not in {bg, wall} for x in np.unique(board)):
                        continue
                    operations = []
                    for a, b in zip(hp, vp):
                        match = [op for op, mapping in enumerate(SIDE_MAPS)
                                 if outer.get(mapping[hside]) == a and outer.get(mapping[vside]) == b]
                        if len(match) != 1:
                            break
                        operations.append(match[0])
                    if len(operations) != len(hp):
                        continue
                    candidates.append(dict(background=bg, wall=wall, markers=tuple(sorted(markers)),
                                           outer=outer, hside=hside, vside=vside, bounds=bounds,
                                           program=hp, vprogram=vp, operations=tuple(operations),
                                           board=board, walls=walls))
    if len(candidates) != 1:
        raise ValueError("guides must determine exactly one square board and cyclic program")
    return candidates[0]


def _predict(info, depth=None, operations=None, mode="cycle"):
    board = info["board"]
    depth = _depths(~info["walls"]) if depth is None else depth
    ops = info["operations"] if operations is None else operations
    output = board.copy()
    for d in range(int(depth.max()) + 1):
        index = d % len(ops) if mode == "cycle" else min(d, len(ops) - 1)
        layer = depth == d
        # D4-invariant walls imply D4-invariant depth classes.
        output[layer] = _transform(board, ops[index])[layer]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the input, then independently reorient every contour-depth class."""
    return _predict(_decode(input_grid))


def _coordinate_output(info, depth):
    """Validate cell transport by signed coordinates, without _transform."""
    n = info["board"].shape[0]
    result = info["board"].copy()
    occupancy = np.zeros((n, n), dtype=int)
    for d in range(int(depth.max()) + 1):
        j = d % len(info["program"])
        matrix = np.zeros((2, 2), dtype=int)
        for source, color in ((info["hside"], info["program"][j]),
                              (info["vside"], info["vprogram"][j])):
            target = next(s for s, value in info["outer"].items() if value == color)
            matrix += np.outer(NORMALS[target], NORMALS[source])
        source = np.array(np.where(depth == d))
        target = matrix @ source
        target += ((matrix.min(axis=1) < 0) * (n - 1))[:, None]
        assert np.all(depth[target[0], target[1]] == d), "a shell left its depth class"
        np.add.at(occupancy, (target[0], target[1]), 1)
        result[target[0], target[1]] = info["board"][source[0], source[1]]
    assert np.all(occupancy[~info["walls"]] == 1), "transport must be a bijection"
    assert np.all(occupancy[info["walls"]] == 0), "walls must not move"
    return result


def _shortcut_checks(info):
    depth = _depths(~info["walls"])
    answer = _predict(info, depth)
    n = depth.shape[0]
    rr, cc = np.indices(depth.shape)
    rectangle = np.minimum.reduce((rr, cc, n - 1 - rr, n - 1 - cc))
    return {
        "global_symmetry": all(not np.array_equal(answer, _transform(info["board"], op)) for op in range(8)),
        "rectangle_depth": not np.array_equal(answer, _predict(info, rectangle)),
        "four_neighbor_depth": not np.array_equal(answer, _predict(info, _depths(~info["walls"], False))),
        "no_cycle": not np.array_equal(answer, _predict(info, depth, mode="hold")),
        "phase_shift": not np.array_equal(answer, _predict(info, depth, info["operations"][1:] + info["operations"][:1])),
        "alternating": not np.array_equal(answer, _predict(info, depth, (info["operations"][0], next(op for op in info["operations"] if op != info["operations"][0])))),
    }


def _symmetric(seed_mask):
    return np.logical_or.reduce([_transform(seed_mask, op) for op in range(8)])


def _scaffold(rng, n, first):
    mask = np.zeros((n, n), dtype=bool)
    if first:
        # Internal off-diagonal islands leave a blank edge possible and expose
        # both genuine shape erosion and diagonal contact with the scaffold.
        r = int(rng.integers(2, 5))
        c = int(rng.integers(2, 6))
        if r == c:
            c += 1
        mask[r, c] = True
        if rng.random() < .5:
            mask[r, c + 1] = True
        if rng.random() < .35:
            b = int(rng.integers(1, 3))
            mask[b, b:n - b] = True
        return _symmetric(mask)
    for _ in range(int(rng.integers(0, 5))):
        kind = int(rng.integers(5))
        if kind == 0:  # Edge indentation, mirrored and rotated around the board.
            c = int(rng.integers(1, n - 2))
            mask[:int(rng.integers(1, 4)), c:min(n - 1, c + int(rng.integers(1, 4)))] = True
        elif kind == 1:  # Off-axis islands or short connected patches.
            r, c = int(rng.integers(1, n - 2)), int(rng.integers(1, n - 2))
            mask[r:r + int(rng.integers(1, 3)), c:c + int(rng.integers(1, 3))] = True
        elif kind == 2:  # A concentric wall can separate an inner room.
            b = int(rng.integers(1, max(2, n // 2 - 1)))
            mask[b, b:n - b] = True
        elif kind == 3:  # Orthogonal L-shaped wall fragments.
            r, c = int(rng.integers(1, n - 2)), int(rng.integers(1, n - 2))
            span = int(rng.integers(2, max(3, n // 2)))
            mask[r, c:min(n - 1, c + span)] = True
            mask[r:min(n - 1, r + span), c] = True
        else:  # A split through the center; even boards get paired central lines.
            mask[n // 2, :] = True
    result = _symmetric(mask)
    return result if np.any(~result) else np.zeros((n, n), dtype=bool)


def _motif(rng, walls, bg, wall, inks, blank_side=None):
    n = walls.shape[0]
    board = np.full((n, n), bg, dtype=int)
    board[walls] = wall
    allowed = ~walls
    if blank_side is not None:
        _side(allowed, blank_side)[:] = False
    coords = np.argwhere(allowed)
    density = float(rng.uniform(.06, .28))
    count = max(8, int(len(coords) * density))
    for i in rng.permutation(len(coords))[:count]:
        board[tuple(coords[i])] = int(rng.choice(inks))
    # Independent strokes and bends give geometric motifs as well as scatter.
    for _ in range(int(rng.integers(1, 5))):
        r, c = (int(v) for v in coords[int(rng.integers(len(coords)))])
        color = int(rng.choice(inks))
        dr, dc = DIR4[int(rng.integers(4))]
        length = int(rng.integers(2, max(3, n // 2)))
        for step in range(length):
            rr, cc = r + step * dr, c + step * dc
            if 0 <= rr < n and 0 <= cc < n and allowed[rr, cc]:
                board[rr, cc] = color
        if rng.random() < .5:
            r, c = r + (length - 1) * dr, c + (length - 1) * dc
            for step in range(1, int(rng.integers(2, 5))):
                rr, cc = r + step * dc, c + step * dr
                if 0 <= rr < n and 0 <= cc < n and allowed[rr, cc]:
                    board[rr, cc] = color
    # Make every sampled ink role visible, without protected corner anchors.
    for color, i in zip(inks, rng.permutation(len(coords))):
        board[tuple(coords[i])] = int(color)
    return board


def _paint_controls(grid, outer):
    for side, color in outer.items():
        _side(grid, side)[1:-1] = color


def _assemble(rng, board, bg, wall, markers, program, inner, outer):
    n = board.shape[0]
    h, w = int(rng.integers(n + 6, 31)), int(rng.integers(n + 6, 31))
    r, c = int(rng.integers(3, h - n - 2)), int(rng.integers(3, w - n - 2))
    grid = np.full((h, w), bg, dtype=int)
    grid[r:r + n, c:c + n] = board
    hside = next(s for s in inner if s in ("T", "B"))
    vside = next(s for s in inner if s in ("L", "R"))
    hp = tuple(markers[x] for x in program)
    vp = tuple(markers[1 - x] for x in program)
    for side, values, reverse in ((hside, hp, vside == "R"), (vside, vp, hside == "B")):
        line = _guide_line(grid, (r, c, n), side)
        line[:] = wall
        if reverse:
            line = line[::-1]
        line[:len(values)] = values
    _paint_controls(grid, outer)
    return grid


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    first_corner = int(rng.integers(4))
    inner = (SIDES[first_corner], SIDES[(first_corner + 1) % 4])
    hside = next(s for s in inner if s in ("T", "B"))
    vside = next(s for s in inner if s in ("L", "R"))
    qturn = int(rng.choice((1, 3)))
    words = ((0, 0, 1), (0, 1, 1), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 1, 1))
    word = words[int(rng.integers(len(words)))]
    for attempt in range(160):
        colors = [int(x) for x in rng.permutation(10)]
        bg, wall, markers = colors[0], colors[1], tuple(colors[2:4])
        inks = colors[4:4 + int(rng.integers(2, 5))]
        n = int(rng.integers(18, 25))
        walls = _scaffold(rng, n, True)
        if _depths(~walls).max() < len(word) + 1:
            continue
        board = _motif(rng, walls, bg, wall, inks, hside)
        outer = {SIDE_MAPS[qturn][hside]: markers[0], SIDE_MAPS[qturn][vside]: markers[1]}
        grid = _assemble(rng, board, bg, wall, markers, word, inner, outer)
        info = _decode(grid)
        if not all(_shortcut_checks(info).values()):
            continue
        answer = _predict(info)
        swapped = grid.copy()
        _paint_controls(swapped, {s: markers[1] if v == markers[0] else markers[0] for s, v in outer.items()})
        swapped_answer = solve(swapped)
        if np.array_equal(answer, swapped_answer):
            continue
        puzzle["train"].extend(({"input": grid, "output": answer},
                                {"input": swapped, "output": swapped_answer}))
        break
    else:
        raise RuntimeError("bounded teaching-witness construction exhausted")
    seen = {_key(pair["input"]) for pair in puzzle["train"]}
    for index in range(2, nt + ne):
        is_test = index >= nt
        for attempt in range(80):
            colors = [int(x) for x in rng.permutation(10)]
            bg, wall, markers = colors[0], colors[1], tuple(colors[2:4])
            inks = colors[4:4 + int(rng.integers(1, 5))]
            n = int(rng.integers(9, 25))
            walls = _scaffold(rng, n, False)
            if np.count_nonzero(~walls) < n * n // 3:
                continue
            board = _motif(rng, walls, bg, wall, inks)
            p = int(rng.integers(2 if is_test else 1, 6))
            # Extra training may use constant words; tests below require
            # visibly different actions at different contour depths.
            program = tuple(int(x) for x in rng.integers(0, 2, size=p))
            corner = (first_corner + int(rng.choice((2, 3) if is_test else (0, 1)))) % 4
            inner = (SIDES[corner], SIDES[(corner + 1) % 4])
            outer_corner = int(rng.integers(4))
            outer_sides = (SIDES[outer_corner], SIDES[(outer_corner + 1) % 4])
            outer_colors = markers[::-1] if rng.integers(2) else markers
            outer = dict(zip(outer_sides, outer_colors))
            grid = _assemble(rng, board, bg, wall, markers, program, inner, outer)
            if _key(grid) in seen:
                continue
            info = _decode(grid)
            answer = _predict(info)
            if is_test:
                operations = sorted(set(info["operations"]))
                if len(operations) != 2 or any(np.array_equal(answer, _transform(board, op)) for op in range(8)):
                    continue
                reverse_ops = tuple(operations[1] if op == operations[0] else operations[0]
                                    for op in info["operations"])
                if np.array_equal(answer, _predict(info, operations=reverse_ops)):
                    continue
            puzzle["test" if is_test else "train"].append({"input": grid, "output": answer})
            seen.add(_key(grid))
            break
        else:
            raise RuntimeError("bounded example construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            depth = _bfs_depths(~info["walls"])
            assert np.array_equal(depth, _depths(~info["walls"])), "erosion and boundary distance disagree"
            assert np.array_equal(out, _coordinate_output(info, depth)), "wrong contour transport"
            assert np.array_equal(out == info["wall"], info["walls"]), "scaffold changed"
            assert np.array_equal(np.sort(out.ravel()), np.sort(info["board"].ravel())), "cell colors were lost or created"
            for d in range(int(depth.max()) + 1):
                assert np.array_equal(np.sort(out[depth == d]), np.sort(info["board"][depth == d])), "a color crossed depth classes"
            analyses[split].append(info)
            if split == "test":
                operations = sorted(set(info["operations"]))
                assert len(operations) == 2, "tests must exercise both program correspondences"
                assert all(not np.array_equal(out, _transform(info["board"], op)) for op in range(8)), "test collapsed to a global symmetry"
                reverse_ops = tuple(operations[1] if op == operations[0] else operations[0]
                                    for op in info["operations"])
                assert not np.array_equal(out, _predict(info, operations=reverse_ops)), "test ignores marker identity"
    a, b = analyses["train"][:2]
    assert np.array_equal(a["board"], b["board"]) and a["bounds"] == b["bounds"]
    assert (a["hside"], a["vside"], a["program"], a["vprogram"]) == (b["hside"], b["vside"], b["program"], b["vprogram"])
    assert set(a["outer"]) == set(b["outer"]) and all(a["outer"][s] != b["outer"][s] for s in a["outer"])
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]), "marker matching must matter"
    assert all(_shortcut_checks(a).values()), "teaching example does not reject the documented shortcuts"
    assert any(np.all(_side(a["board"], side) == a["background"]) for side in SIDES), "guide-based crop witness missing"
    assert any(op in (1, 3) for op in a["operations"]) and any(op >= 4 for op in a["operations"])
    train_corners = {(x["hside"], x["vside"]) for x in analyses["train"]}
    assert all((x["hside"], x["vside"]) not in train_corners for x in analyses["test"]), "test should use a new guide corner"


def _components(mask):
    remaining = set(tuple(int(v) for v in p) for p in np.argwhere(mask))
    count = 0
    while remaining:
        count += 1
        q = [remaining.pop()]
        while q:
            r, c = q.pop()
            for dr, dc in DIR4:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    q.append(neighbor)
    return count


def self_test():
    """Bounded 42-seed audit of geometry, programs, symmetry, and mutations."""
    shapes, sizes, walls, periods, operations, regions, palettes, puzzles = (set() for _ in range(8))
    corners = {"train": set(), "test": set()}
    depths_seen, wall_counts, combinations = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], again[s]) for k in ("input", "output"))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                depth = _depths(~info["walls"])
                shapes.add(pair["input"].shape)
                sizes.add(info["board"].shape[0])
                walls.add(_key(info["walls"]))
                periods.add(len(info["program"]))
                operations.update(info["operations"])
                regions.add(_components(~info["walls"]))
                palettes.add((info["background"], info["wall"], info["markers"]))
                corners[split].add((info["hside"], info["vside"]))
                depths_seen.add(int(depth.max()))
                wall_counts.add(int(info["walls"].sum()))
                combinations.add((len(info["program"]), info["operations"], int(depth.max()), _components(~info["walls"])))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                assert all(np.array_equal(depth, _transform(depth, op)) for op in range(8))
    assert len(puzzles) == 42 and len(shapes) >= 35 and len(walls) >= 80
    assert sizes >= set(range(9, 25)) and periods == set(range(1, 6))
    assert operations == set(range(8)) and len(regions) >= 4
    assert len(palettes) >= 70 and all(len(v) == 4 for v in corners.values())
    assert len(depths_seen) >= 5 and len(wall_counts) >= 30 and len(combinations) >= 90
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1200 + 17 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for kind in ("output_cell", "crop", "guide", "scaffold"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if kind == "output_cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "crop":
            pair["output"] = pair["output"][1:, :]
        elif kind == "guide":
            line = _guide_line(pair["input"], info["bounds"], info["hside"])
            line[len(line) // 2] = info["background"]
        else:
            r, c, n = info["bounds"]
            candidates = [(rr, cc) for rr in range(n) for cc in range(n)
                          if len({(rr, cc), (cc, n - 1 - rr), (n - 1 - rr, n - 1 - cc), (n - 1 - cc, rr)}) > 1]
            rr, cc = candidates[0]
            pair["input"][r + rr, c + cc] = info["background"] if info["walls"][rr, cc] else info["wall"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError(f"mutation accepted: {kind}")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, legal counts, contour and correspondence invariants")
