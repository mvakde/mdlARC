"""A worked maze pair supplies the ink for a missing maze companion.

No evidence files, random state, training outputs, or cached examples are used
by solve. Generation writes no artifacts. All retained diagnostics are bounded.
"""
from collections import deque
from itertools import product

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _neighbors(p, shape):
    for dr, dc in _DIRECTIONS:
        q = p[0] + dr, p[1] + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _search(walls, start, goal):
    """BFS; multiplicities saturate at two, so ties cannot silently disappear."""
    distance = {start: 0}
    ways = {start: 1}
    parent = {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, walls.shape):
            if walls[q]:
                continue
            if q not in distance:
                distance[q] = distance[p] + 1
                ways[q] = ways[p]
                parent[q] = p
                queue.append(q)
            elif distance[q] == distance[p] + 1:
                ways[q] = min(2, ways[q] + ways[p])
    if goal not in distance:
        return None, 0, distance
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1], ways[goal], distance


def _terminals(maze, terminal):
    points = [tuple(map(int, p)) for p in np.argwhere(maze == terminal)]
    if len(points) != 2:
        raise ValueError("a maze must contain exactly two terminals")
    return points


def _route(maze, wall, terminal):
    a, b = _terminals(maze, terminal)
    path, count, _ = _search(maze == wall, a, b)
    if count != 1 or len(path) < 3:
        raise ValueError("a nontrivial unique shortest route is required")
    return path


def _paint(maze, path, ink):
    out = maze.copy()
    for p in path[1:-1]:
        out[p] = ink
    return out


def _longer_alternative(walls, path):
    # Blocking an interior route cell certifies an alternate when one survives.
    for p in path[1:-1]:
        blocked = walls.copy()
        blocked[p] = True
        other, _, _ = _search(blocked, path[0], path[-1])
        if other is not None:
            assert len(other) > len(path)
            return other
    return None


def _turns(path):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _components(a):
    seen = np.zeros(a.shape, dtype=bool)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if seen[r, c]:
                continue
            color = int(a[r, c])
            points, queue = [], [(r, c)]
            seen[r, c] = True
            while queue:
                p = queue.pop()
                points.append(p)
                for q in _neighbors(p, a.shape):
                    if not seen[q] and a[q] == color:
                        seen[q] = True
                        queue.append(q)
            yield color, points


def _frames(a):
    """Isolated monochrome outline components, in arbitrary colors."""
    frames = {}
    for color, points in _components(a):
        r0, c0 = map(int, np.min(points, axis=0))
        r1, c1 = map(int, np.max(points, axis=0))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if (h >= 7 and w >= 7 and len(points) == 2 * (h + w) - 4
                and all(r in (r0, r1) or c in (c0, c1) for r, c in points)):
            frames.setdefault(color, []).append((r0, c0, h, w, color))
    return frames


def _inside(a, b):
    return (b[0] < a[0] and b[1] < a[1]
            and a[0] + a[2] < b[0] + b[2]
            and a[1] + a[3] < b[1] + b[3])


def _disjoint(a, b):
    return (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
            or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _crop(a, box):
    r, c, h, w = box[:4]
    return a[r + 1:r + h - 1, c + 1:c + w - 1].copy()


def _extract(a):
    frames = _frames(a)
    scenes = []
    for color, panels in frames.items():
        if len(panels) != 4:
            continue
        panels = sorted(panels)
        groups = []
        for other, boxes in frames.items():
            if other == color:
                continue
            for box in boxes:
                members = tuple(i for i, p in enumerate(panels) if _inside(p, box))
                if (len(members) == 2 and _disjoint(panels[members[0]], panels[members[1]])
                        and panels[members[0]][2:4] == panels[members[1]][2:4]):
                    groups.append((box, members))
        for i, (box0, members0) in enumerate(groups):
            for box1, members1 in groups[i + 1:]:
                if not _disjoint(box0, box1) or set(members0) & set(members1):
                    continue
                contents = [[_crop(a, panels[j]) for j in members]
                            for members in (members0, members1)]
                blanks = [(g, s) for g in range(2) for s in range(2)
                          if np.all(contents[g][s] == contents[g][s][0, 0])]
                if len(blanks) != 1:
                    continue
                g, s = blanks[0]
                scenes.append(dict(reference=contents[1 - g], target=contents[g][1 - s],
                                   background=int(contents[g][s][0, 0]),
                                   panels=panels, boxes=[box0, box1],
                                   reference_boxes=[panels[j] for j in (members0, members1)[1 - g]],
                                   target_box=panels[(members0, members1)[g][1 - s]],
                                   blank_box=panels[(members0, members1)[g][s]]))
    if len(scenes) != 1:
        raise ValueError("expected one grouping of four panels and one blank")
    return scenes[0]


def _relation(scene):
    """The complete pair supplies a supported route relation and its ink."""
    background = scene["background"]
    candidates = []
    for source, completed in (scene["reference"], scene["reference"][::-1]):
        colors = set(map(int, np.unique(source)))
        changed = source != completed
        if len(colors) != 3 or background not in colors or not np.any(changed):
            continue
        old, new = np.unique(source[changed]), np.unique(completed[changed])
        if len(old) != 1 or old[0] != background or len(new) != 1 or int(new[0]) in colors:
            continue
        terminal_options = [c for c in colors - {background} if np.count_nonzero(source == c) == 2]
        if len(terminal_options) != 1:
            continue
        terminal = terminal_options[0]
        wall = next(iter(colors - {background, terminal}))
        try:
            path = _route(source, wall, terminal)
        except ValueError:
            continue
        ink = int(new[0])
        if np.array_equal(_paint(source, path, ink), completed):
            candidates.append((source, completed, wall, terminal, ink, path))
    if len(candidates) != 1:
        raise ValueError("the complete pair must demonstrate exactly one route relation")
    source, completed, wall, terminal, ink, path = candidates[0]
    target = scene["target"]
    if set(map(int, np.unique(target))) != {background, wall, terminal}:
        raise ValueError("reference and target maze roles must correspond")
    return source, completed, wall, terminal, ink, path


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the reconstructed blank interior, using this input alone."""
    scene = _extract(_grid(input_grid))
    _, _, wall, terminal, ink, _ = _relation(scene)
    target = scene["target"]
    return _paint(target, _route(target, wall, terminal), ink)


def _candidate_walls(rng, shape, mode):
    h, w = shape
    walls = np.zeros(shape, dtype=bool)
    if mode == "scatter":
        return rng.random(shape) < rng.uniform(.23, .43)
    if mode == "island":
        r0 = int(rng.integers(1, h - 2))
        r1 = int(rng.integers(r0 + 2, h))
        c0 = int(rng.integers(1, w - 1))
        c1 = int(rng.integers(c0 + 1, w))
        walls[r0:r1, c0:c1] = True
    else:
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            length, thickness = int(rng.integers(2, max(h, w))), int(rng.integers(1, 3))
            if rng.integers(2):
                walls[r:min(h, r + length), c:min(w, c + thickness)] = True
            else:
                walls[r:min(h, r + thickness), c:min(w, c + length)] = True
    return walls


def _make_maze(rng, shape, mode):
    """Logical roles: 0 background, 1 wall, 2 terminal. Bounded rejection."""
    for _ in range(128):
        walls = _candidate_walls(rng, shape, mode)
        free = [tuple(map(int, p)) for p in np.argwhere(~walls)]
        if walls.sum() < 3 or len(free) < 8:
            continue
        for _ in range(8):
            i, j = rng.choice(len(free), 2, replace=False)
            start, goal = free[int(i)], free[int(j)]
            path, count, _ = _search(walls, start, goal)
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if (count != 1 or len(path) < 6 or len(path) - 1 <= manhattan
                    or _turns(path) < 2 or len(free) <= len(path) + 1):
                continue
            if _longer_alternative(walls, path) is None:
                continue
            maze = walls.astype(np.int8)
            maze[start] = maze[goal] = 2
            return maze
    # A vertical interior slit has a unique short route around its top and a
    # strictly longer route around its bottom. This is the same rule, not a retry
    # with weakened uniqueness or a stored puzzle. All requested shapes >= 5.
    h, w = shape
    maze = np.zeros(shape, dtype=np.int8)
    c = w // 2
    maze[1:h - 1, c] = 1
    maze[1, c - 1] = maze[1, c + 1] = 2
    return maze


def _outline(a, box, color):
    r, c, h, w = box[:4]
    a[r, c:c + w] = a[r + h - 1, c:c + w] = color
    a[r:r + h, c] = a[r:r + h, c + w - 1] = color


def _layout(rng, shapes):
    # Choose group orientations and a packing axis jointly for feasibility;
    # neither determines reference direction or target role.
    choices = []
    for axes in product((0, 1), repeat=2):
        dims = []
        for shape, axis in zip(shapes, axes):
            ph, pw = shape[0] + 2, shape[1] + 2
            dims.append([2 * ph + 3, pw + 2] if axis == 0 else [ph + 2, 2 * pw + 3])
        for packing in (0, 1):
            minimum = [max(d[k] for d in dims) if k != packing
                       else sum(d[k] for d in dims) + 1 for k in (0, 1)]
            if max(minimum) <= 30:
                choices.append((axes, dims, packing, minimum))
    axes, dims, packing, minimum = choices[int(rng.integers(len(choices)))]
    dims = [d.copy() for d in dims]
    canvas_shape = [int(rng.integers(n, 31)) for n in minimum]
    allocations = rng.multinomial(canvas_shape[packing] - minimum[packing], [0.2] * 5)
    dims[0][packing] += int(allocations[0])
    dims[1][packing] += int(allocations[1])
    starts = [int(allocations[2]), int(allocations[2]) + dims[0][packing] + 1 + int(allocations[3])]
    boxes, panels = [], []
    for i, (shape, axis) in enumerate(zip(shapes, axes)):
        cross = 1 - packing
        dims[i][cross] = int(rng.integers(dims[i][cross], canvas_shape[cross] + 1))
        origin = [0, 0]
        origin[packing] = starts[i]
        origin[cross] = int(rng.integers(canvas_shape[cross] - dims[i][cross] + 1))
        boxes.append(tuple(origin + dims[i]))
        phw = [shape[0] + 2, shape[1] + 2]
        gap = int(rng.integers(1, dims[i][axis] - 2 * phw[axis] - 1))
        first = int(rng.integers(1, dims[i][axis] - 2 * phw[axis] - gap))
        positions = []
        for along in (first, first + phw[axis] + gap):
            p = origin.copy()
            p[axis] += along
            p[1 - axis] += int(rng.integers(1, dims[i][1 - axis] - phw[1 - axis]))
            positions.append(tuple(p + phw))
        panels.append(positions)
    return tuple(canvas_shape), boxes, panels


def _assemble(rng, reference, target, colors, ink):
    background, wall, terminal = colors
    available = [c for c in range(10) if c not in (*colors, ink)]
    frame, group0, group1, paper0, paper1 = map(int, rng.choice(available, 5, replace=False))
    ref = np.array(colors, dtype=np.int8)[reference]
    tgt = np.array(colors, dtype=np.int8)[target]
    completed = _paint(ref, _route(ref, wall, terminal), ink)
    output = _paint(tgt, _route(tgt, wall, terminal), ink)
    groups = [[ref, completed], [tgt, np.full(tgt.shape, background, dtype=np.int8)]]
    for group in groups:
        if rng.integers(2):
            group.reverse()
    if rng.integers(2):
        groups.reverse()
    shape, boxes, panels = _layout(rng, [g[0].shape for g in groups])
    rr, cc = np.indices(shape)
    pattern = (rr // int(rng.integers(1, 3)) + cc // int(rng.integers(1, 3)) + int(rng.integers(2))) % 2
    canvas = np.where(pattern, paper0, paper1).astype(np.int8)
    for box, local, group, color in zip(boxes, panels, groups, (group0, group1)):
        _outline(canvas, box, color)
        for panel, content in zip(local, group):
            _outline(canvas, panel, frame)
            r, c, h, w = panel
            canvas[r + 1:r + h - 1, c + 1:c + w - 1] = content
    return {"input": canvas, "output": output}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.choice(10, 3, replace=False)))
    inks = rng.permutation([c for c in range(10) if c not in colors])
    modes = ("island", "segments", "scatter")
    shared_shape = tuple(map(int, rng.integers(5, 9, size=2)))
    shared_target = _make_maze(rng, shared_shape, str(rng.choice(modes)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            ink = int(inks[i % len(inks)]) if split == "train" else int(rng.choice(inks))
            if split == "train" and i < 2:
                target = shared_target.copy()
            else:
                shape = tuple(map(int, rng.integers(8, 11, size=2) if split == "test"
                                      else rng.integers(5, 9, size=2)))
                if split == "test" and max(shape) == 8:
                    shape = (9, shape[1])
                target = _make_maze(rng, shape, str(rng.choice(modes)))
            # The reference is always a fresh maze, never a copied route mask.
            for attempt in range(16):
                shape = tuple(map(int, rng.integers(5, 9, size=2)))
                reference = _make_maze(rng, shape, modes[(i + attempt) % 3])
                if _key(reference) == _key(target):
                    continue
                pair = _assemble(rng, reference, target, colors, ink)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct reference and target scenes")
    return puzzle


def _check_maze(maze, background, wall, terminal):
    assert np.count_nonzero(maze == wall) >= 3
    path = _route(maze, wall, terminal)
    assert all(maze[p] == background for p in path[1:-1])
    distance = abs(path[0][0] - path[-1][0]) + abs(path[0][1] - path[-1][1])
    assert len(path) - 1 > distance and _turns(path) >= 2
    assert np.count_nonzero(maze == background) > len(path) - 2
    assert _longer_alternative(maze == wall, path) is not None
    return path


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            scene = _extract(inp)
            source, completed, wall, terminal, ink, _ = _relation(scene)
            background, target = scene["background"], scene["target"]
            roles.add((background, wall, terminal))
            assert _key(source) != _key(target)
            assert out.shape == target.shape
            _check_maze(source, background, wall, terminal)
            path = _check_maze(target, background, wall, terminal)
            # Distance-sum characterization checks all geodesic cells, rather
            # than relying only on the parent-pointer reconstruction.
            _, count, d0 = _search(target == wall, path[0], path[-1])
            _, _, d1 = _search(target == wall, path[-1], path[0])
            assert count == 1
            expected_mask = np.zeros(target.shape, dtype=bool)
            for p in d0.keys() & d1.keys():
                expected_mask[p] = d0[p] + d1[p] == len(path) - 1 and target[p] == background
            assert np.array_equal(out != target, expected_mask)
            assert np.all(out[expected_mask] == ink)
            assert np.array_equal(out[~expected_mask], target[~expected_mask])
            assert np.array_equal(solve(inp), out)
            assert completed.shape == source.shape
            if split == "train":
                assert max(target.shape) <= 8
                training.append((_key(target), ink))
            else:
                assert 9 <= max(target.shape) <= 10
    assert len(roles) == 1
    assert training[0][0] == training[1][0] and training[0][1] != training[1][1]


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def _diagnostics():
    pair = generate_puzzle(917, 2, 1)["test"][0]
    inp, out = pair["input"], pair["output"]
    original = inp.copy()
    scene = _extract(inp)
    _, _, wall, terminal, ink, _ = _relation(scene)
    replacement = next(c for c in range(10) if c not in set(map(int, np.unique(inp))))
    twin = inp.copy()
    # Only the informative completed-reference ink changes. The entire target,
    # blank, framing, terminals and reference maze stay pixel-identical.
    twin[twin == ink] = replacement
    expected = np.where(out == ink, replacement, out).astype(np.int8)
    assert np.array_equal(solve(twin), expected) and not np.array_equal(out, expected)
    for k in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
    assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
    # Complete-pair order has no semantics, including when it opposes the target.
    swapped = inp.copy()
    a, b = scene["reference_boxes"]
    for box, content in ((a, _crop(inp, b)), (b, _crop(inp, a))):
        r, c, h, w = box[:4]
        swapped[r + 1:r + h - 1, c + 1:c + w - 1] = content
    assert np.array_equal(solve(swapped), out)
    assert np.array_equal(inp, original)
    target = scene["target"]
    path = _route(target, wall, terminal)
    other = _longer_alternative(target == wall, path)
    assert not np.array_equal(_paint(target, other, ink), out)
    assert not np.array_equal(np.where(target == scene["background"], ink, target), out)
    # Destroying reference support must be detected, not silently ignored.
    broken = inp.copy()
    broken[broken == ink] = scene["background"]
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("an uninformative complete pair was accepted")


def self_test():
    """42 seeds, streaming aggregates; no grids or execution traces saved."""
    shapes, maze_shapes, wall_counts, lengths, turns, palettes, structures = (set() for _ in range(7))
    orientations, endpoint_kinds, layouts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _extract(pair["input"])
                _, _, wall, terminal, ink, _ = _relation(scene)
                target = scene["target"]
                path = _route(target, wall, terminal)
                shapes.add(pair["input"].shape)
                maze_shapes.add(target.shape)
                wall_counts.add(int(np.count_nonzero(target == wall)))
                lengths.add(len(path) - 1)
                turns.add(_turns(path))
                palettes.add((scene["background"], wall, terminal, ink))
                structures.add(_key((target == wall).astype(np.uint8)))
                h, w = target.shape
                endpoint_kinds.add(tuple(sorted(0 if r in (0, h - 1) or c in (0, w - 1) else 1
                                                  for r, c in (path[0], path[-1]))))
                for box in scene["boxes"]:
                    members = [p for p in scene["panels"] if _inside(p, box)]
                    orientations.add("h" if _disjoint((0, members[0][1], 1, members[0][3]),
                                                       (0, members[1][1], 1, members[1][3])) else "v")
                layouts.add(tuple(tuple(b[:4]) for b in scene["boxes"]))
    assert len(shapes) >= 15 and len(maze_shapes) >= 15
    assert len(wall_counts) >= 15 and len(lengths) >= 12 and len(turns) >= 4
    assert len(palettes) >= 60 and len(structures) >= 100 and len(layouts) >= 150
    assert orientations == {"h", "v"} and len(endpoint_kinds) == 3
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(42 + a * 10 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    assert _same(puzzle, generate_puzzle(0))
    assert not _same(puzzle, generate_puzzle(1))
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    _diagnostics()
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(shapes),
            "target_shapes": len(maze_shapes), "wall_cell_counts": len(wall_counts),
            "route_lengths": len(lengths), "turn_counts": len(turns),
            "role_palettes": len(palettes), "wall_masks": len(structures),
            "layouts": len(layouts), "independent_semantic_review": False}
