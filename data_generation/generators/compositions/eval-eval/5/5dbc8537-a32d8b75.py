"""Seam wave: infer a rectangle assembly, then texture its hop-distance phases.

No parent generator, external files, latent seed information, or training labels
are used by solve. Only standard-library modules and NumPy are required.
"""
from collections import Counter, deque
import numpy as np


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(int, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        component, stack = [], [start]
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _contact_graph(rectangles):
    """A positive shared side is an edge; a point contact is not."""
    graph = [set() for _ in rectangles]
    for i, (y, x, h, w) in enumerate(rectangles):
        for j, (a, b, c, d) in enumerate(rectangles[:i]):
            vertical = ((y + h == a or a + c == y)
                        and max(x, b) < min(x + w, b + d))
            horizontal = ((x + w == b or b + d == x)
                          and max(y, a) < min(y + h, a + c))
            if vertical or horizontal:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _distances(graph, root):
    distance = [-1] * len(graph)
    distance[root] = 0
    queue = deque([root])
    while queue:
        v = queue.popleft()
        for u in sorted(graph[v]):
            if distance[u] < 0:
                distance[u] = distance[v] + 1
                queue.append(u)
    if -1 in distance:
        raise ValueError("The assembly is disconnected")
    return distance


def _packing(mask, dimensions, max_nodes=None):
    """Find at most two boundary decompositions, grouping identical copies.

    Dead-state memoization is used only for states with no completion. Successful
    states are not merged: different earlier seams can affect the later wave.
    A finite max_nodes is used only to reject expensive generation attempts.
    Input-only solving is exhaustive and has no sampling budget.
    """
    h, w = mask.shape
    coordinates = list(map(tuple, np.argwhere(mask)))
    if not coordinates:
        return [], 0
    index = {p: i for i, p in enumerate(coordinates)}
    shapes = list(dict.fromkeys(dimensions))
    remaining = [dimensions.count(s) for s in shapes]
    areas = [a * b for a, b in shapes]
    placements, by_cell = [], [[] for _ in coordinates]
    for group, (rh, rw) in enumerate(shapes):
        for y in range(h - rh + 1):
            for x in range(w - rw + 1):
                if not np.all(mask[y:y + rh, x:x + rw]):
                    continue
                ids = [index[y + dy, x + dx]
                       for dy in range(rh) for dx in range(rw)]
                bits = sum(1 << i for i in ids)
                p = len(placements)
                placements.append((group, bits, (y, x, rh, rw)))
                for i in ids:
                    by_cell[i].append(p)
    full = (1 << len(coordinates)) - 1
    solutions, dead, nodes = [], set(), 0

    def search(covered, path):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise ValueError("Generation packing budget exceeded")
        if covered == full:
            solutions.append(tuple(sorted(path)))
            return True
        state = covered, tuple(remaining)
        if state in dead:
            return False
        if sum(n * a for n, a in zip(remaining, areas)) < (full ^ covered).bit_count():
            dead.add(state)
            return False
        options = None
        for i in range(len(coordinates)):
            if covered >> i & 1:
                continue
            legal = [p for p in by_cell[i]
                     if remaining[placements[p][0]] > 0
                     and not (covered & placements[p][1])]
            if not legal:
                dead.add(state)
                return False
            if options is None or len(legal) < len(options):
                options = legal
            if len(legal) == 1:
                break
        found = False
        for p in options:
            group, bits, rectangle = placements[p]
            remaining[group] -= 1
            found |= search(covered | bits, path + [rectangle])
            remaining[group] += 1
            if len(solutions) >= 2:
                return True
        if not found:
            dead.add(state)
        return found

    search(0, [])
    return solutions, nodes


def _tray_objects(tray):
    vacancy = int(tray[0, 0])
    chips, texture, chip_color = [], None, None
    for component in _components(tray != vacancy):
        rows, columns = zip(*component)
        y, x = min(rows), min(columns)
        h, w = max(rows) - y + 1, max(columns) - x + 1
        block = tray[y:y + h, x:x + w]
        if len(component) != h * w or np.any(block == vacancy):
            raise ValueError("Source objects must be complete filled rectangles")
        # A background moat includes diagonals. This also protects the binary tile.
        for yy, xx in component:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    a, b = yy + dy, xx + dx
                    if (0 <= a < len(tray) and 0 <= b < tray.shape[1]
                            and tray[a, b] != vacancy
                            and not (y <= a < y + h and x <= b < x + w)):
                        raise ValueError("Source components touch diagonally")
        colors = np.unique(block)
        if len(colors) == 1:
            c = int(colors[0])
            if chip_color is not None and c != chip_color:
                raise ValueError("Solid chips must share one color")
            chip_color = c
            chips.append((h, w))
        elif len(colors) == 2 and h == w and h >= 2 and texture is None:
            texture = block.copy()
        else:
            raise ValueError("Exactly one complete binary square is required")
    if not chips or texture is None or chip_color in texture:
        raise ValueError("Missing or ambiguous source roles")
    if len(np.unique(tray)) != 4:
        raise ValueError("The tray must have four distinct local color roles")
    return vacancy, chip_color, chips, texture


def _decode(input_grid):
    grid = _grid(input_grid)
    answers = []
    for axis in (0, 1):
        extent = grid.shape[axis]
        for cut in range(1, extent - 1):
            line = grid[cut, :] if axis == 0 else grid[:, cut]
            divider = int(line[0])
            if not np.all(line == divider) or np.count_nonzero(grid == divider) != line.size:
                continue
            a, b = ((grid[:cut], grid[cut + 1:]) if axis == 0
                    else (grid[:, :cut], grid[:, cut + 1:]))
            for board, tray, first in ((a, b, True), (b, a, False)):
                try:
                    vacancy, chip, dimensions, texture = _tray_objects(tray)
                    colors = set(map(int, np.unique(board)))
                    background = colors - {vacancy, chip}
                    if len(colors) != 3 or len(background) != 1 or np.count_nonzero(board == chip) != 1:
                        continue
                    background = next(iter(background))
                    roles = {vacancy, chip, background, divider, *map(int, np.unique(texture))}
                    if len(roles) != 6 or background in tray:
                        continue
                    mask = (board == vacancy) | (board == chip)
                    if len(_components(mask)) != 1 or max(board.shape) * len(texture) > 30:
                        continue
                    side = ("top" if first else "bottom") if axis == 0 else ("left" if first else "right")
                    answers.append(dict(board=board.copy(), tray=tray.copy(), mask=mask,
                                        vacancy=vacancy, chip=chip, background=background,
                                        divider=divider, dimensions=dimensions, texture=texture,
                                        marker=tuple(map(int, np.argwhere(board == chip)[0])), side=side))
                except ValueError:
                    continue
    if len(answers) != 1:
        raise ValueError("Input must identify one visible destination/tray split")
    return answers[0]


def _analysis(input_grid):
    info = _decode(input_grid)
    solutions, nodes = _packing(info["mask"], info["dimensions"])
    if len(solutions) != 1:
        raise ValueError("The finite inventory must give exactly one rectangle decomposition")
    rectangles = solutions[0]
    my, mx = info["marker"]
    roots = [i for i, (y, x, h, w) in enumerate(rectangles)
             if y <= my < y + h and x <= mx < x + w]
    if len(roots) != 1:
        raise ValueError("The root must belong to one placed rectangle")
    graph = _contact_graph(rectangles)
    distances = _distances(graph, roots[0])
    info.update(rectangles=rectangles, root=roots[0], graph=graph,
                distances=distances, nodes=nodes)
    return info


def _render(info, phases=None, texture=None):
    texture = info["texture"] if texture is None else texture
    n = len(texture)
    h, w = info["board"].shape
    out = np.full((h * n, w * n), info["background"], int)
    a, b = map(int, np.unique(texture))
    exchanged = a + b - texture
    phases = [d % 2 for d in info["distances"]] if phases is None else phases
    for (y, x, rh, rw), phase in zip(info["rectangles"], phases):
        out[y * n:(y + rh) * n, x * n:(x + rw) * n] = np.tile(
            exchanged if phase else texture, (rh, rw))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions, rectangle boundaries and phases from this input."""
    return _render(_analysis(input_grid))


def _pixel_wave(info):
    """A-ablation: count vacancy-cell edges instead of inferred rectangle hops."""
    mask, marker, texture = info["mask"], info["marker"], info["texture"]
    distance = np.full(mask.shape, -1, int)
    distance[marker] = 0
    queue = deque([marker])
    while queue:
        y, x = queue.popleft()
        for a, b in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (0 <= a < len(mask) and 0 <= b < mask.shape[1]
                    and mask[a, b] and distance[a, b] < 0):
                distance[a, b] = distance[y, x] + 1
                queue.append((a, b))
    n = len(texture)
    out = np.full(tuple(s * n for s in mask.shape), info["background"], int)
    colors = np.unique(texture)
    for y, x in np.argwhere(mask):
        block = texture if distance[y, x] % 2 == 0 else int(colors.sum()) - texture
        out[y * n:(y + 1) * n, x * n:(x + 1) * n] = block
    return out


def _alternatives(info):
    n = len(info["rectangles"])
    yield "no_exchange", _render(info, [0] * n)
    yield "all_exchange", _render(info, [1] * n)
    yield "one_hop", _render(info, [int(d > 0) for d in info["distances"]])
    yield "degree", _render(info, [len(g) % 2 for g in info["graph"]])
    order = sorted(range(n), key=lambda i: (
        info["dimensions"].index(info["rectangles"][i][2:]), info["rectangles"][i]))
    phases = [0] * n
    for rank, i in enumerate(order):
        phases[i] = rank % 2
    yield "tray_order", _render(info, phases)
    yield "pixel_distance", _pixel_wave(info)
    yield "rotate_texture", _render(info, texture=np.rot90(info["texture"]))
    yield "reflect_texture", _render(info, texture=info["texture"][:, ::-1])


def _triangle(graph):
    return any(graph[i] & graph[j] for i in range(len(graph)) for j in graph[i])


def _texture(rng, n, colors):
    for _ in range(100):
        tile = rng.choice(colors, size=(n, n)).astype(int)
        if (len(np.unique(tile)) == 2
                and all(not np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3))
                and not np.array_equal(tile, tile[:, ::-1])
                and not np.array_equal(tile, tile[::-1])):
            return tile
    tile = np.full((n, n), colors[0], int)
    tile[0, 0] = colors[1]
    return tile


def _grow(rng, h, w, count, allow_repeat, force_repeat):
    mask = np.zeros((h, w), bool)
    rectangles = []
    max_dim = min(4 if count >= 7 else 5, h - 2, w - 2)
    pool = [(a, b) for a in range(1, max_dim + 1) for b in range(1, max_dim + 1)]
    for i in range(count):
        used = [r[2:] for r in rectangles]
        if force_repeat and i == 1:
            rh, rw = used[0]
        elif allow_repeat and rectangles and rng.random() < .18:
            rh, rw = used[int(rng.integers(len(used)))]
        else:
            choices = pool if allow_repeat else [d for d in pool if d not in used]
            if not choices:
                return None
            rh, rw = choices[int(rng.integers(len(choices)))]
        options, weights = [], []
        multi_contact_preference = float(rng.uniform(1, 10))
        for y in range(h - rh + 1):
            for x in range(w - rw + 1):
                if np.any(mask[y:y + rh, x:x + rw]):
                    continue
                neighbors = _contact_graph(rectangles + [(y, x, rh, rw)])[-1]
                if not rectangles or neighbors:
                    options.append((y, x, rh, rw))
                    weights.append(1 if len(neighbors) < 2 else multi_contact_preference)
        if not options:
            return None
        probabilities = np.asarray(weights, float)
        probabilities /= probabilities.sum()
        rectangle = options[int(rng.choice(len(options), p=probabilities))]
        rectangles.append(rectangle)
        y, x, rh, rw = rectangle
        mask[y:y + rh, x:x + rw] = True
    if np.all(mask):
        return None
    return mask, tuple(sorted(rectangles))


def _scatter(rng, tiles, shape, vacancy):
    h, w = shape
    if any(t.shape[0] > h or t.shape[1] > w for t in tiles):
        return None
    for _ in range(12):
        occupied = np.zeros(shape, bool)
        positions = {}
        # Sorting is a fit aid only. Coordinates are sampled independently.
        order = sorted(rng.permutation(len(tiles)), key=lambda i: -tiles[i].size)
        for i in order:
            rh, rw = tiles[i].shape
            legal = []
            for y in range(h - rh + 1):
                for x in range(w - rw + 1):
                    if y == 0 and x == 0:
                        continue
                    if not occupied[max(0, y - 1):min(h, y + rh + 1),
                                    max(0, x - 1):min(w, x + rw + 1)].any():
                        legal.append((y, x))
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            positions[i] = (y, x)
            occupied[y:y + rh, x:x + rw] = True
        if len(positions) == len(tiles):
            tray = np.full(shape, vacancy, int)
            for i, (y, x) in positions.items():
                rh, rw = tiles[i].shape
                tray[y:y + rh, x:x + rw] = tiles[i]
            return tray
    return None


def _make_pair(rng, side, n, count, allow_repeat, force_repeat, surplus, witness):
    for _ in range(220):
        max_extent = min(13, 30 // n)
        minimum = max(7, max_extent - 2) if count >= 7 else 7
        h, w = map(int, rng.integers(minimum, max_extent + 1, size=2))
        grown = _grow(rng, h, w, count, allow_repeat, force_repeat)
        if grown is None:
            continue
        mask, rectangles = grown
        dimensions = [r[2:] for r in rectangles]
        try:
            solutions, _ = _packing(mask, dimensions, max_nodes=6000)
        except ValueError:
            continue
        if solutions != [rectangles]:
            continue
        graph = _contact_graph(rectangles)
        roots = []
        for root, rectangle in enumerate(rectangles):
            distance = _distances(graph, root)
            if rectangle[2] * rectangle[3] > 1 and (not witness or max(distance) >= 2):
                roots.append(root)
        if not roots or (witness and not _triangle(graph)):
            continue
        root = roots[int(rng.integers(len(roots)))]
        distance = _distances(graph, root)
        # Local roles are independent of geometry and of the preceding example.
        background, vacancy, chip, divider, a, b = map(int, rng.choice(10, 6, replace=False))
        tile = _texture(rng, n, [a, b])
        inventory = dimensions.copy()
        if surplus:
            candidates = [(rh, rw) for rh in range(1, 7) for rw in range(1, 7)
                          if (rh, rw) not in inventory]
            rng.shuffle(candidates)
            # Prefer locally plausible candidates; their full completion is checked.
            def local_fit(d):
                rh, rw = d
                return any(mask[y:y + rh, x:x + rw].all()
                           for y in range(h - rh + 1) for x in range(w - rw + 1))
            if rng.random() < .8:
                candidates.sort(key=lambda d: not local_fit(d))
            desired = int(rng.integers(1, 3))
            added = 0
            for candidate in candidates:
                try:
                    augmented, _ = _packing(mask, inventory + [candidate], max_nodes=6000)
                except ValueError:
                    continue
                if augmented == [rectangles]:
                    inventory.append(candidate)
                    added += 1
                    if added == desired:
                        break
            if not added:
                continue
        board = np.full(mask.shape, background, int)
        board[mask] = vacancy
        ry, rx, rh, rw = rectangles[root]
        marker = ry + int(rng.integers(rh)), rx + int(rng.integers(rw))
        board[marker] = chip
        source_tiles = [np.full(d, chip, int) for d in inventory] + [tile]
        shape = (h, 29 - w) if side in ("left", "right") else (29 - h, w)
        tray = _scatter(rng, source_tiles, shape, vacancy)
        if tray is None:
            continue
        # Trim only surplus outer tray extent, keeping independent blank slack.
        rows, columns = np.nonzero(tray != vacancy)
        if side in ("left", "right"):
            low = max(columns) + 1
            tray = tray[:, :int(rng.integers(low, tray.shape[1] + 1))]
            line = np.full((h, 1), divider, int)
            parts = [board, line, tray] if side == "left" else [tray, line, board]
            inp = np.concatenate(parts, axis=1)
        else:
            low = max(rows) + 1
            tray = tray[:int(rng.integers(low, len(tray) + 1))]
            line = np.full((1, w), divider, int)
            parts = [board, line, tray] if side == "top" else [tray, line, board]
            inp = np.concatenate(parts, axis=0)
        forward = dict(board=board, background=background, rectangles=rectangles,
                       texture=tile, distances=distance)
        out = _render(forward)
        try:
            recovered = _analysis(inp)
        except ValueError:
            continue
        if not np.array_equal(_render(recovered), out):
            continue
        if witness and any(np.array_equal(candidate, out) for _, candidate in _alternatives(recovered)):
            continue
        return {"input": inp.copy(), "output": out.copy()}
    raise RuntimeError("No unique, informative seam-wave construction in bounded attempts")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    allow_repeat = bool(rng.random() < .65)
    allow_surplus = bool(rng.random() < .55)
    sides = list(rng.permutation(["left", "right", "top", "bottom"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", nt), ("test", nq)):
        for i in range(total):
            training = split == "train"
            witness = training and i < 2
            if regime == 0:
                n = 2 if training else int(rng.integers(3, 5))
                count = int(rng.integers(4 if witness else 3, 7))
            elif regime == 1:
                n = int(rng.integers(2, 4))
                count = int(rng.integers(4 if witness else 3, 6)) if training else int(rng.integers(6, 10))
            else:
                n = int(rng.integers(2, 4))
                count = int(rng.integers(4 if witness else 3, 8))
            side = str(rng.choice(sides[:2] if training else sides[2:])) if regime == 2 else str(rng.choice(sides))
            surplus = allow_surplus and ((training and i == 0) or rng.random() < .5)
            force_repeat = allow_repeat and training and i == 1
            for _ in range(8):
                pair = _make_pair(rng, side, n, count, allow_repeat, force_repeat, surplus, witness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        assert _key(inp) not in seen, "Duplicate input"
        seen.add(_key(inp))
        info = _analysis(inp)
        n = len(info["texture"])
        assert out.shape == tuple(s * n for s in info["board"].shape)
        assert np.array_equal(out, _render(info)), "Incorrect seam-wave output"
        used = Counter(r[2:] for r in info["rectangles"])
        available = Counter(info["dimensions"])
        assert all(available[d] >= count for d, count in used.items())
        # Verify blocks directly after inference, including root and full margins.
        owner = np.full(info["mask"].shape, -1, int)
        for i, (y, x, h, w) in enumerate(info["rectangles"]):
            assert np.all(owner[y:y + h, x:x + w] == -1)
            owner[y:y + h, x:x + w] = i
        assert np.array_equal(owner >= 0, info["mask"])
        first, second = map(int, np.unique(info["texture"]))
        for y in range(len(owner)):
            for x in range(owner.shape[1]):
                block = out[y * n:(y + 1) * n, x * n:(x + 1) * n]
                i = int(owner[y, x])
                if i < 0:
                    assert np.all(block == info["background"])
                else:
                    expected = info["texture"] if info["distances"][i] % 2 == 0 else first + second - info["texture"]
                    assert np.array_equal(block, expected)
        info.update(repeated=any(v > 1 for v in used.values()),
                    surplus=len(info["dimensions"]) - len(info["rectangles"]))
        records.append(info)
    train, test = records[:nt], records[nt:]
    for pair, info in zip(puzzle["train"][:2], train[:2]):
        root = info["rectangles"][info["root"]]
        assert root[2] * root[3] > 1 and max(info["distances"]) >= 2
        assert _triangle(info["graph"]), "Training must teach shortest routes in cycles"
        assert any(info["distances"][i] == info["distances"][j]
                   for i in range(len(info["graph"])) for j in info["graph"][i])
        for name, candidate in _alternatives(info):
            assert not np.array_equal(candidate, pair["output"]), f"Training admits {name} shortcut"
    assert not any(i["repeated"] for i in test) or any(i["repeated"] for i in train)
    assert not any(i["surplus"] for i in test) or any(i["surplus"] for i in train)
    # Verify that one of the three promised structural shifts is actually visible.
    scale = min(len(i["texture"]) for i in test) > max(len(i["texture"]) for i in train)
    packing = min(len(i["rectangles"]) for i in test) > max(len(i["rectangles"]) for i in train)
    side = not ({i["side"] for i in train} & {i["side"] for i in test})
    assert scale or packing or side, "Missing structural train/test shift"


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded semantic fixtures plus 42 reproducible structurally varied seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(17)) == _signature(puzzles[17])
    infos = [_analysis(pair["input"]) for p in puzzles for s in p for pair in p[s]]
    assert {i["side"] for i in infos} == {"left", "right", "top", "bottom"}
    assert {len(i["texture"]) for i in infos} == {2, 3, 4}
    assert len({i["board"].shape for i in infos}) >= 20
    assert len({tuple(r[2:] for r in i["rectangles"]) for i in infos}) >= 80
    assert max(len(i["rectangles"]) for i in infos) >= 8
    assert max(max(i["distances"]) for i in infos) >= 3
    assert max(max(map(len, i["graph"])) for i in infos) >= 4
    assert any(len(i["dimensions"]) > len(i["rectangles"]) for i in infos)
    assert any(len(i["dimensions"]) == len(i["rectangles"]) for i in infos)
    assert any(max(Counter(r[2:] for r in i["rectangles"]).values()) > 1 for i in infos)
    assert any(i["mask"][0].any() or i["mask"][-1].any()
               or i["mask"][:, 0].any() or i["mask"][:, -1].any() for i in infos)
    # Finite inventory, duplicate identity, ambiguity, and no independent rotation.
    assert not _packing(np.ones((1, 4), bool), [(1, 2)])[0]
    assert len(_packing(np.ones((1, 4), bool), [(1, 2), (1, 2)])[0]) == 1
    assert len(_packing(np.ones((1, 4), bool), [(1, 4), (1, 2), (1, 2)])[0]) == 2
    assert not _packing(np.ones((2, 1), bool), [(1, 2)])[0]
    # Triangle plus a tail: shortest routes, equal-frontier contact, and corner exclusion.
    rectangles = ((0, 0, 1, 4), (1, 0, 2, 2), (1, 2, 1, 2), (3, 0, 1, 2))
    graph = _contact_graph(rectangles)
    assert _distances(graph, 0) == [0, 1, 1, 2]
    assert _contact_graph(((0, 0, 1, 1), (1, 1, 1, 1))) == [set(), set()]
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(3100 + 10 * nt + nq, nt, nq)
            validate_generated_puzzle(p)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()}
                    for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("Seam-wave self-test passed (42 seeds and every legal count combination).")
