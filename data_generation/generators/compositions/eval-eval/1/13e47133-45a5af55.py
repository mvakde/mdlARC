"""Thickest-place ripples: infer erosion cores, then propagate a stripe code.

Only standard-library modules and NumPy are used.  No parent generator is
imported, and solving has no access to randomness or generation state.
"""
from collections import deque
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int16, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        component = np.zeros(mask.shape, dtype=bool)
        while q:
            r, c = q.popleft()
            component[r, c] = True
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if (0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        result.append(component)
    return result


def _erosion_depth(component, diagonal=True):
    """Boundary cells have depth zero; off-board positions are outside."""
    h, w = component.shape
    current = component.copy()
    depth = np.full(component.shape, -1, dtype=np.int16)
    level = 0
    while np.any(current):
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in (DIR8 if diagonal else DIR4):
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _travel_distance(component, sources):
    """Multi-source four-neighbor shortest paths restricted to one region."""
    h, w = component.shape
    distance = np.full(component.shape, -1, dtype=np.int16)
    q = deque((int(r), int(c)) for r, c in np.argwhere(sources))
    distance[sources] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and component[rr, cc]
                    and distance[rr, cc] < 0):
                distance[rr, cc] = distance[r, c] + 1
                q.append((rr, cc))
    assert np.all(distance[component] >= 0)
    return distance


def _parse(input_grid):
    a = _grid(input_grid)
    uniform = np.all(a == a[:, :1], axis=1)
    starts = np.flatnonzero(~uniform)
    assert len(starts), "the board must begin with a mixed row"
    start = int(starts[0])
    assert start >= 2, "at least one code row and a separator are required"
    wall_color = int(a[start - 1, 0])
    code = a[:start - 1, 0].copy()
    assert wall_color not in code, "the separator color is reserved for walls"
    board = a[start:].copy()
    colors = set(int(c) for c in np.unique(board))
    assert wall_color in colors and len(colors) == 2
    neutral = next(c for c in colors if c != wall_color)
    walls = board == wall_color
    regions = _components(~walls)
    assert len(regions) >= 2
    return code, board, neutral, wall_color, walls, regions


def _put_shells(output, component, distance, code, neutral):
    output[component] = neutral
    reached = component & (distance >= 0) & (distance < len(code))
    output[reached] = code[distance[reached]]


def _render(input_grid):
    code, board, neutral, wall_color, walls, components = _parse(input_grid)
    output = board.copy()
    regions = []
    for component in components:
        depth = _erosion_depth(component)
        core = component & (depth == int(depth.max()))
        distance = _travel_distance(component, core)
        _put_shells(output, component, distance, code, neutral)
        regions.append({"mask": component, "depth": depth,
                        "core": core, "distance": distance})
    return output, {"code": code, "board": board, "neutral": neutral,
                    "wall_color": wall_color, "walls": walls, "regions": regions}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible legend and solve any input satisfying task_spec.md."""
    return _render(input_grid)[0].astype(int)


def _chebyshev_depth(component):
    """Validation construction: distance to the nearest outside cell minus one."""
    inside = np.argwhere(component) + 1
    outside = np.argwhere(~np.pad(component, 1))
    rr = np.abs(inside[:, None, 0] - outside[None, :, 0])
    cc = np.abs(inside[:, None, 1] - outside[None, :, 1])
    values = np.maximum(rr, cc).min(axis=1) - 1
    depth = np.full(component.shape, -1, dtype=np.int16)
    depth[component] = values
    return depth


def _reference_output(analysis):
    """Authored cross-check via direct core distance and repeated dilation."""
    output = analysis["board"].copy()
    h, w = output.shape
    for region in analysis["regions"]:
        mask = region["mask"]
        depth = _chebyshev_depth(mask)
        assert np.array_equal(depth, region["depth"])
        frontier = mask & (depth == int(depth.max()))
        seen = frontier.copy()
        output[mask] = analysis["neutral"]
        for color in analysis["code"]:
            output[frontier] = color
            padded = np.pad(frontier, 1)
            next_frontier = np.zeros(mask.shape, dtype=bool)
            for dr, dc in DIR4:
                next_frontier |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
            frontier = next_frontier & mask & ~seen
            seen |= frontier
    return output


def _unblocked_distance(core):
    rr, cc = np.indices(core.shape)
    distance = np.full(core.shape, 1000, dtype=np.int16)
    for r, c in np.argwhere(core):
        distance = np.minimum(distance, np.abs(rr - r) + np.abs(cc - c))
    return distance


def _alternative(analysis, mode):
    """Concrete, fallible shortcuts, used only as demonstration checks."""
    code = analysis["code"]
    if mode == "reverse_code":
        code = code[::-1]
    elif mode == "collapse_rows":
        code = code[np.r_[True, code[1:] != code[:-1]]]
    elif mode == "ignore_code":
        code = np.full(len(code), code[0], dtype=int)
    output = analysis["board"].copy()
    for region in analysis["regions"]:
        mask, core = region["mask"], region["core"]
        distance = region["distance"]
        if mode == "boundary_bands":
            distance = region["depth"]
        elif mode == "box_core":
            rows, cols = np.nonzero(mask)
            rr, cc = np.indices(mask.shape)
            box = np.minimum(np.minimum(rr - rows.min(), rows.max() - rr),
                             np.minimum(cc - cols.min(), cols.max() - cc))
            fake = mask & (box == box[mask].max())
            distance = _travel_distance(mask, fake)
        elif mode == "four_neighbor_core":
            depth = _erosion_depth(mask, diagonal=False)
            fake = mask & (depth == depth.max())
            distance = _travel_distance(mask, fake)
        elif mode == "unblocked":
            distance = _unblocked_distance(core)
        elif mode == "single_core":
            fake = np.zeros(mask.shape, dtype=bool)
            r, c = np.argwhere(core)[0]
            fake[r, c] = True
            distance = _travel_distance(mask, fake)
        if mode == "saturate":
            output[mask] = code[np.minimum(distance[mask], len(code) - 1)]
        elif mode == "cycle":
            output[mask] = code[distance[mask] % len(code)]
        else:
            _put_shells(output, mask, distance, code, analysis["neutral"])
    return output


SHORTCUTS = ("boundary_bands", "box_core", "four_neighbor_core", "unblocked",
             "single_core", "collapse_rows", "reverse_code", "ignore_code",
             "saturate", "cycle")


def _witnesses(output, analysis):
    return {name for name in SHORTCUTS
            if not np.array_equal(output, _alternative(analysis, name))}


def _solid_block(walls):
    return bool(np.any(walls[:-1, :-1] & walls[1:, :-1]
                       & walls[:-1, 1:] & walls[1:, 1:]))


def _rectangular(mask):
    r, c = np.nonzero(mask)
    return int(mask.sum()) == (r.max() - r.min() + 1) * (c.max() - c.min() + 1)


def _split_room(rng, walls):
    """Cut one rectangular open component into two, retaining room width >=3."""
    choices = []
    h, w = walls.shape
    for mask in _components(~walls):
        if not _rectangular(mask):
            continue
        rows, cols = np.nonzero(mask)
        a, b, c, d = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        if b - a + 1 >= 7 and (
                (c > 0 and np.any(walls[a + 3:b - 2, c - 1]))
                or (d + 1 < w and np.any(walls[a + 3:b - 2, d + 1]))):
            choices.append((0, a, b, c, d))
        if d - c + 1 >= 7 and (
                (a > 0 and np.any(walls[a - 1, c + 3:d - 2]))
                or (b + 1 < h and np.any(walls[b + 1, c + 3:d - 2]))):
            choices.append((1, a, b, c, d))
    if not choices:
        return walls
    axis, a, b, c, d = choices[int(rng.integers(len(choices)))]
    new = walls.copy()
    if axis == 0:
        q = int(rng.integers(a + 3, b - 2))
        new[q, max(0, c - 1):min(w, d + 2)] = True
    else:
        q = int(rng.integers(c + 3, d - 2))
        new[max(0, a - 1):min(h, b + 2), q] = True
    assert not _solid_block(new)
    return new if len(_components(new)) == 1 else walls


def _add_finger(rng, walls):
    h, w = walls.shape
    nodes = np.argwhere(walls)
    for _ in range(35):
        r, c = map(int, nodes[int(rng.integers(len(nodes)))])
        dr, dc = DIR4[int(rng.integers(4))]
        length = int(rng.integers(3, 10))
        points = [(r + t * dr, c + t * dc) for t in range(1, length + 1)]
        if not all(1 <= rr < h - 1 and 1 <= cc < w - 1 and not walls[rr, cc]
                   for rr, cc in points):
            continue
        new = walls.copy()
        for rr, cc in points:
            new[rr, cc] = True
        if not _solid_block(new):
            return new
    return walls


def _attach_box(rng, walls):
    """A U-shaped enclosure attached to a wall creates a hole and a new room."""
    choices = []
    for mask in _components(~walls):
        if not _rectangular(mask):
            continue
        for transpose in (False, True):
            work, region = (walls.T, mask.T) if transpose else (walls, mask)
            rows, cols = np.nonzero(region)
            a, b, c, d = map(int, (rows.min(), rows.max(), cols.min(), cols.max()))
            if b - a + 1 < 7 or d - c + 1 < 5:
                continue
            for side in (-1, 1):
                anchor = c - 1 if side == 1 else d + 1
                if 0 <= anchor < work.shape[1] and np.all(work[a:b + 1, anchor]):
                    choices.append((transpose, a, b, c, d, side, anchor))
    if not choices:
        return walls
    transpose, a, b, c, d, side, anchor = choices[int(rng.integers(len(choices)))]
    new = walls.copy()
    work = new.T if transpose else new
    top = int(rng.integers(a + 1, b - 3))
    bottom = int(rng.integers(top + 3, b))
    extent = int(rng.integers(3, d - c + 1))
    far = anchor + side * extent
    lo, hi = sorted((anchor, far))
    work[top, lo:hi + 1] = True
    work[bottom, lo:hi + 1] = True
    work[top:bottom + 1, far] = True
    return walls if _solid_block(new) else new


def _jog_wall(rng, walls):
    """Replace an isolated straight interval by a displaced orthogonal interval."""
    for _ in range(40):
        transpose = bool(rng.integers(2))
        work = walls.T if transpose else walls
        nodes = np.argwhere(work)
        r, c = map(int, nodes[int(rng.integers(len(nodes)))])
        end = r + int(rng.integers(3, 9))
        shift = int(rng.choice((-3, -2, -1, 1, 2, 3)))
        lo, hi = sorted((c, c + shift))
        if r < 1 or end + 1 >= work.shape[0] or lo < 1 or hi + 1 >= work.shape[1]:
            continue
        window = work[r - 1:end + 2, lo - 1:hi + 2]
        expected = np.zeros(window.shape, dtype=bool)
        expected[:, c - lo + 1] = True
        if not np.array_equal(window, expected):
            continue
        new = walls.copy()
        modified = new.T if transpose else new
        modified[r + 1:end, c] = False
        modified[r, lo:hi + 1] = True
        modified[end, lo:hi + 1] = True
        modified[r:end + 1, c + shift] = True
        if not _solid_block(new):
            return new
    return walls


def _general_layout(rng, shape, test):
    h, w = shape
    walls = np.zeros(shape, dtype=bool)
    column = int(rng.integers(3, w - 3))
    walls[:, column] = True
    target = int(rng.integers(4 if test else 2, 9 if test else 7))
    for _ in range(target - 2):
        walls = _split_room(rng, walls)
    for _ in range(int(rng.integers(0, 3 if test else 2))):
        walls = _attach_box(rng, walls)
    for _ in range(int(rng.integers(1 if test else 0, 5 if test else 4))):
        walls = _add_finger(rng, walls)
    for _ in range(int(rng.integers(1 if test else 0, 4 if test else 3))):
        walls = _jog_wall(rng, walls)
    return walls


def _code(rng, length, neutral, wall_color, kind):
    colors = [c for c in range(10) if c not in (neutral, wall_color)]
    a, b, c = [int(x) for x in rng.choice(colors, 3, replace=False)]
    if kind == 0:
        code = [a, b, b, c]
        code.extend(int(x) for x in rng.choice(colors, length - 4))
        code[-1] = c
        return np.asarray(code, dtype=int)
    if kind == 1:
        code = [neutral, a, b]
        code.extend([a] * (length - 3))
        return np.asarray(code, dtype=int)
    available = [c for c in range(10) if c != wall_color]
    palette_size = int(rng.integers(1, min(len(available), length) + 1))
    palette = rng.choice(available, palette_size, replace=False)
    code = [int(c) for c in rng.permutation(palette)]
    # Insert visits into an ordered code. Each sampled color remains visible;
    # adjacent insertions preserve row thickness rather than compressing runs.
    while len(code) < length:
        position = int(rng.integers(len(code) + 1))
        color = (code[position - 1] if position and rng.random() < 0.5
                 else int(rng.choice(palette)))
        code.insert(position, color)
    return np.asarray(code, dtype=int)


def _assemble(walls, code, neutral, wall_color):
    board = np.full(walls.shape, neutral, dtype=int)
    board[walls] = wall_color
    header = np.repeat(np.asarray(code, dtype=int)[:, None], walls.shape[1], axis=1)
    separator = np.full((1, walls.shape[1]), wall_color, dtype=int)
    inp = np.concatenate((header, separator, board), axis=0)
    return {"input": inp, "output": solve(inp)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _require_integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _require_integer(num_train, "num_train", 2, 10)
    num_test = _require_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    total = num_train + num_test
    neutrals = [int(c) for c in rng.permutation(10)]
    # A shuffled cyclic assignment gives distinct local roles and varies both
    # role colors within even the smallest legal puzzle.
    offset = int(rng.integers(1, 10))
    wall_colors = neutrals[offset:] + neutrals[:offset]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_shape = None
    for index in range(total):
        neutral, wall_color = neutrals[index % 10], wall_colors[index % 10]
        test = index >= num_train
        kind = index if index < 2 else 2
        # Keep a demonstrable shift even at ten training examples: if room
        # counts >=4 have already appeared, the first test extrapolates the
        # finite code to an actually used eighth/ninth shell instead.
        longer_code = (test and index == num_train
                       and any(len(_parse(p["input"])[-1]) >= 4
                               for p in puzzle["train"]))
        for attempt in range(80):
            if kind == 0:
                length = 7
                h = int(rng.integers(13, 16))
                w = int(rng.integers(17, 23))
                column = int(rng.integers(10, w - 4))
                row = int(rng.integers(4, 6))
                gap = int(rng.integers(2, 5))
                walls = np.zeros((h, w), dtype=bool)
                walls[:, column] = True
                walls[row, gap:column + 1] = True
            elif kind == 1:
                length = int(rng.integers(3, 6))
                h = int(rng.choice((13, 17)))
                w = int(rng.integers(17, 25))
                column = int(rng.integers(9, w - 4))
                gap = 3 if h == 13 else 4
                walls = np.zeros((h, w), dtype=bool)
                walls[:, column] = True
                walls[h // 2, gap:column + 1] = True
            else:
                length = (int(rng.integers(8, 10)) if longer_code
                          else int(rng.integers(1 if not test else 2, 8 if not test else 10)))
                h = int(rng.integers(9 if not test else 12, 30 - length))
                w = int(rng.integers(9 if not test else 13, 31))
                turns = int(rng.integers(4))
                shape = (w, h) if turns % 2 else (h, w)
                walls = np.rot90(_general_layout(rng, shape, test), turns).copy()
            if kind < 2:
                walls = np.rot90(walls, int(rng.integers(4))).copy()
                # A rotated spanning wall may touch only the side edges.
                # Extend a short branch to the top to keep the header boundary
                # visibly identifiable; the resulting thin room follows the
                # same all-deepest-cells rule as every other room.
                if not np.any(walls[0]):
                    column = 1 if rng.integers(2) else walls.shape[1] - 2
                    stop = int(np.flatnonzero(walls[:, column])[0])
                    walls[:stop + 1, column] = True
            if rng.integers(2):
                walls = walls[:, ::-1].copy()
            # The prefix decoder needs the first board row to be mixed. This
            # is a visible layout constraint, not a hidden orientation cue.
            if np.all(walls[0]) or not np.any(walls[0]):
                continue
            if walls.shape[0] + length + 1 > 30:
                continue
            if kind == 1 and walls.shape == first_shape:
                continue
            code = _code(rng, length, neutral, wall_color, kind)
            pair = _assemble(walls, code, neutral, wall_color)
            if _key(pair["input"]) in seen:
                continue
            if kind < 2:
                output, analysis = _render(pair["input"])
                witnesses = _witnesses(output, analysis)
                if kind == 0 and not {"unblocked", "collapse_rows", "reverse_code",
                                      "saturate", "cycle", "boundary_bands", "box_core"} <= witnesses:
                    continue
                if kind == 1:
                    if "four_neighbor_core" not in witnesses or "single_core" not in witnesses:
                        continue
                    if not any(len(_components(r["core"])) >= 2 for r in analysis["regions"]):
                        continue
            if test and len(_components(~walls)) < 4:
                continue
            if longer_code:
                _, analysis = _render(pair["input"])
                if code[-1] == neutral or not any(
                        int(r["distance"].max()) >= length - 1 for r in analysis["regions"]):
                    continue
            seen.add(_key(pair["input"]))
            puzzle["test" if test else "train"].append(pair)
            if kind == 0:
                first_shape = walls.shape
            break
        else:
            raise RuntimeError(f"bounded construction failed for example {index}")
    # Construction roles never determine presentation order or solution rules.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training_witnesses = set(), set()
    roles, shapes, counts = set(), set(), set()
    training, testing = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, analysis = _render(inp)
            walls, neutral = analysis["walls"], analysis["neutral"]
            assert len(_components(walls)) == 1 and not _solid_block(walls)
            assert np.array_equal(out, predicted)
            assert np.array_equal(out, _reference_output(analysis))
            assert out.shape == analysis["board"].shape
            assert np.array_equal(out == analysis["wall_color"], walls)
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            for region in analysis["regions"]:
                core, distance, mask = region["core"], region["distance"], region["mask"]
                assert np.all(out[core] == analysis["code"][0])
                assert np.all(out[mask & (distance >= len(analysis["code"]))] == neutral)
            assert not np.array_equal(inp, out)
            roles.add((neutral, analysis["wall_color"]))
            shapes.add(out.shape)
            counts.add(len(analysis["regions"]))
            if split == "train":
                training.append(analysis)
                training_witnesses |= _witnesses(out, analysis)
            else:
                assert len(analysis["regions"]) >= 4
                testing.append(analysis)
    assert training_witnesses == set(SHORTCUTS), "demonstrations permit a listed shortcut"
    assert any(a["code"][0] == a["neutral"] for a in training)
    assert any(len(_components(r["core"])) >= 2 for a in training for r in a["regions"])
    assert len({n for n, _ in roles}) >= min(3, len(seen))
    assert len({w for _, w in roles}) >= min(3, len(seen))
    assert len(shapes) >= 2 and len(counts) >= 2
    largest_train_code = max(len(a["code"]) for a in training)
    largest_train_count = max(len(a["regions"]) for a in training)
    assert any(len(a["regions"]) > largest_train_count or (
        len(a["code"]) > largest_train_code and a["code"][-1] != a["neutral"]
        and any(int(r["distance"].max()) >= len(a["code"]) - 1 for r in a["regions"]))
        for a in testing), "no visible test extrapolation"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed suite, semantic fixtures, corruption and count checks."""
    puzzles, shapes, geometries, palettes, codes, region_counts = (set() for _ in range(6))
    core_counts, radii, lengths, colors, neutral_colors, wall_colors = (set() for _ in range(6))
    concave = enclosed = fragmented = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            output, a = _render(pair["input"])
            assert np.array_equal(output, pair["output"])
            assert np.array_equal(_reference_output(a), pair["output"])
            shapes.add(output.shape)
            geometries.add(_key(a["walls"]))
            palettes.add(len(np.unique(a["code"])))
            lengths.add(len(a["code"]))
            normalized = {}
            codes.add(tuple(normalized.setdefault(int(c), len(normalized)) for c in a["code"]))
            colors.update(map(int, np.unique(pair["input"])))
            neutral_colors.add(a["neutral"])
            wall_colors.add(a["wall_color"])
            region_counts.add(len(a["regions"]))
            for r in a["regions"]:
                mask = r["mask"]
                concave += not _rectangular(mask)
                enclosed += not (mask[0].any() or mask[-1].any()
                                 or mask[:, 0].any() or mask[:, -1].any())
                islands = len(_components(r["core"]))
                fragmented += islands > 1
                core_counts.add(islands)
                radii.add(int(r["distance"].max()))
    assert len(puzzles) == 42
    assert len(shapes) >= 60 and len(geometries) >= 150
    assert len(codes) >= 50 and len(palettes) >= 5
    assert lengths == set(range(1, 10))
    assert colors == neutral_colors == wall_colors == set(range(10))
    assert len(region_counts) >= 5 and len(core_counts) >= 3
    assert concave >= 120 and enclosed >= 5 and fragmented >= 42
    assert max(radii) >= 20 and min(radii) <= 2
    assert _puzzle_key(generate_puzzle(9173, 7, 3)) == _puzzle_key(generate_puzzle(9173, 7, 3))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * train + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3.0, 1), (3, False), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rectangular cores form full plateaus; every tied point is a source.
    walls = np.zeros((7, 12), dtype=bool)
    walls[:, 6] = True
    fixture = _assemble(walls, [2, 3], 0, 1)
    output = fixture["output"]
    assert np.all(output[2:5, 2:4] == 2)
    assert output[0, 0] == 0 and output[1, 2] == 3
    # D=0 is legal: all cells of a one-cell-wide room are core sources.
    thin = np.zeros((5, 7), dtype=bool)
    thin[:, 1] = True
    fixture = _assemble(thin, [4, 5, 6], 0, 1)
    assert np.all(fixture["output"][:, 0] == 4)
    # An unsampled but legal boundary member: a 28-row code above a one-row
    # board, reaching the 30-row input and 30-column width limits.
    tiny = np.zeros((1, 30), dtype=bool)
    tiny[:, 1] = True
    long_code = ([2, 0, 3, 4, 5, 6, 7, 8, 9] * 4)[:28]
    fixture = _assemble(tiny, long_code, 0, 1)
    assert fixture["input"].shape == (30, 30)
    assert fixture["output"].shape == (1, 30)
    assert np.all(fixture["output"][~tiny] == 2)
    predicted, analysis = _render(fixture["input"])
    assert np.array_equal(predicted, _reference_output(analysis))
    # Reject a plausible-looking but wrong color and a changed divider.
    original = generate_puzzle(81, 2, 1)
    for role in ("core", "wall"):
        broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()}
                      for p in original[s]] for s in ("train", "test")}
        _, a = _render(broken["test"][0]["input"])
        positions = a["walls"] if role == "wall" else a["regions"][0]["core"]
        r, c = np.argwhere(positions)[0]
        old = int(broken["test"][0]["output"][r, c])
        broken["test"][0]["output"][r, c] = (old + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted corrupted {role}")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "board_shapes": len(shapes),
            "wall_geometries": len(geometries), "normalized_codes": len(codes),
            "code_lengths": sorted(lengths), "palette_sizes": sorted(palettes),
            "region_counts": sorted(region_counts), "core_island_counts": sorted(core_counts),
            "radii": sorted(radii), "concave_regions": int(concave),
            "enclosed_regions": int(enclosed), "fragmented_cores": int(fragmented),
            "independent_verification": False}


if __name__ == "__main__":
    print(self_test())
