"""Restore a reflected maze, then draw its unique shortest orthogonal route.

The first row is a five-swatch color key (background, wall, terminal,
erasure, route); the second row is a wall-colored separator. Only the field
below that separator participates in reflection and routing. No files are
read, written, or cached by this module.
"""
from collections import deque
from hashlib import sha256

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _read_input(value):
    a = _grid(value)
    if a.shape[0] < 5 or a.shape[1] < 5:
        raise ValueError("Missing color key or field")
    roles = tuple(int(x) for x in a[0, :5])
    bg, wall, terminal, erase, ink = roles
    if (len(set(roles)) != 5 or np.any(a[0, 5:] != bg)
            or np.any(a[1] != wall)):
        raise ValueError("Malformed five-swatch key")
    field = a[2:].copy()
    if not set(np.unique(field)).issubset({bg, wall, terminal, erase}):
        raise ValueError("Unexpected field color")
    return a, field, roles


def _orbit(p, shape, axes):
    r, c = p
    h, w = shape
    rr, cc = axes[0] - r, axes[1] - c
    rows = sorted({r, rr}) if 0 <= rr < h else [r]
    cols = sorted({c, cc}) if 0 <= cc < w else [c]
    return tuple((y, x) for y in rows for x in cols)


def _orbits(shape, axes):
    seen = set()
    for r in range(shape[0]):
        for c in range(shape[1]):
            if (r, c) not in seen:
                orbit = _orbit((r, c), shape, axes)
                seen.update(orbit)
                yield orbit


def _axis(field, erase, dimension):
    """Greatest supported reflected overlap; tied axes are invalid.

    At least half the lines, and four visible nonself cell comparisons, must
    participate. All visible reflected comparisons must agree. The maximal
    overlap convention excludes accidental folds of tiny uniform edge strips.
    """
    a = field if dimension == 0 else field.T
    n = a.shape[0]
    supported = []
    for s in range(1, 2 * n - 2):
        overlap = n - abs(s - (n - 1))
        if overlap * 2 < n:
            continue
        i = np.arange(max(0, s - n + 1), min(n, (s + 1) // 2))
        if not len(i):
            continue
        x, y = a[i], a[s - i]
        visible = (x != erase) & (y != erase)
        if int(visible.sum()) >= 4 and np.all(x[visible] == y[visible]):
            supported.append((overlap, s))
    if not supported:
        raise ValueError("No supported reflection fold")
    largest = max(x[0] for x in supported)
    best = [s for overlap, s in supported if overlap == largest]
    if len(best) != 1:
        raise ValueError("Ambiguous maximal reflection fold")
    return best[0]


def _restore(field, erase):
    axes = (_axis(field, erase, 0), _axis(field, erase, 1))
    clean = field.copy()
    for orbit in _orbits(field.shape, axes):
        witnesses = {int(field[p]) for p in orbit if field[p] != erase}
        if len(witnesses) != 1:
            raise ValueError("Empty or contradictory reflection orbit")
        color = next(iter(witnesses))
        for p in orbit:
            clean[p] = color
    return clean, axes


def _neighbors(p, shape):
    r, c = p
    h, w = shape
    for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= y < h and 0 <= x < w:
            yield y, x


def _path(blocked, start, goal):
    """BFS counts shortest routes, saturating at two; no tie breaking rule."""
    if blocked[start] or blocked[goal]:
        return [], 0
    distance, ways, previous = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, blocked.shape):
            if blocked[q]:
                continue
            d = distance[p] + 1
            if q not in distance:
                distance[q] = d
                ways[q] = ways[p]
                previous[q] = p
                queue.append(q)
            elif distance[q] == d:
                ways[q] = min(2, ways[q] + ways[p])
    if goal not in distance:
        return [], 0
    route = [goal]
    while route[-1] != start:
        route.append(previous[route[-1]])
    return route[::-1], ways[goal]


def _endpoints(clean, terminal):
    cells = [tuple(map(int, p)) for p in np.argwhere(clean == terminal)]
    if len(cells) != 2:
        raise ValueError("The repaired field must have exactly two terminals")
    return cells


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input's key, surviving reflection evidence, and maze."""
    result, field, roles = _read_input(input_grid)
    bg, wall, terminal, erase, ink = roles
    clean, _ = _restore(field, erase)
    start, goal = _endpoints(clean, terminal)
    route, count = _path(clean == wall, start, goal)
    if count != 1 or len(route) < 3:
        raise ValueError("A unique nontrivial shortest route is required")
    output = clean.copy()
    for p in route[1:-1]:
        if clean[p] != bg:
            raise ValueError("Route intermediates must be background")
        output[p] = ink
    result[2:] = output
    return result


def _turns(route):
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(route, route[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _critical_walls(blocked, route, axes):
    # Removing any returned wall bridges route positions over >2 steps.
    # Four-member orbits allow erasure needing both folds at this very wall.
    indices = {p: i for i, p in enumerate(route)}
    result = []
    for point in np.argwhere(blocked):
        p = tuple(map(int, point))
        if len(_orbit(p, blocked.shape, axes)) != 4:
            continue
        adjacent = [indices[q] for q in _neighbors(p, blocked.shape) if q in indices]
        if len(adjacent) >= 2 and max(adjacent) - min(adjacent) > 2:
            result.append(p)
    return result


def _has_longer_alternative(blocked, route):
    for p in route[1:-1]:
        other = blocked.copy()
        other[p] = True
        alternative, _ = _path(other, route[0], route[-1])
        if alternative:
            assert len(alternative) > len(route)
            return True
    return False


def _random_maze(rng):
    h, w = int(rng.integers(11, 25)), int(rng.integers(9, 26))
    fringe, side = int(rng.integers(1, 5)), int(rng.integers(0, 4))
    axes = h - 1 + fringe, w - 1 + side
    rr, cc = np.arange(h), np.arange(w)
    rr = np.where(axes[0] - rr < h, np.minimum(rr, axes[0] - rr), rr)
    cc = np.where(axes[1] - cc < w, np.minimum(cc, axes[1] - cc), cc)
    raw = rng.random((h, w)) < rng.uniform(0.4, 0.65)
    blocked = raw[rr[:, None], cc[None, :]].copy()
    r = int(rng.integers(fringe))
    c = int(rng.integers(side, (axes[1] - 2) // 2 + 1))
    start, goal = (r, c), (r, axes[1] - c)
    blocked[start] = blocked[goal] = False
    return blocked, start, goal, axes


def _gate_maze(rng):
    # Each wall has a reflected pair of doors. The terminal-side fringe makes
    # the upper doors nearer, while lower doors provide longer alternatives.
    fringe = int(rng.integers(1, 5))
    h = int(rng.integers(max(12, fringe + 10), 27))
    count = int(rng.choice((3, 5, 7)))
    half = count // 2
    widths = [1] + [int(rng.integers(1, 3)) for _ in range(half)]
    widths = widths + widths[-2::-1]
    w = 1 + count + sum(widths)
    axes = h - 1 + fringe, w - 1
    low, high = fringe + 1, (axes[0] - 1) // 2 - 1
    depths = [int(rng.integers(low + 1, high + 1))]
    depths += [int(rng.integers(low, depths[0]))]
    depths += [int(rng.integers(low, high + 1)) for _ in range(max(0, half - 1))]
    depths = depths + depths[-2::-1]
    blocked = np.zeros((h, w), dtype=bool)
    c = 1
    for width, depth in zip(widths, depths):
        blocked[:, c:c + width] = True
        blocked[depth, c:c + width] = False
        blocked[axes[0] - depth, c:c + width] = False
        c += width + 1
    r = int(rng.integers(fringe))
    return blocked, (r, 0), (r, w - 1), axes


def _mask_field(rng, clean, axes, route, critical, terminal_loss, hard):
    # Internal role IDs: 0 background, 1 wall, 2 terminal, 3 erased.
    h, w = clean.shape
    mask = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(4))
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(0.04, 0.16)
    elif style in (1, 2):
        for _ in range(int(rng.integers(1, 4))):
            bh, bw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            if style == 2:
                bh = 1
                bw = int(rng.integers(3, min(w, 9) + 1))
            r, c = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
            mask[r:r + bh, c:c + bw] = True
    else:
        p = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(8, 30))):
            mask[p] = True
            ns = list(_neighbors(p, clean.shape))
            p = ns[int(rng.integers(len(ns)))]
    protected = set()
    p = critical[int(rng.integers(len(critical)))]
    orbit = _orbit(p, clean.shape, axes)
    if hard:
        survivor = (axes[0] - p[0], axes[1] - p[1])
        protected.update(q for q in orbit if q != survivor)
    else:
        protected.add(p)
    for q in orbit:
        mask[q] = q in protected
    # Also expose background repair away from the route, and on the route.
    route_set = set(route)
    for on_route in (False, True):
        candidates = [tuple(map(int, q)) for q in np.argwhere(clean == 0)
                      if (tuple(map(int, q)) in route_set) == on_route
                      and len(_orbit(tuple(map(int, q)), clean.shape, axes)) > 1]
        if not candidates:
            return None
        q = candidates[int(rng.integers(len(candidates)))]
        protected.add(q)
        mask[q] = True
    ends = [tuple(map(int, q)) for q in np.argwhere(clean == 2)]
    mask[clean == 2] = False
    if terminal_loss:
        q = ends[int(rng.integers(2))]
        protected.add(q)
        mask[q] = True
    for orbit in _orbits(clean.shape, axes):
        erased = [q for q in orbit if mask[q]]
        if not hard and len(erased) > 1:
            forced = [q for q in erased if q in protected]
            if len(forced) > 1:
                return None
            keep = forced[0] if forced else erased[int(rng.integers(len(erased)))]
            for q in erased:
                mask[q] = q == keep
        if all(mask[q] for q in orbit):
            options = [q for q in orbit if q not in protected]
            if not options:
                return None
            mask[options[int(rng.integers(len(options)))]] = False
    inp = clean.copy()
    inp[mask] = 3
    try:
        repaired, inferred = _restore(inp, 3)
    except ValueError:
        return None
    if inferred != axes or not np.array_equal(repaired, clean):
        return None
    return inp


def _make_pair(rng, roles, terminal_loss, hard):
    want_random = rng.random() < 0.28
    # Candidates are held only in memory; retry bounds never create logs/files.
    for attempt in range(16000 if want_random else 160):
        builder = _random_maze if want_random and attempt < 15840 else _gate_maze
        blocked, start, goal, axes = builder(rng)
        route, count = _path(blocked, start, goal)
        if (count != 1 or len(route) - 1 <= abs(start[0] - goal[0])
                + abs(start[1] - goal[1]) + 3 or _turns(route) < 2):
            continue
        critical = _critical_walls(blocked, route, axes)
        if not critical or int((~blocked).sum()) <= len(route) + 3:
            continue
        clean = blocked.astype(np.int8)
        clean[start] = clean[goal] = 2
        try:
            if (_axis(clean, 3, 0), _axis(clean, 3, 1)) != axes:
                continue
        except ValueError:
            continue
        if not _has_longer_alternative(blocked, route):
            continue
        inp = _mask_field(rng, clean, axes, route, critical, terminal_loss, hard)
        if inp is None:
            continue
        output = clean.copy()
        for p in route[1:-1]:
            output[p] = 4
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        inp, output = np.rot90(inp, turns), np.rot90(output, turns)
        if flip:
            inp, output = inp[:, ::-1], output[:, ::-1]
        palette = np.asarray(roles, dtype=np.int8)
        inp, output = palette[inp], palette[output]
        h, w = inp.shape
        # Rotation acts on the field; the key remains readable at the top.
        pair = {}
        for name, field in (("input", inp), ("output", output)):
            grid = np.full((h + 2, w), roles[0], dtype=np.int8)
            grid[0, :5] = palette
            grid[1] = roles[1]
            grid[2:] = field
            pair[name] = grid
        return pair
    raise RuntimeError("Bounded maze construction exhausted")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    # The minimum two demonstrations separate wall-hard and terminal-easy
    # repair; tests combine the two. Order carries no semantic information.
    modes = [(False, True), (True, False)]
    modes += [(bool(rng.integers(2)), False) if rng.random() < 0.55
              else (False, True) for _ in range(nt - 2)]
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plan in (("train", modes), ("test", [(True, True)] * ns)):
        for terminal_loss, hard in plan:
            for _ in range(24):
                pair = _make_pair(rng, roles, terminal_loss, hard)
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    _, field, roles = _read_input(pair["input"])
    bg, wall, terminal, erase, ink = roles
    clean, axes = _restore(field, erase)
    endpoints = _endpoints(clean, terminal)
    route, count = _path(clean == wall, *endpoints)
    mask = field == erase
    diagonal = 0
    for r, c in np.argwhere(mask):
        rr, cc = axes[0] - int(r), axes[1] - int(c)
        if (0 <= rr < field.shape[0] and 0 <= cc < field.shape[1]
                and mask[rr, c] and mask[r, cc] and not mask[rr, cc]):
            diagonal += 1
    # Wall-only ablation: grant terminal/background repair, leave all erased
    # walls traversable. A shorter path proves repair alters route constraints.
    omitted = clean == wall
    omitted[mask] = False
    shortcut, _ = _path(omitted, *endpoints)
    route_set = set(route)
    repaired_bg_offroute = sum(clean[p] == bg and p not in route_set
                              for p in zip(*np.nonzero(mask)))
    return {"roles": roles, "axes": axes, "clean": clean, "route": route,
            "unique": count == 1, "diagonal": diagonal,
            "terminal_loss": int(np.count_nonzero(mask & (clean == terminal))),
            "wall_loss": int(np.count_nonzero(mask & (clean == wall))),
            "background_loss": int(np.count_nonzero(mask & (clean == bg))),
            "wall_ablation_shortens": bool(shortcut and len(shortcut) < len(route)),
            "offroute_repair": repaired_bg_offroute,
            "turns": _turns(route), "erased": int(mask.sum())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    roles, seen, demos = None, set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen, "Duplicate input"
            seen.add(sig)
            f = _features(pair)
            roles = f["roles"] if roles is None else roles
            assert f["roles"] == roles, "Key conventions must agree within a puzzle"
            bg, wall, terminal, erase, ink = roles
            clean, route = f["clean"], f["route"]
            assert f["unique"] and len(route) >= 5 and f["turns"] >= 2
            assert len(route) - 1 > sum(abs(a - b) for a, b in zip(route[0], route[-1]))
            assert f["wall_loss"] and f["background_loss"] and f["offroute_repair"]
            assert f["wall_ablation_shortens"], "Missing walls must change shortest routing"
            assert _has_longer_alternative(clean == wall, route), "Shortest selection needs a competitor"
            assert np.array_equal(inp[:2], out[:2])
            assert not np.any(out[2:] == erase)
            expected = clean.copy()
            for p in route[1:-1]:
                assert clean[p] == bg
                expected[p] = ink
            assert np.array_equal(out[2:], expected), "Wrong restored maze or route"
            assert np.array_equal(solve(inp), out)
            assert np.array_equal(out[2:][clean == wall], clean[clean == wall])
            assert np.array_equal(out[2:][clean == terminal], clean[clean == terminal])
            if split == "train":
                demos.append(f)
            else:
                assert f["terminal_loss"] == 1 and f["diagonal"] > 0
    assert any(f["terminal_loss"] == 0 and f["diagonal"] > 0 for f in demos)
    assert any(f["terminal_loss"] == 1 and f["diagonal"] == 0 for f in demos)


def self_test():
    """Bounded 42-seed study, no retained puzzles, logs, files, or traces."""
    shapes, palettes, structures, lengths, offsets, parities = (set() for _ in range(6))
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                f = _features(pair)
                shapes.add(inp.shape)
                palettes.add(f["roles"])
                lengths.add(len(f["route"]))
                axes = f["axes"]
                h, w = f["clean"].shape
                offsets.add((int(np.sign(axes[0] - h + 1)), int(np.sign(axes[1] - w + 1))))
                parities.add((axes[0] % 2, axes[1] % 2))
                ids = {c: i for i, c in enumerate(f["roles"])}
                normalized = np.array([[ids[int(c)] for c in row] for row in inp], dtype=np.int8)
                structures.add((inp.shape, normalized.tobytes()))
                digest.update(inp.tobytes())
                assert f["wall_ablation_shortens"]
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(palettes) >= 30
    assert len(shapes) >= 35 and len(structures) >= 190 and len(lengths) >= 20
    assert len(offsets) >= 4 and len(parities) == 4
    first, again = generate_puzzle(73, 2, 1), generate_puzzle(73, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first
               for p, q in zip(first[s], again[s]) for k in ("input", "output"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Corrupt a repaired wall, a terminal, and an off-route background cell.
    for role_index in (0, 1, 2):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in first[s]] for s in first}
        pair = bad["test"][0]
        f = _features(pair)
        candidates = [tuple(map(int, p)) for p in np.argwhere(f["clean"] == f["roles"][role_index])
                      if tuple(map(int, p)) not in set(f["route"][1:-1])]
        r, c = candidates[0]
        pair["output"][r + 2, c] = f["roles"][4]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Validator accepted a corrupted preserved cell")
