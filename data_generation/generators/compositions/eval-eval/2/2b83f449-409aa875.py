"""Five-step signals on a locally rewired, nonbranching railway.

Only standard-library modules and NumPy are used. No filesystem access, hidden
conventions, retained puzzles, or parent imports participate in generation/solve.
"""
from collections import Counter, defaultdict
import hashlib
import numpy as np


DISTANCE = 5
COLLISION = 9
DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
HOST_SEEDS = (0, 344364348, 450990551, 602610515, 833904596, 743724382,
              384354989, 536250805, 344109484, 248131358, 716412893, 98827215,
              895387790, 947328049, 913536869, 713044779, 98308054, 970313608,
              422547170, 67261276, 941541196, 456596784, 621227350, 920463607,
              857070889, 406379379, 253990322, 933973968, 537019892, 687737188,
              970109308, 841376060, 950958064, 745469198, 429130417, 748218555,
              819570395, 347842976, 94730248, 971367267, 906065301, 876936592)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    offsets = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                    if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], {start}
        while todo:
            r, c = todo.pop()
            found = {(r + dr, c + dc) for dr, dc in offsets} & remaining
            remaining -= found
            cells |= found
            todo.extend(sorted(found))
        result.append(cells)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    wings = ((r - 1, c - dc), (r + 1, c - dc)) if dr == 0 else \
            ((r - dr, c - 1), (r - dr, c + 1))
    return (tip, direction, (tip,) + wings)


def _read_arrow(cells):
    if len(cells) != 3:
        return None
    matches = [_arrow(p, d) for p in sorted(cells) for d in DIRS
               if set(_arrow(p, d)[2]) == cells]
    return matches[0] if len(matches) == 1 else None


def _read_bar(cells):
    if len(cells) != 3:
        return None
    points = sorted(cells)
    if len({p[0] for p in points}) == 1 and points[-1][1] - points[0][1] == 2:
        return points[1], (0, 1)
    if len({p[1] for p in points}) == 1 and points[-1][0] - points[0][0] == 2:
        return points[1], (1, 0)
    return None


def _neighbors(mask, p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in DIRS
            if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
            and mask[r + dr, c + dc]]


def _paths(mask):
    result = []
    for component in _components(mask):
        degrees = {p: len(_neighbors(mask, p)) for p in component}
        assert set(degrees.values()) <= {1, 2}, "Branches/isolated tracks forbidden"
        ends = sorted(p for p in component if degrees[p] == 1)
        assert len(ends) == 2, "Closed tracks forbidden"
        order, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            following = [p for p in _neighbors(mask, current) if p != previous]
            assert len(following) == 1
            previous, current = current, following[0]
            order.append(current)
        assert len(order) == len(component)
        result.append(order)
    return result


def _rewire(raw, bars):
    """Agreeing simultaneous writes, including compatible shared patch cells."""
    mask, writes, patch = raw.copy(), {}, set()
    for (r, c), axis in bars:
        assert 1 <= r < raw.shape[0] - 1 and 1 <= c < raw.shape[1] - 1
        # A horizontal bar becomes a vertical wall (and conversely).
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                value = cc != c if axis == (0, 1) else rr != r
                p = (rr, cc)
                assert p not in writes or writes[p] == value, "Conflicting switches"
                writes[p] = value
                patch.add(p)
    for p, value in writes.items():
        mask[p] = value
    return mask, writes, patch


def _decode(grid):
    """Identify role colors by monochrome geometry, never by frequency."""
    assert 0 in grid and COLLISION not in grid, "0 background; 9 output-only"
    track_colors, divider_colors, arrows = [], [], []
    bars = []
    for color in sorted(set(map(int, np.unique(grid))) - {0}):
        components = _components(grid == color, diagonal=True)
        possible_bars = [_read_bar(c) for c in components]
        possible_arrows = [_read_arrow(c) for c in components]
        if all(p is not None for p in possible_bars):
            divider_colors.append(color)
            bars.extend(possible_bars)
        elif all(p is not None for p in possible_arrows):
            arrows.extend((a, color) for a in possible_arrows)
        else:
            track_colors.append(color)
    assert len(track_colors) == len(divider_colors) == 1 and arrows
    track, divider = track_colors[0], divider_colors[0]
    assert len({axis for _, axis in bars}) == 1
    raw = grid == track
    for (tip, _, _), _ in arrows:
        raw[tip] = True
    mask, writes, patch = _rewire(raw, bars)
    paths = _paths(mask)
    assert paths
    # Arrow wings are annotations outside the railway, not additional rails.
    occupied = set()
    for (tip, direction, cells), _ in arrows:
        assert tip not in patch and mask[tip]
        for wing in cells[1:]:
            assert not raw[wing] and not mask[wing] and wing not in patch
        assert not any((r + dr, c + dc) in occupied for r, c in cells
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        occupied.update(cells)
        assert (tip[0] + direction[0], tip[1] + direction[1]) in _neighbors(mask, tip)
    # Every switch's six side-port cells belong to two unbroken input lanes.
    for (r, c), axis in bars:
        ports = ([(r + dr, c + dc) for dr in (-1, 1) for dc in (-1, 0, 1)]
                 if axis == (0, 1) else
                 [(r + dr, c + dc) for dc in (-1, 1) for dr in (-1, 0, 1)])
        assert all(raw[p] for p in ports)
    return track, divider, arrows, raw, mask, writes, patch, paths, bars


def _route(mask, arrow):
    tip, direction, _ = arrow
    current = (tip[0] + direction[0], tip[1] + direction[1])
    assert current in _neighbors(mask, tip)
    route, previous = [tip, current], tip
    for _ in range(DISTANCE - 1):
        following = [p for p in _neighbors(mask, current) if p != previous]
        assert len(following) == 1, "A shot cannot leave a terminal early"
        previous, current = current, following[0]
        route.append(current)
    return tuple(route)


def _paint(grid, arrows, destinations, structural=None):
    out = grid.copy() if structural is None else structural.copy()
    owners = {arrow[0]: arrow[2] for arrow, _ in arrows}
    hits = defaultdict(list)
    for (_, color), point in zip(arrows, destinations):
        hits[point].append(color)
    for point, colors in hits.items():
        color = COLLISION if len(colors) > 1 else colors[0]
        for cell in owners.get(point, (point,)):
            if 0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]:
                out[cell] = color
    return out, hits


def _evaluate(grid):
    track, divider, arrows, raw, mask, writes, patch, paths, bars = _decode(grid)
    structural = grid.copy()
    for p, value in writes.items():
        structural[p] = track if value else divider
    routes = [_route(mask, a) for a, _ in arrows]
    destinations = [r[-1] for r in routes]
    out, hits = _paint(grid, arrows, destinations, structural)
    assert max(map(len, hits.values())) <= 2, "A simple path has two approaches"
    tips = {a[0] for a, _ in arrows}
    straight = [(a[0][0] + DISTANCE * a[1][0], a[0][1] + DISTANCE * a[1][1])
                for a, _ in arrows]
    straight_out, _ = _paint(grid, arrows, straight, structural)
    curved_sites = {r[-1] for r, p in zip(routes, straight) if r[-1] != p}
    info = dict(track=track, divider=divider, arrows=arrows, bars=bars, paths=paths,
                mask=mask, patch=patch, routes=routes, hits=hits,
                collisions=sum(len(v) == 2 for v in hits.values()),
                routed_collisions=sum(len(v) == 2 and p in curved_sites for p, v in hits.items()),
                strikes=sum(p in tips for p in hits),
                singles=sum(len(v) == 1 for v in hits.values()),
                curved=sum(r[-1] != p for r, p in zip(routes, straight)),
                essential_a=not np.array_equal(out, straight_out),
                essential_b=not np.array_equal(out, structural))
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """One snapshot: rotate bars, then walk five edges and paint final impacts."""
    return _evaluate(_grid(input_grid))[0]


def _compatible(arrows):
    occupied = set()
    for arrow in arrows:
        if any((r + dr, c + dc) in occupied for r, c in arrow[2]
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            return False
        occupied.update(arrow[2])
    return True


def _network(rng, lanes, width):
    """Sample a lane topology, cutting U-turn traps rather than fixing a motif."""
    h = 2 * lanes + 3
    raw = np.zeros((h, width), dtype=bool)
    extents = []
    for i in range(lanes):
        left, right = int(rng.integers(1, 4)), width - 1 - int(rng.integers(1, 4))
        raw[2 + 2 * i, left:right + 1] = True
        extents.append((left, right))
    centers, previous = [], []
    for i in range(lanes - 1):
        low = max(extents[i][0], extents[i + 1][0]) + 2
        high = min(extents[i][1], extents[i + 1][1]) - 2
        selected = []
        desired = int(rng.choice((0, 1, 1, 1, 2)))
        for c in rng.permutation(np.arange(low, high + 1)):
            c = int(c)
            if len(selected) >= desired:
                break
            if all(abs(c - p) >= 2 for p in previous) and all(abs(c - p) >= 4 for p in selected):
                selected.append(c)
        centers.extend((3 + 2 * i, c) for c in sorted(selected))
        previous = selected
    if not centers:
        centers = [(3, width // 2)]
    # Same-side neighboring ports would produce a U bend; open that lane.
    for i in range(lanes):
        row = 2 + 2 * i
        ports = sorted((c, r) for r, c in centers if abs(r - row) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                if b - a < 4:
                    return None
                raw[row, int(rng.integers(a + 2, b - 1))] = False
    # Additional breaks change component counts and internal terminals.
    for _ in range(int(rng.integers(4))):
        i = int(rng.integers(lanes))
        row, (a, b) = 2 + 2 * i, extents[i]
        choices = [c for c in range(a + 2, b - 1)
                   if all(abs(c - x) >= 2 for r, x in centers if abs(r - row) == 1)
                   and np.all(raw[row, c - 1:c + 2])]
        if choices:
            raw[row, choices[int(rng.integers(len(choices)))]] = False
    bars = [(p, (0, 1)) for p in centers]
    try:
        mask, writes, patch = _rewire(raw, bars)
        paths = _paths(mask)
    except AssertionError:
        return None
    return raw, mask, writes, patch, paths, bars


def _candidates(network):
    raw, mask, _, patch, _, bars = network
    bar_cells = {(r, c + d) for (r, c), _ in bars for d in (-1, 0, 1)}
    choices, routes = [], []
    for tip in map(tuple, np.argwhere(raw).tolist()):
        if tip in patch:
            continue
        for direction in ((0, 1), (0, -1)):
            arrow = _arrow(tip, direction)
            if not all(0 <= r < raw.shape[0] and 0 <= c < raw.shape[1] for r, c in arrow[2]):
                continue
            if any(raw[p] or mask[p] or p in patch or p in bar_cells for p in arrow[2][1:]):
                continue
            try:
                route = _route(mask, arrow)
            except AssertionError:
                continue
            choices.append(arrow)
            routes.append(route)
    return choices, routes


def _sources(rng, network, kind, desired):
    choices, routes = _candidates(network)
    def curved(i):
        tip, d, _ = choices[i]
        return routes[i][-1] != (tip[0] + 5 * d[0], tip[1] + 5 * d[1])
    by_end, by_tip = defaultdict(list), defaultdict(list)
    for i, (arrow, route) in enumerate(zip(choices, routes)):
        by_end[route[-1]].append(i)
        by_tip[arrow[0]].append(i)
    collisions = []
    for indices in by_end.values():
        if len(indices) == 2 and any(curved(i) for i in indices) and \
                _compatible([choices[i] for i in indices]):
            collisions.append(tuple(indices))
    strikes = [(i, j) for i, route in enumerate(routes) for j in by_tip[route[-1]]
               if i != j and _compatible([choices[i], choices[j]])]
    rng.shuffle(collisions)
    rng.shuffle(strikes)
    selected = None
    if kind == "collision" and collisions:
        selected = list(collisions[0])
    elif kind == "strike" and strikes:
        selected = list(strikes[0])
    elif kind == "both":
        for collision in collisions[:50]:
            for strike in strikes[:150]:
                combined = list(dict.fromkeys(collision + strike))
                if _compatible([choices[i] for i in combined]):
                    selected = combined
                    break
            if selected is not None:
                break
    elif kind == "free" and choices:
        selected = [int(rng.integers(len(choices)))]
    if selected is None:
        return None
    # At least one signal must actually use a bend; topology is never decoration.
    for i in rng.permutation(len(choices)):
        i = int(i)
        if not any(curved(j) for j in selected) and curved(i) and \
                _compatible([choices[j] for j in selected] + [choices[i]]):
            selected.append(i)
            break
    if not any(curved(i) for i in selected):
        return None
    for i in rng.permutation(len(choices)):
        if len(selected) >= desired:
            break
        i = int(i)
        if i not in selected and _compatible([choices[j] for j in selected] + [choices[i]]):
            selected.append(i)
    return [choices[i] for i in selected]


def _make_pair(rng, track, divider, kind, split, rotation, shift):
    for attempt in range(80):
        if shift == "network":
            lanes = int(rng.integers(2, 5) if split == "train" else rng.integers(5, 9))
        else:
            lanes = int(rng.integers(2, 8))
        width = int(rng.integers(19, 31))
        network = _network(rng, lanes, width)
        if network is None:
            continue
        desired = int(rng.integers(3, 7) if split == "train" else rng.integers(5, 11))
        arrows = _sources(rng, network, kind, desired)
        if arrows is None:
            continue
        raw, _, _, _, _, bars = network
        grid = np.where(raw, track, 0).astype(np.int8)
        for (r, c), _ in bars:
            grid[r, c - 1:c + 2] = divider
        colors = list(map(int, rng.permutation([c for c in range(1, 9) if c not in (track, divider)])))
        # Cyclic assignment guarantees visibly different colors for the first
        # two witnesses, while some extra sources reuse colors.
        for i, arrow in enumerate(arrows):
            for p in arrow[2]:
                grid[p] = colors[i % len(colors)]
        if rng.random() < .5:
            grid = np.fliplr(grid).copy()
        turns = rotation
        if split == "test" and shift == "orientation":
            turns += 1
        grid = np.rot90(grid, turns).copy()
        try:
            out, info = _evaluate(grid)
        except AssertionError:
            # Dense source placement can hide every long track run. Reject
            # that presentation if role recovery is no longer unique.
            continue
        if not info["essential_a"] or not info["essential_b"]:
            continue
        if kind in ("collision", "both") and not info["collisions"]:
            continue
        if kind in ("strike", "both") and not any(
                p == arrow[0] and len(info["hits"][p]) == 1 and
                any(out[q] != grid[q] for q in arrow[2])
                for p in info["hits"] for arrow, _ in info["arrows"]):
            continue
        return {"input": grid, "output": out}
    raise RuntimeError("Bounded network/source construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    track, divider = map(int, rng.choice(np.arange(1, 9), 2, replace=False))
    rotation = int(rng.integers(4))
    shift = str(rng.choice(("network", "orientation")))
    kinds = ["collision", "strike"] + [str(rng.choice(("free", "collision", "strike", "both")))
                                       for _ in range(num_train - 2)]
    rng.shuffle(kinds)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", kinds), ("test", ["both"] * num_test)):
        for kind in specs:
            for _ in range(10):
                pair = _make_pair(rng, track, divider, kind, split, rotation, shift)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions = set(), set()
    teaching = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            expected, info = _evaluate(inp)
            assert np.array_equal(out, expected), "Incorrect output"
            conventions.add((info["track"], info["divider"]))
            assert info["curved"] and info["essential_a"] and info["essential_b"]
            for route in info["routes"]:
                assert len(route) == DISTANCE + 1 and len(set(route)) == DISTANCE + 1
                assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                           for a, b in zip(route, route[1:]))
            # Impact changes are confined to final track sites or entire target
            # arrows. Every unused input cell and every unstruck arrow survives.
            allowed = set(info["patch"]) | set(info["hits"])
            for arrow, _ in info["arrows"]:
                if arrow[0] in info["hits"]:
                    allowed.update(arrow[2])
                else:
                    assert all(inp[p] == out[p] for p in arrow[2])
            assert set(map(tuple, np.argwhere(inp != out).tolist())) <= allowed
            if split == "train":
                teaching["collision"] += info["collisions"]
                teaching["routed_collision"] += info["routed_collisions"]
                teaching["single"] += info["singles"]
                teaching["curved"] += info["curved"]
                teaching["strike"] += sum(arrow[0] in info["hits"] and
                                          len(info["hits"][arrow[0]]) == 1 and
                                          any(inp[p] != out[p] for p in arrow[2])
                                          for arrow, _ in info["arrows"])
    assert len(conventions) == 1
    assert all(teaching[k] for k in ("collision", "routed_collision", "single", "curved", "strike")), \
        "Demonstrations must expose all queried mechanisms"


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _mechanism_tests():
    # Hand-specified new-family examples, not generated labels or parent pairs.
    # Both tracks run from column 2 to 21; the divider center is (4,11).
    def fixture(specs):
        inp = np.zeros((9, 24), dtype=np.int8)
        inp[3, 2:22] = inp[5, 2:22] = 8
        inp[4, 10:13] = 7
        expected = inp.copy()
        expected[3:6, 10:13] = 8
        expected[3:6, 11] = 7
        for tip, direction, color in specs:
            for p in _arrow(tip, direction)[2]:
                inp[p] = expected[p] = color
        return inp, expected

    # Distinct straight endpoints (3,9) and (5,13) coincide on the rewired
    # railway: the second pulse moves right two, up two, then left one.
    inp, expected = fixture([((3, 4), (0, 1), 1), ((5, 8), (0, 1), 2)])
    expected[3, 9] = 9
    assert np.array_equal(solve(inp), expected)
    # Shooter at (3,3) strikes the tip (3,8). That struck arrow still fires:
    # two right, two down, one left, putting its ORIGINAL color at (5,9).
    inp, expected = fixture([((3, 3), (0, 1), 1), ((3, 8), (0, 1), 2)])
    for p in _arrow((3, 8), (0, 1))[2]:
        expected[p] = 1
    expected[5, 9] = 2
    snapshot = inp.copy()
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(inp, snapshot)
    # Recoloring object identity must not affect geometry or routing.
    swapped = inp.copy()
    swapped[inp == 1] = 4
    swapped[inp == 2] = 6
    recolored = expected.copy()
    recolored[expected == 1] = 4
    recolored[expected == 2] = 6
    assert np.array_equal(solve(swapped), recolored)
    # The family rejects branches, cycles and shots ending outside a path.
    for bad_mask in (np.ones((2, 2), dtype=bool),
                     np.array([[0, 1, 0], [1, 1, 1]], dtype=bool)):
        try:
            _paths(bad_mask)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid topology accepted")
    try:
        _route(np.ones((1, 4), dtype=bool), _arrow((0, 0), (0, 1)))
    except AssertionError:
        pass
    else:
        raise AssertionError("Short path accepted")


def self_test():
    """Stream 42 puzzles; retain hashes/counters, never a corpus of grid copies."""
    _mechanism_tests()
    signatures, shapes, palettes, structures = set(), set(), set(), set()
    arrow_counts, bar_counts, path_counts, axes = set(), set(), set(), set()
    totals = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                out, info = _evaluate(inp)
                assert np.array_equal(out, pair["output"])
                assert np.array_equal(solve(inp.tolist()), out)
                shapes.add(tuple(inp.shape))
                palettes.add((info["track"], info["divider"]))
                arrow_counts.add(len(info["arrows"]))
                bar_counts.add(len(info["bars"]))
                path_counts.add(len(info["paths"]))
                axes.update(axis for _, axis in info["bars"])
                normalized = inp.copy()
                normalized[inp == info["track"]] = 8
                normalized[inp == info["divider"]] = 7
                for arrow, _ in info["arrows"]:
                    for p in arrow[2]:
                        normalized[p] = 1
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized.tobytes()).hexdigest())
                totals["pairs"] += 1
                for key in ("collisions", "routed_collisions", "strikes", "singles", "curved"):
                    totals[key] += info[key]
                # Exact rotate/reflection equivariance checks both lane axes.
                if seed == 0:
                    assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
                    assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and len(palettes) >= 20
    assert len(arrow_counts) >= 5 and len(bar_counts) >= 6 and len(path_counts) >= 5
    assert axes == {(0, 1), (1, 0)}
    # All count combinations, including the minimum teaching budget.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(8128 + 10 * nt + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(42, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), distinct_shapes=len(shapes),
                normalized_structures=len(structures), role_palettes=len(palettes),
                arrow_counts=sorted(arrow_counts), divider_counts=sorted(bar_counts),
                component_counts=sorted(path_counts), **dict(totals))
