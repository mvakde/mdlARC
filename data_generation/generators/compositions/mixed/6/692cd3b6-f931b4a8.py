"""Count, build a two-scale gated field, then mark its shortest-route union.

Only NumPy and the standard library are used. Generation and tests are bounded,
serial, and in memory; importing this module or generating a puzzle writes no files.
"""
from collections import deque
import hashlib

import numpy as np


BACKGROUND = 0
CONNECTOR = 4
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PANEL_SIZES = tuple(range(7, 16))
HOST_SEEDS = (0, 321207319, 621816406, 427824689, 105797182, 110869391,
              432121322, 938509122, 935654074, 300928048, 435452628,
              575205320, 51829115, 20842053, 955112701, 235883175,
              867249915, 121467210, 199195751, 742860763, 17470504,
              744328825, 513947488, 913586447, 782723470, 414739993,
              612166927, 148851738, 637881119, 726158257, 75675664,
              869791825, 771473036, 972509942, 591263486, 209026250,
              981338678, 695026095, 85368713, 990452802, 235787804,
              135172866)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return array


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _decode(input_grid):
    grid = _grid(input_grid)
    if grid.shape[0] != grid.shape[1] or grid.shape[0] % 2:
        raise ValueError("the input must have four equal square panels")
    n = grid.shape[0] // 2
    if n not in PANEL_SIZES:
        raise ValueError("panel side must be 7..15")
    left, right = grid[:n, :n], grid[:n, n:]
    carrier, stencil = grid[n:, :n], grid[n:, n:]
    h = int(np.count_nonzero(left))
    w = int(np.count_nonzero(right))
    if (h, w) not in ((n, n), (n, 2*n), (2*n, n)):
        raise ValueError("counter dimensions must be n by n, n by 2n, or 2n by n")
    wall_colors = np.unique(carrier[carrier != BACKGROUND])
    if len(wall_colors) != 1 or int(wall_colors[0]) == CONNECTOR:
        raise ValueError("carrier must contain background and one wall ink")
    wall = int(wall_colors[0])
    if not np.any(carrier == BACKGROUND):
        raise ValueError("carrier must have traversable background")
    terminal_colors = np.unique(stencil[(stencil != BACKGROUND) & (stencil != wall)])
    if len(terminal_colors) != 1 or int(terminal_colors[0]) == CONNECTOR:
        raise ValueError("stencil must add one distinct terminal ink")
    terminal = int(terminal_colors[0])
    if np.any(grid == CONNECTOR):
        raise ValueError("connector ink 4 is reserved for the output")
    expected_native_count = 2 if (h, w) == (n, n) else 1
    if np.count_nonzero(stencil == terminal) != expected_native_count:
        raise ValueError("native stencil has the wrong number of terminals")
    return n, h, w, left, right, carrier, stencil, wall, terminal


def _resize(array, h, w):
    n = array.shape[0]
    return array[(np.arange(h)*n//h)[:, None],
                 (np.arange(w)*n//w)[None, :]].copy()


def _tile(array, h, w):
    n = array.shape[0]
    return array[np.arange(h)[:, None] % n, np.arange(w)[None, :] % n]


def _install(carrier, stencil, h, w):
    base = _resize(carrier, h, w)
    overlay = _tile(stencil, h, w)
    return np.where(overlay != BACKGROUND, overlay, base).astype(np.int8)


def _chambers(field, wall, terminal):
    """Read and validate complete seven-wall shells in the assembled field."""
    centers = tuple(tuple(map(int, p)) for p in np.argwhere(field == terminal))
    if len(centers) != 2:
        raise ValueError("the assembled field must have exactly two terminals")
    h, w = field.shape
    areas, gates = [], []
    for r, c in centers:
        if not (1 <= r < h-1 and 1 <= c < w-1):
            raise ValueError("a terminal has an incomplete chamber")
        areas.append({(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)})
        if any(field[r+dr, c+dc] != wall for dr in (-1, 1) for dc in (-1, 1)):
            raise ValueError("chamber corners must be walls")
        openings = []
        for d, (dr, dc) in enumerate(DIRECTIONS):
            value = int(field[r+dr, c+dc])
            if value == BACKGROUND:
                openings.append(d)
            elif value != wall:
                raise ValueError("invalid chamber side")
        if len(openings) != 1:
            raise ValueError("each terminal must have exactly one gate")
        direction = openings[0]
        dr, dc = DIRECTIONS[direction]
        outside = (r+2*dr, c+2*dc)
        if not (0 <= outside[0] < h and 0 <= outside[1] < w
                and field[outside] == BACKGROUND):
            raise ValueError("a gate must lead to an in-bounds background cell")
        gates.append(direction)
    if not areas[0].isdisjoint(areas[1]):
        raise ValueError("chambers overlap")
    return centers, tuple(gates)


def _distances(field, wall, start, diagonal=False):
    distance = np.full(field.shape, -1, dtype=np.int16)
    distance[start] = 0
    queue = deque([start])
    steps = DIRECTIONS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else DIRECTIONS
    h, w = field.shape
    while queue:
        r, c = queue.popleft()
        value = int(distance[r, c]) + 1
        for dr, dc in steps:
            a, b = r+dr, c+dc
            if (0 <= a < h and 0 <= b < w and distance[a, b] < 0
                    and field[a, b] != wall):
                distance[a, b] = value
                queue.append((a, b))
    return distance


def _route_union(field, wall, terminal, diagonal=False):
    centers = tuple(tuple(map(int, p)) for p in np.argwhere(field == terminal))
    if len(centers) != 2:
        raise ValueError("exactly two terminals are required")
    a = _distances(field, wall, centers[0], diagonal)
    b = _distances(field, wall, centers[1], diagonal)
    length = int(a[centers[1]])
    if length < 0:
        raise ValueError("terminals are disconnected")
    mask = (field == BACKGROUND) & (a >= 0) & (b >= 0) & (a+b == length)
    return mask, length, a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all conventions, counters, layers, and routes from this input alone."""
    _, h, w, _, _, carrier, stencil, wall, terminal = _decode(input_grid)
    field = _install(carrier, stencil, h, w)
    _chambers(field, wall, terminal)
    mask, _, _ = _route_union(field, wall, terminal)
    output = field.copy()
    output[mask] = CONNECTOR
    return output


def _obstacles(n, rng, light=False):
    """Bars, rectangles, elbows, little islands, or mixtures of these."""
    mask = np.zeros((n, n), dtype=bool)
    for _ in range(int(rng.integers(1, 4 if light else 6))):
        r, c = map(int, rng.integers(n, size=2))
        a, b = map(int, rng.integers(1, max(3, n//2+1), size=2))
        kind = int(rng.integers(4))
        if kind == 0:
            mask[r:min(n, r+a), c:min(n, c+b)] = True
        elif kind == 1:
            mask[r, max(0, c-b):min(n, c+b+1)] = True
        elif kind == 2:
            mask[max(0, r-a):min(n, r+a+1), c] = True
            mask[r, c:min(n, c+b)] = True
        else:
            locations = rng.choice(n*n, int(rng.integers(1, n+1)), replace=False)
            mask.ravel()[locations] = True
    return mask


def _native_layout(n, kind, rng):
    if kind == "single":
        options = [(a, b) for a in range(1, n-1) for b in range(a+4, n-1)]
        a, b = options[int(rng.integers(len(options)))]
        row = int(rng.integers(1, n-1))
        centers, gates = ((row, a), (row, b)), (1, 3)
        if rng.integers(2):
            centers = tuple((c, r) for r, c in centers)
            gates = (2, 0)
        return centers, gates
    if kind == "repeat":
        return ((int(rng.integers(2, n-2)), int(rng.integers(2, n-2))),), (int(rng.integers(4)),)
    points = [(r, c) for r in range(1, n-1) for c in range(1, n-1)]
    first_choices = [p for p in points if max(p[0]-1, n-2-p[0], p[1]-1, n-2-p[1]) >= 4]
    first = first_choices[int(rng.integers(len(first_choices)))]
    second_choices = [p for p in points if max(abs(p[i]-first[i]) for i in range(2)) >= 4]
    centers = (first, second_choices[int(rng.integers(len(second_choices)))])
    gates = []
    for r, c in centers:
        choices = [d for d, (dr, dc) in enumerate(DIRECTIONS)
                   if 0 <= r+2*dr < n and 0 <= c+2*dc < n]
        gates.append(choices[int(rng.integers(len(choices)))])
    return centers, tuple(gates)


def _stencil(n, centers, gates, wall, terminal, rng):
    stencil = np.zeros((n, n), dtype=np.int8)
    protected = np.zeros((n, n), dtype=bool)
    for (r, c), d in zip(centers, gates):
        dr, dc = DIRECTIONS[d]
        if not (0 <= r+2*dr < n and 0 <= c+2*dc < n):
            raise ValueError("native gate is trapped")
        stencil[r-1:r+2, c-1:c+2] = wall
        stencil[r, c] = terminal
        stencil[r+dr, c+dc] = BACKGROUND
        protected[r-1:r+2, c-1:c+2] = True
        protected[r+2*dr, c+2*dc] = True
    extra = _obstacles(n, rng, light=True) if rng.random() < .75 else np.zeros((n, n), bool)
    stencil[extra & ~protected] = wall
    # Check that another shell or its extra walls did not obstruct an exit.
    for (r, c), d in zip(centers, gates):
        dr, dc = DIRECTIONS[d]
        if stencil[r+dr, c+dc] or stencil[r+2*dr, c+2*dc]:
            raise ValueError("native exits conflict")
    return stencil


def _counter(n, count, rng, force_multi=False):
    panel = np.zeros((n, n), dtype=np.int8)
    style = int(rng.integers(4))
    if style == 0:
        positions = rng.choice(n*n, count, replace=False)
    elif style == 1:
        # A compact raster with a possibly incomplete last row.
        width = int(rng.integers((count+n-1)//n, n+1))
        height = (count+width-1)//width
        r, c = int(rng.integers(n-height+1)), int(rng.integers(n-width+1))
        positions = np.array([(r+i//width)*n+c+i % width for i in range(count)])
    elif style == 2:
        rows, cols = rng.permutation(n), rng.permutation(n)
        positions = rows[np.arange(count)//n]*n + cols[np.arange(count) % n]
    else:
        # Connected irregular occupancy; sorted frontiers make replay stable.
        start = tuple(map(int, rng.integers(n, size=2)))
        points, frontier = set(), {start}
        for _ in range(count):
            choices = sorted(frontier)
            r, c = choices[int(rng.integers(len(choices)))]
            frontier.remove((r, c))
            points.add((r, c))
            for dr, dc in DIRECTIONS:
                p = (r+dr, c+dc)
                if 0 <= p[0] < n and 0 <= p[1] < n and p not in points:
                    frontier.add(p)
        positions = np.array([r*n+c for r, c in sorted(points)])
    inks = np.array([c for c in range(1, 10) if c != CONNECTOR], dtype=np.int8)
    k = int(rng.integers(2, 5)) if force_multi or rng.random() < .6 else 1
    selected = rng.choice(inks, k, replace=False)
    values = np.concatenate((selected, rng.choice(selected, count-k)))
    rng.shuffle(values)
    panel.ravel()[positions] = values
    return np.rot90(panel, int(rng.integers(4))).copy()


def _alternative_mask(field, wall, terminal):
    try:
        return _route_union(field, wall, terminal)[0]
    except ValueError:
        return None


def _evidence(pair):
    n, h, w, left, right, carrier, stencil, wall, terminal = _decode(pair["input"])
    field = _install(carrier, stencil, h, w)
    union, length, distance = _route_union(field, wall, terminal)
    evidence = {"route_union" if int(union.sum()) > length-1 else "single_route"}
    tiled = _tile(stencil, h, w)
    base = _resize(carrier, h, w)
    if np.any((field == BACKGROUND) & (distance >= 0) & ~union):
        evidence.add("shortest_not_flood")
    without_carrier = _alternative_mask(tiled, wall, terminal)
    if without_carrier is not None and not np.array_equal(union, without_carrier):
        evidence.add("carrier_changes_routes")
    tiled_carrier = np.where(tiled != BACKGROUND, tiled, _tile(carrier, h, w))
    if not np.array_equal(tiled_carrier, field):
        evidence.add("carrier_resize")
    wrong_mask = _alternative_mask(tiled_carrier, wall, terminal)
    if not np.array_equal(union, wrong_mask):
        evidence.add("resize_changes_routes")
    scaled_stencil = _resize(stencil, h, w)
    if not np.array_equal(np.where(scaled_stencil != BACKGROUND, scaled_stencil, base), field):
        evidence.add("native_stencil")
    if np.any((tiled == BACKGROUND) & (base == wall)):
        evidence.add("transparent_stencil")
    if np.any((tiled != BACKGROUND) & (tiled != base)):
        evidence.add("opaque_overwrite")
    centers, _ = _chambers(field, wall, terminal)
    rr, cc = np.indices(field.shape)
    manhattan = [abs(rr-r)+abs(cc-c) for r, c in centers]
    shortest_straight = int(np.abs(np.subtract(centers[0], centers[1])).sum())
    if not np.array_equal(union, (field == BACKGROUND) & (sum(manhattan) == shortest_straight)):
        evidence.add("obstacle_aware")
    diagonal = _route_union(field, wall, terminal, diagonal=True)[0]
    if not np.array_equal(union, diagonal):
        evidence.add("four_neighbors")
    if h != w:
        evidence.add("counter_order")
    for panel, dimension in ((left, h), (right, w)):
        r, c = np.nonzero(panel)
        if (int(r.max()-r.min()+1)*int(c.max()-c.min()+1)) != dimension:
            evidence.add("counter_occupancy")
        _, frequencies = np.unique(panel[panel != BACKGROUND], return_counts=True)
        if int(frequencies.max()) != dimension:
            evidence.add("all_counter_inks")
        if len(frequencies) != dimension:
            evidence.add("cells_not_colors")
    return evidence


REQUIRED_EVIDENCE = frozenset((
    "single_route", "route_union", "shortest_not_flood", "carrier_changes_routes",
    "carrier_resize", "resize_changes_routes", "native_stencil", "transparent_stencil",
    "opaque_overwrite", "obstacle_aware", "four_neighbors", "counter_order",
    "counter_occupancy", "all_counter_inks", "cells_not_colors"))


def _sample_pair(n, kind, rng, need_multi=False):
    # At most 240 attempts, retaining only the current candidate.
    for _ in range(240):
        try:
            centers, gates = _native_layout(n, kind, rng)
            h, w = (n, n)
            if kind == "repeat":
                h, w = (n, 2*n) if rng.integers(2) else (2*n, n)
            wall, terminal = map(int, rng.choice([c for c in range(1, 10) if c != CONNECTOR], 2, replace=False))
            stencil = _stencil(n, centers, gates, wall, terminal, rng)
            tiled = _tile(stencil, h, w)
            installed_centers = tuple(tuple(map(int, p)) for p in np.argwhere(tiled == terminal))
            carrier_mask = _obstacles(n, rng)
            for r, c in installed_centers:
                d = gates[0] if kind == "repeat" else gates[centers.index((r, c))]
                dr, dc = DIRECTIONS[d]
                for step in (1, 2):
                    a, b = r+step*dr, c+step*dc
                    carrier_mask[a*n//h, b*n//w] = False
            if kind == "single":
                (r0, c0), (r1, c1) = centers
                carrier_mask[min(r0, r1):max(r0, r1)+1, min(c0, c1):max(c0, c1)+1] = False
                # No added stencil obstacle may cross the demonstrated straight route.
                segment = stencil[min(r0, r1):max(r0, r1)+1, min(c0, c1):max(c0, c1)+1]
                if np.any((segment != BACKGROUND) & (segment != terminal)):
                    continue
            if not carrier_mask.any() or carrier_mask.all():
                continue
            carrier = np.where(carrier_mask, wall, BACKGROUND).astype(np.int8)
            field = _install(carrier, stencil, h, w)
            _chambers(field, wall, terminal)
            mask, length, distance = _route_union(field, wall, terminal)
            if not np.any((field == BACKGROUND) & (distance >= 0) & ~mask):
                continue
            if kind == "single" and int(mask.sum()) != length-1:
                continue
            if need_multi and int(mask.sum()) <= length-1:
                continue
            if need_multi:
                no_carrier = _alternative_mask(tiled, wall, terminal)
                wrong_field = np.where(tiled != BACKGROUND, tiled, _tile(carrier, h, w))
                wrong_route = _alternative_mask(wrong_field, wall, terminal)
                if (no_carrier is None or np.array_equal(mask, no_carrier)
                        or np.array_equal(mask, wrong_route)):
                    continue
            left = _counter(n, h, rng, force_multi=need_multi)
            right = _counter(n, w, rng)
            inp = np.block([[left, right], [carrier, stencil]])
            out = field.copy()
            out[mask] = CONNECTOR
            pair = {"input": inp, "output": out}
            if need_multi and not (REQUIRED_EVIDENCE - {"single_route"}) <= _evidence(pair):
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded sampling could not construct the requested route geometry")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    bank = tuple(map(int, rng.choice(PANEL_SIZES, 4, replace=False)))
    # Two witnesses establish the operations even at the minimum legal count.
    puzzle = {"train": [_sample_pair(bank[0], "single", rng),
                         _sample_pair(bank[1], "repeat", rng, need_multi=True)], "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < count:
            sizes = bank if split == "train" else tuple(n for n in PANEL_SIZES
                         if n not in {p["input"].shape[0]//2 for p in puzzle["train"]})
            for _ in range(16):
                n = int(rng.choice(sizes))
                kind = "general" if rng.random() < .45 else "repeat"
                pair = _sample_pair(n, kind, rng)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys"
    assert all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists"
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_sizes, test_sizes, training_evidence = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            n, h, w, _, _, carrier, stencil, wall, terminal = _decode(inp)
            field = _install(carrier, stencil, h, w)
            centers, gates = _chambers(field, wall, terminal)
            union, length, distance = _route_union(field, wall, terminal)
            assert out.shape == (h, w), "counters do not give the output dimensions"
            changed = out != field
            assert np.array_equal(changed, union), "changed cells are not the full shortest union"
            assert np.all(out[union] == CONNECTOR), "incorrect route ink"
            assert np.all(out[~union] == field[~union]), "walls, terminals, or background changed"
            assert int(np.count_nonzero(out == terminal)) == 2
            assert int(union.sum()) >= length-1 > 0
            assert np.any((field == BACKGROUND) & (distance >= 0) & ~union), "flood-fill ambiguity"
            for (r, c), d in zip(centers, gates):
                dr, dc = DIRECTIONS[d]
                assert union[r+dr, c+dc], "a mandatory gate was not colored"
            assert np.array_equal(solve(inp), out)
            if split == "train":
                train_sizes.add(n)
                training_evidence |= _evidence(pair)
            else:
                test_sizes.add(n)
    assert len(train_sizes) >= 2, "training needs two panel sizes"
    assert train_sizes.isdisjoint(test_sizes), "test must transfer to an unseen panel size"
    assert REQUIRED_EVIDENCE <= training_evidence, "missing training evidence: " + str(REQUIRED_EVIDENCE-training_evidence)


def _features(pair):
    n, h, w, _, _, carrier, stencil, wall, terminal = _decode(pair["input"])
    field = _install(carrier, stencil, h, w)
    centers, gates = _chambers(field, wall, terminal)
    mask, length, _ = _route_union(field, wall, terminal)
    return {"n": n, "shape": (h, w), "scale": (h//n, w//n), "gates": gates,
            "separation": tuple(abs(centers[0][i]-centers[1][i]) for i in range(2)),
            "wall": wall, "terminal": terminal, "route_cells": int(mask.sum()),
            "length": length, "multiple": int(mask.sum()) > length-1,
            "edge_route": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
            "carrier": _key(carrier == wall), "stencil": _key(stencil != BACKGROUND)}


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def self_test():
    """Streaming 42-seed audit, count boundaries, corruption, and hand witnesses."""
    fingerprints, features = set(), {"train": [], "test": []}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in features:
            features[split].extend(_features(pair) for pair in puzzle[split])
    assert len(fingerprints) == 42
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for split, rows in features.items():
        assert len({f["n"] for f in rows}) >= 7, split
        assert {(1, 1), (1, 2), (2, 1)} <= {f["scale"] for f in rows}
        assert len({f["carrier"] for f in rows}) >= 30
        assert len({f["stencil"] for f in rows}) >= 30
        assert len({f["route_cells"] for f in rows}) >= 15
        assert len({f["gates"] for f in rows}) >= 6
        assert {False, True} <= {f["multiple"] for f in rows}
        assert any(f["edge_route"] for f in rows)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+10*a+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # A symmetric obstacle has two equal detours: neither one path nor a
    # filled rectangle is the intended answer. This is not a sampled label.
    field = np.zeros((7, 9), dtype=np.int8)
    field[3, 1], field[3, 7], field[3, 4] = 5, 5, 2
    mask, length, _ = _route_union(field, 2, 5)
    assert length == 8 and mask[2, 4] and mask[4, 4] and not mask[3, 4]
    assert int(mask.sum()) > length-1 and not mask[0].any()
    # Hand-specified B witnesses separate enlarged carrier cells from native
    # stencil period and check the transparent positions numerically.
    carrier = np.array([[0, 2], [2, 0]])
    stencil = np.array([[0, 5], [0, 0]])
    assert np.array_equal(_install(carrier, stencil, 2, 4), [[0, 5, 2, 5], [2, 2, 0, 0]])
    assert np.array_equal(_install(carrier, stencil, 4, 2), [[0, 5], [0, 2], [2, 5], [2, 0]])
    # Counter arrangement and ink identities carry no additional meaning.
    p = generate_puzzle(0)["test"][0]
    changed = p["input"].copy()
    n = changed.shape[0]//2
    for panel in (changed[:n, :n], changed[:n, n:]):
        occupied = int(np.count_nonzero(panel))
        panel[:] = 0
        for i in range(occupied):
            panel[i//n, i % n] = 3
    assert np.array_equal(solve(changed), p["output"])
    # Pair-level recoloring is recoverable from the input, including counter
    # colors that happen to reuse either semantic ink.
    _, _, _, _, _, _, _, wall, terminal = _decode(p["input"])
    changed = p["input"].copy()
    changed[p["input"] == wall] = terminal
    changed[p["input"] == terminal] = wall
    expected = p["output"].copy()
    expected[p["output"] == wall] = terminal
    expected[p["output"] == terminal] = wall
    assert np.array_equal(solve(changed), expected)
    # Damaging one answer, or choosing one shortest path, must fail validation.
    puzzle = generate_puzzle(0)
    for damage in ("wall", "missing_route"):
        corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]]
                     for s in ("train", "test")}
        pair = next(p for p in corrupted["train"] if "route_union" in _evidence(p))
        if damage == "wall":
            n, _, _, _, _, _, _, wall, _ = _decode(pair["input"])
            pos = tuple(np.argwhere(pair["output"] == wall)[0])
        else:
            pos = tuple(np.argwhere(pair["output"] == CONNECTOR)[0])
        pair["output"][pos] = BACKGROUND
        try:
            validate_generated_puzzle(corrupted)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupted output accepted")
    return {"seeds_checked": len(HOST_SEEDS), "distinct_puzzles": len(fingerprints),
            "by_split": {s: {"pairs": len(rows), "panel_sizes": sorted({f["n"] for f in rows}),
                             "output_shapes": len({f["shape"] for f in rows}),
                             "carrier_masks": len({f["carrier"] for f in rows}),
                             "stencil_masks": len({f["stencil"] for f in rows}),
                             "gate_pairs": len({f["gates"] for f in rows}),
                             "route_sizes": len({f["route_cells"] for f in rows}),
                             "multiple_routes": sum(f["multiple"] for f in rows),
                             "edge_routes": sum(f["edge_route"] for f in rows)}
                         for s, rows in features.items()},
            "independent_solver": False}
