"""A distributed minority key, aligned and broadcast by contour depth.

Only Python's standard library and NumPy are used. No parent module, source
fixture, seed, or generation state is used by the input-only solver.
"""
from collections import Counter, deque
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer) and
             np.all((grid >= 0) & (grid <= 9)), "integer colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    groups = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        group = np.zeros_like(mask)
        queue = deque([(sr, sc)])
        seen[sr, sc] = True
        while queue:
            r, c = queue.popleft()
            group[r, c] = True
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        groups.append(group)
    return groups


def _depth(mask, diagonal=True):
    """Simultaneous erosion, explicitly treating the padded exterior as false."""
    h, w = mask.shape
    current = mask.copy()
    depth = np.full((h, w), -1, dtype=np.int16)
    level = 0
    while np.any(current):
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in (DIR8 if diagonal else DIR4):
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _depth_bfs(mask):
    """A second depth computation for self-test, using boundary-distance BFS."""
    h, w = mask.shape
    depth = np.full((h, w), -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc])
               for dr, dc in DIR8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                queue.append((rr, cc))
    return depth


def _paths(mask, depth, length, row_direction=None):
    h, w = mask.shape

    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and mask[r, c]

    result = []
    for r, c in np.argwhere(depth == 0):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            if row_direction is not None and dr != row_direction:
                continue
            for dc in (-1, 1):
                if (inside(r - dr, c) or inside(r, c - dc) or
                        not all(inside(rr, cc) for rr, cc in
                                ((r + dr, c), (r, c + dc), (r + dr, c + dc)))):
                    continue
                cells = tuple((r + k * dr, c + k * dc) for k in range(length))
                if all(inside(rr, cc) and depth[rr, cc] == k
                       for k, (rr, cc) in enumerate(cells)):
                    result.append((cells, dr, dc))
    return result


def _geometry(wall, length):
    _require(len(_components(wall)) == 1, "divider must be connected")
    _require(sum(bool(np.any(edge)) for edge in
                 (wall[0], wall[-1], wall[:, 0], wall[:, -1])) >= 2,
             "divider must meet two canvas edges")
    _require(not np.any(wall[:-1, :-1] & wall[1:, :-1] &
                        wall[:-1, 1:] & wall[1:, 1:]), "divider must be thin")
    masks = _components(~wall)
    _require(len(masks) >= 3, "at least three chambers are required")
    depths = [_depth(m) for m in masks]
    _require(all(_paths(m, d, length) for m, d in zip(masks, depths)),
             "each chamber needs sufficient corner-key capacity")
    return masks, depths


def _decode(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    ranks = np.argsort(-counts, kind="stable")
    neutral, divider = map(int, ranks[:2])
    _require(counts[neutral] > counts[divider] > counts[ranks[2]] > 0,
             "neutral and divider must have strict frequency ranks")
    wall = grid == divider
    # Geometry can first be segmented without knowing the shared key length.
    masks = _components(~wall)
    depths = [_depth(m) for m in masks]
    cue = (grid != neutral) & ~wall
    _require(np.any(cue), "the shared key length must be visible")
    length = 1 + max(int(d[m & cue].max()) for m, d in zip(masks, depths)
                     if np.any(m & cue))
    _require(1 <= length <= 4, "key length must be 1..4")
    masks, depths = _geometry(wall, length)
    keys, directions = [], []
    for mask, depth in zip(masks, depths):
        positions = [tuple(map(int, p)) for p in np.argwhere(mask & cue)]
        key = [neutral] * length
        occupied = set()
        for r, c in positions:
            k = int(depth[r, c])
            _require(k not in occupied and k < length, "one visible cue per key depth")
            occupied.add(k)
            key[k] = int(grid[r, c])
        options = [(cells, dr, dc) for cells, dr, dc in _paths(mask, depth, length)
                   if all((r, c) == cells[int(depth[r, c])] for r, c in positions)]
        _require(bool(options), "key cues must lie on a convex-corner diagonal")
        _require(all(tuple(int(grid[r, c]) for r, c in cells) == tuple(key)
                     for cells, _, _ in options), "key reading must be unique")
        keys.append(key)
        directions.append(tuple(sorted({(dr, dc) for _, dr, dc in options})))
    keys = np.asarray(keys, dtype=np.int8)
    minority, common, donors = [], [], []
    for k in range(length):
        frequencies = Counter(map(int, keys[:, k]))
        _require(len(frequencies) == 2 and sorted(frequencies.values()) == [1, len(masks) - 1],
                 "each key depth requires a repeated color and one visible exception")
        rare = next(c for c, count in frequencies.items() if count == 1)
        usual = next(c for c, count in frequencies.items() if count == len(masks) - 1)
        minority.append(rare)
        common.append(usual)
        donors.append(int(np.flatnonzero(keys[:, k] == rare)[0]))
    return dict(grid=grid, neutral=neutral, divider=divider, wall=wall, masks=masks,
                depths=depths, length=length, keys=keys, minority=tuple(minority),
                common=tuple(common), donors=tuple(donors), directions=directions)


def _paint(wall, masks, depths, period, divider):
    result = np.full(wall.shape, divider, dtype=np.int8)
    colors = np.asarray(period, dtype=np.int8)
    for mask, depth in zip(masks, depths):
        result[mask] = colors[depth[mask] % len(colors)]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, key slots, votes and contours exclusively from input."""
    a = _decode(input_grid)
    return _paint(a["wall"], a["masks"], a["depths"], a["minority"], a["divider"])


def _contour_witness(a):
    colors = np.asarray(a["minority"], dtype=np.int8)
    for mask, depth in zip(a["masks"], a["depths"]):
        rr, cc = np.nonzero(mask)
        rectangular = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                          cc - cc.min(), cc.max() - cc))
        actual = colors[depth[mask] % len(colors)]
        if (np.any(actual != colors[rectangular % len(colors)]) and
                np.any(actual != colors[_depth(mask, False)[mask] % len(colors)])):
            return True
    return False


def _partition(rng, shape, minimum, target):
    h, w = shape
    rooms = [(0, h - 1, 0, w - 1)]
    wall = np.zeros(shape, dtype=bool)
    while len(rooms) < target:
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, span in ((0, b - a + 1), (1, d - c + 1))
                   if span >= 2 * minimum + 1]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + minimum, b - minimum + 1))
            wall[cut, max(0, c - 1):min(w, d + 2)] = True
            rooms.extend(((a, cut - 1, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + minimum, d - minimum + 1))
            wall[max(0, a - 1):min(h, b + 2), cut] = True
            rooms.extend(((a, b, c, cut - 1), (a, b, cut + 1, d)))
    return wall, rooms


def _attach(rng, wall, rooms, minimum):
    """An enclosure remains attached to an existing side wall, never free noise."""
    options = []
    for a, b, c, d in rooms:
        for transpose in (False, True):
            aa, bb, cc, dd = (c, d, a, b) if transpose else (a, b, c, d)
            work = wall.T if transpose else wall
            if bb - aa + 1 < minimum + 6 or dd - cc + 1 < minimum + 4:
                continue
            for sign, side in ((1, cc - 1), (-1, dd + 1)):
                if 0 <= side < work.shape[1] and np.all(work[aa:bb + 1, side]):
                    options.append((transpose, aa, bb, cc, dd, sign, side))
    if not options:
        return wall
    transpose, a, b, c, d, sign, side = options[int(rng.integers(len(options)))]
    result = wall.copy()
    work = result.T if transpose else result
    top = int(rng.integers(a + 2, b - minimum - 2))
    bottom = int(rng.integers(top + minimum + 1, b - 1))
    far = side + sign * int(rng.integers(minimum + 1, d - c - 1))
    lo, hi = sorted((side, far))
    work[top, lo:hi + 1] = work[bottom, lo:hi + 1] = True
    work[top:bottom + 1, far] = True
    return result


def _step(rng, wall):
    for _ in range(35):
        transpose = bool(rng.integers(2))
        work = wall.T if transpose else wall
        points = np.argwhere(work)
        r, c = map(int, points[int(rng.integers(len(points)))])
        end = r + int(rng.integers(3, 12))
        offset = int(rng.choice((-3, -2, -1, 1, 2, 3)))
        lo, hi = sorted((c, c + offset))
        if r < 1 or end + 1 >= work.shape[0] or lo < 1 or hi + 1 >= work.shape[1]:
            continue
        block = work[r - 1:end + 2, lo - 1:hi + 2]
        expected = np.zeros_like(block)
        expected[:, c - lo + 1] = True
        if not np.array_equal(block, expected):
            continue
        result = wall.copy()
        modified = result.T if transpose else result
        modified[r + 1:end, c] = False
        modified[r, lo:hi + 1] = modified[end, lo:hi + 1] = True
        modified[r:end + 1, c + offset] = True
        return result
    return wall


def _fallback():
    wall = np.zeros((25, 29), dtype=bool)
    wall[:, 9] = True
    wall[12, 9:] = True
    wall[6:10, 9] = False
    wall[5, 9:12] = wall[10, 9:12] = True
    wall[5:11, 11] = True
    return wall


def _make_layout(rng, length, witness, test):
    minimum = max(3, 2 * length - 1, 2 * length + 1 if witness else 0)
    low = max(9, 2 * minimum + 2)
    for attempt in range(45):
        if attempt == 44:
            wall = _fallback()
        else:
            shape = (30, 30) if rng.random() < 0.14 else tuple(map(int, rng.integers(low, 31, size=2)))
            wall, rooms = _partition(rng, shape, minimum, int(rng.integers(3, 9)))
            if rng.random() < (0.5 if test else 0.3):
                wall = _attach(rng, wall, rooms, 2 * length - 1 if length > 1 else 3)
            for _ in range(int(rng.integers(1 if witness else 0, 7 if test else 5))):
                wall = _step(rng, wall)
        try:
            masks, depths = _geometry(wall, length)
            _require(len(masks) <= 8, "sampler chamber limit")
            _require(all(_paths(m, d, length, direction) for m, d in zip(masks, depths)
                         for direction in (-1, 1)), "sampler needs both corner orientations")
            if witness:
                _require(max(int(d.max()) for d in depths) >= length, "witness must repeat its cycle")
                probe = dict(masks=masks, depths=depths, minority=tuple(range(length)))
                _require(_contour_witness(probe), "witness needs a diagnostic concavity")
            return wall, masks, depths
        except ValueError:
            if attempt == 44:
                raise
    raise AssertionError("bounded layout loop must return")


def _make_pair(rng, length, witness, test, neutral, divider):
    wall, masks, depths = _make_layout(rng, length, witness, test)
    n = len(masks)
    available = np.asarray([c for c in range(10) if c not in (neutral, divider)])
    palette = list(map(int, rng.permutation(available)))
    if witness and length == 2:
        common, minority = (palette[0], palette[1]), (palette[1], palette[0])
    elif witness and length == 3:
        p, q, r = palette[:3]
        common, minority = (neutral, r, p), (p, neutral, q)
    else:
        common = tuple(map(int, rng.choice([c for c in range(10) if c != divider], size=length)))
        if test:
            # Visible anchors make the orientation shift observable even when
            # some later key entries are neutral or repeated.
            common = (int(rng.choice(available)),) + common[1:]
        minority = tuple(int(rng.choice([c for c in range(10)
                                        if c not in (divider, c0) and
                                        not (test and k == 0 and c == neutral)]))
                         for k, c0 in enumerate(common))
        # Common and minority are unequal, so at least one of them visibly
        # establishes the final index even when the other is neutral.
    donors = list(map(int, rng.integers(n, size=length)))
    if witness:
        donors[:2] = list(map(int, rng.choice(n, 2, replace=False)))
    keys = np.tile(np.asarray(common, dtype=np.int8), (n, 1))
    for k, donor in enumerate(donors):
        keys[donor, k] = minority[k]
    grid = np.full(wall.shape, neutral, dtype=np.int8)
    grid[wall] = divider
    upward = set(map(int, rng.choice(n, 2, replace=False))) if test else set()
    for i, (mask, depth) in enumerate(zip(masks, depths)):
        direction = -1 if i in upward else (int(rng.choice((-1, 1))) if test else 1)
        options = _paths(mask, depth, length, direction)
        cells, _, _ = options[int(rng.integers(len(options)))]
        for k, cell in enumerate(cells):
            grid[cell] = keys[i, k]
    output = _paint(wall, masks, depths, minority, divider)
    pair = {"input": grid, "output": output}
    # Check actual visible encoding, including neutral gaps and role ranks.
    a = _decode(grid)
    _require(a["length"] == length and a["minority"] == minority,
             "constructed instruction must be recoverable")
    return pair


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    _require(nt > ne, "num_train must be greater than num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    role_order = list(map(int, rng.permutation(10)))
    offset = int(rng.integers(1, 10))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + ne):
        test = i >= nt
        witness = i < 2
        length = i + 2 if witness else int(rng.integers(2 if test else 1, 5 if test else 4))
        neutral, divider = role_order[i % 10], role_order[(i + offset) % 10]
        for _ in range(20):
            pair = _make_pair(rng, length, witness, test, neutral, divider)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise AssertionError("failed bounded duplicate rejection")
        seen.add(_key(pair["input"]))
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(analyses):
    two = [a for a in analyses if a["length"] == 2 and
           a["minority"] == a["common"][::-1] and len(set(a["minority"])) == 2]
    three = [a for a in analyses if a["length"] == 3 and
             a["minority"][1] == a["neutral"] and a["common"][0] == a["neutral"] and
             len(set(a["minority"])) == 3]
    for options in (two, three):
        _require(any(len(set(a["donors"])) >= 2 and
                     all(tuple(k) != a["minority"] for k in a["keys"]) and
                     max(int(d.max()) for d in a["depths"]) >= a["length"] and
                     _contour_witness(a) for a in options),
                 "training needs recombination, repetition and diagnostic contour witnesses")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", ne, 1, 4)
    _require(nt > ne, "training count must exceed test count")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _decode(inp)
            expected = _paint(a["wall"], a["masks"], a["depths"], a["minority"], a["divider"])
            _require(np.array_equal(out, expected), "output must broadcast the minority contour cycle")
            _require(np.array_equal(out == a["divider"], a["wall"]), "preserve exactly the divider")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "output colors must be visible")
            if split == "train":
                train.append(a)
                _require(all(any(dr == 1 for dr, _ in ds) for ds in a["directions"]),
                         "training keys must admit downward readings")
            else:
                _require(sum(bool(ds) and all(dr == -1 for dr, _ in ds)
                             for ds in a["directions"]) >= 2, "tests require visible upward keys")
    _witnesses(train)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded diversity, counterfactual, interface and corruption checks."""
    # Exercise the guaranteed backup explicitly, rather than relying on a
    # finite seed sample to happen to exhaust the normal layout search.
    backup = _fallback()
    for length in range(1, 5):
        masks, depths = _geometry(backup, length)
        assert len(masks) == 3
        assert all(_paths(m, d, length, dr) for m, d in zip(masks, depths)
                   for dr in (-1, 1))
        if length in (2, 3):
            assert max(int(d.max()) for d in depths) >= length
            assert _contour_witness(dict(masks=masks, depths=depths,
                                         minority=tuple(range(length))))
    puzzles, shapes, networks, roles = set(), set(), set(), set()
    regions, lengths, directions, holes = set(), set(), set(), 0
    rectangular_disagreements = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert all(np.array_equal(d, _depth_bfs(m)) for m, d in zip(a["masks"], a["depths"]))
                shapes.add(a["grid"].shape)
                networks.add(_key(a["wall"].astype(np.int8)))
                roles.add((a["neutral"], a["divider"]))
                regions.add(len(a["masks"]))
                lengths.add(a["length"])
                directions.update(direction for ds in a["directions"] for direction in ds)
                rectangular_disagreements += _contour_witness(a)
                holes += sum(not np.any(m[0]) and not np.any(m[-1]) and
                             not np.any(m[:, 0]) and not np.any(m[:, -1]) for m in a["masks"])
        for pair in p["train"][:2]:
            a = _decode(pair["input"])
            majority = _paint(a["wall"], a["masks"], a["depths"], a["common"], a["divider"])
            assert not np.array_equal(majority, pair["output"])
            own = np.full(a["grid"].shape, a["divider"], dtype=np.int8)
            for m, d, key in zip(a["masks"], a["depths"], a["keys"]):
                own[m] = key[d[m] % a["length"]]
            assert not np.array_equal(own, pair["output"])
            assert not np.array_equal(pair["input"], pair["output"])
    assert len(puzzles) == 42 and len(shapes) >= 50 and len(networks) >= 120
    assert len(roles) >= 50 and len(regions) >= 5 and lengths == {1, 2, 3, 4}
    assert directions == set((dr, dc) for dr in (-1, 1) for dc in (-1, 1))
    assert rectangular_disagreements >= 84 and holes >= 1
    assert _puzzle_key(generate_puzzle(832235409, 2, 1)) == _puzzle_key(generate_puzzle(832235409, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    base = generate_puzzle(0, 2, 1)
    for defect in ("output", "divider", "missing_vote", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad["train"][0]
        a = _decode(pair["input"])
        if defect == "output":
            r, c = map(int, np.argwhere(~a["wall"])[-1])
            pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        elif defect == "divider":
            r, c = map(int, np.argwhere(a["wall"])[0])
            pair["output"][r, c] = a["neutral"]
        elif defect == "missing_vote":
            donor = a["donors"][0]
            r, c = map(int, np.argwhere(a["masks"][donor] & (a["depths"][donor] == 0) &
                                      (pair["input"] != a["neutral"]))[0])
            pair["input"][r, c] = a["common"][0]
        else:
            bad["test"][0] = {k: v.copy() for k, v in pair.items()}
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError("validator accepted " + defect)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, legal counts, counterfactuals and mutations)")
