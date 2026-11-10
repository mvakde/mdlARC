"""Census docks: rewired paths gather beads into width-matched cargo.

All semantic colors are visible. No parent code, labels, seeds, or training
examples are consulted by solve(). See task_spec.md for the complete rule.
"""

from itertools import permutations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _near(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _paths(cells):
    """Order every component, rejecting branches, isolated cells, and cycles."""
    result = []
    for comp in _components(cells):
        adj = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                   if (p[0] + dr, p[1] + dc) in comp] for p in comp}
        assert all(len(ns) in (1, 2) for ns in adj.values()), "branched/isolated rail"
        ends = sorted(p for p in comp if len(adj[p]) == 1)
        assert len(ends) == 2, "closed rail loop"
        path, previous = [ends[0]], None
        while path[-1] != ends[1]:
            nxt = [q for q in adj[path[-1]] if q != previous]
            assert len(nxt) == 1
            previous = path[-1]
            path.append(nxt[0])
        assert len(path) == len(comp)
        result.append(path)
    return result


def _fork(cells):
    """Parse a base and three positive spokes, independently of their lengths."""
    parses = []
    for out in DIRS:
        perp = (-out[1], out[0])
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            a, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((a, q, length))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        a, b, stop = runs[1]
        if a + b != width or not 1 <= b - a <= 3:
            continue
        parses.append(dict(out=out, perp=perp, kmin=kmin, qmin=qmin,
                           width=width, runs=runs, span=b - a, stop=stop,
                           q0=a, cells=cells))
    assert len(parses) == 1, "receiver needs one base and three positive spokes"
    return parses[0]


def _fork_cell(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    return (k * fork["out"][0] + q * fork["perp"][0],
            k * fork["out"][1] + q * fork["perp"][1])


def _rewire(grid, bg, rail, bead, bar):
    answer = grid.copy()
    answer[answer == bead] = rail
    writes, centers, overlaps = {}, [], 0
    for comp in _components(_cells(grid == bar)):
        assert len(comp) == 3
        rs, cs = zip(*comp)
        horizontal = len(set(rs)) == 1 and max(cs) - min(cs) == 2
        vertical = len(set(cs)) == 1 and max(rs) - min(rs) == 2
        assert horizontal or vertical, "switch must be a straight triomino"
        r, c = sum(rs) // 3, sum(cs) // 3
        assert 0 < r < grid.shape[0] - 1 and 0 < c < grid.shape[1] - 1
        centers.append((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r + dr, c + dc
                on_original = dr == 0 if horizontal else dc == 0
                if on_original:
                    assert int(grid[p]) == bar
                else:
                    assert int(grid[p]) in (rail, bead)
                on_turned = dc == 0 if horizontal else dr == 0
                value = bar if on_turned else rail
                if on_turned:
                    assert grid[p] != bead, "bead cannot be replaced by a gate"
                if p in writes:
                    assert writes[p] == value, "conflicting switch neighborhoods"
                    overlaps += 1
                writes[p] = value
    assert centers
    for p, value in writes.items():
        answer[p] = value
    return answer, centers, overlaps


def _analyze(input_grid, turn=True):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.all((grid >= 0) & (grid <= 9))
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "background border required"
    colors = set(map(int, np.unique(grid))) - {bg}
    assert len(colors) == 4
    comps = {color: _components(_cells(grid == color)) for color in colors}
    fork_options = []
    for color in sorted(colors):
        try:
            fs = [_fork(c) for c in comps[color]]
        except AssertionError:
            continue
        fork_options.append((color, fs))
    assert len(fork_options) == 1, "receiver role must be visible by shape"
    structure, forks = fork_options[0]
    candidates = []
    for rail, bead, bar in permutations(sorted(colors - {structure})):
        if not all(len(c) == 1 for c in comps[bead]):
            continue
        try:
            structural, centers, overlaps = _rewire(grid, bg, rail, bead, bar)
            if not turn:
                structural = grid.copy()
                structural[structural == bead] = rail
            paths = _paths(_cells(structural == rail))
            endpoints = {p: i for i, path in enumerate(paths) for p in (path[0], path[-1])}
            docks = []
            for source in forks:
                fork = dict(source)
                face = {_fork_cell(fork, fork["stop"] + 1, q)
                        for q in range(fork["q0"], fork["q0"] + fork["span"])}
                contacts = {_fork_cell(fork, fork["stop"] + 2, q)
                            for q in range(fork["q0"], fork["q0"] + fork["span"])}
                contacts &= set(endpoints)
                assert len(contacts) == 1, "each dock must attach to exactly one rail terminal"
                contact = next(iter(contacts))
                assert all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                           and grid[r, c] == bg for r, c in face), "cargo slot must be empty"
                fork.update(face=face, contact=contact, path=endpoints[contact])
                docks.append(fork)
            assert len({f["contact"] for f in docks}) == len(docks)
            beads = _cells(grid == bead)
            assert beads <= _cells(structural == rail)
            counts = [len(beads & set(path)) for path in paths]
            for pi, count in enumerate(counts):
                assert sum(f["path"] == pi and f["span"] == count for f in docks) <= 1
            candidates.append(dict(grid=grid, bg=bg, rail=rail, bead=bead, bar=bar,
                                   structure=structure, structural=structural,
                                   paths=paths, docks=docks, beads=beads,
                                   counts=counts, centers=centers, overlaps=overlaps))
        except AssertionError:
            continue
    assert len(candidates) == 1, "rail, bead, and switch roles must have one interpretation"
    return candidates[0]


def _render(info, use_key=True):
    output = info["structural"].copy()
    successful, rejected, orphan, routes, packets = [], [], [], set(), set()
    for pi, path in enumerate(info["paths"]):
        cargo = info["beads"] & set(path)
        if not cargo:
            continue
        docks = [f for f in info["docks"] if f["path"] == pi]
        matches = [f for f in docks if f["span"] == len(cargo)]
        if not use_key and docks:
            # Deliberate shortcut: choose the dock nearest the group, ignoring width.
            matches = [min(docks, key=lambda f: sum(abs(path.index(p) - path.index(f["contact"]))
                                                   for p in cargo))]
        if not matches:
            (rejected if docks else orphan).append(pi)
            continue
        assert len(matches) == 1
        dock = matches[0]
        ordered = path if path[0] == dock["contact"] else path[::-1]
        last = max(ordered.index(p) for p in cargo)
        route = set(ordered[:last + 1])
        assert routes.isdisjoint(route) and packets.isdisjoint(dock["face"])
        _paint(output, route, info["structure"])
        _paint(output, dock["face"], info["bead"])
        routes |= route
        packets |= dock["face"]
        successful.append(dict(path=pi, count=len(cargo), dock=dock, route=route,
                               untraveled=len(path) - last - 1))
    return output, dict(successful=successful, rejected=rejected, orphan=orphan,
                        routes=routes, packets=packets)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles and destinations using this input alone."""
    return _render(_analyze(input_grid))[0]


def _core(rng, lanes, width, fallback=False):
    """Create a whole switch network; no parent generators are called."""
    rows = []
    for i in range(lanes - 1):
        if fallback:
            rows.append([3 if i % 2 == 0 else width - 4])
            continue
        previous = rows[-1] if rows else []
        wanted = int(rng.choice([0, 1, 1, 1, 2, 2, 3]))
        chosen = []
        for c in map(int, rng.permutation(np.arange(2, width - 2))):
            if len(chosen) >= wanted:
                break
            if all(abs(c - old) >= 4 for old in chosen) and all(abs(c - old) >= 2 for old in previous):
                chosen.append(c)
        rows.append(sorted(chosen))
    centers = [(2 * i + 1, c) for i, row in enumerate(rows) for c in row]
    if not centers:
        raise AssertionError("no switches")
    grid = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    grid[::2] = 1
    for r, c in centers:
        grid[r, c - 1:c + 2] = 3
    # Optional cuts open cycles and add undocked terminals. Unlike gravity in A,
    # attraction to a dock also permits paths that turn back along the lane axis.
    if not fallback:
        possible = [(r, c) for r in range(0, grid.shape[0], 2) for c in range(2, width - 2)
                    if all(abs(c - x) >= 2 for y, x in centers if abs(y - r) == 1)]
        rng.shuffle(possible)
        for r, c in possible[:int(rng.integers(0, 4))]:
            if all(grid[r, cc] == 1 for cc in range(c - 1, c + 2)):
                grid[r, c] = 0
    structural, _, _ = _rewire(grid, 0, 1, 2, 3)
    paths = _paths(_cells(structural == 1))
    return grid, structural, paths, centers


def _make_dock(rng, endpoint, span, shape, mode=None):
    """Place a receiver outside the left/right side of the rail core."""
    r, c = endpoint
    inward = 1 if c == 6 else -1
    center = int(rng.integers(2, 4) if mode == "long_center" else rng.integers(1, 4))
    guard, gap = ((1, 1), (1, 2), (2, 1))[int(rng.integers(3))]
    q0 = guard + gap
    width = span + 2 * q0
    offset = (span - 1) // 2 + (int(rng.integers(2)) if span == 2 else 0)
    top = r - offset - q0
    base = c - inward * (center + 2)
    left, right = map(int, rng.integers(1, center + 2, size=2))
    if mode == "long_center":
        left, right = map(int, rng.integers(1, center, size=2))
    elif mode == "unequal" and left == right:
        right = left % (center + 1) + 1
    structure = {(top + q, base) for q in range(width)}
    for a, b, length in ((0, guard, left), (q0, q0 + span, center),
                          (width - guard, width, right)):
        structure |= {(top + q, base + inward * k)
                      for q in range(a, b) for k in range(1, length + 1)}
    packet = {(top + q, c - inward) for q in range(q0, q0 + span)}
    assert all(0 < rr < shape[0] - 1 and 0 < cc < shape[1] - 1
               for rr, cc in structure | packet)
    return dict(cells=structure, face=packet, contact=endpoint, span=span)


def _beads_for(rng, path, eligible, count, far_from=None, existing=frozenset()):
    choices = [p for p in path if p in eligible]
    if far_from is not None:
        ordered = path if path[0] == far_from else path[::-1]
        # Prefer the half closer to the competing, wrong-size dock. Leave a
        # nontraversed suffix so coloring the whole path is visibly incorrect.
        preferred = [p for p in choices if .52 * len(path) <= ordered.index(p) < len(path) - 2]
        rest = [p for p in choices if p not in preferred]
        rng.shuffle(preferred)
        rng.shuffle(rest)
        choices = preferred + rest
    else:
        rng.shuffle(choices)
    selected = set()
    blocked = set(existing)
    for p in choices:
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for q in blocked | selected):
            selected.add(p)
            if len(selected) == count:
                return selected
    raise AssertionError("not enough separated bead positions")


def _construct_pair(rng, is_test, witness, rotation):
    for attempt in range(160):
        fallback = attempt >= 120
        lanes = int(rng.integers(8, 11) if is_test else rng.integers(5, 7))
        width = int(rng.integers(11, 18) if is_test else rng.integers(11, 16))
        if fallback:
            lanes, width = (9, 15) if is_test else (6, 15)
        try:
            core, rewired, local_paths, centers = _core(rng, lanes, width, fallback)
            height = core.shape[0]
            # Margins accommodate forks of different widths and positive lengths.
            shape = height + 10, width + 12
            grid = np.zeros(shape, dtype=np.int8)
            grid[5:5 + height, 6:6 + width] = core
            structural = grid.copy()
            structural[5:5 + height, 6:6 + width] = rewired
            paths = [[(r + 5, c + 6) for r, c in p] for p in local_paths]
            candidates = [i for i, path in enumerate(paths)
                          if all(p[1] in (6, width + 5) for p in (path[0], path[-1]))
                          and max(r for r, _ in path) - min(r for r, _ in path) >= (12 if is_test else 4)
                          and len(path) >= 19]
            rng.shuffle(candidates)
            assert candidates
            eligible = _cells(grid == 1) & _cells(structural == 1)
            built = None
            for main in candidates:
                path = paths[main]
                spans = [1, 3]
                rng.shuffle(spans)
                docks = []
                for endpoint, span in zip((path[0], path[-1]), spans):
                    mode = "long_center" if witness == span == 1 else "unequal" if witness == span == 3 else None
                    dock = _make_dock(rng, endpoint, span, shape, mode)
                    assert all(dock["cells"].isdisjoint(_near(f["cells"] | f["face"]))
                               and dock["face"].isdisjoint(_near(f["cells"] | f["face"])) for f in docks)
                    dock["path"] = main
                    docks.append(dock)
                other_options = [(i, p) for i, ps in enumerate(paths) if i != main
                                 for p in (ps[0], ps[-1]) if p[1] in (6, width + 5)]
                rng.shuffle(other_options)
                for pi, p in other_options:
                    dock = _make_dock(rng, p, 2, shape)
                    if any(not (dock["cells"] | dock["face"]).isdisjoint(_near(f["cells"] | f["face"]))
                           for f in docks):
                        continue
                    dock["path"] = pi
                    docks.append(dock)
                    break
                if len(docks) < 3:
                    continue
                # A fourth dock is optional and belongs to a further component.
                if rng.random() < (.85 if is_test else .55):
                    for pi, p in other_options:
                        if pi in {f["path"] for f in docks}:
                            continue
                        dock = _make_dock(rng, p, int(rng.integers(1, 4)), shape)
                        if any(not (dock["cells"] | dock["face"]).isdisjoint(_near(f["cells"] | f["face"]))
                               for f in docks):
                            continue
                        dock["path"] = pi
                        docks.append(dock)
                        break
                orphan_options = [i for i in range(len(paths)) if i not in {f["path"] for f in docks}
                                  and any(p in eligible for p in paths[i])]
                if not orphan_options:
                    continue
                built = main, docks, orphan_options
                break
            assert built is not None
            main, docks, orphan_options = built
            main_count = witness if witness in (1, 3) else int(rng.choice([1, 3]))
            chosen = next(f for f in docks[:2] if f["span"] == main_count)
            main_beads = _beads_for(rng, paths[main], eligible, main_count, chosen["contact"])
            # Reject accidental agreement with the distance-only alternative.
            wrong = next(f for f in docks[:2] if f is not chosen)
            distance = lambda f: sum(abs(paths[main].index(p) - paths[main].index(f["contact"])) for p in main_beads)
            assert distance(wrong) < distance(chosen)
            beads = set(main_beads)
            second_active = witness != 1
            second_count = 2 if second_active else 1
            beads |= _beads_for(rng, paths[docks[2]["path"]], eligible, second_count, existing=beads)
            if len(docks) == 4:
                # Ensure a rejected dock remains in the two-success witness.
                count = 1 if docks[3]["span"] != 1 else 2
                beads |= _beads_for(rng, paths[docks[3]["path"]], eligible, count, existing=beads)
            orphan = int(rng.choice(orphan_options))
            beads |= _beads_for(rng, paths[orphan], eligible, 1, existing=beads)
            for dock in docks:
                _paint(grid, dock["cells"], 4)
                _paint(structural, dock["cells"], 4)
            _paint(grid, beads, 2)
            # Labels use construction geometry, not solve() or its role parser.
            output = structural.copy()
            for pi, path in enumerate(paths):
                cargo = beads & set(path)
                matches = [f for f in docks if f["path"] == pi and f["span"] == len(cargo)]
                if not cargo or not matches:
                    continue
                assert len(matches) == 1
                dock = matches[0]
                order = path if path[0] == dock["contact"] else path[::-1]
                farthest = max(order.index(p) for p in cargo)
                _paint(output, set(order[:farthest + 1]), 4)
                _paint(output, dock["face"], 2)
            # Every pair can have a completely new five-role palette.
            palette = np.asarray(rng.choice(10, 5, replace=False), dtype=np.int8)
            grid, output = palette[grid], palette[output]
            if rng.integers(2):
                grid, output = np.fliplr(grid), np.fliplr(output)
            grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
            info = _analyze(grid)
            predicted, detail = _render(info)
            assert np.array_equal(output, predicted)
            assert _render(info, use_key=False)[1]["packets"] != detail["packets"]
            assert _render(_analyze(grid, turn=False))[1]["packets"] != detail["packets"]
            assert any(s["untraveled"] >= 2 for s in detail["successful"])
            assert detail["orphan"]
            return {"input": grid, "output": output}
        except AssertionError:
            continue
    raise RuntimeError("bounded construction exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    witness_positions = list(map(int, rng.permutation(num_train)[:2]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = (1 if i == witness_positions[0] else 3 if i == witness_positions[1] else 0) if split == "train" else 0
            pair_rotation = (rotation + i % 2) % 4 if split == "train" else (rotation + 2 + i % 2) % 4
            for _ in range(20):
                pair = _construct_pair(rng, split == "test", witness, pair_rotation)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_counts = set(), set()
    training_reject = training_multiple = training_long_center = training_unequal = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (np.asarray(pair[k]) for k in ("input", "output"))
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = inp.shape, inp.astype(np.int8).tobytes()
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            expected, details = _render(info)
            assert np.array_equal(out, expected) and not np.array_equal(inp, out)
            assert 3 <= len(info["docks"]) <= 4
            assert {f["span"] for f in info["docks"]} == {1, 2, 3}
            assert details["successful"] and details["orphan"]
            # Exact bead conservation applies to accepted groups; rejection is explicit.
            assert np.count_nonzero(out == info["bead"]) == sum(s["count"] for s in details["successful"])
            assert all(out[p] == info["structure"] for f in info["docks"] for p in f["cells"])
            assert np.count_nonzero(inp == info["bar"]) == np.count_nonzero(out == info["bar"])
            assert _render(info, use_key=False)[1]["packets"] != details["packets"], "width key is inessential"
            assert _render(_analyze(inp, turn=False))[1]["packets"] != details["packets"], "rewiring is inessential"
            assert any(s["untraveled"] >= 2 for s in details["successful"])
            competing = [pi for pi in range(len(info["paths"]))
                         if {f["span"] for f in info["docks"] if f["path"] == pi} == {1, 3}]
            assert len(competing) == 1
            main_path = info["paths"][competing[0]]
            axis = 0 if info["docks"][0]["out"][0] else 1
            cross = 1 - axis
            reach = max(p[cross] for p in main_path) - min(p[cross] for p in main_path)
            assert (reach <= 10 if split == "train" else reach >= 12), "missing routing shift"
            if split == "train":
                train_counts.update(s["count"] for s in details["successful"])
                training_reject |= bool(details["rejected"])
                training_multiple |= len(details["successful"]) >= 2
                for success in details["successful"]:
                    a, b, c = (run[2] for run in success["dock"]["runs"])
                    training_long_center |= b > max(a, c)
                    training_unequal |= a != c
    assert train_counts == {1, 2, 3}, "show all three count-to-width matches"
    assert training_reject and training_multiple, "show mismatch and independent grouping"
    assert training_long_center and training_unequal, "show the central tip determines insertion"


def _puzzle_key(puzzle):
    return tuple((s, p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded semantic/variation checks on 42 seeds plus API and corruption cases."""
    shapes, layouts, palettes, bar_counts, bead_counts, forks, signatures = set(), set(), set(), set(), set(), set(), set()
    overlaps = breaks = turned_back = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                pal = tuple(info[k] for k in ("bg", "rail", "bead", "bar", "structure"))
                palettes.add(pal)
                normalized = np.zeros(pair["input"].shape, dtype=np.int8)
                for i, color in enumerate(pal):
                    normalized[pair["input"] == color] = i
                layouts.add((normalized.shape, normalized.tobytes()))
                bar_counts.add(len(info["centers"]))
                bead_counts.add(len(info["beads"]))
                overlaps += bool(info["overlaps"])
                for f in info["docks"]:
                    forks.add((f["width"], tuple(f["runs"])))
                # Internal degree-one rail endpoints evidence cuts, regardless of rotation.
                contacts = {f["contact"] for f in info["docks"]}
                axis = 0 if info["docks"][0]["out"][0] else 1
                coords = [p[axis] for path in info["paths"] for p in path]
                breaks += any(p[axis] not in (min(coords), max(coords))
                              for path in info["paths"] for p in (path[0], path[-1]) if p not in contacts)
                cross = 1 - axis
                turned_back += any(len({int(np.sign(b[cross] - a[cross])) for a, b in zip(path, path[1:])
                                        if b[cross] != a[cross]}) == 2 for path in info["paths"])
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(layouts) >= 190 and len(palettes) >= 150
    assert len(bar_counts) >= 8 and len(bead_counts) >= 4 and len(forks) >= 60
    assert overlaps > 0 and breaks > 0 and turned_back > 0
    for role in range(5):
        assert {p[role] for p in palettes} == set(range(10))
    a = generate_puzzle(314159, 2, 1)
    assert _puzzle_key(a) == _puzzle_key(generate_puzzle(314159, 2, 1))
    # Geometric symmetries and an arbitrary color permutation preserve the rule.
    pair = a["test"][0]
    recolor = np.array([6, 4, 7, 9, 0, 2, 8, 1, 5, 3], dtype=np.int8)
    for rotation in range(4):
        inp = recolor[np.rot90(np.fliplr(pair["input"]), rotation)]
        expected = recolor[np.rot90(np.fliplr(pair["output"]), rotation)]
        assert np.array_equal(solve(inp), expected)
    for nt, nq in ((10, 4), (10, 1), (5, 4)):
        validate_generated_puzzle(generate_puzzle(700 + nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    info = _analyze(corrupted["test"][0]["input"])
    successful = _render(info)[1]["successful"][0]
    p = next(iter(successful["dock"]["face"]))
    corrupted["test"][0]["output"][p] = info["bg"]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("cargo corruption accepted")
    malformed = {(0, q) for q in range(7)} | {(k, q) for k in (1, 2) for q in (0, 6)}
    try:
        _fork(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError("two-pronged receiver accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), normalized_layouts=len(layouts),
                palettes=len(palettes), switch_counts=sorted(bar_counts), bead_counts=sorted(bead_counts),
                fork_geometries=len(forks), overlapping_pairs=overlaps,
                cut_pairs=breaks, returning_path_pairs=turned_back)
