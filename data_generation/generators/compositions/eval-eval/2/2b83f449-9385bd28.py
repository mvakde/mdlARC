"""Kinetic stencil stacking: switched rails determine rectangle occlusion.

No parent code, files, seeds, or demonstrations are consulted by solve().
The four infrastructure colors are fixed; all stencil mappings are visible.
"""
from itertools import combinations

import numpy as np


BG, RAIL, SWITCH, TURNED = 0, 8, 7, 6
CUES = (1, 2, 3, 4, 5, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 898408797, 302673580, 876875567, 995665375, 300843909,
              267706042, 622039728, 789660409, 170214147, 67943125,
              994417010, 24447974, 959652222, 934579124, 246209924,
              223183384, 946914195, 888035744, 840318741, 240899220,
              462633639, 867137163, 888890240, 171186274, 845477623,
              849444846, 125220765, 6527160, 777687712, 761319503,
              204439371, 991410871, 658121653, 189790486, 556451515,
              360237655, 630643682, 614194116, 919935205, 39001641,
              292121695)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _inside(cell, bounds):
    r, c = cell
    r0, c0, r1, c1 = bounds
    return r0 <= r <= r1 and c0 <= c <= c1


def _intersects(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _corners(bounds, diagonal):
    r0, c0, r1, c1 = bounds
    if diagonal == 0:
        return {(r0, c0), (r0, c0+1), (r0+1, c0),
                (r1, c1), (r1, c1-1), (r1-1, c1)}
    return {(r0, c1), (r0, c1-1), (r0+1, c1),
            (r1, c0), (r1, c0+1), (r1-1, c0)}


def _recognize(cells):
    if len(cells) != 6:
        return None
    rows, cols = zip(*cells)
    b = min(rows), min(cols), max(rows), max(cols)
    h, w = b[2]-b[0]+1, b[3]-b[1]+1
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        return None
    for d in (0, 1):
        if cells == _corners(b, d):
            return b, d
    return None


def _neighbors(cell, cells):
    r, c = cell
    return [(r+dr, c+dc) for dr, dc in DIRS if (r+dr, c+dc) in cells]


def _paths(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        component, todo = set(), [start]
        while todo:
            q = todo.pop()
            if q in component:
                continue
            component.add(q)
            todo.extend(_neighbors(q, cells))
        remaining.difference_update(component)
        endpoints = sorted(q for q in component if len(_neighbors(q, cells)) == 1)
        if len(endpoints) != 2 or any(len(_neighbors(q, cells)) not in (1, 2) for q in component):
            raise ValueError("rails must form simple paths, without branches or cycles")
        path, previous, current = [endpoints[0]], None, endpoints[0]
        while current != endpoints[1]:
            choices = [q for q in _neighbors(current, cells) if q != previous]
            if len(choices) != 1:
                raise ValueError("ambiguous path")
            previous, current = current, choices[0]
            path.append(current)
        if len(path) != len(component):
            raise ValueError("untraced rail cells")
        steps = {b[0]-a[0] for a, b in zip(path, path[1:]) if b[0] != a[0]}
        if len(steps) > 1:
            raise ValueError("internal gravity traps are outside the family")
        result.append(path)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _network(scene, token_cells, turn=True):
    """Simultaneous local writes, then global connectivity validation."""
    h, w = scene.shape
    raw = {tuple(map(int, q)) for q in np.argwhere(scene == RAIL)} | set(token_cells)
    if any((r+1, c) in raw for r, c in raw):
        raise ValueError("input rails must be horizontal and separated")
    bars = {tuple(map(int, q)) for q in np.argwhere(scene == SWITCH)}
    remaining, centers, writes, overlaps = set(bars), [], {}, 0
    while remaining:
        start = min(remaining)
        block, todo = set(), [start]
        while todo:
            q = todo.pop()
            if q in block:
                continue
            block.add(q)
            todo.extend(_neighbors(q, bars))
        remaining.difference_update(block)
        r, first = min(block)
        if block != {(r, first+i) for i in range(3)}:
            raise ValueError("switches must be isolated horizontal three-cell bars")
        c = first+1
        if not (0 < r < h-1 and 0 < c < w-1):
            raise ValueError("switch neighborhood leaves the grid")
        if any((rr, cc) not in raw for rr in (r-1, r+1) for cc in range(c-1, c+2)):
            raise ValueError("switch must sit between intact rails")
        if (r-1, c) in token_cells or (r+1, c) in token_cells:
            raise ValueError("a switch cannot cut through a bead")
        for rr in range(r-1, r+2):
            for cc in range(c-1, c+2):
                val = TURNED if cc == c else RAIL
                if (rr, cc) in writes:
                    if writes[rr, cc] != val:
                        raise ValueError("conflicting switch neighborhoods")
                    overlaps += 1
                writes[rr, cc] = val
        centers.append((r, c))
    if not centers:
        raise ValueError("at least one switch is required")
    rail = np.zeros(scene.shape, dtype=np.int8)
    for q in raw:
        rail[q] = RAIL
    if turn:
        for q, val in writes.items():
            rail[q] = val
    paths = _paths({tuple(map(int, q)) for q in np.argwhere(rail == RAIL)})
    endpoints = {q for p in paths for q in (p[0], p[-1])}
    if not set(token_cells) <= endpoints:
        raise ValueError("beads must start at exposed terminals")
    return dict(rail=rail, paths=paths, centers=sorted(centers), overlaps=overlaps,
                raw=raw, endpoints=endpoints)


def _settle(net, source):
    """Color-preserving queues: resident low bead first, incoming bead second."""
    by_cell = {q: cue for cue, q in source.items()}
    positions, depths = {}, {}
    collisions = singles = levels = 0
    for p in net["paths"]:
        ends = (p[0], p[-1])
        occupied = [q for q in ends if q in by_cell]
        if not occupied:
            continue
        if ends[0][0] == ends[1][0]:
            for q in occupied:
                positions[by_cell[q]], depths[by_cell[q]] = q, 0
            levels += len(occupied)
        else:
            low = max(ends, key=lambda q: q[0])
            ordered = p if p[0] == low else p[::-1]
            keys = ([by_cell[low]] if low in by_cell else [])
            keys += [by_cell[q] for q in occupied if q != low]
            for depth, cue in enumerate(keys):
                positions[cue], depths[cue] = ordered[depth], depth
            collisions += len(keys) == 2
            singles += len(keys) == 1 and positions[keys[0]] != source[keys[0]]
    if set(positions) != set(source) or len(set(positions.values())) != len(source):
        raise ValueError("beads were lost or duplicated")
    return dict(positions=positions, depths=depths, collisions=collisions,
                singles=singles, levels=levels)


def _parse(grid):
    """Identify the isolated table, two Ls and singleton key of each color."""
    h, w = grid.shape
    if np.any(grid == TURNED):
        raise ValueError("turned-switch color is output-only")
    candidates = []
    locations = {cue: {tuple(map(int, q)) for q in np.argwhere(grid == cue)} for cue in CUES}
    for c in range(w-1):
        rows = np.flatnonzero(np.isin(grid[:, c], CUES))
        runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1)+1)
        for run in runs:
            n = len(run)
            if not 2 <= n <= 6:
                continue
            r0, r1 = int(run[0]), int(run[-1])
            cues = list(map(int, grid[r0:r1+1, c]))
            fills = list(map(int, grid[r0:r1+1, c+1]))
            if len(set(cues)) != n or any(x not in (BG, *CUES) for x in fills):
                continue
            table = (r0, c, r1, c+1)
            if any(grid[rr, cc] != BG
                   for rr in range(max(0, r0-1), min(h, r1+2))
                   for cc in range(max(0, c-1), min(w, c+3))
                   if not _inside((rr, cc), table)):
                continue
            table_cells = {(rr, cc) for rr in range(r0, r1+1) for cc in (c, c+1)}
            objects, source = [], {}
            for rank, (cue, fill) in enumerate(zip(cues, fills)):
                cells = locations[cue] - table_cells
                if len(cells) != 7:
                    break
                choices = []
                for token in sorted(cells):
                    recognized = _recognize(cells-{token})
                    if recognized is not None:
                        b, d = recognized
                        rr, cc = token
                        adjacent_rails = sum(
                            0 <= rr+dr < h and 0 <= cc+dc < w
                            and grid[rr+dr, cc+dc] == RAIL for dr, dc in DIRS)
                        if not _intersects(b, table) and adjacent_rails == 1:
                            choices.append((token, b, d))
                if len(choices) != 1:
                    break
                token, b, d = choices[0]
                source[cue] = token
                objects.append(dict(cue=cue, fill=fill, rank=rank, bounds=b,
                                    markers=cells-{token}, diagonal=d))
            if len(objects) != n:
                continue
            scene = grid.copy()
            scene[r0:r1+1, c:c+2] = BG
            if set(map(int, np.unique(scene))) - {BG, RAIL, SWITCH, *cues}:
                continue
            try:
                net = _network(scene, set(source.values()))
                state = _settle(net, source)
            except ValueError:
                continue
            # Every nonmechanical colored input cell belongs to its stencil.
            markers = set().union(*(o["markers"] for o in objects))
            if markers & (net["raw"] | set(map(tuple, np.argwhere(net["rail"] != BG)))):
                continue
            candidates.append(dict(table=table, objects=objects, source=source,
                                   network=net, state=state, scene=scene))
    if len(candidates) != 1:
        raise ValueError("input must have a unique table/stencil/rail interpretation")
    return candidates[0]


def _order(parsed, variant="normal"):
    state, objects = parsed["state"], parsed["objects"]
    if variant == "static":
        return list(objects)
    if variant == "initial":
        return sorted(objects, key=lambda o: (-parsed["source"][o["cue"]][0], o["rank"]))
    if variant == "no_turn":
        net = _network(parsed["scene"], set(parsed["source"].values()), turn=False)
        state = _settle(net, parsed["source"])
    return sorted(objects, key=lambda o: (
        -state["positions"][o["cue"]][0],
        0 if variant == "no_depth" else state["depths"][o["cue"]],
        -o["rank"] if variant == "reverse_ties" else o["rank"]))


def _render(grid, parsed, variant="normal"):
    result = grid.copy()
    net, state = parsed["network"], parsed["state"]
    for q in parsed["source"].values():
        result[q] = RAIL
    # This is the canvas plane. Paint including old markers and backgrounds.
    if variant != "no_boxes":
        for obj in reversed(_order(parsed, variant)):
            r0, c0, r1, c1 = obj["bounds"]
            result[r0:r1+1, c0:c1+1] = obj["cue"] if variant == "no_mapping" else obj["fill"]
            for q in obj["markers"]:
                result[q] = obj["cue"]
    # Mechanical infrastructure lies above every stencil layer.
    mask = net["rail"] != BG
    result[mask] = net["rail"][mask]
    for cue, q in state["positions"].items():
        result[q] = cue
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _array(input_grid)
    return _render(grid, _parse(grid))


def _witnesses(grid, parsed):
    expected = _render(grid, parsed)
    mechanical = parsed["network"]["rail"] != BG
    canvas = ~mechanical
    found = set()
    for name in ("static", "initial", "no_turn", "no_depth", "reverse_ties", "no_boxes", "no_mapping"):
        if np.any((expected != _render(grid, parsed, name)) & canvas):
            found.add(name)
    state = parsed["state"]
    for name in ("collisions", "singles", "levels"):
        if state[name]:
            found.add(name)
    objects = _order(parsed)
    for i, upper in enumerate(objects):
        for lower in objects[i+1:]:
            for q in lower["markers"]:
                if (_inside(q, upper["bounds"]) and not mechanical[q]
                        and expected[q] != lower["cue"]):
                    found.add("occlusion")
            if upper["fill"] == BG and lower["fill"] != BG:
                a, b = upper["bounds"], lower["bounds"]
                for r in range(max(a[0], b[0]), min(a[2], b[2])+1):
                    for c in range(max(a[1], b[1]), min(a[3], b[3])+1):
                        q = r, c
                        if (not mechanical[q] and q not in upper["markers"]
                                and expected[q] == BG):
                            found.add("clearing")
    return found


def _build_core(rng, lanes, width):
    rows, centers = [], []
    for i in range(lanes-1):
        chosen = []
        previous = rows[-1] if rows else []
        wanted = int(rng.choice([0, 1, 1, 1, 2, 2, 3]))
        for c in rng.permutation(np.arange(2, width-2)):
            c = int(c)
            if len(chosen) >= wanted:
                break
            if all(abs(c-p) >= 4 for p in chosen) and all(abs(c-p) >= 2 for p in previous):
                chosen.append(c)
        rows.append(sorted(chosen))
        centers.extend((2*i+1, c) for c in chosen)
    if not centers:
        centers = [(1, int(rng.integers(2, width-2)))]
    breaks = []
    for r in range(0, 2*lanes-1, 2):
        ports = sorted((c, y) for y, c in centers if abs(y-r) == 1)
        for (a, side_a), (b, side_b) in zip(ports, ports[1:]):
            if side_a == side_b:
                breaks.append((r, int(rng.integers(a+2, b-1))))
    for _ in range(int(rng.integers(1, 6))):
        pool = [(r, c) for r in range(0, 2*lanes-1, 2) for c in range(2, width-2)
                if all(abs(c-x) >= 2 for y, x in centers if abs(y-r) == 1)
                and all(abs(c-x) >= 3 for y, x in breaks if y == r)]
        if not pool:
            break
        breaks.append(pool[int(rng.integers(len(pool)))])
    core = np.zeros((2*lanes-1, width), dtype=np.int8)
    core[::2, :] = RAIL
    for q in breaks:
        core[q] = BG
    for r, c in centers:
        core[r, c-1:c+2] = SWITCH
    return core


def _choose_keys(rng, net, lesson):
    vertical = [p for p in net["paths"] if p[0][0] != p[-1][0]]
    level = [p for p in net["paths"] if p[0][0] == p[-1][0]]
    if lesson == "queue":
        if len(vertical) < 2 or not level:
            return None
        chosen = list(rng.choice(len(vertical), 2, replace=False))
        double, single = vertical[int(chosen[0])], vertical[int(chosen[1])]
        keys = [double[0], double[-1], min((single[0], single[-1]), key=lambda q: q[0]),
                level[int(rng.integers(len(level)))][0]]
        return keys
    if lesson == "ties":
        pairs = [(a, b) for a, b in combinations(vertical, 2)
                 if max(a[0][0], a[-1][0]) == max(b[0][0], b[-1][0])]
        if not pairs:
            return None
        a, b = pairs[int(rng.integers(len(pairs)))]
        keys = [min((a[0], a[-1]), key=lambda q: q[0]),
                min((b[0], b[-1]), key=lambda q: q[0])]
        other = sorted(net["endpoints"] - {a[0], a[-1], b[0], b[-1]})
        if not other:
            return None
        count = int(rng.integers(1, min(4, len(other))+1))
        keys.extend(other[int(i)] for i in rng.choice(len(other), count, replace=False))
        return keys
    endpoints = sorted(net["endpoints"])
    count = int(rng.integers(2, min(6, len(endpoints))+1))
    return [endpoints[int(i)] for i in rng.choice(len(endpoints), count, replace=False)]


def _sample_bounds(rng, h, w):
    shape = int(rng.integers(5))
    if shape == 0:
        hh, ww = 2, int(rng.integers(4, w+1))
    elif shape == 1:
        hh, ww = int(rng.integers(4, h+1)), 2
    elif shape == 2:
        hh = ww = int(rng.integers(3, min(h, w, 11)+1))
    else:
        hh, ww = int(rng.integers(3, h+1)), int(rng.integers(3, w+1))
    r, c = int(rng.integers(h-hh+1)), int(rng.integers(w-ww+1))
    return r, c, r+hh-1, c+ww-1


def _candidate(rng, lesson, diagonal, test, clearing):
    h, w = int(rng.integers(16, 31)), int(rng.integers(16, 31))
    lanes = int(rng.integers(3, min(8, (h-5)//2)+1))
    width = int(rng.integers(9, min(23, w-4)+1))
    core = _build_core(rng, lanes, width)
    r0 = int(rng.integers(1, h-core.shape[0]))
    c0 = int(rng.integers(1, w-core.shape[1]))
    grid = np.zeros((h, w), dtype=np.int8)
    grid[r0:r0+core.shape[0], c0:c0+core.shape[1]] = core
    try:
        net = _network(grid, set())
    except ValueError:
        return None
    keys = _choose_keys(rng, net, lesson)
    if keys is None:
        return None
    n = len(keys)
    cues = list(map(int, rng.choice(CUES, n, replace=False)))
    fills = list(map(int, rng.choice((BG, *CUES), n, replace=rng.random() < .25)))
    if clearing:
        fills[int(rng.integers(n))] = BG
    diagonals = [diagonal]*n
    if test:
        diagonals = list(map(int, rng.integers(2, size=n)))
        diagonals[0], diagonals[1] = 0, 1
        rng.shuffle(diagonals)
    for cue, q in zip(cues, keys):
        grid[q] = cue
    # Choose an arbitrary locally isolated table, not an edge-only gutter.
    locations = [(r, c) for r in range(h-n+1) for c in range(w-1)
                 if not grid[max(0, r-1):min(h, r+n+1), max(0, c-1):min(w, c+3)].any()]
    if not locations:
        return None
    tr, tc = locations[int(rng.integers(len(locations)))]
    table = tr, tc, tr+n-1, tc+1
    ranks = list(map(int, rng.permutation(n)))
    for row, i in enumerate(ranks):
        grid[tr+row, tc:tc+2] = cues[i], fills[i]
    reserved = {(r, c) for r in range(max(0, tr-1), min(h, tr+n+1))
                for c in range(max(0, tc-1), min(w, tc+3))}
    used = {tuple(map(int, q)) for q in np.argwhere(grid != BG)} | reserved
    for i, cue in enumerate(cues):
        placed = False
        for _ in range(160):
            b = _sample_bounds(rng, h, w)
            markers = _corners(b, diagonals[i])
            if _intersects(b, table) or markers & used or markers & set(map(tuple, np.argwhere(net["rail"] != BG))):
                continue
            # A bead must remain a singleton of its color, apart from its Ls.
            if any(abs(r-keys[i][0])+abs(c-keys[i][1]) <= 1 for r, c in markers):
                continue
            for q in markers:
                grid[q] = cue
            used.update(markers)
            placed = True
            break
        if not placed:
            return None
    try:
        parsed = _parse(grid)
    except ValueError:
        return None
    witness = _witnesses(grid, parsed)
    required = {"no_boxes", "no_mapping"}
    if lesson == "queue":
        required |= {"no_depth", "static", "collisions", "singles", "levels", "occlusion"}
    if lesson == "ties":
        required |= {"reverse_ties", "initial", "no_turn", "singles"}
    if clearing:
        required.add("clearing")
    if not required <= witness:
        return None
    return dict(input=grid, output=_render(grid, parsed))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be 1..4 and less than num_train")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    slots = list(map(int, rng.permutation(num_train)))
    use_clearing = bool(rng.random() < .55)
    lessons = {slots[0]: ("queue", 0), slots[1]: ("ties", 1)}
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson, diagonal = lessons.get(i, ("free", int(rng.integers(2)))) if split == "train" else ("free", 0)
            for _ in range(600):
                pair = _candidate(rng, lesson, diagonal, split == "test",
                                  use_clearing and split == "train" and i == slots[0])
                if pair is not None and _signature(pair["input"]) not in seen:
                    # Clearing is tested only if demonstrated, although literal
                    # background swatches have a fixed input-visible meaning.
                    if split == "test" and not use_clearing:
                        if "clearing" in _witnesses(pair["input"], _parse(pair["input"])):
                            continue
                    seen.add(_signature(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded sampling did not find a readable witnessed example")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, witnesses, train_diagonals = set(), set(), set()
    tests = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            parsed = _parse(inp)
            assert np.array_equal(out, _render(inp, parsed))
            net, state = parsed["network"], parsed["state"]
            # Every rail/switch is conserved according to simultaneous writes;
            # keys stay color-distinct and each occurs once on the final rails.
            for q in map(tuple, np.argwhere(net["rail"] != BG)):
                expected = next((cue for cue, p in state["positions"].items() if p == q), int(net["rail"][q]))
                assert out[q] == expected
            for cue in parsed["source"]:
                assert sum(out[q] == cue for q in map(tuple, np.argwhere(net["rail"] == RAIL))) == 1
            t = parsed["table"]
            assert np.array_equal(inp[t[0]:t[2]+1, t[1]:t[3]+1], out[t[0]:t[2]+1, t[1]:t[3]+1])
            w = _witnesses(inp, parsed)
            diagonals = {o["diagonal"] for o in parsed["objects"]}
            if split == "train":
                witnesses |= w
                assert len(diagonals) == 1, "generated training scenes teach one diagonal at a time"
                train_diagonals |= diagonals
            else:
                assert diagonals == {0, 1}, "generated tests recombine both taught diagonals"
                tests.append(w)
    assert {"static", "initial", "no_turn", "no_depth", "reverse_ties", "no_boxes",
            "no_mapping", "collisions", "singles", "levels", "occlusion"} <= witnesses
    assert train_diagonals == {0, 1}
    if any("clearing" in w for w in tests):
        assert "clearing" in witnesses


def self_test():
    """Bounded 42-seed semantic/variation checks; no files or parent imports."""
    # Hand-computed identity-preserving collision and single-bead motion.
    scene = np.zeros((3, 9), dtype=np.int8)
    scene[::2] = RAIL
    scene[1, 3:6] = SWITCH
    net = _network(scene, {(0, 0), (2, 0), (0, 8)})
    state = _settle(net, {1: (0, 0), 2: (2, 0), 3: (0, 8)})
    assert state["positions"] == {1: (2, 1), 2: (2, 0), 3: (2, 8)}
    assert state["depths"] == {1: 1, 2: 0, 3: 0}
    fake = dict(state=state, objects=[dict(cue=i, rank=i-1) for i in (1, 2, 3)])
    assert [o["cue"] for o in _order(fake)] == [2, 3, 1]
    assert [o["cue"] for o in _order(fake, "reverse_ties")] == [3, 2, 1]
    level = _settle(dict(paths=[[(0, i) for i in range(4)]]), {1: (0, 0), 2: (0, 3)})
    assert level["positions"] == {1: (0, 0), 2: (0, 3)}
    assert _recognize(_corners((0, 0, 1, 3), 0)) is not None
    assert _recognize(_corners((0, 0, 3, 1), 1)) is not None
    assert _recognize({(r, c) for r in range(2) for c in range(3)}) is None
    # A closed loop and a branch are invalid even when they have no beads.
    for cells in ({(0, 0), (0, 1), (1, 0), (1, 1)},
                  {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}):
        try:
            _paths(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rail topology accepted")
    signatures, shapes, layouts, counts, tables = set(), set(), set(), set(), set()
    ranges = {"switches": set(), "paths": set(), "overlaps": set(), "thin": set()}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                parsed = _parse(inp)
                assert np.array_equal(solve(inp.tolist()), pair["output"])
                shapes.add(inp.shape)
                layouts.add((inp.shape, parsed["network"]["rail"].tobytes(),
                             tuple(o["bounds"] for o in parsed["objects"])))
                counts.add(len(parsed["objects"]))
                tables.add(parsed["table"][:2])
                ranges["switches"].add(len(parsed["network"]["centers"]))
                ranges["paths"].add(len(parsed["network"]["paths"]))
                ranges["overlaps"].add(parsed["network"]["overlaps"])
                ranges["thin"].add(sum(min(o["bounds"][2]-o["bounds"][0]+1,
                                              o["bounds"][3]-o["bounds"][1]+1) == 2 for o in parsed["objects"]))
    assert len(signatures) == 42 and len(layouts) >= 150 and len(shapes) >= 50
    assert {2, 3, 4, 5, 6} <= counts and len(tables) >= 40
    assert len(ranges["switches"]) >= 6 and len(ranges["paths"]) >= 6
    assert 0 in ranges["overlaps"] and max(ranges["overlaps"]) > 0
    assert 0 in ranges["thin"] and max(ranges["thin"]) > 0
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
