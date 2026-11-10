"""Repair periodic framed motifs, then stitch their checkpoint fragments.

Only the standard library and NumPy are needed. No files, examples, random
state, construction metadata, or parent modules are consulted by solve().
Searches and self-tests are in memory; no execution traces are saved.
"""
from collections import deque
from functools import lru_cache
from heapq import heappop, heappush
import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _neighbors(height, width):
    return [tuple(rr * width + cc
                  for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                  if 0 <= rr < height and 0 <= cc < width)
            for r in range(height) for c in range(width)]


def _fragments(grid, color, neighbors):
    remaining = set(map(int, np.flatnonzero(grid.ravel() == color)))
    fragments = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], {start}
        for cell in todo:
            for other in neighbors[cell]:
                if other in remaining:
                    remaining.remove(other)
                    part.add(other)
                    todo.append(other)
        ends = sorted(p for p in part if len(part.intersection(neighbors[p])) <= 1)
        if len(part) > 1 and (len(ends) != 2 or any(
                len(part.intersection(neighbors[p])) > 2 for p in part)):
            raise ValueError("checkpoint fragment is branched or cyclic")
        ordered = [ends[0]]
        visited = {ends[0]}
        while len(ordered) < len(part):
            choices = [p for p in neighbors[ordered[-1]] if p in part - visited]
            if len(choices) != 1:
                raise ValueError("checkpoint fragment is not a path")
            ordered.append(choices[0])
            visited.add(choices[0])
        fragments.append(tuple(ordered))
    return fragments


def _direction(a, b, width):
    return b // width - a // width, b % width - a % width


def _bends(path, width):
    directions = [_direction(a, b, width) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _geometric_bends(a, direction, b, entry, width):
    """Obstacle-free lower bound, including turns at both fragment ends.

    Straight and L routes suffice for a lower bound: any longer route needs
    at least as many direction changes; a reversal is counted as just one.
    """
    dr, dc = b // width - a // width, b % width - a % width
    vertical = (1 if dr > 0 else -1, 0)
    horizontal = (0, 1 if dc > 0 else -1)
    orders = ((vertical, horizontal), (horizontal, vertical)) if dr and dc else (
        (vertical,) if dr else (horizontal,) if dc else (),)
    result = []
    for middle in orders:
        ds = ((direction,) if direction is not None else ()) + middle
        ds += (entry,) if entry is not None else ()
        result.append(sum(x != y for x, y in zip(ds, ds[1:])))
    return min(result)


def _route(grid, color, background, *, bend_first=False, max_states=None,
           use_heuristic=True):
    """Exact search over induced simple paths, modulo traversal reversal.

    Costs are (new background cells, bends), in that order. The optional
    reversed objective is diagnostic only. An MST over relaxed connections
    and unavoidable fragment bends gives admissible A* lower bounds. No
    checkpoint order, endpoint order, alternation, or corner is prescribed.
    max_states is used only to reject costly generation proposals.
    """
    height, width = grid.shape
    neighbors = _neighbors(height, width)
    neighbor_masks = [sum(1 << p for p in ns) for ns in neighbors]
    fragments = _fragments(grid, color, neighbors)
    if not fragments:
        raise ValueError("route has no checkpoints")
    intrinsic = [_bends(p, width) for p in fragments]
    if len(fragments) == 1:
        return {"path": fragments[0], "cost": (0, intrinsic[0]), "states": 0}
    count, n = len(fragments), grid.size
    all_mask = (1 << count) - 1
    owners, entries = {}, {}
    for i, frag in enumerate(fragments):
        for p in (frag, frag[::-1]):
            owners[p[0]] = i
            entries[p[0]] = _direction(p[0], p[1], width) if len(p) > 1 else None
    endpoints = tuple(owners)
    values = grid.ravel()
    distances = {}
    for end in endpoints:
        ds = [10000] * n
        ds[end] = 0
        queue = deque([end])
        while queue:
            current = queue.popleft()
            for other in neighbors[current]:
                if ds[other] != 10000:
                    continue
                if values[other] == background:
                    ds[other] = ds[current] + 1
                    queue.append(other)
                elif other in owners:
                    ds[other] = min(ds[other], ds[current] + 1)
        distances[end] = ds
    length_weights, bend_weights = [], []
    for frag in fragments:
        length_row, bend_row = [], []
        for other in fragments:
            length_row.append(max(0, min(distances[a][b] - 1
                              for a in (frag[0], frag[-1])
                              for b in (other[0], other[-1]))))
            bend_row.append(min(_geometric_bends(
                a, tuple(-x for x in entries[a]) if entries[a] else None,
                b, entries[b], width)
                for a in (frag[0], frag[-1]) for b in (other[0], other[-1])))
        length_weights.append(length_row)
        bend_weights.append(bend_row)

    @lru_cache(None)
    def mst(mask, bend):
        nodes = [i for i in range(count) if mask >> i & 1]
        if len(nodes) < 2:
            return 0
        weights = bend_weights if bend else length_weights
        reached, cost = {nodes[0]}, 0
        while len(reached) < len(nodes):
            value, node = min((weights[a][b], b)
                              for a in reached for b in nodes if b not in reached)
            cost += value
            reached.add(node)
        return cost

    queue, serial = [], 0

    def push(path, used, mask, added, bends, direction):
        nonlocal serial
        rem = all_mask ^ mask
        length_lb = bend_lb = 0
        if rem and use_heuristic:
            possible = [e for e in endpoints if rem >> owners[e] & 1]
            length_lb = max(0, min(distances[e][path[-1]] - 1 for e in possible))
            length_lb += mst(rem, False)
            bend_lb = min(_geometric_bends(path[-1], direction, e, entries[e], width)
                          for e in possible)
            bend_lb += mst(rem, True) + sum(intrinsic[i]
                                          for i in range(count) if rem >> i & 1)
        key = (bends + bend_lb, added + length_lb) if bend_first else (
            added + length_lb, bends + bend_lb)
        heappush(queue, (*key, serial, added, bends, mask, path, used, direction))
        serial += 1

    for i, frag in enumerate(fragments):
        for p in dict.fromkeys((frag, frag[::-1])):
            direction = _direction(p[-2], p[-1], width) if len(p) > 1 else None
            push(p, sum(1 << a for a in p), 1 << i, 0, intrinsic[i], direction)
    best, paths, expanded = None, set(), 0
    while queue:
        k1, k2, _, added, bends, mask, path, used, direction = heappop(queue)
        if best is not None and (k1, k2) > best:
            break
        if mask == all_mask:
            best = (k1, k2)
            paths.add(min(path, path[::-1]))
            if len(paths) > 1:
                return {"path": None, "cost": (added, bends), "states": expanded}
            continue
        expanded += 1
        if max_states is not None and expanded > max_states:
            raise RuntimeError("generation route-search budget exceeded")
        for point in neighbors[path[-1]]:
            if used >> point & 1:
                continue
            if values[point] == background:
                cells, new_mask = (point,), mask
            elif point in owners and not (mask >> owners[point] & 1):
                frag = fragments[owners[point]]
                cells = frag if frag[0] == point else frag[::-1]
                new_mask = mask | (1 << owners[point])
            else:
                continue
            new_path, new_used = path, used
            new_added, new_bends, new_dir = added, bends, direction
            for cell in cells:
                # A newly painted cell may touch only the preceding path cell.
                if (neighbor_masks[cell] & new_used).bit_count() != 1:
                    break
                d = _direction(new_path[-1], cell, width)
                new_bends += int(new_dir is not None and new_dir != d)
                new_dir = d
                new_added += int(values[cell] == background)
                new_used |= 1 << cell
                new_path += (cell,)
            else:
                push(new_path, new_used, new_mask, new_added, new_bends, new_dir)
    if not paths:
        raise ValueError("no feasible nonbranching completion")
    path = next(iter(paths))
    return {"path": path,
            "cost": (len(path) - sum(map(len, fragments)), _bends(path, width)),
            "states": expanded}


def _recover(motif, structure, accent):
    """All supported strict-majority periods must agree on the repaired cells."""
    cross, length = motif.shape
    candidates = []
    for period in range(1, length // 3 + 1):
        template = np.empty((cross, period), dtype=np.int8)
        valid = True
        for phase in range(period):
            group = motif[:, phase::period]
            votes = np.count_nonzero(group == accent, axis=1)
            if np.any(2 * votes == group.shape[1]):
                valid = False
                break
            template[:, phase] = np.where(2 * votes > group.shape[1], accent, structure)
        if not valid or len(np.unique(template)) != 2:
            continue
        repaired = template[:, np.arange(length) % period]
        errors = int(np.count_nonzero(repaired != motif))
        if errors <= max(1, motif.size // 8):
            candidates.append((period, repaired, errors))
    if not candidates or any(not np.array_equal(candidates[0][1], c[1])
                             for c in candidates):
        raise ValueError("periodic repair is missing or ambiguous")
    return candidates[0][1], candidates[0][0], candidates[0][2]


def _parse(grid):
    """Normalize parallel compartments; locate motifs from interior structure."""
    structure = int(grid[0, 0])
    if not (np.all(grid[0] == structure) and np.all(grid[-1] == structure)
            and np.all(grid[:, 0] == structure) and np.all(grid[:, -1] == structure)):
        raise ValueError("outer frame is not uniform")
    rows = np.flatnonzero(np.all(grid == structure, axis=1))
    cols = np.flatnonzero(np.all(grid == structure, axis=0))
    if len(rows) > 2 and len(cols) > 2:
        raise ValueError("dividers must be parallel")
    if len(rows) > 2:
        transpose = False
    elif len(cols) > 2:
        transpose = True
    else:
        rr, cc = np.where(grid[1:-1, 1:-1] == structure)
        if not len(rr):
            raise ValueError("missing binary motif")
        transpose = int(rr.max() - rr.min()) > int(cc.max() - cc.min())
    normalized = grid.T.copy() if transpose else grid.copy()
    dividers = np.flatnonzero(np.all(normalized == structure, axis=1)).tolist()
    background = int(normalized[1, 1])
    if background == structure:
        raise ValueError("background must differ from structure")
    lanes = []
    for first, last in zip(dividers, dividers[1:]):
        lane = normalized[first:last+1].copy()
        rr, cc = np.where(lane[1:-1, 1:-1] == structure)
        if not len(rr):
            raise ValueError("missing compartment motif")
        r0, r1 = int(rr.min()) + 1, int(rr.max()) + 2
        c0, c1 = int(cc.min()) + 1, int(cc.max()) + 2
        rectangle = (slice(r0, r1), slice(c0, c1))
        motif = lane[rectangle]
        accents = set(map(int, np.unique(motif))) - {structure}
        if len(accents) != 1 or background in accents or motif.shape[1] <= motif.shape[0]:
            raise ValueError("motif must be a long structural/accent rectangle")
        accent, = accents
        if np.count_nonzero(lane == accent) != np.count_nonzero(motif == accent):
            raise ValueError("accent checkpoints must start inside their motif")
        lanes.append({"first": first, "last": last, "grid": lane,
                      "rectangle": rectangle, "accent": accent})
    if not lanes:
        raise ValueError("no compartments")
    return normalized, bool(transpose), structure, background, lanes


def _solve_details(input_grid, *, repair=True, bend_first=False, ignore_marks=False,
                   diagnostic_budget=None):
    a = _grid(input_grid)
    normalized, transposed, structure, background, lanes = _parse(a)
    answer = normalized.copy()
    details = []
    # Every compartment is repaired before its checkpoints are routed.
    for lane in lanes:
        g = lane["grid"].copy()
        rect, accent = lane["rectangle"], lane["accent"]
        clean, period, errors = _recover(g[rect], structure, accent)
        if repair:
            g[rect] = clean
        restored = g.copy()
        if ignore_marks:
            g[(g != structure) & (g != background) & (g != accent)] = background
        result = _route(g, accent, background, bend_first=bend_first,
                        max_states=diagnostic_budget)
        if result["path"] is None:
            raise ValueError("equal-cost route tie")
        out = restored.copy()
        out.ravel()[list(result["path"])] = accent
        if ignore_marks:
            # Diagnostic output shows the competing route even at ignored marks.
            out = g.copy()
            out.ravel()[list(result["path"])] = accent
        answer[lane["first"]:lane["last"]+1] = out
        details.append({**lane, **result, "repaired": restored,
                        "period": period, "errors": errors,
                        "output": out, "transposed": transposed})
    return (answer.T.copy() if transposed else answer), details


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive periodic repairs and unique optimal routes entirely from input."""
    return _solve_details(input_grid)[0]


def _make_lane(rng, width, palette, accent, period, copies, *, tradeoff=False,
               jog=False, polarity=0):
    structure, background = palette[:2]
    markers = [c for c in palette[2:] if c != accent]
    tail = int(rng.integers(0, 2))
    length = period * copies + tail
    cross = int(rng.integers(4, 6) if jog else rng.integers(2, 5))
    top = int(rng.integers(3, 5) if tradeoff else rng.integers(2, 4))
    bottom = int(rng.integers(2, 4))
    height = cross + top + bottom + 2
    # Keep every inner corner blank, so the background is visibly encoded.
    spare = width - length - 2
    left = int(rng.integers(1, spare))
    r, c = top + 1, left + 1
    g = np.full((height, width), background, dtype=np.int8)
    g[[0, -1], :] = structure
    g[:, [0, -1]] = structure
    motif = np.full((cross, length), structure, dtype=np.int8)
    phase = 1 if tradeoff or jog else int(rng.integers(1, period-1))
    motif[:, phase::period] = accent
    if jog:
        # Repeated one-cell-wide bent fragments, with vertical terminal runs.
        # This is vector-slice periodicity, rather than identical scalar rows.
        turn = int(rng.integers(1, cross-2))
        motif[turn+1:, phase::period] = structure
        motif[turn:, phase+1::period] = accent
    g[r:r+cross, c:c+length] = motif
    x = c + phase
    if tradeoff:
        # The nearer corridor has a local notch: length-first accepts two extra
        # turns; bend-first takes a longer outer rectangle. Side is randomized.
        g[r-1, x+1] = markers[0]
        g[r-2, x+period-1] = markers[1]
        g[r+cross, x] = markers[2]
    else:
        # One blocked terminal breaks the otherwise tied two stitch parities.
        g[r-1, x] = markers[0]
    result = _route(g, accent, background, max_states=2000)
    if result["path"] is None:
        raise ValueError("ambiguous route proposal")
    if not tradeoff:
        # Extra inert fragments delete alternatives without hiding a different
        # optimal route. The parity-blocking mark above remains answer-causal.
        protected = set(result["path"])
        for color in markers[1:1+int(rng.integers(0, 3))]:
            upper = bool(rng.integers(0, 2))
            row = int(rng.integers(1, r)) if upper else int(rng.integers(r+cross, height-1))
            col = int(rng.integers(2, width-2))
            run = int(rng.integers(1, min(4, width-1-col)))
            cells = [(row, col+k) for k in range(run)]
            if all(g[a, b] == background and a*width+b not in protected for a, b in cells):
                for a, b in cells:
                    g[a, b] = color
    if tradeoff:
        alternate = _route(g, accent, background, bend_first=True, max_states=2000)
        if (alternate["path"] is None or not (
                result["cost"][0] < alternate["cost"][0]
                and result["cost"][1] > alternate["cost"][1])):
            raise ValueError("missing length-before-bend witness")
    out = g.copy()
    out.ravel()[list(result["path"])] = accent
    noisy = motif.copy()
    # Both error directions appear. Distinct phase/row groups preserve majority.
    true_cells = [(a, b) for a, b in np.argwhere(motif == accent)
                  if 0 < b < length-1]
    false_cells = [(a, b) for a, b in np.argwhere(motif == structure)
                   if 0 < b < length-1]
    rng.shuffle(true_cells)
    rng.shuffle(false_cells)
    if jog and polarity == 1:
        # The second demonstration makes the causal dependency particularly
        # visible: a false neighboring checkpoint creates a degree-three spur.
        # Periodic repair removes it before a simple route can even exist.
        ns = _neighbors(cross, length)
        flat = motif.ravel()
        branching = [(a, b) for a, b in false_cells if any(
            flat[q] == accent and sum(flat[t] == accent for t in ns[q]) == 2
            for q in ns[a*length+b])]
        if not branching:
            raise ValueError("missing branch-removal demonstration")
        false_cells = branching
    remove, insert = true_cells[0], false_cells[0]
    if polarity == 0:
        noisy[remove] = structure
    elif polarity == 1:
        noisy[insert] = accent
    else:
        noisy[remove] = structure
        noisy[insert] = accent
    recovered, p, _ = _recover(noisy, structure, accent)
    if p != period or not np.array_equal(recovered, motif):
        raise ValueError("ambiguous motif proposal")
    g[r:r+cross, c:c+length] = noisy
    # Reflection changes the obstacle side and periodic phase together.
    if bool(rng.integers(0, 2)):
        g, out = g[::-1].copy(), out[::-1].copy()
    if bool(rng.integers(0, 2)):
        g, out = g[:, ::-1].copy(), out[:, ::-1].copy()
    return g, out


def _make_pair(rng, palette, count, *, tradeoff, jog, test, rotation, polarity):
    for attempt in range(96):
        try:
            plans = []
            for i in range(count):
                special = tradeoff and i == 0
                bent = jog and i == count-1 and not special
                period = 4 if special or bent else int(rng.integers(3, 6))
                copies = (int(rng.integers(5, 7)) if period == 3 else 5) if test else 4
                plans.append((period, copies, special, bent))
            minimum = max(p*n + 5 for p, n, _, _ in plans)
            width = int(rng.integers(minimum, 31))
            ordinary = [int(c) for c in rng.permutation(palette[2:])]
            lanes = [_make_lane(rng, width, palette, ordinary[i], p, n,
                                tradeoff=special, jog=bent, polarity=polarity)
                     for i, (p, n, special, bent) in enumerate(plans)]
            inp = np.concatenate([lanes[0][0]] + [g[1:] for g, _ in lanes[1:]])
            out = np.concatenate([lanes[0][1]] + [g[1:] for _, g in lanes[1:]])
            if max(inp.shape) > 30:
                continue
            inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
            return {"input": inp, "output": out}
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError("finite pair-generation budget exhausted")


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
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    first_count = int(rng.integers(1, 3))
    first_axis = int(rng.integers(0, 2))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(num_train + num_test):
        test = index >= num_train
        count = first_count if index == 0 else 3-first_count if index == 1 else int(rng.integers(1, 3))
        axis = first_axis if index == 0 else 1-first_axis if index == 1 or test else int(rng.integers(0, 2))
        rotation = axis + 2 * int(rng.integers(0, 2))
        for _ in range(16):
            pair = _make_pair(rng, palette, count,
                              tradeoff=index == 0 or (index > 1 and rng.random() < 0.2),
                              jog=index == 1 or (index > 1 and rng.random() < 0.6),
                              test=test, rotation=rotation,
                              polarity=0 if index == 0 else 1 if index == 1 else 2)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen:
                seen.add(key)
                puzzle["test" if test else "train"].append(pair)
                break
        else:
            raise RuntimeError("distinct-input generation budget exhausted")
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    expected, details = _solve_details(inp)
    assert np.array_equal(expected, out), "incorrect output"
    normalized, _, structure, background, _ = _parse(inp)
    assert not np.array_equal(inp, out)
    for d in details:
        original, repaired, answer = d["grid"], d["repaired"], d["output"]
        rectangle, accent = d["rectangle"], d["accent"]
        assert np.all(original[[1, 1, -2, -2], [1, -2, 1, -2]] == background)
        for color in set(map(int, np.unique(original))) - {structure, background, accent}:
            assert len(_fragments(original, color, _neighbors(*original.shape))) == 1
        mask = np.zeros(original.shape, dtype=bool)
        mask[rectangle] = True
        assert 1 <= d["errors"] <= max(1, original[rectangle].size // 8)
        assert original[rectangle].shape[1] // d["period"] >= 4
        assert np.array_equal(answer[rectangle], repaired[rectangle])
        immutable = (~mask) & (original != background)
        assert np.array_equal(original[immutable], answer[immutable])
        added = (answer != repaired)
        assert np.any(added)
        assert np.all(repaired[added] == background)
        assert np.all(answer[added] == accent)
        assert len(_fragments(repaired, accent, _neighbors(*repaired.shape))) >= 2
        output_fragments = _fragments(answer, accent, _neighbors(*answer.shape))
        assert len(output_fragments) == 1
        assert set(d["path"]) == set(output_fragments[0])
        assert np.all(answer[repaired == structure] == structure)
        assert not np.array_equal(repaired, answer), "omitting B must change the answer"
        # Omitting A cannot correct structure/accent cells inside the motif;
        # B only adds on background, which never occurs inside that rectangle.
        assert np.any(original[rectangle] != answer[rectangle])
    assert normalized.size <= 900
    return details


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, descriptors = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            _, _, structure, background, _ = _parse(inp)
            roles.add((structure, background))
            descriptors[split].append(_inspect_pair(pair))
    assert len(roles) == 1
    train = descriptors["train"]
    assert {len(train[0]), len(train[1])} == {1, 2}
    assert train[0][0]["transposed"] != train[1][0]["transposed"]
    assert any(_bends(f, d["grid"].shape[1]) > 0
               for d in train[1] for f in _fragments(
                   d["repaired"], d["accent"], _neighbors(*d["grid"].shape)))
    has_raw_branch = False
    for d in train[1]:
        try:
            _fragments(d["grid"], d["accent"], _neighbors(*d["grid"].shape))
        except ValueError:
            has_raw_branch = True
    assert has_raw_branch, "repair must restore route feasibility in training"
    witnesses = 0
    for d in train[0]:
        alternate = _route(d["repaired"], d["accent"], next(iter(roles))[1], bend_first=True)
        witnesses += int(alternate["path"] is not None
                         and d["cost"][0] < alternate["cost"][0]
                         and d["cost"][1] > alternate["cost"][1])
    assert witnesses, "training must separate length-first from bend-first"
    for ds in descriptors["test"]:
        assert all(5 <= len(_fragments(d["repaired"], d["accent"],
                                       _neighbors(*d["grid"].shape))) <= 6 for d in ds)
    for ds in descriptors["train"]:
        assert all(len(_fragments(d["repaired"], d["accent"],
                                  _neighbors(*d["grid"].shape))) == 4 for d in ds)
    return None


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, bounded memory, semantic checks, ablations and diversity.

    Returns descriptive compact-report-ready metrics; writes no artifacts.
    This shares the solver with validation and is not independent approval.
    """
    shapes, normalized_structures, signatures = set(), set(), set()
    periods, thicknesses, fragments, roles, counts = set(), set(), set(), set(), set()
    bends, errors, lengths = set(), set(), set()
    total_pairs = hidden_bends = jogged = max_states = no_a_failures = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                inp = pair["input"]
                shapes.add(inp.shape)
                remap = {}
                signature = bytes(remap.setdefault(int(c), len(remap)) for c in inp.ravel())
                normalized_structures.add((inp.shape, signature))
                _, _, wall, bg, ds = _parse(inp)
                roles.add((wall, bg))
                counts.add(len(ds))
                _, ds = _solve_details(inp)
                for d in ds:
                    p = d["path"]
                    width = d["grid"].shape[1]
                    periods.add(d["period"])
                    thicknesses.add(d["grid"][d["rectangle"]].shape[0])
                    lengths.add(d["grid"][d["rectangle"]].shape[1])
                    errors.add(d["errors"])
                    bends.add(d["cost"][1])
                    parts = _fragments(d["repaired"], d["accent"], _neighbors(*d["grid"].shape))
                    fragments.add(len(parts))
                    jogged += int(any(_bends(f, width) for f in parts))
                    max_states = max(max_states, d["states"])
                    for a, b, c in zip(p, p[1:], p[2:]):
                        hidden_bends += int(_direction(a, b, width) != _direction(b, c, width)
                                            and d["repaired"].ravel()[b] == bg)
        # Omitting A is infeasible in the second demonstration: its inserted
        # false checkpoint creates a branch. This certificate is local and
        # conclusive, so no exponential search on malformed fragments is needed.
        _, _, _, bg, raw_lanes = _parse(puzzle["train"][1]["input"])
        infeasible_raw = False
        for d in raw_lanes:
            try:
                _fragments(d["grid"], d["accent"], _neighbors(*d["grid"].shape))
            except ValueError:
                infeasible_raw = True
        assert infeasible_raw
        no_a_failures += 1
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(normalized_structures) >= 180
    assert periods == {3, 4, 5} and counts == {1, 2}
    assert {2, 3, 4, 5}.issubset(thicknesses)
    assert {4, 5, 6}.issubset(fragments)
    assert errors == {1, 2} and len(lengths) >= 9
    assert len(roles) >= 20 and len(bends) >= 5
    assert hidden_bends >= 500 and jogged >= 42 and no_a_failures == 42
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(12345+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Ambiguity must be rejected rather than resolved by a coordinate tie-break.
    tied = np.ones((6, 13), dtype=np.int8)
    tied[1:-1, 1:-1] = 0
    tied[2:4, 2:11] = 1
    tied[2:4, 3:11:3] = 2
    assert _route(tied, 2, 0)["path"] is None
    # Check the heuristic against unheuristic exhaustive search on small cases.
    # These are diagnostic routes, not saved templates used by generation.
    for marker in ((1, 3), (4, 3), (1, 6)):
        small = tied.copy()
        small[marker] = 3
        compared = []
        for heuristic in (True, False):
            try:
                result = _route(small, 2, 0, use_heuristic=heuristic, max_states=50000)
                compared.append((result["cost"], result["path"]))
            except ValueError:
                compared.append(None)
        assert compared[0] == compared[1]
    return {"seeds": 42, "pairs": total_pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "color_normalized_inputs": len(normalized_structures),
            "periods": sorted(periods), "motif_thicknesses": sorted(thicknesses),
            "motif_lengths": sorted(lengths), "fragment_counts": sorted(fragments),
            "route_bends": sorted(bends), "hidden_bends": hidden_bends,
            "jogged_motifs": jogged, "repair_ablation_differences": no_a_failures,
            "raw_checkpoint_infeasibility_certificates": no_a_failures,
            "maximum_route_search_states": max_states, "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
