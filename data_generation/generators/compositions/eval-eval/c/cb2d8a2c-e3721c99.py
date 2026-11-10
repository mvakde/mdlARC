"""Self-addressed baffles: topology filters competing unary route commands.

Only the standard library and NumPy are used. solve() reads no construction
state, examples, files, or seed. The leading edge of each baffle is straight;
the rest of its connected silhouette and its cavities can be irregular.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer 2D arrays, size 1..30, colors 0..9")
    return a


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, cells = deque([(r, c)]), []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        groups.append(cells)
    return groups


def _mask(cells):
    rc = np.asarray(cells, dtype=int)
    lo, hi = rc.min(axis=0), rc.max(axis=0) + 1
    mask = np.zeros(tuple(hi-lo), dtype=bool)
    mask[tuple((rc-lo).T)] = True
    return mask, tuple(map(int, lo))


def _cavities(mask):
    inverse = ~np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = inverse.shape
    return [cells for cells in _components(inverse)
            if all(0 < r < h-1 and 0 < c < w-1 for r, c in cells)]


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _decode(input_grid):
    """Parse visible roles, then simulate collisions in seed-forward coordinates."""
    grid = _grid(input_grid)
    colors, frequencies = np.unique(grid, return_counts=True)
    assert sum(frequencies == frequencies.max()) == 1, "background must be dominant"
    bg = int(colors[frequencies.argmax()])
    seeds = []
    for color, frequency in zip(colors, frequencies):
        if color == bg or frequency != 1:
            continue
        r, c = map(int, np.argwhere(grid == color)[0])
        flags = (r == 0, c == 0, r == grid.shape[0]-1, c == grid.shape[1]-1)
        if sum(flags) == 1:
            seeds.append((int(color), flags.index(True)))
    assert len(seeds) == 1, "one noncorner singleton boundary seed required"
    route, rotation = seeds[0]
    a = np.rot90(grid, -rotation).copy()
    height, full_width = a.shape
    seed_col = int(np.argwhere(a == route)[0, 1])
    separators = [(c, int(a[0, c])) for c in range(1, full_width-1)
                  if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    assert len(separators) == 1, "one complete separator perpendicular to leading edges"
    separator, sep_color = separators[0]
    if seed_col > separator:
        panel, work = a[:, :separator], a[:, separator+1:]
        offset, panel_side = separator+1, "left"
    else:
        panel, work = a[:, separator+1:], a[:, :separator]
        offset, panel_side = 0, "right"
    assert min(panel.shape) >= 2 and work.shape[1] >= 3
    assert not np.any(panel == route) and not np.any(panel == sep_color)
    mapping, entries = {}, []
    for color in np.unique(panel):
        color = int(color)
        if color == bg:
            continue
        groups = _components(panel == color)
        assert len(groups) == 1, "each legend color must have one prototype"
        mask, pos = _mask(groups[0])
        cavities = _cavities(mask)
        k = len(cavities)
        assert 0 <= k <= 4 and k not in mapping, "legend classes must be unique"
        for cavity in cavities:
            assert all(panel[pos[0]+r-1, pos[1]+c-1] == bg for r, c in cavity)
        mapping[k] = color
        entries.append((pos, k, color))
    assert 2 <= len(mapping) <= 4
    entries.sort()
    neutral_roles = set(map(int, np.unique(work))) - {bg, route} - set(mapping.values())
    assert len(neutral_roles) == 1 and sep_color not in neutral_roles
    neutral = neutral_roles.pop()
    width = work.shape[1]
    current = seed_col-offset
    assert 0 < current < width-1
    gates = []
    for cells in _components((work != bg) & (work != route)):
        mask, (top, lo) = _mask(cells)
        h, w = mask.shape
        bottom, hi = top+h-1, lo+w-1
        assert top > 0 and bottom < height-1 and np.all(mask[0])
        assert np.any(work[tuple(np.asarray(cells).T)] == neutral)
        cavities = _cavities(mask)
        k = len(cavities)
        assert 0 <= k <= 4
        for cavity in cavities:
            assert all(work[top+r-1, lo+c-1] == bg for r, c in cavity)
        pin_cells = [(r, c) for r, c in cells if work[r, c] != neutral]
        assert pin_cells and all(r == top and int(work[r, c]) in mapping.values()
                                 for r, c in pin_cells), "pins belong on the leading edge"
        color = mapping.get(k)
        gate = dict(cells=cells, mask=mask, top=top, bottom=bottom, lo=lo, hi=hi,
                    holes=k, areas=tuple(len(x) for x in cavities), color=color,
                    n=0, m=len(pin_cells), side=0, target=None, active=False,
                    crossing=False, turn=None)
        if color is not None:
            relevant = sorted(c for r, c in pin_cells if work[r, c] == color)
            other = sorted(c for r, c in pin_cells if work[r, c] != color)
            n, m = len(relevant), len(other)
            assert 1 <= n <= 3 and 0 <= m <= 3
            low = relevant == [lo+2*j for j in range(n)]
            high = relevant == sorted(hi-2*j for j in range(n))
            assert low != high, "the matching pins must specify one endpoint"
            side = 1 if low else -1
            expected_other = sorted(hi-2*j for j in range(m)) if low else [lo+2*j for j in range(m)]
            assert other == expected_other, "decoy pins alternate from the other end"
            assert w >= max(3, 2*(n+m)+1), "commands need a neutral gap"
            target = hi+n+1 if low else lo-n-1
            assert 0 <= target < width, "encoded bypass must fit the work rectangle"
            gate.update(n=n, m=m, side=side, target=target)
        else:
            positions = sorted(c for r, c in pin_cells)
            assert any(positions == sorted({lo+2*j for j in range(n)} |
                                           {hi-2*j for j in range(m)})
                       and w >= 2*(n+m)+1
                       for n in range(4) for m in range(4) if n+m > 0)
        gates.append(gate)
    gates.sort(key=lambda x: x["top"])
    assert 1 <= len(gates) <= 4
    previous_bottom, start = -1, 0
    path = np.zeros(work.shape, dtype=bool)
    output = a.copy()
    out_work = output[:, offset:offset+width]
    for gate in gates:
        assert gate["top"] > previous_bottom, "forward bands cannot overlap"
        cells = tuple(np.asarray(gate["cells"]).T)
        out_work[cells] = bg if gate["color"] is None else gate["color"]
        intercepts = gate["lo"] <= current <= gate["hi"]
        if gate["color"] is not None:
            turn = gate["top"] - gate["n"] - 1
            assert turn >= 1 and turn > previous_bottom
            if intercepts:
                gate.update(active=True, turn=turn)
                assert turn >= start
                for r in range(start, turn+1):
                    path[r, current] = True
                target = gate["target"]
                step = 1 if target > current else -1
                for c in range(current, target+step, step):
                    path[turn, c] = True
                current, start = target, turn
        previous_bottom = gate["bottom"]
    path[start:, current] = True
    for gate in gates:
        cells = tuple(np.asarray(gate["cells"]).T)
        if gate["color"] is None:
            gate["crossing"] = bool(path[cells].any())
        else:
            assert not path[cells].any(), "route cannot cross a surviving baffle"
    out_work[path] = route
    assert len(_components(path)) == 1 and path[0].sum() == path[-1].sum() == 1
    return np.rot90(output, rotation).copy(), dict(
        rotation=rotation, panel_side=panel_side, roles=(bg, route, neutral, sep_color),
        mapping=mapping, order=tuple(x[1] for x in entries), gates=gates,
        path=path, forward_shape=work.shape, offset=offset)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from the visible input; never mutate the caller's grid."""
    return _decode(input_grid)[0]


def _prototype(k, rng):
    if k == 0:
        mask = np.ones((int(rng.integers(2, 4)), 3), dtype=bool)
    elif k <= 3:
        mask = np.ones((2*k+1, 3), dtype=bool)
        mask[1:2*k:2, 1] = False
    else:
        mask = np.ones((5, 5), dtype=bool)
        mask[1::2, 1::2] = False
    if rng.random() < .4:
        mask = np.rot90(mask).copy()
    if rng.random() < .25 and mask.shape[1] < 7:
        c = int(rng.integers(mask.shape[1]))
        mask = np.insert(mask, c, mask[:, c], axis=1)
    assert len(_components(mask)) == 1 and len(_cavities(mask)) == k
    return mask


def _body(k, h, w, rng, keep_height=False):
    """Unequal connected cavities in separated compartments; sculpt the rear."""
    mask = np.ones((h, w), dtype=bool)
    if k:
        assert h >= 3 and w >= 2*k+1
        lengths = np.ones(k, dtype=int)
        lengths += rng.multinomial(w-2-(k-1)-k, np.full(k, 1/k))
        col = 1
        for length in lengths:
            length = int(length)
            cells = {(int(rng.integers(h-2)), int(rng.integers(length)))}
            area = int(rng.integers(1, (h-2)*length+1))
            while len(cells) < area:
                frontier = sorted({(rr, cc) for r, c in cells
                                   for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                                   if 0 <= rr < h-2 and 0 <= cc < length
                                   and (rr, cc) not in cells})
                cells.add(frontier[int(rng.integers(len(frontier)))])
            cavity = np.zeros((h-2, length), dtype=bool)
            cavity[tuple(np.asarray(sorted(cells)).T)] = True
            for island in _cavities(cavity):
                cavity[tuple((np.asarray(island)-1).T)] = True
            mask[1:h-1, col:col+length][cavity] = False
            col += length+1
    # Every modification preserves the topology and the complete leading edge.
    for _ in range(6):
        candidate = mask.copy()
        c = int(rng.integers(w))
        depth = int(rng.integers(1, h)) if h > 1 else 0
        if depth:
            candidate[h-depth:, c:min(w, c+int(rng.integers(1, 4)))] = False
            if ((not keep_height or candidate[-1].any())
                    and len(_components(candidate)) == 1
                    and len(_cavities(candidate)) == k):
                mask = candidate
    return mask


def _options(width, current, k, n, m, status, forced_side):
    minimum = max(4, 2*k+1, 2*(n+m)+1)
    options = []
    for lo in range(width):
        for hi in range(lo+minimum-1, width):
            intercepts = lo <= current <= hi
            if status in ("active", "erase") and not intercepts:
                continue
            if status == "miss" and intercepts:
                continue
            for side in (-1, 1):
                if forced_side is not None and side != forced_side:
                    continue
                target = hi+n+1 if side == 1 else lo-n-1
                if status != "erase" and not 0 <= target < width:
                    continue
                options.append((lo, hi, side, target))
    return options


def _plan(rng, width, jobs):
    failed = set()

    def visit(i, current):
        if i == len(jobs):
            return []
        if (i, current) in failed:
            return None
        k, n, m, status, side = jobs[i]
        candidates = _options(width, current, k, n, m, status, side)
        for j in rng.permutation(len(candidates)):
            item = candidates[int(j)]
            next_current = item[3] if status == "active" else current
            rest = visit(i+1, next_current)
            if rest is not None:
                return [item] + rest
        failed.add((i, current))
        return None

    for initial in rng.permutation(np.arange(1, width-1)):
        spans = visit(0, int(initial))
        if spans is not None:
            return int(initial), spans
    return None


def _make_pair(rng, keys, jobs, roles, rotation, mirror, rich):
    bg, route, neutral, separator = roles
    class_colors = list(map(int, rng.choice([c for c in range(10) if c not in roles],
                                           len(keys), replace=False)))
    mapping = dict(zip(keys, class_colors))
    glyphs = [_prototype(k, rng) for k in keys]
    panel_width = max(x.shape[1] for x in glyphs) + 2
    legend_height = sum(x.shape[0] for x in glyphs) + len(keys)+1
    max_width = 29-panel_width
    base_width = int(rng.integers(18, max_width+1))
    chosen = None
    for width in dict.fromkeys((base_width, max_width)):
        chosen = _plan(rng, width, jobs)
        if chosen is not None:
            break
    if chosen is None:
        return None
    initial, spans = chosen
    heights = [int(rng.integers(3, 7 if rich else 5)) for _ in jobs]
    for i, job in enumerate(jobs):
        if job[0] == 0 and rng.random() < .35:
            heights[i] = int(rng.integers(1, 4))
    if rich:
        # A visible extension beyond the <=4-cell training body heights.
        # Rich tests have at most three bodies, leaving room for this anchor.
        heights[0] = int(rng.integers(5, 7))
    distances = [job[1]+1 for job in jobs]
    while 2+sum(heights)+sum(distances) > 30:
        reducible = [i for i, h in enumerate(heights)
                     if h > (5 if rich and i == 0 else 3 if jobs[i][0] else 1)]
        assert reducible
        heights[int(rng.choice(reducible))] -= 1
    minimum = max(legend_height, 2+sum(heights)+sum(distances))
    height = int(rng.integers(minimum, 31))
    slack = height-2-sum(heights)-sum(distances)
    gaps = rng.multinomial(slack, np.full(len(jobs)+1, 1/(len(jobs)+1)))
    full_width = panel_width+1+width
    a = np.full((height, full_width), bg, dtype=np.int8)
    a[:, panel_width] = separator
    r = 1 + int(rng.integers(height-legend_height+1))
    for glyph, color in zip(glyphs, class_colors):
        c = int(rng.integers(1, panel_width-glyph.shape[1]))
        a[r:r+glyph.shape[0], c:c+glyph.shape[1]][glyph] = color
        r += glyph.shape[0]+1
    offset = panel_width+1
    a[0, offset+initial] = route
    b = a.copy()
    work, out_work = a[:, offset:], b[:, offset:]
    path = np.zeros_like(work, dtype=bool)
    cursor, current, start = 1, initial, 0
    for i, ((k, n, m, status, _), (lo, hi, side, target), h, d) in enumerate(
            zip(jobs, spans, heights, distances)):
        cursor += int(gaps[i])
        top = cursor+d
        mask = _body(k, h, hi-lo+1, rng, keep_height=rich and i == 0)
        work[top:top+h, lo:hi+1][mask] = neutral
        chosen_color = mapping[k] if status != "erase" else int(rng.choice(class_colors))
        other_colors = [c for c in class_colors if c != chosen_color]
        for j in range(n):
            c = lo+2*j if side == 1 else hi-2*j
            work[top, c] = chosen_color
        for j in range(m):
            c = hi-2*j if side == 1 else lo+2*j
            work[top, c] = int(rng.choice(other_colors))
        out_work[top:top+h, lo:hi+1][mask] = bg if status == "erase" else mapping[k]
        if status == "active":
            turn = top-d
            path[start:turn+1, current] = True
            path[turn, min(current, target):max(current, target)+1] = True
            current, start = target, turn
        cursor = top+h
    path[start:, current] = True
    out_work[path] = route
    if mirror:
        a, b = a[:, ::-1], b[:, ::-1]
    a, b = np.rot90(a, rotation).copy(), np.rot90(b, rotation).copy()
    prediction, _ = _decode(a)
    assert np.array_equal(prediction, b), "constructor and input-only decoder disagree"
    return {"input": a, "output": b}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.random() < .4)
    maximum = int(rng.choice((2, 3)))
    regime = int(rng.integers(3))  # held-out direction, longer sequence, richer bodies
    directions = list(map(int, rng.permutation(4)))
    train_dirs = directions[:2] if regime == 0 else directions
    test_dirs = directions[2:] if regime == 0 else directions
    repeated = bool(rng.random() < .6)
    first_side = int(rng.choice((-1, 1)))
    first_keys = list(map(int, rng.choice(([0, 2], [1, 3], [0, 3], [2, 4], [1, 4]))))
    first_keys.sort(reverse=True)
    second_keys = list(map(int, rng.choice(5, int(rng.integers(2, 5)), replace=False)))
    rng.shuffle(second_keys)
    # This repeated-side demonstration needs two copies of its first class.
    # Two four-hole bodies with clearance four need a 23-cell work field,
    # leaving too little space for that class's five-cell-wide prototype.
    # Use another listed class here; four-hole bodies remain unrestricted in
    # other examples and the input-only solver.
    if second_keys[0] == 4:
        second_keys[0], second_keys[1] = second_keys[1], second_keys[0]
    first_class = next(k for k in first_keys if k != 1)
    second_class = next(k for k in first_keys if k != first_class)
    absent = int(rng.choice([k for k in range(5) if k not in first_keys]))
    teaching = [
        (first_keys, [(first_class, 1, 2, "active", first_side),
                      (absent, 1, 1, "erase", None),
                      (second_class, 2, 1, "active", first_side if repeated else -first_side)]),
        (second_keys, [(second_keys[0], maximum, 0, "active", -first_side),
                       (second_keys[1], 1, 1, "miss", None),
                       (second_keys[0], 1, 2, "active", -first_side)])]
    puzzle, seen = {"train": [], "test": []}, set()
    teaching_mirror = bool(rng.integers(2))
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                keys, jobs = teaching[index]
            else:
                size = int(rng.integers(2, 5))
                keys = list(map(int, rng.choice(5, size, replace=False)))
                rng.shuffle(keys)
                number = (4 if split == "test" and regime == 1 else
                          int(rng.integers(1, 5 if split == "test" and regime != 2 else 4)))
                jobs = []
                for j in range(number):
                    status = "active" if j == 0 else str(rng.choice(("active", "erase", "miss"), p=(.55,.3,.15)))
                    pool = keys if status != "erase" else [k for k in range(5) if k not in keys]
                    k = int(rng.choice(pool))
                    n = int(rng.integers(1, maximum+1))
                    m = int(rng.integers(0, 4))
                    jobs.append((k, n, m, status, None))
            for attempt in range(40):
                local_roles = roles if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
                dirs = train_dirs if split == "train" else test_dirs
                pair = _make_pair(rng, keys, jobs, local_roles, int(rng.choice(dirs)),
                                  teaching_mirror if split == "train" and index < 2
                                  else bool(rng.integers(2)),
                                  split == "test" and regime == 2)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded geometric construction exhausted")
    # The teaching obligations travel with examples, not their displayed index.
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_gates, test_gates = set(), [], []
    by_split = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = _key(a)
            assert key not in seen, "inputs must be distinct"
            seen.add(key)
            prediction, data = _decode(a)
            assert np.array_equal(prediction, b), "wrong route, normalization, or preservation"
            by_split[split].append(data)
            (train_gates if split == "train" else test_gates).extend(data["gates"])
    active = [g for g in train_gates if g["active"]]
    assert {1, 2} <= {g["n"] for g in active}, "teach more than constant clearance"
    assert len({g["holes"] for g in active}) >= 2, "teach topology matching"
    assert {g["side"] for g in active} == {-1, 1}, "teach both endpoint directions"
    assert any(g["m"] > g["n"] for g in active), "wrong-end decoys need a training witness"
    assert any(g["n"] != g["holes"] for g in active), "holes do not directly give clearance"
    assert any(g["color"] is None and g["crossing"] for g in train_gates), "teach erasure opening the route"
    assert any(g["color"] is not None and not g["active"] for g in train_gates), "teach off-ray activation"
    assert {g["n"] for g in test_gates if g["active"]} <= {g["n"] for g in active}
    held_out_direction = ({d["rotation"] for d in by_split["test"]}
                          - {d["rotation"] for d in by_split["train"]})
    longer = max(len(d["gates"]) for d in by_split["test"]) > max(
        len(d["gates"]) for d in by_split["train"])
    taller = max(g["mask"].shape[0] for g in test_gates) > max(
        g["mask"].shape[0] for g in train_gates)
    assert held_out_direction or longer or taller, "a visible train/test shift is required"
    # Recoloring-only omits A; raw-pin counting omits B. Both alter actual labels.
    for pair in puzzle["train"]:
        expected, data = _decode(pair["input"])
        a = np.rot90(np.asarray(pair["input"]), -data["rotation"])
        b = np.rot90(expected, -data["rotation"])
        work_a = a[:, data["offset"]:data["offset"]+data["forward_shape"][1]]
        work_b = b[:, data["offset"]:data["offset"]+data["forward_shape"][1]]
        assert np.any((work_a == data["roles"][0]) & (work_b == data["roles"][1]))


def diversity_features(pair):
    _, data = _decode(pair["input"])
    gates = data["gates"]
    return dict(shape=tuple(np.asarray(pair["input"]).shape), rotation=data["rotation"],
                panel_side=data["panel_side"], roles=data["roles"], keys=tuple(sorted(data["mapping"])),
                order=data["order"], gates=len(gates), holes=tuple(g["holes"] for g in gates),
                sizes=tuple(g["mask"].shape for g in gates),
                areas=tuple(g["areas"] for g in gates),
                pin_counts=tuple((g["n"], g["m"]) for g in gates if g["color"] is not None),
                sides=tuple(g["side"] for g in gates if g["active"]),
                erased=sum(g["color"] is None for g in gates),
                offray=sum(g["color"] is not None and not g["active"] for g in gates),
                erased_crossings=sum(g["crossing"] for g in gates),
                irregular_cavities=sum(not _mask(cells)[0].all() for g in gates
                                       for cells in _cavities(g["mask"])),
                irregular=sum(not np.all(g["mask"]) for g in gates if g["holes"] == 0))


def _literal_regression(extra_hole=False):
    """Hand-designed scene and explicit expected bends, independent of sampling."""
    a = np.zeros((25, 26), dtype=np.int8)
    a[:, 5] = 8
    a[1:3, 1:4] = 2
    a[4:7, 1:4] = 3
    a[5, 2] = 0
    a[8:13, 1:4] = 4
    a[9, 2] = a[11, 2] = 0
    a[0, 15] = 6
    work = a[:, 6:]
    work[5:8, 5:12] = 5
    work[6, 8] = 0
    if extra_hole:
        work[6, 6] = 0
    else:
        work[6, 7] = 0
    work[5, 5] = 3
    work[5, 9] = work[5, 11] = 4
    work[10:13, 11:18] = 5
    work[11, [12, 14, 16]] = 0
    work[10, 11], work[10, 17] = 2, 3
    work[15, :7] = 5
    work[16:18, 3] = 5
    work[15, 0], work[15, 6] = 2, 3
    work[21:24, 9:16] = 5
    work[22, [10, 12]] = 0
    work[21, 9] = 3
    work[21, 13] = work[21, 15] = 4
    b = a.copy()
    out = b[:, 6:]
    foreground = work != 0
    out[5:8, 5:12][foreground[5:8, 5:12]] = 4 if extra_hole else 3
    out[10:13, 11:18] = 0
    out[15:18, :7][foreground[15:18, :7]] = 2
    out[21:24, 9:16][foreground[21:24, 9:16]] = 4
    if extra_hole:
        out[:3, 9] = 6
        out[2, 2:10] = 6
        out[2:14, 2] = 6
        out[13, 2:9] = 6
        out[13:, 8] = 6
    else:
        out[:4, 9] = 6
        out[3, 9:14] = 6
        out[3:19, 13] = 6
        out[18, 6:14] = 6
        out[18:, 6] = 6
    return a, b


def self_test():
    """Bounded topology, physical geometry, count, corruption and diversity checks."""
    diagonal = np.array([[0,1,1], [1,0,1], [1,1,1]], dtype=bool)
    assert len(_cavities(diagonal)) == 1
    diagonal[0, 1] = False
    assert len(_cavities(diagonal)) == 0
    notch = np.array([[1,0,0,1], [1,0,0,1], [1,1,1,1]], dtype=bool)
    assert len(_cavities(notch)) == 0
    for extra_hole in (False, True):
        a, expected = _literal_regression(extra_hole)
        predicted, data = _decode(a)
        assert np.array_equal(predicted, expected)
        # Splitting two empty cells reverses the command and activates the T.
        assert data["gates"][0]["side"] == (-1 if extra_hole else 1)
        assert data["gates"][2]["active"] == extra_hole
        palette = np.array([7,0,9,1,6,2,4,8,3,5], dtype=np.int8)
        for rotation in range(4):
            for mirror in (False, True):
                inp = a[:, ::-1] if mirror else a
                out = expected[:, ::-1] if mirror else expected
                assert np.array_equal(solve(palette[np.rot90(inp, rotation)]),
                                      palette[np.rot90(out, rotation)])
    # Pin-and-span information is identical, but topology reverses the first
    # bypass and changes whether the downstream T activates.
    a, out_a = _literal_regression(False)
    b, out_b = _literal_regression(True)
    assert np.array_equal(np.where(a == 5, 0, a), np.where(b == 5, 0, b))
    assert not np.array_equal(out_a == 6, out_b == 6)
    # Recoloring alone omits A, while counting every pin omits B's filter.
    _, data = _decode(a)
    first = data["gates"][0]
    assert first["n"] == 1 and first["m"] == 2
    assert first["turn"] == 3 and first["top"]-first["n"]-first["m"]-1 == 1
    normalized = a.copy()
    normalized_work = normalized[:, data["offset"]:]
    for gate in data["gates"]:
        normalized_work[tuple(np.asarray(gate["cells"]).T)] = (
            data["roles"][0] if gate["color"] is None else gate["color"])
    assert not np.array_equal(normalized, out_a)
    # Changing an inapplicable pin color among other legend colors has no effect.
    irrelevant = a.copy()
    irrelevant[5, 15] = 2  # work column 9; first object's selected color is 3
    assert np.array_equal(solve(irrelevant), out_a)
    bad, _ = _literal_regression()
    bad[11, 2] = 4  # duplicate one-hole prototypes in two colors
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous legend accepted")
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                features.append(diversity_features(pair))
    assert len(signatures) == 42
    assert len({f["shape"] for f in features}) >= 35
    assert {f["rotation"] for f in features} == set(range(4))
    assert {f["panel_side"] for f in features} == {"left", "right"}
    assert {f["gates"] for f in features} == {1, 2, 3, 4}
    assert {k for f in features for k in f["holes"]} == set(range(5))
    assert {n for f in features for n, m in f["pin_counts"]} == {1, 2, 3}
    assert len({f["keys"] for f in features}) >= 15
    assert len({f["sizes"] for f in features}) >= 100
    assert any(a == b for f in features for a, b in zip(f["sides"], f["sides"][1:]))
    assert any(a != b for f in features for a, b in zip(f["sides"], f["sides"][1:]))
    assert any(f["erased"] == 0 for f in features)
    assert any(f["offray"] for f in features)
    assert any(f["irregular"] for f in features)
    assert any(f["irregular_cavities"] for f in features)
    assert max(area for f in features for areas in f["areas"] for area in areas) >= 6
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(817+n*10+m, n, m))
    for counts in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (3,1.5), (None,1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    p, q = generate_puzzle(31, 2, 1), generate_puzzle(31, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
               for k in ("input", "output"))
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal explicit counts)")
