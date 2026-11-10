"""Wave-bent chamber scaffolds followed by all-shortest doorway routes.

Only NumPy and the Python standard library are required. Generation and checks
are in-memory and bounded; this module does not write artifacts or trace files.
"""

from collections import deque
import hashlib
import numpy as np


_STEP = ((-1, 0), (0, 1), (1, 0), (0, -1))
_CYCLE = (0, -1, 0, 1)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    """Four-connected cell sets, bounded by the canvas."""
    h, w = mask.shape
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, queue = {start}, deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _STEP:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    queue.append(p)
        result.append(cells)
    return result


def _rooms(grid, bg):
    """Doors count as scaffold here, temporarily sealing the two rooms."""
    h, w = grid.shape
    rooms = []
    for cells in _components(grid == bg):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        rs, cs = zip(*cells)
        a, b, l, u = min(rs), max(rs), min(cs), max(cs)
        if len(cells) != (b - a + 1) * (u - l + 1):
            raise ValueError("room interior must be rectangular")
        rooms.append((a - 1, b + 1, l - 1, u + 1))
    if len(rooms) != 2:
        raise ValueError("expected exactly two enclosed rectangular rooms")
    return tuple(sorted(rooms))


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(q, l) for q in range(t, b + 1)}
            | {(q, r) for q in range(t, b + 1)})


def _door_options(grid, box, bg):
    t, b, l, r = box
    candidates = ([(t, c, -1, 0) for c in range(l + 1, r)]
                  + [(b, c, 1, 0) for c in range(l + 1, r)]
                  + [(q, l, 0, -1) for q in range(t + 1, b)]
                  + [(q, r, 0, 1) for q in range(t + 1, b)])
    h, w = grid.shape
    return [(q, c, dr, dc) for q, c, dr, dc in candidates
            if 0 <= q + dr < h and 0 <= c + dc < w
            and grid[q + dr, c + dc] == bg]


def _decode(a):
    bg = int(a[0, 0])
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3:
        raise ValueError("expected background, wall and doorway colors")
    choices = [int(c) for c, n in zip(colors, counts) if n == 2 and c != bg]
    if len(choices) != 1:
        raise ValueError("doorway color must occur exactly twice")
    door = choices[0]
    wall = next(int(c) for c in colors if c not in (bg, door))
    rows, cols = np.where(a != bg)
    top, bottom = int(rows.min()), int(rows.max())
    if cols.min() < 1 or cols.max() >= a.shape[1] - 1:
        raise ValueError("scaffold needs a one-cell lateral margin")
    rooms = _rooms(a, bg)
    doors = []
    shells = []
    for box in rooms:
        shell = _perimeter(box)
        if any(a[p] not in (wall, door) for p in shell):
            raise ValueError("broken input chamber shell")
        markers = [p for p in shell if a[p] == door]
        options = {p[:2]: p[2:] for p in _door_options(a, box, bg)}
        if len(markers) != 1 or markers[0] not in options:
            raise ValueError("each room needs one noncorner outward-clear door")
        p = markers[0]
        doors.append((p, options[p]))
        shells.append(shell)
    if shells[0] & shells[1]:
        raise ValueError("room shells overlap")
    return dict(bg=bg, wall=wall, door=door, top=top, bottom=bottom,
                height=bottom - top + 1, rooms=rooms, doors=tuple(doors),
                shells=shells)


def _shift(a, bg, bottom, mode="bottom"):
    out = np.full_like(a, bg)
    rows, cols = np.where(a != bg)
    if mode == "bottom":
        distances = bottom - rows
    elif mode == "canvas":
        distances = a.shape[0] - 1 - rows
    elif mode == "top":
        distances = rows - int(rows.min())
    else:
        raise ValueError("unknown wave anchor")
    offsets = np.asarray(_CYCLE)[distances % 4]
    dest = cols + offsets
    if np.any((dest < 0) | (dest >= a.shape[1])):
        raise ValueError("wave would clip scaffold")
    out[rows, dest] = a[rows, cols]
    return out


def _distances(passable, start):
    h, w = passable.shape
    open_cells = passable.ravel().tolist()
    distances = [-1] * (h * w)
    s = int(start[0]) * w + int(start[1])
    if not open_cells[s]:
        raise ValueError("blocked terminal")
    distances[s] = 0
    queue = deque([s])
    while queue:
        i = queue.popleft()
        r, c = divmod(i, w)
        neighbors = []
        if r:
            neighbors.append(i - w)
        if c + 1 < w:
            neighbors.append(i + 1)
        if r + 1 < h:
            neighbors.append(i + w)
        if c:
            neighbors.append(i - 1)
        for j in neighbors:
            if open_cells[j] and distances[j] < 0:
                distances[j] = distances[i] + 1
                queue.append(j)
    return np.asarray(distances, dtype=np.int16).reshape(h, w)


def _route(a, info):
    terminals = [tuple(map(int, p)) for p in np.argwhere(a == info["door"])]
    if len(terminals) != 2:
        raise ValueError("expected two shifted doors")
    first, second = terminals
    d0 = _distances(a != info["wall"], first)
    d1 = _distances(a != info["wall"], second)
    length = int(d0[second])
    if length < 0:
        raise ValueError("shifted doors are disconnected")
    union = (d0 >= 0) & (d1 >= 0) & (d0 + d1 == length)
    return union, length, terminals


def _evaluate(a):
    info = _decode(a)
    moved = _shift(a, info["bg"], info["bottom"])
    union, length, terminals = _route(moved, info)
    output = moved.copy()
    output[union & (moved == info["bg"])] = info["door"]
    return output, info, moved, union, length, terminals


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, endpoint, phase and route from this input alone."""
    return _evaluate(_grid(input_grid))[0]


def _make_core(rng, simple):
    """Two closed shells joined by one wall path; doors are added later."""
    if simple:
        h1 = h2 = int(rng.integers(7, 11))
        y1 = y2 = 0
    else:
        h1, h2 = map(int, rng.integers(3, 11, size=2))
        y1, y2 = map(int, rng.integers(0, 11, size=2))
        y1, y2 = y1 - min(y1, y2), y2 - min(y1, y2)
    w1, w2 = map(int, rng.integers(3, 11, size=2))
    gap = int(rng.integers(2, 9))
    left2 = w1 + gap
    core = np.zeros((max(y1 + h1, y2 + h2), left2 + w2), dtype=np.int8)
    boxes = ((y1, y1 + h1 - 1, 0, w1 - 1),
             (y2, y2 + h2 - 1, left2, left2 + w2 - 1))
    for box in boxes:
        for p in _perimeter(box):
            core[p] = 1
    if simple:
        core[0, w1 - 1:left2 + 1] = 1
    else:
        ra = int(rng.integers(y1 + 1, y1 + h1 - 1))
        rb = int(rng.integers(y2 + 1, y2 + h2 - 1))
        elbow = int(rng.integers(w1, left2))
        core[ra, w1 - 1:elbow + 1] = 1
        core[min(ra, rb):max(ra, rb) + 1, elbow] = 1
        core[rb, elbow:left2 + 1] = 1
        if rng.integers(2):
            core = core.T
    if rng.integers(2):
        core = core[::-1]
    if rng.integers(2):
        core = core[:, ::-1]
    return core.copy()


def _shift_mask(mask, bottom):
    """Counterfactual: route first, then shift that route; clip only this foil."""
    rows, cols = np.where(mask)
    dest = cols + np.asarray(_CYCLE)[(bottom - rows) % 4]
    valid = (dest >= 0) & (dest < mask.shape[1])
    result = np.zeros_like(mask)
    result[rows[valid], dest[valid]] = True
    return result


def _witnesses(a, info, moved, union, length, terminals):
    before, _, _ = _route(a, info)
    path_after_wave = _shift_mask(before, info["bottom"])
    br, bc = np.where(before)
    destinations = bc + np.asarray(_CYCLE)[(info["bottom"] - br) % 4]
    # Compare background connector membership, not merely displaced walls.
    route_changes = not np.array_equal(union, before)
    order_changes = not np.array_equal(union, path_after_wave)
    p, q = terminals
    manhattan = abs(p[0] - q[0]) + abs(p[1] - q[1])
    return dict(multiple=int(union.sum()) > length + 1,
                route_changes=route_changes, order_changes=order_changes,
                pre_route_unclipped=bool(np.all((destinations >= 0) & (destinations < a.shape[1]))),
                detour=length > manhattan, length=length,
                union_cells=int(union.sum()))


def _pair(rng, *, simple, residues, bottom_pad=None, avoid_palette=None,
          distinct_residue=None, seen=None, witness=False, moving_doors=False):
    for _ in range(320):
        core = _make_core(rng, simple)
        ch, cw = core.shape
        if (ch < 7 or cw > 28 or ch > 29 or ch % 4 not in residues
                or ch % 4 == distinct_residue):
            continue
        bp = int(rng.integers(0, min(4, 30 - ch) + 1)) if bottom_pad is None else bottom_pad
        if ch + bp > 30:
            continue
        tp = int(rng.integers(0, min(4, 30 - ch - bp) + 1))
        lp = int(rng.integers(1, min(6, 29 - cw) + 1))
        rp = int(rng.integers(1, min(6, 30 - cw - lp) + 1))
        palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
        if palette == avoid_palette:
            continue
        bg, wall, door = palette
        a = np.full((ch + tp + bp, cw + lp + rp), bg, dtype=np.int8)
        a[tp:tp + ch, lp:lp + cw][core == 1] = wall
        try:
            boxes = _rooms(a, bg)
            if simple:
                # Opposed doors on one row: a unique shortest straight route.
                boxes = sorted(boxes, key=lambda b: b[2])
                t, b = max(box[0] for box in boxes), min(box[1] for box in boxes)
                possible = [r for r in range(t + 1, b)
                            if a[r, boxes[0][3] + 1] == bg
                            and a[r, boxes[1][2] - 1] == bg
                            and (not moving_doors or _CYCLE[(tp + ch - 1 - r) % 4] != 0)]
                if not possible:
                    continue
                row = int(rng.choice(possible))
                a[row, boxes[0][3]] = a[row, boxes[1][2]] = door
            else:
                for box in boxes:
                    options = _door_options(a, box, bg)
                    r, c, _, _ = options[int(rng.integers(len(options)))]
                    a[r, c] = door
            output, info, moved, union, length, terminals = _evaluate(a)
            # The transformed outward step of each opening must remain usable.
            for (r, c), (dr, dc) in info["doors"]:
                c += _CYCLE[(info["bottom"] - r) % 4]
                if not (0 <= r + dr < a.shape[0] and 0 <= c + dc < a.shape[1]
                        and moved[r + dr, c + dc] == bg):
                    raise ValueError("deformation blocked an outward doorway")
            facts = _witnesses(a, info, moved, union, length, terminals)
            if simple and facts["multiple"]:
                continue
            if witness and not (facts["multiple"] and facts["order_changes"]
                                and facts["route_changes"] and facts["detour"]
                                and facts["pre_route_unclipped"]
                                and len({box[1] % 4 for box in info["rooms"]}) == 2):
                continue
            if seen is not None and _key(a) in seen:
                continue
            return {"input": a, "output": output}
        except (ValueError, IndexError):
            continue
    raise RuntimeError("bounded sampler could not construct a suitable pair")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held_out = int(rng.integers(4))
    training_residues = set(range(4)) - {held_out}
    seen = set()
    train = []
    first_residue = None
    first_palette = None
    for index in range(ntrain):
        simple = index == 0 or (index > 1 and rng.random() < 0.28)
        pair = _pair(rng, simple=simple, residues=training_residues,
                     bottom_pad=0 if index == 0 else 1 if index == 1 else None,
                     avoid_palette=first_palette if index == 1 else None,
                     distinct_residue=first_residue if index == 1 else None,
                     seen=seen, witness=index == 1, moving_doors=index == 0)
        train.append(pair)
        seen.add(_key(pair["input"]))
        if index == 0:
            info = _decode(pair["input"])
            first_residue = info["height"] % 4
            first_palette = (info["bg"], info["wall"], info["door"])
    rng.shuffle(train)
    test = []
    for _ in range(ntest):
        pair = _pair(rng, simple=bool(rng.random() < 0.25), residues={held_out}, seen=seen)
        test.append(pair)
        seen.add(_key(pair["input"]))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_scaffold(a, info):
    bg = info["bg"]
    _require(len(_components(a != bg)) == 1, "input scaffold is not connected")
    _require(info["height"] >= 7, "scaffold does not expose a full wave cycle")
    shell = info["shells"][0] | info["shells"][1]
    bridge = set(map(tuple, np.argwhere(a == info["wall"]).tolist())) - shell
    _require(bool(bridge), "wall bridge missing")
    _require(len(_components(np.array([[((r, c) in bridge) for c in range(a.shape[1])]
                                      for r in range(a.shape[0])], dtype=bool))) == 1,
             "wall bridge disconnected")
    degrees = [sum((r + dr, c + dc) in bridge for dr, dc in _STEP) for r, c in bridge]
    _require(max(degrees) <= 2 and degrees.count(1) == 2, "bridge must be one simple path")
    for room_shell in info["shells"]:
        _require(any((r + dr, c + dc) in room_shell for r, c in bridge for dr, dc in _STEP),
                 "bridge fails to join both rooms")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, evidence = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "canvas changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            expected, info, moved, union, length, terminals = _evaluate(a)
            _check_scaffold(a, info)
            _require(np.array_equal(out, expected), "output is not the exact wave and route union")
            _require(np.array_equal(out == info["wall"], moved == info["wall"]), "wall altered by routing")
            _require(all(out[p] == info["door"] for p in terminals), "door erased")
            _require(np.any(union & (moved == info["bg"])), "empty connector")
            _require(np.any((moved == info["bg"]) & ~union), "all background filled")
            _require(set(map(int, np.unique(out))) == {info["bg"], info["wall"], info["door"]},
                     "unexpected output colors")
            for r in range(a.shape[0]):
                for color in (info["wall"], info["door"]):
                    before = np.flatnonzero(a[r] == color)
                    after = np.flatnonzero(moved[r] == color)
                    _require(np.array_equal(before + _CYCLE[(info["bottom"] - r) % 4], after),
                             "row translation or foreground conservation failed")
            for (r, c), (dr, dc) in info["doors"]:
                c += _CYCLE[(info["bottom"] - r) % 4]
                _require(0 <= r + dr < a.shape[0] and 0 <= c + dc < a.shape[1]
                         and moved[r + dr, c + dc] == info["bg"], "outward door blocked")
            facts = _witnesses(a, info, moved, union, length, terminals)
            facts.update(info)
            facts["bottom_padding"] = a.shape[0] - info["bottom"] - 1
            facts["top_foil"] = np.array_equal(moved, _shift(a, info["bg"], info["bottom"], "top"))
            facts["canvas_foil"] = np.array_equal(moved, _shift(a, info["bg"], info["bottom"], "canvas"))
            evidence[split].append(facts)
    train = evidence["train"]
    _require({x["multiple"] for x in train} == {False, True}, "training needs unique and branching routes")
    _require(any(x["detour"] and x["order_changes"] and x["route_changes"]
                 and x["pre_route_unclipped"]
                 and len({box[1] % 4 for box in x["rooms"]}) == 2 for x in train),
             "training fails to expose causal route recomputation")
    _require(any(_CYCLE[(x["bottom"] - p[0]) % 4] != 0
                 for x in train for p, _ in x["doors"]), "door translation is not demonstrated")
    _require(len({x["height"] % 4 for x in train}) >= 2, "training phase contrast missing")
    _require(len({x["bottom_padding"] % 4 for x in train}) >= 2, "canvas anchor ambiguous")
    _require(not all(x["top_foil"] for x in train) and not all(x["canvas_foil"] for x in train),
             "training supports a competing phase anchor")
    _require(len({(x["bg"], x["wall"], x["door"]) for x in train}) >= 2, "local palette contrast missing")
    _require(not ({x["height"] % 4 for x in train}
                  & {x["height"] % 4 for x in evidence["test"]}), "held-out height phase missing")
    return None


def self_test():
    """Stream 42 seeds; retain hashes/sets, never a collection of puzzles."""
    signatures, shapes, layouts, palettes, heights, widths, directions, sizes = (set() for _ in range(8))
    branching = detours = edge_routes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
                _require(np.array_equal(solve(a.tolist()), pair["output"]), "input-only solve failed")
                out, info, moved, union, length, terminals = _evaluate(a)
                shapes.add(a.shape)
                palettes.add((info["bg"], info["wall"], info["door"]))
                heights.add(info["height"])
                widths.add(int(np.ptp(np.where(a != info["bg"])[1])) + 1)
                directions.add(tuple(d for _, d in info["doors"]))
                sizes.add(int(union.sum()))
                normalized = np.where(a == info["bg"], 0, np.where(a == info["wall"], 1, 2)).astype(np.int8)
                layouts.add(hashlib.sha256(bytes(a.shape) + normalized.tobytes()).digest())
                facts = _witnesses(a, info, moved, union, length, terminals)
                branching += facts["multiple"]
                detours += facts["detour"]
                edge_routes += bool(union[0].any() or union[-1].any() or union[:, 0].any() or union[:, -1].any())
        signatures.add(digest.digest())
    _require(len(signatures) == 42 and len(layouts) >= 180, "insufficient structural diversity")
    _require(len(shapes) >= 60 and len(palettes) >= 80, "shape or palette diversity missing")
    _require(len(heights) >= 14 and len(widths) >= 14 and len(directions) >= 12, "geometry diversity missing")
    _require(len(sizes) >= 30 and branching >= 80 and detours >= 70, "route diversity missing")
    _require(edge_routes > 0, "no finite-canvas edge-route examples")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 5 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(616922283, 5, 3), generate_puzzle(616922283, 5, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                 for x, y in zip(a[s], b[s]) for k in ("input", "output")), "nondeterminism")


if __name__ == "__main__":
    self_test()
