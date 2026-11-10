"""Contour printing press: clipped color rods ink contours around inserted teeth.

Only NumPy and the standard library are used. Solving reads the current input;
construction parameters and the random generator are never consulted by solve.
No puzzles or execution traces are written by this module.
"""
from collections import deque
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    """Four-connected components, as masks of the original shape."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        queue = deque([(sr, sc)])
        seen[sr, sc] = True
        component = np.zeros(mask.shape, dtype=bool)
        while queue:
            r, c = queue.popleft()
            component[r, c] = True
            for dr, dc in _D4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(component)
    return result


def _erosion(mask, diagonal=True):
    """Solver's contour coordinate: strip all eight-neighbor boundaries."""
    h, w = mask.shape
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    while np.any(current):
        p = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in (_D8 if diagonal else _D4):
            interior &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _bfs_depth(mask):
    """Construction/checking algorithm, distinct from iterative erosion."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc])
               for dr, dc in _D8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                queue.append((rr, cc))
    return depth


def _bbox(mask):
    rr, cc = np.nonzero(mask)
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def _read_upright(a, bg, structure, parts):
    """Recognize a closed chamber block above its matching detached comb."""
    wall, comb = sorted(parts, key=lambda m: _bbox(m)[0])
    t, b, l, r = _bbox(wall)
    et, eb, el, er = _bbox(comb)
    _check(l == el and r == er and et >= b + 2, "comb must be detached below the housing")
    _check(b - t >= 4 and r - l >= 8, "housing too small")
    _check(bool(np.all(wall[t, l:r + 1]) and np.all(wall[b, l:r + 1])
                and np.all(wall[t:b + 1, l]) and np.all(wall[t:b + 1, r])), "closed outer walls required")
    _check(bool(np.all(comb[eb, l:r + 1])), "comb base must match the full bottom wall")
    columns = np.flatnonzero(comb[eb - 1]).tolist()
    _check(1 <= len(columns) <= 3, "one to three comb teeth required")
    expected_comb = np.zeros(a.shape, dtype=bool)
    expected_comb[eb, l:r + 1] = True
    heights = []
    for col in columns:
        row = eb - 1
        while row >= 0 and comb[row, col]:
            row -= 1
        height = eb - row - 1
        _check(1 <= height <= b - t - 2, "tooth leaves no palette space")
        expected_comb[eb - height:eb, col] = True
        heights.append(height)
    _check(np.array_equal(comb, expected_comb), "malformed comb")
    inside = np.zeros(a.shape, dtype=bool)
    inside[t + 1:b, l + 1:r] = True
    rooms = _components(inside & ~wall)
    _check(len(rooms) == len(columns), "one tooth and rod per chamber required")
    expected_input = np.full(a.shape, bg, dtype=np.int8)
    expected_input[wall | comb] = structure
    readings = []
    used = set()
    for room in rooms:
        matching = [i for i, col in enumerate(columns) if room[t + 1, col]]
        _check(len(matching) == 1, "ambiguous chamber to tooth correspondence")
        i = matching[0]
        _check(i not in used, "duplicate tooth correspondence")
        used.add(i)
        col, height = columns[i], heights[i]
        _check(bool(np.all(room[t + 1:b, col])), "rod/tooth axis crosses an internal wall")
        cues = np.argwhere(room & (a != bg))
        _check(len(cues) > 0 and bool(np.all(cues[:, 1] == col)), "chamber cues must form one straight rod")
        length = len(cues)
        _check(np.array_equal(cues[:, 0], np.arange(t + 1, t + length + 1)), "rod must hang contiguously from roof")
        capacity = b - t - height - 1
        _check(capacity <= length <= b - t - 1, "rod must contact or overlap the inserted tooth")
        word = tuple(int(v) for v in a[t + 1:t + length + 1, col])
        _check(all(v not in (bg, structure) for v in word), "rod colors must be distinct from the two structural roles")
        expected_input[t + 1:t + length + 1, col] = word
        shift = length - capacity
        period = word[shift:]
        final_room = room.copy()
        final_room[b - height:b, col] = False
        _check(len(_components(final_room)) == 1, "inserted tooth must leave its chamber connected")
        readings.append(dict(room=room, final_room=final_room, col=col, height=height,
                             word=word, period=period, shift=shift, capacity=capacity))
    _check(np.array_equal(a, expected_input), "unexplained pixels outside the machine")
    return dict(grid=a, bg=bg, structure=structure, wall=wall, comb=comb,
                top=t, bottom=b, left=l, right=r, external_base=eb,
                rooms=sorted(readings, key=lambda v: v["col"]))


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _check(all(int(a[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1))), "background corners required")
    # Neither literal color values nor the modal color determine the hardware.
    candidates = []
    for color in np.unique(a):
        if int(color) == bg:
            continue
        parts = _components(a == color)
        if len(parts) != 2:
            continue
        for turn in range(4):
            upright = np.rot90(a, turn)
            try:
                reading = _read_upright(upright, bg, int(color),
                                        [np.rot90(m, turn) for m in parts])
            except ValueError:
                continue
            reading["turn"] = turn
            candidates.append(reading)
    _check(len(candidates) == 1, "input must have exactly one machine interpretation")
    return candidates[0]


def _print(reading, depth_function=_erosion, variant=None):
    """Ink the actual chamber geometry after docking the comb."""
    a = reading["grid"]
    out = np.full(a.shape, reading["bg"], dtype=np.int8)
    out[reading["wall"]] = reading["structure"]
    delta = reading["bottom"] - reading["external_base"]
    for row, col in np.argwhere(reading["comb"]):
        out[int(row) + delta, col] = reading["structure"]
    for info in reading["rooms"]:
        region = info["final_room"]
        period = info["period"]
        if variant == "ignore_clipping":
            period = info["word"]
        elif variant == "keep_prefix":
            period = info["word"][:info["capacity"]]
        elif variant == "reverse":
            period = period[::-1]
        elif variant == "sort":
            period = tuple(sorted(period))
        elif variant == "sort_reverse":
            period = tuple(sorted(period, reverse=True))
        if variant == "ignore_tooth_boundary":
            depth = depth_function(info["room"])
        elif variant == "row_stripes":
            depth = np.indices(region.shape)[0] - reading["top"] - 1
        else:
            depth = depth_function(region)
        out[region] = np.asarray(period, dtype=np.int8)[depth[region] % len(period)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver. The comb's location supplies the direction and docking."""
    reading = _read(input_grid)
    return np.rot90(_print(reading), -reading["turn"]).copy()


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _geometry(rng, widths, depth, top, left, shape, rich, edge_lane=None):
    """Subdivide a housing, then bend dividers and grow attached wall fingers."""
    bottom = top + depth + 1
    right = left + sum(widths) + len(widths)
    wall = np.zeros(shape, dtype=bool)
    wall[top, left:right + 1] = wall[bottom, left:right + 1] = True
    wall[top:bottom + 1, left] = wall[top:bottom + 1, right] = True
    edges = [left]
    for width in widths:
        edges.append(edges[-1] + width + 1)
    columns = []
    for i, width in enumerate(widths):
        # Off-center rods make large contour lobes possible beside a tall tooth.
        candidates = list(range(edges[i] + 2, edges[i + 1] - 1))
        columns.append(int(rng.choice(candidates)))
    if rich == -1 or edge_lane is not None:
        lane = 0 if rich == -1 else edge_lane
        columns[lane] = edges[lane] + 2 if rng.integers(2) else edges[lane + 1] - 2
    for i, col in enumerate(edges[1:-1]):
        wall[top:bottom + 1, col] = True
        if rich > 0:
            options = [s for s in (-2, -1, 1, 2)
                       if columns[i] + 2 <= col + s <= columns[i + 1] - 2]
            if options:
                shift = int(rng.choice(options))
                a = int(rng.integers(top + 2, bottom - 3))
                b = int(rng.integers(a + 2, bottom - 1))
                low, high = sorted((col, col + shift))
                wall[a + 1:b, col] = False
                wall[a, low:high + 1] = wall[b, low:high + 1] = True
                wall[a:b + 1, col + shift] = True
    # Each finger stays at least two cells from every rod axis. It cannot
    # seal a room or touch another wall except at its single attachment.
    for _ in range(max(0, rich)):
        options = []
        for row in range(top + 2, bottom - 1):
            for col in range(left, right + 1):
                if not wall[row - 1:row + 2, col].all():
                    continue
                for direction in (-1, 1):
                    for length in (2, 3, 4):
                        stop = col + direction * length
                        if not left < stop < right:
                            continue
                        lo, hi = sorted((col + direction, stop))
                        if any(lo - 1 <= p <= hi + 1 for p in columns):
                            continue
                        if not np.any(wall[row - 1:row + 2, lo:hi + 1]):
                            options.append((row, lo, hi))
        if options:
            row, lo, hi = options[int(rng.integers(len(options)))]
            wall[row, lo:hi + 1] = True
    return wall, columns, bottom, right


def _make_pair(rng, n, colors, turn, mirror, witness=False, rich=0, query=False):
    bg, structure = colors
    top, left, bottom_margin, right_margin = [int(rng.integers(1, 3)) for _ in range(4)]
    gap = int(rng.integers(1, 4))
    depth = int(rng.integers(9, 12) if witness else rng.integers(6, 12 if query else 14))
    budget = 30 - left - right_margin - (n + 1)
    minima = [11, 7] if witness else [7] * n
    widths = minima.copy()
    remaining = budget - sum(widths)
    for i in rng.permutation(n):
        extra = int(rng.integers(0, min(remaining, 24 - widths[i]) + 1))
        widths[i] += extra
        remaining -= extra
    max_height = min(depth - 1, 30 - depth - top - gap - bottom_margin - 3)
    query_lane = int(rng.integers(n)) if query else None
    if witness:
        heights = [depth - 3, depth - 2]
    else:
        mode = int(rng.integers(4))
        heights = rng.integers(1, max_height + 1, n).tolist()
        if mode == 0:
            heights = [heights[0]] * n
        elif mode in (1, 2):
            heights.sort(reverse=mode == 2)
    if query:
        heights[query_lane] = depth - int(rng.integers(2, 4))
    external_base = top + depth + 1 + max(heights) + gap + 1
    shape = (external_base + bottom_margin + 1,
             left + sum(widths) + n + right_margin + 1)
    wall, columns, bottom, right = _geometry(rng, widths, depth, top, left, shape,
                                              -1 if witness else rich, query_lane)
    inp = np.full(shape, bg, dtype=np.int8)
    inp[wall] = structure
    inp[external_base, left:right + 1] = structure
    palette = [c for c in range(10) if c not in colors]
    words = []
    for i, (col, height) in enumerate(zip(columns, heights)):
        capacity = depth - height
        length = int(rng.integers(capacity, depth + 1))
        if witness and i == 0:
            prefix = int(rng.integers(1, 4))
            inks = [int(c) for c in rng.choice(palette, 4, replace=False)]
            if inks[:3] in (sorted(inks[:3]), sorted(inks[:3], reverse=True)):
                inks[0], inks[1] = inks[1], inks[0]
            word = [inks[3]] * prefix + inks[:3]
        elif witness:
            word = [int(c) for c in rng.choice(palette, 2, replace=False)]
        elif query and i == query_lane:
            prefix = int(rng.integers(1, min(3, height) + 1))
            inks = [int(c) for c in rng.choice(palette, capacity + 1, replace=False)]
            word = [inks[-1]] * prefix + inks[:-1]
        else:
            size = int(rng.integers(1, min(5, length) + 1))
            inks = rng.choice(palette, size, replace=False)
            word = [int(c) for c in rng.choice(inks, length)]
        words.append(word)
        inp[top + 1:top + len(word) + 1, col] = word
        inp[external_base - height:external_base, col] = structure
    # Labels are constructed directly from geometry and destination arithmetic.
    # The solver separately recognizes pixels and uses iterative erosion.
    inside = np.zeros(shape, dtype=bool)
    inside[top + 1:bottom, left + 1:right] = True
    out = np.full(shape, bg, dtype=np.int8)
    out[wall] = structure
    for room in _components(inside & ~wall):
        i = next(i for i, c in enumerate(columns) if room[top + 1, c])
        col, height = columns[i], heights[i]
        count = depth - height
        period = np.asarray(words[i][-count:], dtype=np.int8)
        room[bottom - height:bottom, col] = False
        depths = _bfs_depth(room)
        out[room] = period[depths[room] % count]
        out[bottom - height:bottom, col] = structure
    if mirror:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a reproducible puzzle with either lane-count or direction transfer."""
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    role_order = [int(c) for c in rng.permutation(10)]
    offsets = [int(c) for c in rng.permutation(np.arange(1, 10))]
    count_transfer = bool(rng.integers(2))
    train_turn = int(rng.integers(4))
    result = {"train": [], "test": []}
    for i in range(nt + ne):
        test = i >= nt
        n = 2 if i == 0 else (3 if test and count_transfer else int(rng.integers(1, 3 if not test else 4)))
        colors = (role_order[i % 10], role_order[(i % 10 + offsets[i // 10]) % 10])
        turn = (train_turn + int(rng.choice((1, 3)))) % 4 if test and not count_transfer else train_turn
        pair = _make_pair(rng, n, colors, turn, bool(rng.integers(2)), witness=i == 0,
                          rich=int(rng.integers(0, 4)) if i else 0, query=test)
        # A legal but over-indented query could have no visible inner layer.
        # One finite fallback removes optional walls, retaining the same rule,
        # split transfer and a fresh moving, multicolor query rod.
        if test and not _query_witness(_read(pair["input"])):
            pair = _make_pair(rng, n, colors, turn, bool(rng.integers(2)), rich=0, query=True)
        result["test" if test else "train"].append(pair)
    # Unique (background, structure) pairs guarantee distinct inputs, even for
    # the maximum 14 examples. Demonstration order has no semantic significance.
    rng.shuffle(result["train"])
    rng.shuffle(result["test"])
    return result


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _witnesses(reading):
    actual = _print(reading)
    result = {name: not np.array_equal(actual, _print(reading, variant=name))
              for name in ("ignore_clipping", "keep_prefix", "reverse", "sort", "sort_reverse",
                           "row_stripes", "ignore_tooth_boundary")}
    result["eight_neighbors"] = not np.array_equal(actual, _print(reading, lambda m: _erosion(m, False)))
    result["stationary"] = any(info["shift"] == 0 for info in reading["rooms"])
    result["lane_specific"] = len({info["shift"] for info in reading["rooms"]}) > 1
    result["not_tooth_height"] = any(0 < info["shift"] < info["height"] for info in reading["rooms"])
    result["repeat"] = any(len(set(info["period"])) >= 3 and
                           int(_erosion(info["final_room"]).max()) >= len(info["period"])
                           for info in reading["rooms"])
    return result


def _query_witness(reading):
    actual = _print(reading)
    return (any(info["shift"] > 0 and len(set(info["period"])) >= 2
                and int(_erosion(info["final_room"]).max()) >= 1 for info in reading["rooms"])
            and all(not np.array_equal(actual, _print(reading, variant=variant))
                    for variant in ("ignore_clipping", "row_stripes", "ignore_tooth_boundary")))


def validate_generated_puzzle(puzzle):
    """Check returned pixels, clipping/contact, contour semantics and evidence."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, readings, evidence = set(), {"train": [], "test": []}, {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape, "printing must preserve canvas dimensions")
            _check(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            reading = _read(inp)
            readings[split].append(reading)
            expected = np.rot90(_print(reading), -reading["turn"])
            _check(np.array_equal(out, expected), "output does not match input-only printing")
            _check(not np.array_equal(inp, out), "identity pair")
            _check(set(np.unique(out)).issubset(set(np.unique(inp))), "invented output color")
            canonical_out = np.rot90(out, reading["turn"])
            _check(bool(np.all(canonical_out[reading["wall"]] == reading["structure"])), "fixed wall changed")
            _check(bool(np.all(canonical_out[reading["comb"]] == reading["bg"])), "external comb not erased")
            for info in reading["rooms"]:
                depth = _erosion(info["final_room"])
                _check(np.array_equal(depth, _bfs_depth(info["final_room"])), "contour implementations disagree")
                # Explicitly simulate clipping the ordered rod one cell at a time.
                cells = list(info["word"])
                for _ in range(info["shift"]):
                    cells.pop(0)
                _check(tuple(cells) == info["period"] and len(cells) == info["capacity"], "ordered clipping mismatch")
            if split == "train":
                for name, value in _witnesses(reading).items():
                    evidence[name] = evidence.get(name, False) or value
            else:
                _check(_query_witness(reading), "query must exercise both contour printing and clipping")
    _check(all(evidence.values()), f"training lacks a distinguishing witness: {evidence}")
    train = readings["train"]
    _check(all(len(r["rooms"]) > max(len(t["rooms"]) for t in train)
               or r["turn"] not in {t["turn"] for t in train}
               for r in readings["test"]), "test must transfer chamber count or press direction")
    return None


def self_test():
    """Bounded 42-seed sweep, count boundaries, equivariance and corruptions.

    Only one puzzle at a time is retained. Diversity sets contain hashes and
    short feature tuples, not duplicate grids. Nothing is written to disk.
    """
    shapes, geometry, signatures, roles, counts, turns, periods, profiles = (set() for _ in range(8))
    shifts, nonrectangular, repeated, stationary = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic generation")
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                reading = _read(pair["input"])
                _check(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solver mismatch")
                shapes.add(pair["input"].shape)
                roles.add((reading["bg"], reading["structure"]))
                counts.add(len(reading["rooms"]))
                turns.add(reading["turn"])
                profiles.add(tuple(i["height"] for i in reading["rooms"]))
                geometry.add(hashlib.sha256(reading["wall"].tobytes()).hexdigest())
                for info in reading["rooms"]:
                    periods.add(len(info["period"]))
                    shifts.add(info["shift"])
                    stationary += info["shift"] == 0
                    repeated += len(set(info["word"])) < len(info["word"])
                    t, b, l, r = _bbox(info["room"])
                    nonrectangular += int(info["room"].sum()) != (b - t + 1) * (r - l + 1)
    _check(len(signatures) == 42 and len(shapes) >= 45 and len(geometry) >= 100, "insufficient geometric variation")
    _check(len(roles) >= 40 and counts == {1, 2, 3} and turns == {0, 1, 2, 3}, "missing role/count/orientation variation")
    _check(len(profiles) >= 60 and len(periods) >= 5 and len(shifts) >= 5, "missing mechanical variation")
    _check(stationary >= 42 and repeated >= 42 and nonrectangular >= 30, "missing contact/content/topology variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9173, nt, ne)
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    p = generate_puzzle(81, 2, 1)
    # The same input-only rule must commute with all whole-scene symmetries and
    # a nontrivial bijective recoloring, including reassignment of background.
    mapping = np.array([(3 * i + 7) % 10 for i in range(10)], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        for turn in range(4):
            for mirror in (False, True):
                a, b = mapping[pair["input"]], mapping[pair["output"]]
                if mirror:
                    a, b = np.fliplr(a), np.fliplr(b)
                _check(np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn)), "symmetry/recoloring mismatch")
    reading = _read(p["train"][0]["input"])
    for row, col in ((reading["top"], reading["left"] + 1),
                     (reading["external_base"], reading["left"]),
                     (reading["top"] + 1, reading["rooms"][0]["col"])):
        malformed = reading["grid"].copy()
        malformed[row, col] = reading["bg"]
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed wall/base/rod accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "wall_geometries": len(geometry), "role_pairs": len(roles),
            "chamber_counts": sorted(counts), "press_directions": len(turns),
            "profiles": len(profiles), "period_lengths": sorted(periods),
            "shifts": sorted(shifts), "stationary_rods": stationary,
            "repeated_color_rods": repeated, "nonrectangular_input_chambers": nonrectangular}
