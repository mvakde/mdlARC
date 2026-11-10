"""Chromatic clockwork ribbon: the next periodic slice steers at neutral walls.

Only NumPy and the standard library are used. No files are read or written by
generation, solving, validation, or self_test; all diagnostic traces stay in RAM.
"""

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(np.all((0 <= a) & (a <= 9)), "bad color")
    return a


def _signature(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _counts(a, b):
    for name, value, lo, hi in (("num_train", a, 2, 10), ("num_test", b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(name + " out of range")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    answer = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        answer.append(cells)
    return answer


def _box(cells):
    r, c = np.array(list(cells)).T
    return int(r.min()), int(c.min()), int(r.max() + 1), int(c.max() + 1)


def _hub_mask(h, w, kind):
    mask = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    elif kind == "cross":
        mask[:] = False
        mask[1:-1, :] = True
        arm = 1 if w % 2 else 2
        mask[:, (w-arm)//2:(w+arm)//2] = True
    return mask


def _slice(box, d, k, distance):
    t, l, b, e = box
    if d == 0:
        return [(t - distance, l + (e-l-k)//2 + j) for j in range(k)]
    if d == 1:
        return [(t + (b-t-k)//2 + j, e - 1 + distance) for j in range(k)]
    if d == 2:
        return [(b - 1 + distance, l + (e-l-k)//2 + j) for j in range(k)]
    return [(t + (b-t-k)//2 + j, l - distance) for j in range(k)]


def _decode(input_grid):
    grid = _grid(input_grid)
    wall = int(grid[0, 0])
    _require(np.all(grid[0] == wall), "missing top key frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == wall, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom key frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "bad key height")
    row = grid[1]
    _require(np.all(grid[1:bottom] == row), "key rows disagree")
    _require(row[0] == row[-1] == wall, "missing side frame")
    runs, j = [], 0
    while j < len(row):
        if row[j] == wall:
            j += 1
            continue
        start = j
        while j < len(row) and row[j] != wall:
            j += 1
        runs.append((start, j))
    _require(2 <= len(runs) <= 4 and all(b-a >= 3 for a, b in runs), "bad panels")
    bg = int(row[runs[0][0] + 1])
    turns = {}
    for a, b in runs:
        _require(np.all(row[a+1:b-1] == bg), "panel interiors must be background")
        left, right = int(row[a]), int(row[b-1])
        _require((left == bg) != (right == bg), "ambiguous side bar")
        pigment, turn = (right, 1) if left == bg else (left, -1)
        _require(pigment not in (bg, wall) and pigment not in turns, "conflicting key")
        turns[pigment] = turn
    field = grid[bottom+1:]
    other = set(map(int, np.unique(field))) - {bg, wall} - set(turns)
    _require(len(other) == 1, "hub color must be unique")
    hub = other.pop()
    cells = set(map(tuple, np.argwhere(field == hub)))
    box = _box(cells)
    t, l, b, e = box
    _require(3 <= b-t <= 7 and 3 <= e-l <= 7, "bad hub size")
    mask = field[t:b, l:e] == hub
    kinds = [kind for kind in ("solid", "clipped", "cross")
             if (np.array_equal(mask, _hub_mask(b-t, e-l, kind))
                 or np.array_equal(mask, _hub_mask(e-l, b-t, kind).T))]
    _require(kinds, "unrecognized hub")
    candidates = []
    for d in range(4):
        transverse = e-l if d % 2 == 0 else b-t
        for k in range(1, 4):
            if transverse < k+2 or (transverse-k) % 2:
                continue
            points = _slice(box, d, k, 1)
            if not all(0 <= r < field.shape[0] and 0 <= c < field.shape[1] for r, c in points):
                continue
            dr, dc = DIRS[d]
            if not all((r-dr, c-dc) in cells and int(field[r, c]) in turns for r, c in points):
                continue
            # Every payload cell is part of one complete rectangular seed word.
            word, seed_cells = [], set()
            for distance in range(1, 31):
                points = _slice(box, d, k, distance)
                if not all(0 <= r < field.shape[0] and 0 <= c < field.shape[1] for r, c in points):
                    break
                values = [int(field[p]) for p in points]
                if all(v in turns for v in values):
                    if len(set(values)) != 1:
                        break
                    word.append(values[0])
                    seed_cells.update(points)
                else:
                    break
            all_payload = set(map(tuple, np.argwhere(np.isin(field, list(turns)))))
            if seed_cells == all_payload and len(word) >= k:
                candidates.append((d, k, word, seed_cells))
    _require(len(candidates) == 1, "seed attachment or width is ambiguous")
    d, k, word, seed_cells = candidates[0]
    last = _slice(box, d, k, len(word))
    # The anchor is the trailing k-by-k square, not the final single slice.
    if d == 0:
        r, c = min(p[0] for p in last), min(p[1] for p in last)
    elif d == 1:
        r, c = min(p[0] for p in last), last[0][1]-k+1
    elif d == 2:
        r, c = last[0][0]-k+1, min(p[1] for p in last)
    else:
        r, c = min(p[0] for p in last), min(p[1] for p in last)
    walls = _components(field == wall)
    for component in walls:
        a, z, b, x = _box(component)
        _require(len(component) == (b-a)*(x-z), "walls must be solid rectangles")
    return dict(field=field, bg=bg, wall=wall, hub=hub, box=box, kind=kinds[0],
                turns=turns, word=word, seed_cells=seed_cells, start=(r, c, d), k=k,
                header=bottom+1, panel_widths=tuple(b-a for a, b in runs),
                walls=walls)


def _front(r, c, d, k):
    if d == 0:
        return [(r-1, c+j) for j in range(k)]
    if d == 1:
        return [(r+j, c+k) for j in range(k)]
    if d == 2:
        return [(r+k, c+j) for j in range(k)]
    return [(r+j, c-1) for j in range(k)]


def _trace(info, mode="normal"):
    """Cell-step semantics. Modes are explicit ablations used only by checks."""
    field, k = info["field"], info["k"]
    bg, wall, turns = info["bg"], info["wall"], info["turns"]
    word = list(info["word"])
    length = len(word)
    if mode == "prefix":
        p = next(p for p in range(1, length+1)
                 if all(v == word[i % p] for i, v in enumerate(word)))
        word = word[:p]
    phase = length % len(word)
    r, c, d = info["start"]
    output = field.copy()
    seen, written = set(), set()
    events, moves = [], []
    hits = set()
    crossings, overwrites = 0, 0
    for _ in range(4 * field.size * len(word) + 1):
        state = r, c, d, phase
        _require(state not in seen, "route cycles")
        seen.add(state)
        points = _front(r, c, d, k)
        if not all(0 <= x < field.shape[0] and 0 <= y < field.shape[1] for x, y in points):
            return output, dict(events=events, moves=moves, written=written, crossings=crossings,
                                overwrites=overwrites, hit_walls=hits, exit_heading=d, final_phase=phase)
        values = [int(field[p]) for p in points]
        pigment = word[phase]
        if wall in values:
            _require(all(v == wall for v in values), "partial face collision")
            components = [i for i, cells in enumerate(info["walls"]) if set(points) <= cells]
            _require(len(components) == 1, "simultaneous different wall components")
            if mode == "stop":
                return output, dict(events=events, moves=moves, written=written, crossings=crossings,
                                    overwrites=overwrites, hit_walls=hits, exit_heading=d, final_phase=phase)
            choice = word[0] if mode == "first" else word[(phase-1) % len(word)] if mode == "last" else pigment
            side = -1 if mode == "left" else 1 if mode == "right" else turns[choice]
            nd = (d+side) % 4
            events.append((len(moves), pigment, d, nd, r, c, phase, components[0]))
            hits.add(components[0])
            d = nd
            if mode == "reset":
                phase = 0
            elif mode == "tick":
                phase = (phase+1) % len(word)
            continue
        _require(all(v == bg for v in values), "route returns to hub or original seed")
        crossings += bool(set(points) & written)
        overwrites += sum(p in written and int(output[p]) != pigment for p in points)
        for p in points:
            output[p] = pigment
        written.update(points)
        moves.append((r, c, d, pigment, phase))
        dr, dc = DIRS[d]
        r, c = r+dr, c+dc
        phase = (phase+1) % len(word)
    raise AssertionError("trace bound exceeded")


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role, instruction, seed slice and phase from this input."""
    return _trace(_decode(input_grid))[0]


def _header(rng, width, bg, wall, turns, height):
    sizes = np.full(len(turns), 5, dtype=int)
    sizes += rng.multinomial(width-int(sizes.sum()), np.full(len(sizes), 1/len(sizes)))
    band = np.full((height, width), bg, dtype=np.int8)
    band[0] = band[-1] = wall
    c = 0
    for pigment, size in zip(rng.permutation(list(turns)), sizes):
        size = int(size)
        band[:, c] = band[:, c+size-1] = wall
        band[1:-1, c+1 if turns[int(pigment)] == -1 else c+size-2] = pigment
        c += size
    return band


def _info_for(field, box, k, word, turns):
    t, l, b, e = box
    length = len(word)
    return dict(field=field, bg=0, wall=1, hub=2, box=box, k=k, word=list(word),
                turns=turns, seed_cells={(r, c) for c in range(e, e+length)
                                        for r in range(t+(b-t-k)//2, t+(b-t+k)//2)},
                start=(t+(b-t-k)//2, e+length-k, 1), walls=_components(field == 1))


def _base_scene(rng, k, word, panel_count):
    hh = int(rng.integers(3, 6))
    # Both axes fit a legible key and leave room for its height after rotation.
    h = int(rng.integers(max(13, k+8, 5*panel_count), 31-hh))
    w = int(rng.integers(max(16, 5*panel_count), 31-hh))
    hub_h = k+2*int(rng.integers(1, 3))
    hub_w = int(rng.integers(3, 7))
    t = int(rng.integers(1, h-hub_h))
    l = int(rng.integers(0, max(1, w-hub_w-len(word)-4)))
    box = t, l, t+hub_h, l+hub_w
    kind = str(rng.choice(("solid", "clipped", "cross")))
    field = np.zeros((h, w), dtype=np.int8)
    region = field[t:t+hub_h, l:l+hub_w]
    region[_hub_mask(hub_h, hub_w, kind)] = 2
    for i, pigment in enumerate(word, 1):
        for p in _slice(box, 1, k, i):
            field[p] = pigment
    return field, box, hh


def _grow(rng, field, box, k, word, turns, target):
    """Insert rectangles on actual routes; no whole-puzzle parent generators."""
    info = _info_for(field, box, k, word, turns)
    output, trace = _trace(info)
    for _ in range(160):
        if len(trace["events"]) >= target and len(trace["hit_walls"]) >= min(target, 3):
            return info, output, trace
        candidates = [m for i, m in enumerate(trace["moves"])
                      if i >= 2 and (i < 2 or trace["moves"][i-1][2] == m[2])]
        if not candidates:
            break
        r, c, d, _, _ = candidates[int(rng.integers(len(candidates)))]
        points = _front(r, c, d, k)
        # A blocker occupies the next face, with independently sampled depth
        # and transverse extent. The current square remains a clear turn corner.
        depth = int(rng.integers(1, 4))
        lo, hi = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        if d == 0:
            a, b, x, y = r-depth, r, c-lo, c+k+hi
        elif d == 1:
            a, b, x, y = r-lo, r+k+hi, c+k, c+k+depth
        elif d == 2:
            a, b, x, y = r+k, r+k+depth, c-lo, c+k+hi
        else:
            a, b, x, y = r-lo, r+k+hi, c-depth, c
        a, b = max(0, a), min(field.shape[0], b)
        x, y = max(0, x), min(field.shape[1], y)
        if a >= b or x >= y or not set(points) <= {(u, v) for u in range(a, b) for v in range(x, y)}:
            continue
        if np.any(field[a:b, x:y] != 0):
            continue
        if np.any(field[max(0, a-1):min(field.shape[0], b+1),
                        max(0, x-1):min(field.shape[1], y+1)] == 1):
            continue
        trial = field.copy()
        trial[a:b, x:y] = 1
        ti = _info_for(trial, box, k, word, turns)
        try:
            to, tt = _trace(ti)
        except AssertionError:
            continue
        # Keep previous walls meaningful, without requiring all walls to be hit
        # in the general solver. Growing a real continuation avoids radial layouts.
        if len(tt["hit_walls"]) < len(ti["walls"]) or len(tt["events"]) <= len(trace["events"]):
            continue
        field, info, output, trace = trial, ti, to, tt
    return info, output, trace


def _different(info, output, mode):
    try:
        return not np.array_equal(_trace(info, mode)[0], output)
    except AssertionError:
        return True


def _witness(info, output, trace):
    sides = {info["turns"][e[1]] for e in trace["events"]}
    return (len(trace["events"]) >= 2 and sides == {-1, 1}
            and any(e[6] != 0 for e in trace["events"])
            and all(_different(info, output, mode)
                    for mode in ("first", "last", "left", "right", "reset", "tick", "stop", "prefix")))


def _make_pair(rng, k, period, target, rotation, teaching=False, allow_crossing=True):
    best = None
    for _ in range(100):
        count = int(rng.integers(2, min(4, period)+1))
        pigments = list(range(3, 3+count))
        # ABA is intentionally a complete word, not a truncated AB cycle.
        if teaching:
            word = [3, 4, 3]
        else:
            word = list(map(int, rng.choice(pigments, size=period)))
            word[0], word[1] = 3, 4
        turns = {p: int(rng.choice((-1, 1))) for p in pigments}
        turns[3], turns[4] = -1, 1
        if rng.random() < .5:
            turns = {p: -s for p, s in turns.items()}
        field, box, hh = _base_scene(rng, k, word, len(turns))
        info, output, trace = _grow(rng, field, box, k, word, turns, target)
        if len(trace["events"]) < target:
            continue
        if teaching and not _witness(info, output, trace):
            continue
        if not allow_crossing and trace["crossings"]:
            continue
        if target <= 3 and len(trace["events"]) > 3:
            continue
        best = info, output, hh
        break
    if best is None:
        # A finite, constructive escape path: scan simple alternating staircases.
        # This is also exercised directly by self_test, not only on random failure.
        info, output, hh = _staircase(rng, k, teaching, period, target)
    else:
        info, output, hh = best
    field = np.ascontiguousarray(np.rot90(info["field"], rotation))
    output = np.ascontiguousarray(np.rot90(output, rotation))
    # Rotation changes neither the meaning of relative L/R nor elapsed phase.
    turns = info["turns"]
    header = _header(rng, field.shape[1], 0, 1, turns, hh)
    raw_input = np.vstack((header, field))
    palette = rng.permutation(10).astype(np.int8)
    inp = palette[raw_input]
    out = palette[output]
    _require(np.array_equal(solve(inp), out), "construction and input reconstruction disagree")
    return {"input": inp, "output": out}


def _staircase(rng, k, teaching, period=3, target=2):
    """Finite escape construction with an alternating east/south staircase.

    Choose gaps by residue so that the next pigment alternates R/L at walls.
    For all generated k/period/target combinations the staircase fits 27x27.
    """
    field = np.zeros((27, 27), dtype=np.int8)
    box = (4, 1, 4+k+2, 4)
    field[4:4+k+2, 1:4] = 2
    word, turns = [3, 4]+[3]*(period-2), {3: 1, 4: -1}
    for i, color in enumerate(word, 1):
        for p in _slice(box, 1, k, i):
            field[p] = color
    info = _info_for(field, box, k, word, turns)
    r, c, d = info["start"]
    elapsed = 0
    for i in range(target):
        side = 1 if i % 2 == 0 else -1
        minimum = 2 if i == 0 else k+1
        gap = next(g for g in range(minimum, minimum+period)
                   if turns[word[(elapsed+g) % period]] == side)
        dr, dc = DIRS[d]
        r, c = r+dr*gap, c+dc*gap
        elapsed += gap
        points = _front(r, c, d, k)
        _require(all(0 <= x < 27 and 0 <= y < 27 and field[x, y] == 0 for x, y in points),
                 "fallback geometry does not fit")
        for p in points:
            field[p] = 1
        d = (d+side) % 4
    info = _info_for(field, box, k, word, turns)
    output, trace = _trace(info)
    _require(not teaching or _witness(info, output, trace), "fallback must teach all ablations")
    return info, output, 3


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("deeper", "wider", "longer")))
    base_rotation = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            teaching = split == "train" and i < 2
            if teaching:
                k, period, target = i+1, 3, 2
            elif split == "train":
                k, period, target = int(rng.integers(1, 3)), int(rng.integers(2, 5)), int(rng.integers(1, 4))
            else:
                k = 3 if regime == "wider" else int(rng.integers(1, 3))
                period = 5 if regime == "longer" else int(rng.integers(3, 5))
                target = int(rng.integers(4, 6)) if regime == "deeper" else int(rng.integers(2, 5))
            for attempt in range(24):
                taught_crossing = any(_trace(_decode(p["input"]))[1]["overwrites"]
                                      for p in puzzle["train"])
                pair = _make_pair(rng, k, period, target, (base_rotation+i) % 4, teaching,
                                  allow_crossing=split == "train" or taught_crossing)
                key = _signature(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError("bounded uniqueness space exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train, test = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            expected, trace = _trace(info)
            _require(np.array_equal(out, expected), "wrong ribbon or crop")
            field = info["field"]
            _require(out.shape == field.shape, "output must omit exactly the key")
            protected = field != info["bg"]
            _require(np.array_equal(out[protected], field[protected]), "changed initial material")
            changed = set(map(tuple, np.argwhere(out != field)))
            _require(changed == trace["written"] and changed, "changed cells must be precisely the extension")
            _require(trace["events"], "generated pair should exercise routing")
            # Collisions do not consume a color: event phase is the total number
            # of visible and generated slices modulo the complete word length.
            for event in trace["events"]:
                steps, pigment, d, nd, _, _, phase, _ = event
                _require(phase == steps % len(info["word"]), "wrong phase at wall")
                _require(pigment == info["word"][phase], "wrong next pigment")
                _require(nd == (d+info["turns"][pigment]) % 4, "turn violates visible key")
            if split == "train":
                train.append((info, expected, trace))
            else:
                test.append((info, expected, trace))
    _require({q[0]["k"] for q in train} >= {1, 2}, "teach thickness from the visible strip")
    _require(sum(_witness(*q) for q in train) >= 2, "two demonstrations must distinguish the shortcuts")
    if any(q[2]["crossings"] for q in test):
        _require(any(q[2]["overwrites"] for q in train), "teach a visible chronological overwrite")
    tk = max(q[0]["k"] for q in train)
    tp = max(len(q[0]["word"]) for q in train)
    td = max(len(q[2]["events"]) for q in train)
    _require(all(q[0]["k"] > tk or len(q[0]["word"]) > tp or len(q[2]["events"]) > td for q in test),
             "test must extend demonstrated width, complete word length, or route depth")


def family_features(pair):
    info = _decode(pair["input"])
    out, trace = _trace(info)
    events = trace["events"]
    wall_shapes = tuple(sorted((b-a, y-x) for a, x, b, y in map(_box, info["walls"])))
    return dict(shape=tuple(info["field"].shape), k=info["k"], word=tuple(info["word"]),
                period=len(info["word"]), direction=info["start"][2], hub_kind=info["kind"],
                hub_shape=(info["box"][2]-info["box"][0], info["box"][3]-info["box"][1]),
                wall_count=len(info["walls"]), wall_shapes=wall_shapes,
                turns=len(events), sides=tuple(sorted({info["turns"][e[1]] for e in events})),
                collision_phases=tuple(e[6] for e in events), crossings=trace["crossings"],
                overwrites=trace["overwrites"],
                length=len(trace["moves"]), header=info["header"], panels=len(info["turns"]),
                background=info["bg"], heading_after_turns=tuple(e[3] for e in events),
                witness=_witness(info, out, trace))


def self_test():
    """42 seeds, all legal count pairs, determinism and semantic mutations.

    Puzzles are released per iteration; no accumulated grids or disk artifacts.
    Returns small measured feature sets, suitable for compact JSON reporting.
    """
    features = {name: set() for name in ("shape", "k", "period", "direction", "hub_kind", "hub_shape",
                                       "wall_count", "wall_shapes", "turns", "header", "panels", "background")}
    digests, witnesses, crossings, pairs = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = family_features(pair)
                pairs += 1
                witnesses += f["witness"]
                crossings += f["crossings"] > 0
                for name in features:
                    features[name].add(f[name])
                for role in ("input", "output"):
                    a = pair[role]
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "nested input failed")
        digests.add(digest.hexdigest())
    _require(len(digests) == 42, "seeds must vary whole puzzles")
    for name, minimum in (("shape", 30), ("k", 3), ("period", 4), ("direction", 4),
                          ("hub_kind", 3), ("wall_count", 3), ("turns", 4), ("background", 10)):
        _require(len(features[name]) >= minimum, "missing structural diversity: " + name)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1700+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "count request ignored")
    first, second = generate_puzzle(912, 2, 1), generate_puzzle(912, 2, 1)
    _require(all(np.array_equal(p[r], q[r]) for s in first for p, q in zip(first[s], second[s])
                 for r in ("input", "output")), "seed determinism failed")
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Check every legal fallback geometry, including held-out widths/lengths.
    for k in (1, 2, 3):
        for period in range(max(2, k), 6):
            for target in range(1, 6):
                info, output, _ = _staircase(np.random.default_rng(k), k, False, period, target)
                _, trace = _trace(info)
                _require(len(trace["events"]) == target, "fallback route depth was lost")
        _staircase(np.random.default_rng(k), k, True)
    # Hand-calculated clock: two rightward cells, two downward cells, then east.
    field = np.zeros((10, 14), dtype=np.int8)
    field[3:6, 0:3] = 2
    field[4, 3:6] = [3, 4, 3]
    field[4, 8] = field[7, 7] = 1
    expected = field.copy()
    for point, pigment in (((4, 6), 3), ((4, 7), 4), ((5, 7), 3), ((6, 7), 3),
                           ((6, 8), 4), ((6, 9), 3), ((6, 10), 3), ((6, 11), 4),
                           ((6, 12), 3), ((6, 13), 3)):
        expected[point] = pigment
    header = _header(np.random.default_rng(17), 14, 0, 1, {3: 1, 4: -1}, 3)
    inp = np.vstack((header, field))
    _require(np.array_equal(solve(inp), expected), "hand-calculated phase/turn regression failed")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("output mutation accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(digests),
                feature_counts={name: len(values) for name, values in features.items()},
                witnesses=witnesses, self_crossing_pairs=crossings)
