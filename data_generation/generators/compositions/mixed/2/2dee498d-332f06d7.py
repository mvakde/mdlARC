"""Registered corridor compaction followed by full-footprint transport.

Only NumPy and the standard library are used. No parent generators, files,
generation metadata, or training examples are consulted by solve().
"""
from collections import deque
import hashlib
import json

import numpy as np


MOVER, GOAL, DIVIDER = 2, 3, 8
TAG_ROWS = 3
STRUCTURAL_COLORS = (0, 1, 4, 5, 6, 7, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _compact(a):
    # Compare the ENTIRE column, including registration and arena cells.
    seen, indices = set(), []
    for c in range(a.shape[1]):
        key = tuple(map(int, a[:, c]))
        if key not in seen:
            seen.add(key)
            indices.append(c)
    return a[:, indices].copy()


def _neighbors(p, shape):
    r, c = p
    for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _distance(mask, sources):
    d = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for source in sources:
        p = tuple(map(int, source))
        if mask[p] and d[p] < 0:
            d[p] = 0
            queue.append(p)
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q] and d[q] < 0:
                d[q] = d[p] + 1
                queue.append(q)
    return d


def _square(a, color):
    points = np.argwhere(a == color)
    _require(len(points) > 0, "missing endpoint marker")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    k = int(hi[0] - lo[0])
    _require(hi[1]-lo[1] == k and len(points) == k*k, "endpoint must be a solid square")
    return tuple(map(int, lo)), k


def _placements(allowed, k):
    return np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))


def _transport(arena):
    """Evaluate visible geometry; reject a tied stopping front instead of choosing one."""
    wall = int(arena[0, 0])
    _require(wall in STRUCTURAL_COLORS, "invalid wall color")
    _require(np.all(arena[0] == wall) and np.all(arena[-1] == wall)
             and np.all(arena[:, 0] == wall) and np.all(arena[:, -1] == wall),
             "arena needs a wall frame")
    remaining = set(map(int, np.unique(arena))) - {wall, MOVER, GOAL}
    _require(len(remaining) == 1, "arena must have four roles")
    floor = remaining.pop()
    _require(floor in STRUCTURAL_COLORS and floor != wall, "invalid floor color")
    start, k = _square(arena, MOVER)
    target, gk = _square(arena, GOAL)
    _require(k == gk, "endpoint sizes differ")
    allowed = arena != wall
    distance = _distance(allowed, np.argwhere(arena == GOAL))
    _require(np.all(distance[allowed] >= 0), "disconnected corridor")
    # A separate route around a wall island would make 'the first throat'
    # ambiguous. Placement cycles within a single wide chamber are fine.
    border = np.zeros(arena.shape, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    wall_distance = _distance(~allowed, np.argwhere(border))
    _require(np.all(wall_distance[~allowed] >= 0), "corridor has an alternate route")
    valid = _placements(allowed, k)
    reach = _distance(valid, [start])
    reachable = reach >= 0
    success = bool(reachable[target])
    destination = target
    if not success:
        windows = np.lib.stride_tricks.sliding_window_view(distance, (k, k))
        fronts = []
        for score in (windows.min(axis=(-1, -2)), windows.max(axis=(-1, -2)),
                      windows.sum(axis=(-1, -2))):
            best = np.argwhere(reachable & (score == score[reachable].min()))
            _require(len(best) == 1, "tied stopping front")
            fronts.append(tuple(map(int, best[0])))
        _require(len(set(fronts)) == 1, "inconsistent stopping front")
        destination = fronts[0]
    result = arena.copy()
    result[arena == MOVER] = floor
    r, c = destination
    result[r:r+k, c:c+k] = MOVER
    # Count bends on one shortest reachable placement route for diagnostics.
    p, headings = destination, []
    while p != start:
        q = next(q for q in _neighbors(p, valid.shape) if reach[q] == reach[p]-1)
        headings.append((q[0]-p[0], q[1]-p[1]))
        p = q
    info = {"wall": wall, "floor": floor, "k": k, "start": start,
            "target": target, "destination": destination, "success": success,
            "travel": int(reach[destination]),
            "turns": sum(a != b for a, b in zip(headings, headings[1:])),
            "placement_freedom": any(sum(bool(valid[q]) for q in _neighbors(tuple(p), valid.shape)) > 2
                                     for p in np.argwhere(reachable))}
    return result, info


def _decode(input_grid):
    a = _grid(input_grid)
    compact = _compact(a)
    _require(compact.shape[0] >= 8 and compact.shape[1] >= 4, "registered map too small")
    _require(np.all(compact[TAG_ROWS] == DIVIDER), "missing divider")
    arena = compact[TAG_ROWS+1:]
    wall = int(arena[0, 0])
    palette = set(map(int, np.unique(arena))) - {MOVER, GOAL}
    _require(len(palette) == 2 and wall in palette, "bad arena palette")
    _require(set(map(int, np.unique(compact[:TAG_ROWS]))) <= palette | {DIVIDER},
             "bad registration colors")
    return compact, arena


def solve(input_grid: np.ndarray) -> np.ndarray:
    compact, arena = _decode(input_grid)
    moved, _ = _transport(arena)
    result = compact.copy()
    result[TAG_ROWS+1:] = moved
    # Do not deduplicate a second time: moving may create newly equal columns.
    return result


def _registered(arena, rng):
    """Reuse tags across unequal terrain, distinguish equal terrain columns.

    The tags are unordered ternary words, not coordinates or movement hints.
    Comparing tags alone therefore loses legitimate distinct columns.
    """
    wall = int(arena[0, 0])
    floor = next(c for c in np.unique(arena) if c not in (wall, MOVER, GOAL))
    groups = {}
    for c in range(arena.shape[1]):
        groups.setdefault(tuple(map(int, arena[:, c])), []).append(c)
    capacity = max(2, max(map(len, groups.values())))
    _require(capacity <= 27, "registration capacity exceeded")
    codes = rng.choice(27, capacity, replace=False)
    tags = np.empty((TAG_ROWS, arena.shape[1]), dtype=np.int8)
    alphabet = (wall, int(floor), DIVIDER)
    for columns in groups.values():
        order = rng.permutation(codes)
        for c, code in zip(columns, order):
            for row in range(TAG_ROWS):
                tags[row, c] = alphabet[(int(code) // (3**row)) % 3]
    return np.vstack((tags, np.full((1, arena.shape[1]), DIVIDER, dtype=np.int8), arena))


def _paint(mask, position, size):
    r, c = position
    mask[r:r+size, c:c+size] = True


def _colored(mask, start, target, k, colors):
    wall, floor = colors
    a = np.full(mask.shape, wall, dtype=np.int8)
    a[mask] = floor
    for position, color in ((start, MOVER), (target, GOAL)):
        r, c = position
        _require(mask[r:r+k, c:c+k].all(), "trimmed marker")
        a[r:r+k, c:c+k] = color
    return a


def _masked_throat(rng, k, colors):
    """Joint geometry/encoding witness: one repeated column opens the raw U."""
    dx = int(rng.integers(k+1, 16-k+1))
    dy = int(rng.integers(3*k+4, 24-k+1))
    t = int(rng.integers(1, k))
    mask = np.zeros((dy+k+2, dx+k+2), dtype=bool)
    mask[1:1+k, 1:dx+k+1] = True
    mask[dy+1:dy+k+1, 1:dx+k+1] = True
    mask[k+1:dy+1, dx+1:dx+k+1] = True
    cut = int(rng.integers(2*k+2, dy-k-1))
    length = int(rng.integers(1, min(3, dy-k-1-cut)+1))
    mask[cut:cut+length, dx+t+1:dx+k+1] = False
    a = _colored(mask, (1, 1), (dy+1, 1), k, colors)
    repeated = dx+1
    if rng.integers(2):
        a = a[::-1].copy()
    if rng.integers(2):
        a = a[:, ::-1].copy()
        repeated = a.shape[1]-1-repeated
    return a, repeated, k-t


def _random_arena(rng, k, blocked, colors, width_limit, must_narrow=False):
    """Grow a fresh orthogonal walk, then paint rooms and variable passages."""
    q = k+1 if must_narrow else k + int(rng.random() < 0.45)
    gap = int(rng.integers(1, 4))
    step = q+gap
    nr_max = min(5, (24-q)//step+1)
    nc_max = min(5, (width_limit-2-q)//step+1)
    _require(nc_max >= 2, "not enough horizontal room")
    nr = int(rng.integers(1, nr_max+1))
    nc = int(rng.integers(2, nc_max+1))
    p = (int(rng.integers(nr)), int(rng.integers(nc)))
    walk, used = [p], {p}
    length = int(rng.integers(2, min(14, nr*nc)+1))
    while len(walk) < length:
        options = [p for p in _neighbors(walk[-1], (nr, nc)) if p not in used]
        if not options:
            break
        p = options[int(rng.integers(len(options)))]
        walk.append(p)
        used.add(p)
    _require(len(walk) >= 2, "walk is too short")
    path = [(1+r*step, 1+c*step) for r, c in walk]
    mask = np.zeros((2+(nr-1)*step+q, 2+(nc-1)*step+q), dtype=bool)
    for p in path:
        _paint(mask, p, q)
    first = int(rng.integers(len(path)-1)) if blocked else -1
    if blocked and first == 0:
        q_required = k
        _require(q == q_required, "wide initial stop lacks a collar approach")
    for i, (a, b) in enumerate(zip(path, path[1:])):
        low = 1 if blocked and i >= first else k
        width = int(rng.integers(low, q+1))
        if must_narrow and i == 0:
            width = k
        if i == first:
            width = int(rng.integers(1, k))
        offset = int(rng.integers(q-width+1))
        if i == first:
            offset = 0
        if a[0] == b[0]:
            left, right = sorted((a[1], b[1]))
            mask[a[0]+offset:a[0]+offset+width, left+q:right] = True
        else:
            top, bottom = sorted((a[0], b[0]))
            mask[top+q:bottom, a[1]+offset:a[1]+offset+width] = True
        if i == first and q > k:
            # Trim only the exit face into a k-wide collar. Earlier rooms
            # can remain wider than k, so the first width change need not stop.
            r, c = a
            if a[0] == b[0]:
                c = c+q-k if b[1] > c else c
                mask[r:r+q, c:c+k] = False
            else:
                r = r+q-k if b[0] > r else r
                mask[r:r+k, c:c+q] = False
            _paint(mask, (r, c), k)

    def terminal(a, b):
        r, c = a
        offset = int(rng.integers(q-k+1))
        if r == b[0]:
            return r+offset, c if c < b[1] else c+q-k
        return (r if r < b[0] else r+q-k), c+offset

    start, target = terminal(path[0], path[1]), terminal(path[-1], path[-2])
    a = _colored(mask, start, target, k, colors)
    if rng.integers(2):
        a = a[::-1].copy()
    if rng.integers(2):
        a = a[:, ::-1].copy()
    return a


def _encode_global(compact, rng, extra):
    w = compact.shape[1]
    indices = list(range(w))
    # Separated early duplicate: defeats a prefix crop and adjacent-run removal.
    indices.insert(int(rng.integers(2, w)), 0)
    # Later duplicates have no spatial meaning. A descending suffix also
    # differs from sorting or preserving last-occurrence order.
    tail = list(map(int, rng.choice(w, extra-1, replace=True)))
    if len(tail) >= 2:
        tail[-2:] = sorted(map(int, rng.choice(w, 2, replace=False)), reverse=True)
    indices.extend(tail)
    return compact[:, indices].copy()


def _make_pair(rng, k, blocked, colors, special=False, test=False, must_narrow=False):
    for attempt in range(100):
        try:
            if special:
                arena, column, extra = _masked_throat(rng, k, colors)
            else:
                arena = _random_arena(rng, k, blocked, colors, 22 if test else 20, must_narrow)
            moved, info = _transport(arena)
            _require(info["success"] != blocked and info["travel"] > 0, "wrong requested outcome")
            compact = _registered(arena, rng)
            if special:
                indices = list(range(compact.shape[1]))
                indices[column:column+1] = [column]*(extra+1)
                inp = compact[:, indices].copy()
                _, raw = _transport(inp[TAG_ROWS+1:])
                _require(raw["success"], "duplicate columns did not mask the throat")
            else:
                budget = min(30-compact.shape[1], 8 if test else 5)
                minimum = 6 if test else 3
                _require(budget >= minimum, "no duplication budget")
                extra = int(rng.integers(minimum, budget+1))
                inp = _encode_global(compact, rng, extra)
            out = compact.copy()
            out[TAG_ROWS+1:] = moved
            _require(np.array_equal(_compact(inp), compact), "registration collision")
            _require(np.array_equal(solve(inp), out), "construction/solution mismatch")
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded corridor construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    demo_k = 2 if regime == 0 else int(rng.integers(2, 4))
    specs = [(demo_k, False, False, True), (demo_k, True, True, False)]
    for _ in range(num_train-2):
        k = 2 if regime == 0 else int(rng.integers(1, 5))
        specs.append((k, bool(k > 1 and rng.integers(2)), False, False))
    rng.shuffle(specs)
    puzzle, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        if split == "test":
            specs = []
            for _ in range(num_test):
                k = int(rng.integers(3, 5)) if regime == 0 else (
                    demo_k if regime == 1 else int(rng.integers(1, 5)))
                specs.append((k, bool(k > 1 and rng.integers(2)), False, False))
        for k, blocked, special, must_narrow in specs:
            for _ in range(20):
                colors = tuple(map(int, rng.choice(STRUCTURAL_COLORS, 2, replace=False)))
                pair = _make_pair(rng, k, blocked, colors, special, split == "test", must_narrow)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    compact, arena = _decode(inp)
    moved, info = _transport(arena)
    _require(inp.shape[1] > compact.shape[1], "no redundant columns")
    _require(out.shape == compact.shape, "output must have compact dimensions")
    _require(np.array_equal(out[:TAG_ROWS+1], compact[:TAG_ROWS+1]), "registration was changed")
    _require(np.array_equal(out[TAG_ROWS+1:], moved), "incorrect movement or scenery")
    _require(info["travel"] > 0, "generated square did not move")
    _require(np.count_nonzero(out[TAG_ROWS+1:] == MOVER) == info["k"]**2, "mover not conserved")
    _require(not np.any((moved == MOVER) & (arena == info["wall"])), "wall collision")
    if not info["success"]:
        _require(np.array_equal(moved[arena == GOAL], arena[arena == GOAL]), "blocked goal changed")
    else:
        _require(not np.any(moved == GOAL), "reached goal not replaced")
    adjacent = inp[:, [0] + [c for c in range(1, inp.shape[1])
                              if not np.array_equal(inp[:, c], inp[:, c-1])]]
    tag_only_count = len({tuple(map(int, compact[:TAG_ROWS, c])) for c in range(compact.shape[1])})
    masked = False
    if not info["success"]:
        try:
            _, raw = _transport(inp[TAG_ROWS+1:])
            masked = raw["success"] and raw["k"] == info["k"]
        except AssertionError:
            pass
    info.update({"extra": inp.shape[1]-compact.shape[1], "shape": inp.shape,
                 "compact_shape": compact.shape, "masked": bool(masked),
                 "global_witness": not np.array_equal(adjacent, compact),
                 "crop_fails": not np.array_equal(inp[:, :compact.shape[1]], compact),
                 "tag_only_fails": tag_only_count < compact.shape[1]})
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _pair_info(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            features[split].append(info)
    train = features["train"]
    _require(any(v["success"] for v in train), "missing successful demonstration")
    _require(any(v["success"] and v["placement_freedom"] for v in train),
             "missing successful wide-room demonstration")
    _require(any(v["masked"] for v in train), "missing causally essential throat witness")
    _require(any(v["global_witness"] and v["crop_fails"] for v in train), "missing global dedup witness")
    _require(any(v["tag_only_fails"] for v in train), "registration alone predicts dedup")
    _require(min(v["extra"] for v in features["test"]) > max(v["extra"] for v in train),
             "missing redundancy train/test shift")


def self_test():
    """Bounded, streaming 42-seed study plus explicit counterfactual checks."""
    signatures, shapes, layouts, sizes, outcomes, turns, extras, palettes = (set() for _ in range(8))
    masked = global_witnesses = freedom = 0
    bare_layouts = set()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                pair_count += 1
                a = pair["input"]
                digest.update(bytes(a.shape) + a.tobytes() + pair["output"].tobytes())
                shapes.add(a.shape)
                compact, arena = _decode(a)
                normalized = compact.copy()
                normalized[compact == info["wall"]] = 0
                normalized[compact == info["floor"]] = 1
                layouts.add(_key(normalized))
                bare_layouts.add(_key(normalized[TAG_ROWS+1:]))
                sizes.add(info["k"])
                outcomes.add((split, info["success"]))
                turns.add(info["turns"])
                extras.add(info["extra"])
                palettes.add((info["wall"], info["floor"]))
                masked += info["masked"]
                global_witnesses += info["global_witness"]
                freedom += info["placement_freedom"]
                # Independent physical invariants; no label is used by solve.
                _require(np.array_equal(solve(a), pair["output"]), "solver disagreement")
                _require(not np.array_equal(compact, pair["output"]), "omitting transport has no effect")
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42 and len(layouts) >= 150, "insufficient cross-seed variation")
    _require(len(bare_layouts) >= 150, "registration hides repeated arena layouts")
    _require(len(shapes) >= 30 and sizes == {1, 2, 3, 4}, "missing geometric scales")
    _require(outcomes == {(s, b) for s in ("train", "test") for b in (False, True)}, "fixed outcomes")
    _require(masked >= 42 and global_witnesses >= 80, "missing parent-necessity evidence")
    _require(max(turns) >= 4 and freedom >= 10, "routes lack turns or wider chambers")
    _require(len(extras) >= 6 and len(palettes) >= 20, "encoding or role variation collapsed")
    for n, m in ((2, 1), (10, 4), (7, 3)):
        a, b = generate_puzzle(117, n, m), generate_puzzle(117, n, m)
        _require(all(np.array_equal(x[role], y[role]) for s in a for x, y in zip(a[s], b[s])
                     for role in ("input", "output")), "nondeterministic puzzle")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Hand-specified geometry: the first forbidden row is 8. A 3-square
    # therefore stops with top-left (5,10), before that row enters its footprint.
    rng = np.random.default_rng(700)
    mask = np.zeros((19, 14), dtype=bool)
    mask[1:5, 1:13] = mask[15:18, 1:13] = True
    mask[4:15, 10:13] = True
    mask[8:10, 11:13] = False
    arena = _colored(mask, (1, 1), (15, 1), 3, (0, 1))
    column, extra = 10, 2
    moved, info = _transport(arena)
    expected = arena.copy()
    expected[1:4, 1:4] = 1
    expected[5:8, 10:13] = MOVER
    _require(info["destination"] == (5, 10) and np.array_equal(moved, expected),
             "hand-specified first-throat destination failed")
    compact = _registered(arena, rng)
    indices = list(range(compact.shape[1]))
    indices[column:column+1] = [column]*(extra+1)
    encoded = compact[:, indices]
    _require(not _transport(arena)[1]["success"] and _transport(encoded[4:])[1]["success"],
             "masked-width counterfactual failed")
    # A 1-cell footprint follows the same connected corridor all the way.
    a = arena.copy()
    start, _ = _square(a, MOVER)
    target, _ = _square(a, GOAL)
    a[(a == MOVER) | (a == GOAL)] = 1
    a[start], a[target] = MOVER, GOAL
    _require(_transport(a)[1]["success"], "point transport still blocks")
    # Negative labels must be rejected, including untouched registration cells.
    puzzle = generate_puzzle(0, 2, 1)
    for where in ("registration", "mover"):
        bad = {s: [{r: a.copy() for r, a in p.items()} for p in puzzle[s]] for s in puzzle}
        pair = bad["train"][0]
        point = (0, 0) if where == "registration" else tuple(np.argwhere(pair["output"] == MOVER)[0])
        pair["output"][point] = (int(pair["output"][point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("validator accepted a corrupted " + where)
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "registered_structures": len(layouts),
            "bare_arena_structures": len(bare_layouts),
            "mover_sizes": sorted(sizes), "reachable_turn_counts": sorted(turns),
            "masked_throat_pairs": masked, "global_dedup_pairs": global_witnesses,
            "wide_placement_pairs": freedom, "structural_palettes": len(palettes),
            "extra_column_counts": sorted(extras), "independent_semantic_oracle": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
