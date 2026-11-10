"""A stamp-carrying trail: unary obstacle clearance, half-turn docks and sampled ink.

Only standard-library modules and NumPy are used. No files are read by the
generator or solver. Coordinates in private helpers use downward travel; a
quarter-turn presents any of the four boundary directions.
"""
from collections import deque
import hashlib
import json

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = deque([start]), []
        while todo:
            r, c = todo.popleft()
            cells.append((int(r), int(c)))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        components.append(cells)
    return components


def _decode(input_grid):
    """Parse visible components and commands, without any generation metadata."""
    original = _grid(input_grid)
    h, w = original.shape
    corners = [int(original[r, c]) for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))]
    _require(len(set(corners)) == 1, "background corners disagree")
    bg = corners[0]
    candidates = []
    for cells in _components(original != bg):
        rs, cs = zip(*cells)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if len(cells) != 9 or (r1 - r0, c1 - c0) != (2, 2):
            continue
        edges = [r0 == 0, c0 == 0, r1 == h - 1, c1 == w - 1]
        if sum(edges) == 1:
            # np.rot90(downward, k): top, left, bottom, right.
            candidates.append(edges.index(True))
    _require(len(candidates) == 1, "expected one boundary stamp")
    k = candidates[0]
    a = np.rot90(original, -k).copy()
    height, width = a.shape
    front_cells = np.argwhere(a[:3] != bg)
    _require(len(front_cells) == 9, "malformed source stamp")
    left = int(front_cells[:, 1].min())
    x = left + 1
    _require(1 <= left and left + 3 < width, "source must avoid canvas corners")
    tile = a[:3, left:left + 3].copy()
    _require(bool(np.all(tile != bg)), "source stamp has a hole")
    source_colors, source_counts = np.unique(tile, return_counts=True)
    _require(int(source_counts.max()) >= 5, "stamp carrier must be a majority")
    carrier = int(source_colors[source_counts.argmax()])
    _require(int(tile[0, 1]) != int(tile[2, 1]), "opposed signal cells must differ")
    _require(carrier not in (int(tile[0, 1]), int(tile[2, 1])), "signal must be a detail")
    tail = a[3:]
    values, counts = np.unique(tail[tail != bg], return_counts=True)
    _require(len(values) == 2, "bars must use two colors")
    bar = int(values[counts.argmax()])
    marker = int(values[counts.argmin()])
    _require(counts.max() > counts.min(), "ambiguous bar colors")
    _require(not ({bar, marker} & set(map(int, source_colors))), "bar and stamp roles overlap")
    rows = list(map(int, np.flatnonzero(np.any(tail != bg, axis=1)) + 3))
    _require(1 <= len(rows) <= 3, "expected one to three gates")
    gates, old_center, current = [], 1, x
    for f in rows:
        cols = np.flatnonzero(a[f] != bg)
        lo, hi = int(cols[0]), int(cols[-1])
        _require(len(cols) == hi - lo + 1, "bar must be contiguous")
        marks = tuple(map(int, np.flatnonzero(a[f] == marker)))
        n = len(marks)
        _require(1 <= n <= 4 and hi - lo + 1 >= 2 * n + 1, "invalid unary bar")
        low = marks == tuple(lo + 2 * j for j in range(n))
        high = marks == tuple(hi - 2 * j for j in reversed(range(n)))
        _require(low != high, "markers must identify one endpoint")
        d, side = n + 1, (1 if low else -1)
        q = hi + d if low else lo - d
        turn, dock = f - d, f + d
        _require(lo <= current <= hi, "gate misses incoming ray")
        _require(1 <= q <= width - 2, "docked stamp exceeds lateral boundary")
        _require(turn >= old_center + 3, "insufficient approach to gate")
        _require(dock <= height - 4, "insufficient exit after dock")
        gates.append(dict(row=f, lo=lo, hi=hi, n=n, d=d, side=side,
                          turn=turn, dock=dock, target=q, incoming=current))
        old_center, current = dock, q
    return dict(a=a, k=k, background=bg, bar=bar, marker=marker, carrier=carrier,
                tile=tile, initial=x, gates=gates)


def _render(info, *, count_offset=1, side_flip=False, operation="halfturn", ink="sample"):
    """Slice-based solver; optional alternatives are only used by validation."""
    a, tile = info["a"], info["tile"].copy()
    out = a.copy()
    out[out == info["marker"]] = info["bar"]
    old, current = 1, info["initial"]
    for g in info["gates"]:
        d = g["n"] + count_offset
        side = -g["side"] if side_flip else g["side"]
        q = g["hi"] + d if side > 0 else g["lo"] - d
        turn, dock = g["row"] - d, g["row"] + d
        if not (1 <= q <= a.shape[1] - 2 and turn >= old + 2 and dock <= a.shape[0] - 3):
            return None
        color = int(tile[2, 1]) if ink == "sample" else info["carrier"]
        out[old + 2:turn + 1, current] = color
        out[turn, min(current, q):max(current, q) + 1] = color
        out[turn:dock - 1, q] = color
        if operation == "halfturn":
            tile = tile[::-1, ::-1].copy()
        elif operation == "row_flip":
            tile = tile[::-1].copy()
        elif operation == "col_flip":
            tile = tile[:, ::-1].copy()
        elif operation != "identity":
            raise ValueError("unknown stamp operation")
        out[dock - 1:dock + 2, q - 1:q + 2] = tile
        old, current = dock, q
    color = int(tile[2, 1]) if ink == "sample" else info["carrier"]
    out[old + 2:, current] = color
    return np.rot90(out, info["k"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every color, heading, distance, dock and tile from this input only."""
    return _render(_decode(input_grid))


def _options(width, current, n, side):
    d, result = n + 1, []
    for lo in range(current + 1):
        for hi in range(max(current, lo + 2 * n), width):
            q = hi + d if side > 0 else lo - d
            if 1 <= q <= width - 2:
                result.append((lo, hi, q))
    return result


def _spans(rng, width, ns, sides):
    failed = set()

    def search(i, current):
        if i == len(ns):
            return []
        if (i, current) in failed:
            return None
        options = _options(width, current, ns[i], sides[i])
        for j in rng.permutation(len(options)):
            lo, hi, q = options[int(j)]
            rest = search(i + 1, q)
            if rest is not None:
                return [(lo, hi, q)] + rest
        failed.add((i, current))
        return None

    for x in rng.permutation(np.arange(2, width - 2)):
        spans = search(0, int(x))
        if spans is not None:
            return int(x), spans
    return None


def _random_counts(rng, maximum, max_gates=3, force_gates=None):
    k = int(rng.integers(1, max_gates + 1)) if force_gates is None else force_gates
    # Minimum canvas height = 5 + 5*k + 2*sum(ns).
    budget = (25 - 5 * k) // 2
    ns = [1] * k
    for j in rng.permutation(k):
        room = min(maximum - 1, budget - sum(ns))
        ns[int(j)] += int(rng.integers(room + 1))
    return ns


def _make_tile(rng, colors, witness, repeat):
    carrier, front, back, auxiliary = colors
    tile = np.full((3, 3), carrier, dtype=np.int8)
    tile[2, 1], tile[0, 1] = front, back
    off_axis = [(r, c) for r in range(3) for c in (0, 2)]
    if witness or rng.random() < 0.7:
        p = off_axis[int(rng.integers(len(off_axis)))]
        tile[p] = auxiliary
    if repeat or rng.random() < 0.5:
        spots = [p for p in off_axis if tile[p] == carrier]
        tile[spots[int(rng.integers(len(spots)))]] = int(rng.choice((front, back)))
    return tile


def _make_pair(rng, palette, ns, sides, k, witness=False, repeat=False, larger=False):
    bg, bar, marker, carrier, front, back, auxiliary = palette
    width = int(rng.integers(13, 31))
    plan = None
    for trial_width in dict.fromkeys((width, max(width, 24), 30)):
        plan = _spans(rng, trial_width, ns, sides)
        if plan is not None:
            width = trial_width
            break
    _require(plan is not None, "no feasible bar spans")
    x, spans = plan
    minimum = 5 + 5 * len(ns) + 2 * sum(ns)
    height = int(rng.integers(max(minimum, 26) if larger else minimum, 31))
    # A fixed total slack is distributed across approach intervals and final exit.
    slack = rng.multinomial(height - minimum, np.full(len(ns) + 1, 1 / (len(ns) + 1)))
    tile = _make_tile(rng, (carrier, front, back, auxiliary), witness, repeat)
    a = np.full((height, width), bg, dtype=np.int8)
    a[:3, x - 1:x + 2] = tile
    positions, old = [], 1
    for i, (n, side, (lo, hi, q)) in enumerate(zip(ns, sides, spans)):
        turn = old + 3 + int(slack[i])
        f, dock = turn + n + 1, turn + 2 * (n + 1)
        a[f, lo:hi + 1] = bar
        for j in range(n):
            a[f, lo + 2 * j if side > 0 else hi - 2 * j] = marker
        positions.append((turn, dock, q))
        old = dock
    # Construct labels from the sampled plan with scalar writes. This does not
    # call the input parser, solve(), or the slice-based output implementation.
    b = a.copy()
    b[b == marker] = bar
    old, current, cargo = 1, x, tile.copy()
    for turn, dock, q in positions:
        color = int(cargo[2, 1])
        for r in range(old + 2, turn + 1):
            b[r, current] = color
        for c in range(min(current, q), max(current, q) + 1):
            b[turn, c] = color
        for r in range(turn, dock - 1):
            b[r, q] = color
        next_cargo = np.empty((3, 3), dtype=np.int8)
        for r in range(3):
            for c in range(3):
                next_cargo[r, c] = cargo[2 - r, 2 - c]
                b[dock - 1 + r, q - 1 + c] = next_cargo[r, c]
        cargo, old, current = next_cargo, dock, q
    for r in range(old + 2, height):
        b[r, current] = int(cargo[2, 1])
    return {"input": np.rot90(a, k).copy(), "output": np.rot90(b, k).copy()}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 2, 10, "num_train")
    m = _count(num_test, 1, 4, "num_test")
    if n <= m:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 7, replace=False)))
    recolor = bool(rng.random() < 0.65)
    maximum = int(rng.integers(2, 5))
    repeat_sides = bool(rng.random() < 0.65)
    count_shift = bool(rng.random() < 0.45)
    directions = list(map(int, rng.permutation(4)))
    # Two visible headings teach sampling in the travel frame; reserve the
    # other two for queries rather than relying on a single upright example.
    train_dirs = directions[:2]
    test_dirs = directions[2:]
    required = list(map(int, rng.permutation(np.arange(1, maximum + 1))))
    if maximum == 4:
        # Any two/two partition has sums <=7 and fits a 30-cell canvas.
        sequences = [required[:2], required[2:]]
    elif maximum == 3:
        sequences = [required[:2], required[2:]]
    else:
        sequences = [required, _random_counts(rng, maximum, 2)]
    sequences += [_random_counts(rng, maximum, 2 if count_shift else 3) for _ in range(n - 2)]
    first_sign = int(rng.choice((-1, 1)))
    puzzle, seen = {"train": [], "test": []}, set()

    def add(split, ns, index):
        start = first_sign if index % 2 == 0 else -first_sign
        if repeat_sides and split == "train" and index == 0:
            sides = [start] * len(ns)
        elif repeat_sides:
            sides = [start] + list(map(int, rng.choice((-1, 1), size=len(ns) - 1)))
        else:
            sides = [start * (-1) ** j for j in range(len(ns))]
        # The second demonstration guarantees the opposite command even if its
        # first gate would otherwise share the first demonstration's sign.
        pool = train_dirs if split == "train" else test_dirs
        k = train_dirs[index] if split == "train" and index < 2 else int(rng.choice(pool))
        for attempt in range(16):
            palette = base_palette
            if recolor:
                rest = list(map(int, rng.permutation([c for c in range(10) if c not in base_palette[1:3]])))
                palette = (rest[0], base_palette[1], base_palette[2], *rest[1:5])
            pair = _make_pair(rng, palette, ns, sides, k,
                              witness=split == "train" and index == 0,
                              repeat=split == "train" and index == 1,
                              larger=split == "test" and count_shift)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise AssertionError("bounded uniqueness retries exhausted")

    for i, ns in enumerate(sequences):
        add("train", ns, i)
    for i in range(m):
        add("test", _random_counts(rng, maximum, force_gates=3 if count_shift else None), i)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _scalar_check(pair):
    """Check every assigned destination and preserved cell from decoded input."""
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "dimensions changed")
    info = _decode(a)
    src, target = info["a"], np.rot90(b, -info["k"])
    expected = src.copy()
    expected[expected == info["marker"]] = info["bar"]
    occupied = src != info["background"]
    old, x = 1, info["initial"]
    cargo = info["tile"].copy()
    for g in info["gates"]:
        turn, dock, q = g["turn"], g["dock"], g["target"]
        path = ([(r, x) for r in range(old + 2, turn + 1)] +
                [(turn, c) for c in range(min(x, q), max(x, q) + 1)] +
                [(r, q) for r in range(turn, dock - 1)])
        for r, c in path:
            _require(not occupied[r, c], "trail collides with an object")
            expected[r, c] = cargo[2, 1]
        next_cargo = np.empty_like(cargo)
        for r in range(3):
            for c in range(3):
                rr, cc = dock - 1 + r, q - 1 + c
                _require(not occupied[rr, cc], "dock collides with an object")
                next_cargo[r, c] = cargo[2 - r, 2 - c]
                expected[rr, cc] = next_cargo[r, c]
        cargo, old, x = next_cargo, dock, q
    for r in range(old + 2, src.shape[0]):
        _require(not occupied[r, x], "exit collides with an object")
        expected[r, x] = cargo[2, 1]
    _require(np.array_equal(expected, target), "incorrect route, stamp, normalization or preservation")
    _require(np.array_equal(solve(a), b), "solver disagrees with scalar check")
    return info


def _d4(tile):
    return [np.rot90(tile, j) for j in range(4)] + [np.rot90(tile[:, ::-1], j) for j in range(4)]


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info = _scalar_check(pair)
            # Dtype-independent input identity, including nested-list callers.
            key = _key(np.asarray(pair["input"], dtype=np.int8))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    train, tests = infos["train"], infos["test"]
    roles = {(i["bar"], i["marker"]) for split in infos.values() for i in split}
    _require(len(roles) == 1, "bar roles changed within puzzle")
    taught = {g["n"] for i in train for g in i["gates"]}
    _require(len(taught) >= 2, "unary distances not distinguished")
    _require({g["n"] for i in tests for g in i["gates"]} <= taught, "untaught unary count")
    _require({g["side"] for i in train for g in i["gates"]} == {-1, 1}, "both commands must be shown")
    _require(any(len(i["gates"]) >= 2 for i in train), "successive stamp turns not shown")
    _require(len({i["k"] for i in train}) >= 2, "travel-relative sampling needs two headings")
    _require(any(len({t.tobytes() for t in _d4(i["tile"])}) == 8 for i in train),
             "training lacks an unambiguous half-turn witness")
    _require(any(np.count_nonzero(i["tile"] == i["tile"][0, 1]) > 1 or
                 np.count_nonzero(i["tile"] == i["tile"][2, 1]) > 1 for i in train),
             "repeated signal colors must be demonstrated")
    def repeats(items):
        return any(any(a["side"] == b["side"] for a, b in zip(i["gates"], i["gates"][1:])) for i in items)
    _require(not repeats(tests) or repeats(train), "repeated command test lacks evidence")
    # Named ablations cover both inherited mechanisms, separately and together.
    for options in ({"count_offset": 0}, {"side_flip": True},
                    {"operation": "identity"}, {"operation": "row_flip"},
                    {"operation": "col_flip"}, {"ink": "carrier"}):
        _require(any((alt := _render(i, **options)) is None or
                     not np.array_equal(alt, np.asarray(p["output"]))
                     for i, p in zip(train, puzzle["train"])),
                 "training does not distinguish " + str(options))


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds plus count/regression checks; never save puzzle payloads."""
    signatures, shapes, structures = set(), set(), set()
    headings, gate_counts, unary_counts, transitions, backgrounds, carriers = (set() for _ in range(6))
    joint, pairs, repeated_tiles = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _decode(pair["input"])
                ns = [g["n"] for g in i["gates"]]
                sides = [g["side"] for g in i["gates"]]
                headings.add(i["k"])
                gate_counts.add(len(ns))
                unary_counts.update(ns)
                transitions.update(zip(sides, sides[1:]))
                backgrounds.add(i["background"])
                carriers.add(i["carrier"])
                shapes.add(pair["input"].shape)
                joint.add((i["k"], len(ns), split))
                role_map = {}
                structures.add((pair["input"].shape, bytes(role_map.setdefault(int(c), len(role_map))
                                                         for c in pair["input"].flat)))
                repeated_tiles += int(np.count_nonzero(i["tile"] == i["tile"][2, 1]) > 1)
                pairs += 1
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(structures) >= 180, "variation collapsed")
    _require(headings == set(range(4)) and gate_counts == {1, 2, 3} and unary_counts == {1, 2, 3, 4},
             "missing structural feature")
    _require(transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}, "missing command transitions")
    _require(len(joint) >= 20 and backgrounds == set(range(10)) and carriers == set(range(10)),
             "joint geometry or palette coverage missing")
    _require(repeated_tiles >= 20, "sample repetition missing")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1200 + 10 * n + m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    _require(_puzzle_digest(p) == _puzzle_digest(generate_puzzle(0)), "seed not deterministic")
    # Color permutation and each orientation preserve the input-only contract.
    permutation = np.array([7, 3, 9, 2, 5, 0, 8, 4, 1, 6], dtype=np.int8)
    for pair in p["test"]:
        for k in range(4):
            a = np.rot90(permutation[pair["input"]], k)
            b = np.rot90(permutation[pair["output"]], k)
            _require(np.array_equal(solve(a), b), "rotation/recoloring failed")
        # Source preservation and every output cell are checked, including noise.
        broken = {s: [{key: value.copy() for key, value in q.items()} for q in p[s]] for s in p}
        broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "heading_gate_split_combinations": len(joint), "repeated_front_signal_tiles": repeated_tiles}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
