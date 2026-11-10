"""Clearance-generated virtual ports join a scrambled set of route tiles.

Only Python's standard library and NumPy are used. solve() reads the input;
construction plans are used solely to build labels, never to predict them.
"""
from collections import Counter
from itertools import permutations, product

import numpy as np


DIRECTIONS = ("down", "right", "up", "left")


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s])
                 for s in ("train", "test"))


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _physical(a, direction):
    # Canonical coordinates: forward is increasing row; transverse is column.
    if direction == "down":
        return a.copy()
    if direction == "up":
        return a[::-1].copy()
    if direction == "right":
        return a.T.copy()
    return a[::-1].T.copy()


def _forward(a, direction):
    if direction in ("down", "up"):
        return _physical(a, direction)
    return a.T.copy() if direction == "right" else a.T[::-1].copy()


def _layout(tiles, order, shape, divider):
    nr, nc = shape
    h, w = tiles[0].shape
    a = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        a[r * (h + 1):r * (h + 1) + h,
          c * (w + 1):c * (w + 1) + w] = tiles[index]
    return a


def _partition(a):
    candidates = []
    for color in map(int, np.unique(a)):
        rr = list(map(int, np.flatnonzero(np.all(a == color, axis=1))))
        cc = list(map(int, np.flatnonzero(np.all(a == color, axis=0))))
        if not rr and not cc:
            continue
        rb, cb = [-1] + rr + [a.shape[0]], [-1] + cc + [a.shape[1]]
        hs = [v - u - 1 for u, v in zip(rb, rb[1:])]
        ws = [v - u - 1 for u, v in zip(cb, cb[1:])]
        n = len(hs) * len(ws)
        if (not 2 <= n <= 5 or min(hs + ws) < 5
                or len(set(hs)) != 1 or len(set(ws)) != 1):
            continue
        tiles = [a[r + 1:rb[i + 1], c + 1:cb[j + 1]].copy()
                 for i, r in enumerate(rb[:-1]) for j, c in enumerate(cb[:-1])]
        if any(np.any(t == color) for t in tiles):
            continue
        corners = {int(t[r, c]) for t in tiles
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) == 1:
            candidates.append((tiles, color, corners.pop(), (len(hs), len(ws))))
    if len(candidates) != 1:
        raise ValueError("expected one equal-tile partition with a common corner background")
    return candidates[0]


def _read_tile(a, bg, route, bar, marker):
    """Decode one tile in forward coordinates and trace its local route."""
    fsize, width = a.shape
    assert 5 <= fsize <= 14 and 8 <= width <= 30
    seeds = [tuple(map(int, p)) for p in np.argwhere(a == route)]
    assert seeds
    x = seeds[0][1]
    assert 1 <= x <= width - 2
    structure = (a == bar) | (a == marker)
    rows = list(map(int, np.flatnonzero(structure.any(axis=1))))
    if not rows:
        assert seeds == [(0, x), (1, x)]
        assert np.count_nonzero(a != bg) == 2
        return dict(incoming=x, outgoing=None, start=False, end=True,
                    path=(a == route), bars=(), seed=x, seed_f=0)
    assert seeds in ([(0, x)], [(1, x)])
    assert 1 <= len(rows) <= 2
    start = seeds[0][0] == 1
    seed_f = seeds[0][0]
    path = np.zeros(a.shape, dtype=bool)
    current, begin, previous = x, seed_f, -1
    bars = []
    for f in rows:
        assert 0 < f < fsize - 1
        occupied = list(map(int, np.flatnonzero(structure[f])))
        lo, hi = occupied[0], occupied[-1]
        assert occupied == list(range(lo, hi + 1))
        marks = tuple(map(int, np.flatnonzero(a[f] == marker)))
        n = len(marks)
        assert 1 <= n <= 4 and hi - lo + 1 >= 2 * n + 1
        low = marks == tuple(lo + 2 * j for j in range(n))
        high = marks == tuple(hi - 2 * j for j in reversed(range(n)))
        assert low != high, "markers need an asymmetric endpoint and a permanent tail"
        d = n + 1
        sign = 1 if low else -1
        target = hi + d if low else lo - d
        turn = f - d
        assert 1 <= target <= width - 2
        assert lo <= current <= hi and turn >= max(1, begin) and turn > previous
        for y in range(begin, turn + 1):
            path[y, current] = True
        for c in range(min(current, target), max(current, target) + 1):
            path[turn, c] = True
        bars.append(dict(f=f, lo=lo, hi=hi, count=n, side=sign,
                         turn=turn, target=target, entering=current))
        current, begin, previous = target, turn, f
    for y in range(begin, fsize):
        path[y, current] = True
    assert not np.any(path & structure)
    assert np.all(np.isin(a, (bg, route, bar, marker)))
    return dict(incoming=None if start else x, outgoing=current,
                start=start, end=False, path=path, bars=tuple(bars), seed=x, seed_f=seed_f)


def _orders(records, ignore_clearance=False):
    starts = [i for i, t in enumerate(records) if t["start"]]
    ends = [i for i, t in enumerate(records) if t["end"]]
    if len(starts) != 1 or len(ends) != 1:
        return []
    result = []
    for order in permutations(range(len(records))):
        if order[0] != starts[0] or order[-1] != ends[0]:
            continue
        if all((records[a]["seed"] if ignore_clearance else records[a]["outgoing"])
               == records[b]["incoming"] for a, b in zip(order, order[1:])):
            result.append(order)
            if len(result) == 2:
                break
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    tiles, divider, bg, layout = _partition(a)
    other = set(map(int, np.unique(a))) - {divider, bg}
    if len(other) != 3:
        raise ValueError("expected five distinct color roles")
    candidates = []
    for direction in DIRECTIONS:
        canonical = [_forward(t, direction) for t in tiles]
        edges = [np.flatnonzero(t[0] != bg) for t in canonical]
        if (sum(len(e) == 0 for e in edges) != 1
                or any(len(e) > 1 for e in edges)
                or any(np.any(t[-1] != bg) for t in canonical)):
            continue
        port_colors = {int(t[0, e[0]]) for t, e in zip(canonical, edges) if len(e)}
        if len(port_colors) != 1:
            continue
        route = port_colors.pop()
        for bar, marker in permutations(sorted(other - {route})):
            try:
                records = [_read_tile(t, bg, route, bar, marker) for t in canonical]
                orders = _orders(records)
                assert len(orders) == 1
                fsize, width = canonical[0].shape
                assert len(tiles) * (fsize + 1) - 1 <= 30
                candidates.append(dict(tiles=tiles, canonical=canonical, records=records,
                                       order=orders[0], direction=direction,
                                       palette=(bg, divider, route, bar, marker),
                                       layout=layout, fsize=fsize, width=width))
            except (AssertionError, IndexError):
                continue
    if len(candidates) != 1:
        raise ValueError("roles, direction and complete virtual-port chain must be unique")
    return candidates[0]


def _output(parsed, order=None):
    bg, divider, route, bar, marker = parsed["palette"]
    order = parsed["order"] if order is None else order
    completed = []
    for a, record in zip(parsed["canonical"], parsed["records"]):
        b = a.copy()
        b[b == marker] = bar
        b[record["path"]] = route
        completed.append(b)
    result = _layout(completed, order, (len(order), 1), divider)
    for j, index in enumerate(order[:-1]):
        result[(j + 1) * (parsed["fsize"] + 1) - 1,
               parsed["records"][index]["outgoing"]] = route
    return _physical(result, parsed["direction"])


def _local_only(parsed):
    """Ablation of A: complete each tile in its original input slot."""
    bg, divider, route, bar, marker = parsed["palette"]
    completed = []
    for a, record in zip(parsed["canonical"], parsed["records"]):
        b = a.copy()
        b[b == marker] = bar
        b[record["path"]] = route
        completed.append(_physical(b, parsed["direction"]))
    return _layout(completed, tuple(range(len(completed))), parsed["layout"], divider)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, local paths, matching order and final grid from this input."""
    return _output(_decode(input_grid))


def _span_options(width, current, n, side=None, length=None, farther=False, endpoint_entry=False):
    result = []
    d = n + 1
    for lo in range(current + 1):
        for hi in range(max(current, lo + 2 * n), width):
            if length is not None and hi - lo + 1 != length:
                continue
            for sign in ((-1, 1) if side is None else (side,)):
                if endpoint_entry and current != (hi if sign == 1 else lo):
                    continue
                target = hi + d if sign == 1 else lo - d
                far_distance = hi - current if sign == 1 else current - lo
                near_distance = current - lo if sign == 1 else hi - current
                if 1 <= target <= width - 2 and (not farther or far_distance > near_distance):
                    result.append((lo, hi, target, sign))
    return result


def _spans(rng, width, counts, constraints):
    failed, cached = set(), {}

    def visit(j, current):
        if j == len(counts):
            return []
        state = (j, current)
        if state in failed:
            return None
        if state not in cached:
            options = _span_options(width, current, counts[j], **constraints.get(j, {}))
            cached[state] = [options[int(k)] for k in rng.permutation(len(options))]
        for item in cached[state]:
            tail = visit(j + 1, item[2])
            if tail is not None:
                return [item] + tail
        failed.add(state)
        return None

    for initial in map(int, rng.permutation(np.arange(1, width - 1))):
        answer = visit(0, initial)
        if answer is not None:
            return initial, answer
    return None


def _positions(rng, fsize, counts):
    minimum = sum(counts) + 2 * len(counts) + 2
    assert fsize >= minimum
    slack = fsize - minimum
    allocation = rng.multinomial(slack, np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    previous, rows = -1, []
    for j, n in enumerate(counts):
        turn = (1 if j == 0 else previous + 1) + int(allocation[j])
        previous = turn + n + 1
        rows.append(previous)
    return rows


def _draw_tile(rng, fsize, width, counts, spans, initial, start, palette):
    bg, divider, route, bar, marker = palette
    a = np.full((fsize, width), bg, dtype=np.int8)
    b = a.copy()
    seed_f = 1 if start else 0
    a[seed_f, initial] = route
    current, begin = initial, seed_f
    rows = _positions(rng, fsize, counts)
    for f, n, (lo, hi, target, sign) in zip(rows, counts, spans):
        a[f, lo:hi + 1] = bar
        a[f, [lo + 2 * j if sign == 1 else hi - 2 * j for j in range(n)]] = marker
        b[f, lo:hi + 1] = bar
        turn = f - n - 1
        b[begin:turn + 1, current] = route
        b[turn, min(current, target):max(current, target) + 1] = route
        current, begin = target, turn
    b[begin:, current] = route
    return a, b, current


def _counts_for_tile(rng, fsize, maximum, two=False):
    choices = [(n,) for n in range(1, maximum + 1) if n + 4 <= fsize]
    choices += [(a, b) for a, b in product(range(1, maximum + 1), repeat=2)
                if a + b + 6 <= fsize]
    if two:
        choices = [c for c in choices if len(c) == 2]
    return _pick(rng, choices)


def _make_pair(rng, maximum, direction, palette, kind, max_tiles=5):
    if kind == "matching":
        n, fsize = 4, 6
        counts = [(2,), (1,), (int(rng.integers(1, 3)),)]
    elif kind == "returning":
        n, fsize = 3, int(rng.integers(8, 10))
        counts = [(1,), (1, 1)]
    elif kind == "counting":
        first = (3, 4) if maximum == 4 else (1, 3) if maximum == 3 else (1, 2)
        minimum = sum(first) + 6
        n = int(rng.integers(2, 4)) if minimum <= 9 else 2
        limit = 31 // n - 1
        fsize = int(rng.integers(minimum, limit + 1))
        counts = [first] + [_counts_for_tile(rng, fsize, maximum) for _ in range(n - 2)]
    else:
        n = 5 if kind == "long_test" else int(rng.integers(2, max_tiles + 1))
        fsize = int(rng.integers(5, min(14, 31 // n - 1) + 1))
        counts = [_counts_for_tile(rng, fsize, maximum) for _ in range(n - 1)]
    flat_counts = [x for seq in counts for x in seq]
    lower_width = max(8, 3 * max(flat_counts) + 4)
    if kind == "matching":
        lower_width = max(12, lower_width)
    width = int(rng.integers(lower_width, 31))
    if kind == "matching" and rng.random() < .55:
        width = int(rng.integers(lower_width, 15))
    constraints = {}
    if kind == "matching":
        length = int(rng.integers(5, min(8, (width - 2) // 2) + 1))
        sign = int(rng.choice((-1, 1)))
        constraints = {0: dict(side=sign, length=length, farther=True),
                       1: dict(side=-sign, length=length, endpoint_entry=True)}
    elif kind == "counting":
        sign = int(rng.choice((-1, 1)))
        second = sign if rng.random() < .5 else -sign
        constraints = {0: dict(side=sign), 1: dict(side=second)}
    elif kind == "returning":
        sign = int(rng.choice((-1, 1)))
        constraints = {1: dict(side=sign, endpoint_entry=True),
                       2: dict(side=-sign, endpoint_entry=True)}
    for attempt in range(100):
        plan = _spans(rng, width, flat_counts, constraints)
        if plan is None:
            # Width bounds depend on the full sequence, particularly repeated sides.
            width = 30
            continue
        current, spans = plan
        inputs, outputs, exits, cursor = [], [], [], 0
        for i, seq in enumerate(counts):
            local = spans[cursor:cursor + len(seq)]
            cursor += len(seq)
            a, b, current = _draw_tile(rng, fsize, width, seq, local, current, i == 0, palette)
            inputs.append(a)
            outputs.append(b)
            exits.append(current)
        terminal = np.full((fsize, width), palette[0], dtype=np.int8)
        terminal[:2, current] = palette[2]
        inputs.append(terminal)
        outputs.append(terminal.copy())
        # Require unique complete matching, including when positions repeat.
        records = [_read_tile(a, palette[0], *palette[2:]) for a in inputs]
        if _orders(records) != [tuple(range(n))]:
            continue
        tiles = [_physical(a, direction) for a in inputs]
        h, w = tiles[0].shape
        layouts = [(r, n // r) for r in range(1, n + 1) if n % r == 0
                   and r * (h + 1) - 1 <= 30 and (n // r) * (w + 1) - 1 <= 30]
        shape = _pick(rng, layouts)
        order = list(map(int, rng.permutation(n)))
        if kind == "matching":
            # Two intermediate tiles appear in the opposite order to their chain order.
            if order.index(1) < order.index(2):
                j, k = order.index(1), order.index(2)
                order[j], order[k] = order[k], order[j]
        elif order == list(range(n)):
            order.reverse()
        inp = _layout(tiles, order, shape, palette[1])
        out = _layout(outputs, list(range(n)), (n, 1), palette[1])
        for j, x in enumerate(exits):
            out[(j + 1) * (fsize + 1) - 1, x] = palette[2]
        out = _physical(out, direction)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded geometric construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.integers(2, 5))
    dirs = list(rng.permutation(DIRECTIONS))
    shift = _pick(rng, ("direction", "length"))
    train_dirs = dirs[:3] if shift == "direction" else dirs
    test_dirs = dirs[3:] if shift == "direction" else dirs
    shared = bool(rng.random() < .3)
    shared_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    kinds = ["matching", "counting"] + ["returning" if rng.random() < .25 else "ordinary"
                                         for _ in range(num_train - 2)]
    rng.shuffle(kinds)
    for split, count in (("train", num_train), ("test", num_test)):
        for j in range(count):
            kind = kinds[j] if split == "train" else "long_test" if shift == "length" else "ordinary"
            for _ in range(50):
                palette = shared_palette if shared else tuple(map(int, rng.choice(10, 5, replace=False)))
                direction = str(_pick(rng, train_dirs if split == "train" else test_dirs))
                pair = _make_pair(rng, maximum, direction, palette, kind,
                                  max_tiles=4 if shift == "length" and split == "train" else 5)
                if split == "test":
                    teaching = [_features(_decode(p["input"])) for p in puzzle["train"]]
                    queried = _features(_decode(pair["input"]))
                    if queried["same_side"] and not any(f["same_side"] for f in teaching):
                        continue
                    if queried["repeated_ports"] and not any(f["repeated_ports"] for f in teaching):
                        continue
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(parsed):
    records = parsed["records"]
    bars = [b for t in records for b in t["bars"]]
    ordered = [records[i] for i in parsed["order"]]
    links = [t["outgoing"] for t in ordered[:-1]]
    return dict(n=len(records), direction=parsed["direction"], layout=parsed["layout"],
                tile_shape=(parsed["fsize"], parsed["width"]), palette=parsed["palette"],
                counts={b["count"] for b in bars}, sides={b["side"] for b in bars},
                max_bars=max(len(t["bars"]) for t in records),
                repeated_ports=len(set(links)) < len(links),
                same_side=any(a["side"] == b["side"] for t in records
                              for a, b in zip(t["bars"], t["bars"][1:])),
                returns=any(t["incoming"] == t["outgoing"] for t in records if not t["start"] and not t["end"]))


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, b = _grid(pair["input"]), _grid(pair["output"])
    parsed = _decode(a)
    assert np.array_equal(b, _output(parsed)), "wrong order, path, marker replacement or join"
    bg, divider, route, bar, marker = parsed["palette"]
    canonical_out = _forward(b, parsed["direction"])
    n, fsize = len(parsed["records"]), parsed["fsize"]
    assert canonical_out.shape == (n * (fsize + 1) - 1, parsed["width"])
    # Check every copied tile and its allowed changes, independently of the layout builder.
    for slot, index in enumerate(parsed["order"]):
        original = parsed["canonical"][index]
        completed = canonical_out[slot * (fsize + 1):slot * (fsize + 1) + fsize]
        record = parsed["records"][index]
        assert np.all(completed[original == marker] == bar)
        assert np.all(completed[original == bar] == bar)
        assert np.all(completed[original == route] == route)
        assert np.array_equal(completed == route, record["path"])
        assert np.all(completed[(original == bg) & ~record["path"]] == bg)
        if slot < n - 1:
            seam = canonical_out[slot * (fsize + 1) + fsize]
            assert np.flatnonzero(seam == route).tolist() == [record["outgoing"]]
            assert np.count_nonzero(seam == divider) == parsed["width"] - 1
            assert record["outgoing"] == parsed["records"][parsed["order"][slot + 1]]["incoming"]
    # One orthogonally connected, unbranched route from interior start to terminal cap.
    cells = {tuple(map(int, p)) for p in np.argwhere(b == route)}
    todo, visited = [next(iter(cells))], set()
    degrees = []
    while todo:
        p = todo.pop()
        if p in visited:
            continue
        visited.add(p)
        r, c = p
        neighbors = {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)} & cells
        degrees.append(len(neighbors))
        todo.extend(neighbors - visited)
    assert visited == cells and Counter(degrees)[1] == 2 and set(degrees) <= {1, 2}
    assert marker not in set(map(int, np.unique(b)))
    assert not np.array_equal(a, b)
    return parsed


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {}
    for split in ("train", "test"):
        analyses[split] = []
        for pair in puzzle[split]:
            parsed = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            analyses[split].append(parsed)
    train = analyses["train"]
    train_counts = {b["count"] for p in train for t in p["records"] for b in t["bars"]}
    assert {1, 2} <= train_counts
    assert any(_features(p)["max_bars"] == 2 for p in train)
    assert {s for p in train for s in _features(p)["sides"]} == {-1, 1}
    # A genuine ordering witness, plus the two mechanism ablations.
    witnesses = [p for p in train if len(p["records"]) >= 4
                 and [i for i in p["order"] if not p["records"][i]["start"] and not p["records"][i]["end"]]
                 != [i for i, t in enumerate(p["records"]) if not t["start"] and not t["end"]]]
    assert witnesses, "demonstrate ordering of at least two intermediate tiles"
    # The matching witness has a tempting false connector: one-cell clearance
    # reaches the other intermediate tile, while count+1 reaches the correct one.
    def has_false_port(p):
        start = p["records"][p["order"][0]]
        last = start["bars"][-1]
        false_exit = last["hi"] + 1 if last["side"] == 1 else last["lo"] - 1
        true_next = p["order"][1]
        return any(t["incoming"] == false_exit and i != true_next and not t["end"]
                   for i, t in enumerate(p["records"]))
    assert any(has_false_port(p) for p in witnesses), "demonstrate clearance choosing between candidate entries"
    assert any(_orders(p["records"], ignore_clearance=True) != [p["order"]] for p in witnesses)
    assert any(not np.array_equal(_local_only(p), _output(p)) for p in witnesses)
    bars = [b for p in train for t in p["records"] for b in t["bars"]]
    assert any(a["count"] != b["count"] and a["hi"] - a["lo"] == b["hi"] - b["lo"]
               for p in train for a in [x for t in p["records"] for x in t["bars"]]
               for b in [x for t in p["records"] for x in t["bars"]])
    assert any((b["hi"] - b["entering"] > b["entering"] - b["lo"] if b["side"] == 1
                else b["entering"] - b["lo"] > b["hi"] - b["entering"]) for b in bars)
    for p in analyses["test"]:
        assert _features(p)["counts"] <= train_counts
        if _features(p)["same_side"]:
            assert any(_features(t)["same_side"] for t in train)
        if _features(p)["repeated_ports"]:
            assert any(_features(t)["repeated_ports"] for t in train)
    # The shift can be measured from the returned inputs, with no latent flags.
    train_dirs = {p["direction"] for p in train}
    train_n = max(len(p["records"]) for p in train)
    assert all(p["direction"] not in train_dirs or len(p["records"]) > train_n
               for p in analyses["test"])


def self_test() -> None:
    """Bounded checks of 42 seeds, legal counts, counterexamples and diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[7]) == _puzzle_key(generate_puzzle(7))
    features, geometry, shapes = [], set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parsed = _decode(pair["input"])
                features.append((_features(parsed), split))
                roles = np.empty_like(pair["input"])
                for j, color in enumerate(parsed["palette"]):
                    roles[pair["input"] == color] = j
                geometry.add(_key(roles))
                shapes.add(pair["input"].shape)
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
    assert len(geometry) >= 180 and len(shapes) >= 45
    assert {f["n"] for f, _ in features} == {2, 3, 4, 5}
    assert {f["direction"] for f, _ in features} == set(DIRECTIONS)
    assert {n for f, _ in features for n in f["counts"]} == {1, 2, 3, 4}
    assert len({f["tile_shape"] for f, _ in features}) >= 40
    assert any(f["layout"] == (2, 2) for f, _ in features)
    assert any(f["same_side"] for f, _ in features)
    assert any(f["repeated_ports"] for f, _ in features)
    assert any(f["returns"] for f, _ in features)
    assert len({f["palette"] for f, _ in features}) >= 80
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1234, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Literal local-route regression: n=2 -> d=3, not n or constant clearance.
    local = np.zeros((9, 14), dtype=np.int8)
    local[0, 4] = 3
    local[5, 2:7] = (2, 1, 2, 1, 1)
    record = _read_tile(local, 0, 3, 1, 2)
    expected = np.zeros_like(local, dtype=bool)
    expected[:3, 4] = True
    expected[2, 4:10] = True
    expected[2:, 9] = True
    assert record["outgoing"] == 9 and np.array_equal(record["path"], expected)
    # An ambiguous matching graph must be rejected, including identical ports.
    records = [dict(start=True, end=False, incoming=None, outgoing=4),
               dict(start=False, end=False, incoming=4, outgoing=4),
               dict(start=False, end=False, incoming=4, outgoing=4),
               dict(start=False, end=True, incoming=4, outgoing=None)]
    assert len(_orders(records)) == 2
    # Reject a missing bridge and a damaged permanent obstacle.
    p = puzzles[0]
    for role in ("bridge", "bar"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = damaged["test"][0]
        parsed = _decode(pair["input"])
        out = _forward(pair["output"], parsed["direction"])
        if role == "bridge":
            x = parsed["records"][parsed["order"][0]]["outgoing"]
            out[parsed["fsize"], x] = parsed["palette"][1]
        else:
            cell = tuple(np.argwhere(out == parsed["palette"][3])[0])
            out[cell] = parsed["palette"][0]
        pair["output"] = _physical(out, parsed["direction"])
        try:
            validate_generated_puzzle(damaged)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("damaged route or obstacle accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
