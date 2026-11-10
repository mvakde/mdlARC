"""Cable clocks: run-length pulse programs evaluated in cable-path coordinates.

Only NumPy and the standard library are used. Construction knows its drawn paths;
solve reconstructs every path from input cells and never consults construction.
"""

from itertools import permutations
import json
from pathlib import Path

import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (
    0, 432724097, 437455171, 767657390, 708617778, 617399742, 53160775,
    308242864, 361666968, 32051039, 783077152, 729162886, 34398986,
    472823562, 398378596, 431937503, 436370413, 266265226, 649048525,
    256527824, 458042198, 979216566, 306125680, 549335034, 369520275,
    68674371, 191926664, 98735890, 547204598, 382252877, 403126705,
    633447373, 64064127, 511664714, 932677427, 805300494, 577838005,
    609539147, 578558219, 991152923, 405368064, 259243264,
)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer colors required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _glyphs(g, bg):
    """Find the two complementary glyphs using vectorized local comparisons."""
    v = g[1:-1, 1:-1]
    orth = (g[:-2, 1:-1], g[2:, 1:-1], g[1:-1, :-2], g[1:-1, 2:])
    corners = (g[:-2, :-2], g[:-2, 2:], g[2:, :-2], g[2:, 2:])
    d = (v == bg) & (orth[0] != bg)
    k = v != bg
    for x in orth:
        d &= x == orth[0]
        k &= x == bg
    for x in corners:
        d &= x == bg
        k &= x == v
    diamonds, checkers = {}, {}
    for rr, cc in np.argwhere(d):
        r, c = int(rr) + 1, int(cc) + 1
        diamonds.setdefault(int(g[r - 1, c]), []).append((r, c))
    for rr, cc in np.argwhere(k):
        r, c = int(rr) + 1, int(cc) + 1
        checkers.setdefault(int(g[r, c]), []).append((r, c))
    return diamonds, checkers


def _program(g, row, source_col, bg, structural):
    """Nearest-first maximal runs; background gaps never become clock ticks."""
    blocks, c = [], source_col - 2
    while c >= 0:
        color = int(g[row, c])
        if color == bg:
            c -= 1
            continue
        assert color not in structural, "structural color in pulse program"
        near = source_col - c
        length = 0
        while c >= 0 and g[row, c] == color:
            length += 1
            c -= 1
        blocks.append((length, color, near))
    assert 1 <= len(blocks) <= 3, "one to three runs per program"
    assert all(1 <= b[0] <= 5 for b in blocks), "period range"
    return blocks


def _trace(g, rows, left, right, carrier, crossing):
    """Directional tracing keeps the two visits to each crossing independent."""
    cells = set(map(tuple, np.argwhere((g == carrier) | (g == crossing))))
    starts = {(r, left) for r in rows}
    ends = {(r, right) for r in rows}
    neighbors = {}
    for p in cells:
        r, c = p
        edges = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(DIRECTIONS)
                 if (r + dr, c + dc) in cells}
        if p in starts:
            assert set(edges) == {1}, "launch pin must point toward the field"
        elif p in ends:
            assert set(edges) == {3}, "receiver pin must point toward the field"
        elif g[p] == crossing:
            assert len(edges) == 4, "a crossing needs four orthogonal arms"
        else:
            assert len(edges) == 2, "broken cable, contact, or unmarked junction"
        neighbors[p] = edges
    routes, destinations, visited = [], [], set()
    for row in rows:
        p, incoming, route, states = (row, left), 3, [], set()
        while True:
            state = (p, incoming)
            assert state not in states, "cyclic strand"
            states.add(state)
            assert p not in route, "a strand must not cross itself"
            route.append(p)
            visited.add(p)
            if p in ends:
                destinations.append(rows.index(p[0]))
                break
            assert p not in starts or len(route) == 1, "returned to source bank"
            edges = neighbors[p]
            if len(route) == 1 or g[p] == crossing:
                outgoing = (incoming + 2) % 4
            else:
                assert incoming in edges
                outgoing = next(d for d in edges if d != incoming)
            assert outgoing in edges
            p, incoming = edges[outgoing], (outgoing + 2) % 4
        routes.append(route)
    assert sorted(destinations) == list(range(len(rows))), "terminal pairing is not bijective"
    assert visited == cells, "unattached cable fragment"
    for p in cells:
        visits = sum(p in route for route in routes)
        assert visits == (2 if g[p] == crossing else 1), "invalid crossing ownership"
    return routes, destinations


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "blank exterior border"
    candidates = []
    for rotation in range(4):
        g = np.rot90(a, rotation)
        diamonds, checkers = _glyphs(g, bg)
        for marker in sorted(set(diamonds) & set(checkers)):
            ds, ks = diamonds[marker], checkers[marker]
            if not 1 <= len(ds) == len(ks) <= 4:
                continue
            if len({c for _, c in ds}) != 1 or len({c for _, c in ks}) != 1:
                continue
            s, t = ds[0][1], ks[0][1]
            rows = sorted(r for r, _ in ds)
            if s < 3 or t - s < 5 or sorted(r for r, _ in ks) != rows:
                continue
            if any(b - x < 4 for x, b in zip(rows, rows[1:])):
                continue
            if int(np.count_nonzero(g == marker)) != 9 * len(rows):
                continue
            try:
                carrier = int(g[rows[0], s + 2])
                assert carrier not in (bg, marker)
                assert all(g[r, s + 2] == carrier and g[r, t - 2] == carrier for r in rows)
                wire_region = g[:, s + 2:t - 1]
                others = set(map(int, np.unique(wire_region))) - {bg, carrier}
                assert len(others) <= 1, "more than one crossing role"
                crossing = next(iter(others)) if others else -1
                structural = {bg, marker, carrier, crossing}
                programs = [_program(g, r, s, bg, structural) for r in rows]
                allowed = np.zeros(g.shape, dtype=bool)
                allowed[:, s + 2:t - 1] = True
                for r in rows:
                    allowed[r, :s - 1] = True
                    allowed[r - 1:r + 2, s - 1:s + 2] = True
                    allowed[r - 1:r + 2, t - 1:t + 2] = True
                assert np.all(g[~allowed] == bg), "unrelated foreground"
                assert not np.any(g[:, :s + 2] == carrier) and not np.any(g[:, t - 1:] == carrier)
                if crossing != -1:
                    assert not np.any(g[:, :s + 2] == crossing) and not np.any(g[:, t - 1:] == crossing)
                routes, destinations = _trace(g, rows, s + 2, t - 2, carrier, crossing)
                candidates.append(dict(grid=g, rotation=rotation, background=bg, marker=marker,
                                       carrier=carrier, crossing=crossing, rows=rows,
                                       source_col=s, receiver_col=t, programs=programs,
                                       routes=routes, destinations=destinations))
            except (AssertionError, StopIteration, IndexError):
                continue
    assert len(candidates) == 1, f"expected one input interpretation, got {len(candidates)}"
    return candidates[0]


def _winner(blocks, tick, priority="near"):
    firing = [b for b in blocks if tick % b[0] == 0]
    if not firing:
        return None
    if priority == "short":
        return min(firing, key=lambda b: (b[0], b[2]))[1]
    if priority == "long":
        return min(firing, key=lambda b: (-b[0], b[2]))[1]
    if priority == "far":
        return firing[-1][1]
    return firing[0][1]


def _paint_glyph(g, center, pattern, color):
    r, c = center
    view = g[r - 1:r + 2, c - 1:c + 2]
    view[pattern] = color


def _render(info, priority="near", metric="path"):
    """Cell-wise input-only solver, with explicit rival rules for teaching checks."""
    g = info["grid"].copy()
    s, t, rows = info["source_col"], info["receiver_col"], info["rows"]
    for i, (route, blocks) in enumerate(zip(info["routes"], info["programs"])):
        _paint_glyph(g, (rows[i], s), DIAMOND, _winner(blocks, 0, priority))
        skipped = 0
        for tick, p in enumerate(route, 1):
            if metric == "skip_crossings" and info["grid"][p] == info["crossing"]:
                skipped += 1
            time = tick
            if metric == "projected":
                time = p[1] - s - 1
            elif metric == "skip_crossings":
                time -= skipped
            elif metric == "gap_phase":
                time += blocks[0][2] - 2
            color = blocks[0][1] if metric == "solid" else _winner(blocks, time, priority)
            if info["grid"][p] == info["carrier"] and color is not None:
                g[p] = color
        arrival = len(route) + 1
        if metric == "projected":
            arrival = t - s - 2
        elif metric == "skip_crossings":
            arrival -= skipped
        elif metric == "gap_phase":
            arrival += blocks[0][2] - 2
        color = blocks[0][1] if metric == "solid" else _winner(blocks, arrival, priority)
        if color is not None:
            _paint_glyph(g, (route[-1][0], t), CHECKER, color)
    return np.rot90(g, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive pulse periods, priority, routing, phase and terminal samples from input."""
    return _render(_parse(input_grid))


def _routing_moves(target, rng):
    """Realize arbitrary endpoint permutations with a temporary vacant track."""
    n = len(target)
    occupants, moves = list(range(n)) + [None], []
    for lane in map(int, rng.permutation(n)):
        if occupants[lane] == target[lane]:
            continue
        saved = occupants[lane]
        moves.append((lane, n))
        occupants[n], occupants[lane] = saved, None
        vacancy = lane
        while target[vacancy] != saved:
            old = occupants.index(target[vacancy])
            moves.append((old, vacancy))
            occupants[vacancy], occupants[old] = occupants[old], None
            vacancy = old
        moves.append((n, vacancy))
        occupants[vacancy], occupants[n] = occupants[n], None
    assert occupants[:n] == list(target)
    return moves


def _random_lengths(rng, max_period, max_runs):
    count = int(rng.choice(np.arange(1, max_runs + 1), p=([1] if max_runs == 1 else
                         [.62, .38] if max_runs == 2 else [.45, .35, .20])))
    # Eight payload cells leave useful cable space even in nonlocal permutations.
    lengths = []
    for i in range(count):
        room = 8 - sum(lengths) - (count - i - 1)
        lengths.append(int(rng.integers(1, min(max_period, room) + 1)))
    return lengths


def _construct_pair(rng, params, split, lesson, rotation, *, forced_lengths=None,
                    forced_target=None, reflect=None):
    a, b = params["witness"]
    x, y = params["witness_colors"]
    shift = params["shift"]
    if lesson == "priority":
        n, lengths, explicit = 2, [[a, b], [b, a]], [[x, y], [y, x]]
    elif lesson == "metric":
        n, lengths, explicit = 3, [[a], [b], [1]], [[x], [y], [x]]
    else:
        n = 4 if split == "test" and shift == "count" else int(rng.integers(1, 4 if shift == "count" else 5))
        max_runs = 3 if split == "test" and shift == "mixture" else 2
        lengths = [_random_lengths(rng, b, max_runs) for _ in range(n)]
        explicit = [None] * n
        if split == "test" and shift == "mixture":
            lengths[int(rng.integers(n))] = [b, a, 1]
    if forced_lengths is not None:
        lengths = [list(ls) for ls in forced_lengths]
        n, explicit = len(lengths), [None] * len(lengths)

    protected = {x, y}
    if params["vary_palette"]:
        structural = list(map(int, rng.choice([c for c in range(10) if c not in protected], 4, replace=False)))
    else:
        structural = params["palette"][:4]
    bg, carrier, crossing, marker = structural
    labels = [c for c in range(10) if c not in structural]
    pool_size = int(rng.integers(2, len(labels) + 1))
    label_pool = list(map(int, rng.choice(labels, pool_size, replace=False)))
    colors, minimum_gaps = [], []
    for ls, cs in zip(lengths, explicit):
        if cs is None:
            cs = list(map(int, rng.choice(label_pool, len(ls))))
            if len(ls) == 3 and rng.random() < .8:
                cs = list(map(int, rng.choice(labels, 3, replace=False)))
        colors.append(cs)
        minimum_gaps.append([0] + [int(p == q) for p, q in zip(cs, cs[1:])])
    need = max(sum(ls) + sum(gs) for ls, gs in zip(lengths, minimum_gaps))

    target = list(map(int, rng.permutation(n)))
    if lesson is not None:
        target = list(range(n))
        target[0], target[-1] = target[-1], target[0]
    elif rng.random() < .30:
        target = list(range(n))
    if forced_target is not None:
        target = list(forced_target)
    moves = _routing_moves(target, rng)
    if lesson is None and rng.random() < .4:
        lane = int(rng.integers(n))
        candidate = [(lane, n), (n, lane)] + moves
        if need + 11 + 2 * len(candidate) <= 30:
            moves = candidate
    minimum_span = 5 if not moves else 2 * len(moves) + 3
    if need + minimum_span + 8 > 30:
        # A wider program and a dense permutation compete for the same dimension.
        # Keep the program; use a nonlocal transposition that fits in three moves.
        target = list(range(n))
        if n > 1:
            target[0], target[-1] = target[-1], target[0]
        moves = _routing_moves(target, rng)
        minimum_span = 5 if not moves else 2 * len(moves) + 3
    assert need + minimum_span + 8 <= 30
    parts = [1, 1, need, 2, 2] + [2] * max(0, len(moves) - 1)
    extra = int(rng.integers(31 - (need + minimum_span + 8)))
    for _ in range(extra):
        parts[int(rng.integers(len(parts)))] += 1
    left_margin, right_margin, program_width, lead, tail = parts[:5]
    gaps = parts[5:]
    source_col = left_margin + program_width + 1
    start = source_col + 2
    move_columns, column = [], start + lead
    for i in range(len(moves)):
        move_columns.append(column)
        if i < len(gaps):
            column += gaps[i]
    end = (move_columns[-1] + tail) if moves else start + lead + tail
    receiver_col = end + 2
    width = receiver_col + 2 + right_margin

    row_gaps = [int(rng.integers(4, 7)) for _ in range(n - 1)]
    top = int(rng.integers(2, 5))
    rows = [top]
    for gap in row_gaps:
        rows.append(rows[-1] + gap)
    spare = rows[-1] + int(rng.integers(3, min(6, 28 - rows[-1]) + 1))
    bottom = int(rng.integers(1, min(3, 29 - spare) + 1))
    height = spare + bottom + 1
    assert height <= 30 and width <= 30
    tracks = rows + [spare]
    paths = [[(r, start)] for r in rows]
    state, previous = list(range(n)) + [None], start
    for column, (old, new) in zip(move_columns, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((tracks[lane], c) for c in range(previous + 1, column + 1))
        route = state[old]
        step = 1 if tracks[new] > tracks[old] else -1
        paths[route].extend((r, column) for r in range(tracks[old] + step, tracks[new] + step, step))
        state[new], state[old] = state[old], None
        previous = column
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((tracks[lane], c) for c in range(previous + 1, end + 1))

    inp = np.full((height, width), bg, dtype=np.int8)
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            visits.setdefault(p, set()).add(i)
    for p, owners in visits.items():
        assert len(owners) in (1, 2)
        inp[p] = crossing if len(owners) == 2 else carrier
    for row in rows:
        _paint_glyph(inp, (row, source_col), DIAMOND, marker)
        _paint_glyph(inp, (row, receiver_col), CHECKER, marker)

    for i, (ls, cs, gs) in enumerate(zip(lengths, colors, minimum_gaps)):
        spare_space = program_width - sum(ls) - sum(gs)
        allocation = rng.multinomial(spare_space, [1 / (len(ls) + 1)] * (len(ls) + 1))
        gs = [gap + int(add) for gap, add in zip(gs, allocation)]
        cursor = source_col - 1
        for length, color, gap in zip(ls, cs, gs):
            cursor -= gap
            inp[rows[i], cursor - length:cursor] = color
            cursor -= length

    # Ground truth uses authored paths and layered event overwrites, not _parse,
    # _trace, solve, or _winner. Stencils are single logical clock sites.
    out = inp.copy()
    for i, path in enumerate(paths):
        timeline = np.full(len(path) + 2, -1, dtype=int)
        for length, color in reversed(list(zip(lengths[i], colors[i]))):
            timeline[::length] = color
        _paint_glyph(out, (rows[i], source_col), DIAMOND, int(timeline[0]))
        for tick, p in enumerate(path, 1):
            if inp[p] == carrier and timeline[tick] >= 0:
                out[p] = timeline[tick]
        if timeline[-1] >= 0:
            _paint_glyph(out, (path[-1][0], receiver_col), CHECKER, int(timeline[-1]))
    do_reflect = bool(rng.random() < .5) if reflect is None else reflect
    if do_reflect:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _teaching(train):
    evidence = {"singles": set(), "short_near": False, "long_near": False,
                "edges": set(), "lit_receiver": False, "silent_receiver": False,
                "far_receiver": False, "routing_receiver": False}
    for pair in train:
        info = _parse(pair["input"])
        out = np.rot90(np.asarray(pair["output"]), info["rotation"])
        for i, (blocks, path) in enumerate(zip(info["programs"], info["routes"])):
            if len(blocks) == 1 and blocks[0][0] > 1:
                length, color, _ = blocks[0]
                repeats = [t for t, p in enumerate(path, 1)
                           if t % length == 0 and info["grid"][p] == info["carrier"] and out[p] == color]
                quiet = any(t % length and out[p] == info["carrier"]
                            for t, p in enumerate(path, 1) if info["grid"][p] == info["carrier"])
                if len(repeats) >= 2 and quiet:
                    evidence["singles"].add(length)
            for near_idx, near in enumerate(blocks):
                for far in blocks[near_idx + 1:]:
                    if near[1] == far[1]:
                        continue
                    collision = any(t > 0 and t % near[0] == 0 and t % far[0] == 0
                                    and _winner(blocks, t) == near[1] and out[p] == near[1]
                                    for t, p in enumerate(path, 1) if info["grid"][p] == info["carrier"])
                    survives = any(_winner(blocks, t) == far[1] and out[p] == far[1]
                                   for t, p in enumerate(path, 1) if info["grid"][p] == info["carrier"])
                    if collision and survives:
                        evidence["short_near"] |= near[0] < far[0]
                        evidence["long_near"] |= near[0] > far[0]
                        evidence["edges"].add((near[1], far[1]))
            arrival = len(path) + 1
            actual = int(out[path[-1][0], info["receiver_col"]])
            expected = _winner(blocks, arrival)
            evidence["silent_receiver"] |= expected is None and actual == info["marker"]
            evidence["lit_receiver"] |= expected is not None and actual == expected
            evidence["far_receiver"] |= expected is not None and expected != blocks[0][1]
            j = info["destinations"][i]
            wrong = _winner(info["programs"][j], arrival)
            evidence["routing_receiver"] |= (i != j and expected != wrong)
    return evidence


def _assert_teaching(train):
    e = _teaching(train)
    assert len(e["singles"]) >= 2, "two different numerical recurrence lessons"
    assert e["short_near"] and e["long_near"], "both period/proximity orders"
    assert any((b, a) in e["edges"] for a, b in e["edges"]), "no fixed color priority"
    assert e["lit_receiver"] and e["silent_receiver"], "terminal samples need lit and silent cases"
    assert e["far_receiver"], "receiver must visibly sample a farther run too"
    assert e["routing_receiver"], "same-index terminal pairing is not distinguished"
    for priority in ("short", "long", "far"):
        assert any(not np.array_equal(_render(_parse(p["input"]), priority=priority), p["output"])
                   for p in train), f"rival priority fits: {priority}"
    for metric in ("solid", "projected", "skip_crossings", "gap_phase"):
        assert any(not np.array_equal(_render(_parse(p["input"]), metric=metric), p["output"])
                   for p in train), f"rival clock fits: {metric}"


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    params = dict(palette=palette, witness=[(2, 3), (2, 5), (3, 4)][int(rng.integers(3))],
                  witness_colors=palette[4:6], vary_palette=bool(rng.random() < .7),
                  shift=str(rng.choice(["count", "mixture", "rotation"])))
    base = int(rng.integers(4))
    mixed = bool(rng.random() < .5) and params["shift"] != "rotation"
    test_rotation = (base + int(rng.choice([1, 2, 3]))) % 4
    # Place the two mandatory lessons anywhere in the requested training set.
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    for curriculum_attempt in range(100):
        puzzle, seen = {"train": [], "test": []}, set()
        for index in range(nt):
            lesson = "priority" if index == slots[0] else "metric" if index == slots[1] else None
            rotation = int(rng.integers(4)) if mixed else base
            for attempt in range(50):
                pair = _construct_pair(rng, params, "train", lesson, rotation)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-pair construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle["train"].append(pair)
        try:
            _assert_teaching(puzzle["train"])
            assert params["witness"][1] in _teaching(puzzle["train"])["singles"], "largest period needs recurrence"
        except AssertionError:
            continue
        break
    else:
        raise RuntimeError("could not construct a visibly discriminating curriculum")
    for index in range(nq):
        rotation = test_rotation if params["shift"] == "rotation" else int(rng.integers(4)) if mixed else base
        for attempt in range(50):
            pair = _construct_pair(rng, params, "test", None, rotation)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("distinct test construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def inspect_pair(pair):
    """Semantic variation measurements reconstructed from input grids."""
    info = _parse(pair["input"])
    crossings = int(np.count_nonzero(info["grid"] == info["crossing"]))
    lengths = tuple(len(path) for path in info["routes"])
    bends = []
    for path in info["routes"]:
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        bends.append(sum(a != b for a, b in zip(directions, directions[1:])))
    programs = info["programs"]
    return dict(shape=tuple(np.asarray(pair["input"]).shape), direction=(-info["rotation"]) % 4,
                roles=(info["background"], info["carrier"], info["crossing"], info["marker"]),
                cables=len(programs), permutation=tuple(info["destinations"]), crossings=crossings,
                bends=tuple(bends), lengths=lengths, max_runs=max(map(len, programs)),
                periods=tuple(sorted({b[0] for bs in programs for b in bs})),
                row_gaps=tuple(np.diff(info["rows"]).tolist()),
                repeated_colors=any(len({b[1] for b in bs}) < len(bs) for bs in programs),
                nearest_gaps=tuple(bs[0][2] - 2 for bs in programs),
                receiver_colors=tuple(_winner(bs, len(path) + 1) for bs, path in zip(programs, info["routes"])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, _render(info)), "path clock or stencil answer disagrees"
            canonical_out = np.rot90(out, info["rotation"])
            mutable = info["grid"] == info["carrier"]
            for r in info["rows"]:
                mutable[r - 1:r + 2, info["source_col"] - 1:info["source_col"] + 2] |= DIAMOND
                mutable[r - 1:r + 2, info["receiver_col"] - 1:info["receiver_col"] + 2] |= CHECKER
            assert np.array_equal(canonical_out[~mutable], info["grid"][~mutable]), "changed protected cells"
            assert np.array_equal(canonical_out == info["background"], info["grid"] == info["background"])
            infos[split].append(info)
    _assert_teaching(puzzle["train"])
    largest_taught = max(_teaching(puzzle["train"])["singles"])
    assert all(b[0] <= largest_taught for info in infos["test"] for bs in info["programs"] for b in bs)
    # Each puzzle has at least one actual structural generalization, visible here.
    count_shift = min(len(i["rows"]) for i in infos["test"]) > max(len(i["rows"]) for i in infos["train"])
    mixture_shift = min(max(map(len, i["programs"])) for i in infos["test"]) > max(max(map(len, i["programs"])) for i in infos["train"])
    rotation_shift = {i["rotation"] for i in infos["train"]}.isdisjoint({i["rotation"] for i in infos["test"]})
    assert count_shift or mixture_shift or rotation_shift, "missing train/test shift"


def _puzzle_signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded family audit: hand arithmetic, metamorphisms, 42 seeds, all counts."""
    # A hand-written straight cable with eleven wire cells: source tick 0,
    # wire ticks 1..11, receiver tick 12. Near color 2 has period 2, far
    # color 3 period 3. The diamond/checker are clock sites, not 4/5 ticks.
    inp = np.zeros((7, 24), dtype=np.int8)
    inp[2, 1:4] = 3
    inp[2, 4:6] = 2
    _paint_glyph(inp, (2, 7), DIAMOND, 8)
    _paint_glyph(inp, (2, 21), CHECKER, 8)
    inp[2, 9:20] = 6
    expected = inp.copy()
    _paint_glyph(expected, (2, 7), DIAMOND, 2)
    expected[2, 9:20] = [6, 2, 3, 2, 6, 2, 6, 2, 3, 2, 6]
    _paint_glyph(expected, (2, 21), CHECKER, 2)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation))
    lut = np.array([5, 8, 9, 0, 2, 1, 4, 7, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(lut[inp]), lut[expected])

    # Minimal two-pin strand: ticks 1 and 2 on wire, far green at receiver
    # tick 3, despite red on the source. Supports finite-length clipping after
    # informative demonstrations, even though ordinary sampling uses longer wires.
    clipped = np.zeros((7, 15), dtype=np.int8)
    clipped[2, 1:4], clipped[2, 4:6] = 3, 2
    _paint_glyph(clipped, (2, 7), DIAMOND, 8)
    _paint_glyph(clipped, (2, 12), CHECKER, 8)
    clipped[2, 9:11] = 6
    clipped_answer = clipped.copy()
    _paint_glyph(clipped_answer, (2, 7), DIAMOND, 2)
    clipped_answer[2, 10] = 2
    _paint_glyph(clipped_answer, (2, 12), CHECKER, 3)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(clipped, rotation)), np.rot90(clipped_answer, rotation))

    # A legal strand that backtracks longitudinally. It is outside the sampler's
    # usual monotone geometry, but inside the specified visible input grammar.
    bent = np.zeros((15, 29), dtype=np.int8)
    bent[2, 1:4], bent[2, 4:6] = 3, 2
    _paint_glyph(bent, (2, 7), DIAMOND, 8)
    _paint_glyph(bent, (2, 26), CHECKER, 8)
    corners = [(2, 9), (2, 14), (8, 14), (8, 11), (12, 11), (12, 20), (2, 20), (2, 24)]
    hand_path = [corners[0]]
    for a, b in zip(corners, corners[1:]):
        dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
        p = a
        while p != b:
            p = (p[0] + dr, p[1] + dc)
            hand_path.append(p)
    for p in hand_path:
        bent[p] = 6
    assert len(hand_path) == 42
    info = _parse(bent)
    assert info["routes"] == [hand_path]
    result = solve(bent)
    for p, color in {(2, 9): 6, (2, 14): 2, (5, 14): 3, (8, 13): 6,
                     (8, 11): 3, (12, 11): 6, (12, 13): 3, (2, 21): 3,
                     (2, 24): 2, (2, 26): 8}.items():
        assert result[p] == color, "hand backtracking/arrival arithmetic"
    assert np.array_equal(solve(bent[:, ::-1]), result[:, ::-1])

    metrics, signatures, puzzles = [], set(), []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        assert _puzzle_signature(p) == _puzzle_signature(generate_puzzle(seed))
        signatures.add(_puzzle_signature(p))
        puzzles.append(p)
        metrics.extend(inspect_pair(pair) for pair in p["train"] + p["test"])
    assert len(signatures) == 42
    assert len({m["shape"] for m in metrics}) >= 35
    assert {m["direction"] for m in metrics} == {0, 1, 2, 3}
    assert {m["cables"] for m in metrics} == {1, 2, 3, 4}
    assert {m["max_runs"] for m in metrics} == {1, 2, 3}
    assert {p for m in metrics for p in m["periods"]} == {1, 2, 3, 4, 5}
    assert any(m["crossings"] == 0 for m in metrics) and max(m["crossings"] for m in metrics) >= 5
    assert any(m["permutation"] == tuple(range(m["cables"])) for m in metrics)
    assert any(m["permutation"] != tuple(range(m["cables"])) for m in metrics)
    assert any(m["repeated_colors"] for m in metrics)
    assert any(len(set(m["row_gaps"])) > 1 for m in metrics)
    assert max(max(m["bends"]) for m in metrics) >= 4
    assert len({n for m in metrics for n in m["lengths"]}) >= 25
    assert len({m["roles"] for m in metrics}) >= 20
    for p in puzzles:
        local = [inspect_pair(pair) for pair in p["train"] + p["test"]]
        assert len({m["shape"] for m in local}) > 1
        assert len({m["cables"] for m in local}) > 1
        assert len({m["lengths"] for m in local}) > 1

    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, True), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    # Input-only solver supports general legal strand geometry and every small
    # permutation, independently of which ones the 42-seed sample happened to draw.
    rng = np.random.default_rng(7421)
    params = dict(witness=(2, 3), witness_colors=(2, 3),
                  palette=[0, 6, 9, 8, 2, 3, 1, 4, 5, 7],
                  vary_palette=False, shift="rotation")
    for n in range(1, 5):
        for target in permutations(range(n)):
            moves = _routing_moves(target, rng)
            state = list(range(n)) + [None]
            for old, new in moves:
                assert state[old] is not None and state[new] is None
                state[new], state[old] = state[old], None
            assert state[:n] == list(target)
            pair = _construct_pair(rng, params, "train", None, 0,
                                   forced_lengths=[[2]] * n, forced_target=target, reflect=False)
            info = _parse(pair["input"])
            assert info["destinations"] == [target.index(i) for i in range(n)]
            assert np.array_equal(solve(pair["input"]), pair["output"])

    p = puzzles[0]
    for mode in ("protected", "pulse", "receiver"):
        damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = damaged["train"][0]
        info = _parse(pair["input"])
        out = np.rot90(pair["output"], info["rotation"])
        if mode == "protected":
            pos = (0, 0)
        elif mode == "pulse":
            pos = (info["rows"][0], info["source_col"] + 1)
        else:
            pos = (info["routes"][0][-1][0], info["receiver_col"])
        out[pos] = (int(out[pos]) + 1) % 10
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted output corruption: {mode}")

    # This report is descriptive, authored evidence, not independent approval.
    report = dict(seeds=list(HOST_SEEDS), pairs=len(metrics), puzzles=len(signatures),
                  shapes=len({m["shape"] for m in metrics}), directions=sorted({m["direction"] for m in metrics}),
                  cable_counts=sorted({m["cables"] for m in metrics}),
                  max_runs=sorted({m["max_runs"] for m in metrics}),
                  periods=sorted({p for m in metrics for p in m["periods"]}),
                  crossing_range=[min(m["crossings"] for m in metrics), max(m["crossings"] for m in metrics)],
                  path_length_range=[min(n for m in metrics for n in m["lengths"]), max(n for m in metrics for n in m["lengths"])],
                  repeated_color_pairs=sum(m["repeated_colors"] for m in metrics),
                  note="Generator-owned invariants and rival-rule checks only; no independent solver or blind prediction.")
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test_report.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
