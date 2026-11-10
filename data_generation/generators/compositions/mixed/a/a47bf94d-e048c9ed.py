"""Squared bar-adjacency digits carried through an orthogonal cable permutation.

No parent modules, files, random state, or demonstrations are used by solve.
Generation paints answers from constructed paths; solve traces the visible graph.
All diagnostics are in memory; this module never writes grids or traces to disk.
"""
from __future__ import annotations

from itertools import permutations
import hashlib
import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
LENGTHS = (2, 3, 4, 5)
DIGITS = frozenset((1, 4, 6, 9))
HOST_SEEDS = (0, 81989278, 550005676, 64125362, 9745086, 798148734,
              262138722, 55102487, 56158003, 20824448, 602792380, 779258604,
              552940921, 374538010, 475875606, 825195234, 811749189, 211043715,
              691195913, 125433090, 184582787, 27748573, 938212389, 277187332,
              468325827, 970074725, 780106392, 467196917, 275195710, 348319461,
              367157603, 392285706, 100758948, 545448927, 953556737, 987854726,
              581229899, 792871224, 589708422, 311430185, 279950414, 467617420)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must be integral")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        found, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _parse(input_grid):
    """Identify bars/reference geometrically, then trace actual carrier cells."""
    original = _grid(input_grid)
    corners = [int(original[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    _require(len(set(corners)) == 1, "background corners disagree")
    bg = corners[0]
    _require(bg not in DIGITS, "background obscures a digit")
    foreground = set(map(tuple, np.argwhere(original != bg)))
    components = _components(foreground)
    singles = [next(iter(cells)) for cells in components if len(cells) == 1]
    _require(len(singles) == 1, "expected one isolated reference cell")
    marker = singles[0]
    reference_column = int(marker[1])
    bars = []
    for cells in components:
        rows = {p[0] for p in cells}
        if len(rows) == 1 and len(cells) in LENGTHS:
            r = next(iter(rows))
            columns = sorted(p[1] for p in cells)
            _require(columns == list(range(columns[0], columns[-1] + 1)), "broken bar")
            colors = {int(original[p]) for p in cells}
            _require(len(colors) == 1, "bar must be monochrome")
            bars.append((int(r), int(columns[0]), len(cells), colors.pop()))
    bars.sort()
    _require(2 <= len(bars) <= 5, "expected 2..5 source bars")
    left = all(c + length - 1 == reference_column for r, c, length, color in bars)
    right = all(c == reference_column for r, c, length, color in bars)
    _require(left != right, "bars must attach outside one reference bank")
    flipped = right
    g = original[:, ::-1] if flipped else original
    if flipped:
        reference_column = g.shape[1] - 1 - reference_column
        marker = (int(marker[0]), reference_column)
        bars = [(r, g.shape[1] - c - length, length, color)
                for r, c, length, color in bars]
    sc = reference_column + 2
    bar_cells = {(r, x) for r, c, length, color in bars for x in range(c, c + length)}
    foreground = set(map(tuple, np.argwhere(g != bg)))
    wires = foreground - bar_cells - {marker}
    _require(bool(wires), "missing routing field")
    start = sc + 2
    end = max(c for r, c in wires)
    _require(min(c for r, c in wires) == start and end - start >= 6,
             "routing field must lie inward of the source terminals")
    rc = end + 2
    rows = [r for r, c, length, color in bars]
    _require(len(set(rows)) == len(rows), "multiple bars in one row")
    _require(all(b - a >= 4 for a, b in zip(rows, rows[1:])), "terminal sites overlap")
    _require(sorted(r for r, c in wires if c == start) == rows,
             "source ports do not align with bars")
    _require(sorted(r for r, c in wires if c == end) == rows,
             "receiver ports do not form the opposite bank")
    _require(sorted(map(int, np.flatnonzero(g[:, reference_column] != bg))) ==
             sorted(rows + [int(marker[0])]), "reference column must contain only bar tips and marker")
    footprint = np.zeros(g.shape, dtype=bool)
    for r in rows:
        for c in (sc, rc):
            _require(1 <= r < g.shape[0] - 1 and 1 <= c < g.shape[1] - 1,
                     "terminal footprint goes out of bounds")
            _require(bool(np.all(g[r - 1:r + 2, c - 1:c + 2] == bg)),
                     "input terminal footprints must be blank")
            footprint[r - 1:r + 2, c - 1:c + 2] = True
    carrier = int(g[rows[0], start])
    wire_colors = {int(g[p]) for p in wires}
    _require(len(wire_colors) in (1, 2) and not (wire_colors & DIGITS),
             "routing colors must be distinct from numeric digits")
    _require(all(int(g[r, start]) == carrier and int(g[r, end]) == carrier for r in rows),
             "terminal ports must use the carrier color")
    _require(all(color not in wire_colors for r, c, length, color in bars)
             and int(g[marker]) not in wire_colors, "structural color collision")
    cross = next(iter(wire_colors - {carrier}), None)
    endpoints = {(r, start) for r in rows} | {(r, end) for r in rows}
    graph = {}
    crossings = []
    for r, c in sorted(wires):
        edges = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(DIRECTIONS)
                 if (r + dr, c + dc) in wires}
        graph[r, c] = edges
        if (r, c) in endpoints:
            _require(len(edges) == 1 and int(g[r, c]) == carrier, "invalid endpoint")
        elif int(g[r, c]) == cross:
            _require(len(edges) == 4, "crossing marker must have four ports")
            crossings.append((r, c))
        else:
            _require(len(edges) == 2, "branch, unmarked crossing or broken cable")
    permutation, routes, visited = [], [], set()
    for row in rows:
        p, incoming = (row, start), 3
        route, states = [], set()
        while True:
            _require((p, incoming) not in states, "cyclic cable")
            states.add((p, incoming))
            route.append(p)
            visited.add(p)
            if p[1] == end:
                permutation.append(rows.index(p[0]))
                break
            edges = graph[p]
            if p == (row, start):
                outgoing = 1
            elif len(edges) == 4:
                outgoing = (incoming + 2) % 4
            else:
                _require(incoming in edges, "invalid incoming direction")
                outgoing = next(d for d in edges if d != incoming)
            _require(outgoing in edges, "cable cannot continue")
            p = edges[outgoing]
            incoming = (outgoing + 2) % 4
            _require(p[1] != start, "cable returns to source bank")
        routes.append(route)
    _require(sorted(permutation) == list(range(len(rows))), "nonbijective cable pairing")
    _require(visited == wires, "unattached routing fragment")
    return dict(grid=g, background=bg, source_column=sc, receiver_column=rc,
                reference_column=reference_column,
                marker=marker, bars=bars, rows=rows, carrier=carrier, cross=cross,
                wires=wires, routes=routes, crossings=crossings,
                permutation=permutation, footprint=footprint, flipped=flipped)


def _complete(info, labels=None, permutation=None):
    out = info["grid"].copy()
    if labels is None:
        labels = [(length - 1) ** 2 % 10 for r, c, length, color in info["bars"]]
    if permutation is None:
        permutation = info["permutation"]
    for i, j in enumerate(permutation):
        for row, column, stencil in ((info["rows"][i], info["source_column"], DIAMOND),
                                     (info["rows"][j], info["receiver_column"], CHECKER)):
            out[row - 1:row + 2, column - 1:column + 2] = np.where(
                stencil, labels[i], info["background"])
    return out[:, ::-1].copy() if info["flipped"] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute literal units digits and propagate them along input-visible cables."""
    return _complete(_parse(input_grid))


def _vacancy_moves(target, rng):
    """Primitive routing construction: move permutation cycles through a spare lane."""
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = state[j], None
        vacant = j
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = state[n], None
    assert state[:n] == list(target)
    return moves


def _make_pair(rng, palette, lengths, colors, target, source_right):
    n = len(lengths)
    bg, carrier, cross, marker_color = palette
    moves = _vacancy_moves(target, rng)
    if len(moves) <= 5 and rng.random() < .25:
        lane = int(rng.integers(n))
        moves = [(lane, n), (n, lane)] + moves
    lead, tail = (2, 2) if moves else (3, 3)
    gaps = [2] * max(0, len(moves) - 1)
    span = lead + tail + sum(gaps)
    sc = max(lengths) + 2
    budget = 30 - sc - 7 - span
    assert budget >= 0
    offset = int(rng.integers(min(3, budget) + 1))
    sc += offset
    budget -= offset
    right_padding = int(rng.integers(min(3, budget) + 1))
    budget -= right_padding
    pieces = [lead, tail] + gaps
    for _ in range(int(rng.integers(budget + 1))):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, gaps = pieces[0], pieces[1], pieces[2:]
    start = sc + 2
    move_columns = []
    column = start + lead
    for k in range(len(moves)):
        move_columns.append(column)
        if k < len(gaps):
            column += gaps[k]
    end = move_columns[-1] + tail if moves else start + lead + tail
    rc, width = end + 2, end + 5 + right_padding
    max_pitch = min(6, 20 // (n - 1))
    pitch = int(rng.integers(4, max_pitch + 1))
    spare_gap = int(rng.integers(3, min(6, 23 - (n - 1) * pitch) + 1))
    height0 = 7 + (n - 1) * pitch + spare_gap
    top = int(rng.integers(min(3, 30 - height0) + 1))
    bottom = int(rng.integers(min(3, 30 - height0 - top) + 1))
    rows = [3 + top + i * pitch for i in range(n)]
    lanes = rows + [rows[-1] + spare_gap]
    height = lanes[-1] + 4 + bottom
    if rng.random() < .5:
        marker_row = int(rng.integers(1, rows[0] - 1))
    else:
        marker_row = int(rng.integers(lanes[-1] + 2, height - 1))
    paths = [[(r, start)] for r in rows]
    state, previous = list(range(n)) + [None], start
    for column, (old, new) in zip(move_columns, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((lanes[lane], c) for c in range(previous + 1, column + 1))
        route = state[old]
        step = 1 if lanes[new] > lanes[old] else -1
        paths[route].extend((r, column) for r in range(lanes[old] + step, lanes[new] + step, step))
        state[new], state[old] = state[old], None
        previous = column
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((lanes[lane], c) for c in range(previous + 1, end + 1))
    inp = np.full((height, width), bg, dtype=np.int8)
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            visits.setdefault(p, set()).add(i)
    for p, owners in visits.items():
        assert len(owners) <= 2
        inp[p] = cross if len(owners) == 2 else carrier
    inp[marker_row, sc - 2] = marker_color
    for r, length, color in zip(rows, lengths, colors):
        inp[r, sc - 1 - length:sc - 1] = color
    out = inp.copy()
    # Ground truth uses the constructed cable endpoints, never _parse or solve.
    for i, path in enumerate(paths):
        digit = (lengths[i] - 1) ** 2 % 10
        for r, c, pattern in ((rows[i], sc, DIAMOND), (path[-1][0], rc, CHECKER)):
            out[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, digit, bg)
    if rng.random() < .5:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    if source_right:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    assert max(inp.shape) <= 30
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    structural = [int(x) for x in rng.permutation(sorted(set(range(10)) - DIGITS))]
    bg, carrier, cross = structural[:3]
    available = [c for c in range(10) if c not in (bg, carrier, cross)]
    a, b = map(int, rng.choice(available, 2, replace=False))
    marker = int(rng.choice(available))
    palette = (bg, carrier, cross, marker)
    direction_mode = int(rng.integers(3))  # left, right, or both explicitly taught
    count_shift = bool(rng.integers(2))
    first_lengths = list(map(int, rng.permutation(LENGTHS)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            if split == "train" and index == 0:
                n, lengths, colors = 4, first_lengths[:], [a] * 4
            elif split == "train" and index == 1:
                n = int(rng.integers(2, 5))
                lengths, colors = first_lengths[:n], [b] * n
            else:
                n = (5 if count_shift else int(rng.integers(3, 6))) if split == "test" else int(rng.integers(2, 5))
                lengths = list(map(int, rng.choice(LENGTHS, n)))
                colors = list(map(int, rng.choice(available, n)))
            if split == "train":
                target = list(range(n))
                if index == 0 or (index > 1 and rng.random() < .6):
                    j, k = map(int, rng.choice(n, 2, replace=False))
                    target[j], target[k] = target[k], target[j]
            else:
                # A query composes already demonstrated crossings into a longer
                # cycle, or increases the bundle size. No new arithmetic code.
                target = list(map(int, rng.permutation(n)))
                if sum(i != v for i, v in enumerate(target)) < 3:
                    target = list(np.roll(np.arange(n), int(rng.integers(1, n))))
                if all(lengths[target[j]] == lengths[j] for j in range(n)):
                    j = next(j for j in range(n) if target[j] != j)
                    lengths[j], lengths[target[j]] = 2, 3
            source_right = (direction_mode == 1 or
                            (direction_mode == 2 and (bool(index % 2) if split == "train"
                                                      else bool(rng.integers(2)))))
            for _ in range(24):
                pair = _make_pair(rng, palette, lengths, colors, target, source_right)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output changed dimensions")
    info = _parse(inp)
    _require(np.array_equal(_complete(info), out), "incorrect arithmetic, pairing or stencil")
    canonical_out = out[:, ::-1] if info["flipped"] else out
    _require(np.array_equal(info["grid"][~info["footprint"]],
                           canonical_out[~info["footprint"]]), "nonterminal cells changed")
    _require(np.array_equal(info["grid"][:, info["reference_column"]],
                           canonical_out[:, info["reference_column"]]), "reference column changed")
    n = len(info["bars"])
    _require(np.count_nonzero(inp != out) == 9 * n, "expected four plus five new cells per route")
    for i, j in enumerate(info["permutation"]):
        digit = (info["bars"][i][2] - 1) ** 2 % 10
        r, s = info["rows"][i], info["source_column"]
        t, c = info["rows"][j], info["receiver_column"]
        _require(np.count_nonzero(canonical_out[r - 1:r + 2, s - 1:s + 2] == digit) == 4,
                 "source stencil population is not four")
        _require(np.count_nonzero(canonical_out[t - 1:t + 2, c - 1:c + 2] == digit) == 5,
                 "receiver stencil population is not five")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            info = _inspect_pair(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    roles = {(i["background"], i["carrier"], int(i["grid"][i["marker"]]))
             for group in infos.values() for i in group}
    _require(len(roles) == 1, "puzzle structural conventions changed")
    crosses = {i["cross"] for group in infos.values() for i in group if i["cross"] is not None}
    _require(len(crosses) == 1, "crossing role not taught consistently")
    length_colors, color_lengths = {}, {}
    trace_needed = arithmetic_needed = adjacency_needed = modulo_needed = crossing_needed = False
    for pair, info in zip(puzzle["train"], infos["train"]):
        for r, c, length, color in info["bars"]:
            length_colors.setdefault(length, set()).add(color)
            color_lengths.setdefault(color, set()).add(length)
            modulo_needed |= (length - 1) ** 2 >= 10
        out = np.asarray(pair["output"])
        arithmetic_needed |= not np.array_equal(out, _complete(
            info, labels=[length - 1 for r, c, length, color in info["bars"]]))
        adjacency_needed |= not np.array_equal(out, _complete(
            info, labels=[length ** 2 % 10 for r, c, length, color in info["bars"]]))
        trace_needed |= not np.array_equal(out, _complete(info, permutation=list(range(len(info["bars"])))))
        labels = [(bar[2] - 1) ** 2 % 10 for bar in info["bars"]]
        for crossing in info["crossings"]:
            crossing_needed |= len({labels[k] for k, path in enumerate(info["routes"])
                                    if crossing in path}) == 2
    _require(set(length_colors) == set(LENGTHS), "training must expose all four length codes")
    _require(any(len(v) >= 2 for v in length_colors.values()), "length must recur in another color")
    _require(any(len(v) >= 2 for v in color_lengths.values()), "one color must carry different lengths")
    _require(trace_needed and arithmetic_needed and adjacency_needed and modulo_needed and crossing_needed,
             "training does not distinguish both parent mechanisms")
    taught_sides = {i["flipped"] for i in infos["train"]}
    _require(all(i["flipped"] in taught_sides for i in infos["test"]), "untaught bank direction")
    taught_digits = {length: (length - 1) ** 2 % 10 for length in length_colors}
    for pair, info in zip(puzzle["test"], infos["test"]):
        labels = [taught_digits[b[2]] for b in info["bars"]]
        _require(np.array_equal(_complete(info, labels), pair["output"]), "test uses an untaught code")
        _require(not np.array_equal(_complete(info, permutation=list(range(len(labels)))),
                                    pair["output"]), "query does not require cable transport")
    largest_training = max(len(i["bars"]) for i in infos["train"])
    _require(all(len(i["bars"]) > largest_training or
                 sum(j != k for j, k in enumerate(i["permutation"])) >= 3
                 for i in infos["test"]), "query lacks the intended composition shift")
    return None


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = _grid(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and exhaustive small routing/count cases; write no files."""
    shapes, counts, pitches, crossings, layouts, roles, sides = (set() for _ in range(7))
    digests = set()
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                pairs += 1
                shapes.add(pair["input"].shape)
                counts.add(len(info["bars"]))
                pitches.add(info["rows"][1] - info["rows"][0])
                crossings.add(len(info["crossings"]))
                sides.add(info["flipped"])
                roles.add((info["background"], info["carrier"], info["cross"]))
                layouts.add((tuple(info["rows"]), tuple(info["permutation"]),
                             tuple(b[2] for b in info["bars"]), tuple(map(len, info["routes"]))))
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "solver disagrees")
                _require(np.array_equal(before, pair["input"]), "solve mutated its input")
    _require(len(digests) == 42 and len(shapes) >= 35 and len(layouts) >= 150,
             "insufficient structural variation")
    _require(counts == {2, 3, 4, 5} and pitches >= {4, 5, 6} and sides == {False, True},
             "missing count, pitch or source-side coverage")
    _require(0 in crossings and max(crossings) >= 8 and len(roles) >= 15,
             "missing simple/dense or palette-role coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(77123 + nt * 10 + nq, nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0)),
             "seed is not deterministic")
    # Exhaust all 120 five-cable pairings under the largest permitted bar domain.
    rng = np.random.default_rng(123901)
    for target in permutations(range(5)):
        pair = _make_pair(rng, (0, 8, 3, 2), [2, 3, 4, 5, 2], [2] * 5, target, False)
        info = _inspect_pair(pair)
        _require(info["permutation"] in ([target.index(i) for i in range(5)],
                 [4 - target.index(4 - i) for i in range(5)]), "routing construction failed")
    p = generate_puzzle(0)
    p["test"][0]["output"][0, 0] = 1
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("nonterminal corruption accepted")
    return {"seeds": 42, "pairs": pairs, "shapes": len(shapes), "layouts": len(layouts),
            "route_counts": sorted(counts), "pitches": sorted(pitches),
            "crossing_counts": sorted(crossings), "role_palettes": len(roles),
            "five_route_permutations": 120, "legal_count_combinations": 30}
