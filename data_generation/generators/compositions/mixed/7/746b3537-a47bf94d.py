"""Complete a cable diagram, then collapse adjacent equal pixel vectors.

Only NumPy and the standard library are used. Generation constructs paths;
the solver instead discovers a directional graph from the input pixels.
"""
from __future__ import annotations

from itertools import permutations
import hashlib

import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 16948254, 791514428, 676360778, 264712479, 276507056,
              787724801, 671223538, 994153479, 669026353, 476023675,
              869403687, 712210225, 794301122, 263968554, 103706856,
              193111972, 128766218, 265951362, 251352181, 565438516,
              714977957, 42526136, 912488813, 36024433, 609543237,
              732038720, 418229690, 501761874, 829387703, 372662279,
              520096195, 91392134, 758692722, 206906216, 255840740,
              210078872, 103801094, 348448061, 682836209, 845646446,
              177185396)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _keep(a):
    # Both masks refer to this same, completed canvas. No global uniqueness.
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return rows, cols


def collapse(a):
    """Keep the first vector of each consecutive equality run on each axis."""
    r, c = _keep(a)
    return a[np.ix_(r, c)].copy()


def _motif(pattern, bg, color):
    return np.where(pattern, color, bg).astype(np.int8)


def _kind(block, bg):
    colors = set(map(int, np.unique(block))) - {bg}
    if not colors:
        return "blank", None
    if len(colors) != 1:
        raise ValueError("multicolor terminal")
    color = colors.pop()
    if np.all(block == color):
        return "solid", color
    for name, pattern in (("diamond", DIAMOND), ("checker", CHECKER)):
        if np.array_equal(block, _motif(pattern, bg, color)):
            return name, color
    raise ValueError("malformed native 3x3 terminal")


def _trace(field, start, end, centers, carrier, cross):
    visible = set(map(tuple, np.argwhere((field == carrier) | (field == cross))))
    graph = {}
    endpoints = {(start, c): 2 for c in centers}
    endpoints.update({(end, c): 0 for c in centers})
    crossings = set()
    for r, c in visible:
        edges = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(DIRECTIONS)
                 if (r + dr, c + dc) in visible}
        graph[r, c] = edges
        if (r, c) in endpoints:
            if set(edges) != {endpoints[r, c]}:
                raise ValueError("invalid cable endpoint")
        elif len(edges) == 4:
            if cross is None or not any(all(field[edges[d]] == cross for d in ds)
                                        for ds in ((0, 2), (1, 3))):
                raise ValueError("unmarked junction")
            crossings.add((r, c))
        elif len(edges) != 2:
            raise ValueError("broken or branching cable")
    pairing, routes, seen = [], [], set()
    for c in centers:
        p, incoming, route, states = (start, c), 0, [], set()
        while True:
            if (p, incoming) in states:
                raise ValueError("cyclic cable")
            states.add((p, incoming)); seen.add(p); route.append(p)
            if p[0] == end and p in endpoints:
                pairing.append(centers.index(p[1])); break
            if p == (start, c):
                outgoing = 2
            elif p in crossings:
                outgoing = (incoming + 2) % 4
            else:
                choices = set(graph[p]) - {incoming}
                if len(choices) != 1:
                    raise ValueError("ambiguous strand")
                outgoing = choices.pop()
            p = graph[p][outgoing]
            incoming = (outgoing + 2) % 4
            if p[0] == start:
                raise ValueError("route returns to near bank")
        routes.append(route)
    if sorted(pairing) != list(range(len(centers))) or seen != visible:
        raise ValueError("routing is not a complete bijection")
    return pairing, routes, crossings


def _parse(input_grid):
    a = _grid(input_grid)
    corners = (int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1]))
    if len(set(corners)) != 1:
        raise ValueError("background must surround the diagram")
    bg, candidates = corners[0], []
    for transpose in (False, True):
        g = a.T if transpose else a
        for carrier in sorted(set(map(int, np.unique(g))) - {bg}):
            positions = np.argwhere(g == carrier)
            start, end = int(positions[:, 0].min()), int(positions[:, 0].max())
            if start < 4 or end > g.shape[0] - 5 or end - start < 3:
                continue
            centers = list(map(int, np.flatnonzero(g[start] == carrier)))
            if (not 2 <= len(centers) <= 5 or centers != list(np.flatnonzero(g[end] == carrier))
                    or centers[0] < 2 or centers[-1] > g.shape[1] - 3
                    or any(b - c < 4 for c, b in zip(centers, centers[1:]))):
                continue
            try:
                footprint = np.zeros(g.shape, dtype=bool)
                banks = []
                for row in (start - 3, end + 1):
                    bank = []
                    for c in centers:
                        footprint[row:row + 3, c - 1:c + 2] = True
                        bank.append(_kind(g[row:row + 3, c - 1:c + 2], bg))
                    banks.append(bank)
                outside = ~footprint
                outside[start:end + 1] = False
                if not np.all(g[outside] == bg):
                    continue
                field = np.full_like(g, bg)
                field[start:end + 1] = g[start:end + 1]
                others = set(map(int, np.unique(field))) - {bg, carrier}
                if len(others) > 1:
                    continue
                cross = next(iter(others)) if others else None
                labels = {v for bank in banks for _, v in bank if v is not None}
                if not labels.isdisjoint({carrier, cross}):
                    continue
                pairing, routes, crossings = _trace(field, start, end, centers, carrier, cross)
                for phase in (0, 1):
                    route_labels, clues = [], []
                    for i, j in enumerate(pairing):
                        near, far = banks[0][i], banks[1][j]
                        source, receiver = (near, far) if phase == 0 else (far, near)
                        if source[0] not in ("blank", "solid", "diamond") or receiver[0] not in ("blank", "checker"):
                            break
                        colors = {v for _, v in (source, receiver) if v is not None}
                        if len(colors) != 1:
                            break
                        route_labels.append(colors.pop()); clues.append((source[0], receiver[0]))
                    else:
                        candidates.append(dict(grid=g, transpose=transpose, bg=bg,
                                               carrier=carrier, cross=cross, start=start, end=end,
                                               centers=centers, pairing=pairing, routes=routes,
                                               crossings=crossings, footprint=footprint,
                                               phase=phase, labels=route_labels, clues=clues))
            except (ValueError, KeyError, IndexError):
                continue
    if len(candidates) != 1:
        raise ValueError(f"expected one visible interpretation, found {len(candidates)}")
    return candidates[0]


def _complete(info, pairing=None):
    g = info["grid"].copy()
    for i, j in enumerate(info["pairing"] if pairing is None else pairing):
        for bank, index, row in ((0, i, info["start"] - 3), (1, j, info["end"] + 1)):
            pattern = DIAMOND if bank == info["phase"] else CHECKER
            c = info["centers"][index]
            g[row:row + 3, c - 1:c + 2] = _motif(pattern, info["bg"], info["labels"][i])
    return g.T.copy() if info["transpose"] else g


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, endpoint pairs and bank phase solely from input cells."""
    return collapse(_complete(_parse(input_grid)))


def _moves(target, rng):
    # Move cycles through one vacant exterior track; no diagonal swap scaffold.
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for slot in map(int, rng.permutation(n)):
        if state[slot] == target[slot]:
            continue
        saved = state[slot]
        moves.append((slot, n)); state[n], state[slot] = saved, None
        vacant = slot
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant)); state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant)); state[vacant], state[n] = saved, None
    assert state[:n] == list(target)
    return moves


def _field(rng, n, target, detour=False):
    moves = _moves(target, rng)
    if detour and len(moves) <= 6:
        lane = int(rng.integers(n))
        moves = [(lane, n), (n, lane)] + moves
    # Each pair gets irregular inter-terminal spacing and exterior margins.
    gaps = [4] * (n - 1)
    width_budget = 30 - (2 + sum(gaps) + 4 + 3)
    for _ in range(int(rng.integers(min(6, width_budget) + 1))):
        gaps[int(rng.integers(n - 1))] += 1
    centers = [2]
    for gap in gaps:
        centers.append(centers[-1] + gap)
    spare = centers[-1] + 4
    extra_width = 30 - (spare + 3)
    left = int(rng.integers(min(2, extra_width) + 1))
    centers = [c + left for c in centers]; spare += left
    right = 2 + int(rng.integers(min(2, extra_width - left) + 1))
    width = spare + 1 + right
    columns = centers + [spare]
    pieces = [2, 2] + [2] * max(0, len(moves) - 1)
    # Reserve two exterior rows so even the densest marked routing has a
    # genuine equality run. A marked crossing can otherwise split every lead.
    minimum_height = 10 + sum(pieces)
    # For no moves, a lead and tail still form one straight repeated band.
    spare_height = 30 - minimum_height
    top = 2 + int(rng.integers(min(1, spare_height) + 1)); spare_height -= top - 2
    bottom = 1 + int(rng.integers(min(2, spare_height) + 1)); spare_height -= bottom - 1
    for _ in range(int(rng.integers(min(6, spare_height) + 1))):
        pieces[int(rng.integers(len(pieces)))] += 1
    start = top + 3
    rows = [start + pieces[0]]
    for gap in pieces[2:]:
        rows.append(rows[-1] + gap)
    end = (rows[-1] if moves else start + pieces[0]) + pieces[1]
    height = end + 4 + bottom
    paths = [[(start, c)] for c in centers]
    state, previous = list(range(n)) + [None], start
    for row, (old, new) in zip(rows, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        route = state[old]
        step = 1 if new > old else -1
        paths[route].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    field = np.zeros((height, width), dtype=np.int8)
    owners = {}
    for index, path in enumerate(paths):
        for p in path:
            field[p] = 1; owners.setdefault(p, set()).add(index)
    for (r, c), visits in sorted(owners.items()):
        if len(visits) == 2:
            for d in ((0, 2) if rng.integers(2) else (1, 3)):
                dr, dc = DIRECTIONS[d]
                assert field[r + dr, c + dc]
                field[r + dr, c + dc] = 2
    return field, centers, start, end, paths


def _make_pair(rng, n, lesson, orientation, palette, phase):
    target = list(map(int, rng.permutation(n)))
    if lesson == "forward":
        target = list(range(n)); target[0], target[-1] = target[-1], target[0]
    elif rng.random() < .2:
        target = list(range(n))
    field, centers, start, end, paths = _field(rng, n, target, rng.random() < .25)
    bg = palette[0]
    labels = list(map(int, rng.choice(palette[3:], n, replace=False)))
    if lesson is None and rng.random() < .4 and n >= 3:
        labels[-1] = labels[0]
    options = (("solid", "blank"), ("diamond", "blank"), ("blank", "checker"),
               ("solid", "checker"), ("diamond", "checker"))
    if lesson == "forward":
        clues = [options[0]] * n
    elif lesson == "reverse":
        clues = [options[2], options[4], options[0]]
        clues += [options[int(rng.integers(5))] for _ in range(n - 3)]
        rng.shuffle(clues)
    else:
        mode = int(rng.integers(4))
        clues = [options[2] if mode == 0 else options[0] if mode == 1
                 else options[int(rng.integers(5))] for _ in range(n)]
        if all(a == "diamond" and b == "checker" for a, b in clues):
            clues[0] = options[0]
    inp = np.asarray(palette, dtype=np.int8)[field]
    completed = inp.copy()
    # Ground truth uses explicit constructed endpoints, never the solver.
    for i, path in enumerate(paths):
        j = centers.index(path[-1][1])
        for bank, index, row in ((0, i, start - 3), (1, j, end + 1)):
            pattern = DIAMOND if bank == phase else CHECKER
            mode = clues[i][0 if bank == phase else 1]
            c = centers[index]
            completed[row:row + 3, c - 1:c + 2] = _motif(pattern, bg, labels[i])
            block = np.full((3, 3), bg if mode == "blank" else labels[i], dtype=np.int8)
            if mode in ("diamond", "checker"):
                block = _motif(DIAMOND if mode == "diamond" else CHECKER, bg, labels[i])
            inp[row:row + 3, c - 1:c + 2] = block
    if rng.integers(2):
        inp, completed = inp[:, ::-1].copy(), completed[:, ::-1].copy()
    if orientation:
        inp, completed = inp.T.copy(), completed.T.copy()
    return {"input": inp, "output": collapse(completed)}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    phase = int(rng.integers(2))
    shift = int(rng.integers(2))  # first query expands or contracts the bundle
    orientations = [0, 1] + [int(rng.integers(2)) for _ in range(nt - 2)]
    rng.shuffle(orientations)
    lessons = ["forward", "reverse"] + [None] * (nt - 2)
    rng.shuffle(lessons)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", nt), ("test", nq)):
        for index in range(total):
            lesson = lessons[index] if split == "train" else None
            n = (int(rng.integers(3, 5)) if split == "train" else
                 (5 if shift == 0 else 2) if index == 0 else int(rng.integers(2, 6)))
            orientation = orientations[index] if split == "train" else int(rng.integers(2))
            for _ in range(24):
                pair = _make_pair(rng, n, lesson, orientation, palette, phase)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
            seen.add(_key(pair["input"])); puzzle[split].append(pair)
    return puzzle


def inspect_pair(pair):
    info = _parse(pair["input"])
    completed = _complete(info)
    rows, cols = _keep(completed)
    pre_r, pre_c = _keep(np.asarray(pair["input"]))
    wrong = collapse(_complete(info, list(range(len(info["centers"])))))
    return dict(shape=list(np.asarray(pair["input"]).shape), output_shape=list(collapse(completed).shape),
                orientation="horizontal" if info["transpose"] else "vertical", phase=info["phase"],
                routes=len(info["centers"]), pairing=info["pairing"], crossings=len(info["crossings"]),
                pitches=list(map(int, np.diff(info["centers"]))), roles=[info["bg"], info["carrier"], info["cross"]],
                labels=info["labels"], clues=info["clues"], removed_rows=int((~rows).sum()),
                removed_columns=int((~cols).sum()),
                premature_rows_split=int(np.count_nonzero(rows & ~pre_r)),
                premature_columns_split=int(np.count_nonzero(cols & ~pre_c)),
                identity_pairing_wrong=not np.array_equal(wrong, collapse(completed)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, conventions = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen; seen.add(_key(inp))
            info = _parse(inp)
            conventions.add((info["bg"], info["carrier"], info["cross"], info["phase"]))
            completed = _complete(info)
            canonical_completed = completed.T if info["transpose"] else completed
            assert np.array_equal(canonical_completed[~info["footprint"]], info["grid"][~info["footprint"]])
            assert np.array_equal(collapse(completed), out), "wrong completion/compression"
            assert np.array_equal(collapse(out), out), "output still has adjacent equal vectors"
            assert out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1]
            assert not np.array_equal(collapse(inp), out), "B contributes no visible change"
            if split == "train":
                metrics = inspect_pair(pair)
                if metrics["identity_pairing_wrong"]: evidence.add("pairing")
                if metrics["crossings"] and len(set(metrics["labels"])) == metrics["routes"]: evidence.add("crossing")
                if metrics["premature_rows_split"] or metrics["premature_columns_split"]: evidence.add("order")
                if metrics["removed_rows"] and metrics["removed_columns"]: evidence.add("both_axes")
                for a, b in info["clues"]:
                    if a != "blank" and b == "blank": evidence.add("forward")
                    if a == "blank": evidence.add("reverse")
                    if a == "diamond" and b == "checker": evidence.add("preserve")
                    if a == "solid": evidence.add("thin")
    # Absent crossing marks do not change the palette convention.
    assert len({(a, b, d) for a, b, c, d in conventions}) == 1
    assert len({c for a, b, c, d in conventions if c is not None}) <= 1
    assert {"pairing", "crossing", "order", "both_axes", "forward", "reverse", "preserve", "thin"} <= evidence


def self_test():
    """42 seeds, streaming metrics; no grids, traces or logs written to disk."""
    shapes, structures, pairings, pitches, signatures = set(), set(), set(), set(), set()
    orientations, route_counts, clue_modes, role_sets = set(), set(), set(), set()
    row_removals, col_removals, crossings = set(), set(), set()
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]; digest.update(a.tobytes()); digest.update(bytes(a.shape))
                assert np.array_equal(solve(a), pair["output"])
                m = inspect_pair(pair)
                shapes.add(tuple(m["shape"])); pairings.add(tuple(m["pairing"]))
                pitches.add(tuple(m["pitches"])); orientations.add(m["orientation"])
                route_counts.add(m["routes"]); role_sets.add(tuple(m["roles"][:2]))
                row_removals.add(m["removed_rows"]); col_removals.add(m["removed_columns"])
                crossings.add(m["crossings"]); clue_modes.update(tuple(x) for x in m["clues"])
                roles, normalized = {}, []
                for c in a.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structures.add((a.shape, bytes(normalized))); pairs += 1
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 30 and len(structures) >= 150
    assert route_counts == {2, 3, 4, 5} and orientations == {"vertical", "horizontal"}
    assert len(pairings) >= 20 and len(pitches) >= 15 and len(role_sets) >= 20
    assert len(row_removals) >= 5 and len(col_removals) >= 5 and max(crossings) >= 5
    assert {("solid", "blank"), ("diamond", "blank"), ("blank", "checker"),
            ("solid", "checker"), ("diamond", "checker")} <= clue_modes
    # Separated identical vectors survive, while adjacent copies disappear.
    a = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3], [1, 1, 2, 1]])
    assert collapse(a).tolist() == [[1, 2, 1], [3, 4, 3], [1, 2, 1]]
    # Exhaust all 120 five-cable bijections in the native drawing/tracing grammar.
    rng = np.random.default_rng(1701)
    for target in permutations(range(5)):
        field, centers, start, end, paths = _field(rng, 5, target)
        inferred, _, _ = _trace(field, start, end, centers, 1, 2)
        assert inferred == [target.index(i) for i in range(5)]
        assert all(1 <= d <= 30 for d in field.shape)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9201, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (3.2, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(9102, 10, 4), generate_puzzle(9102, 10, 4)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    damaged = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), structures=len(structures),
                pairings=len(pairings), pitch_patterns=len(pitches), palette_roles=len(role_sets),
                row_removals=sorted(row_removals), column_removals=sorted(col_removals),
                crossing_counts=sorted(crossings), exhaustive_five_cable_permutations=120)
