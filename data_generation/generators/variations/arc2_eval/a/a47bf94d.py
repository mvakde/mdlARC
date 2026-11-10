"""Orthogonal cable permutations with complementary terminal completion.

Generation uses a vacant routing track to move arbitrary cycles, while validation
reconstructs a directional graph from the *input*. Neither the tracer nor the
predictor receives a permutation, mask interior, random seed, or query answer.
"""
import numpy as np
from itertools import permutations


_DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
_CHECKER = 1 - _DIAMOND
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _validate_optional_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    a = _validate_optional_count("num_train", num_train, 2, 10)
    b = _validate_optional_count("num_test", num_test, 1, 4)
    if a is None and b is None:
        b = 1 + int(seed) % 4
        low = max(2, b + 1)
        a = low + (int(seed) // 4) % (11 - low)
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _motif(pattern, background, label):
    return np.where(pattern, label, background).astype(np.int8)


def _block_kind(block, background):
    if np.all(block == background):
        return "blank", None
    colors = set(map(int, np.unique(block))) - {background}
    if len(colors) != 1:
        raise AssertionError("terminal contains multiple labels")
    label = next(iter(colors))
    if np.all(block == label):
        return "solid", label
    for name, pattern in (("diamond", _DIAMOND), ("checker", _CHECKER)):
        if np.array_equal(block, _motif(pattern, background, label)):
            return name, label
    raise AssertionError("noncanonical terminal clue")


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _rectangles(grid, color):
    result = []
    for cells in _components(map(tuple, np.argwhere(grid == color))):
        r0, c0 = map(int, np.min(list(cells), axis=0))
        r1, c1 = map(int, np.max(list(cells), axis=0))
        assert r1 > r0 and c1 > c0
        assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
        result.append((r0, r1, c0, c1))
    return result


def _mask_connections(rect, visible, endpoints):
    """A side-adjacent wire is a port only if it points into the mask."""
    r0, r1, c0, c1 = rect
    ports = []
    for c in range(c0, c1 + 1):
        ports.extend([((r0 - 1, c), 2), ((r1 + 1, c), 0)])
    for r in range(r0, r1 + 1):
        ports.extend([((r, c0 - 1), 1), ((r, c1 + 1), 3)])
    true_ports = []
    for p, inward in ports:
        # A strand may turn at the first visible cell (source train 1). An
        # odd visible degree needs the missing edge; a parallel degree-two
        # strand merely touches the side of the rectangle and is not a port.
        degree = sum((p[0] + dr, p[1] + dc) in visible for dr, dc in _DIRECTIONS)
        if p in visible and (degree in (1, 3) or endpoints.get(p) == inward):
            true_ports.append((p, inward))
    assert len(true_ports) >= 2 and len(true_ports) % 2 == 0
    if len(true_ports) == 2:
        return [(true_ports[0], true_ports[1])]
    links = []
    remaining = set(true_ports)
    while remaining:
        a = min(remaining)
        remaining.remove(a)
        p, d = a
        aligned = [b for b in remaining if b[1] == (d + 2) % 4
                   and b[0][1 if d % 2 == 0 else 0] == p[1 if d % 2 == 0 else 0]]
        assert len(aligned) == 1, "mask hides an unspecified pairing"
        b = aligned[0]
        remaining.remove(b)
        links.append((a, b))
    return links


def _trace(grid, start, end, centers, carrier, cross, mask):
    visible = set(map(tuple, np.argwhere(grid == carrier)))
    if cross is not None:
        visible.update(map(tuple, np.argwhere(grid == cross)))
    endpoints = {(start, c): 2 for c in centers}
    endpoints.update({(end, c): 0 for c in centers})
    graph = {p: {} for p in visible}
    for r, c in visible:
        for d, (dr, dc) in enumerate(_DIRECTIONS):
            q = (r + dr, c + dc)
            if q in visible:
                graph[r, c][d] = (q, (d + 2) % 4)
    mask_info = []
    for rect in ([] if mask is None else _rectangles(grid, mask)):
        links = _mask_connections(rect, visible, endpoints)
        for (p, dp), (q, dq) in links:
            assert dp not in graph[p] and dq not in graph[q]
            graph[p][dp] = (q, dq)
            graph[q][dq] = (p, dp)
        mask_info.append((rect, links))
    crossing_cells = []
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1 and endpoints[p] in edges
        elif len(edges) == 4:
            r, c = p
            marked = any(all((r + _DIRECTIONS[d][0], c + _DIRECTIONS[d][1]) in visible
                             and grid[r + _DIRECTIONS[d][0], c + _DIRECTIONS[d][1]] == cross
                             for d in ds) for ds in ((0, 2), (1, 3)))
            assert marked, "unmarked branching intersection"
            crossing_cells.append(p)
        else:
            assert len(edges) == 2, "broken or branching visible cable"
    routes, permutation, all_seen = [], [], set()
    for c in centers:
        p, incoming = (start, c), 0
        route, states = [], set()
        while True:
            assert (p, incoming) not in states, "cable cycle"
            states.add((p, incoming))
            route.append(p)
            all_seen.add(p)
            if p[0] == end and p in endpoints:
                permutation.append(centers.index(p[1]))
                break
            edges = graph[p]
            if p == (start, c):
                outgoing = 2
            else:
                assert incoming in edges
                if len(edges) == 4:
                    outgoing = (incoming + 2) % 4
                else:
                    outgoing = next(d for d in edges if d != incoming)
            p, incoming = edges[outgoing]
            assert p[0] != start, "cable returns to its starting bank"
        routes.append(route)
    assert sorted(permutation) == list(range(len(centers))), "nonbijective routing"
    assert all_seen == visible, "unattached routing fragments"
    return permutation, routes, mask_info, crossing_cells


def _parse_input(grid):
    # The blank exterior identifies background even when a label is repeated.
    corners = [int(grid[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1
    background = corners[0]
    candidates = []
    for transpose in (False, True):
        g = grid.T if transpose else grid
        h, w = g.shape
        for carrier in set(map(int, np.unique(g))) - {background}:
            positions = np.argwhere(g == carrier)
            start, end = int(positions[:, 0].min()), int(positions[:, 0].max())
            if start < 3 or end > h - 4 or end - start < 2:
                continue
            centers = list(map(int, np.flatnonzero(g[start] == carrier)))
            if not 2 <= len(centers) <= 6 or centers != list(np.flatnonzero(g[end] == carrier)):
                continue
            if centers[0] < 1 or centers[-1] >= w - 1:
                continue
            if len(set(np.diff(centers))) != 1 or centers[1] - centers[0] < 4:
                continue
            try:
                footprint = np.zeros(g.shape, dtype=bool)
                banks = []
                for row in (start - 3, end + 1):
                    bank = []
                    for c in centers:
                        footprint[row:row + 3, c - 1:c + 2] = True
                        bank.append(_block_kind(g[row:row + 3, c - 1:c + 2], background))
                    banks.append(bank)
                outside = ~footprint
                outside[start:end + 1] = False
                assert np.all(g[outside] == background)
                field = np.full(g.shape, background, dtype=np.int8)
                field[start:end + 1] = g[start:end + 1]
                other = set(map(int, np.unique(field))) - {background, carrier}
                assert len(other) <= 2
                cross, mask = None, None
                for color in other:
                    try:
                        _rectangles(field, color)
                        assert mask is None
                        mask = color
                    except AssertionError:
                        assert cross is None
                        cross = color
                labels = {v for bank in banks for _, v in bank if v is not None}
                assert labels.isdisjoint({carrier, cross, mask})
                perm, routes, masks, crossings = _trace(field, start, end, centers, carrier, cross, mask)
                candidates.append(dict(transpose=transpose, grid=g, background=background,
                                       carrier=carrier, cross=cross, mask=mask, start=start,
                                       end=end, centers=centers, banks=banks, footprint=footprint,
                                       permutation=perm, routes=routes, masks=masks, crossings=crossings))
            except (AssertionError, StopIteration):
                continue
    assert len(candidates) == 1, f"expected unique input interpretation, got {len(candidates)}"
    return candidates[0]


def _complete(info, phase):
    g = info["grid"].copy()
    bg, centers = info["background"], info["centers"]
    for i, j in enumerate(info["permutation"]):
        near, far = info["banks"][0][i], info["banks"][1][j]
        source, receiver = (near, far) if phase == 0 else (far, near)
        assert source[0] in ("blank", "solid", "diamond")
        assert receiver[0] in ("blank", "checker")
        labels = {b[1] for b in (source, receiver) if b[1] is not None}
        assert len(labels) == 1, "absent or conflicting endpoint clues"
        color = next(iter(labels))
        for bank, index, row in ((0, i, info["start"] - 3), (1, j, info["end"] + 1)):
            pattern = _DIAMOND if bank == phase else _CHECKER
            c = centers[index]
            g[row:row + 3, c - 1:c + 2] = _motif(pattern, bg, color)
    return g.T.copy() if info["transpose"] else g


def _learn_phase(train_pairs):
    possible = {0, 1}
    for pair in train_pairs:
        info = _parse_input(pair["input"])
        matches = set()
        for phase in possible:
            try:
                if np.array_equal(_complete(info, phase), pair["output"]):
                    matches.add(phase)
            except AssertionError:
                pass
        possible &= matches
    assert len(possible) == 1, "demonstrations do not establish one consistent bank rule"
    return possible.pop()


def solve(train_pairs, test_input):
    """Predict using demonstrations and query cells only; never query labels."""
    phase = _learn_phase(train_pairs)
    return _complete(_parse_input(test_input), phase)


def _moves_for_permutation(target, rng):
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


def _orthogonal_field(rng, n, target, extra_detour):
    moves = _moves_for_permutation(target, rng)
    if extra_detour and len(moves) <= 7:
        lane = int(rng.integers(n))
        moves = [(lane, n), (n, lane)] + moves
    max_pitch = min(7, (30 - 3 - 3 - 2) // (n - 1))
    pitch = int(rng.integers(4, max_pitch + 1))
    spare_gap = int(rng.integers(3, min(6, 30 - (n - 1) * pitch - 4) + 1))
    min_width = 2 + (n - 1) * pitch + spare_gap + 1
    available = 30 - min_width
    left = int(rng.integers(min(4, available) + 1))
    right = int(rng.integers(min(4, available - left) + 1))
    centers = [left + 2 + i * pitch for i in range(n)]
    columns = centers + [centers[-1] + spare_gap]
    width = columns[-1] + 1 + right
    # Two cells between moves prevent touching lateral runs. Spare space is
    # allocated among margins, lead lengths, and individual segment lengths.
    gaps = [2] * max(0, len(moves) - 1)
    lead, tail = 2, 2
    minimum = 6 + lead + tail + 1 + sum(gaps)
    if not moves:
        minimum = 11
    budget = 30 - minimum
    top = int(rng.integers(min(4, budget) + 1)); budget -= top
    bottom = int(rng.integers(min(4, budget) + 1)); budget -= bottom
    extra = int(rng.integers(min(9, budget) + 1))
    pieces = [lead, tail] + gaps
    for _ in range(extra):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, gaps = pieces[0], pieces[1], pieces[2:]
    start = top + 3
    move_rows = []
    row = start + lead
    for k in range(len(moves)):
        move_rows.append(row)
        if k < len(gaps):
            row += gaps[k]
    end = (move_rows[-1] + tail) if moves else start + lead + tail
    height = end + 4 + bottom
    paths = [[(start, c)] for c in centers]
    state = list(range(n)) + [None]
    previous = start
    for row, (old, new) in zip(move_rows, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        route = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[route].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    field = np.zeros((height, width), dtype=np.int8)
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            field[p] = 1
            visits.setdefault(p, set()).add(i)
    for (r, c), owners in visits.items():
        if len(owners) > 1:
            assert len(owners) == 2
            axes = (0, 2) if rng.random() < .5 else (1, 3)
            for d in axes:
                dr, dc = _DIRECTIONS[d]
                assert field[r + dr, c + dc] != 0
                field[r + dr, c + dc] = 2
    return field, centers, start, end, paths


def _add_masks(rng, field, centers, start, end, paths, wanted, keep_crossing=False):
    # Test proposed masks against true boundary crossings, then against the
    # visible directional graph. The solver later has access only to the latter.
    original = field.copy()
    visible_original = set(map(tuple, np.argwhere((original == 1) | (original == 2))))
    endpoints = {(start, c): 2 for c in centers}
    endpoints.update({(end, c): 0 for c in centers})
    used = []
    points = sorted(visible_original)
    for _ in range(180):
        if len(used) >= wanted:
            break
        r, c = points[int(rng.integers(len(points)))]
        hh = int(rng.integers(2, min(6, end - start) + 1))
        ww = int(rng.integers(2, min(13, field.shape[1] - 1) + 1))
        r0 = max(start + 1, min(end - hh, r - int(rng.integers(hh))))
        c0 = max(0, min(field.shape[1] - ww, c - int(rng.integers(ww))))
        rect = (r0, r0 + hh - 1, c0, c0 + ww - 1)
        r0, r1, c0, c1 = rect
        if r1 >= end or r0 <= start:
            continue
        if any(not (r1 + 1 < a or b + 1 < r0 or c1 + 1 < x or y + 1 < c0)
               for a, b, x, y in used):
            continue
        inside = lambda p: r0 <= p[0] <= r1 and c0 <= p[1] <= c1
        actual = set()
        for path in paths:
            entry = None
            for a, b in zip(path, path[1:]):
                if not inside(a) and inside(b):
                    entry = a
                if inside(a) and not inside(b):
                    assert entry is not None
                    actual.add(frozenset((entry, b)))
                    entry = None
        if not actual:
            continue
        visible = {p for p in visible_original if not inside(p)}
        try:
            links = _mask_connections(rect, visible, endpoints)
            inferred = {frozenset((a[0], b[0])) for a, b in links}
            assert inferred == actual
            candidate = field.copy()
            candidate[r0:r1 + 1, c0:c1 + 1] = 3
            traced = _trace(candidate, start, end, centers, 1, 2, 3)
            assert not keep_crossing or traced[3]
        except AssertionError:
            continue
        field = candidate
        used.append(rect)
    return field


def _clues(rng, n, lesson):
    options = [("solid", "blank"), ("diamond", "blank"), ("blank", "checker"),
               ("solid", "checker"), ("diamond", "checker")]
    if lesson == "forward":
        clues = [options[int(rng.integers(2))] for _ in range(n)]
        clues[0] = options[0]
    elif lesson == "reverse":
        clues = [options[int(rng.integers(5))] for _ in range(n)]
        clues[0], clues[1] = options[2], options[4]
    else:
        mode = int(rng.integers(6))
        if mode == 0:
            clues = [options[int(rng.integers(2))] for _ in range(n)]
        elif mode == 1:
            clues = [options[2]] * n
        elif mode == 2:
            clues = [options[int(rng.integers(3, 5))] for _ in range(n)]
            clues[0] = options[3]
        else:
            clues = [options[int(rng.integers(5))] for _ in range(n)]
    if all(a == "diamond" and b == "checker" for a, b in clues):
        clues[0] = options[0]
    return [clues[int(i)] for i in rng.permutation(n)]


def _generate_pair(rng, parameters, n, orientation, lesson, regime):
    level = parameters["level"]
    target = list(map(int, rng.permutation(n)))
    kind = rng.random()
    if lesson == "forward":
        target = list(range(n)); target[0], target[-1] = target[-1], target[0]
    elif kind < (.55 if level == 0 else .13):
        target = list(range(n))
    elif kind < .28:
        target = list(reversed(range(n)))
    if regime == "target_test" and parameters["shift"] == "routing" and level:
        # A routing shift can increase permutation difficulty at an overlapping
        # route count. The deterministic reversal is also a bounded fallback.
        for _ in range(12):
            if sum(a > b for k, a in enumerate(target) for b in target[k + 1:]) >= n - 1:
                break
            target = list(map(int, rng.permutation(n)))
        else:
            target = list(reversed(range(n)))
    field, centers, start, end, paths = _orthogonal_field(rng, n, target, rng.random() < .3)
    if level == 0:
        wanted = int(rng.integers(2))
    else:
        wanted = int(rng.integers(5))
    if lesson == "reverse":
        wanted = max(1, wanted)
    if regime == "target_test" and parameters["shift"] == "occlusion":
        wanted = int(rng.integers(2, 5))
    field = _add_masks(rng, field, centers, start, end, paths, wanted,
                       keep_crossing=lesson == "forward") if wanted else field
    palette = parameters["palette"]
    if parameters["vary_palette"] and rng.random() < parameters["vary_palette"]:
        palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    count = int(rng.integers(2, min(n, 6) + 1))
    labels = list(map(int, rng.choice(palette[4:], count, replace=False)))
    route_labels = labels + list(map(int, rng.choice(labels, n - count)))
    rng.shuffle(route_labels)
    if lesson == "forward":
        # Every visible crossing in this demonstration distinguishes its two
        # strands. Other pairs freely use repeated labels and smaller palettes.
        route_labels = list(palette[4:4 + n])
    clues = _clues(rng, n, lesson)
    inp, out = field.copy(), field.copy()
    inp = np.asarray(palette, dtype=np.int8)[inp]
    out = np.asarray(palette, dtype=np.int8)[out]
    for i, path in enumerate(paths):
        j = centers.index(path[-1][1])
        for row, c, pattern, mode in ((start - 3, centers[i], _DIAMOND, clues[i][0]),
                                      (end + 1, centers[j], _CHECKER, clues[i][1])):
            color = route_labels[i]
            out[row:row + 3, c - 1:c + 2] = _motif(pattern, bg, color)
            block = np.full((3, 3), bg if mode == "blank" else color, dtype=np.int8)
            if mode in ("diamond", "checker"):
                block = _motif(pattern, bg, color)
            inp[row:row + 3, c - 1:c + 2] = block
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    if parameters["phase"]:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    if orientation:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _distinct_fallback(pair, seen):
    """Keep geometry and clue forms; enumerate locally legal label assignments.

    At least six nonstructural colors and two labels give at least 30 variants,
    exceeding the maximum 13 earlier pairs plus four excluded source pairs.
    """
    info = _parse_input(pair["input"])
    labels = sorted({v for bank in info["banks"] for _, v in bank if v is not None})
    structural = {info[k] for k in ("background", "carrier", "cross", "mask")}
    available = sorted(set(range(10)) - structural)
    for values in permutations(available, len(labels)):
        lut = np.arange(10, dtype=np.int8)
        for old, new in zip(labels, values):
            lut[old] = new
        candidate = {k: lut[v] for k, v in pair.items()}
        key = (candidate["input"].shape, candidate["input"].tobytes())
        if key not in seen and not _is_source_pair(candidate):
            return candidate
    raise AssertionError("finite label-assignment bound violated")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, rng, num_train, num_test)
    level = int(rng.choice(3, p=[.15, .4, .45]))
    parameters = dict(level=level, phase=int(rng.integers(2)),
                      palette=list(map(int, rng.permutation(10))),
                      vary_palette=0 if level == 0 else (.55 if level == 1 else 1),
                      shift=str(rng.choice(["count", "overlap", "routing", "occlusion"])))
    orientation_mode = int(rng.integers(5)) if level else int(rng.integers(2))
    lesson_indices = list(map(int, rng.choice(nt, 2, replace=False)))
    train_count_base = int(rng.integers(2, 6))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", nt), ("test", nq)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(total):
            lesson = ("forward" if index == lesson_indices[0] else
                      "reverse" if index == lesson_indices[1] else None) if split == "train" else None
            if parameters["shift"] == "count":
                n = 4 if split == "train" else 5
            elif parameters["shift"] in ("routing", "occlusion"):
                n = train_count_base if rng.random() < .65 else int(rng.integers(2, 7))
            else:
                n = int(rng.integers(2, 7))
            if orientation_mode < 2:
                orientation = orientation_mode
            elif orientation_mode == 2:
                orientation = int(rng.integers(2))
            else:
                orientation = int((split == "test") == (orientation_mode == 3))
            for attempt in range(32):
                pair = _generate_pair(rng, parameters, n, orientation, lesson, regime)
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen and not _is_source_pair(pair):
                    break
            else:
                pair = _distinct_fallback(pair, seen)
                signature = (pair["input"].shape, pair["input"].tobytes())
            seen.add(signature)
            puzzle[split].append(pair)
    return puzzle


def _route_clues(info, phase):
    result = []
    for i, j in enumerate(info["permutation"]):
        a, b = info["banks"][0][i], info["banks"][1][j]
        result.append((a, b) if phase == 0 else (b, a))
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= x <= 30 for x in grid.shape)
            assert np.all((grid >= 0) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        assert not np.array_equal(pair["input"], pair["output"])
        key = (pair["input"].shape, pair["input"].tobytes())
        assert key not in seen and not _is_source_pair(pair)
        seen.add(key)
    phase = _learn_phase(puzzle["train"])
    evidence = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _parse_input(pair["input"])
            prediction = _complete(info, phase)
            assert np.array_equal(prediction, pair["output"]), "input-derived answer disagrees"
            output = pair["output"].T if info["transpose"] else pair["output"]
            assert np.array_equal(info["grid"][~info["footprint"]], output[~info["footprint"]])
            if split == "train":
                for source, receiver in _route_clues(info, phase):
                    if source[0] != "blank" and receiver[0] == "blank":
                        evidence.add("forward")
                    if source[0] == "blank":
                        evidence.add("reverse")
                    if source[0] == "diamond":
                        evidence.add("preserve_diamond")
                    if receiver[0] == "checker":
                        evidence.add("preserve_checker")
                    if source[0] != "blank" and receiver[0] != "blank":
                        evidence.add("both")
                    if source[0] == "solid":
                        evidence.add("solid")
                route_labels = [a[1] if a[1] is not None else b[1]
                                for a, b in _route_clues(info, phase)]
                for crossing in info["crossings"]:
                    owners = [i for i, route in enumerate(info["routes"]) if crossing in route]
                    if len({route_labels[i] for i in owners}) == 2:
                        evidence.add("crossing")
    assert {"forward", "reverse", "preserve_diamond", "preserve_checker", "both", "solid", "crossing"} <= evidence


def inspect_pair(pair):
    """Task measurements reconstructed from the returned input, not RNG metadata."""
    info = _parse_input(pair["input"])
    phases = []
    for phase in (0, 1):
        try:
            _complete(info, phase)
            phases.append(phase)
        except AssertionError:
            pass
    assert len(phases) == 1
    phase = phases[0]
    clues = _route_clues(info, phase)
    perm = info["permutation"]
    bends, lengths = [], []
    for route in info["routes"]:
        directions, length = [], 0
        for a, b in zip(route, route[1:]):
            dr, dc = b[0] - a[0], b[1] - a[1]
            length += abs(dr) + abs(dc)
            directions.append((int(np.sign(dr)), int(np.sign(dc))))
        bends.append(sum(a != b for a, b in zip(directions, directions[1:])))
        lengths.append(int(length))
    mask_routes, mask_shapes, elbow_masks, crossing_masks = [], [], 0, 0
    route_masks = [0] * len(perm)
    for rect, links in info["masks"]:
        r0, r1, c0, c1 = rect
        mask_shapes.append((r1 - r0 + 1, c1 - c0 + 1))
        owners = {i for i, route in enumerate(info["routes"])
                  if any(a[0] in route and b[0] in route for a, b in links)}
        for i in owners:
            route_masks[i] += 1
        mask_routes.append(len(owners))
        elbow_masks += int(len(links) == 1 and links[0][0][1] % 2 != links[0][1][1] % 2)
        crossing_masks += int(len({a[1] % 2 for a, b in links}) == 2)
    near, far = info["start"] - 3, info["grid"].shape[0] - info["end"] - 4
    centers = info["centers"]
    labels = [a[1] if a[1] is not None else b[1] for a, b in clues]
    return dict(orientation="horizontal" if info["transpose"] else "vertical", phase=phase,
                shape=pair["input"].shape, routes=len(perm), pitch=centers[1] - centers[0],
                bank_distance=info["end"] - info["start"] + 4,
                margins=(near, far, centers[0] - 1, info["grid"].shape[1] - centers[-1] - 2),
                roles=(info["background"], info["carrier"], info["cross"], info["mask"]),
                colors=len(np.unique(pair["input"])), labels=len(set(labels)),
                repeated_labels=len(set(labels)) < len(labels), permutation=tuple(perm),
                inversions=sum(a > b for k, a in enumerate(perm) for b in perm[k + 1:]),
                crossings=len(info["crossings"]), bends=tuple(bends), lengths=tuple(lengths),
                excursion=any(c < centers[0] or c > centers[-1] for route in info["routes"] for r, c in route),
                mask_count=len(info["masks"]), mask_shapes=mask_shapes, mask_routes=mask_routes,
                elbow_masks=elbow_masks, crossing_masks=crossing_masks,
                max_masks_on_route=max(route_masks),
                forward=sum(a[0] != "blank" and b[0] == "blank" for a, b in clues),
                reverse=sum(a[0] == "blank" for a, b in clues),
                both=sum(a[0] != "blank" and b[0] != "blank" for a, b in clues),
                solid=sum(a[0] == "solid" for a, b in clues),
                canonical=sum(a[0] == "diamond" for a, b in clues))


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[x], q[x])
               for p, q in zip(a[s], b[s]) for x in ("input", "output")) for s in ("train", "test"))


def _source_pairs():
    return [{role: np.array([[int(c) for c in row] for row in text.split()], dtype=np.int8)
             for role, text in zip(("input", "output"), pair)} for pair in _SOURCE_FIXTURES]


def _is_source_pair(pair):
    return any(np.array_equal(pair["input"], source["input"])
               and np.array_equal(pair["output"], source["output"]) for source in _source_pairs())


def self_test() -> None:
    source = _source_pairs()
    # Predictions are computed before looking at the corresponding answer.
    for i, pair in enumerate(source):
        training = source[:3] if i == 3 else [p for j, p in enumerate(source[:3]) if j != i]
        predicted = solve(training, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])
    source_info = _parse_input(source[3]["input"])
    assert source_info["permutation"] == [2, 4, 3, 1, 0]
    elbow = next(links for rect, links in source_info["masks"] if rect == (15, 17, 1, 4))
    assert {elbow[0][0][0], elbow[0][1][0]} == {(16, 5), (18, 2)}
    # All 120 five-cable endpoint permutations fit, including the eight-
    # inversion source test. Test the visible graph, not just the move list.
    routing_rng = np.random.default_rng(71021)
    for target in permutations(range(5)):
        field, centers, start, end, _ = _orthogonal_field(routing_rng, 5, target, False)
        pairing, _, _, _ = _trace(field, start, end, centers, 1, 2, None)
        assert pairing == [target.index(i) for i in range(5)]
        assert max(field.shape) <= 30
    metrics, query_metrics, per_puzzle = [], [], []
    train_counts, test_counts, signatures = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"])); test_counts.add(len(puzzle["test"]))
        local = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                measured = inspect_pair(pair)
                metrics.append(measured)
                local[split].append(measured)
                if split == "test":
                    query_metrics.append(measured)
                signatures.add((pair["input"].shape, pair["input"].tobytes()))
        per_puzzle.append(local)
        for pair in puzzle["test"]:
            predicted = solve([{k: v.copy() for k, v in p.items()} for p in puzzle["train"]], pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == len(metrics)
    assert len({m["shape"] for m in metrics}) >= 40
    assert {m["routes"] for m in metrics} == set(range(2, 7))
    assert {m["pitch"] for m in metrics} >= {4, 5, 6, 7}
    assert {m["phase"] for m in metrics} == {0, 1}
    assert {m["orientation"] for m in metrics} == {"vertical", "horizontal"}
    assert max(m["inversions"] for m in metrics if m["routes"] == 5) >= 8
    assert any(m["crossings"] == 0 and m["inversions"] == 0 for m in metrics)
    assert any(m["elbow_masks"] for m in metrics)
    assert any(m["crossing_masks"] for m in metrics)
    assert any(max(m["mask_routes"], default=0) >= 3 for m in metrics)
    assert any(m["max_masks_on_route"] >= 2 for m in metrics)
    assert {m["mask_count"] for m in metrics} >= {0, 1, 2, 3, 4}
    assert any(m["repeated_labels"] for m in metrics)
    assert len({shape for m in metrics for shape in m["mask_shapes"]}) >= 20
    assert len({m["bank_distance"] for m in metrics}) >= 12
    assert all(len({m["margins"][i] for m in metrics}) >= 4 for i in range(4))
    assert len({m["labels"] for m in metrics}) >= 4
    assert len({m["colors"] for m in query_metrics}) >= 4
    assert any(m["reverse"] == m["routes"] for m in query_metrics)
    assert any(m["forward"] == m["routes"] for m in query_metrics)
    assert any(m["both"] == m["routes"] for m in query_metrics)
    assert any(m["solid"] and m["canonical"] for m in query_metrics)
    assert any(m["orientation"] == "horizontal" and m["reverse"] >= 2
               and max(m["mask_routes"], default=0) >= 2
               and len(set(m["lengths"])) > 1 for m in query_metrics)
    for key in ("routes", "pitch", "shape", "bank_distance", "mask_count", "labels"):
        assert sum(len({m[key] for m in p["train"] + p["test"]}) > 1 for p in per_puzzle) >= 20
    assert sum(len({m["roles"][:2] for m in p["train"] + p["test"]}) > 1
               for p in per_puzzle) >= 15
    regimes = {(tuple(sorted({m["orientation"] for m in p["train"]})),
                tuple(sorted({m["orientation"] for m in p["test"]}))) for p in per_puzzle}
    for a, b in (("vertical", "vertical"), ("horizontal", "horizontal"),
                 ("vertical", "horizontal"), ("horizontal", "vertical")):
        assert ((a,), (b,)) in regimes
    assert any(len(a) == 2 for a, b in regimes)
    assert any({m["routes"] for m in p["train"]} == {4}
               and {m["routes"] for m in p["test"]} == {5} for p in per_puzzle)
    assert any({m["routes"] for m in p["train"]} & {m["routes"] for m in p["test"]}
               for p in per_puzzle)
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(4242, *counts)
        validate_generated_puzzle(p)
        assert all(v is None or len(p[s]) == v for s, v in zip(("train", "test"), counts))
    a = generate_puzzle(8675309, 6, 2)
    assert _puzzles_equal(a, generate_puzzle(8675309, 6, 2))
    assert not _puzzles_equal(a, generate_puzzle(8675310, 6, 2))
    old = a["test"][0]
    different = _distinct_fallback(old, {(old["input"].shape, old["input"].tobytes())})
    assert not np.array_equal(old["input"], different["input"])
    assert np.array_equal(solve(a["train"], different["input"]), different["output"])
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("nonterminal corruption accepted")


# Exact staged source fixtures: regression evidence and content-based exclusion.
_SOURCE_FIXTURES = (('0000000000000000000000\n0000000000000000000000\n0033300000000000000000\n0033388888800880000000\n0033300000800800000000\n0000000889999800000000\n0004000809999004040000\n0040485859999880400000\n0004000809999004040000\n0000000809999000000000\n0022200800800000000000\n0022288800888880000000\n0022200000000000000000\n0000000000000000000000\n0000000000000006060000\n0000088888888880600000\n0000000000000006060000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000', '0000000000000000000000\n0000000000000000000000\n0003000000000002020000\n0030388888800880200000\n0003000000800802020000\n0000000889999800000000\n0004000809999004040000\n0040485859999880400000\n0004000809999004040000\n0000000809999000000000\n0002000800800003030000\n0020288800888880300000\n0002000000000003030000\n0000000000000000000000\n0006000000000006060000\n0060688888888880600000\n0006000000000006060000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000'), ('0000000000000000000000\n0001110222033304440000\n0001110222033304440000\n0001110222033304440000\n0000800080008000800000\n0000800080008885858000\n0000800050000000808000\n0000888888888000808000\n0000000050008000808000\n0000000080008000808000\n0000888585858588808000\n0000800080008000008000\n0000800080008000888000\n0000800080008000800000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000', '0000000000000000000000\n0000100020003000400000\n0001010202030304040000\n0000100020003000400000\n0000800080008000800000\n0000800080008885858000\n0000800050000000808000\n0000888888888000808000\n0000000050008000808000\n0000000080008000808000\n0000888585858588808000\n0000800080008000008000\n0000800080008000888000\n0000800080008000800000\n0004040202010103030000\n0000400020001000300000\n0004040202010103030000\n0000000000000000000000\n0000000000000000000000'), ('0000000000000000000000\n0022200000444066600000\n0022200000444066600000\n0022200000444066600000\n0008000800080008000000\n0099900800080008000000\n0099900888585808000000\n0008000000080899990000\n0008888800990899990000\n0000000800990899990000\n0008885858880808000000\n0008000800000808000000\n0008000800088808000000\n0008000800080008000000\n0000000000101000000000\n0000000000010000000000\n0000000000101000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000', '0000000000000000000000\n0002000100040006000000\n0020201010404060600000\n0002000100040006000000\n0008000800080008000000\n0099900800080008000000\n0099900888585808000000\n0008000000080899990000\n0008888800990899990000\n0000000800990899990000\n0008885858880808000000\n0008000800000808000000\n0008000800088808000000\n0008000800080008000000\n0040402020101060600000\n0004000200010006000000\n0040402020101060600000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000\n0000000000000000000000'), ('00000000000000000000000000\n01110000000222003330066600\n01110000000222003330066600\n01110000000222003330066600\n00800008000080000800008000\n00800008000088880800888000\n00800008099000050500500000\n00800008899888888888888000\n00800000099000050500508000\n00888888899880080800808000\n00000000099080080800808000\n00000000000080050809999900\n00000008888585888809999900\n00000008000080050009999900\n00000008000080088800808000\n09999008000099000500808000\n09999858588599888888808000\n09999008000080000500008000\n00800008000080000800008000\n00800008000080000800008000\n00000000000000000000040400\n00000000000000000000004000\n00000000000000000000040400\n00000000000000000000000000\n00000000000000000000000000\n00000000000000000000000000', '00000000000000000000000000\n00100004000020000300006000\n01010040400202003030060600\n00100004000020000300006000\n00800008000080000800008000\n00800008000088880800888000\n00800008099000050500500000\n00800008899888888888888000\n00800000099000050500508000\n00888888899880080800808000\n00000000099080080800808000\n00000000000080050809999900\n00000008888585888809999900\n00000008000080050009999900\n00000008000080088800808000\n09999008000099000500808000\n09999858588599888888808000\n09999008000080000500008000\n00800008000080000800008000\n00800008000080000800008000\n06060030300101002020040400\n00600003000010000200004000\n06060030300101002020040400\n00000000000000000000000000\n00000000000000000000000000\n00000000000000000000000000'))
