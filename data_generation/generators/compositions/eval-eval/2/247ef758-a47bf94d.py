"""Crossing prints: cables carry transparent templates into local over/under seals.

Only standard Python and NumPy are required. Generation paints from explicit path
records; solve reconstructs a directional graph from the input pixels alone.
No files, cached puzzles, random state, or training answers are used by solve.
"""
from itertools import combinations
import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(a, b):
    a = _integer(a, "num_train", 2, 10)
    b = _integer(b, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if q in cells and q not in seen:
                    seen.add(q)
                    todo.append(q)
    return seen == cells


def _mask(rng, h, w, lesson=False):
    """Tight odd boxes, including holes, sparse diagonals, solids and odd lines."""
    if lesson:
        # Equal-area, incomparable, center-bearing 3x3 templates make the two
        # colors and their occlusion independently observable at a crossing.
        for _ in range(100):
            m = np.zeros((3, 3), bool)
            m.flat[rng.choice(9, 5, replace=False)] = True
            if m[1, 1] and m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any() and _connected(m):
                return m
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    if min(h, w) == 1:
        return np.ones((h, w), bool)
    mode = int(rng.integers(7))
    y, x = np.indices((h, w))
    if mode == 0:
        m = np.ones((h, w), bool)
    elif mode == 1:
        m = (y == 0) | (y == h - 1) | (x == 0) | (x == w - 1)
    elif mode == 2 and h == w:
        m = np.eye(h, dtype=bool)
    elif mode == 3 and h == w:
        m = abs(y - h // 2) + abs(x - w // 2) == h // 2
    else:
        # Independent random walks between all four sides yield many connected
        # masks, rather than a fixed shape dictionary or a required full cross.
        points = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(points)
        m = np.zeros((h, w), bool)
        r, c = points[0]
        m[r, c] = True
        for rr, cc in points[1:]:
            while (r, c) != (rr, cc):
                dr, dc = int(np.sign(rr - r)), int(np.sign(cc - c))
                if dr and dc and rng.random() < .5:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r + dr, c + dc
                m[r, c] = True
        m |= (rng.random((h, w)) < .22) & np.array([
            [m[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any() for c in range(w)] for r in range(h)])
    if rng.random() < .5:
        m = m[:, ::-1]
    if rng.random() < .5:
        m = m[::-1]
    return m.copy()


def _moves(target, rng):
    """Realize a permutation by shuttling whole strands through a vacant lane."""
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = saved, None
        vacant = j
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = saved, None
    assert state[:n] == list(target)
    return moves


def _paths(rng, n, moves, masks):
    rh = max(m.shape[0] // 2 for m in masks)
    rw = max(m.shape[1] // 2 for m in masks)
    pitch, gap = max(4, 2 * rw + 2), max(3, 2 * rh + 1)
    # All slack is budgeted before drawing; no unbounded rejection or clipping.
    width_min = rw + 1 + (n - 1) * pitch + 3 + 2
    spare = int(rng.integers(min(6, 30 - width_min + 3) - 2)) + 3
    slack = 30 - (width_min + spare - 3)
    extras = [0] * (n + 1)
    for _ in range(int(rng.integers(min(6, slack) + 1))):
        extras[int(rng.integers(n + 1))] += 1
    centers = [rw + 1 + extras[0]]
    for i in range(n - 1):
        centers.append(centers[-1] + pitch + extras[i + 1])
    columns = centers + [centers[-1] + spare]
    width = columns[-1] + 2 + extras[-1]
    maxh = max(m.shape[0] for m in masks)
    start, lead, tail = maxh + 2, 2, 2
    gaps = [gap] * (len(moves) - 1)
    height_min = start + lead + sum(gaps) + tail + 2
    assert width <= 30 and height_min <= 30
    slack = 30 - height_min
    top = int(rng.integers(min(2, slack) + 1)); slack -= top
    bottom = int(rng.integers(min(2, slack) + 1)); slack -= bottom
    pieces = [lead, tail] + gaps
    for _ in range(int(rng.integers(min(5, slack) + 1))):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, gaps = pieces[0], pieces[1], pieces[2:]
    start += top
    rows = [start + lead]
    for d in gaps:
        rows.append(rows[-1] + d)
    end = rows[-1] + tail
    height = end + 2 + bottom
    paths = [[(start, c)] for c in centers]
    state, prev = list(range(n)) + [None], start
    for row, (old, new) in zip(rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(prev + 1, row + 1))
        owner = state[old]
        direction = 1 if new > old else -1
        paths[owner].extend((row, c) for c in range(columns[old] + direction, columns[new] + direction, direction))
        state[new], state[old] = owner, None
        prev = row
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(prev + 1, end + 1))
    return (height, width), centers, start, paths


def _make_pair(rng, n, lesson=None):
    # Lessons guarantee visible discriminating evidence even at the legal
    # minimum of two demonstrations. All other scenes sample the same rule.
    if lesson == "transport":
        selected = list(map(int, rng.choice(n, 3, replace=False)))
        target = list(range(n))
        for a, b in zip(selected, selected[1:] + selected[:1]):
            target[a] = b
        moves = _moves(target, rng)
        masks, seen = [], set()
        for _ in range(n):
            for _ in range(100):
                m = _mask(rng, 3, 3, True)
                if m.tobytes() not in seen:
                    break
            else:
                raise AssertionError("lesson mask vocabulary exhausted")
            seen.add(m.tobytes()); masks.append(m)
    elif lesson == "layers":
        lane = int(rng.integers(1, n - 1))
        moves = [(lane, n), (n, lane)]
        masks = [_mask(rng, 3, 3) for _ in range(n)]
        # Two different odd rectangles have shared and private occupied cells.
        masks[lane] = np.zeros((5, 3), bool)
        masks[lane][2, :] = True; masks[lane][:, 1] = True
        for i in range(lane + 1, n):
            masks[i] = np.zeros((3, 5), bool)
            masks[i][1, :] = True; masks[i][:, 2] = True
            masks[i] |= rng.random((3, 5)) < .16
    else:
        target = list(map(int, rng.permutation(n)))
        moves = _moves(target, rng)
        # A return excursion is topologically different from straight identity,
        # although both have the same terminal pairing.
        if not moves or (len(moves) <= 5 and rng.random() < .3):
            lane = int(rng.integers(n - 1))
            moves = [(lane, n), (n, lane)] + moves
        large = n <= 3 and len(moves) <= 4 and rng.random() < .55
        sizes = [(1, 3), (3, 1), (3, 3)] + ([(3, 5), (5, 3), (5, 5)] if large else [])
        masks = [_mask(rng, *sizes[int(rng.integers(len(sizes)))]) for _ in range(n)]
    shape, centers, start, paths = _paths(rng, n, moves, masks)
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, bridge = palette[:3]
    colors = palette[3:3 + n]
    inp = np.full(shape, bg, dtype=np.int8)
    source_cells = []
    for c, m, color in zip(centers, masks, colors):
        h, w = m.shape
        top, left = start - 1 - h, c - w // 2
        cells = [(top + int(r), left + int(x)) for r, x in np.argwhere(m)]
        source_cells.append(cells)
        for p in cells:
            inp[p] = color
    visits, axes = {}, {}
    for i, path in enumerate(paths):
        for k, p in enumerate(path):
            visits.setdefault(p, []).append(i)
            inp[p] = carrier
            if 0 < k < len(path) - 1:
                a, b = path[k - 1], path[k + 1]
                if a[0] == b[0]:
                    axes[p, i] = 1
                elif a[1] == b[1]:
                    axes[p, i] = 0
    crossing_rows = sorted({p[0] for p, owners in visits.items() if len(owners) == 2})
    records = []
    for p in sorted(visits):
        owners = visits[p]
        if len(owners) == 1:
            continue
        assert len(owners) == 2
        by_axis = {axes[p, i]: i for i in owners}
        assert set(by_axis) == {0, 1}
        over = crossing_rows.index(p[0]) % 2 if lesson == "layers" else int(rng.integers(2))
        for d in ((0, 2) if over == 0 else (1, 3)):
            dr, dc = _DIRS[d]
            inp[p[0] + dr, p[1] + dc] = bridge
        records.append((p, by_axis, over))
    assert records
    # Construction oracle uses the paths' ownership records and local masks.
    out = inp.copy()
    for p in visits:
        out[p] = bg
    active = {i for _, owners, _ in records for i in owners.values()}
    for i in active:
        for p in source_cells[i]:
            out[p] = bg
    for (r, c), owners, over in records:
        for axis in (1 - over, over):
            i = owners[axis]
            m = masks[i]
            for y, x in np.argwhere(m):
                out[r + int(y) - m.shape[0] // 2, c + int(x) - m.shape[1] // 2] = colors[i]
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    turns = int(rng.integers(4))
    return {"input": np.rot90(inp, turns).copy(), "output": np.rot90(out, turns).copy()}


def _parse_canonical(g, bg, carrier):
    cells = np.argwhere(g == carrier)
    assert cells.size
    start, end = int(cells[:, 0].min()), int(cells[:, 0].max())
    assert start >= 3 and end >= start + 5 and end < g.shape[0] - 1
    centers = list(map(int, np.flatnonzero(g[start] == carrier)))
    assert 2 <= len(centers) <= 5 and centers == list(np.flatnonzero(g[end] == carrier))
    assert all(b - a >= 4 for a, b in zip(centers, centers[1:]))
    assert np.all(g[end + 1:] == bg)
    source = g[:start]
    labels = sorted(set(map(int, np.unique(source))) - {bg})
    assert len(labels) == len(centers) and carrier not in labels
    templates = []
    for color in labels:
        pts = np.argwhere(source == color)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        h, w = map(int, hi - lo + 1)
        assert h in (1, 3, 5) and w in (1, 3, 5) and max(h, w) >= 3
        assert hi[0] == start - 2 and lo[0] >= 1 and lo[1] >= 1 and hi[1] < g.shape[1] - 1
        center = (int((lo[0] + hi[0]) // 2), int((lo[1] + hi[1]) // 2))
        assert center[1] in centers
        mask = g[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] == color
        assert _connected(mask)
        templates.append(dict(color=color, cells=list(map(tuple, pts)), shape=(h, w), mask=mask,
                              offsets=pts - np.array(center), column=center[1]))
    templates.sort(key=lambda t: t["column"])
    assert [t["column"] for t in templates] == centers
    field = g[start:end + 1]
    other = set(map(int, np.unique(field))) - {bg, carrier}
    assert len(other) == 1 and not other.intersection(labels)
    bridge = other.pop()
    visible = set(map(tuple, np.argwhere((g == carrier) | (g == bridge))))
    graph = {}
    crossings = {}
    endpoints = {(start, c) for c in centers} | {(end, c) for c in centers}
    used_marks = set()
    for r, c in visible:
        edges = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(_DIRS) if (r + dr, c + dc) in visible}
        graph[r, c] = edges
        if (r, c) in endpoints:
            assert len(edges) == 1 and (2 if r == start else 0) in edges and g[r, c] == carrier
        elif len(edges) == 4:
            assert g[r, c] == carrier
            marked = [d for d, p in edges.items() if g[p] == bridge]
            assert marked in ([0, 2], [1, 3])
            crossings[r, c] = dict(over=marked[0] % 2, owners={})
            used_marks.update(edges[d] for d in marked)
        else:
            assert len(edges) == 2, "broken cable, unmarked junction or adjacent strands"
    assert crossings and used_marks == set(map(tuple, np.argwhere(g == bridge)))
    routes, reached, all_seen = [], [], set()
    for i, c in enumerate(centers):
        p, incoming = (start, c), 0
        states, route = set(), []
        while True:
            assert (p, incoming) not in states, "cable cycle"
            states.add((p, incoming)); route.append(p); all_seen.add(p)
            if p[0] == end:
                assert p in endpoints
                reached.append(p[1])
                break
            if p in crossings:
                owners = crossings[p]["owners"]
                assert incoming % 2 not in owners
                owners[incoming % 2] = i
                outgoing = (incoming + 2) % 4
            elif p == (start, c):
                outgoing = 2
            else:
                assert incoming in graph[p]
                outgoing = next(d for d in graph[p] if d != incoming)
            assert outgoing in graph[p]
            p, incoming = graph[p][outgoing], (outgoing + 2) % 4
            assert p[0] != start
        routes.append(route)
    assert sorted(reached) == centers and all_seen == visible
    boxes = []
    for (r, c), cross in crossings.items():
        assert set(cross["owners"]) == {0, 1} and len(set(cross["owners"].values())) == 2
        rh = max(templates[i]["shape"][0] // 2 for i in cross["owners"].values())
        rw = max(templates[i]["shape"][1] // 2 for i in cross["owners"].values())
        assert r - rh >= start and r + rh <= end and c - rw >= 1 and c + rw < g.shape[1] - 1
        boxes.append((r - rh, r + rh, c - rw, c + rw))
    for a, b in combinations(boxes, 2):
        assert a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2], "different crossing prints must be disjoint"
    return dict(grid=g, bg=bg, carrier=carrier, bridge=bridge, templates=templates,
                start=start, end=end, centers=centers, crossings=crossings,
                visible=visible, routes=routes, pairing=tuple(centers.index(c) for c in reached))


def _parse(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert all(int(a[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1)))
    candidates = []
    for rotation in range(4):
        g = np.rot90(a, rotation)
        for carrier in sorted(set(map(int, np.unique(g))) - {bg}):
            try:
                info = _parse_canonical(g, bg, carrier)
                info["rotation"] = rotation
                candidates.append(info)
            except (AssertionError, StopIteration, IndexError, KeyError):
                pass
    assert len(candidates) == 1, f"expected one visible cable interpretation, found {len(candidates)}"
    return candidates[0]


def _render(info, alternative=None):
    out = info["grid"].copy()
    templates = info["templates"]
    active = {i for cross in info["crossings"].values() for i in cross["owners"].values()}
    for p in info["visible"]:
        out[p] = info["bg"]
    for i in active:
        for p in templates[i]["cells"]:
            out[p] = info["bg"]
    for (r, c), cross in info["crossings"].items():
        owners, over = cross["owners"], cross["over"]
        if alternative == "nearest":
            nearest = sorted(range(len(templates)), key=lambda i: (abs(templates[i]["column"] - c), i))
            owners = {0: nearest[0], 1: nearest[1]}
        order = [owners[1 - over], owners[over]]
        if alternative == "reverse_bridge":
            order.reverse()
        elif alternative == "source_order":
            order.sort(reverse=True)
        for i in order:
            t = templates[i]
            offsets = t["offsets"]
            if alternative == "boxes":
                h, w = t["shape"]
                offsets = [(y - h // 2, x - w // 2) for y in range(h) for x in range(w)]
            elif alternative == "point":
                offsets = [(0, 0)]
            for y, x in offsets:
                out[r + int(y), c + int(x)] = t["color"]
    return np.rot90(out, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using this input alone; bridge-arm color means the upper strand."""
    return _render(_parse(input_grid))


def inspect_pair(pair):
    info = _parse(pair["input"])
    correct = _render(info)
    ts = info["templates"]
    active = {i for cross in info["crossings"].values() for i in cross["owners"].values()}
    both_visible = 0
    for (r, c), cross in info["crossings"].items():
        offsets = [{tuple(map(int, p)) for p in ts[i]["offsets"]} for i in cross["owners"].values()]
        both_visible += bool(offsets[0] - offsets[1]) and bool(offsets[1] - offsets[0])
    differences = {mode: int(np.count_nonzero(correct != _render(info, mode)))
                   for mode in ("nearest", "reverse_bridge", "source_order", "boxes", "point")}
    return dict(shape=tuple(np.asarray(pair["input"]).shape), rotation=info["rotation"],
                roles=(info["bg"], info["carrier"], info["bridge"]), count=len(ts), active=len(active),
                crossings=len(info["crossings"]), both_visible=int(both_visible), pairing=info["pairing"],
                identity=info["pairing"] == tuple(range(len(ts))),
                shapes=tuple(t["shape"] for t in ts),
                masks=tuple((t["shape"], t["mask"].tobytes()) for t in ts),
                empty_centers=sum(not t["mask"][t["shape"][0] // 2, t["shape"][1] // 2] for t in ts),
                differences=differences)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed % (1 << 64))
    max_train = int(rng.integers(3, 5))
    lessons = ["transport", "layers"] + [None] * (nt - 2)
    rng.shuffle(lessons)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = lessons[index] if split == "train" else None
            n = int(rng.integers(3 if lesson else 2, max_train + 1)) if split == "train" else max_train + 1
            for _ in range(24):
                pair = _make_pair(rng, n, lesson)
                if _key(pair["input"]) not in seen:
                    if lesson == "transport" and not inspect_pair(pair)["differences"]["nearest"]:
                        continue
                    break
            else:
                raise AssertionError("bounded distinct/evidence proposals exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Structural, graph, rendering, preservation and training-evidence checks."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, _render(info)), "wrong crossing print, erasure, layering or preservation"
            assert not np.array_equal(inp, out)
            canonical = np.rot90(out, info["rotation"])
            active = {i for cross in info["crossings"].values() for i in cross["owners"].values()}
            for i, t in enumerate(info["templates"]):
                assert all(int(canonical[p]) == (info["bg"] if i in active else t["color"]) for p in t["cells"])
            assert not np.any((out == info["carrier"]) | (out == info["bridge"]))
            features[split].append(inspect_pair(pair))
    train = features["train"]
    for shortcut in ("nearest", "reverse_bridge", "source_order", "boxes", "point"):
        assert any(f["differences"][shortcut] for f in train), f"missing distinguishing lesson: {shortcut}"
    assert any(f["active"] < f["count"] for f in train), "unused source preservation is not demonstrated"
    assert any(f["crossings"] >= 2 and f["both_visible"] >= 2 for f in train)
    assert any(len(set(f["shapes"])) > 1 for f in train)
    assert min(f["count"] for f in features["test"]) > max(f["count"] for f in train), "missing bundle-size shift"


def self_test():
    """42 seeds, count boundaries, metamorphic checks and deliberate mutations.

    Runs in memory, retains only hash/feature sets and one current puzzle. Writes
    no samples, logs, checkpoints or execution traces. The return is diagnostic.
    """
    signatures, shapes, masks, roles, rotations, counts = set(), set(), set(), set(), set(), set()
    pairings, crossing_counts, active_counts = set(), set(), set()
    pairs = empty = identities = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                f = inspect_pair(pair)
                signature.update(pair["input"].tobytes()); signature.update(pair["output"].tobytes())
                shapes.add(f["shape"]); masks.update(f["masks"]); roles.add(f["roles"])
                rotations.add(f["rotation"]); counts.add(f["count"]); pairings.add(f["pairing"])
                crossing_counts.add(f["crossings"]); active_counts.add((f["count"], f["active"]))
                pairs += 1; empty += f["empty_centers"] > 0; identities += f["identity"]
        signatures.add(signature.digest())
    assert len(signatures) == 42 and len(shapes) >= 35 and len(masks) >= 80 and len(roles) >= 50
    assert rotations == set(range(4)) and counts == {2, 3, 4, 5}
    assert len(pairings) >= 25 and len(crossing_counts) >= 6 and len(active_counts) >= 6
    assert empty >= 10 and identities >= 42
    p = generate_puzzle(123)
    q = generate_puzzle(123)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in ("input", "output"))
    # Color-role bijections and all rotations must commute with solve.
    rng = np.random.default_rng(90817)
    for pair in p["train"] + p["test"]:
        lut = rng.permutation(10)
        for k in range(4):
            assert np.array_equal(solve(np.rot90(lut[pair["input"]], k)), np.rot90(lut[pair["output"]], k))
    # Change only a crossing's visible bridge direction. Precisely the shared
    # occupied pixels must switch ownership; neither cable identity nor source
    # eligibility may change. This checks the new connecting concept directly.
    for pair in p["train"]:
        info = _parse(pair["input"])
        for (r, c), cross in info["crossings"].items():
            ts = info["templates"]
            masks_here = [{tuple(map(int, q)) for q in ts[i]["offsets"]} for i in cross["owners"].values()]
            common = masks_here[0] & masks_here[1]
            if not common:
                continue
            changed = info["grid"].copy()
            for d, (dr, dc) in enumerate(_DIRS):
                changed[r + dr, c + dc] = info["bridge"] if d % 2 != cross["over"] else info["carrier"]
            changed = np.rot90(changed, -info["rotation"])
            prediction = solve(changed)
            expected = np.zeros(info["grid"].shape, bool)
            for y, x in common:
                expected[r + y, c + x] = True
            assert np.array_equal(prediction != pair["output"], np.rot90(expected, -info["rotation"]))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(91, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for alternative in ("nearest", "reverse_bridge", "source_order", "boxes", "point"):
        i = next(i for i, pair in enumerate(p["train"]) if inspect_pair(pair)["differences"][alternative])
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        damaged["train"][i]["output"] = _render(_parse(p["train"][i]["input"]), alternative)
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepts {alternative}")
    # Selective removal and structural erasure are separate from stamp pixels.
    for defect in ("erase_unused", "retain_wire"):
        damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        for i, pair in enumerate(p["train"]):
            info = _parse(pair["input"])
            active = {j for cross in info["crossings"].values() for j in cross["owners"].values()}
            inactive = set(range(len(info["templates"]))) - active
            if defect == "erase_unused" and not inactive:
                continue
            wrong = np.rot90(pair["output"], info["rotation"]).copy()
            if defect == "erase_unused":
                cell = info["templates"][min(inactive)]["cells"][0]
                wrong[cell] = info["bg"]
            else:
                wrong[info["start"], info["centers"][0]] = info["carrier"]
            damaged["train"][i]["output"] = np.rot90(wrong, -info["rotation"])
            break
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepts {defect}")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                template_masks=len(masks), role_assignments=len(roles), pairings=len(pairings),
                crossing_counts=sorted(crossing_counts), empty_center_pairs=empty,
                identity_pairing_pairs=identities, independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
