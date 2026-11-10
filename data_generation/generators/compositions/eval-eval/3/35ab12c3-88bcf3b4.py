"""Finite shadow threads on sparse directed guide contours.

Only standard-library modules and NumPy are used. No parent module is imported.
The input parser, topology search, and renderer have no generation state.
"""
from __future__ import annotations

import copy
import json
import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    step = (int(np.sign(dr)), int(np.sign(dc)))
    return [(a[0] + k * step[0], a[1] + k * step[1]) for k in range(n + 1)]


def _connected(cells, directions=DIR8):
    if not cells:
        return False
    reached = {min(cells)}
    todo = list(reached)
    while todo:
        p = todo.pop()
        for d in directions:
            q = _add(p, d)
            if q in cells and q not in reached:
                reached.add(q)
                todo.append(q)
    return reached == cells


def _cross(a, b, c):
    return (b[0]-a[0]) * (c[1]-a[1]) - (b[1]-a[1]) * (c[0]-a[0])


def _intersects(a, b, c, d):
    # Continuous geometry detects crossings between diagonal cell centers too.
    if any(max(a[k], b[k]) < min(c[k], d[k]) or
           max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return (_cross(a, b, c) * _cross(a, b, d) <= 0 and
            _cross(c, d, a) * _cross(c, d, b) <= 0)


def _candidate_links(points):
    cells = set(points)
    links = {}
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            line = _segment(a, points[j])
            if line is not None and not cells.intersection(line[1:-1]):
                links[i, j] = line
    return links


def _trace(points, anchor, cue):
    """Unique simple full cycle, otherwise unique full path, rooted by the stub.

    Every landmark participates. Missing segments may not run through another
    landmark, cross, or form an interior chord of the selected contour.
    """
    assert 3 <= len(points) <= 12
    start, second = points.index(anchor), points.index(cue)
    links = _candidate_links(points)
    adjacency = {i: [] for i in range(len(points))}
    for i, j in links:
        adjacency[i].append(j)
        adjacency[j].append(i)
    assert second in adjacency[start]

    def search(closed):
        answers = []
        visits = 0

        def walk(order):
            nonlocal visits
            visits += 1
            assert visits <= 100000, "topology search exceeded family bound"
            if len(answers) >= 2:
                return
            if len(order) == len(points):
                if closed:
                    if start not in adjacency[order[-1]]:
                        return
                    a, b = points[order[-1]], anchor
                    if any(_intersects(a, b, points[order[k]], points[order[k+1]])
                           for k in range(1, len(order) - 2)):
                        return
                answers.append(order)
                return
            for j in adjacency[order[-1]]:
                if j in order:
                    continue
                a, b = points[order[-1]], points[j]
                if any(_intersects(a, b, points[order[k]], points[order[k+1]])
                       for k in range(len(order) - 2)):
                    continue
                walk(order + [j])

        walk([start, second])
        return answers

    cycles = search(True)
    assert len(cycles) <= 1, "ambiguous closed guide"
    choices = cycles or search(False)
    assert len(choices) == 1, "ambiguous or disconnected directed guide"
    order = [points[i] for i in choices[0]]
    closed = bool(cycles)
    sequence = []
    edges = list(zip(order, order[1:]))
    if closed:
        edges.append((order[-1], order[0]))
    for a, b in edges:
        sequence.extend(_segment(a, b)[:-1])
    if not closed:
        sequence.append(order[-1])
    assert len(sequence) == len(set(sequence)), "guide touches itself"
    all_link_cells = set().union(*(set(line) for line in links.values()))
    return sequence, closed, order, all_link_cells - set(sequence)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _decode(value):
    inp = _grid(value)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3, "one background, one guide, one thread required"
    background = int(values[np.argmax(counts)])
    assert int(max(counts)) > inp.size // 2, "background must be a strict majority"
    groups = [(int(c), {tuple(map(int, p)) for p in np.argwhere(inp == c)})
              for c in values if c != background]
    masses = [(c, ps) for c, ps in groups if _connected(ps)]
    guides = [(c, ps) for c, ps in groups if not _connected(ps)]
    assert len(masses) == len(guides) == 1, "roles must be geometrically distinct"
    thread_color, slack = masses[0]
    guide_color, landmarks = guides[0]
    assert 3 <= len(slack) <= 48
    contacts = [(p, a) for p in sorted(slack) for a in sorted(landmarks)
                if _distance(p, a) == 1]
    assert len(contacts) == 1, "thread has one unique landmark attachment"
    root, anchor = contacts[0]
    unit_pairs = [(a, b) for a in sorted(landmarks) for b in sorted(landmarks)
                  if a < b and _distance(a, b) == 1]
    assert len(unit_pairs) == 1 and anchor in unit_pairs[0], "one launch stub required"
    cue = next(p for p in unit_pairs[0] if p != anchor)
    offset = root[0] - anchor[0], root[1] - anchor[1]
    sequence, closed, order, rejected = _trace(sorted(landmarks), anchor, cue)
    guide = set(sequence)
    assert not guide & slack, "slack cannot obscure the guide"
    assert len(slack) <= len(sequence), "thread may not lap or extend the guide"
    translated = [_add(p, offset) for p in sequence[:len(slack)]]
    assert all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]
               for r, c in translated), "finite shadow must fit the canvas"
    assert translated[0] == root and root not in guide
    return inp, dict(background=background, guide_color=guide_color,
                     thread_color=thread_color, slack=slack, landmarks=landmarks,
                     root=root, anchor=anchor, cue=cue, offset=offset,
                     sequence=sequence, guide=guide, closed=closed, order=order,
                     rejected=rejected, translated=translated)


def _render(inp, info, *, sequence=None, length=None, offset=None, visible_budget=False):
    out = np.full_like(inp, info["background"])
    for p in info["guide"]:
        out[p] = info["guide_color"]
    seq = info["sequence"] if sequence is None else sequence
    n = len(info["slack"]) if length is None else length
    d = info["offset"] if offset is None else offset
    drawn = 0
    for p in seq if visible_budget else seq[:n]:
        q = _add(p, d)
        if not (0 <= q[0] < out.shape[0] and 0 <= q[1] < out.shape[1]):
            continue  # Only counterfactuals can leave the generated canvas.
        if q not in info["guide"]:
            out[q] = info["thread_color"]
            drawn += 1
            if visible_budget and drawn == n:
                break
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from colors, contact, direction stub, topology, and count."""
    inp, info = _decode(input_grid)
    return _render(inp, info)


def _simplify(points, closed):
    changed = True
    while changed and len(points) > (3 if closed else 2):
        changed = False
        for i in range(len(points)):
            if not closed and i in (0, len(points) - 1):
                continue
            a, b, c = points[i-1], points[i], points[(i+1) % len(points)]
            if _cross(a, b, c) == 0 and b in (_segment(a, c) or []):
                points = points[:i] + points[i+1:]
                changed = True
                break
    return points


def _shape(rng, kind):
    if kind == "path":
        n = int(rng.integers(3, 7))
        points = [(0, 0)]
        for _ in range(n - 1):
            d = DIR8[int(rng.integers(8))]
            length = int(rng.integers(3, 8))
            points.append((points[-1][0]+length*d[0], points[-1][1]+length*d[1]))
        return _simplify(points, False), False
    if kind == "segment":
        d = DIR8[int(rng.integers(8))]
        n = int(rng.integers(5, 16))
        return [(0, 0), (n*d[0], n*d[1])], False
    if kind == "diamond":
        radius = int(rng.integers(3, 8))
        return [(0, radius), (radius, 2*radius), (2*radius, radius), (radius, 0)], True
    h, w = int(rng.integers(5, 11)), int(rng.integers(5, 13))
    if kind == "notch":
        w = max(w, 8)
        h = max(h, 6)
        left = int(rng.integers(2, w-3))
        right = int(rng.integers(left+2, w-1))
        depth = int(rng.integers(2, h-2))
        return [(0, 0), (0, left), (depth, left), (depth, right),
                (0, right), (0, w), (h, w), (h, 0)], True
    cuts = [int(rng.integers(0, max(1, min(h, w)//3))) for _ in range(4)]
    if kind == "square":
        w, cuts = h, [0]*4
    a, b, c, d = cuts
    points = [(0, a), (0, w-b), (b, w), (h-c, w),
              (h, w-c), (h, d), (h-d, 0), (a, 0)]
    points = [p for i, p in enumerate(points) if p != points[i-1]]
    return _simplify(points, True), True


def _oriented_guide(rng, points, closed, interior_anchor):
    if len(points) != len(set(points)):
        return None
    if closed:
        if rng.random() < .5:
            points = list(reversed(points))
        edge = int(rng.integers(len(points)))
        points = points[edge:] + points[:edge]
        if interior_anchor:
            line = _segment(points[0], points[1])
            if line is None or len(line) < 7:
                return None
            k = int(rng.integers(2, len(line)-3))
            anchor = line[k]
            points = [anchor] + points[1:] + [points[0]]
    elif rng.random() < .5:
        points = list(reversed(points))
    line = _segment(points[0], points[1])
    if line is None or len(line) < 4:
        return None
    points = [points[0], line[1]] + points[1:]
    if len(points) > 12 or len(set(points)) != len(points):
        return None
    anchor, cue = points[:2]
    pairs = [(a, b) for i, a in enumerate(points) for b in points[i+1:]
             if _distance(a, b) == 1]
    if pairs != [(anchor, cue)]:
        return None
    if max(max(p[k] for p in points)-min(p[k] for p in points) for k in (0, 1)) > 19:
        return None
    try:
        sequence, inferred_closed, order, rejected = _trace(sorted(points), anchor, cue)
    except AssertionError:
        return None
    if inferred_closed != closed or order != points:
        return None
    return points, sequence, rejected


def _slack(rng, root, anchor, offset, landmarks, guide, shadow, length, diagonal):
    radius = int(rng.integers(5, 9))
    style = int(rng.integers(3))  # dense growth, sparse growth, directional filament
    forbidden = guide | (shadow - {root})
    available = set()
    for r in range(root[0]-radius, root[0]+radius+1):
        for c in range(root[1]-radius, root[1]+radius+1):
            p = r, c
            dot = (r-anchor[0])*offset[0] + (c-anchor[1])*offset[1]
            if p != root and (p in forbidden or any(_distance(p, v) <= 1 for v in landmarks)
                              or dot <= offset[0]**2+offset[1]**2):
                continue
            if diagonal and _distance(p, root) == 1 and p != root and p in {_add(root, d) for d in DIR4}:
                continue
            available.add(p)
    cells = {root}
    while len(cells) < length:
        frontier = sorted({_add(p, d) for p in cells for d in DIR8} & available - cells)
        if not frontier:
            return None
        scores = []
        for p in frontier:
            neighbors = sum(_add(p, d) in cells for d in DIR8)
            reach = abs(p[0]-root[0]) + abs(p[1]-root[1])
            scores.append((neighbors+1)**3 if style == 0 else
                          1/(neighbors+1)**2 if style == 1 else (reach+1)**2/(neighbors+1))
        weights = np.asarray(scores, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
    return cells


def _fit(rng, landmarks, slack, guide, shadow, colors):
    occupied = set(landmarks) | slack | guide | shadow
    lo = [min(p[k] for p in occupied) for k in (0, 1)]
    lengths = [max(p[k] for p in occupied)-lo[k]+1 for k in (0, 1)]
    if max(lengths) > 30:
        return None
    padding = [int(rng.integers(0, min(7, 30-n)+1)) for n in lengths]
    if rng.random() < .3:
        side = max(lengths[k]+padding[k] for k in (0, 1))
        padding = [side-n for n in lengths]
    dims = [lengths[k]+padding[k] for k in (0, 1)]
    # The uniform canvas visibly determines its own background role.
    while dims[0]*dims[1] <= 2*(len(landmarks)+len(slack)):
        k = 0 if dims[0] <= dims[1] else 1
        if dims[k] == 30:
            k = 1-k
        if dims[k] == 30:
            return None
        dims[k] += 1
        padding[k] += 1
    shift = [int(rng.integers(padding[k]+1))-lo[k] for k in (0, 1)]
    move = lambda p: (p[0]+shift[0], p[1]+shift[1])
    bg, gc, tc = colors
    inp = np.full(dims, bg, dtype=np.int8)
    for p in landmarks:
        inp[move(p)] = gc
    for p in slack:
        inp[move(p)] = tc
    out = np.full_like(inp, bg)
    for p in guide:
        out[move(p)] = gc
    for p in shadow-guide:
        out[move(p)] = tc
    return {"input": inp, "output": out}


def _concave_indices(sequence, closed):
    if not closed:
        return []
    n = len(sequence)
    turns = [_cross(sequence[k-1], sequence[k], sequence[(k+1) % n]) for k in range(n)]
    winding = int(np.sign(sum(turns)))
    return [k for k, turn in enumerate(turns) if turn*winding < 0]


def _make_pair(rng, colors, lesson=None, test_shift=None,
               require_concave=False, require_diagonal=False):
    for attempt in range(700):
        if require_concave:
            kind = "notch"
        elif lesson == "cycle":
            kind = str(rng.choice(["square", "diamond", "cut", "notch"]))
        elif lesson == "path":
            kind = "path"
        elif test_shift == "anchor":
            kind = str(rng.choice(["square", "cut", "notch"]))
        else:
            kind = str(rng.choice(["segment", "path", "square", "diamond", "cut", "notch"],
                                  p=[.12, .22, .14, .15, .2, .17]))
        raw, closed = _shape(rng, kind)
        code = int(rng.integers(8))
        transformed = []
        for r, c in raw:
            if code & 4:
                c = -c
            for _ in range(code % 4):
                r, c = c, -r
            transformed.append((r, c))
        raw = transformed
        oriented = _oriented_guide(rng, raw, closed, test_shift == "anchor")
        if oriented is None:
            continue
        points, sequence, rejected = oriented
        guide_edges = list(zip(points, points[1:])) + ([(points[-1], points[0])] if closed else [])
        if require_diagonal and not any(a[0] != b[0] and a[1] != b[1] for a, b in guide_edges):
            continue
        anchor, cue = points[:2]
        guide = set(sequence)
        if lesson == "cycle" and not rejected:
            continue
        first_turn = next((i for i in range(1, len(sequence)-1)
                           if _cross(sequence[i-1], sequence[i], sequence[i+1]) != 0), len(sequence))
        if lesson is not None:
            low, high = max(5, first_turn+3), min(48, int(len(sequence)*.69))
        elif test_shift == "short":
            low, high = 3, min(48, int(len(sequence)*.69))
        elif test_shift == "length":
            low, high = max(5, int(np.ceil(len(sequence)*.8))), min(48, len(sequence))
        else:
            low, high = 3, min(48, len(sequence))
        concave_indices = _concave_indices(sequence, closed)
        if require_concave:
            interior_turns = [k for k in concave_indices if 0 < k < high-1]
            if not interior_turns:
                continue
            low = max(low, min(interior_turns)+2)
        if high < low:
            continue
        length = int(rng.integers(low, high+1))
        offsets = [d for d in DIR8 if _add(anchor, d) not in guide and
                   sum(_distance(_add(anchor, d), v) == 1 for v in points) == 1]
        rng.shuffle(offsets)
        for offset in offsets:
            root = _add(anchor, offset)
            translated = [_add(p, offset) for p in sequence[:length]]
            shadow = set(translated)
            visible = shadow-guide
            if require_concave and not any(0 < k < length-1 and all(
                    translated[j] not in guide for j in (k-1, k, k+1)) for k in concave_indices):
                continue
            if len(visible) < 3 or translated[-1] in guide:
                continue
            if lesson == "cycle" and (not shadow & guide or
                    len({_add(p, offset) for p in sequence} - guide) <= len(visible)):
                continue
            slack = _slack(rng, root, anchor, offset, points, guide, shadow,
                           length, diagonal=lesson == "path" or rng.random() < .4)
            if slack is None:
                continue
            pair = _fit(rng, points, slack, guide, shadow, colors)
            if pair is None:
                continue
            inp, info = _decode(pair["input"])
            assert np.array_equal(_render(inp, info), pair["output"])
            if lesson == "cycle" and np.array_equal(_render(inp, info, visible_budget=True), pair["output"]):
                continue
            return pair
    raise RuntimeError("bounded construction exhausted; no silent template fallback")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(["length", "anchor"]))
    demo_slots = list(map(int, rng.choice(nt, 2, replace=False)))
    # Swap the two foreground roles between the mandatory demonstrations.
    first_colors = list(map(int, rng.choice(10, 3, replace=False)))
    other_bg = int(rng.choice([c for c in range(10) if c not in first_colors]))
    second_colors = [other_bg, first_colors[2], first_colors[1]]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            lesson = None
            if split == "train" and i == demo_slots[0]:
                lesson, colors = "cycle", first_colors
            elif split == "train" and i == demo_slots[1]:
                lesson, colors = "path", second_colors
            else:
                colors = list(map(int, rng.choice(10, 3, replace=False)))
            for _ in range(20):
                regime = shift if split == "test" else "short" if shift == "length" else None
                pair = _make_pair(rng, colors, lesson, regime)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    # If a query needs a notch or diagonal interpolation, ensure its geometry
    # has observable training evidence even for the minimum two demonstrations.
    for query_field, training_field, slot, colors, lesson, requirements in (
        ("concave", "concave_prefix", demo_slots[0], first_colors, "cycle", {"require_concave": True}),
        ("diagonal", "diagonal", demo_slots[1], second_colors, "path", {"require_diagonal": True}),
    ):
        if (any(_features(p)[query_field] for p in puzzle["test"])
                and not any(_features(p)[training_field] for p in puzzle["train"])):
            for _ in range(20):
                pair = _make_pair(rng, colors, lesson, **requirements)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle["train"][slot] = pair
                    seen.add(key)
                    break
            else:
                raise RuntimeError("geometry demonstration construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _four_root_count(info):
    reached, todo = {info["root"]}, [info["root"]]
    while todo:
        p = todo.pop()
        for d in DIR4:
            q = _add(p, d)
            if q in info["slack"] and q not in reached:
                reached.add(q)
                todo.append(q)
    return len(reached)


def _features(pair):
    inp, i = _decode(pair["input"])
    n = len(i["slack"])
    turns = sum(_cross(a, b, c) != 0 for a, b, c in zip(
        i["sequence"], i["sequence"][1:], i["sequence"][2:]))
    signs = {_cross(i["order"][k-1], i["order"][k], i["order"][(k+1) % len(i["order"])])
             for k in range(len(i["order"]))} if i["closed"] else {0}
    base = i["order"]
    edges = list(zip(base, base[1:])) + ([(base[-1], base[0])] if i["closed"] else [])
    concave_prefix = any(0 < k < n-1 and all(i["translated"][j] not in i["guide"]
                        for j in (k-1, k, k+1)) for k in _concave_indices(i["sequence"], i["closed"]))
    genuine_turns = sum(_cross(a, b, c) != 0 for a, b, c in zip(
        i["sequence"][:n], i["sequence"][1:n], i["sequence"][2:n]))
    return dict(shape=list(inp.shape), background=i["background"], offset=list(i["offset"]),
                closed=i["closed"], vertices=len(base), length=n, guide_length=len(i["sequence"]),
                fraction=round(n/len(i["sequence"]), 3), turns=turns, prefix_turns=genuine_turns,
                concave=any(s > 0 for s in signs) and any(s < 0 for s in signs),
                concave_prefix=concave_prefix,
                diagonal=any(a[0] != b[0] and a[1] != b[1] for a, b in edges),
                rejected_cells=len(i["rejected"]), occluded=len(set(i["translated"]) & i["guide"]),
                four_count=_four_root_count(i),
                interior_anchor=_cross(base[-1], base[0], base[1]) == 0 if i["closed"] else False,
                geometry=repr([(p[0]-i["anchor"][0], p[1]-i["anchor"][1]) for p in base]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, queries = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, info = _decode(pair["input"])
            out = _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert np.array_equal(out, _render(inp, info)), "incorrect directed finite shadow"
            assert not np.array_equal(inp, out)
            assert all(out[p] == info["guide_color"] for p in info["landmarks"])
            assert all(out[p] == info["guide_color"] for p in info["guide"])
            assert out[info["root"]] == info["thread_color"]
            assert np.count_nonzero(out == info["thread_color"]) == len(set(info["translated"])-info["guide"])
            assert np.count_nonzero(out == info["thread_color"]) <= len(info["slack"])
            assert (set(np.unique(out)) <= set(np.unique(inp)))
            if split == "train":
                training.append((inp, info, out))
            else:
                queries.append(info)
    assert len({i["background"] for _, i, _ in training}) >= 2
    assert any(i["closed"] and i["rejected"] for _, i, _ in training), "closure/chord evidence missing"
    assert any(not i["closed"] and len(i["order"]) >= 4 for _, i, _ in training), "bent path evidence missing"
    assert any(not np.array_equal(out, _render(inp, i, visible_budget=True))
               for inp, i, out in training), "hidden positions must visibly spend length"
    assert any(_four_root_count(i) < len(i["slack"]) and not np.array_equal(
        out, _render(inp, i, length=_four_root_count(i))) for inp, i, out in training)
    # Explicit parent ablations: omit A's vector; omit B's finite material budget.
    assert any(not np.array_equal(out, _render(inp, i, offset=(0, 0))) for inp, i, out in training)
    assert any(not np.array_equal(out, _render(inp, i, length=len(i["sequence"])))
               for inp, i, out in training), "whole-contour-copy shortcut unchallenged"
    assert any(i["closed"] and not np.array_equal(out, _render(
        inp, i, sequence=[i["sequence"][0]] + i["sequence"][:0:-1]))
        for inp, i, out in training), "launch direction must affect the answer"
    anchor_inside_edge = lambda i: i["closed"] and _cross(i["order"][-1], i["anchor"], i["cue"]) == 0
    length_shift = (all(len(i["slack"])/len(i["sequence"]) <= .69 for _, i, _ in training)
                    and all(len(i["slack"])/len(i["sequence"]) >= .8 for i in queries))
    anchor_shift = (all(not anchor_inside_edge(i) for _, i, _ in training)
                    and all(anchor_inside_edge(i) for i in queries))
    assert length_shift or anchor_shift, "meaningful visible train/test shift missing"
    train_features = [_features(p) for p in puzzle["train"]]
    query_features = [_features(p) for p in puzzle["test"]]
    for query_field, training_field in (("concave", "concave_prefix"), ("diagonal", "diagonal")):
        assert not any(f[query_field] for f in query_features) or any(f[training_field] for f in train_features), \
            f"query {query_field} geometry lacks training evidence"


def _semantic_regressions():
    # Hand-specified guides and translated prefixes, independent of topology
    # search and generation. In each case one conceptual position is hidden.
    fixtures = [
        ((14, 14), (0, 2, 8),
         {(5, 5), (5, 6), (5, 10), (10, 10), (10, 5)},
         {(6, 4), (7, 3), (8, 3), (8, 2), (9, 2), (9, 1), (10, 1), (10, 2), (11, 1)},
         ({(5, c) for c in range(5, 11)} | {(10, c) for c in range(5, 11)} |
          {(r, 5) for r in range(5, 11)} | {(r, 10) for r in range(5, 11)}),
         [(6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (7, 9), (8, 9), (9, 9)]),
        ((12, 14), (7, 0, 3),
         {(4, 4), (4, 5), (4, 10), (11, 10)},
         {(3, 3), (2, 2), (2, 1), (1, 1), (1, 2), (0, 1), (0, 2), (0, 0), (1, 0)},
         ({(4, c) for c in range(4, 11)} | {(r, 10) for r in range(5, 12)}),
         [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 9), (5, 9)]),
    ]
    for shape, colors, landmarks, slack, guide, shadow in fixtures:
        bg, gc, tc = colors
        inp, expected = np.full(shape, bg, dtype=np.int8), np.full(shape, bg, dtype=np.int8)
        for p in landmarks:
            inp[p] = gc
        for p in slack:
            inp[p] = tc
        for p in guide:
            expected[p] = gc
        for p in set(shadow)-guide:
            expected[p] = tc
        before = inp.copy()
        assert np.array_equal(solve(inp), expected)
        _, info = _decode(inp)
        assert info["translated"] == shadow
        assert np.count_nonzero(expected == tc) == 8 and _four_root_count(info) == 1
        assert np.array_equal(inp, before)


def self_test():
    """Bounded 42-seed semantic, variation, symmetry, and mutation checks."""
    _semantic_regressions()
    features = []
    normalized = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                features.append((split, _features(pair)))
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.tolist()), out)
                before = inp.copy()
                solve(inp)
                assert np.array_equal(inp, before), "solve mutated its input"
                permutation = np.random.default_rng(seed+12345).permutation(10)
                assert np.array_equal(solve(permutation[inp]), permutation[out])
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.ravel())))
        # All eight whole-scene symmetries preserve the visibly encoded direction.
        pair = p["test"][0]
        for k in range(4):
            for reflect in (False, True):
                inp, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                if reflect:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                assert np.array_equal(solve(inp), out)
    fs = [f for _, f in features]
    assert {tuple(f["offset"]) for f in fs} == set(DIR8)
    assert {f["background"] for f in fs} == set(range(10))
    assert {f["closed"] for f in fs} == {False, True}
    assert len({tuple(f["shape"]) for f in fs}) >= 40
    assert len({f["geometry"] for f in fs}) >= 100
    assert len(normalized) >= 180
    assert any(f["concave"] for f in fs) and any(f["diagonal"] for f in fs)
    assert any(f["interior_anchor"] for split, f in features if split == "test")
    assert any(f["fraction"] >= .8 for split, f in features if split == "test")
    assert any(f["prefix_turns"] >= 3 for f in fs)
    assert max(f["length"] for f in fs) >= 30
    assert min(f["length"] for f in fs) <= 5
    # Determinism and every legal count combination, plus type/range rejection.
    encode = lambda p: json.dumps({s: [{k: np.asarray(v).tolist() for k, v in pair.items()}
                                       for pair in ps] for s, ps in p.items()}, separators=(",", ":"))
    assert encode(generate_puzzle(71)) == encode(generate_puzzle(71))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9000+nt*10+nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    for mutation in ("budget", "preservation"):
        damaged = copy.deepcopy(p)
        pair = damaged["test"][0]
        inp, i = _decode(pair["input"])
        if mutation == "budget":
            wrong = _render(inp, i, length=max(1, len(i["slack"])-3))
            assert not np.array_equal(wrong, pair["output"])
            pair["output"] = wrong
        else:
            pair["output"][i["cue"]] = i["background"]
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            continue
        raise AssertionError("wrong semantic output accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed self_test passed")
