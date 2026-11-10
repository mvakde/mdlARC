"""Folded stencil filter: exterior template -> local interior color query.

Only standard library and NumPy are used. The solver reads a single input;
generation plans, seeds, and outputs are never consulted by the input parser.
"""

from itertools import combinations
import json
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        queue = [p]
        for y, x in queue:
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
        result.append(set(queue))
    return result


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys) + 1, min(xs), max(xs) + 1


def _area(box):
    t, b, l, r = box
    return (b - t) * (r - l)


def _outline(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r)} | {(b - 1, x) for x in range(l, r)}
            | {(y, l) for y in range(t, b)} | {(y, r - 1) for y in range(t, b)})


def _corners(box):
    t, b, l, r = box
    return ((t, l), (t, r - 1), (b - 1, l), (b - 1, r - 1))


def _positions(box, q, pattern, outside=False):
    y, x = _corners(box)[q]
    dy, dx = DIRS[q]
    sign = -1 if outside else 1
    return {(y + sign * dy * u, x + sign * dx * v) for u, v in pattern}


def _connected(pattern):
    seen = {min(pattern)}
    queue = list(seen)
    for u, v in queue:
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = u + du, v + dv
                if p in pattern and p not in seen:
                    seen.add(p)
                    queue.append(p)
    return seen == set(pattern)


def _gap(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _parse_roles(a):
    """The unique 5x5 selector identifies frame, padding background, and query."""
    interpretations = []
    for frame in map(int, np.unique(a)):
        components = _components(a == frame)
        boxes = []
        for comp in components:
            box = _bbox(comp)
            if box[1] - box[0] < 5 or box[3] - box[2] < 5 or comp != _outline(box):
                break
            boxes.append(box)
        else:
            small = [b for b in boxes if b[1] - b[0] == b[3] - b[2] == 5]
            records = [b for b in boxes if b[1] - b[0] >= 6 and b[3] - b[2] >= 6]
            if len(small) != 1 or not records or len(records) + 1 != len(boxes):
                continue
            selector = small[0]
            t, b, l, r = selector
            vals, counts = np.unique(a[t + 1:b - 1, l + 1:r - 1], return_counts=True)
            pads = vals[counts == 6]
            if len(pads) != 1:
                continue
            bg = int(pads[0])
            query = frozenset(int(c) for c in vals if c != bg)
            if len(query) != 3 or frame == bg or frame in query:
                continue
            if not all(_gap(x, y) for i, x in enumerate(boxes) for y in boxes[:i]):
                continue
            interpretations.append((bg, frame, query, selector, sorted(records)))
    if len(interpretations) != 1:
        raise ValueError("Missing or ambiguous closed-frame selector")
    return interpretations[0]


def _read(input_grid):
    a = _grid(input_grid)
    bg, frame, query, selector, boxes = _parse_roles(a)
    occupied = np.zeros(a.shape, dtype=bool)
    for t, b, l, r in [selector] + boxes:
        occupied[t:b, l:r] = True
    exterior = [tuple(map(int, p)) for p in np.argwhere((a != bg) & ~occupied)]
    inks = {int(a[p]) for p in exterior}
    if len(inks) != 1:
        raise ValueError("All exterior ornaments must share one ink color")
    ink = inks.pop()
    if ink in query or ink in (bg, frame):
        raise ValueError("Ink must be distinct from labels and structural colors")
    clues = [{q: set() for q in range(4)} for _ in boxes]
    for y, x in exterior:
        options = []
        for i, box in enumerate(boxes):
            for q, (cy, cx) in enumerate(_corners(box)):
                dy, dx = DIRS[q]
                u, v = (cy - y) * dy, (cx - x) * dx
                if 1 <= u <= 3 and 1 <= v <= 3:
                    options.append((i, q, u, v))
        if len(options) != 1:
            raise ValueError("Exterior ink has no unique outward corner association")
        i, q, u, v = options[0]
        clues[i][q].add((u, v))
    patterns = [frozenset(s) for record in clues for s in record.values() if s]
    complete = {s for s in patterns if all(p <= s for p in patterns)}
    if len(complete) != 1:
        raise ValueError("An intact common stencil must contain all partial copies")
    stencil = complete.pop()
    radius = max(max(p) for p in stencil)
    if not (3 <= len(stencil) <= 8 and _connected(stencil)):
        raise ValueError("Stencil must be an 8-connected set of 3..8 cells")
    if len(stencil) == radius ** 2:
        raise ValueError("This family reserves at least one off-stencil probe site")
    interiors = []
    for t, b, l, r in boxes:
        if b - t < 2 * radius + 2 or r - l < 2 * radius + 2:
            raise ValueError("Opposite corner probe neighborhoods must be disjoint")
        interior = a[t + 1:b - 1, l + 1:r - 1]
        colors, counts = np.unique(interior, return_counts=True)
        cmap = {int(c): int(n) for c, n in zip(colors, counts)}
        if set(cmap) & {bg, frame, ink}:
            raise ValueError("Record interiors are solid filler/label fields")
        interiors.append(cmap)
    common = set.intersection(*(set(c) for c in interiors)) - set(query)
    totals = {c: sum(d.get(c, 0) for d in interiors) for c in common}
    if not totals:
        raise ValueError("No common non-query filler")
    largest = max(totals.values())
    fillers = [c for c, n in totals.items() if n == largest]
    if len(fillers) != 1:
        raise ValueError("Filler must be uniquely most frequent among common non-query colors")
    filler = fillers[0]
    records = []
    all_completed = set()
    for i, box in enumerate(boxes):
        gates = []
        for q in range(4):
            probe = _positions(box, q, stencil)
            labels = frozenset(int(a[p]) for p in probe if a[p] != filler)
            marked = bool(clues[i][q])
            outside = _positions(box, q, stencil, True)
            if marked:
                for y, x in outside:
                    if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]):
                        raise ValueError("Completed ornaments must be entirely visible")
                    if occupied[y, x] or a[y, x] not in (bg, ink) or (y, x) in all_completed:
                        raise ValueError("Completed ornaments conflict with another object")
                    # Completion cannot silently join a different corner's fragment.
                    associations = [(j, k) for j, other in enumerate(boxes)
                                    for k, (cy, cx) in enumerate(_corners(other))
                                    if 1 <= (cy-y)*DIRS[k][0] <= 3
                                    and 1 <= (cx-x)*DIRS[k][1] <= 3]
                    if associations != [(i, q)]:
                        raise ValueError("A completed ornament would have ambiguous ownership")
                all_completed |= outside
            gates.append({"q": q, "marked": marked, "clue": frozenset(clues[i][q]),
                          "labels": labels, "probe": probe, "outside": outside,
                          "passes": marked and query <= labels})
        records.append({"box": box, "gates": gates, "positive": any(g["passes"] for g in gates),
                        "labels": frozenset(set(interiors[i]) - {filler})})
    return {"array": a, "background": bg, "frame": frame, "filler": filler, "ink": ink,
            "query": query, "selector": selector, "stencil": stencil, "radius": radius,
            "records": records}


def _answer(info):
    out = info["array"].copy()
    for record in info["records"]:
        if record["positive"]:
            t, b, l, r = record["box"]
            out[t:b, l:r] = info["background"]
        for gate in record["gates"]:
            if gate["marked"]:
                for p in gate["outside"]:
                    out[p] = info["background"] if record["positive"] else info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the stencil, fold its addresses, filter records, finish survivors."""
    return _answer(_read(input_grid))


def _sample_stencil(rng, radius, minimum=3, asymmetric=False, forbidden=()):
    domain = [(u, v) for u in range(1, radius + 1) for v in range(1, radius + 1)]
    # Enumerate a domain of at most nine cells, not a stored shape dictionary.
    candidates = [frozenset(c) for n in range(minimum, min(8, radius ** 2 - 1) + 1)
                  for c in combinations(domain, n)
                  if max(max(p) for p in c) == radius and _connected(c)
                  and (not asymmetric or frozenset((v, u) for u, v in c) != frozenset(c))
                  and frozenset(c) not in forbidden]
    if not candidates:
        raise ValueError("No compatible unseen stencil")
    return candidates[int(rng.integers(len(candidates)))]


def _pack(rng, shape, sizes):
    h, w = shape
    boxes = {}
    for i in sorted(range(len(sizes)), key=lambda k: -sizes[k][0] * sizes[k][1]):
        sh, sw = sizes[i]
        options = [(y, y + sh, x, x + sw) for y in range(h - sh + 1)
                   for x in range(w - sw + 1)
                   if all(_gap((y, y + sh, x, x + sw), b) for b in boxes.values())]
        if not options:
            return None
        if rng.random() < .65:
            edges = [b for b in options if b[0] == 0 or b[2] == 0 or b[1] == h or b[3] == w]
            options = edges or options
        boxes[i] = options[int(rng.integers(len(options)))]
    return [boxes[i] for i in range(len(sizes))]


def _layout(rng, radius, count, matched=False):
    minimum = 2 * radius + 2
    for attempt in range(240):
        if count == 2 and radius == 3:
            h, w = int(rng.integers(29, 31)), int(rng.integers(20, 31))
            sizes = [(minimum, int(rng.integers(minimum, min(14, w-12) + 1))) for _ in range(count)]
        else:
            low = 26 if count >= 4 else 23 if count == 3 else 21 if count == 2 else 17
            h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
            cap = 4 if count >= 2 else 10
            sizes = [(int(rng.integers(minimum, minimum+cap+1)),
                      int(rng.integers(minimum, minimum+cap+1))) for _ in range(count)]
        if attempt > 100:
            h, w = 30, 30
            sizes = [(minimum, minimum) for _ in range(count)]
        if matched:
            sizes = [sizes[0]] * count
        if rng.random() < .5:
            h, w = w, h
            sizes = [(b, a) for a, b in sizes]
        envelopes = [(a + 2 * radius, b + 2 * radius) for a, b in sizes] + [(5, 5)]
        packed = _pack(rng, (h, w), envelopes)
        if packed is not None:
            frames = [(t+radius, b-radius, l+radius, r-radius) for t, b, l, r in packed[:-1]]
            return (h, w), frames, packed[-1]
    raise RuntimeError("Bounded free packing exhausted")


def _subset(rng, values, n):
    order = rng.permutation(len(values))[:n]
    return [values[int(i)] for i in order]


def _failure_marks(rng, query, extras, capacity, missing=None):
    if missing is None:
        n = int(rng.integers(3))
        marks = _subset(rng, query, n)
    else:
        marks = [q for q in query if q != missing]
    marks += _subset(rng, extras, int(rng.integers(min(2, len(extras)) + 1)))
    marks = marks[:capacity]
    if marks:
        target = int(rng.integers(len(marks), capacity + 1))
        preferred = [q for q in marks if q in query] or marks
        marks += [int(rng.choice(preferred)) for _ in range(target - len(marks))]
    return marks


def _make_pair(rng, package, stencil, roles, sparse=False):
    radius = max(max(p) for p in stencil)
    count = 2 if package in ("identity", "scope") else int(rng.integers(1, 5 if radius == 2 else 3))
    shape, boxes, selector = _layout(rng, radius, count, package == "identity")
    colors = list(map(int, rng.permutation(10))) if roles is None else list(roles)
    bg, frame, filler, ink = colors[:4]
    label_count = int(rng.integers(4, 7))
    labels = colors[4:4+label_count]
    query = _subset(rng, labels, 3)
    extras = [x for x in labels if x not in query]
    a = np.full(shape, bg, dtype=np.int8)
    for box in [selector] + boxes:
        for p in _outline(box):
            a[p] = frame
    t, b, l, r = selector
    for n, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        a[t+1+int(n)//3, l+1+int(n)%3] = color
    active, marks, retained = [], [], []
    order = list(rng.permutation(count))
    neg_id, pos_id = (int(order[0]), int(order[1])) if count >= 2 else (0, 0)
    for i, box in enumerate(boxes):
        t, b, l, r = box
        a[t+1:b-1, l+1:r-1] = filler
        local = {}
        if package == "identity" and i == neg_id:
            chosen = list(range(4))
            for q in range(3):
                local[q] = [x for x in query if x != query[q]] + [extras[0]]
            local[3] = []
        elif package == "scope" and i == neg_id:
            first = int(rng.integers(4))
            chosen = [first, first ^ 1]
            for j, q in enumerate(chosen):
                local[q] = [x for x in query if x != query[j]] + [extras[0], query[(j+1) % 3]]
            local[first ^ 2] = list(query)  # an unmarked passing corner stays inactive
        elif package in ("identity", "scope") and i == pos_id:
            chosen = _subset(rng, list(range(4)), 2)
            passing = chosen[0]
            local[passing] = list(query) if package == "identity" else list(query) + [extras[0], query[0]]
            local[chosen[1]] = [query[0], query[0]]
        else:
            chosen = _subset(rng, list(range(4)), int(rng.integers(5)))
            if i == 0 and len(chosen) < 2:
                chosen = _subset(rng, list(range(4)), 2)
            for q in range(4):
                if rng.random() < .42:
                    m = list(query) + _subset(rng, extras, int(rng.integers(min(2, len(extras)) + 1)))
                    m = m[:len(stencil)]
                    if len(m) < len(stencil) and rng.random() < .5:
                        m += [query[0]]
                    local[q] = m
                else:
                    local[q] = _failure_marks(rng, query, extras, len(stencil))
        assigned = {}
        for q in range(4):
            coords = sorted(stencil)
            rng.shuffle(coords)
            values = local.get(q, [])
            assigned[q] = {}
            y, x = _corners(box)[q]
            dy, dx = DIRS[q]
            for uv, color in zip(coords, values):
                a[y+dy*uv[0], x+dx*uv[1]] = color
                assigned[q][uv] = color
        if package in ("identity", "scope") and i == neg_id:
            # Put a missing query color outside the actual aperture. A square-
            # window or whole-record test now makes the wrong deletion.
            q = chosen[0]
            used = {int(a[p]) for p in _positions(box, q, stencil)}
            missing = next(x for x in query if x not in used)
            holes = [(u, v) for u in range(1, radius+1) for v in range(1, radius+1)
                     if (u, v) not in stencil]
            uv = holes[int(rng.integers(len(holes)))]
            y, x = _corners(box)[q]
            dy, dx = DIRS[q]
            a[y+dy*uv[0], x+dx*uv[1]] = missing
        active.append(chosen)
        marks.append(assigned)
        retained.append({})
    occurrences = [(i, q) for i, qs in enumerate(active) for q in qs]
    if package == "identity":
        reference = (neg_id, 0)
    elif package == "scope":
        reference = (neg_id, active[neg_id][0])
    else:
        reference = occurrences[int(rng.integers(len(occurrences)))]
    for i, q in occurrences:
        if (i, q) == reference or (package == "free" and rng.random() < .20):
            keep = set(stencil)
        else:
            n = int(rng.integers(1, min(2, len(stencil)-1)+1)) if sparse else int(rng.integers(1, len(stencil)))
            keep = set(_subset(rng, sorted(stencil), n))
        if package in ("identity", "scope") and i == pos_id and q == active[i][0]:
            # The retained ornament alone samples only one query identity.
            uv = next(p for p, c in marks[i][q].items() if c == query[0])
            keep = {uv}
        retained[i][q] = keep
        for p in _positions(boxes[i], q, keep, True):
            a[p] = ink
    # Construction truth: read fields at the known geometric addresses; do
    # not call solve or _read to create a target.
    out = a.copy()
    for i, box in enumerate(boxes):
        positive = any(set(query) <= {int(a[p]) for p in _positions(box, q, stencil)} for q in active[i])
        if positive:
            t, b, l, r = box
            out[t:b, l:r] = bg
        for q in active[i]:
            for p in _positions(box, q, stencil, True):
                out[p] = bg if positive else ink
    return {"input": a, "output": out}


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.permutation(10))) if rng.random() < .28 else None
    puzzle = {"train": [], "test": []}
    seen, train_stencils = set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            package = ("identity" if index == 0 else "scope" if index == 1 else "free") if split == "train" else "free"
            radius = 2 if package == "identity" else 3 if package == "scope" or split == "test" else int(rng.integers(2, 4))
            stencil = _sample_stencil(rng, radius, 5 if package == "scope" else 3,
                                      package == "identity", train_stencils if split == "test" else ())
            for attempt in range(80):
                pair = _make_pair(rng, package, stencil, roles, split == "test")
                try:
                    info = _read(pair["input"])
                    assert np.array_equal(_answer(info), pair["output"])
                    assert _key(pair["input"]) not in seen
                    break
                except (ValueError, AssertionError):
                    continue
            else:
                raise RuntimeError("Bounded field/role sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                train_stencils.add(stencil)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(infos):
    records = [(info, record) for info in infos for record in info["records"]]
    positives = [(i, r) for i, r in records if r["positive"]]
    negatives = [(i, r) for i, r in records if not r["positive"]]
    assert positives and negatives, "Training must teach deletion and preservation"
    assert any(i["query"] <= r["labels"] for i, r in negatives), "No whole-record membership counterexample"
    assert any(i["query"] <= set().union(*(g["labels"] for g in r["gates"] if g["marked"]))
               for i, r in negatives), "No corner-pooling counterexample"
    assert any(all(any(g["marked"] and g["labels"] & i["query"] == i["query"]-{q}
                          for g in r["gates"]) for q in i["query"]) for i, r in negatives)
    assert any(any(not g["marked"] and i["query"] <= g["labels"] for g in r["gates"])
               for i, r in negatives), "No unmarked-corner counterexample"
    assert any(any(g["passes"] and g["labels"] == i["query"] for g in r["gates"])
               for i, r in positives), "No exact-set positive"
    assert any(any(g["passes"] and g["labels"]-i["query"]
                          and len([p for p in g["probe"] if i["array"][p] in i["query"]]) > 3
                          for g in r["gates"]) for i, r in positives), "No extra/repeated-label positive"
    assert any(any(g["passes"] and not i["query"] <= {
        int(i["array"][p]) for p in _positions(r["box"], g["q"], g["clue"])}
                   for g in r["gates"]) for i, r in positives), "No completion-dependent deletion"
    assert any(any(g["passes"] for g in r["gates"]) and any(g["marked"] and not g["passes"] for g in r["gates"])
               for _, r in positives), "No any-corner versus all-corners witness"
    assert any(i["stencil"] != frozenset((v, u) for u, v in i["stencil"])
               and any(g["marked"] and g["clue"] != i["stencil"] and g["q"] in (1, 2)
                       for g in r["gates"]) for i, r in negatives), "No surviving reflected asymmetric partial"
    assert any(any(g["marked"] and i["query"] <= {int(i["array"][p]) for p in
                 _positions(r["box"], g["q"], {(u, v) for u in range(1, i["radius"]+1)
                                               for v in range(1, i["radius"]+1)})}
                 for g in r["gates"]) for i, r in negatives), "No square-window counterexample"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos = {"train": [], "test": []}
    seen = set()
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            assert np.array_equal(_answer(info), out), "Incorrect filtered/completed output"
            t, b, l, r = info["selector"]
            assert np.array_equal(inp[t:b, l:r], out[t:b, l:r])
            for record in info["records"]:
                t, b, l, r = record["box"]
                if record["positive"]:
                    assert np.all(out[t:b, l:r] == info["background"])
                else:
                    assert np.array_equal(inp[t:b, l:r], out[t:b, l:r])
            infos[split].append(info)
    _coverage(infos["train"])
    trained = {i["stencil"] for i in infos["train"]}
    assert all(i["stencil"] not in trained and i["radius"] == 3 for i in infos["test"])
    return None


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _counterfactuals(puzzle):
    """Change only the query, then only the aperture; both can flip deletion."""
    for pair in puzzle["train"]:
        info = _read(pair["input"])
        if info["radius"] != 2 or len(info["stencil"]) != 3:
            continue
        options = [r for r in info["records"] if not r["positive"]
                   and all(g["marked"] for g in r["gates"])
                   and r["gates"][0]["clue"] == info["stencil"]
                   and len(r["gates"][0]["labels"]) == 3
                   and len(r["gates"][0]["labels"] & info["query"]) == 2]
        if not options:
            continue
        record = options[0]
        gate = record["gates"][0]
        # Parent B necessity: selecting the hard negative's identities deletes
        # it, without changing any field, frame, or ornament.
        changed = pair["input"].copy()
        t, b, l, r = info["selector"]
        query_points = [tuple(map(int, p)) for p in np.argwhere(changed[t+1:b-1, l+1:r-1] != info["background"])]
        for (y, x), color in zip(query_points, sorted(gate["labels"])):
            changed[t+1+y, l+1+x] = color
        answer = solve(changed)
        rt, rb, rl, rr = record["box"]
        assert np.all(answer[rt:rb, rl:rr] == info["background"])
        # Parent A necessity: swap the extra-label address for the decoy's
        # missing-query address. Only exterior ornaments change; fields and
        # selector stay byte-identical, but the formerly negative record passes.
        box = record["box"]
        cy, cx = _corners(box)[0]
        removed = next(uv for uv in sorted(info["stencil"])
                       if int(info["array"][cy+uv[0], cx+uv[1]]) not in info["query"])
        hole = next(uv for uv in ((1, 1), (1, 2), (2, 1), (2, 2)) if uv not in info["stencil"])
        replacement = (set(info["stencil"]) - {removed}) | {hole}
        changed = pair["input"].copy()
        for obj in info["records"]:
            for g in obj["gates"]:
                if not g["marked"]:
                    continue
                for p in g["outside"]:
                    changed[p] = info["background"]
                keep = replacement if g["clue"] == info["stencil"] else set(g["clue"]) & replacement
                keep = keep or {min(replacement)}
                for p in _positions(obj["box"], g["q"], keep, True):
                    changed[p] = info["ink"]
        assert np.array_equal(changed[t:b, l:r], info["array"][t:b, l:r])
        for obj in info["records"]:
            ot, ob, ol, oright = obj["box"]
            assert np.array_equal(changed[ot:ob, ol:oright], info["array"][ot:ob, ol:oright])
        answer = solve(changed)
        assert np.all(answer[rt:rb, rl:rr] == info["background"])
        return
    raise AssertionError("No identity teaching scene for counterfactual audit")


def self_test():
    """Bounded 42-seed diversity audit plus family-specific counterfactuals."""
    signatures, shapes, stencils, layouts, roles, counts, frame_shapes = (set() for _ in range(7))
    ink_cells_changed = delete_cells_changed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed < 2:
            assert _signature(generate_puzzle(seed)) == _signature(puzzle)
        validate_generated_puzzle(puzzle)
        _counterfactuals(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                original = a.copy()
                assert np.array_equal(solve(a), out) and np.array_equal(a, original)
                info = _read(a)
                shapes.add(a.shape)
                stencils.add(info["stencil"])
                roles.add(tuple(info[k] for k in ("background", "frame", "filler", "ink")))
                counts.add(len(info["records"]))
                layouts.add((a.shape, info["selector"], tuple(r["box"] for r in info["records"])))
                for record in info["records"]:
                    t, b, l, r = record["box"]
                    frame_shapes.add((b-t, r-l))
                    if record["positive"]:
                        delete_cells_changed += int(np.count_nonzero(a[t:b, l:r] != out[t:b, l:r]))
                    else:
                        ink_cells_changed += sum(len(g["outside"])-len(g["clue"]) for g in record["gates"] if g["marked"])
                # Color permutation preserves the semantics, including black.
                if seed < 3:
                    permutation = np.random.default_rng(seed+1000).permutation(10)
                    assert np.array_equal(solve(permutation[a]), permutation[out])
                # Relocation of selector identities changes no record decision.
                if seed < 3:
                    t, b, l, r = info["selector"]
                    moved = a.copy()
                    moved[t+1:b-1, l+1:r-1] = np.roll(moved[t+1:b-1, l+1:r-1], 1, axis=0)
                    expected = out.copy()
                    expected[t:b, l:r] = moved[t:b, l:r]
                    assert np.array_equal(solve(moved), expected)
                if seed < 3:
                    for axis in (0, 1):
                        assert np.array_equal(solve(np.flip(a, axis)), np.flip(out, axis))
                    assert np.array_equal(solve(np.rot90(a)), np.rot90(out))
    assert len(signatures) == 42 and len(shapes) >= 30 and len(stencils) >= 60
    assert len(layouts) >= 190 and len(roles) >= 70 and counts == {1, 2, 3, 4}
    assert len(frame_shapes) >= 30 and ink_cells_changed > 0 and delete_cells_changed > 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(431, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # An unrelated output mutation must be rejected, not just structural errors.
    bad = generate_puzzle(0, 2, 1)
    info = _read(bad["train"][0]["input"])
    t, _, l, _ = info["selector"]
    bad["train"][0]["output"][t, l] = (info["frame"]+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Unrelated output mutation accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "stencils": len(stencils), "layouts": len(layouts), "role_assignments": len(roles),
            "record_counts": sorted(counts), "frame_shapes": len(frame_shapes)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
