"""Forward echoes: first later compatible donor on one folded thread.

Only NumPy and the standard library are used. No parent generator is imported.
"""
from __future__ import annotations

import numpy as np

_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_WITNESS_ORDER = ((0, 1), (1, 2), (1, 3), (2, 4), (4, 0), (1, 0))
_WITNESS_VALUES = ((0, 1), (0, 1), (1, 0), (0, 0), (2, 3), (3, 3), (1, 2))
_FALLBACK = ((1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (1, 3),
             (2, 3), (2, 2), (2, 1), (3, 1), (4, 1), (4, 2), (4, 3))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a


def _components(cells):
    todo, parts = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _support(center, kind):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if kind == "square" or dr == 0 or dc == 0}


def _trace(cells):
    graph = {p: [q for dr, dc in _D4
                 if (q := (p[0] + dr, p[1] + dc)) in cells] for p in cells}
    ends = sorted(p for p in graph if len(graph[p]) == 1)
    if len(ends) != 2 or any(len(v) not in (1, 2) for v in graph.values()):
        raise ValueError("The reconstructed thread must be an open simple chain")
    if ends[0][0] == ends[1][0]:
        raise ValueError("Thread endpoints must have different rows")
    start = min(ends, key=lambda p: p[0])
    previous, current, ordered, visited = None, start, [], set()
    while True:
        if current in visited:
            raise ValueError("Cyclic thread")
        ordered.append(current)
        visited.add(current)
        nxt = [p for p in graph[current] if p != previous]
        if not nxt:
            break
        previous, current = current, nxt[0]
    if len(visited) != len(cells):
        raise ValueError("Disconnected thread")
    return ordered


def _parse_roles(grid, bg, line):
    tokens = []
    remaining = set(map(tuple, np.argwhere((grid != bg) & (grid != line))))
    for cells in _components(remaining):
        if len(cells) not in (5, 9):
            raise ValueError("Beads must be complete crosses or 3x3 squares")
        rs, cs = zip(*cells)
        center = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
        kind = "cross" if len(cells) == 5 else "square"
        if _support(center, kind) != cells:
            raise ValueError("Malformed bead support")
        body = {int(grid[p]) for p in cells if p != center}
        if len(body) != 1:
            raise ValueError("Bead bodies must be uniform")
        tokens.append(dict(center=center, kind=kind, body=body.pop(),
                           value=int(grid[center]), cells=cells))
    if not 3 <= len(tokens) <= 10:
        raise ValueError("Expected 3..10 separated beads")
    neutral = set(map(tuple, np.argwhere(grid == line)))
    if len(neutral) < 2:
        raise ValueError("Missing neutral thread")
    skeleton = set(neutral)
    for t in tokens:
        r, c = t["center"]
        ports = [(dr, dc) for dr, dc in _D4 if (r + 2*dr, c + 2*dc) in neutral]
        if len(ports) not in (0, 2):
            raise ValueError("A bead must have two thread ports or no ports")
        t["on_thread"] = len(ports) == 2
        if ports:
            skeleton.add((r, c))
            skeleton.update((r + dr, c + dc) for dr, dc in ports)
        elif any((r + dr, c + dc) in neutral for r, c in t["cells"] for dr, dc in _D4):
            raise ValueError("Off-thread beads must be isolated from the thread")
    path = _trace(skeleton)
    if path[0] not in neutral or path[-1] not in neutral:
        raise ValueError("Endpoints must be bare thread cells")
    order = {p: i for i, p in enumerate(path)}
    beads = sorted((t for t in tokens if t["on_thread"]), key=lambda t: order[t["center"]])
    if not 3 <= len(beads) <= 8:
        raise ValueError("Expected 3..8 beads on the thread")
    return dict(background=bg, line=line, tokens=tokens, beads=beads, path=path)


def _interpret(value):
    grid = _grid(value)
    hist = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(hist.argmax())
    if hist[bg] * 2 <= grid.size:
        raise ValueError("The background must occupy a strict majority")
    candidates = []
    for line in np.flatnonzero(hist):
        if line == bg:
            continue
        try:
            candidates.append(_parse_roles(grid, bg, int(line)))
        except (ValueError, KeyError):
            continue
    if len(candidates) != 1:
        raise ValueError("Thread and bead interpretation must be unique")
    return candidates[0]


def _compose(beads, policy="first", recursive=False):
    """Return surviving (original bead, new center) pairs, in bead order."""
    chosen = {}
    for i, bead in enumerate(beads):
        donors = [j for j in range(i + 1, len(beads)) if beads[j]["body"] == bead["value"]]
        if donors:
            if policy == "last":
                j = donors[-1]
            elif policy == "nearest":
                r, c = bead["center"]
                j = min(donors, key=lambda j: ((beads[j]["center"][0]-r)**2
                                              + (beads[j]["center"][1]-c)**2, j))
            else:
                j = donors[0]
            chosen[i] = j
    values = {}
    for i in reversed(range(len(beads))):
        if i in chosen:
            j = chosen[i]
            values[i] = values.get(j, beads[j]["value"]) if recursive else beads[j]["value"]
    return [(beads[i], values[i]) for i in sorted(values)]


def _render(composed, background):
    if not composed:
        # Useful for diagnostics; legal generated inputs always have a survivor.
        return np.full((1, 3), background, dtype=np.int8)
    out = np.full((4*len(composed)-1, 3), background, dtype=np.int8)
    for i, (t, value) in enumerate(composed):
        for r, c in _support((4*i+1, 1), t["kind"]):
            out[r, c] = t["body"]
        out[4*i+1, 1] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, trace topology and perform original-value forward lookups."""
    info = _interpret(input_grid)
    return _render(_compose(info["beads"]), info["background"])


def _features(info):
    beads, bg = info["beads"], info["background"]
    composed = _compose(beads)
    actual = _render(composed, bg)
    differs = lambda seq, **kw: not np.array_equal(actual, _render(_compose(seq, **kw), bg))
    results = {id(t): v for t, v in composed}
    donors = []
    earlier_only = conflicting = shared = 0
    for i, t in enumerate(beads):
        later = [j for j in range(i+1, len(beads)) if beads[j]["body"] == t["value"]]
        earlier_only += not later and any(b["body"] == t["value"] for b in beads[:i])
        conflicting += len({beads[j]["value"] for j in later}) > 1
        if later:
            donors.append(later[0])
    shared = len(donors) - len(set(donors))
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(info["path"], info["path"][1:])]
    return dict(
        beads=len(beads), off_thread=len(info["tokens"])-len(beads), survivors=len(composed),
        conflicting=conflicting, earlier_only=earlier_only, shared=shared,
        last_failure=differs(beads, policy="last"), nearest_failure=differs(beads, policy="nearest"),
        reverse_failure=differs(beads[::-1]),
        spatial_failure=differs(sorted(beads, key=lambda t: t["center"])),
        recursive_failure=differs(beads, recursive=True),
        raw_failure=not np.array_equal(actual, _render([(t, t["value"]) for t in beads], bg)),
        solid=sum(t["body"] != t["value"] and v == t["body"] for t, v in composed),
        monochrome_donor=sum(beads[j]["body"] == beads[j]["value"] for j in donors),
        discarded_donor=sum(id(beads[j]) not in results for j in donors),
        unchanged=sum(t["value"] == v for t, v in composed),
        background=bg, line=info["line"], kinds=tuple(sorted({t["kind"] for t in beads})),
        length=len(info["path"]), turns=sum(a != b for a, b in zip(steps, steps[1:])),
        upward=(-1, 0) in steps,
        reversal=(0, 1) in steps and (0, -1) in steps,
        above=min(p[0] for p in info["path"]) < info["path"][0][0],
        layout=tuple((t["center"][0]-beads[0]["center"][0],
                      t["center"][1]-beads[0]["center"][1]) for t in beads),
    )


def _walk(rng, length):
    """Self-avoiding coarse walk; subdivision keeps parallel edges separated."""
    for _ in range(80):
        path = [(int(rng.integers(5)), int(rng.integers(5)))]
        used = set(path)
        previous = None
        for _ in range(length-1):
            r, c = path[-1]
            options = [(r+dr, c+dc) for dr, dc in _D4
                       if 0 <= r+dr < 5 and 0 <= c+dc < 5 and (r+dr, c+dc) not in used]
            if not options:
                break
            weights = [2.5 if (q[0]-r, q[1]-c) == previous else 1.0 for q in options]
            weights = np.asarray(weights)
            q = options[int(rng.choice(len(options), p=weights/weights.sum()))]
            previous = q[0]-r, q[1]-c
            path.append(q)
            used.add(q)
        if len(path) == length:
            return path
    return list(_FALLBACK)


def _subdivide(rng, nodes, gap_low, gap_high):
    rows = np.r_[0, np.cumsum(rng.integers(gap_low, gap_high+1, 4))]
    cols = np.r_[0, np.cumsum(rng.integers(gap_low, gap_high+1, 4))]
    points = [(int(rows[r]), int(cols[c])) for r, c in nodes]
    d0 = (int(np.sign(points[0][0]-points[1][0])), int(np.sign(points[0][1]-points[1][1])))
    d1 = (int(np.sign(points[-1][0]-points[-2][0])), int(np.sign(points[-1][1]-points[-2][1])))
    start = points[0][0]+2*d0[0], points[0][1]+2*d0[1]
    finish = points[-1][0]+2*d1[0], points[-1][1]+2*d1[1]
    backbone = []
    for a, b in zip([start]+points, points+[finish]):
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        distance = abs(b[0]-a[0])+abs(b[1]-a[1])
        backbone.extend((a[0]+k*dr, a[1]+k*dc) for k in range(distance))
    backbone.append(finish)
    ordered = _trace(set(backbone))
    if ordered[0] != start:
        points.reverse()
    return ordered, points


def _layout(rng, relations, kinds, roles, off_count, gaps, special, fallback=False):
    n = len(relations)
    if fallback:
        nodes = list(_FALLBACK)
    else:
        length = int(rng.integers(max(n, 11) if special else n, min(17, n+7)+1))
        nodes = _walk(rng, length)
    path, nodes = _subdivide(rng, nodes, *gaps)
    if fallback and special:
        indices = [0, 6, 8, 9, 11, 12]
    else:
        indices = sorted(map(int, rng.choice(len(nodes), n, replace=False)))
    centers = [nodes[i] for i in indices]
    supports = [_support(p, k) for p, k in zip(centers, kinds)]
    occupied = set().union(*supports)
    all_cells = set(path) | occupied
    r0, c0 = min(r for r, c in all_cells), min(c for r, c in all_cells)
    bh = max(r for r, c in all_cells)-r0+1
    bw = max(c for r, c in all_cells)-c0+1
    if max(bh, bw) > 30:
        raise ValueError("Layout exceeds the canvas")
    h = int(rng.integers(max(bh, 12 if off_count else 6), 31))
    w = int(rng.integers(max(bw, 12 if off_count else 6), 31))
    if fallback:
        h = w = 30
    top, left = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
    shift = lambda p: (p[0]-r0+top, p[1]-c0+left)
    path = [shift(p) for p in path]
    centers = [shift(p) for p in centers]
    supports = [_support(p, k) for p, k in zip(centers, kinds)]
    occupied = set().union(*supports)
    neutral = set(path)-occupied
    bg, line, palette = roles
    grid = np.full((h, w), bg, dtype=np.int8)
    for p in neutral:
        grid[p] = line
    beads = []
    for p, cells, kind, (body, value) in zip(centers, supports, kinds, relations):
        for q in cells:
            grid[q] = palette[body]
        grid[p] = palette[value]
        beads.append(dict(center=p, kind=kind, body=palette[body], value=palette[value]))
    for _ in range(off_count):
        kind = str(rng.choice(sorted(set(kinds))))
        blocked = occupied | neutral
        forbidden = blocked | {(r+dr, c+dc) for r, c in blocked for dr, dc in _D4}
        options = []
        for r in range(1, h-1):
            for c in range(1, w-1):
                cells = _support((r, c), kind)
                if cells.isdisjoint(forbidden):
                    options.append(((r, c), cells))
        if not options:
            raise ValueError("No isolated space for an off-thread bead")
        p, cells = options[int(rng.integers(len(options)))]
        # A conflicting color-compatible outsider; scope, rather than equality,
        # decides whether it is consulted. Its identity is visible in this grid.
        body, value = (1, 4) if special else tuple(map(int, rng.integers(0, 5, 2)))
        for q in cells:
            grid[q] = palette[body]
        grid[p] = palette[value]
        occupied.update(cells)
    if np.count_nonzero(grid == bg)*2 <= grid.size:
        raise ValueError("Insufficient background")
    output = _render(_compose(beads), bg)
    return dict(input=grid, output=output)


def _random_relations(rng, n, dense):
    k = int(rng.integers(2 if dense else 3, 5 if dense else 9))
    relations = [tuple(map(int, row)) for row in rng.integers(0, k, (n, 2))]
    # At least one success, with no recursion or hidden generation convention.
    i = int(rng.integers(n-1))
    j = int(rng.integers(i+1, n))
    relations[j] = (relations[i][1], relations[j][1])
    return relations


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, v, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(("more_beads", "longer_gaps", "denser_relations")))
    morphology = str(rng.choice(("square", "cross", "mixed")))
    varied_roles = bool(rng.random() < .55)
    stable_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    witness_modes = ["order", "values"] + ["random"]*(num_train-2)
    rng.shuffle(witness_modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = witness_modes[index] if split == "train" else "random"
            if witness == "order":
                relations = list(_WITNESS_ORDER)
            elif witness == "values":
                relations = list(_WITNESS_VALUES)
            else:
                n = 8 if split == "test" and mode == "more_beads" else int(rng.integers(3, 8 if split == "train" else 9))
                relations = _random_relations(rng, n, split == "test" and mode == "denser_relations")
            n = len(relations)
            kinds = ([morphology]*n if morphology != "mixed" else
                     ["square" if i % 2 == 0 else "cross" for i in range(n)])
            if morphology == "mixed":
                rng.shuffle(kinds)
            if varied_roles:
                bg, line = tuple(map(int, rng.choice(10, 2, replace=False)))
                if split == "train" and index == 1:
                    first = _interpret(puzzle["train"][0]["input"])
                    if (bg, line) == (first["background"], first["line"]):
                        options = [(a, b) for a in range(10) for b in range(10)
                                   if a != b and (a, b) != (bg, line)]
                        bg, line = options[int(rng.integers(len(options)))]
            else:
                bg, line = stable_roles
            palette = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, line)])))
            roles = bg, line, palette
            gaps = ((5, 6) if split == "test" else (4, 4)) if mode == "longer_gaps" else (4, 6)
            off_count = 1 if witness == "order" else int(rng.integers(0, 3))
            for attempt in range(121):
                try:
                    pair = _layout(rng, relations, kinds, roles, off_count, gaps,
                                   witness == "order", fallback=attempt == 120)
                    info = _interpret(pair["input"])
                    if not np.array_equal(solve(pair["input"]), pair["output"]):
                        continue
                    f = _features(info)
                    if witness == "order" and not all(f[k] for k in (
                            "spatial_failure", "nearest_failure", "reverse_failure", "recursive_failure", "above")):
                        continue
                    if f["survivors"] == 0 or _key(pair["input"]) in seen:
                        continue
                    break
                except (ValueError, KeyError):
                    continue
            else:
                raise RuntimeError("Bounded construction could not produce a valid unique example")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            info = _interpret(inp)
            f = _features(info)
            assert 1 <= f["survivors"] <= 7
            assert np.array_equal(out, solve(inp)), "Forward lookup or bead packing mismatch"
            assert out.shape == (4*f["survivors"]-1, 3)
            assert info["line"] not in out, "Neutral connectors must not be emitted"
            assert np.all(out[3::4] == info["background"]), "One blank separator row is required"
            features[split].append(f)
    train = features["train"]
    for k in ("conflicting", "earlier_only", "shared", "last_failure", "nearest_failure",
              "reverse_failure", "spatial_failure", "recursive_failure", "raw_failure",
              "solid", "monochrome_donor", "discarded_donor", "unchanged", "off_thread"):
        assert any(f[k] for f in train), f"Missing training evidence: {k}"
    assert any(f["above"] and f["reverse_failure"] for f in train), "Higher endpoint must be taught"
    train_roles = {(f["background"], f["line"]) for f in train}
    if len(train_roles) == 1:
        assert {(f["background"], f["line"]) for f in features["test"]} <= train_roles
    test_kinds = set().union(*(set(f["kinds"]) for f in features["test"]))
    assert test_kinds <= set().union(*(set(f["kinds"]) for f in train))


def self_test():
    """Bounded semantic, counterfactual, count, determinism and diversity checks."""
    puzzles, records, signatures = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        signatures.add(tuple(tuple((_key(q["input"]), _key(q["output"])) for q in p[s])
                             for s in ("train", "test")))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _interpret(pair["input"])
                f = _features(info)
                f["shape"] = pair["input"].shape
                records.append(f)
                copied = pair["input"].copy()
                assert np.array_equal(solve(copied), pair["output"])
                assert np.array_equal(copied, pair["input"]), "solve mutated its input"
    assert len(signatures) == 42
    for field, minimum in (("shape", 40), ("layout", 100), ("length", 35), ("turns", 8),
                           ("beads", 5), ("survivors", 5), ("background", 10), ("line", 10)):
        assert len({f[field] for f in records}) >= minimum, (field, minimum)
    for field in ("upward", "reversal", "above", "nearest_failure", "recursive_failure", "solid"):
        assert sum(bool(f[field]) for f in records) >= 5, field
    assert {("square",), ("cross",), ("cross", "square")} <= {f["kinds"] for f in records}
    first = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], puzzles[0][s])
               for k in ("input", "output"))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: np.array(v, copy=True) for k, v in q.items()} for q in first[s]] for s in first}
    bad["train"][0]["output"][1, 1] = (int(bad["train"][0]["output"][1, 1])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted composition accepted")
    for cells in ({(0, 0), (0, 1), (0, 2)},
                  {(0, 1), (1, 0), (1, 1), (1, 2)},
                  {(0, 0), (0, 1), (1, 0), (1, 1)},
                  {(0, 0), (1, 0), (3, 3)}):
        try:
            _trace(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("Tied, branched, cyclic or disconnected thread accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({f["shape"] for f in records}),
            "bead_layouts": len({f["layout"] for f in records}),
            "thread_lengths": len({f["length"] for f in records}),
            "turn_counts": sorted({f["turns"] for f in records}),
            "bead_counts": sorted({f["beads"] for f in records}),
            "survivor_counts": sorted({f["survivors"] for f in records})}
