"""Relational sedimentation: a lookup can act as a geometric adhesive.

Only standard-library modules and NumPy are used.  No parent code or data is
loaded.  The solver discovers every color role and the settling direction from
the input.  See task_spec.md for the rule and deliberately bounded sampling.
"""

import heapq
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
          "cells must be integers")
    _need(np.all((a >= 0) & (a <= 9)), "bad grid colors")
    return a.astype(np.int8, copy=True)


def _parts(points):
    todo = set(points)
    answer = []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        answer.append(part)
    return answer


def _support(center, h, w, kind):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-(h // 2), h // 2 + 1)
            for dc in range(-(w // 2), w // 2 + 1)
            if kind == "rectangle" or dr == 0 or dc == 0}


def _box(token):
    r, c = token["center"]
    rh, cw = token["h"] // 2, token["w"] // 2
    return {(y, x) for y in range(r-rh, r+rh+1) for x in range(c-cw, c+cw+1)}


def _stubs(token):
    r, c = token["center"]
    return {(r-token["h"]//2-1, c), (r+token["h"]//2+1, c),
            (r, c-token["w"]//2-1), (r, c+token["w"]//2+1)}


def _color_parts(token, center_color, grouping="components"):
    colors = {p: token["body"] for p in token["cells"]}
    colors[token["center"]] = center_color
    if grouping == "before":
        # Diagnostic: segment before, rather than after, center replacement.
        old = {p: token["body"] for p in token["cells"]}
        old[token["center"]] = token["value"]
        return [(colors[min(part)], part) for color in sorted(set(old.values()))
                for part in _parts(p for p in old if old[p] == color)]
    result = []
    for color in sorted(set(colors.values())):
        cells = {p for p in colors if colors[p] == color}
        groups = _parts(cells) if grouping == "components" else [cells]
        result.extend((color, part) for part in groups)
    return result


def _decisions(tokens, selection="ends", lookup="one"):
    result = []
    for i, t in enumerate(tokens):
        donors = [u for j, u in enumerate(tokens) if j != i and u["body"] == t["value"]]
        keep = (t["eligible"] or selection == "all") and bool(donors)
        value = donors[0]["value"] if donors else t["value"]
        if lookup == "none":
            value = t["value"]
        elif lookup == "extra" and donors:
            further = [u for u in tokens if u["body"] == value]
            if further:
                value = further[0]["value"]
        result.append((keep, value, donors))
    return result


def _render(a, selection="ends", lookup="one", grouping="components", near=False):
    out = a["grid"].copy()
    bands = {b["route"]: b for b in a["bands"]}
    placements = []
    for t, (keep, value, _) in zip(a["tokens"], _decisions(a["tokens"], selection, lookup)):
        for p in t["cells"]:
            out[p] = a["background"]
        if not keep:
            continue
        for color, cells in _color_parts(t, value, grouping):
            b = bands[color]
            offset = b["top"] - min(r for r, c in cells) if near else b["bottom"] - max(r for r, c in cells)
            target = {(r+offset, c) for r, c in cells}
            _need(all(b["top"] <= r <= b["bottom"] and b["left"] < c < b["right"]
                      for r, c in target), "stamp crosses a destination boundary")
            placements.append((b["rail"], target))
    # Overlapping same-destination stamps are a union, not a physical stack.
    for color, cells in placements:
        for p in cells:
            out[p] = color
    return out, placements


def _parse_canonical(g, bg):
    h, w = g.shape
    rows = []
    for row in g:
        nz = np.flatnonzero(row != bg)
        info = None
        if len(nz) >= max(13, w-4):
            l, r = int(nz[0]), int(nz[-1])
            if (row[l] == row[r] and row[l] != row[l+1] and
                    np.all(row[l+1:r] == row[l+1])):
                info = (l, r, int(row[l+1]), int(row[l]))
        rows.append(info)
    found = [i for i, row in enumerate(rows) if row is not None]
    _need(found and found[0] >= 5, "no band bank")
    first = found[0]
    _need(np.all(g[first-1] == bg), "missing source/bank separator")
    bands, i = [], first
    while i < h:
        if rows[i] is None:
            _need(np.all(g[i] == bg), "unparsed cells below source")
            i += 1
            continue
        top, info = i, rows[i]
        while i+1 < h and rows[i+1] == info:
            i += 1
        l, r, route, rail = info
        bands.append(dict(top=top, bottom=i, left=l, right=r, route=route, rail=rail))
        i += 1
    _need(3 <= len(bands) <= 6, "need three to six distinct address bands")
    _need(all(y["top"] > x["bottom"]+1 for x, y in zip(bands, bands[1:])), "bands touch")
    routes = {b["route"] for b in bands}
    _need(len(routes) == len(bands) and bg not in routes, "ambiguous band addresses")
    source_colors = set(map(int, np.unique(g[:first]))) - {bg}
    line_candidates = source_colors - routes
    _need(len(line_candidates) == 1, "line role is not unique")
    line = line_candidates.pop()
    _need(all(line != b["rail"] for b in bands), "line cannot be a rail color")
    points = {tuple(map(int, p)) for p in np.argwhere((g[:first] != bg) & (g[:first] != line))}
    tokens = []
    for cells in _parts(points):
        rr, cc = zip(*cells)
        th, tw = max(rr)-min(rr)+1, max(cc)-min(cc)+1
        _need(th >= 3 and tw >= 3 and th % 2 == tw % 2 == 1, "not a centered odd stencil")
        center = ((min(rr)+max(rr))//2, (min(cc)+max(cc))//2)
        kinds = [k for k in ("rectangle", "cross") if _support(center, th, tw, k) == cells]
        _need(len(kinds) == 1, "unrecognized stencil support")
        bodies = {int(g[p]) for p in cells if p != center}
        _need(len(bodies) == 1, "nonuniform stencil body")
        body, value = bodies.pop(), int(g[center])
        _need(body in routes and value in routes, "stencil identity has no address band")
        tokens.append(dict(center=center, h=th, w=tw, kind=kinds[0], cells=cells,
                           body=body, value=value))
    _need(len(tokens) >= 3, "missing donor evidence")
    graph = {tuple(map(int, p)) for p in np.argwhere(g[:first] == line)}
    degrees = {p: sum((p[0]+dr, p[1]+dc) in graph for dr, dc in _D4) for p in graph}
    ends = {p for p in graph if degrees[p] == 1}
    _need(len(graph) >= 3 and len(_parts(graph)) == 1 and len(ends) == 2 and
          all(d in (1, 2) for d in degrees.values()), "line must be one open unbranched path")
    for t in tokens:
        t["eligible"] = bool(_stubs(t) & ends)
        _need(len(_stubs(t) & ends) <= 1, "two ends select the same token")
        # The line may touch a token's exterior, but never enter its box.
        _need(not _box(t) & graph, "line crosses a stencil bounding box")
    _need(all(sum(p in _stubs(t) for t in tokens) <= 1 for p in ends), "shared endpoint")
    mapping = {}
    for t in tokens:
        _need(t["body"] not in mapping or mapping[t["body"]] == t["value"], "conflicting donors")
        mapping[t["body"]] = t["value"]
    _need(any(t["eligible"] for t in tokens), "no selected stencil")
    return dict(grid=g, background=bg, line=line, bands=bands, tokens=tokens, graph=graph)


def _analyze(value):
    g = _grid(value)
    bg = int(g[0, 0])
    _need(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), "corners must reveal background")
    candidates = []
    for rotation in range(4):
        try:
            a = _parse_canonical(np.rot90(g, rotation).copy(), bg)
            out, places = _render(a)
        except AssertionError:
            continue
        a.update(rotation=rotation, output=out, placements=places)
        candidates.append(a)
    _need(len(candidates) == 1, "source/bank orientation is ambiguous")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read original donors, fuse decoded colors, then independently settle masks."""
    a = _analyze(input_grid)
    return np.rot90(a["output"], -a["rotation"]).copy()


def _route(rng, shape, blocked, start, end):
    h, w = shape
    if start in blocked or end in blocked or start == end:
        return None
    costs = rng.uniform(1., 1.4, size=shape)
    distance, previous, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p == end:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1] if len(path) >= 3 else None
        for dr, dc in _D4:
            q = (p[0]+dr, p[1]+dc)
            if not (1 <= q[0] < h-1 and 1 <= q[1] < w-1) or q in blocked:
                continue
            new = cost + float(costs[q])
            if new < distance.get(q, float("inf")):
                distance[q], previous[q] = new, p
                heapq.heappush(queue, (new, q))
    return None


def _relations(rng, mode):
    m = int(rng.integers(4 if mode == "fusion" else 3, 6))
    n = int(rng.integers(4, 8))
    mapping = {i: int(rng.integers(m)) for i in range(m)}
    if mode == "fusion":
        mapping.update({0: 1, 1: 0, 2: 3})
        # Index 0 fuses, index 1 fails: identity 3 has no physical donor.
        bodies = [0, 2, 1, 0]
        choices = [i for i in range(m) if i != 3]
        eligible = 2
    elif mode == "chain":
        mapping.update({0: 1, 1: 2, 2: 0 if m == 3 else 3})
        bodies, choices, eligible = [0, 1, 2, 0], list(range(m)), 2
    else:
        mapping[0] = 1
        bodies = [0, int(rng.integers(m)), 1, int(rng.integers(m))]
        choices, eligible = list(range(m)), int(rng.integers(1, 3))
    bodies += [int(rng.choice(choices)) for _ in range(n-4)]
    return m, [(b, mapping[b]) for b in bodies], eligible


def _geometry(rng, specs, stage, width, left, right, eligible, interior):
    """Random packing plus weighted path routing; no coordinate templates."""
    n = len(specs)
    omit = 3 if interior else -1
    for _ in range(22):
        tokens = [None]*n
        occupied = set()
        order = sorted((i for i in range(n) if i != omit), key=lambda i: -specs[i][0]*specs[i][1])
        for i in order:
            th, tw, kind, body, value = specs[i]
            rh, cw = th//2, tw//2
            centers = [(r, c) for r in range(1+rh, stage-1-rh)
                       for c in range(left+1+cw, right-cw)]
            if not centers:
                break
            for j in rng.permutation(len(centers)):
                center = centers[int(j)]
                t = dict(center=center, h=th, w=tw, kind=kind, body=body, value=value,
                         eligible=i < eligible, cells=_support(center, th, tw, kind))
                box = _box(t)
                if any((r+dr, c+dc) in occupied for r, c in box
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                    continue
                tokens[i] = t
                occupied.update(box)
                break
            if tokens[i] is None:
                break
        if any(tokens[i] is None for i in order):
            continue
        stub_lists = []
        for i in range(eligible):
            stubs = sorted(p for p in _stubs(tokens[i]) if p not in occupied and
                           1 <= p[0] < stage-1 and 1 <= p[1] < width-1)
            stub_lists.append(stubs)
        if any(not s for s in stub_lists):
            continue
        for _ in range(14):
            start = stub_lists[0][int(rng.integers(len(stub_lists[0])))]
            if eligible == 2:
                end = stub_lists[1][int(rng.integers(len(stub_lists[1])))]
            else:
                free = [(r, c) for r in range(1, stage-1) for c in range(1, width-1)
                        if (r, c) not in occupied and
                        all((r, c) not in _stubs(t) and
                            not any((r+dr, c+dc) in t["cells"] for dr, dc in _D4)
                            for t in tokens if t is not None)]
                if not free:
                    break
                end = free[int(rng.integers(len(free)))]
            path = _route(rng, (stage, width), occupied, start, end)
            if path is None:
                continue
            ends, line = {path[0], path[-1]}, set(path)
            if {i for i, t in enumerate(tokens) if t is not None and _stubs(t) & ends} != set(range(eligible)):
                continue
            if any(p not in _stubs(t) and any((p[0]+dr, p[1]+dc) in t["cells"] for dr, dc in _D4)
                   for p in ends for t in tokens if t is not None):
                continue
            if interior:
                th, tw, kind, body, value = specs[omit]
                options = []
                for r in range(1+th//2, stage-1-th//2):
                    for c in range(left+1+tw//2, right-tw//2):
                        t = dict(center=(r, c), h=th, w=tw, kind=kind, body=body, value=value,
                                 eligible=False, cells=_support((r, c), th, tw, kind))
                        box = _box(t)
                        if (not _stubs(t) & (line-ends) or _stubs(t) & ends or box & line or
                                any((p[0]+dr, p[1]+dc) in t["cells"] for p in ends for dr, dc in _D4) or
                                any((y+dr, x+dc) in occupied for y, x in box
                                    for dr in (-1, 0, 1) for dc in (-1, 0, 1))):
                            continue
                        options.append(t)
                if not options:
                    continue
                tokens[omit] = options[int(rng.integers(len(options)))]
            return tokens, line
    return None


def _make_pair(rng, mode, is_test, turn):
    for _ in range(100):
        m, relations, eligible = _relations(rng, mode)
        specs = []
        for i, (body, value) in enumerate(relations):
            kind = str(rng.choice(["rectangle", "cross"]))
            th = 5 if rng.random() < .13 else 3
            tw = 5 if rng.random() < .3 else 3
            if mode == "fusion" and i == 0:
                kind = "cross"
            if mode == "chain" and i < 2:
                kind = "cross" if i == 0 else "rectangle"
            if is_test and i == 0:
                tw = int(rng.choice([7, 9]))
            if i >= eligible:
                th = 3
            specs.append((th, tw, kind, body, value))
        # Reserve enough depth even for a rigid body-color interpretation. This
        # makes the contrasting component-gravity hypothesis easy to exhibit.
        mini = []
        for th, tw, kind, body, value in specs:
            center = (th//2, tw//2)
            mini.append(dict(center=center, h=th, w=tw, kind=kind, body=body, value=value,
                             cells=_support(center, th, tw, kind), eligible=len(mini) < eligible))
        required = [1]*m
        for t, (keep, value, _) in zip(mini, _decisions(mini)):
            if keep:
                required[t["body"]] = max(required[t["body"]], t["h"])
                for color, part in _color_parts(t, value):
                    required[color] = max(required[color], max(r for r, c in part)-min(r for r, c in part)+1)
        cap = min(16, 30-sum(required)-m-1)
        if cap < 10:
            continue
        stage = int(rng.integers(10, cap+1))
        width = int(rng.integers(24 if is_test else 20, 31))
        insets = [(int(rng.integers(3)), int(rng.integers(3))) for _ in range(m)]
        left, right = max(x[0] for x in insets), width-1-max(x[1] for x in insets)
        layout = _geometry(rng, specs, stage, width, left, right, eligible, mode == "chain")
        if layout is None:
            continue
        tokens, line_cells = layout
        depths, gaps, margin = required[:], [1]*m, 1
        budget = 30-stage-sum(depths)-sum(gaps)-margin
        for _ in range(int(rng.integers(budget+1))):
            choice = int(rng.integers(2*m+1))
            if choice < m:
                depths[choice] += 1
            elif choice < 2*m:
                gaps[choice-m] += 1
            else:
                margin += 1
        palette = list(map(int, rng.permutation(10)))
        bg, line = palette[:2]
        identities, extras = palette[2:2+m], palette[2+m:]
        common = int(rng.choice(extras))
        rails = [int(rng.choice([x for x in identities if x != identities[i]]))
                 if rng.random() < .35 else common if rng.random() < .6 else int(rng.choice(extras))
                 for i in range(m)]
        grid = np.full((stage+sum(depths)+sum(gaps)+margin, width), bg, dtype=np.int8)
        bands, cursor = [], stage
        order = list(map(int, rng.permutation(m)))
        for identity in order:
            cursor += gaps[identity]
            l, inset_r = insets[identity]
            r = width-1-inset_r
            b = dict(top=cursor, bottom=cursor+depths[identity]-1, left=l, right=r,
                     route=identities[identity], rail=rails[identity])
            grid[b["top"]:b["bottom"]+1, l:r+1] = b["route"]
            grid[b["top"]:b["bottom"]+1, [l, r]] = b["rail"]
            bands.append(b)
            cursor += depths[identity]
        for p in line_cells:
            grid[p] = line
        for t in tokens:
            t["body"], t["value"] = identities[t["body"]], identities[t["value"]]
            for p in t["cells"]:
                grid[p] = t["body"]
            grid[t["center"]] = t["value"]
        # Label from sampled geometry/relations, then separately rediscover it.
        a = dict(grid=grid, background=bg, bands=bands, tokens=tokens)
        output, _ = _render(a)
        pair = dict(input=np.rot90(grid, turn).copy(), output=np.rot90(output, turn).copy())
        try:
            parsed = _analyze(pair["input"])
            _need(np.array_equal(solve(pair["input"]), pair["output"]), "construction/parse disagreement")
            f = _features(parsed)
            _need(f["survivors"] >= 1, "no active stencil")
            if mode == "fusion":
                _need(f["fusion_witness"] and f["failed"] and f["selection_witness"], "fusion lesson lost")
            if mode == "chain":
                _need(f["gravity_witness"] and f["simultaneous_witness"] and f["interior_compatible"], "chain lesson lost")
        except AssertionError:
            continue
        return pair
    raise RuntimeError("bounded sampler could not fit a valid scene")


def _key(g):
    return g.shape, g.tobytes()


def _features(a):
    t = a["tokens"]
    decisions = _decisions(t)
    active = [i for i, (keep, _, _) in enumerate(decisions) if keep]
    compatible = [i for i, (keep, _, donors) in enumerate(decisions) if not t[i]["eligible"] and donors]
    expected = a["output"]

    def different(**kwargs):
        try:
            return not np.array_equal(_render(a, **kwargs)[0], expected)
        except AssertionError:
            return True  # A proposed shortcut can also violate visible capacity.

    selection_witnesses = []
    for i in compatible:
        extra_tokens = [dict(x, eligible=x["eligible"] or j == i) for j, x in enumerate(t)]
        try:
            extra_out, _ = _render(dict(a, tokens=extra_tokens))
        except AssertionError:
            continue
        if not np.array_equal(extra_out, expected):
            selection_witnesses.append(i)
    footprints = [cells for _, cells in a["placements"]]
    overlap = sum(map(len, footprints)) - len(set().union(*footprints)) if footprints else 0
    signatures = []
    for i in active:
        for color, cells in _color_parts(t[i], decisions[i][1]):
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            signatures.append(tuple(sorted((r-r0, c-c0) for r, c in cells)))
    return dict(shape=a["grid"].shape, rotation=a["rotation"], background=a["background"],
                line=a["line"], tokens=len(t), bands=len(a["bands"]),
                depths=tuple(b["bottom"]-b["top"]+1 for b in a["bands"]),
                eligible=sum(x["eligible"] for x in t), survivors=len(active),
                failed=sum(x["eligible"] and not d[2] for x, d in zip(t, decisions)),
                fused=sum(t[i]["body"] == decisions[i][1] and t[i]["value"] != t[i]["body"] for i in active),
                collapsed=sum(t[i]["kind"] == "cross" and decisions[i][1] != t[i]["body"] for i in active),
                interior_compatible=sum(bool(_stubs(t[i]) & a["graph"]) for i in selection_witnesses),
                overlap=overlap, unused=len(a["bands"])-len({a["grid"][min(cells)] for _, cells in a["placements"]}),
                cross=sum(x["kind"] == "cross" for x in t), rectangle=sum(x["kind"] == "rectangle" for x in t),
                widths=tuple(x["w"] for x in t), signatures=tuple(signatures),
                fusion_witness=different(grouping="before"),
                gravity_witness=different(grouping="rigid"),
                lookup_witness=different(lookup="none"), selection_witness=bool(selection_witnesses),
                simultaneous_witness=different(lookup="extra"), far_witness=different(near=True))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    turns = list(map(int, rng.permutation(4)))[:2 if rng.random() < .45 else 1]
    lessons = ["fusion", "chain"] + ["random"]*(num_train-2)
    rng.shuffle(lessons)
    seen, puzzle = set(), {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = lessons[i] if split == "train" else "random"
            turn = turns[i % len(turns)] if split == "train" else int(rng.choice(turns))
            for _ in range(12):
                pair = _make_pair(rng, mode, split == "test", turn)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate-input rejection exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _need(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
          len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions changed")
            key = _key(inp)
            _need(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            canonical_out = np.rot90(out, a["rotation"])
            _need(np.array_equal(canonical_out, a["output"]), "incorrect settling or lookup")
            token_cells = set().union(*(t["cells"] for t in a["tokens"]))
            _need(all(canonical_out[p] == a["background"] for p in token_cells), "source not cleared")
            _need(all(canonical_out[p] == a["line"] for p in a["graph"]), "line changed")
            for b in a["bands"]:
                _need(all(canonical_out[r, c] == b["rail"] for r in range(b["top"], b["bottom"]+1)
                          for c in (b["left"], b["right"])), "rail changed")
            _need(not np.array_equal(inp, out), "identity pair")
            f = _features(a)
            _need(f["survivors"] >= 1, "no settling stencil")
            features[split].append(f)
    train, test = features["train"], features["test"]
    for name in ("fusion_witness", "gravity_witness", "lookup_witness", "selection_witness",
                 "simultaneous_witness", "far_witness", "failed", "interior_compatible"):
        _need(any(f[name] for f in train), "training lacks " + name)
    _need({f["rotation"] for f in test} <= {f["rotation"] for f in train}, "untrained direction")
    _need(min(max(f["widths"]) for f in test) > max(max(f["widths"]) for f in train),
          "test must transfer to a wider stencil")


def self_test():
    """Bounded 42-seed structural survey, ablations, and negative checks."""
    records, keys = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        keys.add(tuple(_key(q["input"]) for s in ("train", "test") for q in p[s]))
        local = [_features(_analyze(q["input"])) for s in ("train", "test") for q in p[s]]
        _need(len({f["shape"] for f in local}) >= 2, "fixed canvas")
        records.extend(local)
    _need(len(keys) == 42, "duplicate puzzles across seeds")
    _need(len({f["shape"] for f in records}) >= 40, "narrow canvas distribution")
    _need({f["rotation"] for f in records} == set(range(4)), "missing gravity direction")
    _need({3, 4, 5} <= {f["bands"] for f in records}, "narrow band counts")
    _need({4, 5, 6, 7} <= {f["tokens"] for f in records}, "narrow token counts")
    _need(len({f["depths"] for f in records}) >= 35, "narrow destination geometry")
    _need(len({s for f in records for s in f["signatures"]}) >= 20, "narrow settled masks")
    _need(len({(f["background"], f["line"]) for f in records}) >= 35, "narrow role palettes")
    _need(all(any(f[n] for f in records) for n in ("fused", "collapsed", "overlap", "rectangle", "cross")),
          "missing topology/union cases")
    p, q = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _need(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
              for k in ("input", "output")), "nondeterminism")
    for counts in ((10, 4), (10, 1), (5, 4)):
        p = generate_puzzle(701+sum(counts), *counts)
        _need(tuple(len(p[s]) for s in ("train", "test")) == counts, "ignored counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A pixel-level hand calculation: a 3x3 cross with a foreign center has
    # four disconnected body cells; its body sediment is a three-cell bar.
    t = dict(center=(2, 4), h=3, w=3, kind="cross", body=1, value=2,
             cells=_support((2, 4), 3, 3, "cross"), eligible=True)
    pieces = _color_parts(t, 3)
    body = [cells for color, cells in pieces if color == 1]
    _need(len(body) == 4, "foreign center must separate four arms")
    landed = {(10-max(r for r, c in cells)+r, c) for cells in body for r, c in cells}
    _need(landed == {(10, 3), (10, 4), (10, 5)}, "incorrect union sediment")
    _need(len(_color_parts(t, 1)) == 1, "matching center must fuse the cross")
    # Hand-built member outside the sampler's size/count ranges: a 7x11 cross
    # and six bands. Calculate its two outcomes directly, without _render.
    for donor_value in (1, 3):
        inp = np.zeros((30, 21), dtype=np.int8)
        cursor, band_rows = 11, {}
        for identity in range(1, 7):
            cursor += 1
            depth = 7 if identity == 1 else 1
            band_rows[identity] = cursor+depth-1
            inp[cursor:cursor+depth, 1:20] = identity
            inp[cursor:cursor+depth, [0, 20]] = 7
            cursor += depth
        big = _support((4, 8), 7, 11, "cross")
        donor = _support((8, 17), 3, 3, "rectangle")
        third = _support((3, 17), 3, 3, "rectangle")
        for cells, color in ((big, 1), (donor, 2), (third, 3)):
            for cell in cells:
                inp[cell] = color
        inp[4, 8], inp[8, 17], inp[3, 17] = 2, donor_value, 4
        for cell in ((8, 8), (9, 8), (9, 9), (9, 10)):
            inp[cell] = 8
        expected = inp.copy()
        for cell in big | donor | third:
            expected[cell] = 0
        if donor_value == 1:
            for r, c in big:
                expected[r+11, c] = 7
        else:
            expected[16:19, 8] = 7
            expected[18, 3:14] = 7
            expected[band_rows[3], 8] = 7
        for rotation in range(4):
            _need(np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation)),
                  "large-stencil/six-band hand calculation failed")
    bad = generate_puzzle(991, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
