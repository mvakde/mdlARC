"""Scoped relay windows: geometry restricts simultaneous color lookup.

Only the public input is needed by solve. See task_spec.md for the family rule.
No parent module, file, seed, or generated example is used during solving.
"""
from __future__ import annotations

import heapq
import numpy as np


D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, part = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _corners(box, diagonal):
    r0, c0, r1, c1 = box
    if diagonal == 0:
        return {(r0, c0), (r0, c0+1), (r0+1, c0),
                (r1, c1), (r1, c1-1), (r1-1, c1)}
    return {(r0, c1), (r0, c1-1), (r0+1, c1),
            (r1, c0), (r1, c0+1), (r1-1, c0)}


def _inside(p, box):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _area(box):
    return (box[2]-box[0]+1) * (box[3]-box[1]+1)


def _support(center, kind):
    r, c = center
    return {(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if kind == "square" or dr == 0 or dc == 0}


def _touch(a, b):
    return bool(a & b) or any((r+dr, c+dc) in b for r, c in a for dr, dc in D4)


def _graph(cells):
    return {p: {(p[0]+dr, p[1]+dc) for dr, dc in D4} & cells for p in cells}


def _parse_roles(grid, bg, line):
    cells = {tuple(map(int, p)) for p in np.argwhere((grid != bg) & (grid != line))}
    fragments, tokens = {}, []
    for part in _components(cells):
        if len(part) == 3:
            rs, cs = zip(*part)
            values = {int(grid[p]) for p in part}
            _check(len(values) == 1 and max(rs)-min(rs) == max(cs)-min(cs) == 1,
                   "malformed L fragment")
            cue = values.pop()
            fragments.setdefault(cue, set()).update(part)
            continue
        _check(len(part) in (5, 9), "unrecognized foreground component")
        rs, cs = zip(*part)
        center = ((min(rs)+max(rs))//2, (min(cs)+max(cs))//2)
        kind = "cross" if len(part) == 5 else "square"
        _check(_support(center, kind) == part, "uncentered token")
        bodies = {int(grid[p]) for p in part if p != center}
        _check(len(bodies) == 1, "nonuniform token body")
        body, value = bodies.pop(), int(grid[center])
        _check(value not in (bg, line), "token identity uses structural color")
        tokens.append(dict(center=center, cells=part, kind=kind, body=body, value=value))
    frames = {}
    for cue, markers in fragments.items():
        rs, cs = zip(*markers)
        box = min(rs), min(cs), max(rs), max(cs)
        _check(len(markers) == 6 and min(box[2]-box[0]+1, box[3]-box[1]+1) >= 4,
               "frame needs two separated L corners")
        _check(any(markers == _corners(box, d) for d in (0, 1)), "corners disagree")
        frames[cue] = dict(cue=cue, box=box, markers=markers, area=_area(box))
    _check(bool(frames) and bool(tokens), "missing frames or tokens")
    line_cells = {tuple(map(int, p)) for p in np.argwhere(grid == line)}
    graph = _graph(line_cells)
    _check(bool(graph) and all(len(adj) in (1, 2) for adj in graph.values()), "branched line")
    ends = {p for p, adj in graph.items() if len(adj) == 1}
    parts = _components(line_cells)
    _check(all(len(p) >= 3 and len(p & ends) == 2 for p in parts), "line is closed or too short")
    selected = []
    for i, token in enumerate(tokens):
        r, c = token["center"]
        token["ends"] = {(r+2*dr, c+2*dc) for dr, dc in D4} & ends
        _check(len(token["ends"]) <= 1, "two line ends select one token")
        token["eligible"] = bool(token["ends"])
        if token["eligible"]:
            _check(token["body"] in frames, "selected body lacks a frame")
            _check(_inside(token["center"], frames[token["body"]]["box"]),
                   "root center outside its own scope")
            selected.append(i)
    _check(bool(selected), "no selected roots")
    _check(len({tokens[i]["body"] for i in selected}) == len(selected), "duplicate selected frame")
    _check(all(sum(p in t["ends"] for t in tokens) <= 1 for p in ends), "ambiguous end attachment")
    return dict(bg=bg, line=line, frames=frames, tokens=tokens,
                line_cells=line_cells, graph=graph, line_parts=parts)


def _read(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _check(np.count_nonzero(counts == counts[bg]) == 1, "ambiguous background")
    interpretations = []
    for line in np.flatnonzero(counts):
        if line == bg:
            continue
        try:
            interpretations.append(_parse_roles(grid, bg, int(line)))
        except ValueError:
            pass
    _check(len(interpretations) == 1, "ambiguous line/frame/token interpretation")
    return interpretations[0]


def _actions(p, *, selection="end", scope=True, updated=False):
    """All donor reads come from the input. Alternate policies audit shortcuts."""
    tokens, frames = p["tokens"], p["frames"]
    retained, fills, donors = {}, {}, {}
    for i, t in enumerate(tokens):
        if t["body"] not in frames:
            continue
        eligible = t["eligible"] if selection == "end" else (
            _touch(t["cells"], p["line_cells"]) if selection == "touch" else True)
        if not eligible or not _inside(t["center"], frames[t["body"]]["box"]):
            continue
        ds = [j for j, d in enumerate(tokens) if j != i and d["body"] == t["value"]
              and (not scope or _inside(d["center"], frames[t["body"]]["box"]))]
        if not ds:
            continue
        if not scope:
            # An explicitly defined nearest-global-donor alternative.
            ds = [min(ds, key=lambda j: (abs(tokens[j]["center"][0]-t["center"][0])
                                        + abs(tokens[j]["center"][1]-t["center"][1]),
                                        tokens[j]["center"]))]
        vals = {tokens[j]["value"] for j in ds}
        _check(len(vals) == 1, "conflicting donors within selected scope")
        z = vals.pop()
        retained[i], fills[t["body"]], donors[i] = z, z, ds
    if updated:
        # Wrong policy: donate a survivor's already changed center.
        changed = {i: retained.get(ds[0], tokens[ds[0]]["value"]) for i, ds in donors.items()}
        retained = changed
        fills = {tokens[i]["body"]: z for i, z in retained.items()}
    areas = [frames[cue]["area"] for cue in fills]
    _check(len(set(areas)) == len(areas), "active frames have equal area")
    return retained, fills, donors


def _render(grid, p, retained, fills, reverse=False, complete=True):
    result = grid.copy()
    for token in p["tokens"]:
        for cell in token["cells"]:
            result[cell] = p["bg"]
    if complete:
        for cue in sorted(fills, key=lambda c: p["frames"][c]["area"], reverse=not reverse):
            frame = p["frames"][cue]
            r0, c0, r1, c1 = frame["box"]
            result[r0:r1+1, c0:c1+1] = fills[cue]
            for cell in frame["markers"]:
                result[cell] = cue
    for cell in p["line_cells"]:
        result[cell] = p["line"]
    for i, z in retained.items():
        token = p["tokens"][i]
        for cell in token["cells"]:
            result[cell] = token["body"]
        result[token["center"]] = z
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    p = _read(grid)
    retained, fills, _ = _actions(p)
    return _render(grid, p, retained, fills)


def _pixel_check(grid, p, output):
    """Check ownership cell by cell instead of repainting bounding-box layers."""
    retained, fills, _ = _actions(p)
    token_owner = {cell: i for i, t in enumerate(p["tokens"]) for cell in t["cells"]}
    for cell in np.ndindex(grid.shape):
        i = token_owner.get(cell)
        if cell in p["line_cells"]:
            expected = p["line"]
        elif i in retained:
            t = p["tokens"][i]
            expected = retained[i] if cell == t["center"] else t["body"]
        else:
            covering = [cue for cue in fills if _inside(cell, p["frames"][cue]["box"])]
            if covering:
                cue = min(covering, key=lambda c: p["frames"][c]["area"])
                expected = cue if cell in p["frames"][cue]["markers"] else fills[cue]
            else:
                expected = p["bg"] if i is not None else int(grid[cell])
        _check(int(output[cell]) == expected, "incorrect pixel ownership")


def _features(grid, p=None):
    p = _read(grid) if p is None else p
    kept, fills, donors = _actions(p)
    out = _render(grid, p, kept, fills)
    tokens, frames = p["tokens"], p["frames"]
    alternatives = {}
    for name, kw in (("ignore_scope", dict(scope=False)), ("ignore_endpoints", dict(selection="all")),
                     ("touching", dict(selection="touch")), ("updated_reads", dict(updated=True))):
        try:
            ks, fs, _ = _actions(p, **kw)
            alternatives[name] = not np.array_equal(out, _render(grid, p, ks, fs))
        except ValueError:
            alternatives[name] = True
    alternatives["ignore_completion"] = not np.array_equal(out, _render(grid, p, kept, fills, complete=False))
    alternatives["reverse_priority"] = not np.array_equal(out, _render(grid, p, kept, fills, reverse=True))
    straddling = sum(any(not _inside(cell, frames[tokens[i]["body"]]["box"])
                         for cell in tokens[j]["cells"]) for i, ds in donors.items() for j in ds)
    nearer_wrong = False
    for i, z in kept.items():
        t = tokens[i]
        distance = lambda other: abs(other["center"][0]-t["center"][0]) + abs(other["center"][1]-t["center"][1])
        wrong = [distance(d) for j, d in enumerate(tokens) if j != i and d["body"] == t["value"]
                 and d["value"] != z and not _inside(d["center"], frames[t["body"]]["box"])]
        right = [distance(d) for j, d in enumerate(tokens) if j != i and d["body"] == t["value"]
                 and d["value"] == z]
        nearer_wrong |= bool(wrong and right and min(wrong) < min(right))
    occluded = sum(int(out[cell]) != frame["cue"] and cell not in p["line_cells"]
                   and not any(cell in tokens[i]["cells"] for i in kept)
                   for frame in frames.values() for cell in frame["markers"])
    unselected_effect = interior_effect = False
    for i, t in enumerate(tokens):
        if t["eligible"] or t["body"] not in frames or not _inside(t["center"], frames[t["body"]]["box"]):
            continue
        vals = {d["value"] for j, d in enumerate(tokens) if j != i and d["body"] == t["value"]
                and _inside(d["center"], frames[t["body"]]["box"])}
        if len(vals) == 1:
            z = vals.pop()
            changed = any(int(out[cell]) != (z if cell == t["center"] else t["body"])
                          for cell in t["cells"])
            unselected_effect |= changed
            interior_effect |= changed and _touch(t["cells"], p["line_cells"])
    return dict(shape=grid.shape, frames=len(frames), tokens=len(tokens), active=len(fills),
                lines=len(p["line_parts"]), bg=p["bg"], line=p["line"],
                kinds=tuple(sorted({t["kind"] for t in tokens})),
                areas=tuple(sorted(f["area"] for f in frames.values())),
                boxes=tuple(sorted(f["box"] for f in frames.values())),
                border_donors=straddling, covered_corners=occluded,
                donor_counts=tuple(sorted(len(ds) for ds in donors.values())),
                retained_donor=any(j in kept for ds in donors.values() for j in ds),
                nearer_wrong=nearer_wrong,
                unselected_effect=unselected_effect, interior_effect=interior_effect,
                **alternatives)


def _box(rng, shape, edge=False):
    h, w = shape
    hh = int(rng.integers(6, min(h, 20)+1))
    ww = int(rng.integers(6, min(w, 20)+1))
    r, c = int(rng.integers(h-hh+1)), int(rng.integers(w-ww+1))
    if edge:
        axis = int(rng.integers(4))
        if axis == 0:
            r = 0
        elif axis == 1:
            r = h-hh
        elif axis == 2:
            c = 0
        else:
            c = w-ww
    return r, c, r+hh-1, c+ww-1


def _route(rng, shape, occupied, start, end):
    if start in occupied or end in occupied or start == end:
        return None
    costs = rng.uniform(1, 1.7, size=shape)
    queue, distance, previous = [(0., start)], {start: 0.}, {}
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p == end:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1]
        for dr, dc in D4:
            q = p[0]+dr, p[1]+dc
            if not (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]) or q in occupied:
                continue
            new = cost+float(costs[q])
            if new < distance.get(q, float("inf")):
                distance[q], previous[q] = new, p
                heapq.heappush(queue, (new, q))
    return None


def _lines(rng, shape, centers, occupied, roots, interior=False, separate=False):
    stubs = lambda p: [(p[0]+2*dr, p[1]+2*dc) for dr, dc in D4
                      if 0 <= p[0]+2*dr < shape[0] and 0 <= p[1]+2*dc < shape[1]
                      and (p[0]+2*dr, p[1]+2*dc) not in occupied]
    all_stubs = set(q for center in centers for q in stubs(center))
    bare = [(r, c) for r in range(shape[0]) for c in range(shape[1])
            if (r, c) not in occupied and (r, c) not in all_stubs]
    if not bare:
        return None
    groups = [[i] for i in roots] if separate else [roots]
    line_cells = set()
    for group in groups:
        start_opts = stubs(centers[group[0]])
        if not start_opts:
            return None
        start = start_opts[int(rng.integers(len(start_opts)))]
        if len(group) == 2:
            ends = stubs(centers[group[1]])
            if not ends:
                return None
            end = ends[int(rng.integers(len(ends)))]
        else:
            end = bare[int(rng.integers(len(bare)))]
        blocked = set(occupied)
        for r, c in line_cells:
            blocked.add((r, c))
            blocked.update((r+dr, c+dc) for dr, dc in D4)
        if interior:
            waypoints = stubs(centers[1])
            if not waypoints:
                return None
            wp = waypoints[int(rng.integers(len(waypoints)))]
            first = _route(rng, shape, blocked | {end}, start, wp)
            if first is None or len(first) < 2:
                return None
            extra_block = set(first[:-1])
            for r, c in first[:-1]:
                extra_block.update((r+dr, c+dc) for dr, dc in D4)
            # Only the shared waypoint can join the two paths.
            extra_block.discard(wp)
            for dr, dc in D4:
                q = wp[0]+dr, wp[1]+dc
                if q not in first and not any(abs(q[0]-p[0])+abs(q[1]-p[1]) == 1
                                               for p in first[:-1]):
                    extra_block.discard(q)
            second = _route(rng, shape, blocked | extra_block, wp, end)
            path = None if second is None else first + second[1:]
        else:
            path = _route(rng, shape, blocked, start, end)
        if path is None or len(path) < 3:
            return None
        line_cells.update(path)
    graph = _graph(line_cells)
    if not all(len(adj) in (1, 2) for adj in graph.values()):
        return None
    ends = {p for p, adj in graph.items() if len(adj) == 1}
    actual = {i for i, center in enumerate(centers) if set(stubs(center)) & ends}
    return line_cells if actual == set(roots) else None


def _candidate(rng, params, mode, split):
    # These are randomized constraint-based placements, not position templates.
    shape = int(rng.integers(17, 31)), int(rng.integers(17, 31))
    colors = list(map(int, rng.permutation(10))) if params["varied"] else (
        list(params["roles"]) + list(map(int, rng.permutation(params["identities"]))))
    bg, line, a, b, c, d, e = colors[:7]
    nframes = params["train_frames"] + (split == "test")
    frames, markers = {}, set()
    for i, cue in enumerate(colors[2:2+nframes]):
        placed = False
        for _ in range(40):
            box = _box(rng, shape, edge=rng.random() < (.55 if split == "test" else .2))
            cells = _corners(box, int(rng.integers(2)))
            if _touch(cells, markers):
                continue
            if i == 1:
                aa = frames[a]["box"]
                overlap_h = min(aa[2], box[2])-max(aa[0], box[0])+1
                overlap_w = min(aa[3], box[3])-max(aa[1], box[1])+1
                if min(overlap_h, overlap_w) < 4 or _area(box) == _area(aa):
                    continue
                if mode == "scope" and all(_inside(p, aa) for p in cells):
                    continue
            if any(box == f["box"] for f in frames.values()):
                continue
            frames[cue] = dict(cue=cue, box=box, area=_area(box), markers=cells)
            markers.update(cells)
            placed = True
            break
        if not placed:
            return None
    kinds = [params["morphology"]]*4
    if params["morphology"] == "mixed":
        kinds = list(map(str, rng.choice(["square", "cross"], size=4)))
    centers, tokens, occupied = [], [], set(markers)
    relations = [(a, b), (b, c), (c, d), (b, e)]
    n_extra = int(rng.integers(0, 4))
    for i in range(4+n_extra):
        kind = kinds[i] if i < 4 else (str(rng.choice(["square", "cross"]))
                                      if params["morphology"] == "mixed" else params["morphology"])
        options = []
        for r in range(1, shape[0]-1):
            for cc in range(1, shape[1]-1):
                center = r, cc
                if i == 0 and not _inside(center, frames[a]["box"]):
                    continue
                if i == 1 and not (_inside(center, frames[a]["box"]) and _inside(center, frames[b]["box"])):
                    continue
                if i == 2 and not _inside(center, frames[b]["box"]):
                    continue
                if i == 3 and _inside(center, frames[a]["box"]):
                    continue
                footprint = _support(center, "square")
                if _touch(footprint, occupied):
                    continue
                options.append(center)
        if not options:
            return None
        # Some examples teach center-only boundary membership, before tests use it.
        if i == 1 and mode == "snapshot":
            border = [p for p in options if any(not _inside(q, frames[a]["box"])
                                               for q in _support(p, kind))]
            if border:
                options = border
        center = options[int(rng.integers(len(options)))]
        if i < 4:
            body, value = relations[i]
        else:
            body, value = map(int, rng.choice(colors[2:], 2))
            if body == b and _inside(center, frames[a]["box"]):
                value = c
            if body == c and _inside(center, frames[b]["box"]):
                value = d
        cells = _support(center, kind)
        centers.append(center)
        tokens.append(dict(center=center, kind=kind, cells=cells, body=body, value=value))
        occupied.update(cells)
    roots = [0] if mode == "scope" else [0, 1]
    if mode == "random":
        roots = [0] if rng.random() < .4 else [0, 1]
        # Random visible relations, including shared terminals and donor reuse.
        # Root bodies identify their frames, but no identity chain is prescribed.
        for i, t in enumerate(tokens):
            if i not in roots:
                t["body"] = int(rng.choice(colors[2:]))
            t["value"] = int(rng.choice(colors[2:]))
        for i in reversed(roots):
            t = tokens[i]
            options = [j for j, donor in enumerate(tokens) if j != i
                       and _inside(donor["center"], frames[t["body"]]["box"])]
            if not options:
                return None
            j = options[int(rng.integers(len(options)))]
            t["value"] = tokens[j]["body"]
            ds = [donor for k, donor in enumerate(tokens) if k != i and donor["body"] == t["value"]
                  and _inside(donor["center"], frames[t["body"]]["box"])]
            z = int(rng.choice(colors[2:]))
            for donor in ds:
                donor["value"] = z
    line_cells = _lines(rng, shape, centers, occupied, roots, interior=mode == "scope",
                        separate=mode != "scope" and rng.random() < .35)
    if line_cells is None:
        return None
    grid = np.full(shape, bg, dtype=np.int8)
    for cue, frame in frames.items():
        for p in frame["markers"]:
            grid[p] = cue
    for t in tokens:
        for p in t["cells"]:
            grid[p] = t["body"]
        grid[t["center"]] = t["value"]
    for p in line_cells:
        grid[p] = line
    # Build labels from construction objects; solving rediscovers these objects.
    for i, t in enumerate(tokens):
        t["eligible"] = i in roots
    constructed = dict(bg=bg, line=line, frames=frames, tokens=tokens, line_cells=line_cells)
    try:
        retained, fills, _ = _actions(constructed)
    except ValueError:
        return None
    if mode == "random" and (len(retained) != len(roots) or any(
            z in (tokens[i]["body"], tokens[i]["value"]) for i, z in retained.items())):
        return None
    output = _render(grid, constructed, retained, fills)
    # Audit the actual returned orientation, including all shortcut witnesses.
    k = int(rng.integers(4))
    grid, output = np.rot90(grid, k), np.rot90(output, k)
    if rng.random() < .5:
        grid, output = np.fliplr(grid), np.fliplr(output)
    try:
        parsed = _read(grid)
        f = _features(grid, parsed)
        if mode == "scope" and not (f["ignore_scope"] and f["ignore_endpoints"]
                                    and f["touching"] and f["nearer_wrong"]):
            return None
        if mode == "snapshot" and not (f["updated_reads"] and f["reverse_priority"]
                                       and f["covered_corners"] and f["border_donors"]):
            return None
        if not np.array_equal(solve(grid), output):
            return None
    except ValueError:
        return None
    return dict(input=grid.copy(), output=output.copy())


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, n, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= int(n) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(10)))
    params = dict(roles=colors[:2], identities=colors[2:], varied=bool(rng.random() < .7),
                  morphology=str(rng.choice(["square", "cross", "mixed"])),
                  train_frames=int(rng.integers(2, 4)))
    modes = ["scope", "snapshot"] + list(map(str, rng.choice(["scope", "snapshot", "random"],
                                                            size=num_train-2, p=[.15, .15, .7])))
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = modes[i] if split == "train" else str(rng.choice(["scope", "snapshot", "random"],
                                                                   p=[.15, .2, .65]))
            for _ in range(1200):
                pair = _candidate(rng, params, mode, split)
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("bounded geometric construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, feats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _check(grid.shape == output.shape, "output changed canvas size")
            _check(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            p = _read(grid)
            _pixel_check(grid, p, output)
            _check(np.array_equal(solve(grid), output), "input-only solve mismatch")
            _check(not np.array_equal(grid, output), "vacuous pair")
            f = _features(grid, p)
            _check(f["ignore_completion"], "frame completion has no effect")
            feats[split].append(f)
    training = feats["train"]
    for feature in ("ignore_scope", "ignore_endpoints", "touching", "updated_reads", "reverse_priority"):
        _check(any(f[feature] for f in training), f"training does not distinguish {feature}")
    _check(any(f["covered_corners"] for f in training), "corner occlusion not taught")
    _check(any(f["retained_donor"] for f in training), "retained donors not taught")
    _check(any(f["border_donors"] for f in training), "center-only membership not taught")
    _check(any(f["nearer_wrong"] for f in training), "nearer outside-donor contrast not taught")
    _check(any(f["unselected_effect"] and f["interior_effect"] for f in training),
           "no concrete compatible interior-token removal witness")
    _check(max(f["frames"] for f in feats["test"]) == max(f["frames"] for f in training)+1,
           "missing structural train/test shift")
    _check(len({(f["shape"], f["boxes"]) for f in training+feats["test"]}) > 1,
           "geometry does not vary")


def self_test():
    """Bounded 42-seed sweep, concrete semantic witnesses and mutation checks."""
    records, puzzles, keys = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        signature = tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s])
        assert signature not in keys
        keys.add(signature)
        for s in ("train", "test"):
            for pair in p[s]:
                records.append(_features(pair["input"]))
    assert len({f["shape"] for f in records}) >= 50
    assert len({f["boxes"] for f in records}) >= 150
    assert {f["frames"] for f in records} >= {2, 3, 4}
    assert len({f["tokens"] for f in records}) >= 4
    assert {f["active"] for f in records} == {1, 2}
    assert {f["lines"] for f in records} >= {1, 2}
    assert {f["bg"] for f in records} == set(range(10))
    assert {f["line"] for f in records} == set(range(10))
    assert any(len(f["kinds"]) == 2 for f in records)
    assert sum(f["border_donors"] > 0 for f in records) >= 10
    assert any(max(f["shape"]) == 30 for f in records)
    assert any(max(f["shape"]) <= 20 for f in records)
    assert any(max(f["donor_counts"]) > 1 for f in records)
    again = generate_puzzle(0)
    assert all(np.array_equal(x[r], y[r]) for s in again for x, y in zip(again[s], puzzles[0][s])
               for r in ("input", "output"))
    # Rule-supported members beyond the sampler's successful/non-solid roots.
    scope_pair = next(pair for pair in puzzles[0]["train"]
                      if _features(pair["input"])["active"] == 1
                      and _features(pair["input"])["nearer_wrong"])
    scope_grid = scope_pair["input"]
    parsed = _read(scope_grid)
    retained, _, donors = _actions(parsed)
    i = next(iter(retained))
    root = parsed["tokens"][i]
    for result_color in (root["body"], root["value"]):
        variant = scope_grid.copy()
        for j in donors[i]:
            variant[parsed["tokens"][j]["center"]] = result_color
        answer = solve(variant)
        assert int(answer[root["center"]]) == result_color
        _pixel_check(variant, _read(variant), answer)
    absent = next(c for c in range(10) if c not in (parsed["bg"], parsed["line"])
                  and not any(t["body"] == c and j != i and _inside(t["center"],
                      parsed["frames"][root["body"]]["box"]) for j, t in enumerate(parsed["tokens"])))
    variant = scope_grid.copy()
    variant[root["center"]] = absent
    answer = solve(variant)
    assert not _actions(_read(variant))[0]
    assert all(int(answer[cell]) == parsed["bg"] for t in parsed["tokens"] for cell in t["cells"])
    _pixel_check(variant, _read(variant), answer)
    # The rule commutes with rigid motions and any bijective color reassignment.
    for k in range(4):
        variant = np.rot90(scope_grid, k)
        assert np.array_equal(solve(variant), np.rot90(scope_pair["output"], k))
    relabel = np.array([9, 2, 6, 0, 5, 7, 1, 8, 4, 3], dtype=np.int8)
    assert np.array_equal(solve(relabel[scope_grid]), relabel[scope_pair["output"]])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(4242+nt*10+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [dict(input=x["input"].copy(), output=x["output"].copy()) for x in puzzles[0][s]]
           for s in ("train", "test")}
    p = _read(bad["train"][0]["input"])
    cell = min(p["line_cells"])
    bad["train"][0]["output"][cell] = p["bg"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("line corruption accepted")
    return dict(seeds=42, pairs=len(records), shapes=len({f["shape"] for f in records}),
                layouts=len({f["boxes"] for f in records}),
                border_pairs=sum(f["border_donors"] > 0 for f in records))


if __name__ == "__main__":
    print(self_test())
