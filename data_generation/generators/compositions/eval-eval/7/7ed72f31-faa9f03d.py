"""Symmetry broadcast routers: commands are copied; cable shapes are not.

Only Python's standard library and NumPy are used. The public solver reads one
grid, and neither reads files nor consults demonstrations or generation state.
"""
from __future__ import annotations

import copy
import json
import numpy as np

SYMMETRY, DIRECTION, REPAIR = 2, 4, 6
ANNOTATIONS = {SYMMETRY, DIRECTION, REPAIR}
DATA_COLORS = (0, 1, 3, 5, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be a 1..30 by 1..30 integer array of colors 0..9")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + i*d[0], a[1] + i*d[1]) for i in range(length + 1)]


def _polyline(vertices):
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(p, d, shape):
    result = []
    q = _add(p, d)
    while _inside(q, shape):
        result.append(q)
        q = _add(q, d)
    return result


def _halo(cells, shape):
    return {q for p in cells for d in ((0, 0),) + DIRS
            if _inside(q := _add(p, d), shape)}


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, found = [first], {first}
        while stack:
            for q in _neighbors(stack.pop(), remaining):
                remaining.remove(q)
                found.add(q)
                stack.append(q)
        result.append(found)
    return result


def _operator(inp):
    cells = {tuple(map(int, p)) for p in np.argwhere(inp == SYMMETRY)}
    assert cells and len(_components(cells)) == 1, "one symmetry operator is required"
    rs, cs = {r for r, _ in cells}, {c for _, c in cells}
    if len(cells) == 1:
        return "P", min(cells), cells
    if len(rs) == 1 and len(cells) == max(cs) - min(cs) + 1:
        return "H", (min(rs), min(cs)), cells
    if len(cs) == 1 and len(cells) == max(rs) - min(rs) + 1:
        return "V", (min(rs), min(cs)), cells
    raise AssertionError("symmetry marker must be a straight segment or singleton")


def _map(p, kind, origin):
    r, c = p
    a, b = origin
    return ((2*a-r, c) if kind == "H" else
            (r, 2*b-c) if kind == "V" else (2*a-r, 2*b-c))


def _reconstruct(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert bg not in ANNOTATIONS and counts[bg] > inp.size // 2
    colors = set(map(int, np.unique(inp))) - ANNOTATIONS - {bg}
    assert colors
    work = inp.copy()
    pending = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
    # Markers can replace a bend, pivot, endpoint, or straight waypoint. They
    # are assigned only when all currently visible trace neighbors agree.
    while pending:
        assignments = {}
        for p in sorted(pending):
            witnesses = {int(work[q]) for d in DIRS
                         if _inside(q := _add(p, d), inp.shape) and int(work[q]) in colors}
            if len(witnesses) == 1:
                assignments[p] = next(iter(witnesses))
        assert assignments, "repair marker has no unique neighboring color"
        for p, color in assignments.items():
            work[p] = color
            pending.remove(p)
    # One-cell omissions are inferred simultaneously, without extending ends.
    gaps = {}
    for p0 in np.argwhere(work == bg):
        p = tuple(map(int, p0))
        witnesses = set()
        for d in ((1, 0), (0, 1)):
            a, b = _add(p, d), _add(p, (-d[0], -d[1]))
            if _inside(a, inp.shape) and _inside(b, inp.shape):
                if int(work[a]) == int(work[b]) and int(work[a]) in colors:
                    witnesses.add(int(work[a]))
        assert len(witnesses) <= 1, "gap has competing color assignments"
        if witnesses:
            gaps[p] = next(iter(witnesses))
    for p, color in gaps.items():
        work[p] = color
    layers = {c: {tuple(map(int, p)) for p in np.argwhere(work == c)} for c in colors}
    # Input cables do not cross. Their separate identities therefore require
    # no learned or invisible color precedence.
    for c, cells in layers.items():
        assert all(len(_neighbors(p, cells)) <= 3 for p in cells)
        for p in cells:
            assert all(int(work[q]) not in colors - {c} for d in DIRS
                       if _inside(q := _add(p, d), inp.shape))
    return bg, work, layers, gaps


def _edit(p, d, color, cells, shape):
    assert p in cells and _add(p, d) not in cells
    old_start = _add(p, (-d[0], -d[1]))
    assert old_start in cells
    perpendicular = (-d[1], d[0])
    assert (_add(p, perpendicular) in cells or
            _add(p, (-perpendicular[0], -perpendicular[1])) in cells)
    previous, current, walk = p, old_start, []
    while True:
        assert current not in walk, "terminal branch loops"
        walk.append(current)
        following = [q for q in _neighbors(current, cells) if q != previous]
        if not following:
            assert _boundary(current, shape), "old branch must terminate on the boundary"
            break
        assert len(following) == 1, "old branch is not uniquely traceable"
        previous, current = current, following[0]
    new = set(_ray(p, d, shape))
    assert new
    return {"pivot": p, "direction": d, "color": color,
            "old": set(walk), "walk": [p] + walk, "new": new}


def _analyse(inp):
    kind, origin, gate = _operator(inp)
    bg, work, layers, gaps = _reconstruct(inp)
    edits, commands = [], []
    for q0 in np.argwhere(inp == DIRECTION):
        q = tuple(map(int, q0))
        candidates = []
        for d in DIRS:
            p = q[0]-d[0], q[1]-d[1]
            if not _inside(p, inp.shape) or int(work[p]) not in layers:
                continue
            color = int(work[p])
            try:
                local = _edit(p, d, color, layers[color], inp.shape)
                pp, qq = _map(p, kind, origin), _map(q, kind, origin)
                assert pp != p and _inside(pp, inp.shape) and _inside(qq, inp.shape)
                assert int(work[qq]) == bg, "only one member of a command pair is drawn"
                remote_color = int(work[pp])
                assert remote_color in layers
                dd = qq[0]-pp[0], qq[1]-pp[1]
                remote = _edit(pp, dd, remote_color, layers[remote_color], inp.shape)
                candidates.append((local, remote, q, qq))
            except AssertionError:
                continue
        assert len(candidates) == 1, "command does not have a unique local and symmetric pivot"
        a, b, q, qq = candidates[0]
        edits.extend((a, b))
        commands.append({"source": q, "image": qq, "local": a, "remote": b})
    assert commands
    assert len({e["pivot"] for e in edits}) == len(edits)
    for i, a in enumerate(edits):
        assert not a["new"] & gate
        for b in edits[i+1:]:
            assert not a["old"] & b["old"]
            assert a["pivot"] not in b["old"] and b["pivot"] not in a["old"]
            assert a["color"] == b["color"] or not a["new"] & b["new"]
    return {"kind": kind, "origin": origin, "gate": gate, "background": bg,
            "work": work, "layers": layers, "gaps": gaps,
            "edits": edits, "commands": commands}


def _render(shape, bg, gate, layers, edits):
    kept = {c: set(s) for c, s in layers.items()}
    for e in edits:
        kept[e["color"]].difference_update(e["old"])
    out = np.full(shape, bg, dtype=np.int8)
    for p in gate:
        out[p] = SYMMETRY
    for c, cells in kept.items():
        for p in cells:
            out[p] = c
    rays = {}
    for e in edits:
        for p in e["new"]:
            assert p not in gate
            assert p not in rays or rays[p] == e["color"]
            rays[p] = e["color"]
    for p, c in rays.items():
        out[p] = c
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair cables and broadcast each routing command using the input gate."""
    inp = _grid(input_grid)
    a = _analyse(inp)
    return _render(inp.shape, a["background"], a["gate"], a["layers"], a["edits"])


def _simple(walk):
    cells = set(walk)
    return (len(cells) == len(walk) and all(len(_neighbors(p, cells)) <= 2 for p in cells)
            and len(_neighbors(walk[0], cells)) == 1
            and len(_neighbors(walk[-1], cells)) == 1)


def _reach(p, d, shape):
    return (p[0] if d == (-1, 0) else shape[0]-1-p[0] if d == (1, 0)
            else p[1] if d == (0, -1) else shape[1]-1-p[1])


def _old_walk(rng, p, d, shape):
    vertices = [p]
    current_d = (-d[0], -d[1])
    bends = int(rng.choice(4, p=[.23, .35, .30, .12]))
    for _ in range(bends):
        room = _reach(vertices[-1], current_d, shape) - 1
        if room < 2:
            return None
        length = int(rng.integers(2, min(8, room) + 1))
        last = vertices[-1]
        vertices.append((last[0]+length*current_d[0], last[1]+length*current_d[1]))
        side = int(rng.choice((-1, 1)))
        current_d = (-side*current_d[1], side*current_d[0])
    room = _reach(vertices[-1], current_d, shape)
    if room < 1:
        return None
    last = vertices[-1]
    vertices.append((last[0]+room*current_d[0], last[1]+room*current_d[1]))
    walk = _polyline(vertices)
    return walk if _simple(walk) else None


def _route(rng, p, d, shape, blocked):
    q = _add(p, d)
    if p in blocked or q in blocked:
        return None
    for _ in range(45):
        old = _old_walk(rng, p, d, shape)
        if old is None or set(old) & blocked or q in old:
            continue
        side = int(rng.choice((-1, 1)))
        heading = (-side*d[1], side*d[0])
        vertices = [p]
        for _ in range(int(rng.integers(1, 4))):
            room = _reach(vertices[-1], heading, shape) - 1
            if room < 2:
                break
            length = int(rng.integers(2, min(6, room) + 1))
            last = vertices[-1]
            vertices.append((last[0]+length*heading[0], last[1]+length*heading[1]))
            sign = int(rng.choice((-1, 1)))
            heading = (-sign*heading[1], sign*heading[0])
        if len(vertices) < 2:
            continue
        retained = _polyline(vertices)
        full = list(reversed(old)) + retained[1:]
        cells = set(full)
        if (not _simple(full) or cells & blocked or q in cells
                or _neighbors(q, cells) != [p]):
            continue
        # A minority of routes have an unequal extra retained arm at the pivot.
        if rng.random() < .20:
            other = (-retained[1][0]+p[0], -retained[1][1]+p[1])
            arm = _ray(p, other, shape)[:int(rng.integers(2, 5))]
            if (len(arm) >= 2 and not set(arm) & (cells | blocked)
                    and all(not _neighbors(x, cells - {p}) for x in arm)):
                cells.update(arm)
        return cells, set(old[1:])
    return None


def _gate(rng, kind, shape, command_count=1):
    h, w = shape
    def coordinate(size):
        # Richer scenes need enough room on BOTH sides for the reflected
        # addresses. This changes only sampling, not the interpretation of T.
        if command_count == 1:
            low, high = 5, size - 5
        else:
            radius = 4 if command_count == 2 else 2
            low, high = max(5, size//2-radius), min(size-5, size//2+radius+1)
        return int(rng.integers(low, high))
    if kind == "P":
        origin = (coordinate(h), coordinate(w))
        return origin, {origin}
    if kind == "H":
        origin = (coordinate(h), int(rng.integers(1, w-3)))
        length = int(rng.integers(2, min(8, w-origin[1])+1))
        return origin, {(origin[0], origin[1]+i) for i in range(length)}
    origin = (int(rng.integers(1, h-3)), coordinate(w))
    length = int(rng.integers(2, min(8, h-origin[0])+1))
    return origin, {(origin[0]+i, origin[1]) for i in range(length)}


def _scene(rng, kind, command_count, pool, bg, *, witness=False, crossing=False,
           witness_colors=None):
    low = 15 if command_count == 1 else 22 if command_count == 2 else 27
    shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
    origin, gate = _gate(rng, kind, shape, command_count)
    blocked = _halo(gate, shape)
    layers, edits, qs = {}, [], []
    sign = int(rng.choice((-1, 1)))
    candidates = []
    for r in range(1, shape[0]-1):
        for c in range(1, shape[1]-1):
            p = (r, c)
            pp = _map(p, kind, origin)
            if (not _inside(pp, shape) or _boundary(pp, shape)
                    or abs(r-pp[0])+abs(c-pp[1]) < 6):
                continue
            coord = r-origin[0] if kind != "V" else c-origin[1]
            if coord*sign < 2:
                continue
            if witness and kind == "P" and abs(c-origin[1]) < 2:
                continue
            candidates.append(p)
    for index in range(command_count):
        placed = False
        rng.shuffle(candidates)
        for p in candidates[:100]:
            pp = _map(p, kind, origin)
            dirs = list(DIRS)
            rng.shuffle(dirs)
            if witness and kind in ("H", "V"):
                dirs = [d for d in dirs if (d[0] != 0 if kind == "H" else d[1] != 0)]
            for d in dirs:
                q = _add(p, d)
                qq = _map(q, kind, origin)
                dd = (qq[0]-pp[0], qq[1]-pp[1])
                if (not _inside(q, shape) or not _inside(qq, shape)
                        or q in blocked or qq in blocked or pp in blocked):
                    continue
                colors = (list(witness_colors) if witness_colors is not None else
                          list(map(int, rng.choice(pool, size=2, replace=False))))
                if not witness and rng.random() < .25:
                    colors[1] = colors[0]
                local = _route(rng, p, d, shape, blocked | {qq, pp})
                if local is None:
                    continue
                local_cells, local_old = local
                remote = _route(rng, pp, dd, shape,
                                blocked | _halo(local_cells | {q}, shape))
                if remote is None:
                    continue
                remote_cells, remote_old = remote
                trial = [dict(pivot=p, direction=d, color=colors[0], old=local_old,
                              new=set(_ray(p, d, shape))),
                         dict(pivot=pp, direction=dd, color=colors[1], old=remote_old,
                              new=set(_ray(pp, dd, shape)))]
                all_edits = edits + trial
                if witness:
                    # A bent old route refutes deleting only a straight ray.
                    def bent(e):
                        cells = e["old"] | {e["pivot"]}
                        return any(len(ns := _neighbors(x, cells)) == 2
                                   and (ns[0][0]+ns[1][0], ns[0][1]+ns[1][1]) != (2*x[0], 2*x[1])
                                   for x in cells)
                    if not any(bent(e) for e in trial):
                        continue
                if any(e["new"] & gate for e in all_edits):
                    continue
                if any(a["color"] != b["color"] and a["new"] & b["new"]
                       for i, a in enumerate(all_edits) for b in all_edits[i+1:]):
                    continue
                for color, cells in zip(colors, (local_cells, remote_cells)):
                    layers.setdefault(color, set()).update(cells)
                edits.extend(trial)
                qs.append(q)
                blocked.update(_halo(local_cells | remote_cells | {q, qq}, shape))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    # An unaddressed cable defeats "redirect everything" and color-based rules.
    # The first witness includes a transverse retained stroke under a new ray.
    if crossing:
        options = [(e, p) for e in edits for p in sorted(e["new"])
                   if abs(p[0]-e["pivot"][0])+abs(p[1]-e["pivot"][1]) >= 3]
        rng.shuffle(options)
        added = False
        for e, p in options:
            perp = (-e["direction"][1], e["direction"][0])
            left, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            cells = {(p[0]+i*perp[0], p[1]+i*perp[1]) for i in range(-left, right+1)}
            if all(_inside(x, shape) for x in cells) and not cells & blocked:
                color = int(rng.choice(sorted({f["color"] for f in edits} - {e["color"]})))
                layers.setdefault(color, set()).update(cells)
                blocked.update(_halo(cells, shape))
                added = True
                break
        if not added:
            return None
    else:
        added = False
        for _ in range(70):
            p = (int(rng.integers(2, shape[0]-2)), int(rng.integers(2, shape[1]-2)))
            d = DIRS[int(rng.integers(4))]
            route = _route(rng, p, d, shape, blocked)
            if route is not None:
                cells, _ = route
                color = int(rng.choice(sorted({e["color"] for e in edits})))
                layers.setdefault(color, set()).update(cells)
                blocked.update(_halo(cells, shape))
                added = True
                break
        if not added:
            return None
    inp = np.full(shape, bg, dtype=np.int8)
    for p in gate:
        inp[p] = SYMMETRY
    for color, cells in layers.items():
        for p in cells:
            inp[p] = color
    for q in qs:
        if inp[q] != bg:
            return None
        inp[q] = DIRECTION
    out = _render(shape, bg, gate, layers, edits)
    return inp, out, layers, edits


def _damage(rng, scene, *, witness=False):
    inp, out, layers, edits = scene
    original = inp.copy()
    bg = int(np.bincount(inp.ravel(), minlength=10).argmax())
    if bg in ANNOTATIONS or np.count_nonzero(inp == bg) <= inp.size // 2:
        return None
    try:
        a = _analyse(inp)
        assert a["layers"] == layers and np.array_equal(solve(inp), out)
    except AssertionError:
        return None
    removed = set().union(*(e["old"] for e in edits))
    retained = set().union(*layers.values()) - removed
    priorities = []
    if witness:
        priorities = [edits[1]["pivot"], min(removed), min(retained)]
    sites = sorted(set().union(*layers.values()))
    rng.shuffle(sites)
    sites = list(dict.fromkeys(priorities + sites))
    wanted = int(rng.integers(3 if witness else 1, 9))
    successes = 0
    for p in sites:
        prior = int(inp[p])
        inp[p] = REPAIR
        try:
            recovered = _analyse(inp)
            assert recovered["layers"] == layers
            assert np.array_equal(solve(inp), out)
        except AssertionError:
            inp[p] = prior
        else:
            successes += 1
            if successes >= wanted:
                break
    if successes == 0:
        return None
    sites = [p for p in sites if inp[p] == original[p]]
    rng.shuffle(sites)
    wanted = int(rng.integers(1 if witness else 0, 5))
    successes = 0
    for p in sites:
        if successes >= wanted:
            break
        c = int(inp[p])
        ns = _neighbors(p, layers[c])
        if len(ns) != 2 or (ns[0][0]+ns[1][0], ns[0][1]+ns[1][1]) != (2*p[0], 2*p[1]):
            continue
        inp[p] = bg
        try:
            recovered = _analyse(inp)
            assert recovered["layers"] == layers
            assert np.array_equal(solve(inp), out)
        except AssertionError:
            inp[p] = c
        else:
            successes += 1
    # Preserve the promised evidence, as measured from the returned input.
    if witness:
        marked = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
        if not (marked & removed and marked & retained and len(marked) >= 3):
            return None
    for color in layers:
        mask = out == color
        if np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]):
            return None
    if not any(int(inp[p]) in layers and int(out[p]) == bg for p in removed):
        return None
    return {"input": inp, "output": out}


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(DATA_COLORS)))
    # Annotation roles are fixed; ordinary colors can be background or cable.
    pool = palette[:int(rng.integers(2, 6))]
    backgrounds = [c for c in palette if c not in pool]
    steady_bg = bool(rng.integers(2))
    line_kind = str(rng.choice(("H", "V")))
    teaching_colors = list(map(int, rng.choice(pool, size=2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = split == "train" and i < 2
            crossing = split == "train" and i == 0
            kind = (line_kind if i == 0 else "P") if witness else str(rng.choice(("H", "V", "P")))
            commands = 1 if witness else (int(rng.integers(1, 3)) if split == "train"
                                          else int(rng.choice((2, 3), p=[.85, .15])))
            bg = backgrounds[0] if steady_bg else int(rng.choice(backgrounds))
            witness_colors = (teaching_colors if i == 0 else teaching_colors[::-1]) if witness else None
            for _ in range(100):
                scene = _scene(rng, kind, commands, pool, bg, witness=witness, crossing=crossing,
                               witness_colors=witness_colors)
                if scene is None:
                    continue
                pair = _damage(rng, scene, witness=witness)
                if pair is not None and _fingerprint(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError(f"bounded scene construction exhausted: seed={seed}, "
                                   f"counts={num_train}/{num_test}, {split}[{i}], "
                                   f"kind={kind}, commands={commands}")
            puzzle[split].append(pair)
            seen.add(_fingerprint(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    a = _analyse(inp)
    edits = a["edits"]
    kept = {c: s - set().union(*(e["old"] for e in edits if e["color"] == c))
            for c, s in a["layers"].items()}
    removed = set().union(*(e["old"] for e in edits))
    marked = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
    bends = []
    for e in edits:
        walk = e["walk"]
        headings = [(b[0]-p[0], b[1]-p[1]) for p, b in zip(walk, walk[1:])]
        bends.append(sum(d != f for d, f in zip(headings, headings[1:])))
    return {"analysis": a, "kind": a["kind"], "shape": inp.shape,
            "commands": len(a["commands"]), "payloads": len(a["layers"]),
            "components": sum(len(_components(s)) for s in a["layers"].values()),
            "bends": bends, "rays": [len(e["new"]) for e in edits],
            "deleted_markers": len(marked & removed),
            "retained_markers": len(marked - removed),
            "marked_pivots": sum(e["pivot"] in marked for e in edits),
            "gap_count": len(a["gaps"]),
            "ray_crossings": sum(len(e["new"] & s) for e in edits for c, s in kept.items()
                                 if c != e["color"]),
            "new_old_hits": sum(len(e["new"] & f["old"]) for e in edits for f in edits if e is not f),
            "unaddressed": sum(len(_components(s)) for s in a["layers"].values()) - len(edits),
            "different_remote_colors": sum(x["local"]["color"] != x["remote"]["color"] for x in a["commands"]),
            "changed_directions": sum(x["local"]["direction"] != x["remote"]["direction"] for x in a["commands"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen
            seen.add(key)
            f = _features(inp)
            a = f["analysis"]
            assert np.array_equal(solve(inp), out), "full-grid solution mismatch"
            assert np.array_equal(out[inp == SYMMETRY], inp[inp == SYMMETRY])
            assert not np.any(np.isin(out, [DIRECTION, REPAIR]))
            assert f["unaddressed"] >= 1
            assert f["retained_markers"] + f["deleted_markers"] >= 1
            # Omitting A: only the visibly marked local pivots get rerouted.
            local_only = _render(inp.shape, a["background"], a["gate"], a["layers"],
                                 [x["local"] for x in a["commands"]])
            assert not np.array_equal(local_only, out), "symmetry has no effect"
            # Omitting B: purely additive symmetry cannot erase a cable cell.
            assert any(inp[p] in a["layers"] and out[p] == a["background"]
                       for e in a["edits"] for p in e["old"]), "terminal deletion is hidden"
            # Simultaneous semantics do not depend on command enumeration.
            assert np.array_equal(_render(inp.shape, a["background"], a["gate"], a["layers"],
                                         list(reversed(a["edits"]))), out)
            if split == "train":
                training.append(f)
    assert "P" in {f["kind"] for f in training}
    assert {f["kind"] for f in training} & {"H", "V"}
    assert any(f["changed_directions"] and f["different_remote_colors"] for f in training)
    assert any(f["ray_crossings"] for f in training), "ray precedence needs a demonstration"
    assert any(f["deleted_markers"] and f["retained_markers"] and f["marked_pivots"] for f in training)
    local_colors = {x["local"]["color"] for f in training for x in f["analysis"]["commands"]}
    remote_colors = {x["remote"]["color"] for f in training for x in f["analysis"]["commands"]}
    assert local_colors & remote_colors, "a color must have both local and remote roles"
    assert any(max(f["bends"]) > 0 for f in training), "bent deletion needs a demonstration"
    assert any(x["local"]["pivot"][0] != x["remote"]["pivot"][0]
               and x["local"]["pivot"][1] != x["remote"]["pivot"][1]
               for f in training if f["kind"] == "P" for x in f["analysis"]["commands"])


def self_test():
    """Bounded semantic, equivariance, mutation, count, and 42-seed checks."""
    features, signatures, normalized = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_fingerprint(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                features.append(_features(inp))
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                # The rule is invariant to rotations/reflections, even though
                # the finite operator need not span the paired pivot positions.
                transformed_in = np.rot90(np.fliplr(inp), seed % 4).copy()
                transformed_out = np.rot90(np.fliplr(out), seed % 4).copy()
                assert np.array_equal(solve(transformed_in), transformed_out)
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len({f["shape"] for f in features}) >= 50
    assert {f["kind"] for f in features} == {"H", "V", "P"}
    assert {f["commands"] for f in features} >= {1, 2, 3}
    assert {b for f in features for b in f["bends"]} >= {0, 1, 2, 3}
    assert len({f["components"] for f in features}) >= 3
    assert len({f["payloads"] for f in features}) >= 3
    for key in ("deleted_markers", "retained_markers", "marked_pivots", "gap_count",
                "ray_crossings", "new_old_hits", "different_remote_colors", "changed_directions"):
        assert sum(f[key] for f in features) > 0, key
    assert min(n for f in features for n in f["rays"]) == 1
    assert max(n for f in features for n in f["rays"]) >= 20
    p = generate_puzzle(123)
    repeated = generate_puzzle(123)
    assert all(np.array_equal(p[s][i][k], repeated[s][i][k])
               for s in p for i in range(len(p[s])) for k in ("input", "output"))
    # Exact count boundaries and type errors. The external verifier exercises
    # every legal count combination as well.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(500+nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Mutations cover preservation, remote color, deletion, retained repair,
    # pivot repair, and ray precedence, rather than just array structure.
    original = generate_puzzle(0)
    targeted = set()
    for split in ("train", "test"):
        for i, pair in enumerate(original[split]):
            inp, out = pair["input"], pair["output"]
            a = _analyse(inp)
            removed = set().union(*(e["old"] for e in a["edits"]))
            marks = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
            sites = {"gate": min(a["gate"]), "remote pivot": a["commands"][0]["remote"]["pivot"],
                     "remote ray": a["commands"][0]["image"],
                     "deletion": next(p for p in sorted(removed) if out[p] == a["background"])}
            if marks - removed:
                sites["retained repair"] = min(marks - removed)
            for e in a["edits"]:
                if e["pivot"] in marks:
                    sites["marked pivot"] = e["pivot"]
                for c, cells in a["layers"].items():
                    kept = cells - set().union(*(f["old"] for f in a["edits"] if f["color"] == c))
                    if c != e["color"] and kept & e["new"]:
                        sites["ray crossing"] = min(kept & e["new"])
            for name, point in sites.items():
                if name in targeted:
                    continue
                bad = copy.deepcopy(original)
                bad[split][i]["output"][point] = (int(out[point]) + 1) % 10
                try:
                    validate_generated_puzzle(bad)
                except (AssertionError, ValueError):
                    targeted.add(name)
                else:
                    raise AssertionError("corrupt output accepted")
    assert targeted >= {"gate", "remote pivot", "remote ray", "deletion", "retained repair",
                        "marked pivot", "ray crossing"}
    return {"seeds": 42, "pairs": len(features), "distinct_structures": len(normalized),
            "grid_shapes": len({f["shape"] for f in features}),
            "command_counts": sorted({f["commands"] for f in features}),
            "old_branch_bends": sorted({b for f in features for b in f["bends"]}),
            "note": "Author-owned checks; not independent semantic approval."}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
