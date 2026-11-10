"""Invisible zippers: sparse virtual rails select and shape unwound threads.

Only standard library and NumPy are used. No parent code or data is loaded.
The reader reconstructs geometry from the supplied grid, never a sampled plan.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((g >= 0) & (g <= 9))), "ARC colors required")
    return g.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _segment(a, b):
    _require((a[0] == b[0]) != (a[1] == b[1]), "nonzero orthogonal segment required")
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    n = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return [(a[0] + i*d[0], a[1] + i*d[1]) for i in range(n + 1)]


def _polyline(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_segment(a, b)[1:])
    return result


def _normalise(inp):
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if np.all(g[:, 0] == token) and np.count_nonzero(g == token) == g.shape[0]:
            candidates.append((g.copy(), k, token))
    _require(len(candidates) == 1, "one unique full-edge spool strip required")
    g, rotation, token = candidates[0]
    _require(g.shape[1] >= 4, "no room for rails")
    bg = int(g[0, 1])
    _require(bg != token and bool(np.all(g[:, 1] == bg)), "clear approach column required")
    return g, rotation, token, bg


def _complete(g, bg, token):
    """Temporary geometry only: connect consecutive same-color row/column knots.

    Orthogonality is a fixed family convention, not an inferred color command.
    Different colors cannot claim the same empty cell. Existing pins also block.
    Singleton colors naturally contribute no interpolated links.
    """
    virtual = g.copy()
    claims = {}
    for color in sorted(set(map(int, np.unique(g))) - {bg, token}):
        points = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        by_row, by_col = {}, {}
        for r, c in points:
            by_row.setdefault(r, []).append(c)
            by_col.setdefault(c, []).append(r)
        links = []
        for r, cols in by_row.items():
            cols.sort()
            links.extend(((r, a), (r, b)) for a, b in zip(cols, cols[1:]))
        for c, rows in by_col.items():
            rows.sort()
            links.extend(((a, c), (b, c)) for a, b in zip(rows, rows[1:]))
        for a, b in links:
            for p in _segment(a, b):
                _require(int(g[p]) in (bg, color), "interpolation crosses another object")
                _require(p not in claims or claims[p] == color, "virtual rail collision")
                claims[p] = color
    for p, color in claims.items():
        virtual[p] = color
    _require(bool(np.all(virtual[:, 1] == bg)), "interpolation blocks the approach")
    return virtual


def _paired(g, p, d, bg, token, matching=True):
    n = (-d[1], d[0])
    a, b = _add(p, n), _add(p, (-n[0], -n[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return False
    ca, cb = int(g[a]), int(g[b])
    return ca not in (bg, token) and cb not in (bg, token) and (ca == cb or not matching)


def _trace(g, row, bg, token, matching=True):
    p, d = (row, 1), (0, 1)
    path, directions, seen, turns = [p], [], set(), []
    while True:
        state = p, d
        _require(state not in seen and len(seen) <= g.size*4, "cyclic route")
        seen.add(state)
        q = _add(p, d)
        if not _inside(g, q):
            break
        if int(g[q]) == bg:
            p = q
            path.append(p)
            directions.append(d)
            continue
        options = []
        for turn in ((-d[1], d[0]), (d[1], -d[0])):
            q = _add(p, turn)
            if (_inside(g, q) and int(g[q]) == bg
                    and _paired(g, q, turn, bg, token, matching)):
                options.append(turn)
        _require(len(options) <= 1, "two supported turns")
        if not options:
            break
        turns.append((p, d, options[0]))
        d = options[0]
    _require(len(path) == len(set(path)), "route revisits a cell")
    _require(len(path) >= 3 and path[-1][1] != 0, "degenerate thread")
    return {"start": (row, 0), "path": path, "turns": turns,
            "directions": directions, "end": path[-1], "end_direction": d,
            "open": not _inside(g, _add(path[-1], d))}


def _analyse(inp, interpolate=True):
    original = _grid(inp)
    g, rotation, token, bg = _normalise(original)
    virtual = _complete(g, bg, token) if interpolate else g.copy()
    routes, mismatches = [], []
    for row in range(1, g.shape[0] - 1):
        for col in range(1, g.shape[1]):
            if int(virtual[row, col]) != bg:
                break
            a, b = int(virtual[row - 1, col]), int(virtual[row + 1, col])
            if a not in (bg, token) and b not in (bg, token):
                if a == b:
                    routes.append(_trace(virtual, row, bg, token))
                else:
                    mismatches.append((row, col, a, b))
                # Nearer mismatches cannot be skipped to select a farther pair.
                break
    cells = [p for route in routes for p in route["path"]]
    _require(len(cells) == len(set(cells)), "threads must be disjoint")
    _require(all(int(g[p]) == bg for p in cells), "thread covers an input object")
    return {"grid": g, "virtual": virtual, "rotation": rotation,
            "token": token, "bg": bg, "routes": routes, "mismatches": mismatches}


def _answer(info, routes=None):
    out = info["grid"].copy()
    for route in info["routes"] if routes is None else routes:
        out[route["start"]] = info["bg"]
        for p in route["path"]:
            out[p] = info["token"]
    return np.rot90(out, -info["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Unwind every selected spool using only its visible input geometry."""
    return _answer(_analyse(input_grid))


def _signature(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _geometry_proposal(rng, h, w, bends, closed):
    """Two translated stair rails, never a parent puzzle or a stored template.

    A diagonal displacement d has normal component one on both straight axes.
    Thus S and S+2d enclose the same bent polyline S+d. Segment lengths >=3
    keep separate rail landmarks from inventing cross-links under interpolation.
    """
    side = int(rng.choice([-1, 1]))
    d = (side, 1)
    lengths = [int(rng.integers(3, 8)) for _ in range(bends + 1)]
    points = [(0, 0)]
    for i, n in enumerate(lengths):
        direction = (0, 1) if i % 2 == 0 else (-side, 0)
        p = points[-1]
        points.append((p[0] + n*direction[0], p[1] + n*direction[1]))
    twin = [(r + 2*d[0], c + 2*d[1]) for r, c in points]
    lo = min(r for r, c in points + twin)
    hi = max(r for r, c in points + twin)
    max_col = max(c for r, c in points + twin)
    if hi - lo >= h or max_col + 3 >= w:
        return None
    shift_r = int(rng.integers(-lo, h - hi))
    shift_c = int(rng.integers(2, min(5, w - 2 - max_col) + 1))
    points = [(r + shift_r, c + shift_c) for r, c in points]
    twin = [(r + shift_r, c + shift_c) for r, c in twin]
    middle = [(r + d[0], c + d[1]) for r, c in points]
    path = [(middle[0][0], c) for c in range(1, middle[0][1] + 1)]
    path.extend(_polyline(middle)[1:])
    final_d = (0, 1) if bends % 2 == 0 else (-side, 0)
    cap = None
    if closed:
        tail = int(rng.integers(0, 4))
        cap = (middle[-1][0] + (tail + 1)*final_d[0],
               middle[-1][1] + (tail + 1)*final_d[1])
        if not (0 <= cap[0] < h and 2 <= cap[1] < w):
            return None
        for _ in range(tail):
            path.append(_add(path[-1], final_d))
    else:
        while True:
            q = _add(path[-1], final_d)
            if not (0 <= q[0] < h and 0 <= q[1] < w):
                break
            path.append(q)
    wall_a, wall_b = set(_polyline(points)), set(_polyline(twin))
    path_set = set(path)
    if wall_a & wall_b or path_set & (wall_a | wall_b):
        return None
    if cap is not None and cap in (wall_a | wall_b | path_set):
        return None
    return {"a": points, "b": twin, "wall_a": wall_a, "wall_b": wall_b,
            "path": path, "cap": cap}


def _build_scene(rng, plans, rotation, colors):
    """Bounded rejection packing with no reserved row bands or shape library."""
    token, bg = colors[:2]
    count = len(plans)
    for attempt in range(100):
        # Relax packing pressure on later attempts without changing the rule.
        low_h = 16 if count == 2 else 22
        h = int(rng.integers(low_h, 31)) if attempt < 35 else 30
        w = int(rng.integers(18, 31)) if attempt < 35 else 30
        inp = np.full((h, w), bg, dtype=np.int8)
        full = inp.copy()
        inp[:, 0] = full[:, 0] = token
        used_paths, used_walls = set(), set()
        intended, color_index, placed = [], 2, True
        # Largest staircase first, so straight channels fill the remaining space.
        for bends, matched, closed in sorted(plans, key=lambda x: -x[0]):
            ca = colors[color_index]
            color_index += 1
            if matched:
                cb = ca
            else:
                cb = colors[color_index]
                color_index += 1
            cap_color = None
            if closed:
                cap_color = colors[color_index]
                color_index += 1
            _require(color_index <= 10, "palette budget")
            accepted = None
            for _ in range(100):
                proposal = _geometry_proposal(rng, h, w, bends, closed)
                if proposal is None:
                    continue
                path_cells = set(proposal["path"])
                walls = proposal["wall_a"] | proposal["wall_b"]
                if proposal["cap"] is not None:
                    walls.add(proposal["cap"])
                if (path_cells & (used_paths | used_walls)
                        or walls & (used_paths | used_walls)):
                    continue
                accepted = proposal
                break
            if accepted is None:
                placed = False
                break
            for key, wall_key, color in (("a", "wall_a", ca), ("b", "wall_b", cb)):
                for p in accepted[key]:
                    inp[p] = color
                for p in accepted[wall_key]:
                    full[p] = color
            if accepted["cap"] is not None:
                inp[accepted["cap"]] = full[accepted["cap"]] = cap_color
            used_paths.update(path_cells)
            used_walls.update(walls)
            if matched:
                intended.append({"start": (accepted["path"][0][0], 0),
                                 "path": accepted["path"]})
        if not placed:
            continue
        try:
            analysis = _analyse(inp)
            actual = {r["start"]: r["path"] for r in analysis["routes"]}
            wanted = {r["start"]: r["path"] for r in intended}
            _require(actual == wanted, "visible routing differs from sampled geometry")
            _require(np.array_equal(analysis["virtual"], full), "unintended interpolation link")
            _require(analysis["mismatches"], "inactive opening missing")
            _require(_no_match_difference(analysis), "color matching must change the result")
            # Expected answer is rendered from the independently retained plan.
            # solve() has no access to this plan, and validator re-reads the input.
            out = inp.copy()
            for route in intended:
                out[route["start"]] = bg
                for p in route["path"]:
                    out[p] = token
            return {"input": np.rot90(inp, rotation).copy(),
                    "output": np.rot90(out, rotation).copy()}
        except AssertionError:
            continue
    raise RuntimeError("bounded scene packing exhausted")


def _no_match_difference(info):
    """Counterfactual witness: a geometrically routable mismatched entrance.

    This is an ablation of the same reader, not an independent reference solver.
    One extra thread suffices; no arbitrary priority among conflicting routes.
    """
    for row, col, ca, cb in info["mismatches"]:
        try:
            route = _trace(info["virtual"], row, info["bg"], info["token"], matching=False)
        except AssertionError:
            continue
        if len(route["path"]) >= 4:
            return True
    return False


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_colors = tuple(map(int, rng.permutation(10)))
    palette_mode = int(rng.integers(3))
    base_rotation = int(rng.integers(4))
    reflect = bool(rng.integers(2))
    orientation_mode = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    # Both minimal demonstrations are needed. Their positions are shuffled.
    demonstrations = [0, 1]
    rng.shuffle(demonstrations)
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(int(count)):
            if split == "train" and index < 2:
                demo = demonstrations[index]
                if demo == 0:
                    plans = [(2, True, False), (0, False, True)]
                else:
                    plans = [(1, True, True), (0, True, False), (0, False, False)]
            elif split == "train":
                n = int(rng.integers(2, 5))
                plans = [(int(rng.integers(3)), True, bool(rng.integers(2))),
                         (int(rng.integers(2)), False, False)]
                plans += [(int(rng.integers(2)), True, False) for _ in range(n - 2)]
            else:
                n = int(rng.integers(2, 5))
                plans = [(int(rng.integers(3, 5)), True, bool(rng.integers(2))),
                         (int(rng.integers(2)), False, False)]
                plans += [(int(rng.integers(2)), True, False) for _ in range(n - 2)]
            colors = base_colors
            if palette_mode == 1:
                colors = (base_colors[0],) + tuple(map(int, rng.permutation(base_colors[1:])))
            elif palette_mode == 2:
                colors = tuple(map(int, rng.permutation(10)))
            rotation = base_rotation
            if orientation_mode == 1:
                rotation = (rotation + index + (split == "test")) % 4
            elif orientation_mode == 2 and split == "test":
                rotation = (rotation + int(rng.integers(1, 4))) % 4
            for _ in range(12):
                pair = _build_scene(rng, plans, rotation, colors)
                if reflect:
                    pair = {k: np.fliplr(v).copy() for k, v in pair.items()}
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(info):
    routes, g = info["routes"], info["grid"]
    return {"shape": tuple(g.shape), "rotation": info["rotation"],
            "token": info["token"], "bg": info["bg"],
            "active": len(routes), "mismatches": len(info["mismatches"]),
            "palette": len(np.unique(g)),
            "turns": tuple(len(r["turns"]) for r in routes),
            "lengths": tuple(len(r["path"]) for r in routes),
            "open": sum(r["open"] for r in routes),
            "closed": sum(not r["open"] for r in routes),
            "virtual_added": int(np.count_nonzero(g != info["virtual"])),
            "transverse": tuple(d[0] for r in routes for p, old, d in r["turns"] if d[0])}


def _check_rail_structure(info):
    """Validate actual completed chains, including their visible translation."""
    g, virtual = info["grid"], info["virtual"]
    for color in sorted(set(map(int, np.unique(g))) - {info["bg"], info["token"]}):
        knots = set(map(tuple, np.argwhere(g == color).tolist()))
        if len(knots) == 1:
            continue  # A stopping pin, with no implicit extension.
        remaining = set(map(tuple, np.argwhere(virtual == color).tolist()))
        components = []
        while remaining:
            start = min(remaining)
            component, stack = {start}, [start]
            remaining.remove(start)
            while stack:
                p = stack.pop()
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = _add(p, d)
                    if q in remaining:
                        remaining.remove(q)
                        component.add(q)
                        stack.append(q)
            degrees = {p: sum(_add(p, d) in component for d in
                             ((1, 0), (-1, 0), (0, 1), (0, -1))) for p in component}
            _require(len(component) >= 4 and all(v in (1, 2) for v in degrees.values()),
                     "rail must be a simple nontrivial chain")
            endpoints = {p for p, degree in degrees.items() if degree == 1}
            _require(len(endpoints) == 2 and endpoints <= knots, "rail endpoints must be knots")
            for p, degree in degrees.items():
                neighbors = [_add(p, d) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
                             if _add(p, d) in component]
                if degree == 2 and neighbors[0][0] != neighbors[1][0] and neighbors[0][1] != neighbors[1][1]:
                    _require(p in knots, "every rail corner must be visible")
            components.append(component)
        _require(len(components) in (1, 2), "one chain or one matched pair per identity")
        if len(components) == 2:
            a, b = components
            pa, pb = min(a), min(b)
            offset = pb[0] - pa[0], pb[1] - pa[1]
            _require(abs(offset[0]) == abs(offset[1]) == 2,
                     "paired rails must be diagonally displaced by two cells")
            _require({_add(p, offset) for p in a} == b, "paired rails must be translated copies")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas preserved")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyse(inp)
            _check_rail_structure(info)
            _require(info["routes"] and info["mismatches"], "both entrance types required")
            _require(np.array_equal(_answer(info), out), "wrong thread output")
            a, b = info["grid"], np.rot90(out, info["rotation"])
            guides = (a != info["bg"]) & (a != info["token"])
            _require(np.array_equal(a[guides], b[guides]), "input knots and pins must survive")
            expected_changes = sum(len(r["path"]) + 1 for r in info["routes"])
            _require(np.count_nonzero(a != b) == expected_changes, "unrelated cells changed")
            expected_tokens = a.shape[0] + sum(len(r["path"]) - 1 for r in info["routes"])
            _require(np.count_nonzero(b == info["token"]) == expected_tokens, "thread length accounting")
            for row, col, ca, cb in info["mismatches"]:
                _require(b[row, 0] == info["token"], "mismatched spool moved")
            raw = _analyse(inp, interpolate=False)
            _require(not np.array_equal(_answer(raw), out), "sparse completion must affect the answer")
            _require(_no_match_difference(info), "guide equality must affect the answer")
            # Materializing the virtual walls is a tempting but incorrect shortcut.
            _require(bool(np.any((a != info["virtual"]) & (b == a))), "virtual rails must remain invisible")
            records[split].append(_measure(info))
    train = records["train"]
    train_turns = {n for r in train for n in r["turns"]}
    _require({0, 1, 2} <= train_turns, "demonstrate straight, elbow, repeated turns")
    _require(any(r["open"] for r in train) and any(r["closed"] for r in train), "both stops demonstrated")
    _require(len({r["active"] for r in train}) >= 2, "demonstrate variable mover counts")
    _require(all(max(r["turns"]) <= 2 for r in train), "training complexity regime")
    _require(all(max(r["turns"]) >= 3 for r in records["test"]), "test must compose more turns")
    return None


def self_test():
    """Streaming 42-seed checks: retain hashes/feature sets, not puzzle payloads."""
    shapes, structures, identities, turns, active, palettes = set(), set(), set(), set(), set(), set()
    rotations, transverse, tokens, backgrounds = set(), set(), set(), set()
    lengths, additions, closed, opened, pair_count = [], [], 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                _require(np.array_equal(solve(inp.copy()), out), "input-only solve")
                info = _analyse(inp)
                m = _measure(info)
                shapes.add(m["shape"])
                rotations.add(m["rotation"])
                tokens.add(m["token"])
                backgrounds.add(m["bg"])
                turns.update(m["turns"])
                active.add(m["active"])
                palettes.add(m["palette"])
                transverse.update(m["transverse"])
                lengths.extend(m["lengths"])
                additions.append(m["virtual_added"])
                closed += m["closed"]
                opened += m["open"]
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape) + norm).hexdigest())
                pair_count += 1
        identities.add(digest.hexdigest())
    _require(len(identities) == 42, "seed diversity")
    _require(len(shapes) >= 30 and len(structures) >= 150, "structural variation")
    _require(turns == set(range(5)), "zero through four turns")
    _require(active == {1, 2, 3}, "one through three active spools")
    _require(rotations == set(range(4)) and transverse == {-1, 1}, "directions")
    _require(tokens == set(range(10)) and backgrounds == set(range(10)), "literal role variation")
    _require(len(palettes) >= 4 and min(additions) >= 8, "palette and interpolation variation")
    _require(closed > 0 and opened > 0 and max(lengths) >= 30, "terminal/length variation")
    a, b = generate_puzzle(20261022, 3, 2), generate_puzzle(20261022, 3, 2)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "determinism")
    # The convention commutes with geometry and color-role renaming.
    example = a["test"][0]
    permutation = np.random.default_rng(17).permutation(10)
    _require(np.array_equal(solve(permutation[example["input"]]),
                            permutation[example["output"]]), "palette equivariance")
    for rotation in range(4):
        for reflect in (False, True):
            inp, out = (np.rot90(example[k], rotation) for k in ("input", "output"))
            if reflect:
                inp, out = np.fliplr(inp), np.fliplr(out)
            _require(np.array_equal(solve(inp), out), "dihedral equivariance")
    for n in range(2, 11):
        for q in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(2000 + 10*n + q, n, q)
            _require(len(p["train"]) == n and len(p["test"]) == q, "explicit counts")
    for n, q in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, n, q)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Missing one thread cell and completing an invisible rail must both fail.
    for mutation in range(2):
        bad = generate_puzzle(0, 2, 1)
        pair = bad["test"][0]
        if mutation == 0:
            p = tuple(map(int, np.argwhere(pair["input"] != pair["output"])[0]))
            pair["output"][p] = pair["input"][p]
        else:
            info = _analyse(pair["input"])
            p = tuple(map(int, np.argwhere(info["grid"] != info["virtual"])[0]))
            normalised_output = np.rot90(pair["output"], info["rotation"]).copy()
            normalised_output[p] = info["virtual"][p]
            pair["output"] = np.rot90(normalised_output, -info["rotation"]).copy()
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong completion or thread accepted")
    return {"valid": True, "seeds": 42, "pairs": pair_count,
            "distinct_puzzles": len(identities), "shapes": len(shapes),
            "color_normalized_structures": len(structures), "turns": sorted(turns),
            "active_counts": sorted(active), "palette_sizes": sorted(palettes),
            "thread_length_range": [min(lengths), max(lengths)],
            "virtual_added_range": [min(additions), max(additions)],
            "open_threads": opened, "closed_threads": closed,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
