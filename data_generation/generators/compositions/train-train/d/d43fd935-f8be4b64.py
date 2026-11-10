"""Rooted relay circuits: alignment starts power, cross arms pass it onward.

Only the standard library and NumPy are used. No parent generator is imported.
The parser, static visibility graph, and final rendering all read the input.
"""

from collections import deque
import hashlib
import json

import numpy as np


DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape),
             "grid dimensions must lie in 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must lie in 0..9")
    return a.astype(np.int8, copy=False)


def _feet(center):
    r, c = center
    return {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}


def _arms(center):
    return _feet(center) - {center}


def _anchor_cells(anchor):
    r, c = anchor
    return {(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)}


def _alignment(center, anchor):
    r, c = center
    ar, ac = anchor
    if r in (ar, ar + 1):
        return (0, 1) if c < ac else (0, -1)
    if c in (ac, ac + 1):
        return (1, 0) if r < ar else (-1, 0)
    return None


def _parse(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > g.size // 2, "unique majority background required")
    squares = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(g[r, c])
            if color != bg and np.all(g[r:r + 2, c:c + 2] == color):
                squares.append((r, c, color))
    _require(len(squares) == 1, "one solid non-background 2x2 station required")
    ar, ac, station = squares[0]
    anchor = (ar, ac)
    fixed = _anchor_cells(anchor)
    _require(counts[station] == 4, "station color is exclusive to its square")
    candidates = []
    for arm in (int(v) for v in np.unique(g) if int(v) not in (bg, station)):
        positions = {tuple(map(int, p)) for p in np.argwhere(g == arm)}
        centers = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                   if int(g[r, c]) not in (arm, station)
                   and _arms((r, c)) <= positions]
        if not centers or len(positions) != 4 * len(centers):
            continue
        if set().union(*(_arms(p) for p in centers)) != positions:
            continue
        if len({r for r, _ in centers}) != len(centers):
            continue
        if len({c for _, c in centers}) != len(centers):
            continue
        allowed, good = set(fixed), True
        for p in centers:
            foot = _feet(p)
            if foot & allowed:
                good = False
                break
            allowed.update(foot)
        if good and all((int(g[r, c]) == bg or (r, c) in allowed)
                        for r in range(h) for c in range(w)):
            candidates.append((arm, tuple(sorted(centers))))
    _require(len(candidates) == 1, "cross arm color must be geometrically unique")
    arm, centers = candidates[0]
    for p in centers:
        _require(min(abs(x - r) + abs(y - c)
                     for x, y in _feet(p) for r, c in fixed) >= 2,
                 "station and crosses need a background gap")
    return g, bg, station, arm, anchor, centers


def _network(shape, anchor, centers):
    """Trace on original structure; every first arm has an owning relay."""
    h, w = shape
    station = _anchor_cells(anchor)
    owner = {p: i for i, center in enumerate(centers) for p in _arms(center)}
    routes, edges, roots = {}, [set() for _ in centers], set()
    for i, (r, c) in enumerate(centers):
        for dr, dc in DIRECTIONS:
            rr, cc = r + 2 * dr, c + 2 * dc
            path, hit = [], None
            while 0 <= rr < h and 0 <= cc < w:
                p = (rr, cc)
                if p in station:
                    hit = "station"
                    break
                if p in owner:
                    hit = owner[p]
                    edges[i].add(hit)
                    break
                path.append(p)
                rr, cc = rr + dr, cc + dc
            routes[i, dr, dc] = (tuple(path), hit)
            if hit == "station" and _alignment((r, c), anchor) == (dr, dc):
                roots.add(i)
    return routes, edges, roots


def _reachable(roots, edges, banned=()):
    banned = set(banned)
    depth = {i: 0 for i in roots if i not in banned}
    queue = deque(sorted(depth))
    while queue:
        i = queue.popleft()
        for j in sorted(edges[i]):
            if j not in depth and j not in banned:
                depth[j] = depth[i] + 1
                queue.append(j)
    return depth


def _render(g, centers, routes, active, *, vertical_first=False,
            skip_background=None, through=False, anchor=None):
    out = g.copy()
    horizontal, vertical = {}, {}
    protected = set().union(*(_feet(p) for p in centers))
    if anchor is not None:
        protected |= _anchor_cells(anchor)
    directions = DIRECTIONS[2:] + DIRECTIONS[:2] if vertical_first else DIRECTIONS
    for dr, dc in directions:
        layer = horizontal if dr == 0 else vertical
        for i in sorted(active):
            r, c = centers[i]
            value = int(g[r, c])
            if skip_background is not None and value == skip_background:
                continue
            path = routes[i, dr, dc][0]
            if through:
                path = []
                rr, cc = r + 2 * dr, c + 2 * dc
                while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    if (rr, cc) not in protected:
                        path.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
            for p in path:
                out[p] = value
                layer[p] = i
    return out, horizontal, vertical


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family input, independent of generator or seed."""
    g, _, _, _, anchor, centers = _parse(input_grid)
    routes, edges, roots = _network(g.shape, anchor, centers)
    depth = _reachable(roots, edges)
    return _render(g, centers, routes, depth)[0]


def _analyze(input_grid):
    g, bg, station, arm, anchor, centers = _parse(input_grid)
    routes, edges, roots = _network(g.shape, anchor, centers)
    depth = _reachable(roots, edges)
    active, dormant = set(depth), set(range(len(centers))) - set(depth)
    out, horizontal, vertical = _render(g, centers, routes, active)
    colored_crossings, erasures = [], []
    for p in horizontal.keys() & vertical.keys():
        hi, vi = horizontal[p], vertical[p]
        hv, vv = int(g[centers[hi]]), int(g[centers[vi]])
        if hv != vv and hv != bg:
            if vv == bg:
                erasures.append((p, hi, vi))
            else:
                colored_crossings.append((p, hi, vi))
    bg_nodes = {i for i in active if int(g[centers[i]]) == bg}
    no_blank = _reachable(roots, edges, bg_nodes)
    no_blank_descendants = active - set(no_blank) - bg_nodes
    stats = {
        "active": active, "dormant": dormant, "roots": roots, "depth": depth,
        "max_depth": max(depth.values(), default=-1),
        "colored_crossings": colored_crossings, "erasures": erasures,
        "blank_descendants": no_blank_descendants,
        "dormant_edges": sum(j in dormant for i in dormant for j in edges[i]),
        "alignment_ablation": not np.array_equal(
            out, _render(g, centers, routes, range(len(centers)))[0]),
        "relay_ablation": not np.array_equal(
            out, _render(g, centers, routes, roots)[0]),
        "one_pass_ablation": not np.array_equal(
            out, _render(g, centers, routes, {i for i in active if depth[i] <= 1})[0]),
        "two_pass_ablation": not np.array_equal(
            out, _render(g, centers, routes, {i for i in active if depth[i] <= 2})[0]),
        "stop_ablation": not np.array_equal(
            out, _render(g, centers, routes, active, through=True, anchor=anchor)[0]),
    }
    return {"grid": g, "background": bg, "station_color": station, "arm_color": arm,
            "anchor": anchor, "centers": centers, "routes": routes, "edges": edges,
            "output": out, **stats}


def _fits(p, centers, anchor, shape):
    r, c = p
    if not (1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1):
        return False
    if any(r == x or c == y or abs(r - x) + abs(c - y) < 5 for x, y in centers):
        return False
    return min(abs(x - rr) + abs(y - cc)
               for x, y in _feet(p) for rr, cc in _anchor_cells(anchor)) >= 2


def _attached_candidates(parent, centers, anchor, shape):
    r, c = parent
    possible = ([(r + d, x) for d in (-1, 1) for x in range(1, shape[1] - 1)]
                + [(x, c + d) for d in (-1, 1) for x in range(1, shape[0] - 1)])
    return [p for p in possible if _fits(p, centers, anchor, shape)]


def _sample_geometry(rng, total, cluster, prefer_deep):
    h = int(rng.integers(14, 31))
    w = h if rng.random() < .25 else int(rng.integers(14, 31))
    shape = (h, w)
    anchor = (int(rng.integers(h - 1)), int(rng.integers(w - 1)))
    places = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    roots = [p for p in places if _alignment(p, anchor) is not None
             and _fits(p, [], anchor, shape)]
    if not roots:
        return None
    centers = [roots[int(rng.integers(len(roots)))]]
    dormant_target = int(rng.integers(2, 4)) if cluster else int(rng.integers(1, 3))
    active_target = max(1, total - dormant_target)
    for _ in range(active_target - 1):
        parent = centers[-1] if prefer_deep and rng.random() < .8 else centers[int(rng.integers(len(centers)))]
        candidates = _attached_candidates(parent, centers, anchor, shape)
        if not candidates:
            return None
        centers.append(candidates[int(rng.integers(len(candidates)))])
    for k in range(total - len(centers)):
        if cluster and k == 1:
            candidates = _attached_candidates(centers[-1], centers, anchor, shape)
        else:
            candidates = [p for p in places if _fits(p, centers, anchor, shape)]
        if not candidates:
            return None
        centers.append(candidates[int(rng.integers(len(candidates)))])
    routes, edges, roots = _network(shape, anchor, centers)
    depth = _reachable(roots, edges)
    return shape, anchor, tuple(centers), routes, edges, roots, depth


def _build(shape, anchor, centers, roles, signals):
    bg, station, arm = roles
    g = np.full(shape, bg, dtype=np.int8)
    for p in _anchor_cells(anchor):
        g[p] = station
    for center, value in zip(centers, signals):
        for p in _arms(center):
            g[p] = arm
        g[center] = value
    return g


def _make_pair(rng, roles, kind, shift, split):
    bg, _, _ = roles
    pool = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    mandatory = kind in ("colored", "blank")
    is_cluster = split == "test" and shift == "islands"
    for _ in range(2200):
        if kind == "simple":
            total = int(rng.integers(2, 6))
        elif split == "test" and shift == "depth":
            total = int(rng.integers(6, 10))
        else:
            total = int(rng.integers(5 if mandatory else 4, 10))
        geometry = _sample_geometry(rng, total, is_cluster,
                                    prefer_deep=mandatory or shift == "depth")
        if geometry is None:
            continue
        shape, anchor, centers, routes, edges, roots, depth = geometry
        active = set(depth)
        dormant = set(range(total)) - active
        max_depth = max(depth.values(), default=-1)
        if not roots or not dormant or len(active) < (3 if mandatory else 1):
            continue
        if mandatory and max_depth < 2:
            continue
        if split == "train" and shift == "depth" and max_depth > 2:
            continue
        if split == "test" and shift == "depth" and max_depth < 3:
            continue
        dormant_edges = sum(j in dormant for i in dormant for j in edges[i])
        if shift == "islands" and ((split == "train" and dormant_edges)
                                    or (split == "test" and not dormant_edges)):
            continue
        if kind == "simple" and len(active) > 2:
            continue
        palette_size = int(rng.integers(2, 5))
        signals = [pool[int(rng.integers(palette_size))] for _ in centers]
        g = _build(shape, anchor, centers, roles, signals)
        _, horizontal, vertical = _render(g, centers, routes, active)
        crossings = [(p, horizontal[p], vertical[p])
                     for p in horizontal.keys() & vertical.keys()
                     if horizontal[p] != vertical[p]]
        if mandatory and not crossings:
            continue
        if kind == "blank":
            choices = []
            for p, hi, vi in crossings:
                without = _reachable(roots, edges, (vi,))
                if active - set(without) - {vi}:
                    choices.append((p, hi, vi))
            if not choices:
                continue
            _, hi, vi = choices[int(rng.integers(len(choices)))]
            signals[vi] = bg
            signals[hi] = pool[0]
        elif kind == "colored":
            _, hi, vi = crossings[int(rng.integers(len(crossings)))]
            signals[hi], signals[vi] = pool[0], pool[1]
        elif kind != "simple" and rng.random() < .65:
            candidates = sorted(active)
            for i in rng.choice(candidates, size=min(len(candidates), int(rng.integers(1, 3))), replace=False):
                signals[int(i)] = bg
        g = _build(shape, anchor, centers, roles, signals)
        try:
            a = _analyze(g)
        except AssertionError:
            continue
        if not a["alignment_ablation"] or np.array_equal(a["output"], g):
            continue
        if len(active) > len(roots) and not a["relay_ablation"]:
            continue
        if mandatory and not a["stop_ablation"]:
            continue
        if kind == "colored" and not a["colored_crossings"]:
            continue
        if kind == "colored" and not a["one_pass_ablation"]:
            continue
        if kind == "blank" and (not a["erasures"] or not a["blank_descendants"]):
            continue
        if split == "test" and shift == "depth" and not a["two_pass_ablation"]:
            continue
        return {"input": g, "output": a["output"]}
    raise RuntimeError(f"could not construct {split}/{kind}/{shift} example")


def _check_counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = "depth" if rng.integers(2) == 0 else "islands"
    stable_roles = bool(rng.integers(2))
    roles = tuple(int(v) for v in rng.permutation(10)[:3])
    kinds = ["colored", "blank"]
    kinds += [("simple", "mixed", "mixed")[int(rng.integers(3))]
              for _ in range(num_train - 2)]
    rng.shuffle(kinds)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, schedule in (("train", kinds), ("test", ["mixed"] * num_test)):
        for kind in schedule:
            pair_roles = roles if stable_roles else tuple(int(v) for v in rng.permutation(10)[:3])
            for _ in range(10):
                pair = _make_pair(rng, pair_roles, kind, shift, split)
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "exactly train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "exactly input/output pair keys required")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape, "dimensions are preserved")
            _require(_key(g) not in seen, "inputs must be distinct")
            seen.add(_key(g))
            a = _analyze(g)
            _require(np.array_equal(a["output"], out), "wrong relay output")
            _require(not np.array_equal(g, out), "each example changes visibly")
            _require(a["roots"] and a["dormant"], "powered and dormant objects required")
            _require(a["alignment_ablation"], "station filtering must affect answer")
            _require(all(a["edges"][i] <= a["active"] for i in a["active"]),
                     "every arm reached by a powered relay must become powered")
            _require(not (a["roots"] & a["dormant"]),
                     "a visible center-to-station connection cannot remain dormant")
            for center in a["centers"]:
                for p in _feet(center):
                    _require(g[p] == out[p], "cross arms and centers must be protected")
            for p in _anchor_cells(a["anchor"]):
                _require(g[p] == out[p], "station must be preserved")
            if a["max_depth"] > 0:
                _require(a["relay_ablation"], "relay activation must affect answer")
            analyses[split].append(a)
    train, test = analyses["train"], analyses["test"]
    _require(any(a["max_depth"] >= 2 and a["one_pass_ablation"] and a["colored_crossings"]
                 and a["stop_ablation"] for a in train),
             "training needs a colored multi-hop and stopping witness")
    _require(any(a["erasures"] and a["blank_descendants"]
                 and a["max_depth"] >= 2 for a in train),
             "training must demonstrate an active blank relay and erasure")
    depth_shift = (max(a["max_depth"] for a in train) <= 2
                   and all(a["max_depth"] >= 3 and a["two_pass_ablation"] for a in test))
    island_shift = (all(a["dormant_edges"] == 0 for a in train)
                    and all(a["dormant_edges"] > 0 for a in test))
    _require(depth_shift or island_shift, "a structural train/test shift is required")


def _puzzle_digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
              for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded semantic, ablation, count, determinism, and 42-seed diversity checks."""
    # Small, hand-checkable circuit: root -> elbow -> remote emitter, plus an
    # unpowered two-cross cycle. These coordinates are test fixtures only.
    centers = ((10, 2), (11, 11), (5, 12), (16, 6), (17, 14))
    fixture = _build((20, 18), (2, 2), centers, (0, 8, 3), (0, 2, 7, 5, 6))
    expected = solve(fixture)
    _require(expected[0, 12] == 7 and expected[5, 0] == 7,
             "a blank root must power its colored descendants")
    _require(expected[5, 2] == 0, "a blank vertical ray must erase a colored horizontal ray")
    _require(expected[4, 11] == 0, "an arm must stop the ray before cells beyond it")
    _require(expected[0, 6] == 0 and expected[0, 14] == 0,
             "an unpowered cycle must not switch itself on")
    colored_fixture = fixture.copy()
    colored_fixture[10, 2] = 4
    _require(solve(colored_fixture)[5, 2] == 4,
             "vertical precedence must also hold for contrasting colored rays")
    moved_station = fixture.copy()
    moved_station[2:4, 2:4] = 0
    moved_station[0:2, 8:10] = 8
    _require(np.array_equal(solve(moved_station), moved_station),
             "without an aligned root there must be no power")
    permutation = np.array([9, 5, 0, 4, 7, 1, 8, 3, 2, 6], dtype=np.int8)
    _require(np.array_equal(solve(permutation[fixture]), permutation[expected]),
             "literal colors must not determine roles or activation")
    padded = np.pad(fixture, ((1, 2), (2, 1)), constant_values=0)
    _require(np.array_equal(solve(padded)[1:21, 2:20], expected),
             "translation and extra background must preserve the interior rule")
    shapes, normalized, palettes, profiles, signatures = set(), set(), set(), set(), set()
    object_counts, active_counts, root_counts, backgrounds = set(), set(), set(), set()
    square_count = colored_only_tests = blank_tests = depth_shifts = island_shifts = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_digest(p))
        train, test = [], []
        for split in ("train", "test"):
            for pair in p[split]:
                g, out = pair["input"], pair["output"]
                a = _analyze(g)
                _require(np.array_equal(solve(g.tolist()), out), "input-only solve mismatch")
                shapes.add(g.shape)
                mapping = {}
                normalized.add((g.shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in g.flat)))
                palettes.add((a["background"], a["station_color"], a["arm_color"]))
                backgrounds.add(a["background"])
                object_counts.add(len(a["centers"]))
                active_counts.add(len(a["active"]))
                root_counts.add(len(a["roots"]))
                profiles.add((len(a["centers"]), len(a["active"]), len(a["roots"]),
                              a["max_depth"], a["dormant_edges"]))
                square_count += g.shape[0] == g.shape[1]
                (train if split == "train" else test).append(a)
                if split == "test":
                    has_blank = any(g[a["centers"][i]] == a["background"] for i in a["active"])
                    blank_tests += has_blank
                    colored_only_tests += not has_blank
        depth_shifts += max(a["max_depth"] for a in train) <= 2 and all(a["max_depth"] >= 3 for a in test)
        island_shifts += all(not a["dormant_edges"] for a in train) and all(a["dormant_edges"] for a in test)
    _require(len(signatures) == 42, "seeds should create different puzzles")
    _require(len(shapes) >= 45 and len(normalized) >= 180, "geometric diversity too narrow")
    _require(len(palettes) >= 30 and len(backgrounds) >= 8, "role variation too narrow")
    _require(len(profiles) >= 30 and len(object_counts) >= 6 and len(active_counts) >= 5,
             "network topologies and counts must vary")
    _require(square_count >= 10 and blank_tests >= 8 and colored_only_tests >= 5,
             "shapes and signal compositions must vary")
    _require(depth_shifts >= 8 and island_shifts >= 8, "both structural shifts must occur")
    _require(_puzzle_digest(generate_puzzle(123456, 2, 1)) ==
             _puzzle_digest(generate_puzzle(123456, 2, 1)), "seed is not deterministic")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(91000 + 10 * nt + ns, nt, ns)
            _require((len(q["train"]), len(q["test"])) == (nt, ns), "counts not honored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "shapes": len(shapes), "normalized_inputs": len(normalized),
            "role_palettes": len(palettes), "network_profiles": len(profiles),
            "object_counts": sorted(object_counts), "active_counts": sorted(active_counts),
            "root_counts": sorted(root_counts), "depth_shifts": int(depth_shifts),
            "island_shifts": int(island_shifts), "blank_tests": int(blank_tests),
            "colored_only_tests": int(colored_only_tests)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
