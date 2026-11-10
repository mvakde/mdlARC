"""Close labeled circuits behind an opaque panel, then reveal its interior.

The renderer constructs complete paths. The input reader independently traces
visible pixels; it never receives construction descriptors or an example output.
No files, external dependencies, cached scenes, or execution traces are used.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from itertools import permutations

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "noninteger cell"
    assert np.all((0 <= g) & (g <= 9)), "color bounds"
    return g


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(cells):
    unseen = set(cells)
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        group, stack = {p}, [p]
        while stack:
            y, x = stack.pop()
            for dy, dx in _DIRS:
                q = (y + dy, x + dx)
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    stack.append(q)
        yield group


def _border(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _find_panel(g, bg):
    candidates = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for points in _components(map(tuple, np.argwhere(g == color))):
            ys, xs = zip(*points)
            box = tuple(map(int, (min(ys), max(ys), min(xs), max(xs))))
            t, b, l, r = box
            if b - t < 4 or r - l < 4 or points != _border(box):
                continue
            inside = g[t + 1:b, l + 1:r]
            if np.all(inside == inside[0, 0]) and int(inside[0, 0]) != color:
                candidates.append((box, color, int(inside[0, 0])))
    assert len(candidates) == 1, "panel must have a unique complete frame"
    return candidates[0]


def _read_input(input_grid):
    """Parse a geometric circuit drawing without seeing any answer."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    box, insulator, fill = _find_panel(g, bg)
    t, b, l, r = box
    assert 0 < t < b < g.shape[0] - 1 and 0 < l < r < g.shape[1] - 1
    points = {tuple(map(int, p)) for p in np.argwhere(g != bg)
              if not (t <= p[0] <= b and l <= p[1] <= r)}
    graph = {p: {d: (p[0] + dy, p[1] + dx) for d, (dy, dx) in enumerate(_DIRS)
                 if (p[0] + dy, p[1] + dx) in points} for p in points}
    assert points and all(len(edges) in (1, 2, 4) for edges in graph.values()), "broken cable"
    carrier_colors = {int(g[p]) for p, edges in graph.items() if len(edges) == 2}
    assert len(carrier_colors) == 1, "carrier must be uniform between ports and crossings"
    carrier = carrier_colors.pop()
    assert carrier not in {bg, insulator, fill}
    ports, crossings = {}, set()
    for p, edges in graph.items():
        y, x = p
        if len(edges) == 4:
            assert int(g[p]) == insulator, "crossings must use the frame color"
            crossings.add(p)
        elif len(edges) == 2:
            assert int(g[p]) == carrier
        else:
            if y == t - 1 and l + 1 < x < r - 1:
                side, inward = "top", 2
            elif y == b + 1 and l + 1 < x < r - 1:
                side, inward = "bottom", 0
            elif x == l - 1 and t + 1 < y < b - 1:
                side, inward = "left", 1
            elif x == r + 1 and t + 1 < y < b - 1:
                side, inward = "right", 3
            else:
                raise AssertionError("free cable endpoint is not a panel port")
            assert (inward + 2) % 4 in edges, "port must approach the frame normally"
            assert int(g[p]) not in {bg, insulator, fill}
            ports[p] = side
    sides = set(ports.values())
    assert len(sides) == 2 and len(sides & {"top", "bottom"}) == 1
    assert len(sides & {"left", "right"}) == 1, "port banks must be adjacent"
    starts = sorted(p for p in ports if ports[p] in {"top", "bottom"})
    ends = {p for p in ports if ports[p] in {"left", "right"}}
    n = len(starts)
    assert 2 <= n <= 4 and len(ends) == n
    assert all(v - u >= 3 for u, v in zip(sorted(p[1] for p in starts), sorted(p[1] for p in starts)[1:]))
    assert all(v - u >= 3 for u, v in zip(sorted(p[0] for p in ends), sorted(p[0] for p in ends)[1:]))
    routes, used, arrivals = [], set(), set()
    for start in starts:
        p, previous, trace, states = start, None, [], set()
        while True:
            assert (p, previous) not in states, "cable cycle"
            states.add((p, previous))
            trace.append(p)
            used.add(p)
            if p != start and p in ports:
                assert p in ends and p not in arrivals, "nonbijective cable pairing"
                arrivals.add(p)
                break
            edges = graph[p]
            if previous is None:
                following = next(iter(edges.values()))
            elif len(edges) == 4:
                direction = (p[0] - previous[0], p[1] - previous[1])
                following = (p[0] + direction[0], p[1] + direction[1])
                assert following in edges.values()
            else:
                options = [q for q in edges.values() if q != previous]
                assert len(options) == 1
                following = options[0]
            previous, p = p, following
        labels = {int(g[q]) for q in (start, p)} - {carrier}
        assert len(labels) == 1, "each circuit needs an agreeing endpoint label"
        routes.append({"vertical": start, "horizontal": p, "label": labels.pop(), "trace": trace})
    assert arrivals == ends and used == points, "unattached exterior material"
    labels = {route["label"] for route in routes}
    assert len(labels) >= 2 and labels.isdisjoint({bg, carrier, insulator, fill})
    return {"grid": g, "box": box, "background": bg, "fill": fill,
            "insulator": insulator, "carrier": carrier, "routes": routes,
            "crossings": crossings, "sides": tuple(sorted(sides))}


def _closure_cells(route, box, rays=False):
    t, b, l, r = box
    vy, vx = route["vertical"]
    hy, hx = route["horizontal"]
    if rays:
        return ({(y, vx) for y in range(t + 1, b)}
                | {(hy, x) for x in range(l + 1, r)})
    ys = range(t + 1, hy + 1) if vy < t else range(hy, b)
    xs = range(vx, r) if hx > r else range(l + 1, vx + 1)
    return {(y, vx) for y in ys} | {(hy, x) for x in xs}


def _reveal(info, ordered=False, rays=False, uncolored=False):
    t, b, l, r = info["box"]
    out = np.full((b - t - 1, r - l - 1), info["fill"], dtype=np.int8)
    owners = {}
    routes = info["routes"]
    # A plausible route-blind rule matches ports by distance to the shared
    # corner. Use this rotation/reflection invariant ordering for the ablation.
    vertical_near_first = sorted((route["vertical"] for route in routes),
                                 key=lambda p: -p[1] if routes[0]["horizontal"][1] > r else p[1])
    horizontal_near_first = sorted((route["horizontal"] for route in routes),
                                   key=lambda p: p[0] if routes[0]["vertical"][0] < t else -p[0])
    ordered_partners = dict(zip(vertical_near_first, horizontal_near_first))
    for i, route in enumerate(routes):
        route = dict(route)
        if ordered:
            route["horizontal"] = ordered_partners[route["vertical"]]
        for p in _closure_cells(route, info["box"], rays):
            owners.setdefault(p, []).append(info["carrier"] if uncolored else route["label"])
    for (y, x), colors in owners.items():
        assert len(colors) <= 2
        out[y - t - 1, x - l - 1] = colors[0] if len(colors) == 1 else info["insulator"]
    return out, owners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Trace the exterior, close each loop with one hidden elbow, crop."""
    return _reveal(_read_input(input_grid))[0]


def _axis(rng, n):
    """Allocate an entire axis first, so no sampling can exceed 30 cells."""
    values = [1] + [2] * (n - 1) + [3, 2] + [3] * (n - 1) + [2, 1]
    extra = int(rng.integers(min(30 - (5 * n + 5), 12) + 1))
    for _ in range(extra):
        values[int(rng.integers(len(values)))] += 1
    k, q = 1, values[0]
    tracks = [q]
    for _ in range(n - 1):
        q += values[k]
        tracks.append(q)
        k += 1
    q += values[k]
    near = q
    k += 1
    q += values[k]
    k += 1
    ports = [q]
    for _ in range(n - 1):
        q += values[k]
        ports.append(q)
        k += 1
    q += values[k]
    far = q
    size = far + values[k + 1] + 1
    assert size <= 30
    return size, near, far, ports, tracks


def _polyline(vertices):
    path = [vertices[0]]
    for (a, b), (c, d) in zip(vertices, vertices[1:]):
        assert (a == c) != (b == d)
        dy, dx = int(np.sign(c - a)), int(np.sign(d - b))
        while (a, b) != (c, d):
            a, b = a + dy, b + dx
            path.append((a, b))
    assert len(path) == len(set(path))
    return path


def _orient(g, turn, flip):
    result = np.rot90(g, turn)
    return (result[:, ::-1] if flip else result).copy()


def _make_pair(rng, n, lesson, turn, flip, target=None):
    h, t, b, rows, top_tracks = _axis(rng, n)
    w, near, far, reflected_cols, reflected_tracks = _axis(rng, n)
    l, r = w - 1 - far, w - 1 - near
    cols = sorted(w - 1 - x for x in reflected_cols)
    right_tracks = [w - 1 - x for x in reflected_tracks]
    top_tracks = list(map(int, rng.permutation(top_tracks)))
    right_tracks = list(map(int, rng.permutation(right_tracks)))
    permutation = (list(target) if target is not None else list(reversed(range(n))) if lesson == 0
                   else list(range(n)) if lesson == 1 else list(map(int, rng.permutation(n))))
    assert sorted(permutation) == list(range(n))
    palette = list(map(int, rng.permutation(10)))
    bg, frame, fill, carrier = palette[:4]
    if lesson == 0 or (lesson != 1 and rng.random() < .35):
        fill = bg
    labels = palette[4:4 + n]
    if lesson > 1 and n > 2 and rng.random() < .4:
        labels[-1] = labels[0]
    modes = list(map(int, rng.integers(0, 3, size=n)))  # 0: vertical, 1: horizontal, 2: both
    if lesson == 0:
        modes = [0] * n
    elif lesson == 1:
        modes[0], modes[1] = 1, 2
    outside_owners, hidden_owners = {}, {}
    paths = []
    for i, j in enumerate(permutation):
        path = _polyline([(t - 1, cols[i]), (top_tracks[i], cols[i]),
                          (top_tracks[i], right_tracks[j]), (rows[j], right_tracks[j]),
                          (rows[j], r + 1)])
        paths.append(path)
        for p in path:
            outside_owners.setdefault(p, []).append(i)
        # Construct the complete concealed path, including frame cells, before
        # applying the cover. Truth does not call solve or the reader.
        hidden = _polyline([(t, cols[i]), (rows[j], cols[i]), (rows[j], r)])
        for y, x in hidden:
            if t < y < b and l < x < r:
                hidden_owners.setdefault((y, x), []).append(i)
    inp = np.full((h, w), bg, dtype=np.int8)
    for p, owners in outside_owners.items():
        assert len(owners) <= 2
        inp[p] = carrier if len(owners) == 1 else frame
    for i, path in enumerate(paths):
        if modes[i] != 1:
            inp[path[0]] = labels[i]
        if modes[i] != 0:
            inp[path[-1]] = labels[i]
    for p in _border((t, b, l, r)):
        inp[p] = frame
    inp[t + 1:b, l + 1:r] = fill
    out = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    for (y, x), owners in hidden_owners.items():
        assert len(owners) <= 2
        out[y - t - 1, x - l - 1] = labels[owners[0]] if len(owners) == 1 else frame
    return {"input": _orient(inp, turn, flip), "output": _orient(out, turn, flip)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    training_max = int(rng.integers(2, 4))
    base_turn, base_flip = int(rng.integers(4)), bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else 2
            n = (2 if lesson == 0 else training_max if lesson == 1 else
                 int(rng.integers(2, training_max + 1))) if split == "train" else training_max + 1
            # The second demonstration reflects the first corner. Later cases
            # freely combine the already demonstrated adjacency and symmetry.
            turn = base_turn if lesson < 2 else int(rng.integers(4))
            flip = base_flip if lesson == 0 else not base_flip if lesson == 1 else bool(rng.integers(2))
            for _ in range(16):
                pair = _make_pair(rng, n, lesson, turn, flip)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("duplicate inputs after 16 bounded attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, output):
    routes = info["routes"]
    g, carrier = info["grid"], info["carrier"]
    ends = sorted(route["horizontal"] for route in routes)
    modes = tuple((int(g[q["vertical"]]) != carrier, int(g[q["horizontal"]]) != carrier)
                  for q in routes)
    t, b, l, r = info["box"]
    return {"circuits": len(routes), "corner": info["sides"], "shape": g.shape,
            "output_shape": output.shape, "ports": tuple((q["vertical"], q["horizontal"]) for q in routes),
            "permutation": tuple(ends.index(q["horizontal"]) for q in routes),
            "external_crossings": len(info["crossings"]),
            "hidden_crossings": int(np.count_nonzero(output == info["insulator"])),
            "labels": len({q["label"] for q in routes}), "clues": modes,
            "fill_matches": info["fill"] == info["background"],
            "roles": tuple(info[k] for k in ("background", "insulator", "fill", "carrier")),
            "box": (t, b, l, r), "lengths": tuple(len(q["trace"]) for q in routes)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _key(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            info = _read_input(inp)
            expected, owners = _reveal(info)
            assert np.array_equal(out, expected), "wrong closed-circuit crop"
            assert all(1 <= len(v) <= 2 for v in owners.values())
            assert np.count_nonzero(out == info["insulator"]) == sum(len(v) == 2 for v in owners.values())
            assert info["carrier"] not in set(map(int, np.unique(out))), "unpropagated neutral wire"
            assert set(map(int, np.unique(out))) <= ({info["fill"], info["insulator"]}
                                                   | {q["label"] for q in info["routes"]})
            assert np.any(out == info["fill"]), "no empty panel cells"
            assert not np.array_equal(out, _reveal(info, rays=True)[0]), "closure can be omitted"
            features[split].append(_features(info, out))
    train = features["train"]
    assert train[0]["circuits"] == 2 and train[0]["hidden_crossings"] == 0
    assert train[0]["fill_matches"] and not train[1]["fill_matches"]
    assert train[1]["external_crossings"] > 0 and train[1]["hidden_crossings"] > 0
    assert train[1]["labels"] == train[1]["circuits"]
    clue_modes = {mode for f in train[:2] for mode in f["clues"]}
    assert {(True, False), (False, True), (True, True)} <= clue_modes
    # Omitting A: pair the banks by distance to their shared corner, not tracing.
    # Omitting B: extend straight rays, without closing each circuit by an elbow.
    assert any(not np.array_equal(np.asarray(pair["output"]), _reveal(_read_input(pair["input"]), ordered=True)[0])
               for pair in puzzle["train"][:2]), "routing has no demonstrated effect"
    assert all(f["circuits"] > max(q["circuits"] for q in train) for f in features["test"])


def self_test():
    """42 streamed seeds, all legal counts, bad inputs, and two concept ablations.

    Returns small aggregate measurements for an optional caller-owned report.
    No samples or logs are written. Only the current puzzle is retained.
    """
    signatures, shapes, output_shapes, layouts, corners, roles = set(), set(), set(), set(), set(), set()
    counts, crosses, clues, repeats, fill_modes = Counter(), Counter(), set(), 0, set()
    ablations = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                prediction = solve(inp)
                assert np.array_equal(prediction, out) and np.array_equal(inp, before)
                info = _read_input(inp)
                f = _features(info, out)
                shapes.add(inp.shape)
                output_shapes.add(out.shape)
                layouts.add((f["box"], f["ports"], f["permutation"]))
                corners.add(f["corner"])
                roles.add(f["roles"])
                counts[f["circuits"]] += 1
                crosses[(f["external_crossings"] > 0, f["hidden_crossings"] > 0)] += 1
                clues.update(f["clues"])
                repeats += int(f["labels"] < f["circuits"])
                fill_modes.add(f["fill_matches"])
                for name, kwargs in (("ordered_ports", {"ordered": True}), ("straight_rays", {"rays": True}),
                                     ("no_labels", {"uncolored": True})):
                    ablations[name] += int(not np.array_equal(out, _reveal(info, **kwargs)[0]))
                for g in (inp, out):
                    digest.update(bytes(g.shape))
                    digest.update(g.tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 35 and len(output_shapes) >= 25
    assert len(layouts) >= 190 and len(roles) >= 180
    assert len(corners) == 4 and set(counts) == {2, 3, 4}
    assert clues == {(True, False), (False, True), (True, True)}
    assert fill_modes == {False, True} and repeats > 0
    assert crosses[(True, True)] > 0 and crosses[(False, False)] > 0
    assert ablations["ordered_ports"] >= 42 and ablations["straight_rays"] == 210
    assert ablations["no_labels"] == 210
    # Exhaust every supported permutation at every count and all eight square
    # symmetries. This tests input tracing, crop orientation, and crossing pixels.
    circuit_rng = np.random.default_rng(7221)
    permutation_cases = 0
    for n in (2, 3, 4):
        for target in permutations(range(n)):
            for turn in range(4):
                for flip in (False, True):
                    pair = _make_pair(circuit_rng, n, 2, turn, flip, target)
                    assert np.array_equal(solve(pair["input"]), pair["output"])
                    permutation_cases += 1
    # All 30 legal explicit count combinations are exercised with distinct seeds.
    legal_counts = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
            legal_counts += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    one, two = generate_puzzle(123, 10, 4), generate_puzzle(123, 10, 4)
    assert all(_key(p[k]) == _key(q[k]) for s in one for p, q in zip(one[s], two[s]) for k in p)
    damaged = generate_puzzle(11, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Input consistency is checked rather than silently selecting one clue.
    bad = generate_puzzle(17, 2, 1)["train"][1]["input"].copy()
    info = _read_input(bad)
    route = next(q for q in info["routes"] if bad[q["vertical"]] != info["carrier"]
                 and bad[q["horizontal"]] != info["carrier"])
    replacement = next(c for c in range(10) if c not in set(info[k] for k in
                       ("background", "insulator", "fill", "carrier")) and c != route["label"])
    bad[route["horizontal"]] = replacement
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting endpoint labels accepted")
    bad = generate_puzzle(17, 2, 1)["train"][1]["input"].copy()
    info = _read_input(bad)
    bad[min(info["crossings"])] = info["carrier"]
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("unmarked four-way junction accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes), "layouts": len(layouts),
            "role_assignments": len(roles), "corner_choices": len(corners),
            "circuit_counts": dict(sorted(counts.items())), "repeated_label_pairs": repeats,
            "crossing_presence": [{"exterior": a, "hidden": b, "pairs": count}
                                  for (a, b), count in sorted(crosses.items())],
            "changed_by_ablation": dict(ablations), "legal_count_combinations": legal_counts,
            "permutation_symmetry_cases": permutation_cases,
            "independent_semantic_review": False}
