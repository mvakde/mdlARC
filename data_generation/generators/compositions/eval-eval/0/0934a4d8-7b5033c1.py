"""Occluded excursions: mirror recovery followed by a window query on a route.

Only NumPy and the standard library are used. The solver reads only its grid.
Generation labels come from constructed coordinate routes, not from solve().
"""
from itertools import combinations
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_KINDS = ("horizontal", "vertical", "both", "diagonal", "antidiagonal")


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Illegal grid shape"
    assert np.issubdtype(a.dtype, np.integer), "Grid must contain integers"
    assert np.all((0 <= a) & (a <= 9)), "Illegal color"
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _mask(a, background):
    found = []
    for value in np.unique(a):
        color = int(value)
        if color == background:
            continue
        rr, cc = np.nonzero(a == color)
        box = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        r, s, c, d = box
        if len(rr) >= 2 and len(rr) == (s-r)*(d-c):
            found.append((color, box))
    assert len(found) == 1, "Erasure must be the unique rectangular color"
    return found[0]


def _inside(p, box):
    r, s, c, d = box
    return r <= p[0] < s and c <= p[1] < d


def _pack(path, colors, box, background):
    episodes = []
    previous_inside = False
    for p, color in zip(path, colors):
        inside = _inside(p, box)
        if inside:
            if not previous_inside:
                episodes.append([])
            episodes[-1].append(int(color))
        previous_inside = inside
    assert episodes, "Window must touch the route"
    out = np.full((max(map(len, episodes)), len(episodes)), background, dtype=np.int8)
    for col, sequence in enumerate(episodes):
        out[:len(sequence), col] = sequence
    return out, episodes


def _trace_components(full, background):
    cells = {tuple(map(int, p)) for p in np.argwhere(full != background)}
    remaining = set(cells)
    components = []
    while remaining:
        root = min(remaining)
        stack, component = [root], {root}
        remaining.remove(root)
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in _DIRS
                         if (p[0]+dr, p[1]+dc) in component] for p in component}
        ends = sorted(p for p in component if len(neighbors[p]) == 1)
        assert 2 <= len(component) <= 30, "Route length outside 2..30"
        assert len(ends) == 2 and all(len(v) in (1, 2) for v in neighbors.values()), "Not a simple route"
        assert ends[0][0] != ends[1][0], "Endpoint rows must differ"
        current = min(ends, key=lambda p: p[0])
        previous, ordered = None, []
        while True:
            ordered.append(current)
            nxt = [p for p in neighbors[current] if p != previous]
            if not nxt:
                break
            assert len(nxt) == 1 and nxt[0] not in ordered, "Invalid traversal"
            previous, current = current, nxt[0]
        assert len(ordered) == len(component), "Disconnected or cyclic route"
        components.append(ordered)
    return components


def _relations(a, mask_color, background):
    """Detect exact, visibly witnessed reflections, including offset axes.

    Every visible route pixel must have an in-bounds mate. Background cropping
    may leave unpaired background at the margins. Diagonal relations are global
    and square; an axial sum may deviate by at most six from the canvas center.
    """
    h, w = a.shape
    rr, cc = np.indices(a.shape)
    visible = a != mask_color
    foreground = visible & (a != background)
    candidates = []
    for axis, length in ((0, h), (1, w)):
        for total in range(max(0, length-7), min(2*length-1, length+6)):
            r, c = (total-rr, cc) if axis == 0 else (rr, total-cc)
            candidates.append((("h" if axis == 0 else "v", total), r, c))
    if h == w:
        candidates += [(("d", 0), cc, rr), (("a", 0), h-1-cc, w-1-rr)]
    accepted = []
    for name, r, c in candidates:
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        if inside.sum() < a.size/2 or np.any(foreground & ~inside):
            continue
        sr, sc = np.clip(r, 0, h-1), np.clip(c, 0, w-1)
        known = inside & visible & (a[sr, sc] != mask_color)
        moved = (r != rr) | (c != cc)
        if np.count_nonzero(known & foreground & moved) < 4:
            continue
        if not np.array_equal(a[known], a[sr[known], sc[known]]):
            continue
        accepted.append((name, r, c, inside))
    assert accepted, "No visibly supported reflection"
    return accepted


def _infer(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(counts.argmax())
    assert counts[background] > a.size/2, "Background must be a strict majority"
    mask_color, box = _mask(a, background)
    relations = _relations(a, mask_color, background)
    # Union every supported reflection correspondence. Surviving members of
    # an orbit must agree, even if a chain goes through multiple erased cells.
    parent = list(range(a.size))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = np.arange(a.size).reshape(a.shape)
    for _, r, c, inside in relations:
        for x, y in zip(ids[inside], ids[r[inside], c[inside]]):
            x, y = find(int(x)), find(int(y))
            if x != y:
                parent[y] = x
    witnesses = {}
    for index, color in enumerate(a.ravel()):
        if color == mask_color:
            continue
        orbit = find(index)
        assert orbit not in witnesses or witnesses[orbit] == int(color), "Conflicting mirror evidence"
        witnesses[orbit] = int(color)
    full = a.copy()
    for index in np.flatnonzero(a.ravel() == mask_color):
        orbit = find(int(index))
        assert orbit in witnesses, "A hidden orbit has no visible witness"
        full.flat[index] = witnesses[orbit]
    assert mask_color not in full, "Mask color is not a content color"
    components = _trace_components(full, background)
    assert len(components) >= 2, "Scene must contain reflected route copies"
    touched = [p for p in components if any(_inside(q, box) for q in p)]
    assert len(touched) == 1, "The window must touch exactly one route"
    path = touched[0]
    colors = [int(full[p]) for p in path]
    out, episodes = _pack(path, colors, box, background)
    hidden = [p for p in path if _inside(p, box)]
    assert any(full[p] == background for p in zip(*np.nonzero(a == mask_color))), "Window needs hidden background"
    return out, {"background": background, "mask": mask_color, "box": box,
                 "full": full, "path": path, "colors": colors,
                 "components": components, "episodes": episodes,
                 "hidden": hidden, "relations": [r[0] for r in relations]}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return top-aligned, temporally ordered hidden excursions from pixels."""
    return _infer(input_grid)[0]


def _walk(rng, length, persistence):
    path, occupied = [(0, 0)], {(0, 0)}
    last = None
    for _ in range(length-1):
        choices, weights = [], []
        for dr, dc in _DIRS:
            q = path[-1][0]+dr, path[-1][1]+dc
            if q in occupied or sum((q[0]+a, q[1]+b) in occupied for a, b in _DIRS) != 1:
                continue
            rows = [p[0] for p in path] + [q[0]]
            cols = [p[1] for p in path] + [q[1]]
            if max(rows)-min(rows) >= 12 or max(cols)-min(cols) >= 12:
                continue
            choices.append((q, (dr, dc)))
            weights.append(persistence if last == (dr, dc) else 1.0)
        if not choices:
            return None
        prob = np.asarray(weights, dtype=float)
        q, last = choices[int(rng.choice(len(choices), p=prob/prob.sum()))]
        path.append(q)
        occupied.add(q)
    if path[0][0] == path[-1][0] or path[0][1] == path[-1][1]:
        return None
    return path


def _fallbacks(length):
    """Finite serpentine safety net, with many widths, phases and orientations."""
    for width in range(3, 10):
        for start in range(3):
            path = [(0, start)]
            direction = 1
            while len(path) < length:
                end = width if direction == 1 else 0
                while path[-1][1] != end and len(path) < length:
                    path.append((path[-1][0], path[-1][1]+direction))
                if len(path) < length:
                    path.append((path[-1][0]+1, path[-1][1]))
                if len(path) < length:
                    path.append((path[-1][0]+1, path[-1][1]))
                direction *= -1
            if path[-1][0] >= 12 or path[0][1] == path[-1][1]:
                continue
            yield path
            yield [(r, -c) for r, c in path]
            yield [(c, r) for r, c in path]
            yield [(-r, c) for r, c in path]


def _scene(rng, path, kind, crop):
    min_r, min_c = min(p[0] for p in path), min(p[1] for p in path)
    path = [(r-min_r, c-min_c) for r, c in path]
    bh, bw = max(p[0] for p in path)+1, max(p[1] for p in path)+1
    if kind in ("diagonal", "antidiagonal"):
        n = max(bh, bw) + int(rng.integers(0, 3))
        gap = int(rng.integers(1, 4))
        h = w = min(30, 2*n+gap)
        # Place the whole route on one side of the diagonal, away from contact.
        top = int(rng.integers(n-bh+1))
        left = h-n + int(rng.integers(n-bw+1))
        p0 = [(r+top, c+left) for r, c in path]
        p1 = [(c, r) for r, c in p0]
        components = [p0, p1]
        if kind == "antidiagonal":
            components = [[(r, w-1-c) for r, c in p] for p in components]
    else:
        gh, gw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        local_h, local_w = bh+int(rng.integers(0, 3)), bw+int(rng.integers(0, 3))
        h = 2*local_h+gh if kind in ("horizontal", "both") else min(30, local_h+int(rng.integers(1, 8)))
        w = 2*local_w+gw if kind in ("vertical", "both") else min(30, local_w+int(rng.integers(1, 8)))
        h, w = min(30, h), min(30, w)
        top = int(rng.integers(local_h-bh+1)) if kind in ("horizontal", "both") else int(rng.integers(h-bh+1))
        left = int(rng.integers(local_w-bw+1)) if kind in ("vertical", "both") else int(rng.integers(w-bw+1))
        p0 = [(r+top, c+left) for r, c in path]
        components = [p0]
        if kind in ("horizontal", "both"):
            components += [[(h-1-r, c) for r, c in p0]]
        if kind in ("vertical", "both"):
            components += [[(r, w-1-c) for r, c in p] for p in list(components)]
        if crop:
            # Crop only background margins; retain every route and counterpart.
            coords = [p for comp in components for p in comp]
            cuts = [min(3, min(r for r, c in coords)), min(3, h-1-max(r for r, c in coords)),
                    min(3, min(c for r, c in coords)), min(3, w-1-max(c for r, c in coords))]
            cuts = [int(rng.integers(v+1)) for v in cuts]
            components = [[(r-cuts[0], c-cuts[2]) for r, c in p] for p in components]
            h, w = h-cuts[0]-cuts[1], w-cuts[2]-cuts[3]
    return (h, w), components


def _boxes(shape, components, selected, minimum, maximum, unequal, endpoint_mode):
    """Enumerate region intersections from geometry, without reading labels."""
    h, w = shape
    path = components[selected]
    pr, pc = np.asarray(path).T
    rb = sorted(set(range(max(0, int(pr.min())-1), min(h, int(pr.max())+2)+1)) | {0, h})
    cb = sorted(set(range(max(0, int(pc.min())-1), min(w, int(pc.max())+2)+1)) | {0, w})
    boxes = np.asarray([(r, s, c, d) for r, s in combinations(rb, 2) for c, d in combinations(cb, 2)], dtype=int)
    r, s, c, d = boxes.T
    membership = ((pr[None, :] >= r[:, None]) & (pr[None, :] < s[:, None]) &
                  (pc[None, :] >= c[:, None]) & (pc[None, :] < d[:, None]))
    covered = membership.sum(axis=1)
    starts = membership & ~np.concatenate((np.zeros((len(boxes), 1), dtype=bool), membership[:, :-1]), axis=1)
    visits = starts.sum(axis=1)
    area = (s-r)*(d-c)
    good = ((visits >= minimum) & (visits <= maximum) & (covered >= 4) &
            (covered <= len(path)-4) & (area > covered) & (area <= h*w*.23))
    if unequal:
        steps = np.diff(np.asarray(path), axis=0)
        turns = np.any(steps[:-1] != steps[1:], axis=1)
        good &= np.any(membership[:, :-2] & membership[:, 1:-1] & membership[:, 2:] & turns, axis=1)
    if endpoint_mode == "visible":
        good &= ~membership[:, 0] & ~membership[:, -1]
    elif endpoint_mode == "hidden":
        good &= membership[:, 0] | membership[:, -1]
    others = np.asarray([p for i, route in enumerate(components) if i != selected for p in route])
    if len(others):
        ar, ac = others.T
        good &= ~np.any((ar[None, :] >= r[:, None]) & (ar[None, :] < s[:, None]) &
                        (ac[None, :] >= c[:, None]) & (ac[None, :] < d[:, None]), axis=1)
    # Sparse scene plus mask must preserve a strict input background majority.
    good &= h*w-sum(map(len, components))-(area-covered) > h*w/2
    result = []
    for index in np.flatnonzero(good):
        box = tuple(map(int, boxes[index]))
        ordered = [p for p, yes in zip(path, membership[index]) if yes]
        if minimum >= 2 and ordered == sorted(ordered):
            continue
        _, episodes = _pack(path, list(range(len(path))), box, -1)
        if unequal and len({len(x) for x in episodes}) < 2:
            continue
        result.append(box)
    return result


def _sequence(rng, length, palette, style):
    if style == "alternating":
        return [palette[i % 2] for i in range(length)]
    seq = [int(rng.choice(palette))]
    persistence = .8 if style == "long_runs" else (.15 if style == "fine" else .5)
    for _ in range(1, length):
        seq.append(seq[-1] if rng.random() < persistence else int(rng.choice(palette)))
    return seq


def _diagnostics(record):
    hidden, path, colors = record["hidden"], record["path"], record["colors"]
    lookup = dict(zip(path, colors))
    ordered = [lookup[p] for p in hidden]
    row_major = [lookup[p] for p in sorted(hidden)]
    reverse, _ = _pack(path[::-1], colors[::-1], record["box"], record["background"])
    true, _ = _pack(path, colors, record["box"], record["background"])
    runs = [v for i, v in enumerate(ordered) if i == 0 or v != ordered[i-1]]
    steps = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:])]
    return {"spatial_failure": ordered != row_major,
            "reverse_failure": not np.array_equal(true, reverse),
            "recurrence": len(runs) > len(set(runs)),
            "unequal": len({len(x) for x in record["episodes"]}) > 1,
            "inside_color_change": any(len(set(ep)) > 1 for ep in record["episodes"]),
            "bend_inside": any(a != b and all(_inside(p, record["box"]) for p in path[i:i+3])
                               for i, (a, b) in enumerate(zip(steps, steps[1:]))),
            "endpoint_hidden": _inside(path[0], record["box"]) or _inside(path[-1], record["box"])}


def _make_pair(rng, kind, length, selected_side, min_visits, max_visits,
               witness, endpoint_mode, seen, style, allow_above=True):
    def candidates():
        for _ in range(55):
            path = _walk(rng, length, float(rng.choice((.35, 1.0, 2.5, 6.0))))
            if path is not None:
                yield path
        yield from _fallbacks(length)

    for path in candidates():
        shape, components = _scene(rng, path, kind, crop=rng.random() < .65)
        # Higher endpoint is recomputed separately in each reflected copy.
        selected = 0 if selected_side == "first" else len(components)-1
        target = components[selected]
        reverse = target[0][0] > target[-1][0]
        if reverse:
            target = target[::-1]
        if not allow_above and min(p[0] for p in target) < target[0][0]:
            continue
        working = list(components)
        working[selected] = target
        choices = _boxes(shape, working, selected, min_visits, max_visits, witness, endpoint_mode)
        if not choices:
            continue
        for box_index in rng.permutation(len(choices))[:12]:
            box = choices[int(box_index)]
            background, mask_color = map(int, rng.choice(10, 2, replace=False))
            palette = [int(v) for v in rng.choice([c for c in range(10) if c not in (background, mask_color)],
                                                int(rng.integers(2, 7)), replace=False)]
            for _ in range(32):
                base_colors = _sequence(rng, length, palette, style)
                colors = base_colors[::-1] if reverse else base_colors
                output, episodes = _pack(target, colors, box, background)
                hidden = [p for p in target if _inside(p, box)]
                record = {"hidden": hidden, "path": target, "colors": colors,
                          "episodes": episodes, "box": box, "background": background}
                diag = _diagnostics(record)
                if witness and not all(diag[k] for k in ("spatial_failure", "reverse_failure", "recurrence", "inside_color_change")):
                    continue
                full = np.full(shape, background, dtype=np.int8)
                for component in components:
                    for p, color in zip(component, base_colors):
                        full[p] = color
                a = full.copy()
                r, s, c, d = box
                a[r:s, c:d] = mask_color
                if _key(a) in seen:
                    continue
                try:
                    predicted, inferred = _infer(a)
                except AssertionError:
                    break
                if not np.array_equal(inferred["full"], full) or not np.array_equal(predicted, output):
                    break
                return {"input": a, "output": output}
    raise RuntimeError("No valid scene after bounded walk and constructive fallback search")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    kinds = list(rng.choice(_KINDS, 2, replace=False))
    regime = str(rng.choice(("length", "visits", "endpoint")))
    style = str(rng.choice(("mixed", "fine", "long_runs", "alternating")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        allow_above = True
        if split == "test":
            training_records = [_infer(p["input"])[1] for p in puzzle["train"]]
            allow_above = any(min(r for r, c in rec["path"]) < rec["path"][0][0]
                              and _diagnostics(rec)["reverse_failure"] for rec in training_records)
        for index in range(count):
            kind = str(kinds[index % 2])
            witness = split == "train" and index < 2
            if split == "train":
                length = int(rng.integers(18, 24)) if witness else int(rng.integers(8, 24))
                minimum, maximum = (2, 3) if witness else (1, 3)
                endpoint_mode = "visible" if regime == "endpoint" else "any"
                side = "first"
            else:
                length = int(rng.integers(25, 31)) if regime == "length" else int(rng.integers(20, 27))
                minimum, maximum = (4, 5) if regime == "visits" else (2, 4)
                endpoint_mode = "hidden" if regime == "endpoint" else "any"
                side = "last"
            pair = _make_pair(rng, kind, length, side, minimum, maximum, witness,
                              endpoint_mode, seen, style if rng.random() < .65 else "mixed",
                              allow_above=allow_above)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, teaching = set(), []
    above = {"train": False, "test": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "Repeated input"
            seen.add(_key(a))
            predicted, record = _infer(a)
            assert np.array_equal(out, predicted), "Incorrect excursion readout"
            assert record["mask"] not in out, "Erasure leaked into output"
            episodes, bg = record["episodes"], record["background"]
            assert out.shape == (max(map(len, episodes)), len(episodes))
            assert sum(map(len, episodes)) == len(record["hidden"])
            assert np.count_nonzero(out != bg) == len(record["hidden"])
            for col, seq in enumerate(episodes):
                assert np.array_equal(out[:len(seq), col], seq)
                assert np.all(out[len(seq):, col] == bg)
            assert all(a[p] == record["mask"] for p in record["hidden"])
            diagnostic = _diagnostics(record)
            if split == "train":
                teaching.append((len(episodes), diagnostic))
            above[split] |= (min(r for r, c in record["path"]) < record["path"][0][0]
                             and (split == "test" or diagnostic["reverse_failure"]))
    assert any(n >= 2 and d["unequal"] for n, d in teaching), "No segmentation/padding witness"
    assert any(d["spatial_failure"] and d["reverse_failure"] and d["recurrence"] for _, d in teaching), "Weak traversal evidence"
    assert any(d["bend_inside"] and d["inside_color_change"] for _, d in teaching), "No evidence against bend/color segmentation"
    assert not above["test"] or above["train"], "Above-endpoint convention was not demonstrated"


def self_test():
    signatures, shapes, geometries, lengths, visits, relations = set(), set(), set(), set(), set(), set()
    masks, backgrounds, output_shapes = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                output, record = _infer(pair["input"])
                assert np.array_equal(output, pair["output"])
                shapes.add(pair["input"].shape)
                output_shapes.add(output.shape)
                path = record["path"]
                geometries.add(tuple((r-path[0][0], c-path[0][1]) for r, c in path))
                lengths.add(len(path))
                visits.add(len(record["episodes"]))
                relations.update(name[0] for name in record["relations"])
                masks.add(record["mask"])
                backgrounds.add(record["background"])
                # A color bijection preserves every semantic role and the answer.
                perm = np.roll(np.arange(10, dtype=np.int8), 3)
                assert np.array_equal(solve(perm[pair["input"]]), perm[output])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(geometries) >= 100
    assert len(lengths) >= 15 and len(output_shapes) >= 20 and {1, 2, 3, 4} <= visits
    assert {"h", "v", "d", "a"} <= relations and len(masks) == len(backgrounds) == 10
    # Parent A counterfactual: recolor only the visible counterparts of a
    # hidden orbit. The queried route's visible fragments remain identical.
    p = generate_puzzle(0, 2, 1)["train"][0]
    original, record = _infer(p["input"])
    node = record["hidden"][0]
    orbit, frontier = {node}, [node]
    h, w = p["input"].shape
    while frontier:
        r, c = frontier.pop()
        for name, total in record["relations"]:
            q = {"h": (total-r, c), "v": (r, total-c),
                 "d": (c, r), "a": (h-1-c, w-1-r)}[name]
            if 0 <= q[0] < h and 0 <= q[1] < w and q not in orbit:
                orbit.add(q)
                frontier.append(q)
    old = int(record["full"][node])
    replacement = next(v for v in record["colors"] if v != old)
    changed = p["input"].copy()
    for q in orbit:
        if changed[q] != record["mask"]:
            changed[q] = replacement
    assert all(changed[q] == p["input"][q] for q in record["path"] if not _inside(q, record["box"]))
    counterfactual, changed_record = _infer(changed)
    assert changed_record["path"] == record["path"]
    assert not np.array_equal(counterfactual, original), "Mirror evidence failed to affect answer"
    # Parent B ablation: use the correct visit lengths but fill them with the
    # spatially sorted hidden sequence. At least one packed pixel is wrong.
    witness = next(p for p in generate_puzzle(1, 2, 1)["train"] if _diagnostics(_infer(p["input"])[1])["spatial_failure"])
    correct, rec = _infer(witness["input"])
    wrong = correct.copy()
    spatial = [int(rec["full"][q]) for q in sorted(rec["hidden"])]
    cursor = 0
    for col, episode in enumerate(rec["episodes"]):
        wrong[:len(episode), col] = spatial[cursor:cursor+len(episode)]
        cursor += len(episode)
    assert not np.array_equal(correct, wrong), "Spatial ordering ablation did not change answer"
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(90210, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
    # Transform coordinates, then reselect the higher endpoint. The readout
    # remains a packed set of vertical columns, even after rotating the input.
    p = generate_puzzle(17, 2, 1)["train"][0]
    _, rec = _infer(p["input"])
    input_grid, path, colors = p["input"].copy(), rec["path"], rec["colors"]
    for turn in range(4):
        mask, box = _mask(input_grid, rec["background"])
        ordered_path, ordered_colors = (path, colors) if path[0][0] < path[-1][0] else (path[::-1], colors[::-1])
        expected, _ = _pack(ordered_path, ordered_colors, box, rec["background"])
        assert np.array_equal(solve(input_grid), expected)
        reflected = input_grid[:, ::-1].copy()
        reflected_path = [(r, input_grid.shape[1]-1-c) for r, c in ordered_path]
        _, reflected_box = _mask(reflected, rec["background"])
        expected_reflection, _ = _pack(reflected_path, ordered_colors, reflected_box, rec["background"])
        assert np.array_equal(solve(reflected), expected_reflection)
        old_width = input_grid.shape[1]
        input_grid = np.rot90(input_grid).copy()
        path = [(old_width-1-c, r) for r, c in path]
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(3, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect label accepted")
    # Invalid evidence, fully erased orbits, and endpoint/graph violations.
    broken = witness["input"].copy()
    visible = next(q for q in rec["path"] if not _inside(q, rec["box"]))
    broken[visible] = rec["background"]
    erased = np.zeros((10, 12), dtype=np.int8)
    erased[1:9, 1:4] = 9
    for malformed in (broken, erased):
        try:
            solve(malformed)
        except AssertionError:
            pass
        else:
            raise AssertionError("Malformed reflection evidence accepted")
    for cells in ([(1, 1), (1, 2), (1, 3)],
                  [(1, 2), (2, 1), (2, 2), (2, 3)],
                  [(1, 1), (1, 2), (2, 1), (2, 2)],
                  [(1, 1), (2, 1), (4, 4)]):
        malformed = np.zeros((6, 6), dtype=np.int8)
        for point in cells:
            malformed[point] = 1
        try:
            _trace_components(malformed, 0)
        except AssertionError:
            pass
        else:
            raise AssertionError("Tied endpoints, branch, cycle or singleton accepted")
    return {"seeds": 42, "input_shapes": len(shapes), "route_geometries": len(geometries),
            "route_lengths": sorted(lengths), "visits": sorted(visits), "output_shapes": len(output_shapes)}
