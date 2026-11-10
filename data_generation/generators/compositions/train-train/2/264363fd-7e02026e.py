"""Plus relays: exact local stencils coupled by visible directional rays.

Only the standard library and NumPy are used. No parent code is imported.
"""

from collections import deque
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PLUS = ((0, 0),) + DIRS


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending, group = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    pending.append(p)
                    group.append(p)
        groups.append(group)
    return groups


def _bbox(points):
    rr, cc = zip(*points)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _centers(field, signal, marker):
    ink = (field == signal) | (field == marker)
    hit = (ink[1:-1, 1:-1] & ink[:-2, 1:-1] & ink[2:, 1:-1]
           & ink[1:-1, :-2] & ink[1:-1, 2:])
    return {tuple(map(int, point + 1)) for point in np.argwhere(hit)}


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr == 0 and dc:
        return 0, 1 if dc > 0 else -1
    if dc == 0 and dr:
        return 1 if dr > 0 else -1, 0
    return None


def _reachable(centers, start, directions):
    """Distances in the directed center-to-center ray graph, in relay hops."""
    distance = {start: 0}
    pending = deque([start])
    ordered = sorted(centers)
    while pending:
        point = pending.popleft()
        for other in ordered:
            if other not in distance and _direction(point, other) in directions:
                distance[other] = distance[point] + 1
                pending.append(other)
    return distance


def _read_input(grid):
    """Recover all roles, geometry, and ports solely from the supplied input."""
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("the canvas must have a background border")
    groups = _components(grid != bg)
    rectangles = []
    for group in groups:
        r0, r1, c0, c1 = _bbox(group)
        if r1 - r0 >= 3 and c1 - c0 >= 3 and len(group) == (r1-r0)*(c1-c0):
            rectangles.append((group, (r0, r1, c0, c1)))
    if len(groups) != 2 or len(rectangles) != 1:
        raise ValueError("expected one filled field and one detached key")
    panel, box = rectangles[0]
    key = next(group for group in groups if group is not panel)
    r0, r1, c0, c1 = box
    field = grid[r0:r1, c0:c1]
    field_colors = set(map(int, np.unique(field)))
    key_colors = {int(grid[p]) for p in key}
    shared = field_colors & key_colors
    if len(field_colors) != 3 or len(key_colors) != 2 or len(shared) != 1:
        raise ValueError("ambiguous field/key color roles")
    marker = shared.pop()
    ray = next(iter(key_colors - {marker}))
    starts = np.argwhere(field == marker)
    if len(starts) != 1:
        raise ValueError("expected one initial marker")
    start = tuple(map(int, starts[0]))
    sr, sc = start
    if not (0 < sr < field.shape[0]-1 and 0 < sc < field.shape[1]-1):
        raise ValueError("the marked plus must be complete")
    neighbors = {int(field[sr+dr, sc+dc]) for dr, dc in DIRS}
    if len(neighbors) != 1:
        raise ValueError("the marker must have four signal neighbors")
    signal = neighbors.pop()
    rest = field_colors - {signal, marker}
    if len(rest) != 1:
        raise ValueError("ambiguous empty-floor color")
    fill = rest.pop()
    if len({bg, fill, signal, marker, ray}) != 5:
        raise ValueError("the five roles must be distinct")
    key_cells = {p: int(grid[p]) for p in key}
    anchors = [p for p in key if all(
        key_cells.get((p[0]+dr, p[1]+dc)) == marker for dr, dc in PLUS)]
    if len(anchors) != 1:
        raise ValueError("ambiguous charged-plus key anchor")
    kr, kc = anchors[0]
    directions = tuple(d for d in DIRS if all(
        key_cells.get((kr+n*d[0], kc+n*d[1])) == ray for n in (2, 3)))
    expected = {(kr+dr, kc+dc): marker for dr, dc in PLUS}
    for dr, dc in directions:
        for n in (2, 3):
            expected[kr+n*dr, kc+n*dc] = ray
    if not directions or key_cells != expected:
        raise ValueError("key must contain a charged plus and complete two-cell ports")
    centers = _centers(field, signal, marker)
    if start not in centers:
        raise ValueError("marked position is not a plus")
    distance = _reachable(centers, start, directions)
    return dict(bg=bg, fill=fill, signal=signal, marker=marker, ray=ray,
                box=box, key=key, field=field, start=start, directions=directions,
                centers=centers, distance=distance)


def _masks(facts):
    field = facts["field"]
    union = np.zeros(field.shape, dtype=bool)
    rays = np.zeros(field.shape, dtype=bool)
    for r, c in facts["distance"]:
        for dr, dc in PLUS:
            union[r+dr, c+dc] = True
        for dr, dc in facts["directions"]:
            rr, cc = r+dr, c+dc
            while 0 <= rr < field.shape[0] and 0 <= cc < field.shape[1]:
                rays[rr, cc] = True
                rr, cc = rr+dr, cc+dc
    return union, rays


def _answer(grid, facts):
    output = grid.copy()
    for p in facts["key"]:
        output[p] = facts["bg"]
    r0, r1, c0, c1 = facts["box"]
    target = output[r0:r1, c0:c1]
    union, rays = _masks(facts)
    target[rays & (facts["field"] == facts["fill"])] = facts["ray"]
    target[union] = facts["marker"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _answer(grid, _read_input(grid))


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _signature(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _paint_plus(field, point, color):
    for dr, dc in PLUS:
        field[point[0]+dr, point[1]+dc] = color


def _build_pair(rng, profile, edge=False, require_selective=False):
    """Construct a backbone, then rescan the final noisy field, never just intent."""
    # Canonical R/D ports are subsequently rotated/reflected with the whole scene.
    if profile == "simple":
        directions = [(0, 1)]
        if rng.random() < .5:
            directions.append((0, -1))
    else:
        directions = [(0, 1), (1, 0)]
        for d in ((-1, 0), (0, -1)):
            if rng.random() < .45:
                directions.append(d)
        if require_selective and len(directions) == 4:
            directions.remove(((-1, 0), (0, -1))[int(rng.integers(2))])
    ports = tuple(directions)
    offsets = set(PLUS) | {(n*dr, n*dc) for dr, dc in ports for n in (2, 3)}
    kh = max(r for r, c in offsets) - min(r for r, c in offsets) + 1
    kw = max(c for r, c in offsets) - min(c for r, c in offsets) + 1
    bg, fill, signal, marker, ray = map(int, rng.choice(10, 5, replace=False))
    for _attempt in range(220):
        deep = profile == "deep"
        h = int(rng.integers(12, 27))
        w = int(rng.integers(17 if deep else 14, 28-kw))
        gh = int(rng.integers(6, 9))
        ge = int(rng.integers(3, 6)) if deep else 0
        gv = int(rng.integers(4, 8))
        cmax = min(4, w-gh-ge-2)
        rmax = min(6, h-gv-4)
        if cmax < 1 or rmax < 1:
            continue
        if edge and (-1, 0) not in ports and (0, -1) in ports:
            # Touch the west edge, leaving room northward to make omission of
            # a north port visibly consequential in this demonstration.
            if rmax < 2:
                continue
            r, c = 2, 1
        else:
            r = 1 if edge else int(rng.integers(1, rmax+1))
            c = int(rng.integers(1, cmax+1))
        start, bend, corner = (r, c), (r, c+gh), (r+gv, c+gh)
        desired = {start, bend}
        if profile != "simple":
            desired.add(corner)
        if deep:
            desired.add((r+gv, c+gh+ge))
            if rng.random() < .45 and r+gv+2 < h:
                desired.add((r+gv+1, c+gh+ge))
        elif profile == "overlap":
            variant = int(rng.integers(4))
            desired.add((corner[0]+(1 if variant < 2 else 2), corner[1]))
            if variant in (1, 3) and corner[1]+2 < w:
                desired.add((r, corner[1]+1))
                desired.add((corner[0]+1, corner[1]+1))
        # A dormant full plus with an arm on the initial ray. At that arm is
        # also a four-of-five near miss, missing the arm away from the dormant center.
        decoy_gap = int(rng.integers(3, gh-1))
        side = -1 if r >= 2 and rng.random() < .5 else 1
        dormant, near = (r+side, c+decoy_gap), (r, c+decoy_gap)
        density = float(rng.choice((.04, .12, .23, .35)))
        field = np.where(rng.random((h, w)) < density, signal, fill).astype(np.int8)
        if rng.random() < .35:
            pr = int(rng.integers(1, h-4))
            pc = int(rng.integers(1, w-4))
            patch = field[pr:pr+3, pc:pc+3]
            patch[rng.random(patch.shape) < .6] = signal
        # Incidental additional stations need not be isolated objects.
        for _ in range(int(rng.integers(0, 4))):
            desired.add((int(rng.integers(1, h-1)), int(rng.integers(1, w-1))))
        for point in sorted(desired | {dormant}):
            _paint_plus(field, point, signal)
        for dr, dc in PLUS:
            field[near[0]+dr, near[1]+dc] = signal
        # Connect the decoy's ink to the launch plus without completing its
        # missing arm. This defeats a connected-component activation shortcut.
        field[r, c+1:c+decoy_gap+2] = signal
        field[r-side, c+decoy_gap] = fill
        field[start] = marker
        centers = _centers(field, signal, marker)
        distance = _reachable(centers, start, ports)
        if not desired.issubset(centers) or dormant not in centers or dormant in distance:
            continue
        if near in centers or len(centers) > 45:
            continue
        max_depth = max(distance.values())
        if deep and max_depth < 3:
            continue
        if not deep and max_depth > 2:
            continue
        if profile in ("elbow", "overlap") and distance.get(corner) != 2:
            continue
        # Recoverable margins and independently varying field/key placement.
        canvas_h = h + int(rng.integers(2, min(4, 30-h)+1))
        canvas_w = w + kw + int(rng.integers(3, min(5, 30-w-kw)+1))
        grid = np.full((canvas_h, canvas_w), bg, dtype=np.int8)
        fr = int(rng.integers(1, canvas_h-h))
        fc = int(rng.integers(1, canvas_w-w-kw-1))
        grid[fr:fr+h, fc:fc+w] = field
        key_top = int(rng.integers(1, canvas_h-kh))
        key_left = int(rng.integers(fc+w+1, canvas_w-kw))
        kr = key_top - min(x for x, y in offsets)
        kc = key_left - min(y for x, y in offsets)
        for dr, dc in PLUS:
            grid[kr+dr, kc+dc] = marker
        for dr, dc in ports:
            for n in (2, 3):
                grid[kr+n*dr, kc+n*dc] = ray
        grid = np.rot90(grid, int(rng.integers(4))).copy()
        if rng.random() < .5:
            grid = np.fliplr(grid).copy()
        facts = _read_input(grid)
        # Ray paint must visibly travel beyond the local key's radius.
        union, rays = _masks(facts)
        feat = _features(facts)
        if feat["unmatched"] < 4 or feat["ray_cells"] < 3:
            continue
        if not feat["long_ray"]:
            continue
        if not feat["component_counterexample"]:
            continue
        if not _ablation_differences(grid, facts):
            continue
        if require_selective and not _ports_matter(grid, facts):
            continue
        return {"input": grid, "output": _answer(grid, facts)}
    raise RuntimeError(f"could not build a {profile} relay field")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    profiles = ["elbow", "overlap"]
    profiles.extend(str(rng.choice(("simple", "elbow", "overlap"))) for _ in range(nt-2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, plan in (("train", profiles), ("test", ["deep"]*ns)):
        for i, profile in enumerate(plan):
            for _ in range(30):
                pair = _build_pair(rng, profile, edge=(section == "train" and i == 0),
                                   require_selective=(section == "train" and i == 0))
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(facts):
    f, sig, mark = facts["field"], facts["signal"], facts["marker"]
    centers, distance = facts["centers"], facts["distance"]
    union, rays = _masks(facts)
    local_rays = np.zeros(f.shape, dtype=bool)
    for r, c in distance:
        for dr, dc in facts["directions"]:
            for n in range(1, 4):
                rr, cc = r+n*dr, c+n*dc
                if 0 <= rr < f.shape[0] and 0 <= cc < f.shape[1]:
                    local_rays[rr, cc] = True
    long_ray = bool(np.any(rays & (f == facts["fill"]) & ~local_rays))
    dormant = centers - set(distance)
    crossed_dormant = any(any(rays[r+dr, c+dc] for dr, dc in DIRS) for r, c in dormant)
    near_on_ray = False
    for r, c in np.argwhere(rays):
        r, c = int(r), int(c)
        if 0 < r < f.shape[0]-1 and 0 < c < f.shape[1]-1:
            count = sum(f[r+dr, c+dc] in (sig, mark) for dr, dc in PLUS)
            near_on_ray |= count == 4 and (r, c) not in distance
    active = set(distance)
    overlap = any(abs(a[0]-b[0])+abs(a[1]-b[1]) <= 2
                  for a in active for b in active if a != b)
    ink_groups = _components((f == sig) | (f == mark))
    launch_group = next(group for group in ink_groups if facts["start"] in group)
    component_counterexample = any(not union[p] for p in launch_group)
    return dict(depth=max(distance.values()), dormant=len(dormant),
                crossed_dormant=crossed_dormant, near_on_ray=near_on_ray,
                overlap=overlap, active=len(active), centers=len(centers),
                component_counterexample=component_counterexample,
                long_ray=long_ray,
                unmatched=int(np.count_nonzero((f == sig) & ~union)),
                ray_cells=int(np.count_nonzero(rays & (f == facts["fill"]))),
                edge=bool(union[0].any() or union[-1].any()
                          or union[:, 0].any() or union[:, -1].any()))


def _ablation_differences(grid, facts):
    """Compare concrete alternate answers, rather than just latent intentions."""
    expected = _answer(grid, facts)
    f = facts["field"]
    # Without B, treat any interior ink cell as a relay center. Stamps and rays
    # still follow A's key. This intentionally wrong detector admits near misses.
    false_centers = {(r, c) for r in range(1, f.shape[0]-1)
                     for c in range(1, f.shape[1]-1)
                     if f[r, c] in (facts["signal"], facts["marker"])}
    without_b = dict(facts, distance=_reachable(false_centers, facts["start"], facts["directions"]))
    # Without A's relay rule, charge every actual plus, as in parent B.
    without_a = dict(facts, distance={p: 0 for p in facts["centers"]})
    # Also distinguish doing only the initial stamp's rays, with no relaying.
    root_only = dict(facts, distance={facts["start"]: 0})
    return (not np.array_equal(_answer(grid, without_b), expected)
            and not np.array_equal(_answer(grid, without_a), expected)
            and (max(facts["distance"].values()) < 2
                 or not np.array_equal(_answer(grid, root_only), expected)))


def _ports_matter(grid, facts):
    all_ports = dict(facts, directions=DIRS,
                     distance=_reachable(facts["centers"], facts["start"], DIRS))
    return not np.array_equal(_answer(grid, all_ports), _answer(grid, facts))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    selective_training = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = _signature(a)
            assert key not in seen
            seen.add(key)
            facts = _read_input(a)
            expected = _answer(a, facts)
            assert np.array_equal(b, expected)
            feat = _features(facts)
            assert feat["dormant"] and feat["crossed_dormant"] and feat["near_on_ray"]
            assert feat["component_counterexample"]
            assert feat["long_ray"]
            assert feat["unmatched"] >= 4 and feat["ray_cells"] >= 3
            union, rays = _masks(facts)
            r0, r1, c0, c1 = facts["box"]
            out_field = b[r0:r1, c0:c1]
            # B's union is exact, including original matches in large components.
            assert np.array_equal(out_field == facts["marker"], union)
            untouched = ~union & ((facts["field"] != facts["fill"]) | ~rays)
            assert np.array_equal(out_field[untouched], facts["field"][untouched])
            assert all(b[p] == facts["bg"] for p in facts["key"])
            assert _ablation_differences(a, facts)
            if section == "train":
                selective_training |= _ports_matter(a, facts)
            # At least one dormant stencil has cells outside the charged union.
            all_union = np.zeros(facts["field"].shape, dtype=bool)
            for r, c in facts["centers"]:
                for dr, dc in PLUS:
                    all_union[r+dr, c+dc] = True
            assert np.any(all_union & ~union)
            features[section].append(feat)
    assert sum(f["depth"] == 2 for f in features["train"]) >= 2
    assert any(f["overlap"] for f in features["train"])
    assert any(f["edge"] for f in features["train"])
    assert selective_training
    assert all(f["depth"] <= 2 for f in features["train"])
    assert all(f["depth"] >= 3 for f in features["test"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """Bounded semantic, count, determinism, and 42-seed variation checks."""
    shapes, palettes, structures, ports, depths, center_counts = (set() for _ in range(6))
    signatures, field_shapes = set(), set()
    observations = dict(overlap=0, edge=0, multihop=0, crossed_dormant=0, near_on_ray=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p["input"]) for p in puzzle["train"]+puzzle["test"]))
        for pair in puzzle["train"]+puzzle["test"]:
            a, b = pair["input"], pair["output"]
            assert np.array_equal(solve(a), b)
            facts = _read_input(a)
            feat = _features(facts)
            shapes.add(a.shape)
            field_shapes.add(facts["field"].shape)
            palettes.add(tuple(facts[k] for k in ("bg", "fill", "signal", "marker", "ray")))
            ports.add(tuple(sorted(facts["directions"])))
            depths.add(feat["depth"])
            center_counts.add(feat["centers"])
            roles = {}
            structures.add((a.shape, bytes(roles.setdefault(int(x), len(roles)) for x in a.flat)))
            for name in observations:
                observations[name] += (feat["depth"] >= 2) if name == "multihop" else feat[name]
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(field_shapes) >= 40
    assert len(palettes) >= 100 and len(structures) >= 200
    assert len(ports) >= 10 and {1, 2, 3}.issubset(depths)
    assert len(center_counts) >= 8
    assert observations["overlap"] >= 42 and observations["edge"] >= 42
    assert observations["multihop"] >= 126
    assert observations["crossed_dormant"] == 210 and observations["near_on_ray"] == 210
    p = generate_puzzle(8675309, 6, 3)
    assert _equal(p, generate_puzzle(8675309, 6, 3))
    a, b = p["test"][0]["input"], p["test"][0]["output"]
    saved = a.copy()
    permutation = np.random.default_rng(123).permutation(10)
    assert np.array_equal(solve(permutation[a]), permutation[b])
    for k in range(4):
        assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
        assert np.array_equal(solve(np.fliplr(np.rot90(a, k))), np.fliplr(np.rot90(b, k)))
    assert np.array_equal(a, saved)
    # Every legal explicit count combination, with no assumption about defaults.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            q = generate_puzzle(10000+nt*10+ns, nt, ns)
            assert len(q["train"]) == nt and len(q["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Hand-built graph probe: turning needs a relay; an arm crossing is no edge.
    centers = {(2, 2), (2, 7), (6, 7), (6, 10), (3, 5)}
    dist = _reachable(centers, (2, 2), ((0, 1), (1, 0)))
    assert dist == {(2, 2): 0, (2, 7): 1, (6, 7): 2, (6, 10): 3}
    # Mutating an output must be detected by the family-specific validator.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed background cell")
    return dict(seeds=42, pairs=210, shapes=len(shapes), field_shapes=len(field_shapes),
                palettes=len(palettes), structures=len(structures), port_sets=len(ports),
                depths=sorted(depths), center_counts=sorted(center_counts), **observations)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
