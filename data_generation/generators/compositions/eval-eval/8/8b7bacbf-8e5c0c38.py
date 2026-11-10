"""Mirror-restored irrigation: authentic walls define competing paint paths."""
from collections import deque
from itertools import combinations
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=True)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        comp, queue = {root}, [root]
        for r, c in queue:
            for dr, dc in D4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    queue.append(q)
        result.append(comp)
    return result


def _cells(a, color):
    return set(map(tuple, np.argwhere(a == color).tolist()))


def _axis(cells, shape):
    """Count supported cells, including self-reflecting cells, in both directions."""
    best_score, winners = -1, []
    for coordinate in (0, 1):
        for doubled in range(2 * shape[coordinate] - 1):
            supported = {p for p in cells if
                         ((doubled - p[0], p[1]) if coordinate == 0
                          else (p[0], doubled - p[1])) in cells}
            score = len(supported)
            if score > best_score:
                best_score, winners = score, [(coordinate, doubled, supported)]
            elif score == best_score:
                winners.append((coordinate, doubled, supported))
    if len(winners) != 1:
        raise ValueError("Wall color has ambiguous maximum reflection support")
    return winners[0]


def _distances(region, start):
    distance = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in D4:
            p = (r + dr, c + dc)
            if p in region and p not in distance:
                distance[p] = distance[(r, c)] + 1
                queue.append(p)
    return distance


def _holes(shape, walls):
    h, w = shape
    air = {(r, c) for r in range(h) for c in range(w)} - walls
    return [comp for comp in _components(air)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)]


def _paint(base, background, walls, sources, metric="path", tie_first=False):
    out = base.copy()
    regions = _holes(base.shape, walls)
    region_info = []
    for region in regions:
        local = sorted((p, color) for p, color in sources.items() if p in region)
        maps = [_distances(region, p) for p, _ in local]
        ties, detours = set(), set()
        for p in sorted(region):
            if not local:
                continue
            path = [d[p] for d in maps]
            direct = [abs(p[0] - q[0]) + abs(p[1] - q[1]) for q, _ in local]
            nearest = [i for i, d in enumerate(path) if d == min(path)]
            manhattan = [i for i, d in enumerate(direct) if d == min(direct)]
            if len(nearest) > 1 and p not in sources:
                ties.add(p)
            if len(nearest) == len(manhattan) == 1 and nearest != manhattan:
                detours.add(p)
            chosen = nearest if metric == "path" else manhattan
            if p not in sources:
                out[p] = local[chosen[0]][1] if len(chosen) == 1 or tie_first else background
        region_info.append(dict(cells=region, sources=local, ties=ties, detours=detours))
    return out, region_info


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size / 2:
        raise ValueError("The input background must occupy a strict majority")
    wall_colors = [c for c in range(10) if c != bg and counts[c] >= 2]
    paint_colors = [c for c in range(10) if c != bg and counts[c] == 1]
    if not wall_colors or any(counts[c] < 8 for c in wall_colors):
        raise ValueError("Each structural color needs at least eight input cells")
    sources = {next(iter(_cells(a, c))): c for c in paint_colors}
    clean, core, noise, axes = a.copy(), set(), set(), {}
    by_color = {}
    for color in wall_colors:
        cells = _cells(a, color)
        direction, doubled, supported = _axis(cells, a.shape)
        axes[color] = (direction, doubled)
        by_color[color] = supported
        core |= supported
        rejected = cells - supported
        noise |= rejected
        for p in rejected:
            clean[p] = bg
    out, regions = _paint(clean, bg, core, sources)
    enclosed = set().union(*(r["cells"] for r in regions)) if regions else set()
    if not sources or not set(sources) <= enclosed:
        raise ValueError("Every pigment source must be inside a recovered closed network")
    for region in regions:
        boundary = {(r + dr, c + dc) for r, c in region["cells"] for dr, dc in D4} & core
        if len({int(clean[p]) for p in boundary}) != 1:
            raise ValueError("An enclosed region must belong to one structural color")
    return out, dict(bg=bg, clean=clean, core=core, by_color=by_color,
                     noise=noise, axes=axes, sources=sources, regions=regions)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct walls and geodesic pigment ownership from the input alone."""
    return _analyze(input_grid)[0]


def _device(rng, small=False, disconnected=False):
    """Mirrored room graph; unequal room sizes/spacing produce varied channels."""
    ny = int(rng.integers(2, 4 if small else 5))
    nx = int(rng.integers(2, 4 if small else 6))
    widths = list(map(int, rng.integers(1, 3 if small else 4, size=(nx + 1) // 2)))
    widths += widths[:nx // 2][::-1]
    heights = list(map(int, rng.integers(1, 3 if small else 4, size=ny)))
    xgaps = list(map(int, rng.integers(1, 3 if small else 4, size=nx // 2)))
    xgaps += xgaps[:(nx - 1) // 2][::-1]
    ygaps = list(map(int, rng.integers(1, 3 if small else 4, size=ny - 1)))
    xs, ys = [1], [1]
    for c in range(nx - 1):
        xs.append(xs[-1] + widths[c] + xgaps[c])
    for r in range(ny - 1):
        ys.append(ys[-1] + heights[r] + ygaps[r])
    boxes, centers, air = {}, {}, set()
    for r in range(ny):
        for c in range(nx):
            box = {(y, x) for y in range(ys[r], ys[r] + heights[r])
                   for x in range(xs[c], xs[c] + widths[c])}
            boxes[(r, c)] = box
            # Horizontal corridors span entire mirrored room widths. For even
            # widths, vertical corridors use two columns so symmetry is exact.
            centers[(r, c)] = (ys[r] + heights[r] // 2, xs[c] + (widths[c] - 1) // 2)
            air |= box
    all_edges = []
    for r in range(ny):
        for c in range(nx):
            for dr, dc in ((1, 0), (0, 1)):
                q = (r + dr, c + dc)
                if q in boxes:
                    all_edges.append(tuple(sorted(((r, c), q))))
    allowed = [e for e in all_edges if not disconnected or e[0][0] != 0 or e[1][0] == 0]
    connected_nodes = {p for p in boxes if not disconnected or p[0] > 0}
    edges = set()
    # Random spanning tree, then reflection: branched routes and cycles coexist.
    visited = {min(connected_nodes)}
    while visited != connected_nodes:
        frontier = [e for e in allowed if (e[0] in visited) != (e[1] in visited)
                    and set(e) <= connected_nodes]
        e = frontier[int(rng.integers(len(frontier)))]
        edges.add(e)
        visited.update(e)
    for e in allowed:
        if rng.random() < .18:
            edges.add(e)
    edges |= {tuple(sorted(((r, nx - 1 - c) for r, c in e))) for e in edges}
    for p, q in sorted(edges):
        r1, c1 = centers[p]
        r2, c2 = centers[q]
        if p[0] == q[0]:
            air.update((r1, c) for c in range(min(c1, c2), max(c1, c2) + 1))
        else:
            cols = [c1] if widths[p[1]] % 2 else [c1, c1 + 1]
            air.update((r, c) for r in range(r1, r2 + 1) for c in cols)
    steps = D4 if rng.random() < .6 else D8
    core = {(r + dr, c + dc) for r, c in air for dr, dc in steps} - air
    top = min(r for r, c in core)
    left = min(c for r, c in core)
    core = {(r - top, c - left) for r, c in core}
    shape = (max(r for r, c in core) + 1, max(c for r, c in core) + 1)
    if max(shape) > (13 if small else 25):
        return None
    try:
        direction, doubled, kept = _axis(core, shape)
    except ValueError:
        return None
    if direction != 1 or kept != core:
        return None
    regions = _holes(shape, core)
    if not regions:
        return None
    return dict(core=core, shape=shape, axis=doubled, regions=regions)


def _choose_sources(rng, regions, count, lesson):
    region = max(regions, key=len)
    if len(region) < count + 5:
        return None
    candidates = sorted(region)
    rng.shuffle(candidates)
    if lesson == 1:
        candidates = candidates[:min(55, len(candidates))]
        maps = {p: _distances(region, p) for p in candidates}
        pairs = list(combinations(candidates, 2))
        rng.shuffle(pairs)
        for p, q in pairs[:500]:
            tied = detour = False
            for t in region - {p, q}:
                d = maps[p][t] - maps[q][t]
                m = abs(t[0] - p[0]) + abs(t[1] - p[1]) - abs(t[0] - q[0]) - abs(t[1] - q[1])
                tied |= d == 0
                detour |= d * m < 0
                if tied and detour:
                    return [p, q]
        return None
    return candidates[:count]


def _place(rng, devices):
    minimum_h = max(d["shape"][0] for d in devices)
    minimum_w = max(d["shape"][1] for d in devices)
    if len(devices) == 1:
        h = int(rng.integers(min(30, minimum_h + 2), min(30, minimum_h + 9) + 1))
        w = int(rng.integers(min(30, minimum_w + 2), min(30, minimum_w + 9) + 1))
    else:
        h, w = int(rng.integers(max(22, minimum_h), 31)), int(rng.integers(max(22, minimum_w), 31))
    placements, boxes = [], []
    for d in devices:
        dh, dw = d["shape"]
        for _ in range(100):
            rr, cc = int(rng.integers(h - dh + 1)), int(rng.integers(w - dw + 1))
            if any(not (rr + dh < r or r + rh < rr or cc + dw < c or c + cw < cc)
                   for r, c, rh, cw in boxes):
                continue
            boxes.append((rr, cc, dh, dw))
            placements.append((rr, cc))
            break
        else:
            return None
    return (h, w), placements


def _move(cells, offset):
    return {(r + offset[0], c + offset[1]) for r, c in cells}


def _noise(rng, core, region, sources, axis, shape, forbidden, require_plug):
    direction, doubled = axis
    r0, r1 = min(r for r, c in core), max(r for r, c in core)
    c0, c1 = min(c for r, c in core), max(c for r, c in core)
    candidates = {(r, c) for r in range(max(0, r0 - 2), min(shape[0], r1 + 3))
                  for c in range(max(0, c0 - 2), min(shape[1], c1 + 3))} - forbidden - core
    candidates = {p for p in candidates if 2 * p[direction] != doubled and
                  ((doubled - p[0], p[1]) if direction == 0 else (p[0], doubled - p[1])) not in core}
    selected = set()
    ordered = sorted(candidates)
    rng.shuffle(ordered)
    if require_plug:
        seed = sources[0]
        for p in ordered:
            if p not in region or p in sources:
                continue
            reached = _distances(region - {p}, seed)
            if len(region - set(reached) - {p}) < 2:
                continue
            try:
                if _axis(core | {p}, shape) != (direction, doubled, core):
                    continue
            except ValueError:
                continue
            selected.add(p)
            break
        if not selected:
            return None
    budget = int(rng.integers(1, 8))
    for p in ordered:
        if len(selected) >= budget:
            break
        if p in selected:
            continue
        reflected = ((doubled - p[0], p[1]) if direction == 0 else (p[0], doubled - p[1]))
        if reflected in selected:
            continue
        try:
            if _axis(core | selected | {p}, shape) == (direction, doubled, core):
                selected.add(p)
        except ValueError:
            pass
    return selected


def _turn_device(device, rotations):
    """Rotate one structural color independently, before packing the scene."""
    d = dict(device)
    core, regions, shape = d["core"], d["regions"], d["shape"]
    for _ in range(rotations):
        width = shape[1]
        core = {(width - 1 - c, r) for r, c in core}
        regions = [{(width - 1 - c, r) for r, c in region} for region in regions]
        shape = (shape[1], shape[0])
    direction, doubled, _ = _axis(core, shape)
    d.update(core=core, regions=regions, shape=shape, axis=(direction, doubled))
    return d


def _pair(rng, lesson, is_test, taught_direction=None, taught_parity=None):
    for _ in range(220):
        count = 1 if lesson is not None else (2 if rng.random() < .32 else 1)
        devices = [_device(rng, small=i > 0, disconnected=lesson == 0)
                   for i in range(count)]
        if any(d is None for d in devices):
            continue
        if taught_parity is not None and devices[0]["axis"] % 2 != taught_parity:
            continue
        # The first device remains canonical until the final whole-grid turn.
        # Additional devices may have their own horizontal or vertical axis.
        devices = [_turn_device(d, 0 if i == 0 else int(rng.integers(4)))
                   for i, d in enumerate(devices)]
        source_sets = []
        for i, d in enumerate(devices):
            n = (1 if lesson == 0 else 2 if lesson == 1 else
                 3 if is_test and i == 0 else int(rng.integers(1, 3)))
            sources = _choose_sources(rng, d["regions"], n, lesson if i == 0 else None)
            if sources is None:
                break
            source_sets.append(sources)
        if len(source_sets) != count:
            continue
        placed = _place(rng, devices)
        if placed is None:
            continue
        shape, offsets = placed
        palette = list(map(int, rng.permutation(10)))
        bg, wall_colors = palette[0], palette[1:1 + count]
        paints = iter(palette[1 + count:])
        clean = np.full(shape, bg, dtype=int)
        sources, cores, regions, axes, excluded = {}, [], [], [], set()
        for i, (d, offset) in enumerate(zip(devices, offsets)):
            core = _move(d["core"], offset)
            cores.append(core)
            regions.append(_move(max(d["regions"], key=len), offset))
            direction, doubled = d["axis"]
            axes.append((direction, doubled + 2 * offset[direction]))
            for p in core:
                clean[p] = wall_colors[i]
            for p in source_sets[i]:
                q = (p[0] + offset[0], p[1] + offset[1])
                sources[q] = next(paints)
                clean[q] = sources[q]
            rr, cc = offset
            dh, dw = d["shape"]
            excluded |= {(r, c) for r in range(rr, rr + dh) for c in range(cc, cc + dw)}
        inp = clean.copy()
        success = True
        for i, core in enumerate(cores):
            own_box = {(r, c) for r in range(offsets[i][0], offsets[i][0] + devices[i]["shape"][0])
                       for c in range(offsets[i][1], offsets[i][1] + devices[i]["shape"][1])}
            forbidden = (excluded - own_box) | set(sources) | set().union(*cores)
            # Earlier noise cannot be overwritten by another structural color.
            forbidden |= set(map(tuple, np.argwhere(inp != clean).tolist()))
            local_sources = sorted(p for p in sources if p in regions[i])
            additions = _noise(rng, core, regions[i], local_sources, axes[i], shape,
                               forbidden, require_plug=lesson == 0)
            if additions is None:
                success = False
                break
            for p in additions:
                inp[p] = wall_colors[i]
        if not success or np.count_nonzero(inp == bg) <= inp.size / 2:
            continue
        intended, _ = _paint(clean, bg, set().union(*cores), sources)
        # Quarter turns and reflections preserve the hydraulic rule; axis
        # orientation is inferred independently for each color from the input.
        rotation = (int(rng.integers(4)) if taught_direction is None
                    else 2 * int(rng.integers(2)) + (1 - taught_direction))
        inp, intended = np.rot90(inp, rotation), np.rot90(intended, rotation)
        if rng.random() < .5:
            inp, intended = np.fliplr(inp), np.fliplr(intended)
        try:
            prediction, info = _analyze(inp)
        except ValueError:
            continue
        if not np.array_equal(prediction, intended):
            continue
        if any(np.count_nonzero(intended == color) < 2 for color in info["sources"].values()):
            continue
        if lesson == 0 and not any(not r["sources"] for r in info["regions"]):
            continue
        if lesson == 0:
            raw = _without_symmetry(inp, info)
            downstream = sum(p not in info["noise"] and raw[p] != intended[p]
                             for region in info["regions"] for p in region["cells"])
            if downstream < 2:
                continue
        if lesson == 1 and not (any(r["ties"] for r in info["regions"])
                                and any(r["detours"] for r in info["regions"])):
            continue
        return {"input": inp.copy(), "output": intended.copy()}
    raise RuntimeError("Bounded sampling could not produce an informative irrigation network")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    first_direction, first_parity = int(rng.integers(2)), int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson = i if split == "train" and i < 2 else None
            for _ in range(20):
                direction = (first_direction + lesson) % 2 if lesson is not None else None
                parity = (first_parity + lesson) % 2 if lesson is not None else None
                pair = _pair(rng, lesson, split == "test", direction, parity)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _without_symmetry(inp, info):
    walls = set().union(*(_cells(inp, c) for c in info["axes"]))
    return _paint(inp, info["bg"], walls, info["sources"])[0]


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            prediction, info = _analyze(inp)
            assert np.array_equal(prediction, out), "Wrong hydraulic ownership or wall repair"
            assert info["noise"], "Generated examples should demonstrate wall repair"
            for color, core in info["by_color"].items():
                assert _cells(out, color) == core, "Supported wall cells must be preserved"
            for p, color in info["sources"].items():
                assert out[p] == color and np.count_nonzero(out == color) >= 2
            allowed = info["noise"] | set().union(*(r["cells"] for r in info["regions"]))
            assert set(map(tuple, np.argwhere(inp != out).tolist())) <= allowed
            for region in info["regions"]:
                for p in region["ties"]:
                    assert out[p] == info["bg"], "Equal arrival times must remain background"
                if not region["sources"]:
                    assert all(out[p] == info["bg"] for p in region["cells"])
            records[split].append(info)
    first, second = records["train"][:2]
    assert {a[0] for info in (first, second) for a in info["axes"].values()} == {0, 1}
    assert {a[1] % 2 for info in (first, second) for a in info["axes"].values()} == {0, 1}
    assert len(first["sources"]) == 1 and any(not r["sources"] for r in first["regions"])
    assert any(r["ties"] for r in second["regions"])
    assert any(r["detours"] for r in second["regions"]), "Training must distinguish path from direct distance"
    for pair, info in zip(puzzle["train"][:2], (first, second)):
        assert not np.array_equal(info["clean"], pair["output"]), "Omitting paint must change the answer"
    raw = _without_symmetry(np.asarray(puzzle["train"][0]["input"]), first)
    first_output = _grid(puzzle["train"][0]["output"])
    changed_air = [p for r in first["regions"] for p in r["cells"]
                   if p not in first["noise"] and raw[p] != first_output[p]]
    assert len(changed_air) >= 2, "A false plug must affect downstream paint, beyond its own deleted cell"
    assert max(len(r["sources"]) for info in records["train"] for r in info["regions"]) <= 2
    assert all(any(len(r["sources"]) == 3 for r in info["regions"]) for info in records["test"])


def self_test():
    # Hand-authored canal: the axis tab survives, the outlier vanishes, and
    # two equidistant fronts leave the central cell blank. This is not a
    # sampled output reused as a fixture.
    input_rows = ("00000000000", "00000000000", "00000600000", "00666666600",
                  "06200000460", "00666666600", "00600000000", "00000000000", "00000000000")
    output_rows = ("00000000000", "00000000000", "00000600000", "00666666600",
                   "06222044460", "00666666600", "00000000000", "00000000000", "00000000000")
    a = np.array([[int(c) for c in row] for row in input_rows])
    b = np.array([[int(c) for c in row] for row in output_rows])
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn))
    shapes, structures, backgrounds, parities, orientations, wall_counts = set(), set(), set(), set(), set(), set()
    cavity_counts, source_counts, noise_counts, cavity_shapes, fingerprints = set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for s in ("train", "test"):
            for pair in puzzle[s]:
                inp = pair["input"]
                out, info = _analyze(inp)
                assert np.array_equal(out, pair["output"])
                shapes.add(inp.shape)
                backgrounds.add(info["bg"])
                wall_counts.add(len(info["axes"]))
                cavity_counts.add(len(info["regions"]))
                source_counts.add(len(info["sources"]))
                noise_counts.add(len(info["noise"]))
                orientations.update(axis[0] for axis in info["axes"].values())
                parities.update(axis[1] % 2 for axis in info["axes"].values())
                roles = {info["bg"]: 0}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in inp.ravel())
                structures.add((inp.shape, normalized))
                for region in info["regions"]:
                    r0 = min(r for r, c in region["cells"])
                    c0 = min(c for r, c in region["cells"])
                    cavity_shapes.add(tuple(sorted((r-r0, c-c0) for r, c in region["cells"])))
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(shapes) >= 45 and backgrounds == set(range(10))
    assert orientations == parities == {0, 1} and wall_counts == {1, 2}
    assert len(cavity_counts) >= 5 and {1, 2, 3, 4} <= source_counts
    assert len(noise_counts) >= 7 and len(cavity_shapes) >= 120
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(10000 + 10*a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal example counts accepted")
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Two meaningful wrong answers: erase a supported wall; paint a neutral seam.
    for mutation in ("wall", "tie"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["train"][1]
        _, info = _analyze(pair["input"])
        if mutation == "wall":
            position = min(info["core"])
            pair["output"][position] = info["bg"]
        else:
            position = min(set().union(*(r["ties"] for r in info["regions"])))
            pair["output"][position] = next(iter(info["sources"].values()))
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("Validator accepted an invalid wall or seam")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                normalized_structures=len(structures), cavity_shapes=len(cavity_shapes),
                backgrounds=sorted(backgrounds), orientations=sorted(orientations),
                axis_parities=sorted(parities), wall_counts=sorted(wall_counts),
                cavity_counts=sorted(cavity_counts), source_counts=sorted(source_counts),
                noise_counts=sorted(noise_counts))


if __name__ == "__main__":
    print(self_test())
