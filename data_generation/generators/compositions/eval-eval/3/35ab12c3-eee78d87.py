"""Interference cloth: two translated, transparent, periodically woven domains.

All solving conventions are fixed here and visible in the input's four roles.
No parent generator, saved example, seed, or puzzle context is used by solve().
"""
from itertools import permutations

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
ORDERS = tuple((t, r, c) for t in (False, True)
               for r in permutations(range(3)) for c in permutations(range(3)))
DIRECT = ORDERS.index((False, (1, 2, 0), (1, 2, 0)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    _require(n > 0 and (not dr or not dc or abs(dr) == abs(dc)), "unaligned link")
    return [(a[0] + k * (dr // n), a[1] + k * (dc // n)) for k in range(n + 1)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or
           max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _infer_contour(points):
    """Unique noncrossing full cycle, including unit edges; never all chords."""
    n = len(points)
    _require(4 <= n <= 12 and len(set(points)) == n, "contour needs 4..12 unique vertices")
    candidates = []
    rasters = []
    point_set = set(points)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = points[i], points[j]
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if dr and dc and dr != dc:
                continue
            cells = _line(a, b)
            if not point_set.intersection(cells[1:-1]):
                candidates.append((i, j))
                rasters.append(cells)
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(candidates):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    _require(all(len(x) >= 2 for x in adjacency), "no full contour")
    conflicts = [set() for _ in candidates]
    for k, (i, j) in enumerate(candidates):
        for ell, (a, b) in enumerate(candidates[:k]):
            if not {i, j}.intersection((a, b)) and _intersects(points[i], points[j], points[a], points[b]):
                conflicts[k].add(ell)
                conflicts[ell].add(k)
    solutions = {}
    visits = 0

    def visit(order, used, chosen):
        nonlocal visits
        visits += 1
        _require(visits <= 60000, "contour search budget exceeded")
        if len(solutions) > 1:
            return
        if len(order) == n:
            for v, edge in adjacency[order[-1]]:
                if v == 0 and not conflicts[edge].intersection(chosen):
                    solutions[tuple(sorted(chosen + [edge]))] = order
            return
        for v, edge in adjacency[order[-1]]:
            if v not in used and not conflicts[edge].intersection(chosen):
                visit(order + [v], used | {v}, chosen + [edge])

    visit([0], {0}, [])
    _require(len(solutions) == 1, "contour is absent or ambiguous")
    chosen, order = next(iter(solutions.items()))
    boundary = set().union(*(set(rasters[k]) for k in chosen))
    all_links = set().union(*(set(x) for x in rasters))
    vertices = [points[k] for k in order]
    turns = {_cross(vertices[i-1], vertices[i], vertices[(i+1) % n]) for i in range(n)}
    return {"polygon": vertices, "boundary": boundary,
            "rejected": all_links - boundary,
            "concave": any(x < 0 for x in turns) and any(x > 0 for x in turns),
            "diagonal": any(a[0] != b[0] and a[1] != b[1]
                            for a, b in zip(vertices, vertices[1:] + vertices[:1])),
            "unit": any(len(rasters[k]) == 2 for k in chosen)}


def _inside(point, polygon):
    """Strict interior for non-boundary cell centers; horizontal ray parity."""
    r, c = point
    parity = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            x = a[1] + (r - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            if c < x:
                parity = not parity
    return parity


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8 and
            all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())))


MASKS = tuple(m for code in range(1, 511)
              if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                         dtype=bool).reshape(3, 3)))


def _analyze(inp):
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 4 and counts.max() * 2 > inp.size, "need four roles and majority carrier")
    bg = int(colors[np.argmax(counts)])
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(inp == c)]
              for c in colors if c != bg}
    small, large, singleton = [], [], []
    for color, pts in groups.items():
        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        span = hi - lo + 1
        if len(pts) == 1:
            singleton.append((color, pts[0]))
        elif tuple(span) == (3, 3):
            small.append((color, tuple(map(int, lo))))
        elif min(span) >= 7:
            large.append((color, pts))
    _require(len(small) == len(large) == len(singleton) == 1, "ambiguous role grouping")
    source, points = large[0]
    accent, (mr, mc) = small[0]
    echo, marker = singleton[0]
    mask = inp[mr:mr+3, mc:mc+3] == accent
    _require(_full_box(mask), "invalid motif")
    contour = _infer_contour(points)
    anchors = [p for p in points if max(abs(p[0]-marker[0]), abs(p[1]-marker[1])) == 1]
    _require(len(anchors) == 1, "marker must have one adjacent vertex")
    dr, dc = marker[0] - anchors[0][0], marker[1] - anchors[0][1]
    shape = inp.shape
    border = np.zeros(shape, dtype=bool)
    interior = np.zeros(shape, dtype=bool)
    for p in contour["boundary"]:
        border[p] = True
    lo_r, lo_c = np.min(points, axis=0).tolist()
    hi_r, hi_c = np.max(points, axis=0).tolist()
    for r in range(lo_r, hi_r + 1):
        for c in range(lo_c, hi_c + 1):
            if not border[r, c] and _inside((r, c), contour["polygon"]):
                interior[r, c] = True
    _require(bool(interior[mr:mr+3, mc:mc+3].all()), "motif box must lie strictly inside contour")
    _require(not (border | interior)[marker], "marker must be outside source domain")
    domain_points = np.argwhere(border | interior)
    shifted = domain_points + np.array((dr, dc))
    _require(bool(np.all(shifted >= 0) and np.all(shifted < np.array(shape))), "translated domain is clipped")
    return dict(contour, bg=bg, source=source, accent=accent, echo=echo, marker=marker,
                offset=(dr, dc), mask=mask, motif_origin=(mr, mc), origin=(lo_r, lo_c),
                border=border, interior=interior, points=points)


def _shift(mask, offset):
    result = np.zeros_like(mask)
    coords = np.argwhere(mask) + np.array(offset)
    if len(coords):
        _require(bool(np.all(coords >= 0) and np.all(coords < np.array(mask.shape))), "clipped shift")
        result[coords[:, 0], coords[:, 1]] = True
    return result


def _layers(info, order=DIRECT, *, solid=False, move_phase=True, sample_anchor=False):
    mask = info["mask"]
    transpose, row_order, col_order = ORDERS[order]
    tile = (mask.T if transpose else mask)[np.ix_(row_order, col_order)]
    rr, cc = np.indices(info["interior"].shape)
    r0, c0 = info["motif_origin"] if sample_anchor else info["origin"]
    occupancy = tile[(rr-r0) % 3, (cc-c0) % 3]
    cloth = info["interior"] & (True if solid else occupancy)
    other = _shift(cloth, info["offset"]) if move_phase else (
        _shift(info["interior"], info["offset"]) & (True if solid else occupancy))
    return cloth, other


def _render(info, order=DIRECT, *, solid=False, move_phase=True, sample_anchor=False,
            chord_border=False, box_domain=False, fuse=True):
    if box_domain:
        info = dict(info)
        interior = np.zeros_like(info["interior"])
        coords = np.argwhere(info["border"])
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        interior[lo[0]+1:hi[0], lo[1]+1:hi[1]] = True
        info["interior"] = interior & ~info["border"]
    cloth, other = _layers(info, order, solid=solid, move_phase=move_phase, sample_anchor=sample_anchor)
    result = np.full(info["border"].shape, info["bg"], dtype=np.int8)
    result[other] = info["echo"]
    result[cloth] = info["source"]
    if fuse:
        result[cloth & other] = info["accent"]
    border = info["border"].copy()
    if chord_border:
        for p in info["rejected"]:
            border[p] = True
    result[_shift(border, info["offset"])] = info["echo"]
    result[border] = info["source"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the four visibly encoded roles and geometric conventions."""
    return _render(_analyze(_grid(input_grid)))


def _orient(points, code):
    result = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    lo = np.min(result, axis=0)
    return [(r-int(lo[0]), c-int(lo[1])) for r, c in result]


def _propose_shape(rng, family):
    h, w = (int(rng.integers(7, 18)) for _ in range(2))
    if family == "square":
        w = h
    if family in ("rectangle", "square"):
        points = [(0, 0), (0, w), (h, w), (h, 0)]
    elif family == "notch":
        a = int(rng.integers(2, w-3))
        b = int(rng.integers(a+1, w-1))
        depth = int(rng.integers(1, max(2, h-4)))
        points = [(0, 0), (0, w), (h, w), (h, b),
                  (h-depth, b), (h-depth, a), (h, a), (h, 0)]
    elif family == "elbow":
        a, b = (int(rng.integers(4, n-1)) for n in (h, w))
        points = [(0, 0), (0, w), (a, w), (a, b), (h, b), (h, 0)]
    elif family == "stair":
        a, b = int(rng.integers(3, h-3)), int(rng.integers(3, w-3))
        points = [(0, 0), (0, w), (a, w), (a, b+1),
                  (a+2, b+1), (a+2, b), (h, b), (h, 0)]
    elif family == "diamond":
        a, b = int(rng.integers(4, 10)), int(rng.integers(4, 10))
        points = [(0, a), (a, 0), (a+b, b), (b, a+b)]
    else:  # Independently clipped corners; asymmetric octagons through rectangles.
        cuts = [int(rng.integers(0, 4)) for _ in range(4)]
        if not any(cuts):
            cuts[0] = 1
        a, b, c, d = cuts
        points = [(0, a), (0, w-b), (b, w), (h-c, w),
                  (h, w-c), (h, d), (h-d, 0), (a, 0)]
        points = [p for i, p in enumerate(points) if p != points[i-1]]
    return _orient(points, int(rng.integers(8)))


def _shape(rng, family):
    for _ in range(200):
        points = _propose_shape(rng, family)
        try:
            contour = _infer_contour(points)
        except ValueError:
            continue
        intended = set().union(*(set(_line(a, b)) for a, b in zip(points, points[1:] + points[:1])))
        if contour["boundary"] != intended:
            continue
        if family in ("notch", "elbow", "stair") and not contour["concave"]:
            continue
        h, w = np.max(points, axis=0) + 1
        interior = {(r, c) for r in range(int(h)) for c in range(int(w))
                    if (r, c) not in contour["boundary"] and _inside((r, c), contour["polygon"])}
        boxes = [(r, c) for r in range(int(h)-2) for c in range(int(w)-2)
                 if all((r+dr, c+dc) in interior for dr in range(3) for dc in range(3))]
        if boxes:
            return points, contour, interior, boxes
    raise RuntimeError("bounded contour proposal space exhausted")


def _choose_mask(rng, densities):
    density = int(rng.choice(densities))
    options = [m for m in MASKS if int(m.sum()) == density]
    return options[int(rng.integers(len(options)))].copy()


def _make_input(rng, family, mask, offsets, palette):
    points, contour, interior, boxes = _shape(rng, family)
    domain = contour["boundary"] | interior
    anchors = []
    for dr, dc in offsets:
        for v in points:
            marker = (v[0]+dr, v[1]+dc)
            if marker not in domain and sum(max(abs(marker[0]-p[0]), abs(marker[1]-p[1])) == 1
                                           for p in points) == 1:
                anchors.append((v, (dr, dc)))
    _require(bool(anchors), "no marker anchor")
    vertex, offset = anchors[int(rng.integers(len(anchors)))]
    footprint = domain | {(r+offset[0], c+offset[1]) for r, c in domain}
    lo, hi = np.min(list(footprint), axis=0), np.max(list(footprint), axis=0)
    extent = hi-lo+1
    pad_r = int(rng.integers(0, min(6, 31-int(extent[0]))))
    pad_c = int(rng.integers(0, min(6, 31-int(extent[1]))))
    before_r, before_c = int(rng.integers(pad_r+1)), int(rng.integers(pad_c+1))
    sr, sc = before_r-int(lo[0]), before_c-int(lo[1])
    shape = int(extent[0])+pad_r, int(extent[1])+pad_c
    bg, source, echo, accent = palette
    inp = np.full(shape, bg, dtype=np.int8)
    for r, c in points:
        inp[r+sr, c+sc] = source
    inp[vertex[0]+offset[0]+sr, vertex[1]+offset[1]+sc] = echo
    # The sample's location is independent of its occupancy and the copy offset.
    r, c = boxes[int(rng.integers(len(boxes)))]
    inp[r+sr:r+sr+3, c+sc:c+sc+3][mask] = accent
    return inp


def _key(a):
    return a.shape, a.tobytes()


def _matching_orders(inp, out, current):
    info = _analyze(inp)
    return {k for k in current if np.array_equal(_render(info, k), out)}


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
             and low <= value <= high, f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("diagonal_geometry", "denser", "sparser", "new_offset")))
    held = OFFSETS[int(rng.integers(len(OFFSETS)))]
    train_offsets = tuple(d for d in OFFSETS if regime != "new_offset" or d != held)
    train_density = (2, 3, 4) if regime == "denser" else (6, 7, 8) if regime == "sparser" else tuple(range(2, 9))
    test_density = (6, 7, 8) if regime == "denser" else (2, 3, 4) if regime == "sparser" else tuple(range(2, 9))
    orthogonal = ("rectangle", "square", "notch", "elbow", "stair")
    all_families = orthogonal + ("clipped", "diamond")
    puzzle = {"train": [], "test": []}
    used, candidates, train_infos = set(), set(range(len(ORDERS))), []
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            # Two complementary witnesses suffice even for the minimum 2/1 request.
            if split == "train" and index == 0:
                family = "square"  # Contains diagonal chords that must be rejected.
            elif split == "train" and index == 1:
                family = str(rng.choice(("notch", "elbow", "stair")))
            else:
                families = (("clipped", "diamond") if split == "test" and regime == "diagonal_geometry"
                            else orthogonal if split == "train" and regime == "diagonal_geometry" else all_families)
                family = str(rng.choice(families))
            offsets = train_offsets if split == "train" else (held,) if regime == "new_offset" else OFFSETS
            densities = train_density if split == "train" else test_density
            for attempt in range(250):
                mask = _choose_mask(rng, densities)
                if split == "train" and index == 1 and int(mask.sum()) == int(train_infos[0]["mask"].sum()):
                    continue
                palette = [int(x) for x in rng.choice(10, 4, replace=False)]
                inp = _make_input(rng, family, mask, offsets, palette)
                if _key(inp) in used:
                    continue
                info = _analyze(inp)
                out = _render(info)
                cloth, echo_cloth = _layers(info)
                clear = ~(info["border"] | _shift(info["border"], info["offset"]))
                # Every example visibly contains fusion and both single-owner
                # colors. The first witness also has gaps through BOTH sheets;
                # later motifs may be dense enough to cover each other's holes.
                if not all(np.any(x & clear) for x in (cloth & echo_cloth,
                        cloth & ~echo_cloth, echo_cloth & ~cloth)):
                    continue
                if split == "train" and index == 0 and not np.any(clear &
                        info["interior"] & _shift(info["interior"], info["offset"]) & ~cloth & ~echo_cloth):
                    continue
                if split == "train" and index == 1:
                    remaining = _matching_orders(inp, out, candidates)
                    if remaining != {DIRECT}:
                        continue
                    if np.array_equal(out, _render(info, box_domain=True)):
                        continue
                    if info["offset"] == train_infos[0]["offset"]:
                        continue
                if split == "train" and index == 0:
                    if np.array_equal(out, _render(info, chord_border=True)):
                        continue
                    if np.array_equal(out, _render(info, sample_anchor=True)):
                        continue
                pair = {"input": inp, "output": out}
                break
            else:
                raise RuntimeError(f"bounded example search exhausted: {split}/{index}/{regime}")
            puzzle[split].append(pair)
            used.add(_key(inp))
            if split == "train":
                candidates = _matching_orders(inp, out, candidates)
                train_infos.append(info)
    # Witness identities are not tied to their displayed training indices.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference(info):
    """Cellwise stamping check; does not use _render, _layers, or _shift."""
    h, w = info["border"].shape
    r0, c0 = info["origin"]
    dr, dc = info["offset"]
    source = {}
    for r in range(h):
        for c in range(w):
            p = (r, c)
            if p in info["boundary"]:
                source[p] = "edge"
            elif _inside(p, info["polygon"]) and info["mask"][(r-r0+1) % 3, (c-c0+1) % 3]:
                source[p] = "cloth"
    echo = {(r+dr, c+dc): role for (r, c), role in source.items()}
    result = np.full((h, w), info["bg"], dtype=np.int8)
    for r in range(h):
        for c in range(w):
            p = (r, c)
            a, b = source.get(p), echo.get(p)
            if a == "edge":
                result[p] = info["source"]
            elif b == "edge":
                result[p] = info["echo"]
            elif a and b:
                result[p] = info["accent"]
            elif a:
                result[p] = info["source"]
            elif b:
                result[p] = info["echo"]
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    _require(nt > nq, "train count must exceed test count")
    seen, candidates, training = set(), set(range(len(ORDERS))), []
    witnesses = {k: False for k in ("chords", "box", "solid", "phase", "sample", "fusion", "gaps")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must be retained")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(np.array_equal(_reference(info), out), "incorrect patterned domain, copy, fusion, or priority")
            _require(np.array_equal(solve(inp), out), "solve disagrees")
            _require(np.all(out[info["border"]] == info["source"]), "source contour not retained")
            echo_border = _shift(info["border"], info["offset"])
            _require(np.all(out[echo_border & ~info["border"]] == info["echo"]), "echo contour not retained")
            cloth, other = _layers(info)
            clear = ~(info["border"] | echo_border)
            _require(np.array_equal(out == info["accent"], cloth & other & clear), "fusion must follow motif intersection")
            _require(all(np.any(x & clear) for x in (cloth & other, cloth & ~other, other & ~cloth)),
                     "missing visible fusion or single-owner weave")
            _require(np.all(out[~(info["border"] | echo_border | cloth | other)] == info["bg"]), "unexpected painted carrier")
            if split == "train":
                candidates = _matching_orders(inp, out, candidates)
                training.append(info)
                witnesses["gaps"] |= bool(np.any(clear & info["interior"] &
                    _shift(info["interior"], info["offset"]) & ~cloth & ~other))
                for name, kwargs in (("chords", {"chord_border": True}), ("box", {"box_domain": True}),
                        ("solid", {"solid": True}), ("phase", {"move_phase": False}),
                        ("sample", {"sample_anchor": True}), ("fusion", {"fuse": False})):
                    witnesses[name] |= not np.array_equal(out, _render(info, **kwargs))
    _require(candidates == {DIRECT}, "training does not establish motif orientation/phase")
    _require(all(witnesses.values()), f"missing shortcut witnesses: {witnesses}")
    _require(len({i["offset"] for i in training}) >= 2, "training needs two visible translations")
    _require(any(i["concave"] for i in training), "training needs an indented domain")
    _require(len({int(i["mask"].sum()) for i in training}) >= 2, "training needs different motif densities")


def self_test():
    """Bounded structural/semantic regressions and measured diversity over 42 seeds."""
    geometries, motifs, shapes, offsets, palettes, signatures, vertices = (set() for _ in range(7))
    densities = set()
    role_colors = {k: set() for k in ("bg", "source", "echo", "accent")}
    concave = diagonal = unit = boundary_touch = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze(inp)
                origin = np.min(info["points"], axis=0)
                geometries.add(tuple(sorted((r-int(origin[0]), c-int(origin[1])) for r, c in info["points"])))
                motifs.add(info["mask"].tobytes())
                shapes.add(inp.shape)
                offsets.add(info["offset"])
                palettes.add((info["bg"], info["source"], info["echo"], info["accent"]))
                vertices.add(len(info["points"]))
                densities.add(int(info["mask"].sum()))
                for role in role_colors:
                    role_colors[role].add(info[role])
                concave += info["concave"]
                diagonal += info["diagonal"]
                unit += info["unit"]
                border = info["border"] | _shift(info["border"], info["offset"])
                boundary_touch += bool(border[0].any() or border[-1].any() or border[:, 0].any() or border[:, -1].any())
                # Reversing the vertex enumeration cannot change inferred topology.
                _require(_infer_contour(list(reversed(info["points"])))["boundary"] == info["boundary"], "order dependence")
                _require(np.any(inp != out), "identity example")
        if seed in (0, 17, 41):
            replay = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], replay[s])
                         for k in ("input", "output")), "seed nondeterminism")
            pair = puzzle["train"][0]
            inp, out = pair["input"], pair["output"]
            info = _analyze(inp)
            # Structural roles survive a complete literal-color permutation.
            _require(np.array_equal(solve(9-inp), 9-out), "literal-color dependence")
            # Translating the whole scene translates its object-relative phase.
            padded_input = np.pad(inp, ((1, 0), (1, 0)), constant_values=info["bg"])
            padded_output = np.pad(out, ((1, 0), (1, 0)), constant_values=info["bg"])
            _require(np.array_equal(solve(padded_input), padded_output), "canvas-origin dependence")
            # Moving only the stored sample leaves the material meaning unchanged.
            boxes = [(r, c) for r in range(inp.shape[0]-2) for c in range(inp.shape[1]-2)
                     if (r, c) != info["motif_origin"] and info["interior"][r:r+3, c:c+3].all()]
            _require(bool(boxes), "no sample relocation witness")
            moved = inp.copy()
            moved[moved == info["accent"]] = info["bg"]
            r, c = boxes[0]
            moved[r:r+3, c:c+3][info["mask"]] = info["accent"]
            _require(np.array_equal(solve(moved), out), "sample-location dependence")
    _require(len(MASKS) == 321, "incorrect binary mask space")
    _require(len(signatures) == 42 and len(geometries) >= 100 and len(motifs) >= 80, "insufficient structural variation")
    _require(len(shapes) >= 65 and len(palettes) >= 100 and offsets == set(OFFSETS), "insufficient layout/palette/offset variation")
    _require(densities == set(range(2, 9)) and all(v == set(range(10)) for v in role_colors.values()),
             "missing motif density or literal role colors")
    _require(len(vertices) >= 4 and min(concave, diagonal, unit, boundary_touch) > 0, "missing geometric features")
    # Meaningful negative regression: damage a fusion cell, not arbitrary metadata.
    bad = generate_puzzle(808, 2, 1)
    out = bad["test"][0]["output"]
    info = _analyze(bad["test"][0]["input"])
    p = tuple(np.argwhere(out == info["accent"])[0])
    out[p] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted fusion accepted")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "geometries": len(geometries), "motifs": len(motifs),
            "canvas_shapes": len(shapes), "palettes": len(palettes), "offsets": len(offsets),
            "motif_densities": sorted(densities),
            "vertex_counts": sorted(vertices), "concave_pairs": concave, "diagonal_pairs": diagonal,
            "unit_link_pairs": unit, "boundary_pairs": boundary_touch,
            "semantic_verification": "authored invariants and stamping checks; no independent solver"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
