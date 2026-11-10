"""Dock enclosed islands in a gateway-selected chamber, then redraw its coast.

Only NumPy and the standard library are used. All semantic information is visible
in the input. Sampling preferences are never read by solve().
"""
from __future__ import annotations

import numpy as np

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a


def _flood(allowed, start):
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    h, w = allowed.shape
    reached[start] = True
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                stack.append((nr, nc))
    return reached


def _components(mask):
    pending = mask.copy()
    result = []
    while pending.any():
        p = tuple(int(v) for v in np.argwhere(pending)[0])
        component = _flood(pending, p)
        result.append(component)
        pending[component] = False
    return result


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in (_N8 if diagonal else _N4):
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _inner(mask, distance=1):
    result = mask.copy()
    h, w = mask.shape
    for _ in range(distance):
        p = np.pad(result, 1, constant_values=False)
        for dr, dc in _N8:
            result &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _outside(region):
    return _flood(np.pad(~region, 1, constant_values=True), (0, 0))


def _rim(region, diagonal=True):
    outside = _outside(region)
    return _dilate(outside, diagonal)[1:-1, 1:-1] & region


def _checkerboard(mask):
    a, b = mask[:-1, :-1], mask[:-1, 1:]
    c, d = mask[1:, :-1], mask[1:, 1:]
    return bool(np.any((a == d) & (b == c) & (a != b)))


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _analyze(input_grid):
    """Recover and validate the intended domain without construction metadata."""
    grid = _grid(input_grid)
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 3, "exactly three visible roles are required")
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(colors[np.argmax(counts)])
    border = np.zeros(grid.shape, bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    candidates = [int(c) for c, n in zip(colors, counts)
                  if c != bg and 1 <= n <= 2 and np.all(border[grid == c])]
    _require(len(candidates) == 1, "ambiguous gateway color")
    marker = candidates[0]
    solid = int(next(c for c in colors if c not in (bg, marker)))
    _require(np.count_nonzero(grid == solid) > 2, "structure must be distinguishable")
    markers = {}
    for rr, cc in np.argwhere(grid == marker):
        r, c = int(rr), int(cc)
        sides = [s for s, present in (("top", r == 0), ("bottom", r == h - 1),
                                     ("left", c == 0), ("right", c == w - 1)) if present]
        _require(len(sides) == 1 and sides[0] not in markers, "gateways must occupy distinct non-corner sides")
        markers[sides[0]] = (r, c)
    vertical = set(markers) & {"top", "bottom"}
    horizontal = set(markers) & {"left", "right"}
    _require(len(markers) == 1 or (len(vertical) == len(horizontal) == 1), "two gateways must be adjacent")
    start = next(iter(markers.values()))
    target = _flood(grid != solid, start)
    _require(all(target[p] for p in markers.values()), "gateways must select one chamber")
    outer = _outside(target)[1:-1, 1:-1]
    filled = ~outer
    cargo = (grid == solid) & ~outer
    _require(cargo.any(), "chamber has no enclosed cargo")
    _require(np.array_equal(filled & ~target, cargo), "nested free pockets are not allowed")
    _require(not _checkerboard(grid == solid), "corner-only structural joins are not allowed")
    _require(np.all(_inner(filled, 2)[cargo]), "source islands need two-cell exterior clearance")
    coords = np.argwhere(filled)
    r0, c0 = (int(v) for v in coords.min(axis=0))
    r1, c1 = (int(v) for v in coords.max(axis=0))
    frame = r0, r1, c0, c1
    ar = markers[next(iter(horizontal))][0] if horizontal else None
    ac = markers[next(iter(vertical))][1] if vertical else None
    rr, cc = np.indices(grid.shape)
    axes = np.zeros(grid.shape, bool)
    if ar is not None:
        axes |= rr == ar
    if ac is not None:
        axes |= cc == ac
    fixed = cargo & axes
    groups = []
    for v in ("top", "bottom"):
        for s in ("left", "right"):
            if ar is None and v not in markers:
                continue
            if ac is None and s not in markers:
                continue
            mask = cargo & ~axes
            if ar is not None:
                mask &= rr < ar if v == "top" else rr > ar
            if ac is not None:
                mask &= cc < ac if s == "left" else cc > ac
            _require(mask.any(), "all cut halves/quadrants must be nonempty")
            groups.append({"corner": (v, s), "mask": mask, "keep": v in markers or s in markers})
    guide_paths = []
    guide_hits = []
    for side, p in markers.items():
        dr, dc = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}[side]
        ray = []
        r, c = p
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r, c = r + dr, c + dc
        hits = [i for i, q in enumerate(ray) if cargo[q]]
        _require(bool(hits), "each gateway axis must hit enclosed cargo")
        path = ray[:max(hits) + 1]
        _require(all(filled[q] for q in path), "guide cannot leave its chamber")
        guide_paths.append(path)
        guide_hits.append(hits)
    kept_wall = np.zeros_like(cargo)
    erased_components = 0
    for component in _components((grid == solid) & outer):
        if np.any(_dilate(component, False) & target):
            kept_wall |= component
        else:
            erased_components += 1
    info = {"grid": grid, "roles": (bg, solid, marker), "markers": markers,
            "target": target, "filled": filled, "cargo": cargo, "fixed": fixed,
            "frame": frame, "axes": (ar, ac), "groups": groups,
            "guide_paths": guide_paths, "guide_hits": guide_hits,
            "kept_wall": kept_wall, "erased_components": erased_components}
    final = _place_cargo(info, check=True)
    region = filled & ~final
    _require(np.array_equal(_flood(region, start), region), "docking must leave the chamber connected")
    _require(all(region[p] for p in markers.values()), "docking cannot cover a gateway")
    info["final_cargo"] = final
    info["region"] = region
    return info


def _translated(mask, corner, frame):
    r0, r1, c0, c1 = frame
    cells = np.argwhere(mask)
    lo, hi = cells.min(axis=0), cells.max(axis=0)
    dr = (r0 - lo[0]) if corner[0] == "top" else (r1 - hi[0])
    dc = (c0 - lo[1]) if corner[1] == "left" else (c1 - hi[1])
    return cells + np.array([dr, dc])


def _place_cargo(info, frame=None, separate=False, separate_islands=False, move=True, check=False):
    if not move:
        return info["cargo"].copy()
    frame = info["frame"] if frame is None else frame
    result = info["fixed"].copy()
    for group in info["groups"]:
        if not group["keep"]:
            continue
        if separate_islands:
            masks = [component & group["mask"] for component in _components(info["cargo"])
                     if np.any(component & group["mask"])]
        else:
            masks = _components(group["mask"]) if separate else [group["mask"]]
        for mask in masks:
            cells = _translated(mask, group["corner"], frame)
            rows, cols = cells.T
            if check:
                _require(bool(np.all(info["filled"][rows, cols])), "cargo landing outside chamber")
                _require(not result[rows, cols].any(), "cargo landing collision")
            result[rows, cols] = True
    return result


def _render(info, *, canvas=False, nearest=False, separate=False, separate_islands=False,
            wall_hit=False, move=True,
            outline_holes=False, before_docking=False):
    """Mutant switches are used only for diagnostic counterexamples."""
    grid = info["grid"]
    bg, solid, marker = info["roles"]
    frame = (0, grid.shape[0] - 1, 0, grid.shape[1] - 1) if canvas else None
    cargo = _place_cargo(info, frame=frame, separate=separate,
                         separate_islands=separate_islands, move=move)
    region = info["filled"] & ~cargo
    out = np.full(grid.shape, bg, dtype=np.int8)
    out[info["kept_wall"] | cargo] = solid
    for path, hits in zip(info["guide_paths"], info["guide_hits"]):
        if wall_hit:
            first = path[0]
            side = next(s for s, p in info["markers"].items() if p == first)
            dr, dc = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}[side]
            ray = []
            r, c = first
            while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                ray.append((r, c))
                r, c = r + dr, c + dc
            stop = max(i for i, p in enumerate(ray) if grid[p] == solid)
            path = ray[:stop + 1]
        for p in (path[:min(hits) + 1] if nearest else path):
            if out[p] != solid:
                out[p] = marker
    if outline_holes:
        rim = _dilate(np.pad(~region, 1, constant_values=True))[1:-1, 1:-1] & region
    else:
        rim = _rim(info["filled"] & ~info["cargo"] if before_docking else region)
    out[rim & region & (out != solid)] = marker
    for p in info["markers"].values():
        out[p] = marker
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve all inputs satisfying task_spec.md, using input cells only."""
    return _render(_analyze(input_grid))


def _grow(rng, allowed, count):
    points = np.argwhere(allowed)
    if not len(points):
        return None
    r, c = (int(v) for v in points[int(rng.integers(len(points)))])
    cells = {(r, c)}
    for _ in range(count - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4
                           if 0 <= r + dr < allowed.shape[0] and 0 <= c + dc < allowed.shape[1]
                           and allowed[r + dr, c + dc] and (r + dr, c + dc) not in cells})
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    mask = np.zeros_like(allowed)
    for p in cells:
        mask[p] = True
    return mask


def _construct(rng, *, witness, double, island_count, turn, reflect, roles, fallback=False):
    """Build one candidate chamber; no output labels or semantic exceptions here."""
    if fallback:
        h, w = 27, 29
        rh, rw = (13, 13) if witness == 1 else (22, 23)
    else:
        h, w = (int(v) for v in rng.integers(22 if witness == 1 else 17, 31, size=2))
        if witness == 1:
            rh, rw = int(rng.integers(11, 15)), int(rng.integers(11, 15))
        elif witness == 2:
            rh, rw = h - int(rng.integers(1, 4)), w - int(rng.integers(1, 4))
        else:
            rh, rw = int(rng.integers(11, h + 1)), int(rng.integers(11, w + 1))
    c0 = 0 if double else (4 if fallback and witness == 1 else int(rng.integers(0, w - rw + 1)))
    if witness == 1 and not fallback:
        c0 = int(rng.integers(1, w - rw))
    r1, c1 = rh - 1, c0 + rw - 1
    envelope = np.zeros((h, w), bool)
    envelope[:rh, c0:c1 + 1] = True
    filled = envelope.copy()
    # Notches meet the outside but spare the corner landing neighborhoods.
    notches = 0 if fallback else int(rng.integers(0, 4 if witness else 6))
    for _ in range(notches):
        side = str(rng.choice(["left", "right", "bottom", "top"]))
        depth = int(rng.integers(1, 4))
        if side in ("left", "right"):
            start = int(rng.integers(2, rh - 3))
            stop = min(rh - 2, start + int(rng.integers(1, 4)))
            if side == "left":
                filled[start:stop, c0:c0 + depth] = False
            else:
                filled[start:stop, c1 - depth + 1:c1 + 1] = False
        else:
            start = int(rng.integers(c0 + 2, c1 - 2))
            stop = min(c1 - 1, start + int(rng.integers(1, 4)))
            if side == "top":
                filled[:depth, start:stop] = False
            else:
                filled[r1 - depth + 1:r1 + 1, start:stop] = False
    wall = _dilate(envelope) & ~envelope
    thickness = 1 if fallback else int(rng.integers(1, 3))
    if thickness == 2:
        wall |= _dilate(wall) & ~envelope
    wall |= envelope & ~filled
    ac = c0 + 6 if fallback else int(rng.integers(c0 + 4, c1 - 3))
    top = 3 if fallback else int(rng.integers(2, rh - 6))
    bottom = min(r1 - 2, top + (5 if fallback else int(rng.integers(4, 8))))
    ar = (top + bottom) // 2
    cargo = np.zeros((h, w), bool)
    cargo[top:bottom + 1, ac - 1] = True
    teeth = [top, bottom]
    if not fallback and bottom - top >= 6 and rng.random() < .65:
        teeth.append(top + 2)
    for r in teeth:
        length = 2 if fallback else int(rng.integers(1, min(4, c1 - ac - 1)))
        cargo[r, ac:ac + length + 1] = True
        if fallback:
            left_length = (1 if r == top else 2) if double else (2 if r == top else 1)
        else:
            left_length = int(rng.integers(1, min(4, ac - c0 - 1)))
        cargo[r, ac - 1 - left_length:ac - 1] = True
    # Cargo has separate solid components, but all cuts are grouped collectively.
    for index in range(island_count - 1):
        allowed = _inner(filled, 2) & ~_dilate(cargo)
        if fallback and witness == 2 and index == 0:
            addition = np.zeros_like(allowed)
            addition[2:4, 2] = True
            if not np.all(allowed[addition]):
                return None
        else:
            addition = _grow(rng, allowed, 2 if fallback else int(rng.integers(1, 10)))
        if addition is None:
            return None
        cargo |= addition
    if np.any(cargo & ~_inner(filled, 2)):
        return None
    structure = wall | cargo
    # Retained wall branches may extend outside the selected chamber.
    if not fallback and rng.random() < .4:
        available = ~envelope & ~wall
        edge = available & _dilate(wall, False)
        if edge.any():
            p = np.argwhere(edge)[int(rng.integers(np.count_nonzero(edge)))]
            branch = np.zeros_like(wall)
            branch[tuple(p)] = True
            for _ in range(int(rng.integers(1, 5))):
                front = _dilate(branch, False) & available & ~branch
                if not front.any():
                    break
                q = np.argwhere(front)[int(rng.integers(np.count_nonzero(front)))]
                branch[tuple(q)] = True
            structure |= branch
    count = 1 if witness == 1 or fallback else int(rng.integers(0, 5))
    for _ in range(count):
        allowed = ~envelope & ~_dilate(structure)
        addition = _grow(rng, allowed, 3 if fallback else int(rng.integers(1, 10)))
        if addition is None:
            if witness == 1:
                return None
            break
        structure |= addition
    bg, solid, marker = roles
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[structure] = solid
    gateways = [(0, ac)] + ([(ar, 0)] if double else [])
    if any(not filled[p] or structure[p] for p in gateways):
        return None
    for p in gateways:
        grid[p] = marker
    grid = np.rot90(grid, turn).copy()
    return np.fliplr(grid).copy() if reflect else grid


def _features(info, out=None):
    if out is None:
        out = _render(info)
    remaining = _outside(info["region"])[1:-1, 1:-1]
    original_islands = len(_components(info["cargo"]))
    other_regions = _components(info["grid"] != info["roles"][1])
    selected_area = int(info["target"].sum())
    smaller = selected_area < max(int(m.sum()) for m in other_regions)
    mut = {"no_transport": {"move": False}, "canvas_corners": {"canvas": True},
           "nearest_hit": {"nearest": True}, "separate_fragments": {"separate": True},
           "all_holes": {"outline_holes": True}, "old_coast": {"before_docking": True},
           "separate_islands": {"separate_islands": True}, "wall_hit": {"wall_hit": True}}
    failures = {k: not np.array_equal(_render(info, **options), out) for k, options in mut.items()}
    gaps = [b - a - 1 for hits in info["guide_hits"] for a, b in zip(hits, hits[1:]) if b > a + 1]
    deleted = sum(int(g["mask"].sum()) for g in info["groups"] if not g["keep"])
    fragments = max(len(_components(g["mask"])) for g in info["groups"] if g["keep"])
    coast = _rim(info["region"])
    diagonal = int(np.count_nonzero(coast & ~_rim(info["region"], False)))
    # Walls or outside structures farther down a ray must not determine its endpoint.
    noncargo_hit = False
    for side, p in info["markers"].items():
        dr, dc = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}[side]
        path = next(path for path in info["guide_paths"] if path[0] == p)
        r, c = path[-1][0] + dr, path[-1][1] + dc
        while 0 <= r < info["grid"].shape[0] and 0 <= c < info["grid"].shape[1]:
            noncargo_hit |= info["grid"][r, c] == info["roles"][1] and not info["cargo"][r, c]
            r, c = r + dr, c + dc
    r0, r1, c0, c1 = info["frame"]
    return {"shape": info["grid"].shape, "roles": info["roles"], "frame": info["frame"],
            "chamber_shape": (r1 - r0 + 1, c1 - c0 + 1), "marker_count": len(info["markers"]),
            "orientation": tuple(sorted(info["markers"])), "islands": original_islands,
            "smaller": smaller, "selected_area": selected_area, "gaps": tuple(gaps),
            "fragments": fragments, "deleted": deleted, "diagonal": diagonal,
            "fixed_holes": int(np.count_nonzero(info["fixed"] & ~remaining)),
            "erased": info["erased_components"], "ignored_ray_hit": bool(noncargo_hit),
            "notch_area": (r1 - r0 + 1) * (c1 - c0 + 1) - int(info["filled"].sum()),
            "cargo_size": int(info["cargo"].sum()), "guide_spans": tuple(map(len, info["guide_paths"])),
            "failures": failures}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    vary_palette = bool(rng.random() < .65)
    base_turn = int(rng.integers(4))
    orientation_mode = int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index + 1 if split == "train" and index < 2 else 0
            double = witness == 2 if witness else bool(rng.integers(2))
            island_count = 1 if witness == 1 else 2 if witness == 2 else int(rng.integers(3, 5) if split == "test" else rng.integers(1, 3))
            roles = tuple(int(c) for c in rng.choice(10, 3, replace=False)) if vary_palette else palette
            turn = (base_turn if orientation_mode == 0 else (base_turn + (split == "test")) % 4
                    if orientation_mode == 1 else int(rng.integers(4)))
            reflect = bool(rng.integers(2))
            for attempt in range(96):
                grid = _construct(rng, witness=witness, double=double, island_count=island_count,
                                  turn=turn, reflect=reflect, roles=roles, fallback=attempt >= 80)
                if grid is None or _key(grid) in seen:
                    continue
                try:
                    info = _analyze(grid)
                except ValueError:
                    continue
                out = _render(info)
                features = _features(info, out)
                if features["islands"] != island_count:
                    continue
                if witness == 1 and not (features["smaller"] and features["erased"]
                                         and features["ignored_ray_hit"] and all(
                                             value for name, value in features["failures"].items()
                                             if name != "separate_islands")):
                    continue
                if witness == 2 and not (features["deleted"] and features["failures"]["separate_islands"]):
                    continue
                seen.add(_key(grid))
                puzzle[split].append({"input": grid, "output": out})
                break
            else:
                raise RuntimeError("bounded chamber construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned grids and family-specific teaching evidence, not seeds."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            bg, solid, marker = info["roles"]
            assert np.array_equal(out, _render(info))
            assert not np.array_equal(inp, out)
            assert set(np.unique(out)) == {bg, solid, marker}
            assert np.all(out[info["kept_wall"]] == solid)
            assert np.all(out[info["fixed"]] == solid)
            assert all(out[p] == marker for p in info["markers"].values())
            deleted = sum(int(g["mask"].sum()) for g in info["groups"] if not g["keep"])
            assert int((out == solid).sum()) == int(info["kept_wall"].sum() + info["cargo"].sum()) - deleted
            # Guide completion preserves solid hits and never uses a far-side solid as cargo.
            for path in info["guide_paths"]:
                assert all(out[p] in (solid, marker) for p in path)
            f = _features(info, out)
            (training if split == "train" else tests).append(f)
    assert {f["marker_count"] for f in training} == {1, 2}
    assert any(f["smaller"] for f in training)
    assert any(f["deleted"] for f in training)
    assert any(f["erased"] and f["ignored_ray_hit"] for f in training)
    assert any(f["gaps"] and f["fragments"] >= 2 for f in training)
    assert any(f["fixed_holes"] and f["diagonal"] for f in training)
    assert max(f["islands"] for f in training) >= 2
    for shortcut in training[0]["failures"]:
        assert any(f["failures"][shortcut] for f in training), f"missing evidence against {shortcut}"
    assert max(f["islands"] for f in training) < min(f["islands"] for f in tests), "missing count/arrangement test shift"


def self_test():
    """Bounded checks over 42 seeds, counterexamples, and metamorphic changes."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures, shapes, orientations = [], set(), set(), {"train": set(), "test": set()}
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        fs = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, original)
                info = _analyze(inp)
                f = _features(info, out)
                fs.append(f)
                orientations[split].add(f["orientation"])
                shapes.add(f["shape"])
                role_grid = np.zeros_like(inp)
                role_grid[inp == info["roles"][1]] = 1
                role_grid[inp == info["roles"][2]] = 2
                structures.add(_key(role_grid))
        assert len({f["islands"] for f in fs}) >= 3
        assert len({f["frame"] for f in fs}) >= 2
        records.extend(fs)
    assert len(signatures) == 42
    assert len(structures) >= 190 and len(shapes) >= 60
    assert len({f["chamber_shape"] for f in records}) >= 50
    assert len({f["cargo_size"] for f in records}) >= 20
    assert {1, 2, 3, 4} <= {f["islands"] for f in records}
    assert {0, 1, 2} <= {f["erased"] for f in records}
    assert len({f["notch_area"] for f in records}) >= 8
    assert len(orientations["train"]) == 8 and len(orientations["test"]) >= 6
    assert all({f["roles"][k] for f in records} == set(range(10)) for k in range(3))
    assert max(max(f["gaps"], default=0) for f in records) >= 4
    assert max(f["fragments"] for f in records) >= 3
    # Unequal fallback teeth leave an open escape route after docking. Equal
    # teeth on both ends of a spine can seal a free pocket against the wall.
    for witness in (1, 2):
        for turn in range(4):
            for reflection in (False, True):
                inp = _construct(np.random.default_rng(17), witness=witness, double=witness == 2,
                                 island_count=witness, turn=turn, reflect=reflection,
                                 roles=(0, 5, 8), fallback=True)
                f = _features(_analyze(inp))
                assert f["islands"] == witness
                if witness == 1:
                    assert f["smaller"] and f["erased"] and all(
                        v for k, v in f["failures"].items() if k != "separate_islands")
                else:
                    assert f["deleted"] and f["failures"]["separate_islands"]
    for pair in puzzles[0]["train"]:
        for turn in range(4):
            for reflect in (False, True):
                inp = np.rot90(pair["input"], turn).copy()
                out = np.rot90(pair["output"], turn).copy()
                if reflect:
                    inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
                assert np.array_equal(solve(inp), out)
        permutation = np.array([4, 7, 0, 9, 2, 6, 8, 1, 5, 3])
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Inserting an unrelated far-side component must not affect docking or guides.
    pair = next(p for p in puzzles[0]["train"] if
                len(_analyze(p["input"])["markers"]) == 1
                and _features(_analyze(p["input"]))["smaller"])
    info = _analyze(pair["input"])
    bg, solid, _ = info["roles"]
    allowed = ~_dilate(info["filled"]) & ~_dilate(pair["input"] != bg)
    assert allowed.any()
    for p in np.argwhere(allowed):
        changed = pair["input"].copy()
        changed[tuple(p)] = solid
        try:
            answer = solve(changed)
        except ValueError:
            continue
        assert np.array_equal(answer, pair["output"])
        break
    else:
        raise AssertionError("no valid outside insertion")
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(2027, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                        (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    again = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in again for a, b in zip(again[s], puzzles[0][s])
               for k in ("input", "output"))
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in again[s]] for s in again}
    g = damaged["test"][0]["output"]
    g[0, 0] = (int(g[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed chamber self-test passed")
