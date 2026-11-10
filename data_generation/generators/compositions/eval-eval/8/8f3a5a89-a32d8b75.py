"""Exterior stamp loom: selected external edges drive oriented tile unions.

Only NumPy and the Python standard library are used. The input contains all
palette roles and all geometry needed by solve; no training or seed is read.
"""
from __future__ import annotations

import numpy as np


# Clockwise normals: north, east, south, west.
_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=False)


def _flood(allowed, start):
    seen = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return seen
    seen[start] = True
    todo = [start]
    h, w = allowed.shape
    while todo:
        r, c = todo.pop()
        for dr, dc in _DIRS:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and allowed[y, x] and not seen[y, x]:
                seen[y, x] = True
                todo.append((y, x))
    return seen


def _components(mask):
    left = mask.copy()
    result = []
    while left.any():
        start = tuple(int(v) for v in np.argwhere(left)[0])
        comp = _flood(left, start)
        result.append(comp)
        left[comp] = False
    return result


def _halo(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = mask.copy()
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)] if diagonal else _DIRS
    for dr, dc in offsets:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("the outer margin must show the background")
    candidates = []
    for color in np.unique(g):
        if color == bg:
            continue
        comps = _components(g == color)
        if len(comps) != 2:
            continue
        boxes = []
        for comp in comps:
            rr, cc = np.nonzero(comp)
            y, x, b, e = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
            if b - y < 4 or e - x < 4:
                break
            ring = np.zeros(g.shape, bool)
            ring[y, x:e] = ring[b - 1, x:e] = True
            ring[y:b, x] = ring[y:b, e - 1] = True
            if not np.array_equal(comp, ring):
                break
            boxes.append((y, x, b, e))
        if len(boxes) != 2:
            continue
        sizes = [(b - y - 2, e - x - 2) for y, x, b, e in boxes]
        tile_ids = [i for i, (h, w) in enumerate(sizes) if h == w and 2 <= h <= 4]
        map_ids = [i for i, (h, w) in enumerate(sizes) if min(h, w) >= 5]
        if len(tile_ids) == len(map_ids) == 1:
            candidates.append((int(color), boxes[tile_ids[0]], boxes[map_ids[0]]))
    if len(candidates) != 1:
        raise ValueError("expected two unambiguous same-color rectangular frames")
    frame, tb, mb = candidates[0]
    ty, tx, te, tf = tb
    my, mx, me, mf = mb
    tile = g[ty + 1:te - 1, tx + 1:tf - 1].copy()
    scene = g[my + 1:me - 1, mx + 1:mf - 1].copy()
    n = len(tile)
    tile_colors = set(int(v) for v in np.unique(tile))
    if len(tile_colors) != 2 or bg not in tile_colors or frame in tile_colors:
        raise ValueError("tile must explicitly contain background and one ink color")
    ink = next(iter(tile_colors - {bg}))
    values, counts = np.unique(scene, return_counts=True)
    if len(values) != 3 or bg not in values or frame in values or ink in values:
        raise ValueError("map must contain background, walls and a distinct marker")
    singles = [int(v) for v, c in zip(values, counts) if c == 1 and v != bg]
    if len(singles) != 1:
        raise ValueError("map marker must be its unique singleton color")
    marker = singles[0]
    wall = next(int(v) for v in values if v not in (bg, marker))
    if np.count_nonzero(scene == wall) < 2 or max(scene.shape) * n > 30:
        raise ValueError("map has invalid walls or expanded dimensions")
    # No unexplained pixels outside the two complete framed objects.
    allowed = np.zeros(g.shape, bool)
    for y, x, b, e in (tb, mb):
        allowed[y:b, x:e] = True
    if np.any(g[~allowed] != bg):
        raise ValueError("stray pixels outside frames")
    return dict(grid=g, bg=bg, frame=frame, ink=ink, marker=marker,
                wall=wall, tile=tile, scene=scene, tile_box=tb, map_box=mb, n=n)


def _topology(scene, bg, wall, marker):
    start = tuple(int(v) for v in np.argwhere(scene == marker)[0])
    target = _flood(scene != wall, start)
    exterior = _flood(np.pad(~target, 1, constant_values=True), (0, 0))
    h, w = scene.shape
    faces = np.zeros((h, w, 4), bool)
    for k, (dr, dc) in enumerate(_DIRS):
        faces[:, :, k] = target & exterior[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    kept = np.zeros(scene.shape, bool)
    for comp in _components(scene == wall):
        if np.any(_halo(target) & comp):
            kept |= comp
    return dict(target=target, exterior=exterior, faces=faces, kept=kept, start=start)


def _render(d, top, behavior="normal", orientation=(1, 0), exchange=True):
    """Forward block renderer; variants quantify evidence against shortcuts."""
    bg, ink, n, scene = d["bg"], d["ink"], d["n"], d["scene"]
    target, faces = top["target"], top["faces"].copy()
    if behavior == "all_boundaries":
        h, w = scene.shape
        other = np.pad(~target, 1, constant_values=True)
        for k, (dr, dc) in enumerate(_DIRS):
            faces[:, :, k] = target & other[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    if behavior == "all_rooms":
        # Apply the same external-edge rule independently to every free region.
        faces[:] = False
        for region in _components(scene != d["wall"]):
            ex = _flood(np.pad(~region, 1, constant_values=True), (0, 0))
            h, w = scene.shape
            for k, (dr, dc) in enumerate(_DIRS):
                faces[:, :, k] |= region & ex[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    coarse = np.full(scene.shape, bg, int)
    coarse[top["kept"]] = d["wall"]
    if behavior == "keep_all_walls":
        coarse[scene == d["wall"]] = d["wall"]
    coarse[scene == d["marker"]] = d["marker"]
    out = np.repeat(np.repeat(coarse, n, axis=0), n, axis=1)
    base = d["tile"] == (bg if exchange else ink)
    sign, offset = orientation
    stamps = [np.rot90(base, -(sign * k + offset) % 4) for k in range(4)]
    if behavior == "no_rotation":
        stamps = [base] * 4
    for r, c in np.argwhere(faces.any(axis=2) & (scene != d["marker"])):
        directions = np.flatnonzero(faces[r, c])
        bits = np.zeros((n, n), bool)
        for k in directions:
            if behavior == "xor":
                bits ^= stamps[k]
            else:
                bits |= stamps[k]
        if behavior == "one_face":
            bits = stamps[directions[0]]
        elif behavior == "solid_rim":
            bits[:] = True
        elif behavior == "eight_contact":
            pass  # Additional diagonal-only cells are handled below.
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.where(bits, ink, bg)
    if behavior == "eight_contact":
        diagonal_only = target & ~faces.any(axis=2) & (scene != d["marker"])
        p = top["exterior"]
        for r, c in np.argwhere(diagonal_only):
            if p[r:r + 3, c:c + 3].any():
                out[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.where(base, ink, bg)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a complete answer from input pixels alone; never mutates them."""
    d = _decode(input_grid)
    top = _topology(d["scene"], d["bg"], d["wall"], d["marker"])
    return _render(d, top)


def _independent_pixels(d):
    """Validation by synchronous reachability and inverse pixel coordinates.

    This is an authored cross-check, not an independent reference solver.
    It shares frame decoding, but not flood-fill or forward block painting.
    """
    scene, n = d["scene"], d["n"]
    h, w = scene.shape
    free = scene != d["wall"]

    def spread(allowed, reached):
        while True:
            q = np.pad(reached, 1)
            expanded = reached.copy()
            for dr, dc in _DIRS:
                expanded |= q[1 + dr:1 + dr + len(reached), 1 + dc:1 + dc + reached.shape[1]]
            expanded &= allowed
            if np.array_equal(expanded, reached):
                return reached
            reached = expanded

    selected = spread(free, scene == d["marker"])
    complement = np.pad(~selected, 1, constant_values=True)
    edge = np.zeros(complement.shape, bool)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    outside = spread(complement, edge)
    adjacent = np.zeros(scene.shape, bool)
    p = np.pad(selected, 1)
    for dr, dc in _DIRS:
        adjacent |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    walls = scene == d["wall"]
    keep = spread(walls, adjacent & walls)
    out = np.full((h * n, w * n), d["bg"], int)
    tile = d["tile"]
    for y in range(h * n):
        for x in range(w * n):
            r, c, u, v = y // n, x // n, y % n, x % n
            if scene[r, c] == d["marker"]:
                out[y, x] = d["marker"]
            elif keep[r, c]:
                out[y, x] = d["wall"]
            elif selected[r, c]:
                for k, (dr, dc) in enumerate(_DIRS):
                    if not outside[r + 1 + dr, c + 1 + dc]:
                        continue
                    # Inverse lookup into an unrotated north-facing exemplar.
                    sy, sx = ((u, v), (n - 1 - v, u),
                              (n - 1 - u, n - 1 - v), (v, n - 1 - u))[k]
                    if tile[sy, sx] == d["bg"]:
                        out[y, x] = d["ink"]
                        break
    return out


def _motif(rng, n):
    # A complete pattern; no missing row, completion or inferred support.
    for _ in range(100):
        bits = rng.random((n, n)) < rng.uniform(.35, .7)
        if bits.any() and not bits.all() and len({_key(np.rot90(bits, k)) for k in range(4)}) == 4:
            return bits
    bits = np.zeros((n, n), bool)
    bits[0, :2] = True
    return bits


def _place_object(rng, role, region, interior):
    h, w = role.shape
    for attempt in range(35):
        sh = int(rng.integers(1, min(3, h - 2) + 1)) if attempt < 20 else 1
        sw = int(rng.integers(1, min(3, w - 2) + 1)) if attempt < 20 else 1
        shape = np.ones((sh, sw), bool)
        if sh > 1 and sw > 1 and rng.random() < .5:
            shape[1:, 1:] = False
        r, c = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
        mask = np.zeros(role.shape, bool)
        mask[r:r + sh, c:c + sw] = shape
        halo = _halo(mask, True)
        if not np.all(region[halo]):
            continue
        if interior and (r == 0 or c == 0 or r + sh == h or c + sw == w):
            continue
        remains = region & ~mask
        if not remains.any() or len(_components(remains)) != 1:
            continue
        role[mask] = 1
        region[mask] = False
        return True
    return False


def _scene(rng, n, style, witness, rotation):
    limit = 30 // n
    if witness == 1:
        h, w = int(rng.integers(9, 14)), int(rng.integers(12, 16))
        cut = int(rng.integers(4, 6))
    elif witness == 2:
        h, w = int(rng.integers(8, 11)), int(rng.integers(9, 11))
        cut = int(rng.integers(2, 4))
    else:
        h, w = int(rng.integers(5, limit + 1)), int(rng.integers(5, limit + 1))
        cut = int(rng.integers(2, w - 2))
    role = np.zeros((h, w), int)
    if style in ("straight", "branch") or witness:
        role[:, cut] = 1
        if style == "branch" and not witness:
            row = int(rng.integers(2, h - 2))
            if rng.integers(2):
                role[row, :cut] = 1
            else:
                role[row, cut:] = 1
    elif style == "bent":
        position = cut
        for r in range(h):
            nxt = int(np.clip(position + rng.integers(-1, 2), 2, w - 3)) if r and rng.random() < .45 else position
            role[r, min(position, nxt):max(position, nxt) + 1] = 1
            position = nxt
    if witness == 1:
        # A thin mouth teaches three-way union, and a mid-wall lobe teaches
        # that a diagonal-only contact does not request a directional stamp.
        role[:2, 1:cut + 1] = 1
        row = int(rng.integers(3, h - 3))
        role[row:row + 2, cut - 1:cut + 1] = 1
    elif witness == 2:
        row = int(rng.integers(2, h - 3))
        role[row:row + 2, cut:cut + 2] = 1
    else:
        # Boundary-attached lobes coexist with any divider style.
        for _ in range(int(rng.integers(0 if style != "open" else 1, 5))):
            depth = int(rng.integers(1, max(2, min(h, w) // 3)))
            width = int(rng.integers(1, 4))
            side = int(rng.integers(4))
            if side < 2:
                c = int(rng.integers(w))
                if side == 0:
                    role[:depth, c:c + width] = 1
                else:
                    role[-depth:, c:c + width] = 1
            else:
                r = int(rng.integers(h))
                if side == 2:
                    role[r:r + width, :depth] = 1
                else:
                    role[r:r + width, -depth:] = 1
    regions = _components(role == 0)
    # Free rooms all reach the map edge; there are no nested free chambers.
    if not regions or any(m.sum() < 3 or not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()) for m in regions):
        return None
    if witness:
        probe = (h // 2, 0 if witness == 1 else w - 1)
        indices = [i for i, m in enumerate(regions) if m[probe]]
        if not indices:
            return None
        chosen = indices[0]
    else:
        chosen = int(rng.integers(len(regions)))
    target = regions[chosen]
    desired = int(rng.integers(1 if witness else 0, 4))
    placed = 0
    for _ in range(desired):
        placed += _place_object(rng, role, target, True)
    if witness and not placed:
        return None
    others = [m for i, m in enumerate(regions) if i != chosen]
    removed = 0
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        if others:
            removed += _place_object(rng, role, others[int(rng.integers(len(others)))], False)
    if witness and not removed:
        return None
    if np.count_nonzero(role == 1) < 2:
        return None
    # Marker locations do not encode a side or a corner.
    exterior = _flood(np.pad(~target, 1, constant_values=True), (0, 0))
    rim = np.zeros(role.shape, bool)
    for dr, dc in _DIRS:
        rim |= exterior[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    choices = np.argwhere(target & (~rim if witness == 1 else rim)) if witness else np.argwhere(target)
    if not len(choices):
        return None
    r, c = choices[int(rng.integers(len(choices)))]
    role[r, c] = 2
    return np.rot90(role, -rotation).copy()


def _frame(content, color):
    return np.pad(content, 1, constant_values=color)


def _pack(rng, tile, scene, bg, frame):
    a, b = _frame(tile, frame), _frame(scene, frame)
    horizontal = bool(rng.integers(2))
    gap = int(rng.integers(1, 4))
    if rng.integers(2):
        a, b = b, a
    if horizontal:
        h, w = max(len(a), len(b)) + 2, a.shape[1] + b.shape[1] + gap + 2
        g = np.full((h, w), bg, int)
        ya, yb = int(rng.integers(1, h - len(a))), int(rng.integers(1, h - len(b)))
        g[ya:ya + len(a), 1:1 + a.shape[1]] = a
        x = 1 + a.shape[1] + gap
        g[yb:yb + len(b), x:x + b.shape[1]] = b
    else:
        h, w = len(a) + len(b) + gap + 2, max(a.shape[1], b.shape[1]) + 2
        g = np.full((h, w), bg, int)
        xa, xb = int(rng.integers(1, w - a.shape[1])), int(rng.integers(1, w - b.shape[1]))
        g[1:1 + len(a), xa:xa + a.shape[1]] = a
        y = 1 + len(a) + gap
        g[y:y + len(b), xb:xb + b.shape[1]] = b
    return g


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _features(d, top):
    scene, target, faces = d["scene"], top["target"], top["faces"]
    regions = _components(scene != d["wall"])
    sizes = [int(r.sum()) for r in regions]
    sizesel = int(target.sum())
    inside = (scene == d["wall"]) & ~top["exterior"][1:-1, 1:-1]
    active = faces.any(axis=2) & (scene != d["marker"])
    corners = tuple(i for i, v in enumerate((target[0, 0], target[0, -1], target[-1, -1], target[-1, 0])) if v)
    return dict(n=d["n"], map_shape=scene.shape, compartments=len(regions),
                islands=len(_components(inside)), erased=len(_components((scene == d["wall"]) & ~top["kept"])),
                selected="only" if len(sizes) == 1 else "small" if sizesel < max(sizes) else "large",
                corners=corners, on_rim=bool(faces[top["start"]].any()),
                face_counts=tuple(sorted(set(int(v) for v in faces.sum(axis=2)[active]))),
                solo=tuple(k for k in range(4) if np.any(active & (faces.sum(axis=2) == 1) & faces[:, :, k])),
                layout="horizontal" if d["tile_box"][3] <= d["map_box"][1] or d["map_box"][3] <= d["tile_box"][1] else "vertical")


def _shift_geometry(scene, wall):
    """Visible split preference: three rooms or a divider without a straight spine."""
    if len(_components(scene != wall)) >= 3:
        return True
    for comp in _components(scene == wall):
        if comp[0].any() and comp[-1].any() and not np.any(np.all(comp, axis=0)):
            return True
        if comp[:, 0].any() and comp[:, -1].any() and not np.any(np.all(comp, axis=1)):
            return True
    return False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    stable_palette = bool(rng.integers(2))
    palette = rng.choice(10, size=5, replace=False)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_corners = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            for _ in range(500):
                n = witness + 1 if witness else int(rng.choice([2, 3, 4]))
                style = "straight" if witness else str(rng.choice(["straight", "open"])) if split == "train" else str(rng.choice(["bent", "branch"]))
                role = _scene(rng, n, style, witness, rotation if witness else int(rng.integers(4)))
                if role is None:
                    continue
                bg, frame, wall, marker, ink = (int(v) for v in (palette if stable_palette else rng.choice(10, 5, replace=False)))
                scene = np.choose(role, [bg, wall, marker])
                tile = np.where(_motif(rng, n), ink, bg)
                inp = _pack(rng, tile, scene, bg, frame)
                if _key(inp) in seen:
                    continue
                d = _decode(inp)
                top = _topology(scene, bg, wall, marker)
                out = _render(d, top)
                f = _features(d, top)
                if split == "test" and not _shift_geometry(scene, wall):
                    continue
                if witness:
                    if f["selected"] != ("small" if witness == 1 else "large") or len(f["solo"]) < 4:
                        continue
                    if witness == 2 and set(first_corners) & set(f["corners"]):
                        continue
                    if any(np.array_equal(out, _render(d, top, behavior=b)) for b in
                           ("all_boundaries", "all_rooms", "keep_all_walls", "solid_rim", "no_rotation", "one_face", "xor")):
                        continue
                    if witness == 1 and np.array_equal(out, _render(d, top, behavior="eight_contact")):
                        continue
                break
            else:
                raise RuntimeError("bounded scene sampling failed")
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": out})
            if witness == 1:
                first_corners = f["corners"]
        rng.shuffle(puzzle[split])
    return puzzle


_SHORTCUTS = ("all_boundaries", "all_rooms", "keep_all_walls", "solid_rim",
              "no_rotation", "one_face", "xor", "eight_contact")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            top = _topology(d["scene"], d["bg"], d["wall"], d["marker"])
            assert out.shape == tuple(v * d["n"] for v in d["scene"].shape)
            assert np.array_equal(out, _independent_pixels(d))
            assert np.array_equal(out, solve(inp))
            assert set(np.unique(out)) <= {d[k] for k in ("bg", "wall", "marker", "ink")}
            assert np.count_nonzero(out == d["marker"]) == d["n"] ** 2
            assert np.any(out == d["ink"])
            # Every free compartment touches the map edge; nested rooms are outside the generated family.
            for region in _components(d["scene"] != d["wall"]):
                assert region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
            if split == "train":
                train.append((d, top, out, _features(d, top)))
    fs = [f for _, _, _, f in train]
    assert any(f["selected"] == "small" for f in fs) and any(f["selected"] == "large" for f in fs)
    assert not set.intersection(*(set(f["corners"]) for f in fs))
    assert any(f["islands"] for f in fs) and any(f["erased"] for f in fs)
    assert any(f["on_rim"] for f in fs) and any(not f["on_rim"] for f in fs)
    assert set(k for f in fs for k in f["solo"]) == set(range(4))
    assert {1, 2, 3} <= set(k for f in fs for k in f["face_counts"])
    assert {2, 3} <= {f["n"] for f in fs}
    for behavior in _SHORTCUTS:
        assert any(not np.array_equal(out, _render(d, top, behavior=behavior)) for d, top, out, _ in train), behavior
    # Check exchange jointly with all eight direction conventions, rather than
    # assuming a tile cannot imitate a swapped version by a rotation.
    survivors = []
    for exchange in (True, False):
        for sign in (1, -1):
            for offset in range(4):
                if all(np.array_equal(out, _render(d, top, orientation=(sign, offset), exchange=exchange)) for d, top, out, _ in train):
                    survivors.append((exchange, sign, offset))
    assert survivors == [(True, 1, 0)], f"ambiguous training convention: {survivors}"


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                top = _topology(d["scene"], d["bg"], d["wall"], d["marker"])
                f = _features(d, top)
                records.append(f)
                local.append(f)
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
        assert len({f["map_shape"] for f in local}) >= 2
        assert len({f["selected"] for f in local}) >= 2
    assert len(signatures) == 42
    assert {f["n"] for f in records} == {2, 3, 4}
    assert len({f["map_shape"] for f in records}) >= 40
    assert {f["layout"] for f in records} == {"horizontal", "vertical"}
    assert {1, 2, 3} <= {f["compartments"] for f in records}
    assert {0, 1, 2} <= {f["islands"] for f in records}
    assert {0, 1, 2} <= {f["erased"] for f in records}
    assert _puzzle_key(generate_puzzle(923, 2, 1)) == _puzzle_key(generate_puzzle(923, 2, 1))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(401 + 10 * a + b, a, b)
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (6, 5), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _topology_regressions()
    corrupted = generate_puzzle(8, 2, 1)
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


def _topology_regressions():
    """Small constructive facts with known geometry, beyond generated replay."""
    # Border frame is irrelevant here: exercise a 9x12 map directly.
    scene = np.zeros((9, 12), int)
    scene[:, 6] = 1
    scene[4, 2] = 2
    d = dict(scene=scene, n=3, bg=0, ink=4, wall=1, marker=2,
             tile=np.array([[4, 4, 0], [4, 0, 4], [0, 4, 4]]))
    top = _topology(scene, 0, 1, 2)
    assert top["target"].sum() == 54
    assert top["faces"][0, 0].tolist() == [True, False, False, True]
    assert not top["faces"][4, 2].any()
    out = _render(d, top)
    north = np.array([[0, 0, 4], [0, 4, 0], [4, 0, 0]])
    assert np.array_equal(out[0:3, 6:9], north)
    assert np.all(out[12:15, 6:9] == 2)
    # Island insertion does not create any exterior-facing edge nearby.
    island = scene.copy()
    island[4, 4] = 1
    ti = _topology(island, 0, 1, 2)
    assert not ti["exterior"][5, 5]
    assert np.array_equal(ti["faces"], top["faces"])
    # Connecting that island to the divider makes it an exterior intrusion.
    joined = island.copy()
    joined[4, 4:7] = 1
    tj = _topology(joined, 0, 1, 2)
    assert tj["faces"][3, 4, 2] and tj["faces"][4, 3, 1]
    assert not tj["faces"][3, 3].any()  # Diagonal-only contact has no edge normal.
    # A far-side object disappears; moving the marker across swaps retention.
    far = island.copy()
    far[3:5, 9] = 1
    tf = _topology(far, 0, 1, 2)
    assert not tf["kept"][3, 9] and tf["kept"][4, 4]
    far[4, 2], far[6, 9] = 0, 2
    ta = _topology(far, 0, 1, 2)
    assert ta["kept"][3, 9] and not ta["kept"][4, 4]
    # Moving the marker within a room changes only its enlarged position.
    moved = scene.copy()
    moved[4, 2], moved[5, 3] = 0, 2
    tm = _topology(moved, 0, 1, 2)
    assert np.array_equal(tm["faces"], top["faces"])
    # Rotate the MAP alone; the north-facing exemplar stays in its reference
    # orientation. The result must rotate with the map.
    for k in range(4):
        rotated = dict(d, scene=np.rot90(scene, -k).copy())
        tr = _topology(rotated["scene"], 0, 1, 2)
        assert np.array_equal(_render(rotated, tr), np.rot90(out, -k))


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, legal counts, shortcut checks and topology regressions")
