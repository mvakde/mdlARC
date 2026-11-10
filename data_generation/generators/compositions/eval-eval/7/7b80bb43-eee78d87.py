"""Repair fences, then let enclosure select the ink of a periodic wallpaper.

Only three input colors are needed: majority carrier, extensive fence, and a
3x3-bounded swatch. The output has the input dimensions and the same palette.
No files, generation metadata, or demonstrations are consulted by solve().
"""
from collections import deque
from itertools import permutations
import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _at(m, r, c):
    return 0 <= r < m.shape[0] and 0 <= c < m.shape[1] and bool(m[r, c])


def _straight(m):
    """Protect every cell of an uninterrupted orthogonal run of length >=3."""
    result = np.zeros_like(m)
    for a, b in ((m, result), (m.T, result.T)):
        triples = a[:, :-2] & a[:, 1:-1] & a[:, 2:]
        for k in range(3):
            b[:, k:k + triples.shape[1]] |= triples
    return result


def _gaps(m):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        a = m if dc else m.T
        matches = a[:, :-4] & a[:, 4:]
        matches &= ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(matches):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _repairs(m):
    strong = _straight(m)
    result = []
    for r, c, dr, dc in _gaps(m):
        if not (_at(m, r - dr, c - dc) or _at(m, r + 5*dr, c + 5*dc)):
            continue
        matches = []
        for end in (0, 4):
            ar, ac = r + end*dr, c + end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                dy, dx = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar + k*dy, ac + k*dx) for k in (1, 2))
                if (all(_at(m, y, x) and not strong[y, x] for y, x in spur)
                        and not _at(m, ar + 3*dy, ac + 3*dx)):
                    matches.append(spur)
        if matches:
            result.append(((r, c, dr, dc), matches))
    return result


def _clean(m):
    evidence = _repairs(m)
    added, anchors = np.zeros_like(m), np.zeros_like(m)
    for (r, c, dr, dc), _ in evidence:
        for k in (1, 2, 3):
            added[r + k*dr, c + k*dc] = True
        anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
    weak = m & ~_straight(m | added) & ~anchors
    diagonal = np.zeros_like(m)
    for flip in (False, True):
        a = weak[:, ::-1] if flip else weak
        b = diagonal[:, ::-1] if flip else diagonal
        links = a[:-1, :-1] & a[1:, 1:]
        b[:-1, :-1] |= links
        b[1:, 1:] |= links
    return (m & ~diagonal) | added


def _sealed(wall):
    """Four-neighbor carrier reachability from the actual canvas boundary."""
    h, w = wall.shape
    exterior = np.zeros_like(wall)
    q = deque()
    for r, c in ([(r, c) for r in range(h) for c in (0, w-1)]
                 + [(r, c) for c in range(w) for r in (0, h-1)]):
        if not wall[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not wall[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                q.append((y, x))
    return ~wall & ~exterior


def _components(m):
    pending = set(map(tuple, np.argwhere(m).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _decode(grid):
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or int(counts.max()) * 2 <= a.size:
        raise ValueError("input needs three roles and a strict-majority carrier")
    bg = int(colors[np.argmax(counts)])
    swatches = []
    for color in colors:
        if color == bg:
            continue
        pts = np.argwhere(a == color)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if tuple(hi - lo) == (2, 2) and 2 <= len(pts) <= 8:
            r, c = map(int, lo)
            swatches.append((int(color), r, c, a[r:r+3, c:c+3] == color))
    if len(swatches) != 1:
        raise ValueError("input must contain one unambiguous 3x3-bounded swatch")
    accent, r, c, mask = swatches[0]
    fence = next(int(v) for v in colors if v not in (bg, accent))
    return a, bg, fence, accent, mask, (r, c), a == fence


def _paint(wall, mask, bg, fence, accent, order=None):
    if order is None:
        tile = mask[np.ix_((1, 2, 0), (1, 2, 0))]
    else:
        transpose, rows, cols = order
        tile = (mask.T if transpose else mask)[np.ix_(rows, cols)]
    h, w = wall.shape
    occupied = tile[np.arange(h)[:, None] % 3, np.arange(w)[None, :] % 3]
    inside = _sealed(wall)
    out = np.full(wall.shape, bg, dtype=np.int8)
    out[occupied & ~wall] = fence
    out[occupied & inside] = accent
    out[wall] = fence
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, repair the drawing, find sealed faces, and tile from input."""
    _, bg, fence, accent, mask, _, raw = _decode(input_grid)
    return _paint(_clean(raw), mask, bg, fence, accent)


_PERMS = tuple(permutations(range(3)))
_ORDERS = tuple((t, r, c) for t in (False, True) for r in _PERMS for c in _PERMS)
_DIRECT = _ORDERS.index((False, (1, 2, 0), (1, 2, 0)))
_MASKS = []
for _code in range(1, 511):
    _m = np.array([(_code >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3)
    if _m[0].any() and _m[-1].any() and _m[:, 0].any() and _m[:, -1].any():
        _MASKS.append(_m)
_MASKS = np.array(_MASKS)
_DENSITIES = _MASKS.sum(axis=(1, 2))
_AGREE = np.array([[np.array_equal((m.T if t else m)[np.ix_(r, c)],
                                  m[np.ix_((1, 2, 0), (1, 2, 0))])
                    for t, r, c in _ORDERS] for m in _MASKS])


def _choose_mask(rng, eligible):
    ids = np.flatnonzero(eligible)
    density = int(rng.choice(np.unique(_DENSITIES[ids])))
    return int(rng.choice(ids[_DENSITIES[ids] == density]))


def _layout(rng, split, regime):
    if regime == "extent":
        nr, nc = (2, 2) if split == "train" else (2, 3)
    elif regime == "concavity":
        nr, nc = (2, int(rng.choice((2, 3)))) if split == "train" else (3, 3)
    else:  # A shift to longer merged chambers, without a new operation.
        nr, nc = (2, 2) if split == "train" else (2, 3)
    if rng.integers(2):
        nr, nc = nc, nr
    cells = {(r, c) for r in range(nr) for c in range(nc)}
    if regime == "concavity" and split == "test":
        corners = [(0, 0), (0, nc-1), (nr-1, 0), (nr-1, nc-1)]
        for k in rng.choice(4, size=int(rng.integers(1, 3)), replace=False):
            cells.remove(corners[int(k)])
    ys = [int(rng.integers(0, 3))]
    xs = [int(rng.integers(0, 3))]
    for _ in range(nr):
        ys.append(ys[-1] + int(rng.integers(6, 9)))
    for _ in range(nc):
        xs.append(xs[-1] + int(rng.integers(6, 9)))
    shape = ys[-1] + 1 + int(rng.integers(0, 3)), xs[-1] + 1 + int(rng.integers(0, 3))
    # Canonical wall edge -> incident coarse cells. Missing cells produce steps.
    edges = {}
    for r, c in sorted(cells):
        for edge in ((ys[r], xs[c], 0, 1, xs[c+1]-xs[c]),
                     (ys[r+1], xs[c], 0, 1, xs[c+1]-xs[c]),
                     (ys[r], xs[c], 1, 0, ys[r+1]-ys[r]),
                     (ys[r], xs[c+1], 1, 0, ys[r+1]-ys[r])):
            edges.setdefault(edge, set()).add((r, c))
    return shape, edges, cells


def _edge_cells(edge):
    r, c, dr, dc, length = edge
    return [(r + k*dr, c + k*dc) for k in range(length + 1)]


def _draw(shape, edges):
    m = np.zeros(shape, dtype=bool)
    for e in edges:
        for r, c in _edge_cells(e):
            m[r, c] = True
    return m


def _damage_options(edge):
    r, c, dr, dc, length = edge
    options = []
    for offset in range(length - 3):
        gap = tuple((r + (offset+k)*dr, c + (offset+k)*dc) for k in (1, 2, 3))
        for end in (0, 4):
            ar, ac = r + (offset+end)*dr, c + (offset+end)*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                dy, dx = toward*dr + side*dc, toward*dc + side*dr
                options.append((gap, tuple((ar+k*dy, ac+k*dx) for k in (1, 2))))
    return options


def _damage_edge(rng, raw, clean, edge, expected_count):
    options = _damage_options(edge)
    for index in rng.permutation(len(options)):
        gap, spur = options[int(index)]
        if not all(raw[r, c] for r, c in gap):
            continue
        if not all(0 <= r < raw.shape[0] and 0 <= c < raw.shape[1]
                   and not raw[r, c] and not clean[r, c] for r, c in spur):
            continue
        trial = raw.copy()
        for r, c in gap:
            trial[r, c] = False
        for r, c in spur:
            trial[r, c] = True
        evidence = _repairs(trial)
        if (len(evidence) == expected_count and all(len(x[1]) == 1 for x in evidence)
                and np.array_equal(_clean(trial), clean)):
            return trial
    return None


def _add_clutter(rng, raw, clean, wanted):
    for _ in range(wanted):
        length = int(rng.choice((2, 3)))
        for _ in range(45):
            r, c = int(rng.integers(raw.shape[0])), int(rng.integers(raw.shape[1]))
            dc = int(rng.choice((-1, 1)))
            pts = [(r+k, c+k*dc) for k in range(length)]
            if not all(0 <= y < raw.shape[0] and 0 <= x < raw.shape[1]
                       and not raw[y, x] and not clean[y, x] for y, x in pts):
                continue
            trial = raw.copy()
            for y, x in pts:
                trial[y, x] = True
            if np.array_equal(_clean(trial), clean):
                raw = trial
                break
    return raw


def _construct(rng, mask, palette, split, regime, index):
    shape, all_edges, cells = _layout(rng, split, regime)
    axis = index % 2 if split == "train" and index < 2 else int(rng.integers(2))
    boundary = [e for e, owners in all_edges.items() if len(owners) == 1 and e[2] == axis]
    repair_edge = boundary[int(rng.integers(len(boundary)))]
    target = next(iter(all_edges[repair_edge]))
    leak_options = [e for e, owners in all_edges.items() if len(owners) == 1 and target not in owners]
    leak_edge = leak_options[int(rng.integers(len(leak_options)))]
    retained = []
    for edge, owners in all_edges.items():
        remove_probability = .65 if regime == "merging" and split == "test" else .22
        if len(owners) == 2 and target not in owners and rng.random() < remove_probability:
            continue
        retained.append(edge)
    clean = _draw(shape, retained)
    r, c, dr, dc, length = leak_edge
    offset = int(rng.integers(length - 3))
    for k in (1, 2, 3):
        clean[r + (offset+k)*dr, c + (offset+k)*dc] = False
    raw = _damage_edge(rng, clean, clean, repair_edge, 1)
    if raw is None:
        raise AssertionError("primary repair could not be placed")
    count = 1
    extras = int(rng.integers(0, 3))
    for edge_index in rng.permutation(len(retained)):
        edge = retained[int(edge_index)]
        if edge in (leak_edge, repair_edge) or count > extras:
            continue
        trial = _damage_edge(rng, raw, clean, edge, count + 1)
        if trial is not None:
            raw, count = trial, count + 1
    raw = _add_clutter(rng, raw, clean, int(rng.integers(1, 4)))
    bg, fence, accent = palette
    inp = np.where(raw, fence, bg).astype(np.int8)
    sealed = _sealed(clean)
    room_count = len(_components(sealed))
    positions = []
    for y in range(shape[0]-2):
        for x in range(shape[1]-2):
            if (raw[y:y+3, x:x+3] | clean[y:y+3, x:x+3]).any():
                continue
            # A sample outside the only room, or several sealed rooms, exposes
            # that enclosure rather than sample membership selects highlighting.
            if room_count == 1 and sealed[y:y+3, x:x+3].any():
                continue
            if split == "train" and index == 0:
                local = mask[(np.arange(3)[:, None]-y) % 3,
                             (np.arange(3)[None, :]-x) % 3]
                direct = mask[np.ix_((1, 2, 0), (1, 2, 0))]
                if np.array_equal(local, direct):
                    continue
            positions.append((y, x))
    assert positions, "no clear swatch window"
    y, x = positions[int(rng.integers(len(positions)))]
    inp[y:y+3, x:x+3][mask] = accent
    out = _paint(clean, mask, bg, fence, accent)
    pair = {"input": inp, "output": out}
    _check_pair(pair)
    return pair


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    t = _count(num_test, "num_test", 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("extent", "concavity", "merging")))
    palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
    varied_palette = bool(rng.integers(2))
    first = _choose_mask(rng, np.ones(len(_MASKS), dtype=bool))
    candidates = (_DENSITIES != _DENSITIES[first]) & ((_AGREE & _AGREE[first]).sum(axis=1) == 1)
    second = _choose_mask(rng, candidates)
    puzzle, seen, mask_ids = {"train": [], "test": []}, set(), [first, second]
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            if split == "train" and index < 2:
                mask_id = mask_ids[index]
            else:
                allowed = np.ones(len(_MASKS), dtype=bool)
                allowed[mask_ids] = False
                mask_id = _choose_mask(rng, allowed)
                mask_ids.append(mask_id)
            colors = palette
            if varied_palette and (index >= 2 or split == "test") and rng.random() < .7:
                colors = tuple(map(int, rng.choice(10, size=3, replace=False)))
            for _ in range(40):
                try:
                    pair = _construct(rng, _MASKS[mask_id], colors, split, regime, index)
                except AssertionError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, bg, fence, accent, mask, pos, raw = _decode(pair["input"])
    out = _grid(pair["output"])
    assert inp.shape == out.shape
    evidence = _repairs(raw)
    assert evidence and all(len(matches) == 1 for _, matches in evidence)
    wall = _clean(raw)
    assert not wall[pos[0]:pos[0]+3, pos[1]:pos[1]+3].any()
    inside = _sealed(wall)
    rows, cols = np.indices(inp.shape)
    occupied = mask[(rows+1) % 3, (cols+1) % 3]
    # These are full array invariants, including carrier holes and fence priority.
    assert np.array_equal(out[wall], np.full(int(wall.sum()), fence))
    assert np.array_equal(out == accent, occupied & inside)
    assert np.array_equal(out == bg, ~wall & ~occupied)
    assert np.array_equal(out == fence, wall | (occupied & ~inside))
    assert (occupied & inside).any() and (~occupied & inside).any()
    assert (occupied & ~inside & ~wall).any()
    assert np.array_equal(out, solve(inp))
    assert not np.array_equal(inp, out)
    # A must change the wallpaper of distant non-wall cells, not just the fence.
    before_inside = _sealed(wall & raw)
    gained = inside & ~before_inside & ~wall & occupied
    assert int(gained.sum()) >= 2
    # At least one unsupported aperture must remain, with a topological effect.
    indiscriminate = wall.copy()
    for r, c, dr, dc in _gaps(wall):
        for k in (1, 2, 3):
            indiscriminate[r+k*dr, c+k*dc] = True
    overfilled = _sealed(indiscriminate)
    assert (overfilled & ~inside & ~indiscriminate & occupied).any()
    # No solid stencil is allowed: every sealed face exposes motif holes.
    rooms = _components(inside)
    for room in rooms:
        values = [bool(occupied[r, c]) for r, c in room]
        assert any(values) and not all(values)
    assert len(rooms) > 1 or not inside[pos[0]:pos[0]+3, pos[1]:pos[1]+3].any()
    assert not (wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:]).any()
    return {"shape": inp.shape, "palette": (bg, fence, accent), "mask": mask,
            "swatch_position": pos, "repairs": len(evidence),
            "axes": tuple(sorted({e[0][2] for e in evidence})),
            "rooms": len(rooms), "sealed_area": int(inside.sum()),
            "repair_influence": int(gained.sum()), "wall": wall}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train_infos, masks = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(np.asarray(pair["input"]))
            assert key not in seen
            seen.add(key)
            masks.append(info["mask"].tobytes())
            if split == "train":
                train_infos.append(info)
    assert len(set(masks)) == n + t
    assert {axis for info in train_infos for axis in info["axes"]} == {0, 1}
    assert len({int(info["mask"].sum()) for info in train_infos}) >= 2
    assert any(not np.array_equal(
        info["mask"][(np.arange(3)[:, None]-info["swatch_position"][0]) % 3,
                     (np.arange(3)[None, :]-info["swatch_position"][1]) % 3],
        info["mask"][np.ix_((1, 2, 0), (1, 2, 0))]) for info in train_infos)
    orders = set(range(len(_ORDERS)))
    for pair, info in zip(puzzle["train"], train_infos):
        bg, fence, accent = info["palette"]
        orders &= {k for k, order in enumerate(_ORDERS)
                   if np.array_equal(_paint(info["wall"], info["mask"], bg, fence, accent, order),
                                     pair["output"])}
    assert orders == {_DIRECT}, "demonstrations must establish direct cyclic phase"


def self_test():
    """42 streaming seeds; no saved puzzles, logs, or execution traces."""
    signatures, shapes, masks, walls, palettes = set(), set(), set(), set(), set()
    densities, rooms, repair_counts, swatches = set(), set(), set(), set()
    shifts = {"extent": 0, "concavity": 0, "merging": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        regime = str(np.random.default_rng(seed).choice(tuple(shifts)))
        shifts[regime] += 1
        for split in ("train", "test"):
            for pair in p[split]:
                info = _check_pair(pair)
                for role in ("input", "output"):
                    a = pair[role]
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
                shapes.add(info["shape"])
                masks.add(info["mask"].tobytes())
                walls.add(hashlib.sha256(info["wall"].tobytes()).digest())
                palettes.add(info["palette"])
                densities.add(int(info["mask"].sum()))
                rooms.add(info["rooms"])
                repair_counts.add(info["repairs"])
                swatches.add(info["swatch_position"])
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(masks) >= 85 and len(walls) >= 180 and len(palettes) >= 25
    assert densities == set(range(2, 9)) and len(rooms) >= 3
    assert {1, 2, 3} <= repair_counts and len(swatches) >= 40
    assert min(shifts.values()) >= 3
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(15000 + 10*n + t, n, t)
            assert len(p["train"]) == n and len(p["test"]) == t
    p, q = generate_puzzle(940, 3, 1), generate_puzzle(940, 3, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
               for k in ("input", "output"))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Negative regressions: an omitted repair, a destroyed carrier hole, and
    # replacing topology by one fixed central rectangle must all be rejected.
    for mutation in ("repair", "hole", "center"):
        p = generate_puzzle(23, 2, 1)
        pair = p["test"][0]
        inp, bg, fence, accent, mask, _, raw = _decode(pair["input"])
        wall, out = _clean(raw), pair["output"]
        if mutation == "repair":
            (r, c, dr, dc), matches = _repairs(raw)[0]
            for k in (1, 2, 3):
                out[r+k*dr, c+k*dc] = bg
            for y, x in matches[0]:
                out[y, x] = fence
        elif mutation == "hole":
            y, x = np.argwhere(_sealed(wall) & (out == bg))[0]
            out[y, x] = accent
        else:
            h, w = out.shape
            out[out == accent] = fence
            for r in range((h-6)//2, (h-6)//2+6):
                for c in range((w-6)//2, (w-6)//2+6):
                    if not wall[r, c] and mask[(r+1) % 3, (c+1) % 3]:
                        out[r, c] = accent
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            continue
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "motifs": len(masks), "wall_structures": len(walls),
            "palettes": len(palettes), "densities": sorted(densities),
            "room_counts": sorted(rooms), "repair_counts": sorted(repair_counts),
            "regimes": shifts, "independent_semantic_verification": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
