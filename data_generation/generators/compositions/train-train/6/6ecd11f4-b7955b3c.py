"""Magnified damage map + camouflaged plus stamps on a layered-rectangle payload.

Each input holds two separate solid objects on a uniform canvas:

* a large two-colour "damage map": an n-by-m binary grid enlarged by an integer
  factor k (parent A's magnified stencil).  The colour at the map's corners is
  the intact colour; the other colour marks damaged macrocells, which always
  form five-cell plus shapes (parent B's stamps) in compact coordinates;
* a compact n-by-m payload: a scene of solid, overlapping, axis-aligned
  rectangles on a local background (parent B's layered composition) whose
  plus-shaped damaged cells were overpainted with ordinary scene colours, so
  the damage is camouflaged and can only be located through the map.

The output is the repaired payload: every damaged cell takes the value of the
layered rectangle scene reconstructed from the undamaged cells; all other
payload cells are copied unchanged.
"""

from __future__ import annotations

import numpy as np

_MAX = 30


class _Reject(Exception):
    """Internal: candidate construction rejected, retry."""


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _need(ok):
    if not ok:
        raise _Reject()


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _plus(r, c):
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask):
    """4-connected components of a boolean mask as lists of (r, c)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        stack, comp = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)
    return comps


def _bbox(cells):
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1]) and max(a[2], b[2]) < min(a[3], b[3])


# ---------------------------------------------------------------------------
# Input parsing and layered reconstruction (used by solve and the generator)
# ---------------------------------------------------------------------------

def _parse(grid):
    """Split an input into (canvas, map crop, payload crop, scale, damage mask)."""
    grid = np.asarray(grid, dtype=int)
    _check(grid.ndim == 2, "grid must be 2D")
    canvas = int(grid[0, 0])
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _check(np.all(border == canvas), "canvas border must be uniform")
    comps = _components(grid != canvas)
    _check(len(comps) == 2, "expected exactly two objects")
    boxes = []
    for comp in comps:
        r0, r1, c0, c1 = _bbox(comp)
        _check(len(comp) == (r1 - r0) * (c1 - c0), "objects must be solid rectangles")
        boxes.append((r0, r1, c0, c1))
    areas = [(b[1] - b[0]) * (b[3] - b[2]) for b in boxes]
    _check(areas[0] != areas[1], "map and payload must differ in size")
    big, small = (boxes[0], boxes[1]) if areas[0] > areas[1] else (boxes[1], boxes[0])
    stencil = grid[big[0]:big[1], big[2]:big[3]]
    payload = grid[small[0]:small[1], small[2]:small[3]]
    n, m = payload.shape
    H, W = stencil.shape
    _check(H % n == 0 and W % m == 0 and H // n == W // m and H // n >= 2,
           "map must be an integer enlargement of the payload lattice")
    k = H // n
    blocks = stencil.reshape(n, k, m, k).transpose(0, 2, 1, 3).reshape(n, m, k * k)
    _check(np.all(blocks == blocks[:, :, :1]), "map macrocells must be uniform")
    compact = blocks[:, :, 0]
    intact = int(stencil[0, 0])
    _check(np.all(compact[[0, 0, -1, -1], [0, -1, 0, -1]] == intact), "map corners must agree")
    colours = set(int(v) for v in np.unique(compact))
    _check(len(colours) == 2, "map must use exactly two colours")
    damage = compact != intact
    return {"canvas": canvas, "map_box": big, "payload_box": small, "scale": k,
            "payload": payload.copy(), "damage": damage, "intact": intact,
            "mark": next(iter(colours - {intact}))}


def _reconstruct(payload, unknown):
    """Rebuild the layered rectangle scene from cells not flagged unknown.

    Returns (scene, background, colours, bounds, reach) or raises AssertionError.
    """
    h, w = payload.shape
    known = ~unknown
    edge = np.zeros((h, w), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    vals = payload[edge & known]
    _check(len(vals) > 0, "no background evidence")
    counts = np.bincount(vals, minlength=10)
    background = int(np.argmax(counts))
    _check(np.sum(counts == counts[background]) == 1, "background tie on payload border")
    colours = [int(v) for v in np.unique(payload[known]) if int(v) != background]
    _check(len(colours) >= 1, "no visible rectangles")
    bounds, rects = [], []
    for colour in colours:
        pts = np.argwhere(known & (payload == colour))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0) + 1
        bounds.append((int(r0), int(r1), int(c0), int(c1)))
        rect = np.zeros((h, w), bool)
        rect[r0:r1, c0:c1] = True
        rects.append(rect)
    q = len(colours)
    above = np.zeros((q, q), bool)
    for i, colour in enumerate(colours):
        visible = known & (payload == colour)
        for j in range(q):
            if i != j and np.any(visible & rects[j]):
                above[i, j] = True
    reach = above.copy()
    for t in range(q):
        reach |= reach[:, t, None] & reach[t, None, :]
    _check(not np.any(np.diag(reach)), "cyclic layer evidence")
    scene = np.full((h, w), background, int)
    covered = np.zeros((h, w), bool)
    winners = np.zeros((h, w), int)
    for i, rect in enumerate(rects):
        covered |= rect
        top = rect.copy()
        for j, other in enumerate(rects):
            if i != j and not reach[i, j]:
                top &= ~other
        scene[top] = colours[i]
        winners += top
    _check(np.all(winners[covered] == 1), "ambiguous layer order")
    _check(np.array_equal(scene[known], payload[known]), "visible cells contradict rectangles")
    return scene, background, colours, bounds, reach


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    scene, *_ = _reconstruct(info["payload"], info["damage"])
    return scene.astype(int)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _rectangle(rng, n, m, previous):
    kind = int(rng.integers(3))
    if kind == 0:      # wide strip
        rh = int(rng.integers(1, 3))
        rw = int(rng.integers(3, m + 1))
    elif kind == 1:    # tall strip
        rh = int(rng.integers(3, n + 1))
        rw = int(rng.integers(1, 3))
    else:              # block
        rh = int(rng.integers(2, max(3, int(0.75 * n)) + 1))
        rw = int(rng.integers(2, max(3, int(0.75 * m)) + 1))
    rh, rw = min(rh, n), min(rw, m)
    relation = int(rng.choice(3, p=[0.55, 0.2, 0.25]))
    chosen = None
    for _ in range(20):
        if relation == 0 and previous:
            t = previous[int(rng.integers(len(previous)))]
            mr, mc = (t[0] + t[1]) // 2, (t[2] + t[3]) // 2
            r0 = int(np.clip(mr - rh // 2 + rng.integers(-rh // 2 - 1, rh // 2 + 2), 0, n - rh))
            c0 = int(np.clip(mc - rw // 2 + rng.integers(-rw // 2 - 1, rw // 2 + 2), 0, m - rw))
        else:
            r0 = int(rng.integers(n - rh + 1))
            c0 = int(rng.integers(m - rw + 1))
        chosen = (r0, r0 + rh, c0, c0 + rw)
        if chosen in previous:
            continue
        if relation != 1 or not previous or all(not _overlap(chosen, p) for p in previous):
            return chosen
    return chosen


def _bounds_tight(payload, unknown, background, colours, bounds, reach):
    """Every rectangle side bordering hidden cells must be pinned by visible evidence.

    Extending a rectangle by one row/column into a strip that contains hidden
    cells must contradict a visible cell (background, or a layer behind it).
    """
    h, w = payload.shape
    for i, (r0, r1, c0, c1) in enumerate(bounds):
        strips = []
        if r0 > 0:
            strips.append((slice(r0 - 1, r0), slice(c0, c1)))
        if r1 < h:
            strips.append((slice(r1, r1 + 1), slice(c0, c1)))
        if c0 > 0:
            strips.append((slice(r0, r1), slice(c0 - 1, c0)))
        if c1 < w:
            strips.append((slice(r0, r1), slice(c1, c1 + 1)))
        allowed = {background} | {colours[j] for j in range(len(colours)) if reach[i, j]}
        for rs, cs in strips:
            hid = unknown[rs, cs]
            if not np.any(hid):
                continue
            vis = payload[rs, cs][~hid]
            if not any(int(v) in allowed for v in vis):
                return False
    return True


def _make_payload(rng, n, m, rect_range, stamp_range, palette, need):
    """Return (damaged payload, clean scene, damage mask, stamp centres)."""
    n_rect = int(rng.integers(rect_range[0], rect_range[1] + 1))
    chosen = rng.choice(palette, n_rect + 1, replace=False)
    background, colours = int(chosen[0]), [int(v) for v in chosen[1:]]
    rects = [_rectangle(rng, n, m, [])]
    for _ in range(n_rect - 1):
        rects.append(_rectangle(rng, n, m, rects))
    _need(len(set(rects)) == n_rect)
    clean = np.full((n, m), background, int)
    for idx in rng.permutation(n_rect):
        r0, r1, c0, c1 = rects[int(idx)]
        clean[r0:r1, c0:c1] = colours[int(idx)]
    try:
        scene, bg, vis, _, _ = _reconstruct(clean, np.zeros((n, m), bool))
    except AssertionError:
        raise _Reject()
    _need(bg == background and len(vis) == n_rect and np.array_equal(scene, clean))
    edge_vals = np.concatenate((clean[0], clean[-1], clean[:, 0], clean[:, -1]))
    _need(np.mean(edge_vals == background) >= 0.4)
    scene_palette = [background] + colours
    candidates = [(r, c) for r in range(1, n - 1) for c in range(1, m - 1)]
    count = int(rng.integers(stamp_range[0], stamp_range[1] + 1))
    centres = []
    for _ in range(count):
        ok = [p for p in candidates if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 4 for q in centres)]
        if not ok:
            break
        mixed = [p for p in ok if len({int(clean[x]) for x in _plus(*p)}) > 1]
        if mixed and rng.random() < 0.6:
            ok = mixed
        centres.append(ok[int(rng.integers(len(ok)))])
    _need(len(centres) >= stamp_range[0])
    damaged = clean.copy()
    damage = np.zeros((n, m), bool)
    for r, c in centres:
        cells = _plus(r, c)
        under = [int(clean[x]) for x in cells]
        options = [v for v in scene_palette if sum(u != v for u in under) >= 3]
        _need(options)
        colour = int(options[int(rng.integers(len(options)))])
        for x in cells:
            damaged[x] = colour
            damage[x] = True
    _need(np.sum(damaged != clean) >= 3)
    try:
        scene, bg, vis, bounds, reach = _reconstruct(damaged, damage)
    except AssertionError:
        raise _Reject()
    _need(np.array_equal(scene, clean))
    _need(_bounds_tight(damaged, damage, bg, vis, bounds, reach))
    feats = _features(damaged, clean, damage)
    for key in need:
        _need(feats[key])
    return damaged, clean, damage, centres


def _features(damaged, clean, damage):
    """Properties showing which shortcuts a pair refutes."""
    n, m = clean.shape
    _, background, colours, bounds, _ = _reconstruct(damaged, damage)
    hidden = [tuple(int(v) for v in p) for p in np.argwhere(damage)]
    covering = [sum(b[0] <= r < b[1] and b[2] <= c < b[3] for b in bounds) for r, c in hidden]
    # Shortcut: A-style erase of mapped cells to background.
    erase = damaged.copy()
    erase[damage] = background
    # Shortcut: ignore the map and fit rectangles to every cell.
    try:
        blind, *_ = _reconstruct(damaged, np.zeros_like(damage))
        blind_ok = np.array_equal(blind, clean)
    except AssertionError:
        blind_ok = False
    # Shortcut: fill each mapped cell from the majority of its visible 8-neighbours.
    local = damaged.copy()
    for r, c in hidden:
        vals = [int(damaged[rr, cc]) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                if 0 <= rr < n and 0 <= cc < m and not damage[rr, cc]]
        if vals:
            local[r, c] = max(set(vals), key=lambda v: (vals.count(v), -v))
    stamps = _components(damage)
    mixed = sum(len({int(clean[x]) for x in s}) > 1 for s in stamps)
    return {
        "erase_differs": not np.array_equal(erase, clean),
        "blind_differs": not blind_ok,
        "local_differs": not np.array_equal(local, clean),
        "layer_hidden": any(cv >= 2 for cv in covering),
        "mixed_stamp": mixed > 0,
        "uniform_stamp": mixed < len(stamps),
        "stamps": len(stamps),
        "rects": len(colours),
        "unchanged_hidden": bool(np.any(damaged[damage] == clean[damage])),
    }


def _place(rng, H, W, n, m):
    """Choose canvas size and positions for map (H x W) and payload (n x m)."""
    options = []
    if W + m + 3 <= _MAX and max(H, n) + 2 <= _MAX:
        options.append("h")
    if H + n + 3 <= _MAX and max(W, m) + 2 <= _MAX:
        options.append("v")
    _need(options)
    orient = options[int(rng.integers(len(options)))]
    for _ in range(50):
        if orient == "h":
            gap = int(rng.integers(1, 4))
            ml, mr = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            cw = W + m + gap + ml + mr
            if cw > _MAX:
                continue
            ch = min(_MAX, max(H, n) + 2 + int(rng.integers(0, 5)))
            map_left = bool(rng.integers(2))
            mc = ml if map_left else ml + m + gap
            pc = ml + W + gap if map_left else ml
            mrow = int(rng.integers(1, ch - H))
            prow = int(rng.integers(1, ch - n))
        else:
            gap = int(rng.integers(1, 4))
            mt, mb = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            ch = H + n + gap + mt + mb
            if ch > _MAX:
                continue
            cw = min(_MAX, max(W, m) + 2 + int(rng.integers(0, 5)))
            map_top = bool(rng.integers(2))
            mrow = mt if map_top else mt + n + gap
            prow = mt + H + gap if map_top else mt
            mc = int(rng.integers(1, cw - W))
            pc = int(rng.integers(1, cw - m))
        return ch, cw, (mrow, mc), (prow, pc), orient
    raise _Reject()


def _feasible(k, n, m):
    H, W = n * k, m * k
    return ((W + m + 3 <= _MAX and max(H, n) + 2 <= _MAX) or
            (H + n + 3 <= _MAX and max(W, m) + 2 <= _MAX))


def _shapes(k, lo, hi):
    return [(n, m) for n in range(lo, hi + 1) for m in range(lo, hi + 1) if _feasible(k, n, m)]


def _make_pair(rng, k, n, m, roles, rect_range, stamp_range, need):
    canvas, intact, mark = roles
    palette = [v for v in range(10) if v != canvas]
    target = int(rng.integers(rect_range[0], rect_range[1] + 1))
    for attempt in range(400):
        # Keep the sampled layer count for a while before relaxing toward the minimum.
        count = max(rect_range[0], target - attempt // 150)
        try:
            damaged, clean, damage, _ = _make_payload(rng, n, m, (count, count), stamp_range, palette, need)
            ch, cw, (mr, mc), (pr, pc), orient = _place(rng, n * k, m * k, n, m)
        except _Reject:
            continue
        grid = np.full((ch, cw), canvas, int)
        compact = np.where(damage, mark, intact)
        grid[mr:mr + n * k, mc:mc + m * k] = np.kron(compact, np.ones((k, k), int))
        grid[pr:pr + n, pc:pc + m] = damaged
        if not np.array_equal(solve(grid), clean):
            continue
        return grid, clean
    raise RuntimeError("could not build a pair")


def _roles(rng):
    trio = rng.choice(10, 3, replace=False)
    return int(trio[0]), int(trio[1]), int(trio[2])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_test = 1 if num_test is None else _count(num_test, "num_test", 1, 4)
    num_train = max(4, num_test + 1) if num_train is None else _count(num_train, "num_train", 2, 10)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Colour roles: fixed per puzzle, or redrawn for every pair in some seeds.
    local_roles = rng.random() < 0.35
    puzzle_roles = _roles(rng)
    train_scales = [2, 3] if rng.random() < 0.75 else [2, 3, 4]
    seen_combos = set()
    puzzle = {"train": [], "test": []}
    inputs = set()
    first_k = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _attempt in range(60):
                roles = _roles(rng) if local_roles else puzzle_roles
                if split == "train":
                    if index == 0:
                        k = int(rng.choice(train_scales))
                    elif index == 1:
                        k = int(rng.choice([s for s in train_scales if s != first_k]))
                    else:
                        k = int(rng.choice(train_scales))
                    shapes = _shapes(k, 5, 8)
                    rect_range = (2, 4)
                    stamp_range = (1, 2)
                    need = (("erase_differs", "blind_differs", "layer_hidden", "local_differs")
                            if index == 0 else ("mixed_stamp", "erase_differs") if index == 1 else ())
                else:
                    k = int(rng.choice([2, 3, 4]))
                    shapes = [s for s in _shapes(k, 5, 10) if (k,) + s not in seen_combos]
                    rect_range = (3, 5)
                    stamp_range = (1, 3) if rng.random() < 0.3 else (2, 3)
                    need = ("erase_differs",)
                if not shapes:
                    continue
                n, m = shapes[int(rng.integers(len(shapes)))]
                if split == "test" and n * m < 30:
                    continue
                try:
                    grid, out = _make_pair(rng, k, n, m, roles, rect_range, stamp_range, need)
                except RuntimeError:
                    continue
                key = (grid.shape, grid.tobytes())
                if key in inputs:
                    continue
                inputs.add(key)
                if split == "train":
                    seen_combos.add((k, n, m))
                    if index == 0:
                        first_k = k
                puzzle[split].append({"input": grid, "output": out})
                break
            else:
                raise RuntimeError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


# ---------------------------------------------------------------------------
# Validation and self test
# ---------------------------------------------------------------------------

def _pair_info(inp, out):
    info = _parse(inp)
    damage, payload = info["damage"], info["payload"]
    _check(out.shape == payload.shape, "output must have payload dimensions")
    for comp in _components(damage):
        _check(len(comp) == 5 and any(set(_plus(*p)) == set(comp) for p in comp),
               "damaged macrocells must form separate plus stamps")
    scene, background, colours, bounds, reach = _reconstruct(payload, damage)
    _check(np.array_equal(scene, out), "output is not the reconstructed scene")
    _check(np.array_equal(payload[~damage], out[~damage]), "undamaged cells must be copied")
    _check(np.any(payload != out), "pair shows no repair")
    _check(_bounds_tight(payload, damage, background, colours, bounds, reach),
           "rectangle bounds are not pinned by visible evidence")
    _check(info["canvas"] not in set(int(v) for v in np.unique(payload)), "canvas colour reused in payload")
    feats = _features(payload, out, damage)
    feats.update(scale=info["scale"], shape=payload.shape, roles=(info["canvas"], info["intact"], info["mark"]))
    return feats


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "illegal pair counts")
    seen = set()
    stats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for g in (inp, out):
                _check(g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "grid size")
                _check(np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9, "grid colours")
            key = (inp.shape, inp.astype(int).tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            _check(np.array_equal(solve(inp), out), "solve disagrees with output")
            stats[split].append(_pair_info(inp, out))
    train = stats["train"]
    _check(len({s["scale"] for s in train}) >= 2, "training must show at least two scales")
    for feat in ("erase_differs", "blind_differs", "layer_hidden", "mixed_stamp", "local_differs"):
        _check(any(s[feat] for s in train), f"training lacks witness: {feat}")
    combos = {(s["scale"],) + tuple(s["shape"]) for s in train}
    for s in stats["test"]:
        _check((s["scale"],) + tuple(s["shape"]) not in combos, "test repeats a training size/scale")


def self_test() -> None:
    import time
    start = time.time()
    seeds = list(range(42))
    scales, shapes, layouts, rects, stamps = set(), set(), set(), set(), set()
    role_sets, unseen_scale = set(), 0
    for seed in seeds:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        info = [_pair_info(np.asarray(q["input"]), np.asarray(q["output"])) for q in p["train"] + p["test"]]
        for q, s in zip(p["train"] + p["test"], info):
            scales.add(s["scale"])
            shapes.add(tuple(s["shape"]))
            rects.add(s["rects"])
            stamps.add(s["stamps"])
            role_sets.add(s["roles"])
            layouts.add(np.asarray(q["input"]).shape)
        train_k = {s["scale"] for s in info[:len(p["train"])]}
        unseen_scale += any(s["scale"] not in train_k for s in info[len(p["train"]):])
    _check(scales == {2, 3, 4}, "scales lack coverage")
    _check(len(shapes) >= 15 and len(layouts) >= 60, "geometry lacks diversity")
    _check(len(rects) >= 3 and len(stamps) >= 3, "scene counts lack diversity")
    _check(len(role_sets) >= 20, "colour roles lack diversity")
    _check(unseen_scale >= 3, "tests never extrapolate scale")
    a = generate_puzzle(123, 6, 3)
    b = generate_puzzle(123, 6, 3)
    _check(all(np.array_equal(x[r], y[r]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for r in ("input", "output")), "nondeterministic")
    for bad in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5)):
        try:
            generate_puzzle(5, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _check(time.time() - start < 120, "self test too slow")
