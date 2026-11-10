"""Seeded rooms: lone marker cells dye the room that grouped marker walls enclose.

Rule (input-only):
  * background = the strictly most frequent color; marker = the only other color.
  * Partition marker cells into 4-connected components (parent A).  Components
    of two or more cells are WALLS; single cells with no same-colored
    orthogonal neighbor are SEEDS.  Diagonal contact never joins a component.
  * Walls are impermeable; seeds are not.  Flood the non-wall cells with
    4-connectivity to obtain ROOMS (parent B).  Two wall pieces that touch
    only diagonally still seal a room, because the 4-flood cannot pass the gap.
  * Every room that contains at least one seed is dyed: all of its background
    cells become the fixed dye color 8.  Seeds, walls and undyed rooms keep
    their colors; the grid shape is unchanged.

Only the standard library and NumPy are used.
"""
from __future__ import annotations

from collections import deque

import numpy as np

DYE = 8
INPUT_COLORS = tuple(c for c in range(10) if c != DYE)

_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_N8 = _N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------
def _label(mask, conn=4):
    """Label True cells of mask; returns (labels, sizes) with labels 1..n."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    sizes = [0]
    steps = _N4 if conn == 4 else _N8
    current = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labels[r, c] == 0:
                current += 1
                labels[r, c] = current
                queue = deque([(r, c)])
                count = 0
                while queue:
                    y, x = queue.popleft()
                    count += 1
                    for dy, dx in steps:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
                sizes.append(count)
    return labels, np.asarray(sizes, dtype=int)


def _roles(grid):
    """Return (background, marker) or (background, None) for a one-color grid."""
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    bg = int(values[order[0]])
    marker = int(values[order[1]]) if len(values) > 1 else None
    return bg, marker


def _analyze(grid, wall_conn=4, room_conn=4, free_objects_seed=False):
    """Shared engine for the true rule and for the shortcut alternatives."""
    grid = np.asarray(grid, dtype=int)
    bg, marker = _roles(grid)
    out = grid.copy()
    if marker is None:
        return out, {"rooms": 0, "dyed": 0}
    is_marker = grid == marker
    mlab, msize = _label(is_marker, wall_conn)
    cell_size = msize[mlab]
    wall = is_marker & (cell_size >= 2)
    seeds = is_marker & (cell_size == 1)
    rlab, rsize = _label(~wall, room_conn)
    dyed = set(int(v) for v in np.unique(rlab[seeds]) if v > 0)
    if free_objects_seed:
        # Alternative: every marker object that does not partition space
        # (touches exactly one room) behaves like a seed.
        wlab, wsize = _label(wall, 4)
        h, w = grid.shape
        for comp in range(1, len(wsize)):
            ys, xs = np.nonzero(wlab == comp)
            adjacent = set()
            for y, x in zip(ys, xs):
                for dy, dx in _N4:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and rlab[ny, nx] > 0:
                        adjacent.add(int(rlab[ny, nx]))
            if len(adjacent) == 1:
                dyed |= adjacent
    if dyed:
        in_dyed = np.isin(rlab, sorted(dyed))
        out[(grid == bg) & in_dyed] = DYE
    return out, {"rooms": len(rsize) - 1, "dyed": len(dyed)}


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    return _analyze(grid)[0]


# Shortcut / ablation predictors used to prove both parents matter.
def _shortcut_outputs(grid):
    grid = np.asarray(grid, dtype=int)
    bg, marker = _roles(grid)
    out = {}
    # S1: ignore parent A -- every marker cell is a wall, nothing can seed.
    out["all_markers_block"] = grid.copy()
    # S2: any free-standing marker object (dot or small group) seeds its room.
    out["any_object_seeds"] = _analyze(grid, free_objects_seed=True)[0]
    # S3: group markers with 8-connectivity (diagonal contact joins walls).
    out["diagonal_grouping"] = _analyze(grid, wall_conn=8)[0]
    # S4: ignore parent B -- dye the whole background if any seed exists.
    glob = grid.copy()
    if marker is not None:
        mlab, msize = _label(grid == marker, 4)
        if np.any((grid == marker) & (msize[mlab] == 1)):
            glob[grid == bg] = DYE
    out["ignore_rooms"] = glob
    # S5: rooms flood with 8-connectivity (diagonal gaps between walls leak).
    out["diagonal_flood"] = _analyze(grid, room_conn=8)[0]
    return out


# ---------------------------------------------------------------------------
# Wall drawing primitives (all operate on a boolean marker mask)
# ---------------------------------------------------------------------------
def _draw_strip(mask, rng, lo, hi, vertical, thickness, kink_p, dotted):
    """Boundary-to-boundary winding strip inside band [lo, hi] (inclusive).

    Returns a dict describing special features actually drawn.
    """
    work = mask if vertical else mask.T
    length, span = work.shape
    t = thickness
    if hi - lo + 1 < t:
        return None
    c = int(rng.integers(lo, hi - t + 2))
    run = 0
    kinks = 0
    dotted_done = False
    dotted_rows = None
    if dotted and length >= 9 and t == 1:
        start = int(rng.integers(3, length - 5))
        dotted_rows = (start, start + int(rng.integers(2, min(5, length - start - 3) + 1)))
    r = 0
    direction = int(rng.choice([-1, 1]))
    while r < length:
        if dotted_rows is not None and r == dotted_rows[0] and run >= 2:
            # Diagonal chain of single cells: looks like a fence, but leaks.
            a, b = dotted_rows
            d = direction
            ok = True
            cc = c
            for _ in range(a, b + 2):
                cc += d
                if cc < lo or cc > hi:
                    ok = False
                    break
            if ok:
                for rr in range(a, b + 1):
                    c += d
                    work[rr, c] = True
                c += d
                r = b + 1
                run = 0
                dotted_done = True
                dotted_rows = None
                continue
            dotted_rows = None
        if r > 0 and run >= 2 and rng.random() < 0.3:
            if rng.random() < kink_p and r < length - 2:
                nc = c + direction * t
                if lo <= nc and nc + t - 1 <= hi:
                    c = nc
                    kinks += 1
                    run = 0
                    work[r, c:c + t] = True
                    run += 1
                    r += 1
                    continue
            step = int(rng.choice([1, 1, 1, 2, 3]))
            d = int(rng.choice([-1, 1])) if rng.random() < 0.5 else direction
            nc = int(np.clip(c + d * step, lo, hi - t + 1))
            if nc != c:
                left, right = min(c, nc), max(c, nc) + t
                work[r - 1, left:right] = True
                c = nc
                direction = d
                run = 0
        work[r, c:c + t] = True
        run += 1
        r += 1
    return {"kinks": kinks, "dotted": dotted_done}


def _draw_corner(mask, rng, corner):
    """Adjacent-edge cut that isolates a corner pocket."""
    h, w = mask.shape
    depth = int(rng.integers(1, max(2, min(h // 3, 7)) + 1))
    reach = int(rng.integers(1, max(2, min(w // 3, 7)) + 1))
    t = 1 if rng.random() < 0.75 else 2
    work = np.zeros_like(mask)
    # Draw for top-left; flip afterwards.
    for r in range(0, depth + t):
        work[r, reach:reach + t] = True
    for c in range(0, reach + t):
        work[depth:depth + t, c] = True
    if corner in (1, 3):
        work = work[:, ::-1]
    if corner in (2, 3):
        work = work[::-1, :]
    mask |= work
    return True


def _draw_box(mask, rng, cornerless):
    """Closed rectangle outline placed with a one-cell clear margin."""
    h, w = mask.shape
    for _ in range(30):
        bh = int(rng.integers(4, max(5, min(h - 2, 9)) + 1))
        bw = int(rng.integers(4, max(5, min(w - 2, 9)) + 1))
        if bh > h - 2 or bw > w - 2:
            continue
        r0 = int(rng.integers(1, h - bh))
        c0 = int(rng.integers(1, w - bw))
        if mask[r0 - 1:r0 + bh + 1, c0 - 1:c0 + bw + 1].any():
            continue
        box = np.zeros_like(mask)
        box[r0, c0:c0 + bw] = True
        box[r0 + bh - 1, c0:c0 + bw] = True
        box[r0:r0 + bh, c0] = True
        box[r0:r0 + bh, c0 + bw - 1] = True
        if cornerless:
            for y, x in ((r0, c0), (r0, c0 + bw - 1), (r0 + bh - 1, c0), (r0 + bh - 1, c0 + bw - 1)):
                box[y, x] = False
        mask |= box
        return True
    return False


_DECOYS = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _clear4(mask, y, x):
    h, w = mask.shape
    if mask[y, x]:
        return False
    for dy, dx in _N4:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
            return False
    return True


def _has_diag_marker(mask, y, x):
    h, w = mask.shape
    for dy, dx in _DIAG:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
            return True
    return False


def _place_decoy(mask, rng, room_cells, rlab):
    h, w = mask.shape
    order = rng.permutation(len(room_cells))
    for _ in range(3):
        shape = _DECOYS[int(rng.integers(len(_DECOYS)))]
        for idx in order[:80]:
            y0, x0 = room_cells[idx]
            cells = [(y0 + dy, x0 + dx) for dy, dx in shape]
            if not all(0 <= y < h and 0 <= x < w for y, x in cells):
                continue
            room = rlab[y0, x0]
            if any(rlab[y, x] != room for y, x in cells):
                continue
            cellset = set(cells)
            ok = True
            for y, x in cells:
                if mask[y, x]:
                    ok = False
                    break
                for dy, dx in _N4:
                    ny, nx = y + dy, x + dx
                    if (ny, nx) in cellset:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                for y, x in cells:
                    mask[y, x] = True
                return True
    return False


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
_SIZE_TIERS = {"small": (6, 10), "medium": (10, 16), "large": (15, 23), "xl": (21, 30)}


def _sample_shape(rng, tier):
    lo, hi = _SIZE_TIERS[tier]
    h = int(rng.integers(lo, hi + 1))
    w = int(rng.integers(lo, hi + 1))
    if rng.random() < 0.3:
        if rng.random() < 0.5:
            w = int(min(30, w + rng.integers(3, 9)))
        else:
            h = int(min(30, h + rng.integers(3, 9)))
    return h, w


def _build_walls(rng, h, w, plan):
    mask = np.zeros((h, w), dtype=bool)
    layout = plan["layout"]
    kink_p = 0.55 if "kink" in plan["features"] else 0.15
    want_dotted = "fence" in plan["features"]
    dotted_done = False

    def strips(vertical, count):
        nonlocal dotted_done
        span = w if vertical else h
        count = max(1, min(count, (span + 1) // 4))
        # Split the cross dimension into bands separated by >= 2 free cells.
        edges = np.linspace(0, span, count + 1)
        for i in range(count):
            lo = int(edges[i]) + (1 if i > 0 else 0) + 1
            hi = int(edges[i + 1]) - 2
            if hi < lo:
                lo, hi = int(edges[i]) + 1, max(int(edges[i]) + 1, int(edges[i + 1]) - 1)
            hi = min(hi, span - 2)
            lo = max(lo, 1)
            if hi < lo:
                continue
            t = 1 if rng.random() < 0.7 else 2
            dotted = want_dotted and not dotted_done and rng.random() < 0.6
            if dotted:
                t = 1
            info = _draw_strip(mask, rng, lo, hi, vertical, t, kink_p, dotted)
            if info and info["dotted"]:
                dotted_done = True

    if layout == "vertical":
        strips(True, int(rng.integers(1, 5)))
    elif layout == "horizontal":
        strips(False, int(rng.integers(1, 5)))
    elif layout == "cross":
        strips(True, int(rng.integers(1, 3)))
        strips(False, int(rng.integers(1, 3)))
    if layout == "boxes" or rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 3)) if layout == "boxes" else 1):
            cornerless = rng.random() < (0.6 if "kink" in plan["features"] else 0.3)
            _draw_box(mask, rng, cornerless)
    n_corners = int(rng.choice([0, 0, 1, 1, 2])) if layout != "boxes" else int(rng.integers(1, 3))
    corners = rng.permutation(4)[:n_corners]
    for corner in corners:
        _draw_corner(mask, rng, int(corner))
    return mask


def _build_pair(rng, plan):
    """Return (input, output) or None if the attempt fails."""
    h, w = _sample_shape(rng, plan["tier"])
    if plan["layout"] == "boxes":
        h, w = max(h, 9), max(w, 9)
    mask = _build_walls(rng, h, w, plan)
    # Classify the current marker cells and compute rooms.
    mlab, msize = _label(mask, 4)
    wall = mask & (msize[mlab] >= 2)
    rlab, rsize = _label(~wall, 4)
    n_rooms = len(rsize) - 1
    if n_rooms < 2:
        return None
    pre_seeds = mask & (msize[mlab] == 1)
    forced = set(int(v) for v in np.unique(rlab[pre_seeds]) if v > 0)
    eligible = []
    for room in range(1, n_rooms + 1):
        if room in forced:
            continue
        ys, xs = np.nonzero(rlab == room)
        cands = [(y, x) for y, x in zip(ys, xs) if _clear4(mask, y, x)]
        if cands:
            eligible.append(room)
    if not eligible and not forced:
        return None
    rng.shuffle(eligible)
    total_dyeable = len(eligible) + len(forced)
    all_dyed_ok = plan.get("allow_all_dyed", False)
    max_k = len(eligible) if (all_dyed_ok or len(forced) + len(eligible) < n_rooms) else len(eligible) - 1
    min_k = 0 if forced else 1
    if max_k < min_k:
        return None
    k = int(rng.integers(min_k, max_k + 1))
    dyed_rooms = eligible[:k]
    undyed_rooms = [r for r in range(1, n_rooms + 1) if r not in dyed_rooms and r not in forced]
    # Seeds.
    diag_room = None
    if "diag" in plan["features"] and dyed_rooms:
        diag_room = dyed_rooms[0]
    for room in dyed_rooms:
        ys, xs = np.nonzero(rlab == room)
        cells = [(int(y), int(x)) for y, x in zip(ys, xs)]
        if room == diag_room:
            n = 1
        else:
            n = int(rng.choice([1, 1, 1, 1, 2, 2, 3]))
            n = min(n, max(1, len(cells) // 8))
        placed = 0
        prefer_diag = room == diag_room or rng.random() < 0.25
        for attempt in range(n * 40):
            if placed >= n:
                break
            cands = [(y, x) for y, x in cells if _clear4(mask, y, x)]
            if prefer_diag and placed == 0:
                dc = [(y, x) for y, x in cands if _has_diag_marker(wall, y, x)]
                if dc:
                    cands = dc
                elif room == diag_room:
                    return None
            if not cands:
                break
            y, x = cands[int(rng.integers(len(cands)))]
            mask[y, x] = True
            placed += 1
        if placed == 0:
            return None
    # Decoys: grouped marker objects that do not seed.
    n_decoy_rooms = 0
    if undyed_rooms:
        if "decoy" in plan["features"]:
            n_decoy_rooms = max(1, int(rng.integers(1, len(undyed_rooms) + 1)))
        elif rng.random() < 0.35:
            n_decoy_rooms = 1
    rng.shuffle(undyed_rooms)
    for room in undyed_rooms[:n_decoy_rooms]:
        ys, xs = np.nonzero(rlab == room)
        cells = [(int(y), int(x)) for y, x in zip(ys, xs)]
        for _ in range(int(rng.integers(1, 3))):
            _place_decoy(mask, rng, cells, rlab)
    if dyed_rooms and rng.random() < 0.3:
        room = dyed_rooms[int(rng.integers(len(dyed_rooms)))]
        ys, xs = np.nonzero(rlab == room)
        _place_decoy(mask, rng, [(int(y), int(x)) for y, x in zip(ys, xs)], rlab)
    # Background dominance.
    if mask.sum() * 100 >= mask.size * 38:
        return None
    bg, mk = plan["palette"]
    grid = np.where(mask, mk, bg).astype(int)
    out = solve(grid)
    return grid, out


def _pair_features(grid, out):
    """Grid-derived evidence used by the planner and validator."""
    sc = _shortcut_outputs(grid)
    _, info = _analyze(grid)
    refuted = {name for name, pred in sc.items() if not np.array_equal(pred, out)}
    return {"refuted": refuted, "rooms": info["rooms"], "dyed": info["dyed"]}


def _plan_feature_ok(feat, features):
    need = {"decoy": "any_object_seeds", "diag": "diagonal_grouping", "kink": "diagonal_flood"}
    for f in features:
        if f in need and need[f] not in feat["refuted"]:
            return False
    return True


def _check_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("Counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Need 2 <= num_train <= 10, 1 <= num_test <= 4, num_train > num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    num_train, num_test = int(num_train), int(num_test)
    rng = np.random.default_rng(seed)
    total = num_train + num_test

    # Puzzle-level conventions.
    palette_mode = "fixed" if rng.random() < 0.6 else "per_pair"
    base = tuple(int(v) for v in rng.choice(INPUT_COLORS, size=2, replace=False))
    layouts = ["vertical", "horizontal", "cross", "boxes"]
    tiers = ["small", "medium", "large", "xl"]

    # Distribute the three demonstrable distinctions among training pairs.
    train_feats = [set() for _ in range(num_train)]
    for f in ("decoy", "diag", "kink"):
        train_feats[int(rng.integers(num_train))].add(f)
    for fs in train_feats:
        for f in ("decoy", "diag", "kink", "fence"):
            if rng.random() < 0.25:
                fs.add(f)
    test_feats = []
    for _ in range(num_test):
        fs = set(rng.choice(["decoy", "diag", "kink", "fence"], size=2, replace=False).tolist())
        test_feats.append(fs)

    pairs = []
    seen = set()
    for i in range(total):
        is_test = i >= num_train
        features = test_feats[i - num_train] if is_test else train_feats[i]
        for attempt in range(400):
            if palette_mode == "fixed":
                palette = base
            else:
                palette = tuple(int(v) for v in rng.choice(INPUT_COLORS, size=2, replace=False))
            if is_test:
                tier = str(rng.choice(tiers, p=[0.1, 0.3, 0.35, 0.25]))
            else:
                tier = str(rng.choice(tiers, p=[0.25, 0.35, 0.25, 0.15]))
            plan = {
                "layout": str(rng.choice(layouts, p=[0.3, 0.3, 0.25, 0.15])),
                "tier": tier,
                "features": features if attempt < 300 else set(),
                "palette": palette,
                "allow_all_dyed": (not is_test and rng.random() < 0.12),
            }
            built = _build_pair(rng, plan)
            if built is None:
                continue
            grid, out = built
            key = grid.tobytes() + bytes(grid.shape)
            if key in seen:
                continue
            feat = _pair_features(grid, out)
            if feat["rooms"] < 2 or feat["dyed"] < 1:
                continue
            if not _plan_feature_ok(feat, plan["features"]):
                continue
            if is_test:
                special = feat["refuted"] & {"any_object_seeds", "diagonal_grouping", "diagonal_flood"}
                if len(feat["refuted"]) < 3 or not special or feat["rooms"] < 3:
                    continue
            seen.add(key)
            pairs.append({"input": grid, "output": out})
            break
        else:
            raise RuntimeError(f"could not build pair {i} for seed {seed}")
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    validate_generated_puzzle(puzzle)
    return puzzle


# ---------------------------------------------------------------------------
# Validation and self test
# ---------------------------------------------------------------------------
def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_refuted = set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            g = np.asarray(pair["input"])
            o = np.asarray(pair["output"])
            assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
            assert o.shape == g.shape
            assert np.issubdtype(g.dtype, np.integer) and np.issubdtype(o.dtype, np.integer)
            colors = set(np.unique(g).tolist())
            assert len(colors) == 2 and DYE not in colors, "input must use two non-dye colors"
            bg, mk = _roles(g)
            assert (g == bg).sum() > (g == mk).sum(), "background must be strictly dominant"
            key = g.tobytes() + bytes(g.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert np.array_equal(o, solve(g)), "output disagrees with rule"
            changed = o != g
            assert np.all(g[changed] == bg) and np.all(o[changed] == DYE)
            feat = _pair_features(g, o)
            assert feat["rooms"] >= 2 and feat["dyed"] >= 1
            assert {"all_markers_block"} <= feat["refuted"]
            if split == "train":
                train_refuted |= feat["refuted"]
            else:
                assert len(feat["refuted"]) >= 3 and feat["rooms"] >= 3
                assert feat["refuted"] & {"any_object_seeds", "diagonal_grouping", "diagonal_flood"}
    need = {"all_markers_block", "any_object_seeds", "diagonal_grouping", "ignore_rooms", "diagonal_flood"}
    missing = need - train_refuted
    assert not missing, f"training does not refute shortcuts {sorted(missing)}"
    return True


def _hand_cases():
    # Seed diagonal to a wall dyes its room; a domino decoy does not.
    g = np.array([
        [0, 0, 5, 0, 0, 0],
        [0, 0, 5, 0, 0, 0],
        [0, 5, 0, 0, 5, 5],
        [0, 5, 0, 0, 0, 0],
        [0, 5, 0, 0, 0, 0],
    ])
    out = solve(g)
    # Wall pieces (0,2)-(1,2) and (2,1)-(4,1) touch diagonally only: they seal
    # the top-left pocket.  No seeds exist (all components size >= 2).
    assert np.array_equal(out, g)
    g2 = g.copy()
    g2[4, 4] = 5  # lone seed in the right room
    out2 = solve(g2)
    assert out2[0, 3] == DYE and out2[4, 3] == DYE and out2[0, 0] == 0 and out2[4, 0] == 0
    assert out2[4, 4] == 5 and out2[2, 4] == 5
    # 8-connected flood would leak into the top-left pocket.
    assert _shortcut_outputs(g2)["diagonal_flood"][0, 0] == DYE


def self_test():
    _hand_cases()
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    shapes, palettes, room_counts, layouts_seen = set(), set(), set(), set()
    signatures = set()
    refuted_tests = {}
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        for split in ("train", "test"):
            for x, y in zip(p[split], q[split]):
                assert np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        sig = b"".join(np.asarray(pr["input"]).tobytes() for pr in p["train"] + p["test"])
        signatures.add(sig)
        for pr in p["train"] + p["test"]:
            g = np.asarray(pr["input"])
            shapes.add(g.shape)
            palettes.add(_roles(g))
            room_counts.add(_analyze(g)[1]["rooms"])
        for pr in p["test"]:
            for name in _pair_features(np.asarray(pr["input"]), np.asarray(pr["output"]))["refuted"]:
                refuted_tests[name] = refuted_tests.get(name, 0) + 1
    assert len(signatures) == 42
    assert len(shapes) >= 30, len(shapes)
    assert len(palettes) >= 15, len(palettes)
    assert len(room_counts) >= 5, room_counts
    for name in ("any_object_seeds", "diagonal_grouping", "diagonal_flood"):
        assert refuted_tests.get(name, 0) >= 3, refuted_tests
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
