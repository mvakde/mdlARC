"""Fault-guided disassembly: recover panels, scan holes, route damaged cuts.

Only the standard library and NumPy are used. Labels are painted from explicit
construction masks; solve reconstructs those masks from input pixels alone.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, queue = [start], deque([start])
        while queue:
            r, c = queue.popleft()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    found.append(q)
                    queue.append(q)
        result.append(found)
    return result


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            out |= p[dr:dr+h, dc:dc+w]
    return out


def _supported_cuts(mask):
    """A genuine boundary has at least three consecutive signed edge votes."""
    delta = np.diff(mask.astype(np.int8), axis=0)
    support = np.zeros(mask.shape[0]-1, dtype=bool)
    for c in range(mask.shape[1]-2):
        window = delta[:, c:c+3]
        support |= np.all(window == 1, axis=1) | np.all(window == -1, axis=1)
    return [0] + (np.flatnonzero(support)+1).tolist() + [mask.shape[0]]


def _recover(observed):
    """Recover the uniquely supported rectangular partition, on both axes.

    The family requires every real boundary to have three intact signed votes,
    no false boundary to have those votes, and every partition tile to retain
    a strict majority of its true color. No unobserved shape vocabulary is used.
    """
    yy, xx = _supported_cuts(observed), _supported_cuts(observed.T)
    result = np.zeros_like(observed)
    for y0, y1 in zip(yy, yy[1:]):
        for x0, x1 in zip(xx, xx[1:]):
            tile = observed[y0:y1, x0:x1]
            votes = int(tile.sum())
            if 2*votes == tile.size:
                raise ValueError("tied scaffold tile")
            result[y0:y1, x0:x1] = 2*votes > tile.size
    if not result.any() or result.all():
        raise ValueError("scaffold must retain solid and field")
    return result


def _roles_and_markers(inp):
    h, w = inp.shape
    rim = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    colors, counts = np.unique(rim, return_counts=True)
    if len(colors) != 2 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("the rim must identify field and marker uniquely")
    field = int(colors[np.argmax(counts)])
    marker = int(next(c for c in colors if c != field))
    others = set(map(int, np.unique(inp))) - {field, marker}
    if len(others) != 1:
        raise ValueError("expected three visible semantic colors")
    solid = others.pop()
    markers = {}
    for r, c in np.argwhere(inp == marker):
        r, c = int(r), int(c)
        sides = (["top"] if r == 0 else []) + (["bottom"] if r == h-1 else [])
        sides += (["left"] if c == 0 else []) + (["right"] if c == w-1 else [])
        if len(sides) != 1 or sides[0] in markers:
            raise ValueError("markers must be single noncorner cells on different sides")
        markers[sides[0]] = (r, c)
    if len(markers) not in (1, 2):
        raise ValueError("expected one marker or two adjacent markers")
    if len(markers) == 2 and (len(set(markers) & {"top", "bottom"}) != 1
                              or len(set(markers) & {"left", "right"}) != 1):
        raise ValueError("two marked sides must be adjacent")
    return field, solid, marker, markers


def _axes(markers):
    ar = next((r for s, (r, c) in markers.items() if s in ("left", "right")), None)
    ac = next((c for s, (r, c) in markers.items() if s in ("top", "bottom")), None)
    return ar, ac


def _regions(shape, markers):
    rr, cc = np.indices(shape)
    ar, ac = _axes(markers)
    fixed = np.zeros(shape, dtype=bool)
    if ar is not None:
        fixed |= rr == ar
    if ac is not None:
        fixed |= cc == ac
    regions = {}
    for vertical in ("top", "bottom"):
        if ar is None and vertical not in markers:
            continue
        for horizontal in ("left", "right"):
            if ac is None and horizontal not in markers:
                continue
            region = ~fixed
            if ar is not None:
                region = region & ((rr < ar) if vertical == "top" else (rr > ar))
            if ac is not None:
                region = region & ((cc < ac) if horizontal == "left" else (cc > ac))
            regions[(vertical, horizontal)] = region
    return fixed, regions


def _selected(corner, markers):
    return any(side in markers for side in corner)


def _shift(points, corner, shape):
    lo, hi = points.min(axis=0), points.max(axis=0)
    return np.array((-lo[0] if corner[0] == "top" else shape[0]-1-hi[0],
                     -lo[1] if corner[1] == "left" else shape[1]-1-hi[1]))


def _ray(shape, side, position, anchors, nearest=False):
    r, c = position
    vertical = side in ("top", "bottom")
    aligned = np.flatnonzero(anchors[:, c] if vertical else anchors[r, :])
    if not len(aligned):
        raise ValueError("every marker axis must contain a missing-solid cell")
    from_low = side in ("top", "left")
    end = int((min(aligned) if from_low else max(aligned)) if nearest
              else (max(aligned) if from_low else min(aligned)))
    mask = np.zeros(shape, dtype=bool)
    start = 0 if from_low else end
    stop = end+1 if from_low else (shape[0] if vertical else shape[1])
    if vertical:
        mask[start:stop, c] = True
    else:
        mask[r, start:stop] = True
    return mask


def _analyze(inp):
    field, solid, marker, markers = _roles_and_markers(inp)
    base = _recover(inp == solid)
    rim = np.zeros_like(base)
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    if np.any(base & rim):
        raise ValueError("the recovered scaffold must be interior")
    holes = base & (inp == field)
    extras = ~base & (inp == solid)
    for side, point in markers.items():
        _ray(inp.shape, side, point, holes)
    return field, solid, marker, markers, base, holes, extras


def _transform(info, *, all_move=False, nearest=False, scaffold_endpoint=False):
    field, solid, marker, markers, base, holes, extras = info
    fixed, regions = _regions(base.shape, markers)
    moved_base, moved_holes = base & fixed, holes & fixed
    for corner, region in regions.items():
        cells = np.argwhere(base & region)
        if not len(cells) or not _selected(corner, markers):
            continue
        defect_points = np.argwhere(holes & region)
        translation = (_shift(cells, corner, base.shape)
                       if len(defect_points) or all_move else np.array((0, 0)))
        cells = cells + translation
        moved_base[cells[:, 0], cells[:, 1]] = True
        if len(defect_points):
            defect_points = defect_points + translation
            moved_holes[defect_points[:, 0], defect_points[:, 1]] = True
    out = np.where(moved_base, solid, field)
    out[moved_holes] = field
    guides = np.zeros_like(base)
    for side, point in markers.items():
        guides |= _ray(base.shape, side, point, base if scaffold_endpoint else holes, nearest)
    out[guides & ~moved_base & ~moved_holes] = marker
    halo = _dilate(moved_holes) & ~moved_holes
    out[halo] = marker
    return out.astype(np.int64), moved_base, moved_holes, guides, halo


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver: no seed, examples, file access, or generation state."""
    inp = _grid(input_grid)
    return _transform(_analyze(inp))[0]


def _paint_known(base, holes, extras, roles, markers):
    """Construction renderer, using known clean masks rather than recovery."""
    field, solid, marker = roles
    inp = np.where(base, solid, field).astype(np.int64)
    inp[holes] = field
    inp[extras] = solid
    for position in markers.values():
        inp[position] = marker
    ar, ac = _axes(markers)
    groups = {}
    survivors = []
    for r, c in np.argwhere(base):
        r, c = int(r), int(c)
        if r == ar or c == ac:
            survivors.append(((r, c), bool(holes[r, c])))
        else:
            vertical = ("top" if r < ar else "bottom") if ar is not None else next(iter(markers))
            horizontal = ("left" if c < ac else "right") if ac is not None else next(iter(markers))
            groups.setdefault((vertical, horizontal), []).append((r, c))
    for corner, cells in groups.items():
        if not _selected(corner, markers):
            continue
        dr = dc = 0
        if any(holes[p] for p in cells):
            rs, cs = zip(*cells)
            dr = -min(rs) if corner[0] == "top" else base.shape[0]-1-max(rs)
            dc = -min(cs) if corner[1] == "left" else base.shape[1]-1-max(cs)
        survivors.extend((((r+dr, c+dc), bool(holes[r, c])) for r, c in cells))
    out = np.full(base.shape, field, dtype=np.int64)
    final_holes = set()
    for point, missing in survivors:
        if missing:
            final_holes.add(point)
        else:
            out[point] = solid
    for side, (r, c) in markers.items():
        dr, dc = {"top": (1, 0), "bottom": (-1, 0),
                  "left": (0, 1), "right": (0, -1)}[side]
        ray, hits = [], []
        while 0 <= r < base.shape[0] and 0 <= c < base.shape[1]:
            ray.append((r, c))
            if holes[r, c]:
                hits.append(len(ray)-1)
            r, c = r+dr, c+dc
        if not hits:
            raise ValueError("construction is missing an axial fault")
        for point in ray[:max(hits)+1]:
            if point not in final_holes and out[point] == field:
                out[point] = marker
    for r, c in final_holes:
        for rr in range(max(0, r-1), min(base.shape[0], r+2)):
            for cc in range(max(0, c-1), min(base.shape[1], c+2)):
                if (rr, cc) not in final_holes:
                    out[rr, cc] = marker
    return {"input": inp, "output": out}


def _slabs(rng, large=False):
    top = int(rng.integers(4, 7))
    gap = int(rng.integers(3, 5))
    middle = int(rng.integers(4, 7))
    second_gap = int(rng.integers(3, 5))
    bottom = int(rng.integers(3, 5))
    w = int(rng.integers(12, 23 if large else 19))
    ac = int(rng.integers(4, w-4))
    height = top+gap+middle+second_gap+bottom
    mask = np.zeros((height, w), dtype=bool)
    for r0, r1 in ((0, top), (top+gap, top+gap+middle),
                   (height-bottom, height)):
        left = int(rng.integers(0, min(3, ac-2)))
        right = w-int(rng.integers(0, min(3, w-ac-2)))
        mask[r0:r1, left:right] = True
    # The first two panels contain axial faults; the third is an intact scanner
    # decoy. Its off-axis parts still move if their whole cut is damaged.
    return mask, top+gap+middle//2, ac, (top//2, top+gap+middle//2)


def _mosaic(rng, style, large=False):
    n = int(rng.integers(3 if large else 2, 5 if large else 4))
    m = int(rng.integers(3 if large else 2, 5 if large else 4))
    if style == "cavities":
        n = m = 4
    cy, cx = int(rng.integers(n)), int(rng.integers(m))
    heights = rng.integers(3, 6, size=n)
    widths = rng.integers(3, 6, size=m)
    heights[cy], widths[cx] = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    tiles = np.zeros((n, m), dtype=bool)
    tiles[cy, cx] = True
    if style == "rectangle":
        tiles[:] = True
    elif style == "arms":
        tiles[:, cx] = True
        tiles[cy, :] = True
    elif style == "cavities":
        tiles[:] = True
        possible = [(y, x) for y in range(1, n-1) for x in range(1, m-1)
                    if (y, x) != (cy, cx)]
        tiles[possible[int(rng.integers(len(possible)))]] = False
        for y in range(1, n-1):
            for x in range(1, m-1):
                if (y, x) != (cy, cx) and rng.random() < .65:
                    tiles[y, x] = False
    elif style == "separate":
        tiles |= rng.random((n, m)) < rng.uniform(.35, .65)
    else:
        wanted = int(rng.integers(max(2, n*m//3), n*m+1))
        while int(tiles.sum()) < wanted:
            frontier = sorted({(yy, xx) for y, x in np.argwhere(tiles)
                               for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1))
                               if 0 <= yy < n and 0 <= xx < m and not tiles[yy, xx]})
            tiles[frontier[int(rng.integers(len(frontier)))]] = True
    mask = np.repeat(np.repeat(tiles, heights, axis=0), widths, axis=1)
    ar = int(heights[:cy].sum()) + int(rng.integers(2, heights[cy]-2))
    ac = int(widths[:cx].sum()) + int(rng.integers(2, widths[cx]-2))
    return mask, ar, ac


_MOTIFS = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
           ((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (2, 2)),
           ((0, 0), (0, 2)))


def _add_motif(rng, allowed, holes, motif=None):
    motif = _MOTIFS[int(rng.integers(len(_MOTIFS)))] if motif is None else motif
    offsets = np.asarray(motif, dtype=int)
    if rng.random() < .5:
        offsets = offsets[:, ::-1]
    offsets = offsets * rng.choice((-1, 1), size=2)
    candidates = []
    for point in np.argwhere(allowed):
        pts = point + offsets
        if (np.all(pts >= 0) and np.all(pts < allowed.shape)
                and np.all(allowed[pts[:, 0], pts[:, 1]])
                and not np.any(holes[pts[:, 0], pts[:, 1]])):
            candidates.append(pts)
    if not candidates:
        return False
    pts = candidates[int(rng.integers(len(candidates)))]
    holes[pts[:, 0], pts[:, 1]] = True
    return True


def _make_pair(rng, roles, anchor, double, turn, style, large, expanded=False):
    """Bounded rejection sampling; accepts only visibly exact scaffold recovery."""
    for attempt in range(100):
        if anchor == 0:
            damaged_side = str(rng.choice(("left", "right")))
            patch, ar, ac, axial_rows = _slabs(rng, large)
            top_end = int(np.flatnonzero(~patch.any(axis=1))[0])
            # A narrow off-axis tip has two intended sites on its extreme row.
            # Both are absent, while the rest of this wide panel supports the
            # true boundary. Visible-solid box alignment is consequently wrong.
            if damaged_side == "left":
                patch[:top_end, :ac-2] = False
            else:
                patch[:top_end, ac+3:] = False
            double = False
        else:
            patch, ar, ac = _mosaic(rng, style, large)
            if style == "separate" and len(_components(patch)) < 2:
                continue
        ph, pw = patch.shape
        h = int(rng.integers(ph+4, min(30, ph+10)+1))
        w = int(rng.integers(pw+4, min(30, pw+10)+1))
        y0, x0 = int(rng.integers(2, h-ph-1)), int(rng.integers(2, w-pw-1))
        base = np.zeros((h, w), dtype=bool)
        base[y0:y0+ph, x0:x0+pw] = patch
        ar, ac = ar+y0, ac+x0
        markers = {"top": (0, ac)}
        if double:
            markers["right"] = (ar, w-1)
        fixed, regions = _regions(base.shape, markers)
        holes = np.zeros_like(base)
        if anchor == 0:
            for r in axial_rows:
                holes[r+y0, ac] = True
            damaged = [("top", damaged_side)]
            region = regions[damaged[0]]
            pts = np.argwhere(base & region)
            missing_row = pts[pts[:, 0] == pts[:, 0].min()]
            holes[missing_row[:, 0], missing_row[:, 1]] = True
        elif anchor == 1:
            # Adjacent axial/off-axis faults teach merged halos and hole priority.
            holes[ar-1, ac] = holes[ar, ac+1] = True
            holes[ar-1, ac-1] = holes[ar+1, ac+1] = True
            if rng.random() < .5:
                holes[ar, ac] = True
            damaged = [("top", "left"), ("bottom", "right")]
            # Top-right is intact despite a neighboring halo; bottom-left is
            # intact but discarded because neither of its sides is marked.
        else:
            if double and rng.random() < .35:
                holes[ar, ac] = True
            else:
                holes[ar-1, ac] = True
                if double:
                    holes[ar, ac+1] = True
            selected = [corner for corner in regions if _selected(corner, markers)]
            rng.shuffle(selected)
            damaged = selected if expanded else selected[:int(rng.integers(1, min(2, len(selected))+1))]
            for corner in damaged:
                if not _add_motif(rng, base & regions[corner], holes):
                    _add_motif(rng, base & regions[corner], holes, _MOTIFS[0])
            # Faults in the discarded quadrant vanish without leaving a halo.
            if double and (expanded or rng.random() < .6):
                _add_motif(rng, base & regions[("bottom", "left")], holes)
        for corner in damaged:
            for _ in range(int(rng.integers(0, 3))):
                _add_motif(rng, base & regions[corner], holes)
        if anchor != 0 and rng.random() < .5:
            _add_motif(rng, base & fixed, holes, _MOTIFS[0])
        extras = np.zeros_like(base)
        interior = np.ones_like(base)
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        available = interior & ~base
        count = int(rng.integers(1 if anchor in (0, 1) else 0, 7))
        if anchor == 0:
            intact = next(corner for corner in regions if corner not in damaged)
            candidates = np.argwhere(available & regions[intact] & ~_dilate(holes))
            if not len(candidates):
                continue
            extras[tuple(candidates[int(rng.integers(len(candidates)))])] = True
        candidates = np.argwhere(available & ~extras)
        if count > int(extras.sum()):
            indices = rng.choice(len(candidates), count-int(extras.sum()), replace=False)
            pts = candidates[indices]
            extras[pts[:, 0], pts[:, 1]] = True
        pair = _paint_known(base, holes, extras, roles, markers)
        observed = pair["input"] == roles[1]
        try:
            recovered = _recover(observed)
        except ValueError:
            continue
        # These constraints make the geometric inference explicit and exact.
        if not np.array_equal(recovered, base):
            continue
        if (_supported_cuts(observed) != _supported_cuts(base)
                or _supported_cuts(observed.T) != _supported_cuts(base.T)):
            continue
        if not all(np.any(base & region) for region in regions.values()):
            continue
        if anchor == 0:
            info = _analyze(pair["input"])
            if np.array_equal(_transform(info, nearest=True)[0], pair["output"]):
                continue
            if np.array_equal(_transform(info, scaffold_endpoint=True)[0], pair["output"]):
                continue
        return {k: np.rot90(a, turn).copy() for k, a in pair.items()}
    raise RuntimeError("could not construct a readable fault-routing scene in 100 attempts")


def _counts(num_train, num_test):
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
           for n in (num_train, num_test)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 training, 1..4 test, and more training than test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    varying_palette = bool(rng.integers(2))
    regime = int(rng.integers(3))  # scale, held-out rotation, new compositions
    direction = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            anchor = index if split == "train" and index < 2 else 2
            expanded = split == "test" and regime == 0
            double = expanded or anchor == 1 or (anchor == 2 and bool(rng.integers(2)))
            if regime == 1:
                turn = direction if split == "train" else (direction+int(rng.choice((1, 3)))) % 4
            else:
                turn = int(rng.integers(4))
            large = split == "test" and regime == 0
            if regime == 2 and split == "train" and anchor != 0:
                style = str(rng.choice(("rectangle", "arms")))
            elif anchor == 1:
                style = str(rng.choice(("rectangle", "arms", "growth")))
            elif split == "test" and regime == 2:
                style = str(rng.choice(("cavities", "separate")))
            else:
                style = str(rng.choice(("rectangle", "arms", "cavities", "separate", "growth")))
            if varying_palette:
                roles = tuple(map(int, rng.choice(10, 3, replace=False)))
            for _ in range(20):
                pair = _make_pair(rng, roles, anchor, double, turn, style, large, expanded)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("exhausted distinct scenes")
    validate_generated_puzzle(puzzle)
    return puzzle


def _without_a(info):
    field, solid, marker, markers, base, holes, extras = info
    out = np.where(base, solid, field)
    out[holes] = field
    out[_dilate(holes) & ~holes] = marker
    for position in markers.values():
        out[position] = marker
    return out


def _without_b(info, inp):
    field, solid, marker, markers, base, holes, extras = info
    # A-like export of the visible solid, ignoring the missing/extra distinction.
    raw = inp == solid
    fixed, regions = _regions(inp.shape, markers)
    out = np.full_like(inp, field)
    out[fixed & raw] = solid
    for corner, region in regions.items():
        pts = np.argwhere(raw & region)
        if len(pts) and _selected(corner, markers):
            pts += _shift(pts, corner, inp.shape)
            out[pts[:, 0], pts[:, 1]] = solid
    for side, point in markers.items():
        ray = _ray(inp.shape, side, point, raw)
        out[ray & (out == field)] = marker
    return out


def _features(info, out):
    field, solid, marker, markers, base, holes, extras = info
    fixed, regions = _regions(base.shape, markers)
    groups = []
    for corner, region in regions.items():
        cells, defects = base & region, holes & region
        selected = _selected(corner, markers)
        full_pts = np.argwhere(cells)
        visible_pts = np.argwhere(cells & ~holes)
        box_diff = bool(len(full_pts) and (not len(visible_pts)
                        or not np.array_equal(full_pts.min(axis=0), visible_pts.min(axis=0))
                        or not np.array_equal(full_pts.max(axis=0), visible_pts.max(axis=0))))
        groups.append({"corner": corner, "cells": int(cells.sum()),
                       "holes": int(defects.sum()), "selected": selected,
                       "components": len(_components(cells)),
                       "visible_box_diff": box_diff,
                       "decoy": bool((extras & region).any())})
    _, final_base, final_holes, guides, halo = _transform(info)
    edge = np.zeros_like(base)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return {"shape": base.shape, "markers": tuple(sorted(markers)),
            "roles": (field, solid, marker), "holes": int(holes.sum()),
            "extras": int(extras.sum()), "components": len(_components(base)),
            "groups": groups, "final_holes": int(final_holes.sum()),
            "clipped": bool((edge & final_holes).any()),
            "nearest_diff": not np.array_equal(_transform(info, nearest=True)[0], out),
            "scaffold_endpoint_diff": not np.array_equal(_transform(info, scaffold_endpoint=True)[0], out),
            "all_move_diff": not np.array_equal(_transform(info, all_move=True)[0], out)}


def validate_generated_puzzle(puzzle) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train and test")
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            field, solid, marker, markers, base, holes, extras = info
            observed = inp == solid
            assert _supported_cuts(observed) == _supported_cuts(base)
            assert _supported_cuts(observed.T) == _supported_cuts(base.T)
            predicted, final_base, final_holes, guides, halo = _transform(info)
            assert np.array_equal(predicted, out)
            assert np.all(out[final_holes] == field), "all surviving hole centers stay field"
            assert np.all(out[halo] == marker), "exact merged Moore halo"
            assert np.all(out[final_base & ~final_holes & ~halo] == solid)
            assert np.all(out[~final_base & ~guides & ~halo] == field)
            assert all(out[p] == marker for p in markers.values())
            assert not np.array_equal(_without_a(info), out), "A must change this answer"
            assert not np.array_equal(_without_b(info, inp), out), "B must change this answer"
            f = _features(info, out)
            assert all(g["cells"] > 0 for g in f["groups"])
            assert any(g["selected"] and g["holes"] for g in f["groups"])
            # Conservation is about scaffold sites, including missing ones,
            # rather than visible solid counts obscured by the halos.
            discarded = sum(g["cells"] for g in f["groups"] if not g["selected"])
            deleted_holes = sum(g["holes"] for g in f["groups"] if not g["selected"])
            assert int(final_base.sum()) == int(base.sum())-discarded
            assert int(final_holes.sum()) == int(holes.sum())-deleted_holes
            features[split].append(f)
    train = features["train"]
    assert {len(f["markers"]) for f in features["test"]} <= {len(f["markers"]) for f in train}
    assert any(f["nearest_diff"] for f in train), "teach farthest axial fault"
    assert any(f["scaffold_endpoint_diff"] for f in train), "teach faults rather than solid endpoints"
    assert any(f["all_move_diff"] for f in train), "teach intact-region retention"
    assert any(g["selected"] and not g["holes"] and g["decoy"]
               for f in train for g in f["groups"]), "a speck must not activate a cut"
    assert any(g["selected"] and g["holes"] and g["components"] > 1
               for f in train for g in f["groups"]), "teach collective fragment movement"
    assert any(g["selected"] and g["holes"] and g["visible_box_diff"]
               for f in train for g in f["groups"]), "teach boxes that include missing sites"
    assert any(not g["selected"] and not g["holes"]
               for f in train for g in f["groups"]), "teach unconditional opposite-quadrant deletion"
    assert any(f["clipped"] for f in train), "teach halo clipping after movement"


def self_test() -> dict:
    """Bounded 42-seed audit plus meaningful count/symmetry/mutation checks."""
    signatures, shapes, structures, palettes, directions = set(), set(), set(), set(), set()
    canonical_scaffolds, actions, cavity_counts = set(), set(), set()
    holes, extras, components = set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                          for k in ("input", "output"))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze(inp)
                f = _features(info, out)
                pairs += 1
                shapes.add(inp.shape)
                palettes.add(f["roles"])
                directions.add(f["markers"])
                holes.add(f["holes"])
                extras.add(f["extras"])
                components.add(f["components"])
                base = info[4]
                pts = np.argwhere(base)
                low, high = pts.min(axis=0), pts.max(axis=0)+1
                cropped = base[low[0]:high[0], low[1]:high[1]]
                canonical_scaffolds.add(min(_key(np.rot90(b, k))
                                             for b in (cropped, cropped.T) for k in range(4)))
                cavities = 0
                for component in _components(~base):
                    if all(0 < r < base.shape[0]-1 and 0 < c < base.shape[1]-1
                           for r, c in component):
                        cavities += 1
                cavity_counts.add(cavities)
                actions.add(tuple("discard" if not g["selected"] else
                                  "export" if g["holes"] else "stay"
                                  for g in f["groups"]))
                # Normalize palette to semantic roles; structural diversity
                # therefore cannot be satisfied by color swaps alone.
                normalized = np.zeros_like(inp, dtype=np.uint8)
                normalized[inp == info[1]] = 1
                normalized[inp == info[2]] = 2
                structures.add(_key(normalized))
                for k in range(4):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                assert np.array_equal(solve(inp.T), out.T)
        if seed < 3:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], repeat[s]) for k in p)
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(structures) >= 180
    assert len(palettes) >= 30 and len(directions) == 8
    assert len(canonical_scaffolds) >= 100 and len(actions) >= 5
    assert max(cavity_counts) >= 1
    assert all({roles[k] for roles in palettes} == set(range(10)) for k in range(3))
    assert len(holes) >= 8 and len(extras) >= 6 and max(components) >= 3
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            generate_puzzle(42, ntrain, ntest)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Deliberately change one halo cell; validation must reject the label.
    broken = generate_puzzle(17, 2, 1)
    p = broken["train"][0]
    info = _analyze(p["input"])
    halo = _transform(info)[4]
    point = tuple(np.argwhere(halo)[0])
    p["output"][point] = info[0]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad halo label accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "normalized_structures": len(structures),
            "role_triples": len(palettes), "marker_configurations": len(directions),
            "canonical_scaffolds": len(canonical_scaffolds), "action_patterns": len(actions),
            "cavity_counts": sorted(cavity_counts),
            "hole_counts": sorted(holes), "extra_counts": sorted(extras),
            "scaffold_component_counts": sorted(components)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
