"""Anchored lattice stamps followed by the union of clipped empty windows.

Only NumPy and the standard library are used. No parent generator is imported.
The highlight is the family constant 9; input roles use distinct colors 0..8.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
import numpy as np


HOST_SEEDS = (0, 347411101, 373017318, 725247973, 972454489, 176871425,
              666849792, 41285714, 509555030, 298384253, 920665179, 780473287,
              858501529, 889306866, 581657552, 389416836, 601969115, 671767718,
              521069440, 977887354, 832092722, 775074617, 782265935, 614258910,
              578220041, 920971128, 56211933, 135370950, 535837795, 685801271,
              388815127, 221070711, 729131127, 216480160, 21958696, 406238133,
              879888743, 491533124, 968686996, 427332949, 663773506, 751532777)
HIGHLIGHT = 9


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _components(mask):
    """Four-connected components in deterministic row-major order."""
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, group = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    stack.append((a, b))
        result.append(group)
    return result


def _decode(inp):
    """Read the three roles, exact glyph, pitch-two map, and anchor from pixels."""
    values, counts = np.unique(inp, return_counts=True)
    if len(values) != 3 or 9 in values or counts.max() <= inp.size // 2:
        raise ValueError("input needs three non-highlight roles and majority background")
    bg = int(values[np.argmax(counts)])
    foreground = [int(v) for v in values if v != bg]
    groups = {v: _components(inp == v) for v in foreground}
    inks = [v for v in foreground if len(groups[v]) == 2
            and sorted(map(len, groups[v]))[0] == 1
            and sorted(map(len, groups[v]))[1] >= 3]
    if len(inks) != 1:
        raise ValueError("template and exceptional anchor must be unambiguous")
    ink = inks[0]
    marker = next(v for v in foreground if v != ink)
    if not all(len(g) == 1 for g in groups[marker]):
        raise ValueError("ordinary map sites must be isolated")
    glyph = np.array(next(g for g in groups[ink] if len(g) > 1), dtype=int)
    anchor_pixel = next(g[0] for g in groups[ink] if len(g) == 1)
    tr, tc = map(int, glyph.min(axis=0))
    br, bc = map(int, glyph.max(axis=0))
    mask = inp[tr:br+1, tc:bc+1] == ink
    h, w = mask.shape
    if min(h, w) < 2:
        raise ValueError("template must span both axes")
    points = [anchor_pixel] + [g[0] for g in groups[marker]]
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    if not all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in points):
        raise ValueError("map must lie on the fixed pitch-two lattice")
    if not (br+1 < pr or er+1 < tr or bc+1 < pc or ec+1 < tc):
        raise ValueError("map and glyph bounding boxes need a blank separating row/column")
    layout = tuple(sorted(((r-pr)//2, (c-pc)//2) for r, c in points))
    anchor = ((anchor_pixel[0]-pr)//2, (anchor_pixel[1]-pc)//2)
    mh, mw = (er-pr)//2+1, (ec-pc)//2+1
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    if not (0 <= top <= inp.shape[0]-mh*h and 0 <= left <= inp.shape[1]-mw*w):
        raise ValueError("expanded lattice must fit without clipping")
    return {"bg": bg, "ink": ink, "marker": marker, "mask": mask,
            "template_top": (tr, tc), "layout": layout, "anchor": anchor,
            "map_shape": (mh, mw), "origin": (top, left), "points": tuple(points)}


def _assemble(inp, meta):
    """Fresh canvas: source-map pixels never survive as reference obstacles."""
    bg, ink, marker = (meta[k] for k in ("bg", "ink", "marker"))
    out = np.full(inp.shape, bg, dtype=np.int8)
    mask = meta["mask"]
    h, w = mask.shape
    top, left = meta["origin"]
    for r, c in meta["layout"]:
        block = out[top+r*h:top+(r+1)*h, left+c*w:left+(c+1)*w]
        block[mask] = marker if (r, c) == meta["anchor"] else ink
    return out


def _dilate(mask):
    """Union of the clipped 3x3 neighborhoods centered at true positions."""
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    for r in range(3):
        for c in range(3):
            result |= padded[r:r+h, c:c+w]
    return result


def _empty_windows(foreground, clipped=True):
    centers = ~_dilate(foreground)
    if not clipped:
        centers = centers.copy()
        centers[[0, -1], :] = False
        centers[:, [0, -1]] = False
    union = _dilate(centers)
    return centers, union


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    meta = _decode(inp)
    assembled = _assemble(inp, meta)
    _, union = _empty_windows(assembled != meta["bg"])
    assembled[union] = HIGHLIGHT
    return assembled


def _asymmetric(mask):
    others = (mask[::-1], mask[:, ::-1], mask[::-1, ::-1], mask.T,
              mask.T[::-1], mask.T[:, ::-1], mask.T[::-1, ::-1])
    return all(not np.array_equal(mask, x) for x in others)


def _grow(rng, h, w):
    """Tight connected mask, using a monotone path and randomized frontier growth."""
    a = np.zeros((h, w), dtype=bool)
    r = c = 0
    a[r, c] = True
    steps = [0]*(h-1) + [1]*(w-1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        a[r, c] = True
    target = max(h+w-1, int(round(h*w*float(rng.uniform(.30, 1.0)))))
    while int(a.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(a):
            for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= y < h and 0 <= x < w and not a[y, x]:
                    frontier.add((int(y), int(x)))
        a[_pick(rng, sorted(frontier))] = True
    if rng.integers(2):
        a = a[::-1].copy()
    if rng.integers(2):
        a = a[:, ::-1].copy()
    return a


def _ring(rng, dims, notch=True):
    h, w = dims
    a = np.ones(dims, dtype=bool)
    a[1:-1, 1:-1] = False
    if notch:
        a[_pick(rng, ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))] = False
    return a


def _random_mask(rng, dims):
    mode = int(rng.integers(5))
    if mode == 0:
        return np.ones(dims, dtype=bool)
    if mode == 1:
        return _ring(rng, dims, notch=bool(rng.integers(2)))
    if mode == 2 and min(dims) >= 4:
        # A thick connected frame with an off-center rectangular aperture.
        h, w = dims
        a = np.ones(dims, dtype=bool)
        rh, cw = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        r, c = int(rng.integers(1, h-rh)), int(rng.integers(1, w-cw))
        a[r:r+rh, c:c+cw] = False
        return a
    return _grow(rng, *dims)


def _layout(rng, shape, kind):
    mh, mw = shape
    if kind == "teaching":
        return _grow(rng, mh, mw)
    rows, cols = list(range(mh)), list(range(mw))
    if kind == "gaps":
        # Two teaching gaps: physical coordinates must not be compressed.
        rows.remove(int(rng.integers(1, mh-1)))
        cols.remove(int(rng.integers(1, mw-1)))
    elif kind == "free":
        for values in (rows, cols):
            if len(values) >= 3 and rng.random() < .45:
                values.remove(int(rng.integers(1, len(values)-1)))
    density = float(rng.uniform(.12, .90))
    a = np.zeros(shape, dtype=bool)
    for r in rows:
        for c in cols:
            a[r, c] = rng.random() < density
    # Exact extent with variable occupied count, not a fixed stencil.
    for r in (rows[0], rows[-1]):
        a[r, _pick(rng, cols)] = True
    for c in (cols[0], cols[-1]):
        a[_pick(rng, rows), c] = True
    if a.sum() < 2:
        a[rows[-1], cols[-1]] = True
        a[rows[0], cols[0]] = True
    return a


def _place(rng, mask, layout, anchor, colors, boundary_teaching=False):
    """Fit the joint assembly/map extent, then choose independent outer margins."""
    h, w = mask.shape
    mh, mw = layout.shape
    ah, aw = mh*h, mw*w
    ph, pw = 2*mh-1, 2*mw-1
    tr, tc = anchor[0]*h, anchor[1]*w
    options = []
    for pr in range(0 if boundary_teaching else -ph-1, ah+2):
        for pc in range(-pw-1, aw+2):
            if not (pr+ph < tr or tr+h < pr or pc+pw < tc or tc+w < pc):
                continue
            r0, c0 = min(0, pr), min(0, pc)
            bh, bw = max(ah, pr+ph)-r0, max(aw, pc+pw)-c0
            if bh <= (28 if boundary_teaching else 30) and bw <= 30:
                options.append((pr, pc, r0, c0, bh, bw))
    if not options:
        raise RuntimeError("no bounded placement")
    # Compact and broadly padded canvases both occur.
    if rng.random() < .45:
        min_area = min(p[4]*p[5] for p in options)
        options = [p for p in options if p[4]*p[5] <= min_area*1.30]
    pr, pc, r0, c0, bh, bw = _pick(rng, options)
    gh = int(rng.integers(bh+(2 if boundary_teaching else 0), 31))
    gw = int(rng.integers(bw, 31))
    # Unique majority background even for a compact, nearly solid glyph.
    needed = 2*int(mask.sum()+layout.sum())+1
    while gh*gw < needed:
        if gh < 30 and (gw == 30 or gh <= gw):
            gh += 1
        else:
            gw += 1
    top = 2 if boundary_teaching else int(rng.integers(gh-bh+1))-r0
    left = int(rng.integers(gw-bw+1))-c0
    bg, ink, marker = colors
    inp = np.full((gh, gw), bg, dtype=np.int8)
    mr, mc = np.nonzero(mask)
    inp[top+tr+mr, left+tc+mc] = ink
    for r, c in np.argwhere(layout):
        inp[top+pr+2*r, left+pc+2*c] = ink if (r, c) == anchor else marker
    return inp


def _pair_features(inp, out):
    m = _decode(inp)
    assembled = _assemble(inp, m)
    fg = assembled != m["bg"]
    centers, union = _empty_windows(fg)
    _, full_union = _empty_windows(fg, clipped=False)
    _, direct = _empty_windows(inp != m["bg"])
    h, w = m["mask"].shape
    mh, mw = m["map_shape"]
    # Restrict hole checks to occupied template boxes, excluding the exterior.
    tiled_holes = np.zeros_like(fg)
    top, left = m["origin"]
    for r, c in m["layout"]:
        tiled_holes[top+r*h:top+(r+1)*h, left+c*w:left+(c+1)*w] = ~m["mask"]
    isolated_small = any(len(g) < 5 and any(tiled_holes[q] for q in g)
                         for g in _components(centers))
    return m, {"template_shape": (h, w), "map_shape": (mh, mw),
               "occupied": len(m["layout"]), "anchor": m["anchor"],
               "palette": (m["bg"], m["ink"], m["marker"]),
               "highlight_cells": int(union.sum()),
               "union_extra": int((union & ~centers).sum()),
               "clipping_difference": int((union != full_union).sum()),
               "assembly_matters": int((union != direct).sum()),
               "unmarked_holes": int((tiled_holes & ~union).sum()),
               "marked_holes": int((tiled_holes & union).sum()),
               "small_candidate_component": isolated_small,
               "empty_rows": mh-len({r for r, c in m["layout"]}),
               "empty_cols": mw-len({c for r, c in m["layout"]})}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(9, 3, replace=False)))
    consistent_palette = rng.random() < .25
    transpose_shift = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_anchor = first_dims = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(48):
                if split == "train" and index == 0:
                    dims = _pick(rng, ((3, 4), (4, 3), (3, 5), (5, 3)))
                    mask = _ring(rng, dims)
                    map_shape = _pick(rng, ((2, 3), (3, 2), (3, 3)))
                    layout = _layout(rng, map_shape, "teaching")
                elif split == "train" and index == 1:
                    choices = [(5, 5), (5, 6), (6, 5), (5, 7), (7, 5), (6, 6)]
                    dims = _pick(rng, [d for d in choices
                                      if all(x != y for x, y in zip(d, first_dims))])
                    mask = _ring(rng, dims)
                    map_shape = _pick(rng, ((3, 3), (3, 4), (4, 3), (4, 4)))
                    layout = _layout(rng, map_shape, "gaps")
                else:
                    if split == "test":
                        # New longer extent, recombining already demonstrated gaps and pitches.
                        choices = [(h, w, a, b) for h in range(2, 6) for w in range(2, 6)
                                   for a in range(2, 7) for b in range(2, 7)
                                   if 5 <= max(a, b) <= 6 and min(a, b) <= 4
                                   and a*h <= 28 and b*w <= 28
                                   and ((a > b) == transpose_shift)]
                    else:
                        choices = [(h, w, a, b) for h in range(2, 8) for w in range(2, 8)
                                   for a in range(1, 5) for b in range(1, 5)
                                   if a*b >= 2 and a*h <= 28 and b*w <= 28]
                    h, w, a, b = _pick(rng, choices)
                    dims, map_shape = (h, w), (a, b)
                    mask = _random_mask(rng, dims)
                    layout = _layout(rng, map_shape, "free")
                points = [tuple(map(int, p)) for p in np.argwhere(layout)]
                anchors = points
                if split == "train" and index == 0:
                    middle = ((map_shape[0]-1)//2, (map_shape[1]-1)//2)
                    anchors = [p for p in points if p != middle]
                elif split == "train" and index == 1:
                    anchors = [p for p in points if p[0] != first_anchor[0]
                               and p[1] != first_anchor[1]]
                if not anchors:
                    continue
                anchor = _pick(rng, anchors)
                colors = palette if consistent_palette else tuple(map(int, rng.choice(9, 3, replace=False)))
                inp = _place(rng, mask, layout, anchor, colors,
                             boundary_teaching=split == "train" and index == 0)
                out = solve(inp)
                if _key(inp) in seen:
                    continue
                _, f = _pair_features(inp, out)
                if not (f["highlight_cells"] and f["union_extra"] and f["assembly_matters"]):
                    continue
                if split == "train" and index == 0:
                    if not (f["clipping_difference"] and f["unmarked_holes"] and _asymmetric(mask)):
                        continue
                if split == "train" and index == 1:
                    if not (f["marked_holes"] and f["small_candidate_component"]):
                        continue
                seen.add(_key(inp))
                puzzle[split].append({"input": inp, "output": out})
                if split == "train" and index == 0:
                    first_anchor, first_dims = anchor, dims
                break
            else:
                raise RuntimeError("bounded generation retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            m, f = _pair_features(inp, out)
            assembled = _assemble(inp, m)
            fg = assembled != m["bg"]
            centers, union = _empty_windows(fg)
            expected = assembled.copy()
            expected[union] = HIGHLIGHT
            assert np.array_equal(expected, out)
            assert not np.any(union & fg)
            assert np.array_equal(out[fg], assembled[fg])
            population = int(m["mask"].sum())
            assert np.count_nonzero(out == m["marker"]) == population
            assert np.count_nonzero(out == m["ink"]) == population*(len(m["layout"])-1)
            assert f["highlight_cells"] and f["union_extra"] and f["assembly_matters"]
            # Omitting B changes the answer; applying B before A changes the mark mask.
            assert not np.array_equal(out, assembled)
            tr, tc = m["template_top"]
            h, w = m["mask"].shape
            assert np.all(out[tr:tr+h, tc:tc+w][m["mask"]] == m["marker"])
            if split == "train":
                train.append((m, f))
    assert len({m["anchor"][0] for m, f in train}) >= 2
    assert len({m["anchor"][1] for m, f in train}) >= 2
    assert len({m["mask"].shape[0] for m, f in train}) >= 2
    assert len({m["mask"].shape[1] for m, f in train}) >= 2
    assert any(_asymmetric(m["mask"]) for m, f in train)
    assert any(f["unmarked_holes"] for m, f in train)
    assert any(f["marked_holes"] and f["small_candidate_component"] for m, f in train)
    assert any(f["clipping_difference"] for m, f in train)
    assert any(f["empty_rows"] and f["empty_cols"] for m, f in train)
    assert any(all(any((r+(axis == 0), c+(axis == 1)) in m["layout"]
                       for r, c in m["layout"]) for axis in (0, 1)) for m, f in train)


def _scalar_union(fg):
    """Literal window enumeration for bounded internal regression checks."""
    h, w = fg.shape
    result = np.zeros_like(fg)
    for r in range(h):
        for c in range(w):
            box = np.s_[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)]
            if not fg[box].any():
                result[box] = True
    return result


def self_test() -> dict:
    """Stream 42 seeds; retain only small feature sets, hashes, and one puzzle."""
    signatures, shapes, masks, layouts, anchors, palettes, counts = (set() for _ in range(7))
    totals = {"pairs": 0, "edge_diagnostics": 0, "marked_hole_pairs": 0,
              "unmarked_hole_pairs": 0, "small_candidate_pairs": 0,
              "both_gap_pairs": 0, "union_extra_cells": 0, "assembly_difference_cells": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                m, f = _pair_features(inp, out)
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(inp.shape)
                masks.add((m["mask"].shape, m["mask"].tobytes()))
                layouts.add((m["map_shape"], m["layout"]))
                anchors.add(m["anchor"])
                palettes.add(f["palette"])
                counts.add(f["occupied"])
                totals["pairs"] += 1
                totals["edge_diagnostics"] += bool(f["clipping_difference"])
                totals["marked_hole_pairs"] += bool(f["marked_holes"])
                totals["unmarked_hole_pairs"] += bool(f["unmarked_holes"])
                totals["small_candidate_pairs"] += bool(f["small_candidate_component"])
                totals["both_gap_pairs"] += bool(f["empty_rows"] and f["empty_cols"])
                totals["union_extra_cells"] += f["union_extra"]
                totals["assembly_difference_cells"] += f["assembly_matters"]
                assembled = _assemble(inp, m)
                fg = assembled != m["bg"]
                assert np.array_equal(out == 9, _scalar_union(fg))
                assert np.array_equal(solve(inp.tolist()), out)
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 45
    assert len(masks) >= 50 and len(layouts) >= 90 and len(anchors) >= 15
    assert len(palettes) >= 60 and len(counts) >= 12
    assert totals["edge_diagnostics"] >= 42
    assert totals["marked_hole_pairs"] >= 42 and totals["unmarked_hole_pairs"] >= 42
    assert totals["small_candidate_pairs"] >= 42 and totals["both_gap_pairs"] >= 42
    assert max(counts) >= 15
    p, q = generate_puzzle(1234, 2, 1), generate_puzzle(1234, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(10000+10*nt+nq, nt, nq))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (None, 1), (4, None), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Diagonals are obstacles; an enclosed 3x3 opening has one center but nine union cells.
    fg = np.ones((7, 7), dtype=bool)
    fg[2:5, 2:5] = False
    centers, union = _empty_windows(fg)
    assert centers.sum() == 1 and union.sum() == 9
    fg[2, 2] = True
    assert not _empty_windows(fg)[1].any()
    # A two-cell-deep edge band works only because windows are clipped.
    fg = np.ones((6, 7), dtype=bool)
    fg[:2, :] = False
    assert _empty_windows(fg)[1].sum() == 14
    assert not _empty_windows(fg, clipped=False)[1].any()
    # Reject a one-cell corruption rather than merely validating the schema.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": len(HOST_SEEDS), "distinct_puzzles": len(signatures),
            "canvas_shapes": len(shapes), "template_masks": len(masks),
            "map_layouts": len(layouts), "anchors": len(anchors),
            "role_palettes": len(palettes), "occupied_counts": sorted(counts), **totals}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
