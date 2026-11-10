"""Anchored, pitch-two placement maps expanded by a tight template mask."""

import numpy as np


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grow(rng, height, width, density, through=None):
    """A random spanning path and frontier growth; no exponential mask bank."""
    mask = np.zeros((height, width), dtype=bool)
    flip = bool(rng.integers(2))
    middle = through if through is not None else (
        int(rng.integers(height)), int(rng.integers(width)))
    if flip:
        middle = middle[0], width - 1 - middle[1]
    r, c = 0, 0
    mask[r, c] = True
    for target_r, target_c in (middle, (height - 1, width - 1)):
        steps = [0] * (target_r - r) + [1] * (target_c - c)
        rng.shuffle(steps)
        for axis in steps:
            r += axis == 0
            c += axis == 1
            mask[r, c] = True
    goal = max(int(mask.sum()), int(round(height * width * density)))
    while int(mask.sum()) < goal:
        frontier = set()
        for r, c in np.argwhere(mask):
            for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= a < height and 0 <= b < width and not mask[a, b]:
                    frontier.add((int(a), int(b)))
        mask[_pick(rng, sorted(frontier))] = True
    return np.fliplr(mask).copy() if flip else mask


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in (
        np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2),
        mask.T, np.flipud(mask).T, np.fliplr(mask).T, np.rot90(mask, 2).T))


def _template(rng, dims, informative=False):
    h, w = dims
    density = _pick(rng, (0.0, 0.45, 0.6, 0.75, 0.9, 1.0))
    mask = _grow(rng, h, w, density)
    if informative and not _asymmetric(mask):
        # Rectangular, corner-notched masks distinguish every isometry.
        mask[:] = True
        mask[_pick(rng, ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))] = False
    return mask


def _orientation(h, w):
    return int(np.sign(w - h))


def _map_dimensions(rng, dims, orientation, two_axes, complexity, gap_example=False):
    h, w = dims
    limit = 4 if complexity < 0.25 else 7
    if gap_example:
        limit = max(5, limit)
    line = not two_axes and rng.random() < 0.22
    choices = [(a, b) for a in range(1, limit + 1) for b in range(1, limit + 1)
               if 2 <= a*b and a*h <= 28 and b*w <= 28
               and ((min(a, b) == 1) if line else min(a, b) >= 2)
               and (not gap_example or min(a, b) >= 4)
               and (orientation is None or _orientation(a, b) == orientation)]
    if not choices:  # Square regimes cannot contain a nontrivial line.
        choices = [(a, a) for a in range(2, limit + 1) if a*max(h, w) <= 28]
    return _pick(rng, choices)


def _anchor(rng, rows, cols, previous=None, occupied=None, noncentral=False):
    cells = [(r, c) for r in rows for c in cols
             if (previous is None or (r != previous[0] and c != previous[1]))
             and (not noncentral or (r, c) != (rows[-1]//2, cols[-1]//2))
             and (occupied is None or occupied[r, c])]
    if not cells:
        return None
    category = int(rng.integers(4))
    corners = {(rows[0], cols[0]), (rows[0], cols[-1]),
               (rows[-1], cols[0]), (rows[-1], cols[-1])}
    subset = [p for p in cells if (
        p in corners if category == 0 else
        (p[0] in (rows[0], rows[-1]) or p[1] in (cols[0], cols[-1])) if category == 1 else
        (p[0] not in (rows[0], rows[-1]) and p[1] not in (cols[0], cols[-1])) if category == 2
        else True)]
    return _pick(rng, subset or cells)


def _layout(rng, dims, previous=None, connected=False, gap_example=False, noncentral=False):
    h, w = dims
    mode = 2 if gap_example else 0 if connected else int(rng.choice(3, p=(0.50, 0.25, 0.25)))
    rows, cols = list(range(h)), list(range(w))
    if mode == 2:
        # Keep an adjacent step while leaving a whole internal lattice band empty.
        axes = [axis for axis, size in enumerate(dims) if size >= 4]
        if axes:
            for axis in axes if gap_example or rng.random() < 0.35 else [_pick(rng, axes)]:
                values = rows if axis == 0 else cols
                values.remove(int(rng.integers(2, len(values)-1)))
    anchor = _anchor(rng, rows, cols, previous, noncentral=noncentral)
    assert anchor is not None
    if mode == 0:
        return _grow(rng, h, w, float(rng.uniform(0.0, 1.0)), anchor), anchor
    mask = np.zeros(dims, dtype=bool)
    mask[anchor] = True
    # Tight extent and explicit horizontal/vertical pitch witnesses.
    for r in (rows[0], rows[-1]):
        mask[r, _pick(rng, cols)] = True
    for c in (cols[0], cols[-1]):
        mask[_pick(rng, rows), c] = True
    for axis, values in enumerate((rows, cols)):
        adjacent = [v for v in values if v+1 in values]
        if adjacent:
            start = _pick(rng, adjacent)
            other = _pick(rng, cols if axis == 0 else rows)
            for v in (start, start+1):
                mask[(v, other) if axis == 0 else (other, v)] = True
    available = [(r, c) for r in rows for c in cols if not mask[r, c]]
    rng.shuffle(available)
    target = int(round(len(rows)*len(cols)*float(rng.uniform(0.15, 0.95))))
    for point in available[:max(0, target-int(mask.sum()))]:
        mask[point] = True
    return mask, anchor


def _place(rng, mask, layout, anchor, colors):
    """Place on an unbounded plane, then fit the union and add margins."""
    h, w = mask.shape
    mh, mw = layout.shape
    ah, aw = h*mh, w*mw
    ph, pw = 2*mh-1, 2*mw-1
    tr, tc = anchor[0]*h, anchor[1]*w
    rr, cc = np.meshgrid(np.arange(-ph-2, ah+3), np.arange(-pw-2, aw+3), indexing="ij")
    minr, minc = np.minimum(0, rr), np.minimum(0, cc)
    maxr, maxc = np.maximum(ah, rr+ph), np.maximum(aw, cc+pw)
    valid = ((rr+ph < tr) | (tr+h < rr) | (cc+pw < tc) | (tc+w < cc))
    valid &= (maxr-minr <= 30) & (maxc-minc <= 30)
    positions = np.argwhere(valid)
    assert len(positions)
    if rng.random() < 0.35:
        areas = (maxr-minr)[valid]*(maxc-minc)[valid]
        positions = positions[areas <= np.quantile(areas, 0.2)]
    i, j = _pick(rng, positions)
    pr, pc = int(rr[i, j]), int(cc[i, j])
    base_h, base_w = int(maxr[i, j]-minr[i, j]), int(maxc[i, j]-minc[i, j])
    padding = int(rng.integers(5))
    gh = base_h if padding == 0 else 30 if padding == 1 else int(rng.integers(base_h, 31))
    gw = base_w if padding == 0 else 30 if padding == 1 else int(rng.integers(base_w, 31))
    # Two large solid copies can otherwise leave less background than ink.
    # Keep the background identifiable even in the most compact simple cases.
    foreground = int(mask.sum()+layout.sum())
    minimum_area = foreground+max(int(mask.sum())+1, int(layout.sum())-1)+1
    if gh*gw < minimum_area:
        fits = [(a, b) for a in range(gh, 31) for b in range(gw, 31)
                if a*b >= minimum_area]
        smallest = min(a*b for a, b in fits)
        gh, gw = _pick(rng, [(a, b) for a, b in fits if a*b == smallest])
    top = int(rng.integers(gh-base_h+1))-int(minr[i, j])
    left = int(rng.integers(gw-base_w+1))-int(minc[i, j])
    bg, ink, marker = colors
    inp = np.full((gh, gw), bg, dtype=np.int8)
    mr, mc = np.nonzero(mask)
    inp[top+tr+mr, left+tc+mc] = ink
    out = np.full_like(inp, bg)
    for r, c in np.argwhere(layout):
        is_anchor = (r, c) == anchor
        inp[top+pr+2*r, left+pc+2*c] = ink if is_anchor else marker
        out[top+r*h+mr, left+c*w+mc] = marker if is_anchor else ink
    return {"input": inp, "output": out}


def _decode_input(inp):
    """Recover all semantic information from pixels, without a seed or label."""
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    bg = int(values[np.argmax(counts)])
    assert int(counts.max()) > int(np.sort(counts)[-2])
    fg = [int(v) for v in values if v != bg]
    components = {v: _components(inp, v) for v in fg}
    candidates = [v for v in fg if len(components[v]) == 2
                  and sorted(map(len, components[v]))[0] == 1
                  and sorted(map(len, components[v]))[1] >= 3]
    assert len(candidates) == 1
    ink = candidates[0]
    marker = next(v for v in fg if v != ink)
    glyph = next(p for p in components[ink] if len(p) > 1)
    anchor_cell = next(p[0] for p in components[ink] if len(p) == 1)
    assert all(len(p) == 1 for p in components[marker])
    coords = np.array(glyph, dtype=int)
    tr, tc = map(int, coords.min(axis=0))
    br, bc = map(int, coords.max(axis=0))
    h, w = br-tr+1, bc-tc+1
    assert h >= 2 and w >= 2
    mask = inp[tr:br+1, tc:bc+1] == ink
    points = [anchor_cell] + [p[0] for p in components[marker]]
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    assert all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in points)
    assert _rectangles_have_gap((tr, tc, br, bc), (pr, pc, er, ec))
    mh, mw = (er-pr)//2+1, (ec-pc)//2+1
    coarse = tuple(sorted(((int(r)-pr)//2, (int(c)-pc)//2) for r, c in points))
    anchor = (int(anchor_cell[0])-pr)//2, (int(anchor_cell[1])-pc)//2
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    assert 0 <= top and top+mh*h <= inp.shape[0]
    assert 0 <= left and left+mw*w <= inp.shape[1]
    witnesses = tuple(any((r+(axis == 0), c+(axis == 1)) in coarse
                          for r, c in coarse) for axis in (0, 1))
    return {"roles": (bg, ink, marker), "palette": tuple(sorted(map(int, values))),
            "template_h": h, "template_w": w, "mask": tuple(map(int, mask.flat)),
            "map_h": mh, "map_w": mw, "layout": coarse, "anchor": anchor,
            "occupied": len(coarse), "connected": _coarse_connected(coarse),
            "empty_rows": mh-len(set(r for r, _ in coarse)),
            "empty_cols": mw-len(set(c for _, c in coarse)),
            "witnesses": witnesses, "asymmetric": _asymmetric(mask),
            "template_top": tr, "template_left": tc,
            "top": top, "left": left, "map_points": tuple(points)}


def _prediction(inp, meta=None):
    info = _decode_input(inp) if meta is None else meta
    bg, ink, marker = info["roles"]
    mask = np.array(info["mask"], dtype=bool).reshape(info["template_h"], info["template_w"])
    layout = np.zeros((info["map_h"], info["map_w"]), dtype=bool)
    for point in info["layout"]:
        layout[point] = True
    # Kronecker expansion is separate from the generator's point-copy loop.
    stamp = np.full((layout.shape[0]*mask.shape[0], layout.shape[1]*mask.shape[1]), bg, dtype=np.int8)
    stamp[np.kron(layout, mask)] = ink
    exceptional = np.zeros_like(layout)
    exceptional[info["anchor"]] = True
    stamp[np.kron(exceptional, mask)] = marker
    out = np.full_like(inp, bg)
    r, c = info["top"], info["left"]
    out[r:r+stamp.shape[0], c:c+stamp.shape[1]] = stamp
    return out


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    meta = _decode_input(inp)
    assert np.array_equal(_prediction(inp, meta), out)
    assert not np.array_equal(inp, out)
    assert int(np.count_nonzero(out != meta["roles"][0])) == meta["occupied"]*sum(meta["mask"])
    return meta


def _training_coverage(train):
    # Demonstration quality restrictions, separate from semantic reconstruction.
    assert len({m["anchor"][0] for m in train}) >= 2
    assert len({m["anchor"][1] for m in train}) >= 2
    assert len({m["template_h"] for m in train}) >= 2
    assert len({m["template_w"] for m in train}) >= 2
    assert any(m["asymmetric"] for m in train)
    assert any(m["witnesses"] == (True, True) for m in train)
    assert any(m["template_h"] != m["template_w"] for m in train)
    assert any(m["anchor"] != ((m["map_h"]-1)//2, (m["map_w"]-1)//2) for m in train)


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    info = {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert (key, _grid_key(pair["output"])) not in _SOURCE_KEYS
            meta = _analyze_pair(pair["input"], pair["output"])
            assert 2 <= meta["occupied"]
            info[split].append(meta)
    _training_coverage(info["train"])
    for axis in ("empty_rows", "empty_cols"):
        if any(m[axis] for m in info["test"]):
            assert any(m[axis] for m in info["train"])
    if any(not m["connected"] for m in info["test"]):
        assert any(not m["connected"] for m in info["train"])
    # Pitch two and inversion are fixed semantics; local palettes, orientations,
    # and connectivity are visible properties, not hidden conventions.
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed generated puzzle") from exc


def _novel_pair(pair, used):
    key = (_grid_key(pair["input"]), _grid_key(pair["output"]))
    if key[0] not in used and key not in _SOURCE_KEYS:
        return pair
    # Finite fallback: 720 role assignments outnumber all 14 pairs and 4 sources.
    colors = tuple(map(int, np.unique(pair["input"])))
    for a in range(10):
        for b in range(10):
            for c in range(10):
                if len({a, b, c}) != 3:
                    continue
                remap = np.arange(10, dtype=np.int8)
                remap[list(colors)] = (a, b, c)
                candidate = {role: remap[grid] for role, grid in pair.items()}
                key = (_grid_key(candidate["input"]), _grid_key(candidate["output"]))
                if key[0] not in used and key not in _SOURCE_KEYS:
                    return candidate
    raise AssertionError("finite uniqueness fallback exhausted")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(rng, num_train, num_test)
    # Puzzle-level choices control presentation and difficulty, not the rule.
    shift = int(rng.integers(4))
    matched = int(rng.integers(-1, 2))
    orientations = ((-1, 1), (1, -1), (matched, matched), (None, None))[shift]
    complexity = float(rng.random())
    consistent_palette = rng.random() < 0.20
    base_colors = tuple(map(int, rng.choice(10, 3, replace=False)))
    extensions = rng.random() < 0.60
    extension_index = (2 if nt >= 3 and rng.random() < 0.5 else 1) if extensions else None
    previous_anchor = previous_dims = previous_layout = first_palette = None
    puzzle = {"train": [], "test": []}
    used = set()
    for split, regime, count, orient in (("train", "source_train", nt, orientations[0]),
                                        ("test", "target_test", nq, orientations[1])):
        for index in range(count):
            teaching = regime == "source_train" and index < 2
            gap_example = regime == "source_train" and index == extension_index
            dimension_options = [(h, w) for h in range(2, 6) for w in range(2, 6)
                                 if (not teaching or index != 0 or h != w)
                                 and (not teaching or index != 1 or
                                      (h != previous_dims[0] and w != previous_dims[1]))]
            dims = _pick(rng, dimension_options)
            mask = _template(rng, dims, informative=teaching and index == 0)
            local_orientation = (0 if regime == "source_train" and shift in (0, 1)
                                 and rng.random() < 0.35 else orient)
            map_dims = _map_dimensions(rng, dims, local_orientation,
                                      teaching or gap_example, complexity, gap_example)
            prev = previous_anchor if teaching and index == 1 else None
            layout = anchor = None
            if teaching and index == 1 and not gap_example and rng.random() < 0.35:
                if all(a*b <= 28 for a, b in zip(dims, previous_layout.shape)):
                    candidate = _anchor(rng, list(range(previous_layout.shape[0])),
                                        list(range(previous_layout.shape[1])), prev, previous_layout)
                    if candidate is not None:
                        layout, anchor = previous_layout.copy(), candidate
            if layout is None:
                layout, anchor = _layout(rng, map_dims, prev,
                                         connected=teaching or not extensions, gap_example=gap_example,
                                         noncentral=teaching and index == 0)
            colors = base_colors if consistent_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
            if teaching and index == 1 and not consistent_palette and set(colors) == set(first_palette):
                missing = _pick(rng, [v for v in range(10) if v not in colors])
                colors = (missing, colors[1], colors[2])
            pair = _novel_pair(_place(rng, mask, layout, anchor, colors), used)
            used.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
            if teaching and index == 0:
                previous_anchor, previous_dims, previous_layout = anchor, dims, layout
                first_palette = colors
    validate_generated_puzzle(puzzle)
    return puzzle


def diversity_metrics(puzzles):
    """Measurements reconstructed from returned pixels, also used by self_test."""
    rows, per_puzzle = [], []
    for puzzle in puzzles:
        groups = _validate(puzzle)
        for split, metas in groups.items():
            for pair, m in zip(puzzle[split], metas):
                row = dict(m)
                row["split"] = split
                row["shape"] = pair["input"].shape
                row["overlap"] = int(sum(pair["output"][p] != m["roles"][0] for p in m["map_points"]))
                row["edges"] = (m["top"] == 0, m["left"] == 0,
                                m["top"]+m["template_h"]*m["map_h"] == pair["input"].shape[0],
                                m["left"]+m["template_w"]*m["map_w"] == pair["input"].shape[1])
                row["edge"] = any(row["edges"])
                ar, ac = m["anchor"]
                on_row_edge, on_col_edge = ar in (0, m["map_h"]-1), ac in (0, m["map_w"]-1)
                row["anchor_kind"] = ("corner" if on_row_edge and on_col_edge else
                                      "edge" if on_row_edge or on_col_edge else "interior")
                row["anchor_degree"] = sum(p in m["layout"] for p in
                                           ((ar-1, ac), (ar+1, ac), (ar, ac-1), (ar, ac+1)))
                mask = np.array(m["mask"]).reshape(m["template_h"], m["template_w"])
                row["mask_density"] = float(mask.mean())
                row["internal_blanks"] = int(np.count_nonzero(mask[1:-1, 1:-1] == 0))
                row["joined_copies"] = len(_components(pair["output"], m["roles"][1])) < m["occupied"]-1
                rows.append(row)
        train, test = groups["train"], groups["test"]
        signs = [{_orientation(m["map_h"], m["map_w"]) for m in group} for group in (train, test)]
        regime = ("forward" if signs[0] <= {-1, 0} and signs[1] == {1} else
                  "reverse" if signs[0] <= {0, 1} and signs[1] == {-1} else
                  "similar" if len(signs[0]) == len(signs[1]) == 1 and signs[0] == signs[1]
                  else "mixed")
        all_meta = train+test
        per_puzzle.append({"regime": regime,
            "train_anchors": len({m["anchor"] for m in train}),
            "train_anchor_rows": len({m["anchor"][0] for m in train}),
            "train_anchor_cols": len({m["anchor"][1] for m in train}),
            "train_template_heights": len({m["template_h"] for m in train}),
            "train_template_widths": len({m["template_w"] for m in train}),
            "template_heights": len({m["template_h"] for m in all_meta}),
            "template_widths": len({m["template_w"] for m in all_meta}),
            "palettes": len({m["palette"] for m in all_meta}),
            "colors": len(set(v for m in all_meta for v in m["palette"])),
            "map_shapes": len({(m["map_h"], m["map_w"]) for m in all_meta}),
            "occupancies": len({m["occupied"] for m in all_meta}),
            "comparable_anchors": any(a["layout"] == b["layout"] and a["anchor"] != b["anchor"]
                                      for i, a in enumerate(train) for b in train[i+1:]),
            "demonstrated_gaps": any(m["empty_rows"] and m["empty_cols"] for m in train)})
    return rows, per_puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzles_equal(puzzles[17], generate_puzzle(17))
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for i, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3),
                                  (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000+i, nt, nq)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert nq is None or len(puzzle["test"]) == nq
        assert _puzzles_equal(puzzle, generate_puzzle(2000+i, nt, nq))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(10, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    # Original solved contents are semantic regressions, not generated puzzles.
    for inp, out in _source_pairs():
        _analyze_pair(inp, out)
    swapped, original_answer = _source_pairs()[1]
    swapped[13, 15], swapped[11, 13] = int(swapped[11, 13]), int(swapped[13, 15])
    swapped_answer = _prediction(swapped)
    _analyze_pair(swapped, swapped_answer)
    assert np.count_nonzero(swapped_answer != original_answer) == 80
    rows, per_puzzle = diversity_metrics(puzzles)
    for split in ("train", "test"):
        group = [r for r in rows if r["split"] == split]
        assert {2, 3, 4, 5} <= {r["template_h"] for r in group}
        assert {2, 3, 4, 5} <= {r["template_w"] for r in group}
        assert len({(r["map_h"], r["map_w"]) for r in group}) >= 12
        assert len({r["anchor"] for r in group}) >= 12
    assert {p["regime"] for p in per_puzzle} == {"forward", "reverse", "similar", "mixed"}
    assert all(p["train_anchors"] >= 2 for p in per_puzzle)
    assert sum(p["colors"] > 3 for p in per_puzzle) >= 25
    assert any(p["colors"] == 3 for p in per_puzzle)
    assert len({r["shape"] for r in rows}) >= 80
    # Exact contents are excluded; source-sized novel grids remain legal.
    source_input, source_output = _source_pairs()[0]
    source_sized = _novel_pair({"input": source_input, "output": source_output}, set())
    assert source_sized["input"].shape == (22, 22)
    assert (_grid_key(source_sized["input"]), _grid_key(source_sized["output"])) not in _SOURCE_KEYS
    _analyze_pair(source_sized["input"], source_sized["output"])
    assert any(r["shape"] == (30, 30) for r in rows)
    assert min(min(r["shape"]) for r in rows) < 12
    assert len({(r["template_h"], r["template_w"], r["mask"]) for r in rows}) >= 70
    assert len({(r["map_h"], r["map_w"], r["layout"]) for r in rows}) >= 100
    assert {2, 3} <= {r["occupied"] for r in rows}
    assert max(r["occupied"] for r in rows) >= 15
    assert any(r["map_h"] == 1 for r in rows) and any(r["map_w"] == 1 for r in rows)
    assert sum(not r["connected"] for r in rows) >= 20
    assert sum(r["empty_rows"] > 0 for r in rows) >= 10
    assert sum(r["empty_cols"] > 0 for r in rows) >= 10
    assert sum(r["overlap"] > 0 for r in rows) >= 15
    assert sum(r["edge"] for r in rows) >= 30
    assert any(all(r["mask"]) for r in rows)
    assert any(r["mask_density"] < 0.5 for r in rows)
    assert sum(r["internal_blanks"] > 0 for r in rows) >= 20
    assert {r["anchor_kind"] for r in rows} == {"corner", "edge", "interior"}
    assert sum(r["anchor_degree"] == 1 for r in rows) >= 20
    assert all(any(r["edges"][axis] for r in rows) for axis in range(4))
    assert sum(r["joined_copies"] for r in rows) >= 20
    assert sum(p["comparable_anchors"] for p in per_puzzle) >= 5
    assert all(p["map_shapes"] >= 2 and p["occupancies"] >= 2 for p in per_puzzle)
    assert all(len({r["occupied"] for r in rows if (r["map_h"], r["map_w"]) == dims}) >= 3
               for dims in ((3, 3), (4, 4), (4, 3), (3, 4)))
    assert all(len({(r["template_h"], r["template_w"])
                    for r in rows if _orientation(r["map_h"], r["map_w"]) == orient}) >= 12
               for orient in (-1, 0, 1))
    assert any(r["template_h"] >= 4 and r["template_w"] >= 4 and
               r["occupied"] >= 9 and not r["connected"] for r in rows)
    # The old coordinate shortcut must fail substantially in both splits.
    assert all(sum(r["anchor"] != (1, r["map_w"]-2) for r in rows if r["split"] == split) > 20
               for split in ("train", "test"))
    pair = puzzles[0]["train"][0]
    meta = _decode_input(pair["input"])
    changed = pair["output"].copy()
    changed[meta["template_top"], meta["template_left"]] = (int(changed[meta["template_top"], meta["template_left"]])+1) % 10
    try:
        _analyze_pair(pair["input"], changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


def _rectangles_have_gap(a, b):
    at, al, ab, ar = a
    bt, bl, bb, br = b

    return (
        ab + 1 < bt
        or bb + 1 < at
        or ar + 1 < bl
        or br + 1 < al
    )


def _resolve_counts(rng, num_train, num_test):
    def checked(name, value, low, high):
        if value is None:
            return None

        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(f"{name} must be an integer or None")

        value = int(value)
        if not low <= value <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}"
            )
        return value

    train_count = checked("num_train", num_train, 2, 10)
    test_count = checked("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        upper = min(4, train_count - 1)
        if upper < 1:
            raise ValueError(
                "no legal test count for the requested train count"
            )
        test_count = int(rng.integers(1, upper + 1))

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _grid_key(grid):
    canonical = grid.astype(np.int16, copy=False)
    return grid.shape, canonical.tobytes()


def _components(grid, color):
    remaining = {
        tuple(x)
        for x in np.argwhere(grid == color)
    }
    result = []

    while remaining:
        start = remaining.pop()
        component = [start]
        stack = [start]

        while stack:
            r, c = stack.pop()
            for q in (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ):
                if q in remaining:
                    remaining.remove(q)
                    component.append(q)
                    stack.append(q)

        result.append(component)

    return result


def _coarse_connected(points):
    points = set(points)
    seen = {next(iter(points))}
    stack = list(seen)

    while stack:
        r, c = stack.pop()
        for q in (
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ):
            if q in points and q not in seen:
                seen.add(q)
                stack.append(q)

    return seen == points


def _puzzles_equal(a, b):
    if set(a) != set(b):
        return False

    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False

        for pair_a, pair_b in zip(
            a[split],
            b[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_signature(puzzle):
    return tuple(
        (
            split,
            _grid_key(pair["input"]),
            _grid_key(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


# Exact source regressions; these are never sampled as templates.
_SOURCE_GRIDS = (
    (
        '4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444244444444/4444444444442224444444/4444444444442424444444/4444844444444444444444/4444444444444444444444/4484248444444444444444/4444444444444444444444/4444844444444444444444/4444444444444444444444/4444844444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444',
        '4444444444444244444444/4444444444442224444444/4444444444442424444444/4444444444244844244444/4444444442228882224444/4444444442428482424444/4444444444444244444444/4444444444442224444444/4444444444442424444444/4444444444444244444444/4444444444442224444444/4444444444442424444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444/4444444444444444444444',
    ),
    (
        '3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333311133333333333333/3333331133333333333333/3333311333333333333333/3333311133333333333333/3333333333333333333333/3333333333333232323333/3333333333333333333333/3333333333333331333333/3333333333333333333333/3333333333333332333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333',
        '3333333333333333333333/3333333333333333333333/3311111111133333333333/3331131131133333333333/3311311311333333333333/3311111111133333333333/3333322233333333333333/3333332233333333333333/3333322333333333333333/3333322233333333333333/3333311133333333333333/3333331133333333333333/3333311333333333333333/3333311133333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333/3333333333333333333333',
    ),
    (
        '7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7774477777777777777777/7774447777777777777777/7777447777777777777777/7777777777777677777777/7777777777777777777777/7777777777767476777777/7777777777777777777777/7777777777767776777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777',
        '7777777777777777777777/7777777777777777777777/7774477777777777777777/7774447777777777777777/7777447777777777777777/4476674477777777777777/4446664447777777777777/7447667447777777777777/4477774477777777777777/4447774447777777777777/7447777447777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777/7777777777777777777777',
    ),
    (
        '8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888886688888888888/8888888866688888888888/8888888886888888888888/8888888866688888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888848888888888888/8888888888888888888888/8848486848888888888888/8888888888888888888888/8848888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888',
        '8888888888888888888888/8888888888886688888888/8888888888866688888888/8888888888886888888888/8888888888866688888888/8886686684486688888888/8866666644466688888888/8886886884886888888888/8866666644466688888888/8886688888888888888888/8866688888888888888888/8886888888888888888888/8866688888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888/8888888888888888888888',
    ),
)


def _source_pairs():
    return [(np.array([[int(v) for v in row] for row in a.split("/")], dtype=np.int8),
             np.array([[int(v) for v in row] for row in b.split("/")], dtype=np.int8))
            for a, b in _SOURCE_GRIDS]


_SOURCE_KEYS = {(_grid_key(a), _grid_key(b)) for a, b in _source_pairs()}
