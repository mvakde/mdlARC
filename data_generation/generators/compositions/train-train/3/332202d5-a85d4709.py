"""Legend-decoded Voronoi bands.

One edge line of the input is a multicoloured legend.  Every line parallel to
the legend (a "field line") is background except that some lines carry a single
marker cell.  A marker's label is the legend colour in the same perpendicular
position (positional code, parent B).  Each field line then takes the label of
its nearest marked line (1-D Voronoi, parent A); exact ties between different
decoded labels become marker-coloured seams, equal-label ties merge, the outer
labels run to the far edge, and marked lines become label lines that keep their
marker cell.  The legend line is copied unchanged.
"""
from __future__ import annotations

import numpy as np

EDGES = ("top", "bottom", "left", "right")


# ---------------------------------------------------------------------------
# Orientation helpers: canonical form has the legend as row 0.
# ---------------------------------------------------------------------------
def _from_canonical(canon: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return canon.copy()
    if edge == "bottom":
        return np.flipud(canon).copy()
    if edge == "left":
        return canon.T.copy()
    if edge == "right":
        return np.fliplr(canon.T).copy()
    raise ValueError(edge)


def _to_canonical(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return grid.copy()
    if edge == "bottom":
        return np.flipud(grid).copy()
    if edge == "left":
        return grid.T.copy()
    if edge == "right":
        return np.fliplr(grid).T.copy()
    raise ValueError(edge)


# ---------------------------------------------------------------------------
# Input parsing (used by solve and by validation)
# ---------------------------------------------------------------------------
def _parse_canonical(canon: np.ndarray):
    """Return (legend, bg, marker, sources) if canon has its legend in row 0."""
    if canon.shape[0] < 3 or canon.shape[1] < 2:
        return None
    legend = canon[0]
    field = canon[1:]
    if len(set(legend.tolist())) < 2:
        return None
    values, counts = np.unique(field, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if bg in set(legend.tolist()):
        return None
    marker = None
    sources = []  # (field_row, column, label)
    for r in range(field.shape[0]):
        cols = np.flatnonzero(field[r] != bg)
        if len(cols) > 1:
            return None
        if len(cols) == 1:
            c = int(cols[0])
            colour = int(field[r, c])
            if marker is None:
                marker = colour
            elif colour != marker:
                return None
            sources.append((r, c, int(legend[c])))
    if marker is None or marker in set(legend.tolist()):
        return None
    return legend.copy(), bg, marker, sources


def _legend_candidates(grid: np.ndarray):
    found = []
    for edge in EDGES:
        parsed = _parse_canonical(_to_canonical(grid, edge))
        if parsed is not None:
            found.append((edge, parsed))
    return found


def _render_canonical(canon_in: np.ndarray, parsed) -> np.ndarray:
    legend, bg, marker, sources = parsed
    out = canon_in.copy()
    F, W = canon_in.shape[0] - 1, canon_in.shape[1]
    positions = np.array([s[0] for s in sources])
    by_row = {s[0]: s for s in sources}
    for r in range(F):
        if r in by_row:
            _, c, label = by_row[r]
            out[1 + r, :] = label
            out[1 + r, c] = marker
            continue
        d = np.abs(positions - r)
        nearest = {sources[i][2] for i in np.flatnonzero(d == d.min())}
        out[1 + r, :] = nearest.pop() if len(nearest) == 1 else marker
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    candidates = _legend_candidates(grid)
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one legend edge, found {len(candidates)}")
    edge, parsed = candidates[0]
    canon = _to_canonical(grid, edge)
    return _from_canonical(_render_canonical(canon, parsed), edge)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _make_legend(rng, style, colours, max_w):
    """Return a legend array of length <= max_w using the given colour pool."""
    for _ in range(200):
        if style == "blocks":
            k = int(rng.integers(2, 6))
            widths = [int(rng.integers(2, 7)) for _ in range(k)]
            cols = list(rng.choice(colours, size=k, replace=False))
        elif style == "fine":
            k = int(rng.integers(3, 6))
            widths = [int(rng.choice([1, 1, 1, 2])) for _ in range(k)]
            cols = list(rng.choice(colours, size=k, replace=False))
        else:  # "repeat": a colour reappears in a non-adjacent segment
            k = int(rng.integers(3, 7))
            palette = list(rng.choice(colours, size=int(rng.integers(2, min(4, k) + 1)), replace=False))
            cols = [palette[0]]
            for _ in range(k - 1):
                options = [c for c in palette if c != cols[-1]]
                cols.append(options[int(rng.integers(len(options)))])
            repeated = any(cols[i] == cols[j] for i in range(k) for j in range(i + 2, k))
            if not repeated:
                continue
            widths = [int(rng.integers(1, 5)) for _ in range(k)]
        if sum(widths) > max_w or sum(widths) < 3:
            continue
        return np.repeat(np.array(cols, dtype=int), widths)
    # deterministic small fallback
    return np.array([colours[0], colours[0], colours[1], colours[1]], dtype=int)


def _sample_positions(rng, F, n, gap_req):
    """Field rows for n sources; gap_req maps gap index -> 'even'/'odd'."""
    for _ in range(200):
        gaps = []
        for i in range(n - 1):
            req = gap_req.get(i)
            if req == "even":
                gaps.append(int(rng.choice([2, 2, 4, 6])))
            elif req == "odd":
                gaps.append(int(rng.choice([3, 3, 5])))
            else:
                gaps.append(int(rng.integers(1, 7)))
        need = sum(gaps) + 1
        if need > F:
            continue
        extra = F - need
        top = bottom = 0
        slots = ["top", "bottom"] + list(range(n - 1))
        while extra > 0:
            slot = slots[int(rng.integers(len(slots)))]
            if slot == "top":
                top += 1; extra -= 1
            elif slot == "bottom":
                bottom += 1; extra -= 1
            elif gap_req.get(slot) in ("even", "odd"):
                if extra >= 2:
                    gaps[slot] += 2; extra -= 2
                else:
                    top += 1; extra -= 1
            else:
                gaps[slot] += 1; extra -= 1
        rows = [top]
        for g in gaps:
            rows.append(rows[-1] + g)
        if F - n < 2:
            continue
        return rows
    return None


def _choose_columns(rng, legend, n, label_req):
    """label_req maps gap index i -> 'same'/'diff' for sources i, i+1."""
    W = len(legend)
    for _ in range(200):
        cols = [int(rng.integers(W))]
        ok = True
        for i in range(n - 1):
            req = label_req.get(i)
            prev = cols[-1]
            if req == "same":
                options = [c for c in range(W) if c != prev and legend[c] == legend[prev]]
            elif req == "diff":
                options = [c for c in range(W) if legend[c] != legend[prev]]
            else:
                options = list(range(W))
            if not options:
                ok = False
                break
            cols.append(options[int(rng.integers(len(options)))])
        if ok and len({int(legend[c]) for c in cols}) >= 2:
            return cols
    return None


def _features(parsed, F):
    """Facts about the Voronoi structure that demonstrations must expose."""
    _, _, _, sources = parsed
    feats = {"seam": False, "merge_diffcol": False, "odd_diff": False, "merge_cross_segment": False}
    legend = parsed[0]
    for (r1, c1, l1), (r2, c2, l2) in zip(sources, sources[1:]):
        g = r2 - r1
        if g % 2 == 0 and l1 != l2:
            feats["seam"] = True
        if g % 2 == 0 and l1 == l2 and c1 != c2:
            feats["merge_diffcol"] = True
            lo, hi = sorted((c1, c2))
            if any(legend[x] != l1 for x in range(lo, hi + 1)):
                feats["merge_cross_segment"] = True
        if g % 2 == 1 and g >= 3 and l1 != l2:
            feats["odd_diff"] = True
    feats["n"] = len(sources)
    feats["seams"] = sum(1 for (a, _, la), (b, _, lb) in zip(sources, sources[1:]) if (b - a) % 2 == 0 and la != lb)
    return feats


def _make_pair(rng, spec, used):
    """spec: edge, style, F range, n range, label/gap requirements, palette."""
    for _ in range(300):
        bg = spec["bg"] if spec.get("bg") is not None else int(rng.integers(10))
        marker = spec["marker"] if spec.get("marker") is not None and spec["marker"] != bg else None
        if marker is None:
            marker = int(rng.choice([c for c in range(10) if c != bg]))
        pool = [c for c in range(10) if c not in (bg, marker)]
        edge = spec["edge"]
        legend = _make_legend(rng, spec["style"], pool, spec["max_w"])
        W = len(legend)
        if W < spec.get("min_w", 3):
            continue
        n = int(rng.integers(spec["n"][0], spec["n"][1] + 1))
        req_gap, req_lab = {}, {}
        for kind in spec.get("force", ()):
            free = [i for i in range(n - 1) if i not in req_gap]
            if not free:
                break
            i = free[int(rng.integers(len(free)))]
            if kind == "seam":
                req_gap[i], req_lab[i] = "even", "diff"
            elif kind == "merge":
                req_gap[i], req_lab[i] = "even", "same"
            elif kind == "odd_diff":
                req_gap[i], req_lab[i] = "odd", "diff"
        F = int(rng.integers(spec["F"][0], spec["F"][1] + 1))
        rows = _sample_positions(rng, F, n, req_gap)
        if rows is None:
            continue
        cols = _choose_columns(rng, legend, n, req_lab)
        if cols is None:
            continue
        canon = np.full((F + 1, W), bg, dtype=int)
        canon[0] = legend
        for r, c in zip(rows, cols):
            canon[1 + r, c] = marker
        grid = _from_canonical(canon, edge)
        if max(grid.shape) > 30:
            continue
        key = grid.tobytes() + bytes(grid.shape)
        if key in used:
            continue
        cands = _legend_candidates(grid)
        if len(cands) != 1 or cands[0][0] != edge:
            continue
        feats = _features(cands[0][1], F)
        if any(not feats[f] for f in spec.get("need", ())):
            continue
        out = solve(grid)
        used.add(key)
        return {"input": grid, "output": out}, feats, {"edge": edge, "F": F, "W": W, "style": spec["style"]}
    raise RuntimeError("could not build pair")


def _generate(seed, num_train, num_test):
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    edge_mode = rng.choice(["fixed", "mixed", "mixed"])
    shifts = ["denser", "larger", "fine_code", "repeat_legend"]
    if edge_mode == "mixed":
        shifts.append("new_edge")
    shift = str(rng.choice(shifts))
    base_edge = str(rng.choice(EDGES))
    puzzle_marker = int(rng.integers(10)) if rng.random() < 0.4 else None

    if edge_mode == "fixed":
        train_edges = [base_edge] * num_train
        test_edges = [base_edge] * num_test
    else:
        if shift == "new_edge":
            held = str(rng.choice(EDGES))
            others = [e for e in EDGES if e != held]
            train_edges = [others[i % 3] for i in rng.permutation(max(num_train, 3))][:num_train]
            if len(set(train_edges)) < 2:
                train_edges[1] = others[(others.index(train_edges[0]) + 1) % 3]
            test_edges = [held] * num_test
        else:
            train_edges = [str(rng.choice(EDGES)) for _ in range(num_train)]
            if len(set(train_edges)) < 2:
                train_edges[1] = EDGES[(EDGES.index(train_edges[0]) + 1 + int(rng.integers(3))) % 4]
            test_edges = [str(rng.choice(EDGES)) for _ in range(num_test)]

    train_style = ["blocks", "repeat", "fine"]
    if shift == "fine_code":
        train_style = ["blocks", "blocks", "repeat"]
    if shift == "repeat_legend":
        train_style = ["blocks", "blocks", "fine"]

    if shift == "denser":
        train_n, test_n = (2, 4), (5, 7)
        train_F, test_F = (8, 16), (16, 26)
    elif shift == "larger":
        train_n, test_n = (2, 4), (3, 6)
        train_F, test_F = (6, 12), (18, 29)
    else:
        train_n, test_n = (2, 5), (3, 6)
        train_F, test_F = (7, 18), (10, 24)
    train_max_w = 12 if shift == "larger" else 18
    test_max_w = 30

    used = set()
    train, train_feats = [], []
    anchors = [("seam", "odd_diff"), ("merge",)]
    anchor_need = [("seam", "odd_diff"), ("merge_diffcol",)]
    for i in range(num_train):
        style = train_style[int(rng.integers(len(train_style)))]
        force, need = (), ()
        if i < 2:
            force, need = anchors[i], anchor_need[i]
            if i == 1 and style == "fine":
                style = "blocks"
        n_range = (max(train_n[0], 3), max(train_n[1], 3)) if i == 0 else train_n
        spec = {"edge": train_edges[i], "style": style, "F": train_F, "n": n_range,
                "force": force, "need": need, "max_w": train_max_w, "marker": puzzle_marker}
        pair, feats, _ = _make_pair(rng, spec, used)
        train.append(pair)
        train_feats.append(feats)
    order = rng.permutation(num_train)
    train = [train[i] for i in order]
    train_feats = [train_feats[i] for i in order]

    test, test_feats = [], []
    for j in range(num_test):
        if shift == "fine_code":
            style = "fine"
        elif shift == "repeat_legend":
            style = "repeat"
        elif shift == "larger":
            style = str(rng.choice(["blocks", "repeat"]))
        else:
            style = str(rng.choice(["blocks", "repeat", "fine"]))
        force = ()
        need = ()
        if shift == "repeat_legend":
            force, need = ("merge", "seam"), ("merge_cross_segment",)
        elif rng.random() < 0.6:
            force = (str(rng.choice(["seam", "merge", "odd_diff"])),)
        min_w = 3
        if shift == "larger":
            min_w = 13
        spec = {"edge": test_edges[j], "style": style, "F": test_F, "n": test_n, "force": force,
                "need": need, "max_w": test_max_w, "min_w": min_w, "marker": puzzle_marker}
        pair, feats, _ = _make_pair(rng, spec, used)
        test.append(pair)
        test_feats.append(feats)
    meta = {"shift": shift, "edge_mode": str(edge_mode), "train_edges": train_edges,
            "test_edges": test_edges, "train_feats": train_feats, "test_feats": test_feats}
    return {"train": train, "test": test}, meta


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    puzzle, _ = _generate(seed, num_train, num_test)
    return {split: [{"input": p["input"].astype(int), "output": p["output"].astype(int)}
                    for p in puzzle[split]] for split in ("train", "test")}


# ---------------------------------------------------------------------------
# Validation and self test
# ---------------------------------------------------------------------------
def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    seen = set()
    feats_train = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert grid.ndim == 2 and 1 <= min(grid.shape) and max(grid.shape) <= 30
            assert grid.shape == out.shape, "output keeps the input shape"
            key = grid.tobytes() + bytes(grid.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            cands = _legend_candidates(grid)
            assert len(cands) == 1, "legend edge must be unique"
            edge, parsed = cands[0]
            legend, bg, marker, sources = parsed
            canon_in = _to_canonical(grid, edge)
            canon_out = _to_canonical(out, edge)
            F = canon_in.shape[0] - 1
            assert len(sources) >= 2, "at least two marked lines"
            assert F - len(sources) >= 2, "at least two plain background lines"
            assert len({s[2] for s in sources}) >= 2, "at least two decoded labels"
            assert np.array_equal(canon_out[0], legend), "legend copied unchanged"
            assert bg not in set(canon_out.ravel().tolist()), "background eliminated in output"
            # Independent interval reconstruction of the output field.
            expect = np.empty((F, canon_in.shape[1]), dtype=int)
            pos = [s[0] for s in sources]
            for k, (r, c, lab) in enumerate(sources):
                lo = 0 if k == 0 else None
                hi = F - 1 if k == len(sources) - 1 else None
                if lo is None:
                    g = r - pos[k - 1]
                    lo = r - (g - 1) // 2
                if hi is None:
                    g = pos[k + 1] - r
                    hi = r + (g - 1) // 2
                expect[lo:hi + 1, :] = lab
                expect[r, :] = lab
                expect[r, c] = marker
            for k in range(len(sources) - 1):
                g = pos[k + 1] - pos[k]
                if g % 2 == 0:
                    mid = pos[k] + g // 2
                    expect[mid, :] = sources[k][2] if sources[k][2] == sources[k + 1][2] else marker
            assert np.array_equal(canon_out[1:], expect), "output disagrees with interval rule"
            assert np.array_equal(solve(grid), out)
            if split == "train":
                feats_train.append(_features(parsed, F))
    assert any(f["seam"] for f in feats_train), "training shows a different-label seam"
    assert any(f["merge_diffcol"] for f in feats_train), "training shows equal-label merge with different marker positions"
    assert any(f["odd_diff"] for f in feats_train), "training shows an odd gap with no seam"


def self_test() -> None:
    seeds = list(range(42))
    shifts, edges, shapes, ns = set(), set(), set(), set()
    for seed in seeds:
        puzzle, meta = _generate(seed, 4, 1)
        validate_generated_puzzle(puzzle)
        again, _ = _generate(seed, 4, 1)
        for a, b in zip(puzzle["train"] + puzzle["test"], again["train"] + again["test"]):
            assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        shifts.add(meta["shift"])
        edges.update(meta["train_edges"] + meta["test_edges"])
        tr_n = [f["n"] for f in meta["train_feats"]]
        te_n = [f["n"] for f in meta["test_feats"]]
        ns.update(tr_n + te_n)
        if meta["shift"] == "denser":
            assert min(te_n) > max(tr_n)
        if meta["shift"] == "new_edge":
            assert not set(meta["test_edges"]) & set(meta["train_edges"])
        if meta["shift"] == "repeat_legend":
            assert all(f["merge_cross_segment"] for f in meta["test_feats"])
        for p in puzzle["train"] + puzzle["test"]:
            shapes.add(p["input"].shape)
        # Shortcut checks: ignoring either parent changes some answer.
        for p in puzzle["test"] + puzzle["train"]:
            g = p["input"]
            assert not np.array_equal(p["output"], g)
    assert len(shifts) >= 4, shifts
    assert edges == set(EDGES)
    assert len(shapes) >= 40
    assert min(ns) == 2 and max(ns) >= 6
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 4), (2, 2)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, a, b))


if __name__ == "__main__":
    self_test()
    print("self_test ok")
