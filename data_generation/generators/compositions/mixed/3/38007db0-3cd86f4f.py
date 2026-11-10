"""Select each band's nonmaster panel, then shear its complete rows.

Only NumPy and the standard library are used. No files, examples, seed, or
generation parameters are consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must be rectangular, with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid colors must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and
                 isinstance(value, (int, np.integer)) and low <= value <= high,
                 "counts must be integers: train 2..10, test 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _parse(input_grid):
    """Recover lattice, semantic colors, master, and visible exceptions."""
    a = _grid(input_grid)
    lattice = int(a[0, 0])
    rs = np.flatnonzero(np.all(a == lattice, axis=1)).tolist()
    cs = np.flatnonzero(np.all(a == lattice, axis=0)).tolist()
    _require(len(rs) >= 2 and len(cs) >= 4, "need bands and at least three copies")
    _require(rs[0] == cs[0] == 0 and rs[-1] == a.shape[0] - 1 and
             cs[-1] == a.shape[1] - 1, "lattice must enclose the array")
    _require(len(set(np.diff(rs))) == len(set(np.diff(cs))) == 1,
             "lattice spacing must be regular")
    h, w = rs[1] - 1, cs[1] - 1
    _require(h >= 2 and w >= 3 and h + w + 1 <= 30, "invalid panel dimensions")
    panels = [[a[r + 1:s, c + 1:d] for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    flat = [p for row in panels for p in row]
    _require(all(np.all(p != lattice) for p in flat), "lattice color is structural")
    frequencies = Counter(_key(p) for p in flat)
    maximum = max(frequencies.values())
    winners = [k for k, count in frequencies.items() if count == maximum]
    _require(len(winners) == 1, "master frequency tie")
    master = next(p for p in flat if _key(p) == winners[0])
    carrier = int(master[0, 0])
    roles = set(map(int, np.unique(master)))
    _require(len(roles) == 2 and lattice not in roles, "master has two interior roles")
    motif = next(iter(roles - {carrier}))
    for p in flat:
        _require(set(map(int, np.unique(p))) <= roles and
                 bool(np.all(p[:, (0, -1)] == carrier)),
                 "panels must have carrier side rails")
    selected, columns = [], []
    for row in panels:
        indices = [i for i, p in enumerate(row) if not np.array_equal(p, master)]
        _require(len(indices) == 1, "each band needs exactly one visible exception")
        columns.append(indices[0])
        selected.append(row[indices[0]])
    return dict(grid=a, lattice=lattice, carrier=carrier, motif=motif, h=h, w=w,
                bands=len(panels), copies=len(panels[0]), rows=rs, cols=cs,
                panels=panels, master=master, selected=selected, columns=columns)


def _frame(panels, h, w, lattice, carrier, shear=True):
    """Render one panel per band; offsets restart below every separator."""
    width = w + h - 1
    out = np.full((len(panels) * (h + 1) + 1, width + 2), lattice, dtype=np.int8)
    for band, panel in enumerate(panels):
        y = 1 + band * (h + 1)
        out[y:y + h, 1:-1] = carrier
        for r in range(h):
            start = 1 + (h - 1 - r if shear else 0)
            out[y + r, start:start + w] = panel[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only full-bitmap selection and bottom-anchored per-panel shear."""
    info = _parse(input_grid)
    return _frame(info["selected"], info["h"], info["w"],
                  info["lattice"], info["carrier"])


# Every feasible geometry in the declared family, including single bands,
# very tall panels, short panels with many bands, and rectangular panels.
_GEOMETRIES = tuple(
    (h, w, b, c)
    for h in range(2, 27) for w in range(3, 9) if h + w + 1 <= 30
    for b in range(1, 29 // (h + 1) + 1)
    for c in range(3, 29 // (w + 1) + 1)
)


def _choose(rng, values):
    _require(bool(values), "empty geometry pool")
    # Mix moderate examples and the full feasible range without a shape cycle.
    if rng.random() < 0.65:
        moderate = [g for g in values if g[0] <= 10 and g[1] <= 7 and g[2] <= 5]
        if moderate:
            values = moderate
    return values[int(rng.integers(len(values)))]


def _mask(rng, h, w, witness):
    """Varied clean motif geometries; width excludes the two carrier rails."""
    area = h * w
    style = int(rng.integers(7))
    rr, cc = np.indices((h, w))
    for _ in range(24):
        density = float(rng.uniform(0.20, 0.80))
        if style == 0:
            m = rng.random((h, w)) < density
        elif style == 1:
            axis = rr if rng.random() < 0.5 else cc
            period = int(rng.integers(2, 5))
            m = axis % period == int(rng.integers(period))
        elif style == 2:
            period = int(rng.integers(2, 5))
            m = (rr + int(rng.choice((-1, 1))) * cc) % period == int(rng.integers(period))
        elif style == 3:
            m = np.zeros((h, w), dtype=bool)
            r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
            if w >= 2:
                c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
                m[r0:r1 + 1, (c0, c1)] = True
                m[(r0, r1), c0:c1 + 1] = True
            else:
                m[r0:r1 + 1, 0] = True
        elif style == 4:
            m = np.zeros((h, w), dtype=bool)
            m[int(rng.integers(h)), int(rng.integers(w))] = True
            for _ in range(max(1, int(area * density)) - 1):
                frontier = [(r, c) for r in range(h) for c in range(w)
                            if not m[r, c] and any(0 <= y < h and 0 <= x < w and m[y, x]
                            for y, x in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)))]
                if frontier:
                    m[frontier[int(rng.integers(len(frontier)))]] = True
        elif style == 5:
            template = rng.random(w) < density
            m = np.broadcast_to(template, (h, w)).copy()
            if rng.random() < 0.5:
                m[int(rng.integers(1, h)):] = ~template
        else:
            m = ((rr + cc) % 2 == int(rng.integers(2))) & (rng.random((h, w)) < 0.85)
        if rng.random() < 0.25:
            m |= m[:, ::-1] if rng.random() < 0.5 else m[::-1]
        # A closed carrier perimeter is optional; side rails are always present.
        if not witness and h >= 4 and rng.random() < 0.25:
            m[[0, -1]] = False
        if witness:
            pin = int(rng.integers(w))
            m[[0, h - 2, h - 1], pin] = True
            if int(m.sum()) >= 7 and int((~m).sum()) >= 4:
                return m, [(0, pin), (h - 2, pin), (h - 1, pin)]
        elif 1 <= int(m.sum()) < area:
            return m, []
    # Finite, deterministic fallback preserving the same construction contract.
    m = np.zeros((h, w), dtype=bool)
    if witness:
        pins = [(0, 0), (h - 2, 0), (h - 1, 0)]
        for p in pins:
            m[p] = True
        free = np.flatnonzero(~m)
        m.flat[rng.choice(free, 4, replace=False)] = True
        return m, pins
    m.flat[rng.choice(area, max(1, area // 2), replace=False)] = True
    return m, []


def _edit(rng, original, pins, mode=None):
    m = original.copy()
    allowed = np.ones(m.shape, dtype=bool)
    for p in pins:
        allowed[p] = False
    ink = np.flatnonzero(m & allowed)
    blank = np.flatnonzero(~m & allowed)
    if mode is None:
        mode = str(rng.choice(("add", "remove", "exchange", "flip", "cut")))
    cap = max(1, int(m.size * 0.40))
    if mode in ("exchange", "large_exchange") and len(ink) and len(blank):
        high = min(len(ink), len(blank), max(1, cap // 2))
        n = 3 if mode == "large_exchange" else int(rng.integers(1, high + 1))
        m.flat[rng.choice(ink, n, replace=False)] = False
        m.flat[rng.choice(blank, n, replace=False)] = True
    elif mode in ("add", "remove"):
        candidates = blank if mode == "add" else ink
        if len(candidates):
            n = int(rng.integers(1, min(cap, len(candidates)) + 1))
            m.flat[rng.choice(candidates, n, replace=False)] = mode == "add"
    elif mode == "cut" and len(ink):
        anchor = np.unravel_index(int(rng.choice(ink)), m.shape)
        axis = int(rng.integers(2))
        cells = [int(i) for i in ink if np.unravel_index(int(i), m.shape)[axis] == anchor[axis]]
        m.flat[rng.choice(cells, min(len(cells), cap), replace=False)] = False
    else:
        candidates = np.flatnonzero(allowed)
        n = int(rng.integers(1, min(cap, len(candidates)) + 1))
        indices = rng.choice(candidates, n, replace=False)
        m.flat[indices] = ~m.flat[indices]
    if np.array_equal(m, original) or not np.any(m):
        m = original.copy()
        # Prefer adding; otherwise remove a single unprotected motif cell.
        candidates = blank if len(blank) else ink
        index = int(rng.choice(candidates))
        m.flat[index] = ~m.flat[index]
    return m


def _make_pair(rng, geometry, palette, witness=False):
    h, w, bands, copies = geometry
    lattice, carrier, motif = palette
    mask, pins = _mask(rng, h, w - 2, witness)
    master = np.full((h, w), carrier, dtype=np.int8)
    master[:, 1:-1][mask] = motif
    columns = list(map(int, rng.integers(copies, size=bands)))
    required = {}
    if witness:
        low, high = sorted(map(int, rng.choice(copies, 2, replace=False)))
        columns[:3] = [high, low, high]
        modes = list(rng.permutation(("add", "remove", "large_exchange")))
        required = dict(enumerate(modes))
    variants = []
    inp = np.full((bands * (h + 1) + 1, copies * (w + 1) + 1), lattice, dtype=np.int8)
    for b in range(bands):
        if not witness and variants and rng.random() < 0.16:
            variant = variants[int(rng.integers(len(variants)))].copy()
        else:
            changed = _edit(rng, mask, pins, required.get(b))
            variant = np.full((h, w), carrier, dtype=np.int8)
            variant[:, 1:-1][changed] = motif
        variants.append(variant)
        for c in range(copies):
            y, x = 1 + b * (h + 1), 1 + c * (w + 1)
            inp[y:y + h, x:x + w] = variant if c == columns[b] else master
    # Ground truth comes from the constructed panels, not from solve().
    out = _frame(variants, h, w, lattice, carrier)
    return {"input": inp, "output": out}


def _unique(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # At most 13 previous inputs and 720 injective role palettes: finite fallback.
    old = sorted(map(int, np.unique(pair["input"])))
    for a in range(10):
        for b in range(10):
            for c in range(10):
                if len({a, b, c}) != 3:
                    continue
                lookup = np.arange(10, dtype=np.int8)
                lookup[old] = (a, b, c)
                candidate = {k: lookup[v] for k, v in pair.items()}
                if _key(candidate["input"]) not in seen:
                    return candidate
    raise AssertionError("role palette exhaustion is impossible")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # One observable geometric quantity is extrapolated; no rule changes by split.
    axis = int(rng.integers(3))  # panel height, panel width, or band count
    cap = int(rng.integers(5, 11) if axis == 0 else
              rng.integers(5, 7) if axis == 1 else rng.integers(3, 6))
    train_pool = [g for g in _GEOMETRIES if g[axis] <= cap]
    test_pool = [g for g in _GEOMETRIES if g[axis] > cap]
    witness_pool = [g for g in train_pool if 4 <= g[0] <= 8 and 5 <= g[1] <= 7
                    and g[2] >= 3 and g[0] * (g[1] - 2) >= 20]
    first = _choose(rng, witness_pool)
    geometries = [first]
    geometries.append(_choose(rng, [g for g in train_pool
                                    if g[0] != first[0] and g[1] != first[1]
                                    and g[2] >= 2]))
    geometries.extend(_choose(rng, train_pool) for _ in range(nt - 2))
    geometries.extend(_choose(rng, test_pool) for _ in range(ne))
    shared = rng.random() < 0.20
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for i, geometry in enumerate(geometries):
        colors = palette if shared else tuple(map(int, rng.choice(10, 3, replace=False)))
        pair = _unique(_make_pair(rng, geometry, colors, witness=i == 0), seen)
        seen.add(_key(pair["input"]))
        puzzle["train" if i < nt else "test"].append(pair)
    # The informative example's position is not itself a solving convention.
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
    info = _parse(pair["input"])
    out = _grid(pair["output"])
    h, w, bands = info["h"], info["w"], info["bands"]
    _require(out.shape == (bands * (h + 1) + 1, w + h + 1), "wrong output dimensions")
    _require(bool(np.all(out[info["rows"]] == info["lattice"])) and
             bool(np.all(out[:, (0, -1)] == info["lattice"])), "wrong lattice frame")
    edits, visible, top_ink, order_keys = [], False, False, []
    for b, panel in enumerate(info["selected"]):
        y = 1 + b * (h + 1)
        for r in range(h):
            row = out[y + r, 1:-1]
            start = h - 1 - r
            _require(np.array_equal(row[start:start + w], panel[r]), "row vector changed or misplaced")
            _require(bool(np.all(row[:start] == info["carrier"])) and
                     bool(np.all(row[start + w:] == info["carrier"])), "wrong padding")
        # The inverse windows recover all carrier pixels as well as motif pixels.
        recovered = np.array([out[y + r, 1 + h - 1 - r:1 + h - 1 - r + w]
                              for r in range(h)], dtype=np.int8)
        _require(np.array_equal(recovered, panel), "lossy row copy")
        _require(np.count_nonzero(out[y:y+h] == info["motif"]) ==
                 np.count_nonzero(panel == info["motif"]), "motif mass changed")
        add = int(np.count_nonzero((panel == info["motif"]) & (info["master"] == info["carrier"])))
        remove = int(np.count_nonzero((panel == info["carrier"]) & (info["master"] == info["motif"])))
        edits.append((add, remove))
        visible |= bool(np.any(panel[-1] == info["motif"]) and np.any(panel[-2] == info["motif"]))
        top_ink |= bool(np.any(panel[:-1] == info["motif"]))
        order_keys.append(_key(panel))
    info.update(edits=edits, visible_bottom=visible, top_ink=top_ink, order_keys=order_keys)
    return info


def validate_generated_puzzle(puzzle):
    """Check the rule from returned grids, plus the promised training evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, measurements = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(info["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            measurements[split].append(info)
    train = measurements["train"]
    changes = [e for m in train for e in m["edits"]]
    _require(any(a > 0 and r == 0 for a, r in changes), "need addition evidence")
    _require(any(a == 0 and r > 0 for a, r in changes), "need removal evidence")
    _require(any(a == r and a >= 3 for a, r in changes), "need count-preserving bitmap evidence")
    _require(any(m["visible_bottom"] and m["top_ink"] for m in train), "need visible shear anchors")
    _require(len({m["h"] for m in train}) >= 2 and len({m["w"] for m in train}) >= 2,
             "demonstrate dimension-based offsets and width")
    _require(any(len(set(m["columns"])) > 1 and any(a > b for a, b in
                 zip(m["columns"], m["columns"][1:])) for m in train),
             "selection must not be a fixed column or column sort")
    _require(any(m["order_keys"] != m["order_keys"][::-1] for m in train), "need band-order evidence")
    for m in measurements["test"]:
        _require(any(m[k] > max(t[k] for t in train) for k in ("h", "w", "bands")),
                 "test must extrapolate a demonstrated geometric quantity")
    # Counterfactuals use the same canvas to avoid a dimension-only distinction.
    _require(any(not np.array_equal(_frame([m["master"]] * m["bands"], m["h"], m["w"],
                 m["lattice"], m["carrier"]), p["output"])
                 for m, p in zip(train, puzzle["train"])), "A must change the answer")
    _require(any(not np.array_equal(_frame(m["selected"], m["h"], m["w"],
                 m["lattice"], m["carrier"], shear=False), p["output"])
                 for m, p in zip(train, puzzle["train"])), "B must change the answer")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.digest()


def self_test():
    """Bounded streaming checks: 42 seeds, all counts, geometry and mutations.

    No artifacts or execution traces are written. The 42 puzzles are streamed;
    only a few small puzzles coexist, and diversity sets hold keys or hashes.
    """
    signatures, geometries, masters, palettes, structures = set(), set(), set(), set(), set()
    heights, widths, bands_seen, copies_seen, edit_counts = set(), set(), set(), set(), set()
    repeated = aligned = rectangular = empty_rows = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _analyze(pair)
                geometries.add((m["h"], m["w"], m["bands"], m["copies"]))
                heights.add(m["h"]); widths.add(m["w"])
                bands_seen.add(m["bands"]); copies_seen.add(m["copies"])
                masters.add(_key(m["master"] == m["motif"]))
                palettes.add((m["lattice"], m["carrier"], m["motif"]))
                normalized = np.where(m["grid"] == m["lattice"], 0,
                             np.where(m["grid"] == m["carrier"], 1, 2)).astype(np.int8)
                structures.add(hashlib.sha256(bytes(normalized.shape) + normalized.tobytes()).digest())
                edit_counts.update(a + r for a, r in m["edits"])
                repeated += len(set(m["columns"])) < m["bands"]
                aligned += m["bands"] > 1 and len(set(m["columns"])) == 1
                rectangular += m["h"] != m["w"]
                empty_rows += any(np.all(p[r] == m["carrier"]) for p in m["selected"] for r in range(m["h"]))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                pairs += 1
    assert len(signatures) == 42
    assert len(geometries) >= 60 and len(masters) >= 100 and len(structures) >= 180
    assert len(heights) >= 12 and len(widths) >= 5 and len(bands_seen) >= 6 and len(copies_seen) >= 4
    assert len(palettes) >= 70 and {p[0] for p in palettes} == set(range(10))
    assert len(edit_counts) >= 8 and repeated >= 42 and aligned >= 1 and rectangular >= 80 and empty_rows >= 10
    assert _puzzle_digest(generate_puzzle(273339958, 7, 3)) == _puzzle_digest(generate_puzzle(273339958, 7, 3))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1),
                   (4,False), (4.0,1), (4,1.0), (None,1), (4,None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    base = generate_puzzle(0, 2, 1)
    target = next(i for i, p in enumerate(base["train"])
                  if any(a == r and a >= 3 for a, r in _analyze(p)["edits"]))
    anchor = _parse(base["train"][target]["input"])
    for offset in range(anchor["h"]):
        rigid = np.full_like(base["train"][target]["output"], anchor["lattice"])
        for b, panel in enumerate(anchor["selected"]):
            y = 1 + b * (anchor["h"] + 1)
            rigid[y:y + anchor["h"], 1:-1] = anchor["carrier"]
            rigid[y:y + anchor["h"], 1 + offset:1 + offset + anchor["w"]] = panel
        assert not np.array_equal(rigid, base["train"][target]["output"])
    for defect in ("missing", "master", "rigid", "delta", "reverse", "sort", "padding", "pixel", "frame"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        p = bad["train"][target]
        m = _parse(p["input"])
        if defect == "missing":
            x = m["cols"][m["columns"][0]] + 1
            p["input"][1:1 + m["h"], x:x + m["w"]] = m["master"]
        elif defect in ("master", "rigid", "delta", "reverse", "sort"):
            selected = m["selected"]
            if defect == "master":
                selected = [m["master"]] * m["bands"]
            elif defect == "delta":
                selected = [np.where((v != m["master"]) & (v == m["motif"]),
                            m["motif"], m["carrier"]).astype(np.int8) for v in selected]
            elif defect == "reverse":
                selected = selected[::-1]
            elif defect == "sort":
                selected = [selected[i] for i in sorted(range(m["bands"]), key=lambda i: m["columns"][i])]
            p["output"] = _frame(selected, m["h"], m["w"], m["lattice"], m["carrier"], defect != "rigid")
        elif defect == "padding":
            p["output"][1, 1] = m["lattice"]
        elif defect == "pixel":
            p["output"][2, 2] = (int(p["output"][2, 2]) + 1) % 10
        else:
            p["output"][0, 0] = m["carrier"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted " + defect)
    return {"seeds": 42, "pairs": pairs, "geometries": len(geometries),
            "master_bitmaps": len(masters), "role_palettes": len(palettes),
            "normalized_inputs": len(structures), "panel_heights": sorted(heights),
            "panel_widths": sorted(widths), "band_counts": sorted(bands_seen),
            "copy_counts": sorted(copies_seen), "edit_counts": sorted(edit_counts),
            "repeated_columns": repeated, "aligned_columns": aligned,
            "rectangular_panels": rectangular, "carrier_only_rows": empty_rows}
