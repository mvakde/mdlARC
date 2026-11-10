"""Mirror stencils: one simultaneous dilation on a visible tile lattice.

No files, generation state, or training examples are used by solve().
"""
from itertools import combinations
import numbers

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid ARC color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _trim_key(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return frozenset((r-r0, c-c0) for r, c in cells)


def _marker(cells):
    cells = sorted(cells)
    if len(cells) == 1:
        return "P", cells[0]
    _require(len(cells) >= 3 and len(cells) % 2 == 1, "Line marker needs an odd length >= 3")
    rs, cs = {r for r, c in cells}, {c for r, c in cells}
    if len(rs) == 1 and max(cs)-min(cs)+1 == len(cells):
        return "H", (cells[0][0], (min(cs)+max(cs))//2)
    if len(cs) == 1 and max(rs)-min(rs)+1 == len(cells):
        return "V", ((min(rs)+max(rs))//2, cells[0][1])
    raise ValueError("Marker must be a single point or one straight odd line")


def _offsets(payload, center, kind):
    raw = {(r-center[0], c-center[1]) for r, c in payload}
    if kind == "H":
        reflected = {(-r, c) for r, c in raw}
    elif kind == "V":
        reflected = {(r, -c) for r, c in raw}
    else:
        reflected = {(-r, -c) for r, c in raw}
    return raw, reflected


def _targets(seeds, kernel):
    return {(r+dr, c+dc) for r, c in seeds for dr, dc in kernel}


def _inside(cells, rows, cols):
    return {(r, c) for r, c in cells if 0 <= r < rows and 0 <= c < cols}


def _next_generation(active, kernel, rows, cols):
    # Only count a recursive alternative where the entire next stamp fits.
    result = set()
    for q in active:
        requested = _targets({q}, kernel)
        if _inside(requested, rows, cols) == requested:
            result.update(requested)
    return result - active


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][1] + 1 == i:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    divider = int(a[0, 0])
    rr = _runs(np.flatnonzero(np.all(a == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(a == divider, axis=0)))
    _require(len(rr) >= 3 and len(cc) >= 3, "Missing complete divider lattice")
    _require(rr[0][0] == cc[0][0] == 0 and rr[-1][1] == a.shape[0]-1
             and cc[-1][1] == a.shape[1]-1, "Incomplete lattice border")
    thickness = {b-x+1 for x, b in rr+cc}
    hs = {rr[i+1][0]-rr[i][1]-1 for i in range(len(rr)-1)}
    ws = {cc[i+1][0]-cc[i][1]-1 for i in range(len(cc)-1)}
    _require(len(thickness) == len(hs) == len(ws) == 1, "Irregular lattice")
    t, h, w = thickness.pop(), hs.pop(), ws.pop()
    _require(t in (1, 2) and 2 <= h <= 4 and 2 <= w <= 4, "Unsupported tile geometry")
    rows, cols = len(rr)-1, len(cc)-1
    records = {}
    for r in range(rows):
        for c in range(cols):
            y, x = rr[r][1]+1, cc[c][1]+1
            tile = a[y:y+h, x:x+w]
            records.setdefault(_key(tile), [tile.copy(), set()])[1].add((r, c))
    _require(len(records) == 2, "Expected one ordinary pattern and one glyph pattern")
    ordered = sorted(records.values(), key=lambda item: -len(item[1]))
    (ordinary, ordinary_sites), (glyph, seeds) = ordered
    _require(len(ordinary_sites) > len(seeds) >= 1, "Ordinary tiles must be a strict majority")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _require(len(ordinary_colors) in (2, 3) and divider not in ordinary_colors,
             "Ordinary tile must use two or three non-divider colors")
    novel = set(map(int, np.unique(glyph))) - ordinary_colors - {divider}
    _require(len(novel) == 1, "Glyph needs one distinctive ink color")
    ink = novel.pop()
    marker_cells = set(map(tuple, np.argwhere(glyph == divider).tolist()))
    kind, center = _marker(marker_cells)
    payload = set(map(tuple, np.argwhere(glyph == ink).tolist()))
    _require(len(payload) >= 2, "Ink stencil must contain at least two pixels")
    background_mask = (glyph != ink) & (glyph != divider)
    backgrounds = set(map(int, np.unique(glyph[background_mask])))
    _require(len(backgrounds) == 1 and backgrounds <= ordinary_colors, "Invalid glyph background")
    ar, ac = center
    if kind == "H":
        _require(all(r < ar for r, c in payload) or all(r > ar for r, c in payload),
                 "Horizontal payload must lie strictly on one side")
    elif kind == "V":
        _require(all(c < ac for r, c in payload) or all(c > ac for r, c in payload),
                 "Vertical payload must lie strictly on one side")
    else:
        _require((all(r <= ar for r, c in payload) or all(r >= ar for r, c in payload))
                 and (all(c <= ac for r, c in payload) or all(c >= ac for r, c in payload)),
                 "Point payload must lie in one closed quadrant")
    raw, kernel = _offsets(payload, center, kind)
    targets = _targets(seeds, kernel)
    _require(_inside(targets, rows, cols) == targets, "A requested copy leaves the lattice")
    active = seeds | targets
    _require(bool(active-seeds), "At least one new tile is required")
    return {"array": a, "geometry": (h, w, t, rows, cols), "divider": divider,
            "ordinary": ordinary, "glyph": glyph, "ink": ink, "kind": kind,
            "center": center, "payload": payload, "raw": raw, "kernel": kernel,
            "seeds": seeds, "targets": targets, "active": active,
            "background_mask": background_mask}


def _paint(geometry, divider, ordinary, glyph, sites):
    h, w, t, rows, cols = geometry
    a = np.full((rows*(h+t)+t, cols*(w+t)+t), divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            y, x = t+r*(h+t), t+c*(w+t)
            a[y:y+h, x:x+w] = glyph if (r, c) in sites else ordinary
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the input snapshot and apply its mirrored stencil once per seed."""
    f = _decode(input_grid)
    out = f["array"].copy()
    h, w, t, _, _ = f["geometry"]
    for r, c in f["active"] - f["seeds"]:
        y, x = t+r*(h+t), t+c*(w+t)
        out[y:y+h, x:x+w] = f["glyph"]
    return out


def _witnesses(f):
    _, _, _, rows, cols = f["geometry"]
    seeds, kernel, active = f["seeds"], f["kernel"], f["active"]
    unreflected = seeds | _inside(_targets(seeds, f["raw"]), rows, cols)
    ghost = _next_generation(active, kernel, rows, cols)
    kr, kc = min(r for r, c in kernel), min(c for r, c in kernel)
    translated = {(r+kr, c+kc) for r, c in _trim_key(f["raw"])}
    wrong_orientation = seeds | _inside(_targets(seeds, translated), rows, cols)
    competing = [_offsets(f["payload"], f["center"], k)[1]
                 for k in ("H", "V", "P") if k != f["kind"]]
    requests = [_targets({s}, kernel) for s in sorted(seeds)]
    overlap = any(bool((a & b)-seeds) for a, b in combinations(requests, 2))
    return {"no_a_differs": unreflected != active,
            "orientation": _trim_key(f["raw"]) != _trim_key(kernel),
            "orientation_output": wrong_orientation != active,
            "marker_selects": all(seeds | _inside(_targets(seeds, k), rows, cols) != active
                                  for k in competing),
            "one_step": bool(ghost), "overlap": overlap,
            "background_copy": bool(np.any(f["ordinary"][f["background_mask"]]
                                           != f["glyph"][f["background_mask"]]))}


def _sample_design(rng, kind, shape=None):
    shapes = [(h, w) for h in range(2, 5) for w in range(2, 5)
              if (kind != "H" or h >= 3 and w >= 3)
              and (kind != "V" or h >= 3 and w >= 3)]
    h, w = shapes[int(rng.integers(len(shapes)))] if shape is None else shape
    for _ in range(128):
        if kind == "H":
            choices = [(r, s) for r in range(h) for s in (-1, 1)
                       if (r if s < 0 else h-1-r) >= 2]
            ar, side = choices[int(rng.integers(len(choices)))]
            ac = int(rng.integers(1, w-1))
            marker = {(ar, ac+d) for d in (-1, 0, 1)}
            eligible = [(r, c) for r in range(h) for c in range(w) if (r-ar)*side > 0]
        elif kind == "V":
            choices = [(c, s) for c in range(w) for s in (-1, 1)
                       if (c if s < 0 else w-1-c) >= 2]
            ac, side = choices[int(rng.integers(len(choices)))]
            ar = int(rng.integers(1, h-1))
            marker = {(ar+d, ac) for d in (-1, 0, 1)}
            eligible = [(r, c) for r in range(h) for c in range(w) if (c-ac)*side > 0]
        else:
            ar, ac = int(rng.integers(h)), int(rng.integers(w))
            sr, sc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
            marker = {(ar, ac)}
            eligible = [(r, c) for r in range(h) for c in range(w)
                        if (r-ar)*sr >= 0 and (c-ac)*sc >= 0 and (r, c) != (ar, ac)]
        if len(eligible) < 3:
            continue
        count = int(rng.integers(2, min(9, len(eligible), h*w-len(marker)-1)+1))
        payload = {eligible[int(i)] for i in rng.choice(len(eligible), count, replace=False)}
        raw, kernel = _offsets(payload, (ar, ac), kind)
        alternatives = [_offsets(payload, (ar, ac), k)[1] for k in ("H", "V", "P") if k != kind]
        if _trim_key(raw) != _trim_key(kernel) and all(k != kernel for k in alternatives):
            return (h, w), marker, payload, kernel
    return None


def _place_seeds(rng, rows, cols, kernel, count, need_overlap):
    rmin, rmax = min(0, min(r for r, c in kernel)), max(0, max(r for r, c in kernel))
    cmin, cmax = min(0, min(c for r, c in kernel)), max(0, max(c for r, c in kernel))
    legal = [(r, c) for r in range(-rmin, rows-rmax) for c in range(-cmin, cols-cmax)]
    if len(legal) < count:
        return None
    rng.shuffle(legal)
    for _ in range(64):
        chosen = {legal[int(i)] for i in rng.choice(len(legal), count, replace=False)}
        targets = _targets(chosen, kernel)
        active = chosen | targets
        overlap = any(bool((_targets({a}, kernel) & _targets({b}, kernel))-chosen)
                      for a, b in combinations(sorted(chosen), 2))
        ghost = _next_generation(active, kernel, rows, cols)
        if active-chosen and ghost and (not need_overlap or overlap):
            return chosen
    return None


def _make_pair(rng, kind, count, palette, need_overlap=False, familiar=None, forbidden=(),
               forbidden_kernels=()):
    for attempt in range(256):
        design = familiar if familiar is not None and attempt < 192 else _sample_design(rng, kind)
        if design is None:
            continue
        (h, w), marker, payload, kernel = design
        if frozenset(kernel) in forbidden_kernels:
            continue
        t = int(rng.choice((1, 2), p=(.8, .2)))
        max_rows, max_cols = (30-t)//(h+t), (30-t)//(w+t)
        # Roomy boards are useful for one-step witnesses, but geometry is not fixed.
        rows = max_rows if rng.random() < .7 else int(rng.integers(3, max_rows+1))
        cols = max_cols if rng.random() < .7 else int(rng.integers(3, max_cols+1))
        seeds = _place_seeds(rng, rows, cols, kernel, count, need_overlap)
        if seeds is None:
            continue
        divider, ink, bg, *other = palette
        ordinary_colors = [bg] + other
        labels = np.concatenate((np.arange(len(ordinary_colors)),
                                 rng.integers(len(ordinary_colors), size=h*w-len(ordinary_colors))))
        rng.shuffle(labels)
        ordinary = np.asarray(ordinary_colors, dtype=np.int8)[labels.reshape(h, w)]
        glyph = np.full((h, w), bg, dtype=np.int8)
        for q in marker:
            glyph[q] = divider
        for q in payload:
            glyph[q] = ink
        bg_mask = (glyph == bg)
        if not np.any(ordinary[bg_mask] != bg):
            q = tuple(np.argwhere(bg_mask)[0])
            ordinary[q] = other[0]
        # Preserve every ordinary color after the one-cell repair.
        if len(np.unique(ordinary)) != len(ordinary_colors):
            continue
        geometry = (h, w, t, rows, cols)
        inp = _paint(geometry, divider, ordinary, glyph, seeds)
        if _key(inp) in forbidden:
            continue
        center = _marker(marker)[1]
        raw, _ = _offsets(payload, center, kind)
        active = seeds | _targets(seeds, kernel)
        if seeds | _inside(_targets(seeds, raw), rows, cols) == active:
            continue
        out = _paint(geometry, divider, ordinary, glyph, active)
        try:
            witnesses = _witnesses(_decode(inp))
        except ValueError:
            # A whole coarse row of horizontal glyph markers (or a vertical
            # column) can visually merge into an extra divider. Resample it.
            continue
        required = ("no_a_differs", "orientation", "orientation_output", "marker_selects",
                    "one_step", "background_copy") + (("overlap",) if need_overlap else ())
        if not all(witnesses[k] for k in required):
            continue
        return {"input": inp, "output": out}, design
    raise RuntimeError("Bounded construction could not find a readable stencil scene")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    _require(nt > ns, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    kinds = [str(k) for k in rng.choice(("H", "V", "P"), size=int(rng.integers(1, 3)), replace=False)]
    stable_palette = bool(rng.random() < .3)
    base_palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle, used, designs = {"train": [], "test": []}, set(), []
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            kind = kinds[i % len(kinds)] if split == "train" else kinds[(i+nt) % len(kinds)]
            seed_count = (1 if i % 2 == 0 else 2) if split == "train" else int(rng.integers(3, 5))
            palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, int(rng.integers(4, 6)), replace=False)))
            familiar = None
            if split == "test" and i == 0:
                familiar = next(d for k, d in designs if k == kind)
            pair, design = _make_pair(rng, kind, seed_count, palette,
                                      need_overlap=(split == "train" and i == 1),
                                      familiar=familiar, forbidden=used,
                                      forbidden_kernels=({frozenset(designs[0][1][3])}
                                                         if split == "train" and i == 1 else ()))
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if split == "train":
                designs.append((kind, design))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "Invalid example counts")
    seen, training_kinds, training_kernels = set(), set(), set()
    train_overlap = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Output dimensions changed")
            _require(_key(inp) not in seen, "Repeated input")
            seen.add(_key(inp))
            f = _decode(inp)
            expected = _paint(f["geometry"], f["divider"], f["ordinary"], f["glyph"], f["active"])
            _require(np.array_equal(expected, out), "Incorrect one-step whole-tile completion")
            _require(np.array_equal(solve(inp), out), "Solver disagrees")
            # Check all original seeds and dividers, and count changed complete interiors.
            h, w, t, rows, cols = f["geometry"]
            changed_sites = set()
            for r in range(rows):
                for c in range(cols):
                    y, x = t+r*(h+t), t+c*(w+t)
                    if np.any(inp[y:y+h, x:x+w] != out[y:y+h, x:x+w]):
                        changed_sites.add((r, c))
                        _require(np.array_equal(out[y:y+h, x:x+w], f["glyph"]), "Partial tile copy")
            _require(changed_sites == f["active"]-f["seeds"], "Wrong coarse sites changed")
            witnesses = _witnesses(f)
            _require(all(witnesses[k] for k in ("no_a_differs", "orientation", "orientation_output",
                                               "marker_selects", "one_step", "background_copy")),
                     "Missing intended concept or alternative-rule witness")
            if split == "train":
                training_kinds.add(f["kind"])
                training_kernels.add(frozenset(f["kernel"]))
                train_overlap |= witnesses["overlap"]
                _require(1 <= len(f["seeds"]) <= 2, "Training anchor count outside schedule")
            else:
                _require(f["kind"] in training_kinds, "Test operation not demonstrated")
                _require(3 <= len(f["seeds"]) <= 4, "Test must recombine more anchors")
    _require(train_overlap, "Training must show overlapping simultaneous requests")
    _require(len(training_kernels) >= 2, "Training must show distinct ink-derived stencils")
    return True


def self_test():
    """Bounded in-memory semantic and diversity checks; no persisted payloads."""
    shapes, designs, kernels, motifs, palettes, kinds, counts = (set() for _ in range(7))
    overlaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _decode(pair["input"])
                shapes.add(f["geometry"])
                designs.add((f["glyph"].shape, tuple(sorted(f["payload"])), f["kind"], f["center"]))
                kernels.add(tuple(sorted(f["kernel"])))
                roles = {}
                motifs.add((f["ordinary"].shape, tuple(roles.setdefault(int(c), len(roles)) for c in f["ordinary"].flat)))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                kinds.add(f["kind"])
                counts.add(len(f["seeds"]))
                overlaps += _witnesses(f)["overlap"]
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "Prediction mismatch")
                _require(np.array_equal(before, pair["input"]), "Solver mutated input")
    _require(kinds == {"H", "V", "P"} and counts == {1, 2, 3, 4}, "Missing modes or counts")
    _require(len(shapes) >= 30 and len(designs) >= 80 and len(kernels) >= 60,
             "Insufficient structural variation")
    _require(len(motifs) >= 80 and len(palettes) >= 40 and overlaps >= 42, "Insufficient presentation variation")
    # Hand-calculated regressions, including a union at coarse site (2,3).
    ordinary = np.array([[3, 4, 3], [4, 3, 4], [3, 4, 3]], dtype=np.int8)
    glyph = np.array([[9, 9, 9], [6, 3, 3], [6, 3, 6]], dtype=np.int8)
    geometry = (3, 3, 1, 7, 7)
    seeds = {(4, 2), (4, 4)}
    active = seeds | {(3, 1), (2, 1), (2, 3), (3, 3), (2, 5)}
    inp = _paint(geometry, 9, ordinary, glyph, seeds)
    expected = _paint(geometry, 9, ordinary, glyph, active)
    _require(np.array_equal(solve(inp), expected), "Hand-calculated horizontal union failed")
    _require(np.array_equal(solve(inp.T), expected.T), "Transposed vertical case failed")
    glyph_p = np.array([[9, 6, 3], [6, 3, 3], [3, 6, 3]], dtype=np.int8)
    p_seeds = {(4, 4)}
    p_active = p_seeds | {(4, 3), (3, 4), (2, 3)}
    p_input = _paint(geometry, 9, ordinary, glyph_p, p_seeds)
    _require(np.array_equal(solve(p_input), _paint(geometry, 9, ordinary, glyph_p, p_active)),
             "Hand-calculated half-turn with aligned pixels failed")
    # Same anchors, one ink pixel removed: coarse additions must change (B witness).
    smaller = glyph.copy()
    smaller[2, 2] = 3
    small_input = _paint(geometry, 9, ordinary, smaller, seeds)
    small_active = seeds | {(3, 1), (2, 1), (3, 3), (2, 3)}
    _require(np.array_equal(solve(small_input), _paint(geometry, 9, ordinary, smaller, small_active)),
             "Fine-scale ink did not control coarse occupancy")
    _require(_decode(inp)["active"] != _decode(small_input)["active"], "B ablation is vacuous")
    # Counts, determinism and damaged labels are tested without retaining a seed corpus.
    a, b = generate_puzzle(173, 10, 4), generate_puzzle(173, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "Nondeterministic puzzle")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(17000+nt*10+ns, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "Count mismatch")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    a["test"][0]["output"][0, 0] = (int(a["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(a)
    except ValueError:
        pass
    else:
        raise AssertionError("Damaged divider accepted")
    return {"seeds": 42, "pairs": 210, "geometries": len(shapes), "designs": len(designs),
            "kernels": len(kernels), "ordinary_motifs": len(motifs), "palettes": len(palettes),
            "overlap_pairs": overlaps}
