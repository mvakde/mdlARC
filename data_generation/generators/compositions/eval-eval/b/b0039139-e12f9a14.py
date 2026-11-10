"""Finite stencil loom. Standard library and NumPy only; no file I/O at import.

The input's lattice and notched frames encode everything used by solve().
Diagnostics are aggregate, and self_test() retains only one puzzle at a time.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_VARIANTS = ("area", "unit", "no_collision", "infinite", "equal_clock",
             "reset", "no_crop", "no_recolor") + tuple("transform" + str(k) for k in range(1, 8))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "illegal grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "illegal color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    group.append(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _crop(mask):
    pts = np.argwhere(mask)
    _require(len(pts) > 0, "empty stencil")
    lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _transform(a, k):
    return np.rot90(a[:, ::-1] if k >= 4 else a, k % 4).copy()


def _inside(p, d):
    return 0 <= p[0] < d["rows"] and 0 <= p[1] < d["cols"]


def _offset(p, v, n=1):
    return p[0] + n*v[0], p[1] + n*v[1]


def _input_from_data(d):
    h, w = d["h"], d["w"]
    a = np.full((d["rows"]*(h+1)+1, d["cols"]*(w+1)+1), d["bg"], dtype=np.int8)
    a[::h+1, :] = d["guide"]
    a[:, ::w+1] = d["guide"]
    for s in d["sources"]:
        r, c = s["pos"][0]*(h+1), s["pos"][1]*(w+1)
        a[r, c:c+w+2] = a[r+h+1, c:c+w+2] = s["color"]
        a[r:r+h+2, c] = a[r:r+h+2, c+w+1] = s["color"]
        view = a[r+1:r+h+1, c+1:c+w+1]
        view[s["mask"]] = d["ink"]
        nr, nc = s["notch"]
        a[r+nr, c+nc] = d["bg"]
    return a


def _decode(value):
    """Enumerate lattice parses, using input geometry alone, never labels."""
    a = _grid(value)
    bg, guide = int(a[1, 1]) if min(a.shape) > 1 else -1, int(a[0, 0])
    _require(bg != guide, "empty corner does not identify background")
    candidates = []
    for h in (3, 4):
        if (a.shape[0]-1) % (h+1):
            continue
        rows = (a.shape[0]-1)//(h+1)
        if not 4 <= rows <= 7:
            continue
        for w in (3, 4):
            if (a.shape[1]-1) % (w+1):
                continue
            cols = (a.shape[1]-1)//(w+1)
            if not 4 <= cols <= 7:
                continue
            try:
                sources, inks = [], set()
                for rr in range(rows):
                    for cc in range(cols):
                        r, c = rr*(h+1), cc*(w+1)
                        core = a[r+1:r+h+1, c+1:c+w+1]
                        ink = set(map(int, np.unique(core))) - {bg}
                        if not ink:
                            continue
                        _require(len(ink) == 1 and guide not in ink, "mixed core")
                        inks |= ink
                        ring = [(x, y) for x in range(h+2) for y in range(w+2)
                                if x in (0, h+1) or y in (0, w+1)]
                        gaps = [(x, y) for x, y in ring if int(a[r+x, c+y]) == bg]
                        colors = {int(a[r+x, c+y]) for x, y in ring} - {bg}
                        _require(len(gaps) == len(colors) == 1, "invalid notched frame")
                        color = next(iter(colors))
                        _require(color not in {bg, guide} | ink, "frame color aliases a role")
                        nr, nc = gaps[0]
                        _require(not (nr in (0, h+1) and nc in (0, w+1)), "corner notch")
                        direction = ((-1, 0) if nr == 0 else (1, 0) if nr == h+1 else
                                     (0, -1) if nc == 0 else (0, 1))
                        mask = core != bg
                        n = len(_components(mask))
                        _require(1 <= n <= 4 and n == len(_components(mask, True)),
                                 "ambiguous or illegal component count")
                        sources.append({"pos": (rr, cc), "dir": direction,
                                        "mask": mask.copy(), "tile": _crop(mask), "count": n,
                                        "color": color, "notch": gaps[0]})
                _require(len(sources) == 2 and len(inks) == 1, "need two common-ink sources")
                _require(sources[0]["color"] != sources[1]["color"], "payload colors must differ")
                p, q = (s["pos"] for s in sources)
                _require(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 2, "touching source frames")
                corners = {(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)}
                _require(p not in corners and q not in corners, "corner source")
                d = {"h": h, "w": w, "rows": rows, "cols": cols, "bg": bg,
                     "guide": guide, "ink": next(iter(inks)), "sources": sources}
                _require(len({bg, guide, d["ink"], *(s["color"] for s in sources)}) == 5,
                         "color roles must be distinct")
                _require(np.array_equal(_input_from_data(d), a), "unexplained input cells")
                candidates.append(d)
            except AssertionError:
                continue
    _require(len(candidates) == 1, "input must have a unique lattice/frame parse")
    return candidates[0]


def _straight(s, n, d):
    points = []
    for k in range(1, n+1):
        p = _offset(s["pos"], s["dir"], k)
        if not _inside(p, d):
            break
        points.append(p)
    return points


def _routes(d, variant="normal"):
    """Spatial finite-path construction. Source cells are distance zero."""
    sources = d["sources"]
    budgets = [int(s["mask"].sum()) if variant == "area" else
               1 if variant == "unit" else s["count"] for s in sources]
    paths = [_straight(s, n, d) for s, n in zip(sources, budgets)]
    forecast = ([_straight(s, 2*max(d["rows"], d["cols"]), d) for s in sources]
                if variant == "infinite" else paths)
    common = set(forecast[0]) & set(forecast[1])
    u, v = (s["dir"] for s in sources)
    perpendicular = u[0]*v[0] + u[1]*v[1] == 0
    if variant == "normal" and common:
        _require(perpendicular and len(common) == 1, "unsupported straight overlap")
    event = None
    if common and perpendicular and variant != "no_collision":
        _require(len(common) == 1, "ambiguous crossing")
        cross = next(iter(common))
        distances = [p.index(cross)+1 for p in forecast]
        if variant != "equal_clock" or distances[0] == distances[1]:
            if variant == "normal":
                _require(min(distances) >= 2, "encounter too close to a source")
            out_dir = (u[0]+v[0], u[1]+v[1])
            result = []
            for s, n, distance in zip(sources, budgets, distances):
                cut = distance-1
                prefix = _straight(s, min(n, cut), d)
                last = _offset(s["pos"], s["dir"], cut)
                remaining = n if variant == "reset" else max(0, n-cut)
                for k in range(1, remaining+1):
                    p = _offset(last, out_dir, k)
                    if not _inside(p, d):
                        break
                    prefix.append(p)
                result.append(prefix)
            paths = result
            event = {"cross": cross, "distances": tuple(distances), "out_dir": out_dir}
    if variant == "normal":
        occupied = {s["pos"] for s in sources}
        for path in paths:
            _require(path, "a source prints no visible copy")
            for p in path:
                _require(_inside(p, d) and p not in occupied, "route overlap or source overwrite")
                occupied.add(p)
    return paths, event


def _render(d, paths, variant="normal"):
    a = _input_from_data(d)
    for s, path in zip(d["sources"], paths):
        tile = s["mask"] if variant == "no_crop" else s["tile"]
        if variant.startswith("transform"):
            tile = _transform(tile, int(variant[9:]))
        if tile.shape[0] > d["h"] or tile.shape[1] > d["w"]:
            return None
        color = d["ink"] if variant == "no_recolor" else s["color"]
        for rr, cc in path:
            r, c = rr*(d["h"]+1)+1, cc*(d["w"]+1)+1
            a[r:r+tile.shape[0], c:c+tile.shape[1]][tile] = color
    return a


def _alternative(d, variant):
    return _render(d, _routes(d, variant)[0], variant)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, counts, motifs and routes entirely from this input."""
    d = _decode(input_grid)
    paths, _ = _routes(d)
    result = _render(d, paths)
    _require(result is not None, "tile does not fit")
    return result


def _motif(rng, h, w, count, area=None, asymmetric=False, leading=False, small=False):
    """Separated random component growth; no required corner path or density."""
    ah, aw = (3, 3) if small else (h, w)
    mask = None
    for _ in range(48):
        labels = np.zeros((ah, aw), dtype=np.int8)
        placed = 0
        for flat in rng.permutation(ah*aw):
            r, c = divmod(int(flat), aw)
            if not labels[max(0, r-1):r+2, max(0, c-1):c+2].any():
                placed += 1
                labels[r, c] = placed
                if placed == count:
                    break
        if placed != count:
            continue
        target = area if area is not None else count + int(rng.integers(0, 6))
        while np.count_nonzero(labels) < target:
            choices = []
            for r, c in np.argwhere(labels == 0):
                near = set(map(int, labels[max(0, r-1):r+2, max(0, c-1):c+2].ravel())) - {0}
                edge = {int(labels[x, y]) for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                        if 0 <= x < ah and 0 <= y < aw} - {0}
                if len(near) == 1 and edge:
                    choices.append((int(r), int(c), next(iter(near))))
            if not choices:
                break
            r, c, label = choices[int(rng.integers(len(choices)))]
            labels[r, c] = label
        trial = _crop(labels != 0)
        if area is not None and int(trial.sum()) != area:
            continue
        if asymmetric and any(np.array_equal(trial, _transform(trial, k)) for k in range(1, 8)):
            continue
        if leading and trial.shape == (h, w):
            continue
        mask = trial
        break
    if mask is None:
        # Exact small, valid witnesses, also covering rare growth exhaustion.
        patterns = {1: ("100", "111"), 2: ("111", "000", "100"),
                    3: (("101", "000", "111") if area == 5 else ("110", "000", "101")),
                    4: ("101", "000", "101")}
        mask = np.array([[c == "1" for c in row] for row in patterns[count]], dtype=bool)
    _require(len(_components(mask)) == len(_components(mask, True)) == count, "motif growth merged components")
    _require(area is None or int(mask.sum()) == area, "motif area budget failed")
    choices = [(r, c) for r in range(h-mask.shape[0]+1) for c in range(w-mask.shape[1]+1)
               if not leading or r+c > 0]
    _require(choices, "leading padding does not fit")
    r, c = choices[int(rng.integers(len(choices)))]
    padded = np.zeros((h, w), dtype=bool)
    padded[r:r+mask.shape[0], c:c+mask.shape[1]] = mask
    return padded


def _turn_geometry(rng, h, w, rows, cols, specs):
    specs = [dict(s) for s in specs]
    for _ in range(int(rng.integers(4))):
        for s in specs:
            r, c = s["pos"]
            dr, dc = s["dir"]
            s["pos"], s["dir"] = (cols-1-c, r), (-dc, dr)
        h, w, rows, cols = w, h, cols, rows
    if rng.random() < .5:
        for s in specs:
            r, c = s["pos"]
            dr, dc = s["dir"]
            s["pos"], s["dir"] = (r, cols-1-c), (dr, -dc)
    return h, w, rows, cols, specs


def _notch(rng, h, w, direction):
    if direction == (-1, 0):
        return 0, int(rng.integers(1, w+1))
    if direction == (1, 0):
        return h+1, int(rng.integers(1, w+1))
    if direction == (0, -1):
        return int(rng.integers(1, h+1)), 0
    return int(rng.integers(1, h+1)), w+1


def _candidate(rng, mode, palette):
    h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
    if mode == "teach_hit":
        h, w = 3, 4
        rows, cols = int(rng.integers(6, 8)), 5
        r, c = int(rng.integers(2, rows-3)), int(rng.integers(2, cols-1))
        specs = [{"pos": (r, c-2), "dir": (0, 1), "count": 3, "area": 4, "asym": True, "leading": True},
                 {"pos": (r+3, c), "dir": (-1, 0), "count": 3, "area": 5}]
    elif mode in ("teach_miss", "test_hit"):
        rows = int(rng.integers(5, min(7, 29//(h+1))+1))
        cols = int(rng.integers(5, min(7, 29//(w+1))+1))
        r = int(rng.integers(2 if mode == "test_hit" else 1, rows-2))
        distance = 3 if mode == "test_hit" else 2
        c = int(rng.integers(distance, cols-1))
        specs = [{"pos": (r, c-distance), "dir": (0, 1), "count": 4 if mode == "test_hit" else 1},
                 {"pos": (r+2, c), "dir": (-1, 0), "count": int(rng.integers(2, 4)) if mode == "test_hit" else 2}]
        if mode == "teach_miss":
            for s in specs:
                s["area"] = 4
    else:
        rows = int(rng.integers(4, min(7, 29//(h+1))+1))
        cols = int(rng.integers(4, min(7, 29//(w+1))+1))
        corners = {(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)}
        places = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in corners]
        chosen = rng.choice(len(places), 2, replace=False)
        specs = [{"pos": places[int(j)], "dir": _DIRS[int(rng.integers(4))],
                  "count": 4 if mode == "random_test" and i == 0 else
                  int(rng.integers(1, 5 if mode == "random_test" else 4))}
                 for i, j in enumerate(chosen)]
        p, q = (s["pos"] for s in specs)
        _require(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 2, "adjacent random sources")
    h, w, rows, cols, specs = _turn_geometry(rng, h, w, rows, cols, specs)
    d = dict(zip(("bg", "guide", "ink"), palette[:3]))
    d.update(h=h, w=w, rows=rows, cols=cols, sources=[])
    for i, s in enumerate(specs):
        mask = _motif(rng, h, w, s["count"], s.get("area"), s.get("asym", False),
                      s.get("leading", False), mode == "teach_hit")
        d["sources"].append({"pos": s["pos"], "dir": s["dir"], "count": s["count"],
                             "mask": mask, "tile": _crop(mask), "color": palette[3+i],
                             "notch": _notch(rng, h, w, s["dir"])})
    paths, event = _routes(d)
    if mode == "teach_hit":
        _require(event is not None and sorted(event["distances"]) == [2, 3], "lost encounter witness")
        _require(all(len(p) == 3 for p in paths), "clipped encounter witness")
    elif mode == "teach_miss":
        _require(event is None and sorted(map(len, paths)) == [1, 2], "lost missed-encounter witness")
    elif mode == "test_hit":
        _require(event is not None and sorted(event["distances"]) == [2, 3], "lost shifted encounter")
        _require(len(paths[0]) == 4, "four-copy extrapolation must be visible")
    output = _render(d, paths)
    if mode == "teach_hit":
        # All correct copies can fit even when a longer competing route would
        # immediately leave the board. Demand visible evidence against that
        # alternative in the returned scene, rather than assuming it.
        for variant in ("area", "reset", "no_crop"):
            _require(not np.array_equal(_alternative(d, variant), output),
                     "encounter witness hides alternative: " + variant)
    return {"input": _input_from_data(d), "output": output}


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = tuple(map(int, rng.choice(10, 5, replace=False)))
    second = base[1:] + base[:1]  # Every role changes in the other mandatory witness.
    palettes = [base, second, tuple(map(int, rng.choice(10, 5, replace=False)))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for i in range(amount):
            mode = (("teach_hit", "teach_miss")[i] if split == "train" and i < 2 else
                    "test_hit" if split == "test" and i == 0 else "random_" + split)
            palette = palettes[i] if split == "train" and i < 2 else palettes[int(rng.integers(3))]
            if i >= 2 and rng.random() < .35:
                palette = tuple(map(int, rng.choice(10, 5, replace=False)))
            for attempt in range(96):
                try:
                    # The last attempt is constructive, with the same rule.
                    actual_mode = mode if attempt < 95 else ("teach_miss" if split == "train" else "test_hit")
                    pair = _candidate(rng, actual_mode, palette)
                    d = _decode(pair["input"])
                    _require(np.array_equal(solve(pair["input"]), pair["output"]), "construction/parse mismatch")
                    _require(_key(pair["input"]) not in seen, "duplicate input")
                    if split == "test":
                        _require(any(s["count"] == 4 for s in d["sources"]), "test needs new budget")
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded pair construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _potential(d):
    paths = [_straight(s, 2*max(d["rows"], d["cols"]), d) for s in d["sources"]]
    common = set(paths[0]) & set(paths[1])
    u, v = (s["dir"] for s in d["sources"])
    if len(common) == 1 and u[0]*v[0] + u[1]*v[1] == 0:
        cross = next(iter(common))
        return tuple(p.index(cross)+1 for p in paths)
    return None


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "output dimensions changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            d = _decode(a)
            paths, event = _routes(d)
            _require(np.array_equal(_render(d, paths), b), "incorrect output")
            _require(np.any(a != b), "identity pair")
            _require(np.array_equal(a[a != d["bg"]], b[a != d["bg"]]), "original ink or lines changed")
            for s, path in zip(d["sources"], paths):
                _require(len(path) <= s["count"], "budget overspent")
                _require(len(path) == len(set(path)), "self-overlapping route")
            if event is not None:
                _require(event["cross"] not in {p for path in paths for p in path}, "crossing was painted")
            if split == "train":
                _require(all(s["count"] <= 3 for s in d["sources"]), "training budget exceeds curriculum")
                training.append((d, b, paths, event))
            else:
                _require(any(s["count"] == 4 for s in d["sources"]), "test does not extrapolate count")
                tests.append((d, b))
    _require({1, 2, 3} <= {s["count"] for d, _, _, _ in training for s in d["sources"]}, "incomplete count teaching")
    _require(len({(d["bg"], d["guide"], d["ink"]) for d, _, _, _ in training}) >= 2,
             "training needs role reassignment")
    hit = any(e is not None and sorted(e["distances"]) == [2, 3]
              and all(s["count"] == 3 for s in d["sources"])
              and sorted(int(s["mask"].sum()) for s in d["sources"]) == [4, 5]
              and all(len(p) == 3 for p in paths) for d, _, paths, e in training)
    miss = any(e is None and _potential(d) == (2, 2)
               and sorted(s["count"] for s in d["sources"]) == [1, 2]
               and all(int(s["mask"].sum()) == 4 for s in d["sources"])
               and sorted(map(len, paths)) == [1, 2] for d, _, paths, e in training)
    _require(hit and miss, "missing finite collision or missed-encounter witness")
    # The fixed witnesses must actually change the competing answers, not
    # merely advertise construction-time parameters.
    for variant in _VARIANTS:
        fits = all(np.array_equal(_alternative(d, variant), b) for d, b, _, _ in training)
        if fits:
            _require(all(np.array_equal(_alternative(d, variant), b) for d, b in tests),
                     "training leaves test ambiguity: " + variant)
        if variant in ("area", "unit", "no_collision", "infinite", "equal_clock", "reset", "no_crop", "no_recolor"):
            _require(not fits, "training lacks a distinguishing witness: " + variant)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed audit; return compact aggregate measurements only."""
    signatures, shapes, masks, layouts, cell_shapes, headings, turns = set(), set(), set(), set(), set(), set(), set()
    counts, areas = set(), set()
    pairs = hits = misses = clipped = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed in (0, 17, 41):
            _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                d = _decode(pair["input"])
                paths, e = _routes(d)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
                shapes.add(pair["input"].shape)
                cell_shapes.add((d["h"], d["w"]))
                layouts.add((d["rows"], d["cols"], tuple((s["pos"], s["dir"], s["count"]) for s in d["sources"])))
                hits += e is not None
                misses += e is None and _potential(d) is not None
                if e:
                    turns.add(e["out_dir"])
                for s, path in zip(d["sources"], paths):
                    masks.add(_key(s["tile"]))
                    headings.add(s["dir"])
                    counts.add(s["count"])
                    areas.add(int(s["mask"].sum()))
                    clipped += len(path) < s["count"]
    _require(len(signatures) == 42, "cross-seed identity")
    _require(len(shapes) >= 10 and len(layouts) >= 100, "geometry diversity collapsed")
    _require(len(masks) >= 40 and len(areas) >= 5, "motif/component-size diversity collapsed")
    _require(cell_shapes == {(3, 3), (3, 4), (4, 3), (4, 4)}, "cell proportions collapsed")
    _require(headings == set(_DIRS) and len(turns) == 4, "direction diversity collapsed")
    _require(counts == {1, 2, 3, 4} and hits >= 42 and misses >= 42 and clipped > 0, "finite-route coverage collapsed")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(7341, a, b)
        _require((len(p["train"]), len(p["test"])) == (a, b), "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit route fixtures check distance-independent encounters and residual
    # budgets without obtaining their expected paths from the renderer.
    fixture = {"rows": 7, "cols": 7, "sources": [
        {"pos": (2, 0), "dir": (0, 1), "count": 3, "mask": np.ones((1, 3), bool)},
        {"pos": (5, 2), "dir": (-1, 0), "count": 3, "mask": np.ones((1, 3), bool)}]}
    paths, event = _routes(fixture)
    _require(paths == [[(2, 1), (1, 2), (0, 3)], [(4, 2), (3, 2), (2, 3)]],
             "unequal-distance turn fixture failed")
    _require(event["distances"] == (2, 3), "spatial distances changed")
    fixture["sources"][0]["count"] = 1
    fixture["sources"][1].update(pos=(4, 2), count=2)
    paths, event = _routes(fixture)
    _require(event is None and paths == [[(2, 1)], [(3, 2), (2, 2)]],
             "finite reach must suppress the encounter")
    p = generate_puzzle(0)
    palette = (np.arange(10, dtype=np.int8)+3) % 10
    for pair in p["train"]:
        _require(np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]]),
                 "literal colors changed the rule")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][1, 1] = (int(bad["test"][0]["output"][1, 1])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "logical_layouts": len(layouts), "binary_motifs": len(masks),
            "cell_shapes": sorted(cell_shapes), "ink_areas": sorted(areas),
            "encounters": hits, "missed_encounters": misses, "clipped_routes": clipped,
            "independent_solver": False, "blind_prediction": False}
