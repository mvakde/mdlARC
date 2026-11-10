"""Complete landmark drawings and their companions, then contract equal line runs.

No parent module, file, random state, or generated label is used by solve().
The intermediate drawing is used in memory only; no execution traces are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 156171427, 415956069, 510353461, 593602444, 568453619,
              995172128, 4784736, 185071174, 955345690, 173338901, 207765360,
              557769012, 44390960, 29918051, 610047508, 895559515, 17648319,
              317153698, 323106933, 21499705, 550651653, 707414999, 915163594,
              148438078, 244135036, 171768809, 572872009, 336186187, 501416671,
              615341062, 639095228, 272870437, 725520890, 869273795, 233766422,
              604242806, 568316255, 326189514, 730635008, 533442960, 734425648)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid shape outside 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid color outside 0..9")
    return a.astype(np.int8, copy=True)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    if not n or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return ()
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return tuple((a[0] + k * sr, a[1] + k * sc) for k in range(n + 1))


def _candidate_edges(points):
    occupied = set(points)
    result = []
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            raster = _line(a, points[j])
            if raster and not occupied.intersection(raster[1:-1]):
                result.append((i, j))
    return result


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or
           max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _topology(points):
    """Unique noncrossing spanning cycle first, otherwise unique spanning path.

    Edge-set signatures identify reversals/rotations as the same drawing.
    Continuous intersections also exclude crossings between diagonal pixels.
    """
    n = len(points)
    _require(2 <= n <= 12, "source needs 2..12 landmarks")
    edges = _candidate_edges(points)
    adjacency = [[] for _ in points]
    conflicts = [set() for _ in edges]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
        for l, (u, v) in enumerate(edges[:k]):
            if not {i, j}.intersection((u, v)) and _intersect(points[i], points[j], points[u], points[v]):
                conflicts[k].add(l)
                conflicts[l].add(k)
    _require(all(adjacency), "disconnected source landmarks")

    def search(closed):
        solutions, work = {}, 0

        def walk(order, used, selected):
            nonlocal work
            work += 1
            _require(work <= 40000, "ambiguous/excessive topology search")
            if len(solutions) > 1:
                return
            if len(order) == n:
                chosen = selected
                if closed:
                    closing = [k for v, k in adjacency[order[-1]] if v == order[0]]
                    if not closing or conflicts[closing[0]].intersection(selected):
                        return
                    chosen = selected + (closing[0],)
                solutions[tuple(sorted(chosen))] = order
                return
            for v, k in adjacency[order[-1]]:
                if v not in used and not conflicts[k].intersection(selected):
                    walk(order + (v,), used | {v}, selected + (k,))

        for start in ((0,) if closed else range(n)):
            walk((start,), {start}, ())
            if len(solutions) > 1:
                break
        return solutions

    cycles = search(True) if n >= 3 else {}
    _require(len(cycles) <= 1, "competing full contours")
    solutions = cycles or search(False)
    _require(len(solutions) == 1, "no unique full chain")
    signature, order = next(iter(solutions.items()))
    selected = [edges[k] for k in signature]
    lengths = [len(_line(points[i], points[j])) - 1 for i, j in selected]
    _require(min(lengths) >= 2, "family uses nontrivial links only")
    pixels = set().union(*(set(_line(points[i], points[j])) for i, j in selected))
    turns = [_cross(points[order[k - 1]], points[order[k]], points[order[(k + 1) % n]])
             for k in range(n)] if cycles else []
    return {"points": points, "edges": selected, "pixels": pixels,
            "closed": bool(cycles), "lengths": lengths,
            "diagonal": any(points[i][0] != points[j][0] and points[i][1] != points[j][1]
                            for i, j in selected),
            "concave": any(t > 0 for t in turns) and any(t < 0 for t in turns),
            "chords": len(edges) - len(selected)}


def _complete(grid):
    """Recover source groups, topology and companion displacements from input."""
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    _require(int(max(counts)) * 2 > grid.size, "background must be a strict majority")
    groups = {int(c): [tuple(map(int, p)) for p in np.argwhere(grid == c)]
              for c in values if c != bg}
    sources = []
    for c, points in groups.items():
        if len(points) >= 2:
            source = _topology(points)
            source["color"] = c
            sources.append(source)
    _require(1 <= len(sources) <= 3, "need 1..3 sources")
    source_pixels = set()
    for s in sources:
        _require(not source_pixels.intersection(s["pixels"]), "source strokes collide")
        _require(any(grid[p] == bg for p in s["pixels"]), "source already complete")
        source_pixels.update(s["pixels"])
    markers, marked = [], set()
    for c, points in groups.items():
        if len(points) != 1:
            continue
        p = points[0]
        matches = [(i, q) for i, s in enumerate(sources) for q in s["points"]
                   if max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1]
        _require(len(matches) == 1, "singleton must identify exactly one source vertex")
        i, q = matches[0]
        _require(i not in marked and p not in source_pixels, "marker/source conflict")
        marked.add(i)
        d = p[0] - q[0], p[1] - q[1]
        copy_pixels = {(r + d[0], c0 + d[1]) for r, c0 in sources[i]["pixels"]}
        _require(all(0 <= r < grid.shape[0] and 0 <= c0 < grid.shape[1] for r, c0 in copy_pixels),
                 "translated companion outside canvas")
        markers.append({"color": c, "point": p, "source": i, "offset": d, "pixels": copy_pixels})
    _require(markers, "at least one companion required")
    completed, source_only = grid.copy(), grid.copy()
    claimed = set()
    for m in markers:
        for p in m["pixels"]:
            if grid[p] == bg and p not in source_pixels:
                _require(p not in claimed, "ambiguous companion overlap")
                completed[p] = m["color"]
                claimed.add(p)
    copy_only = completed.copy()
    for s in sources:
        for p in s["pixels"]:
            if grid[p] == bg:
                completed[p] = source_only[p] = s["color"]
    _require(np.array_equal(completed[grid != bg], grid[grid != bg]), "changed an existing landmark")
    return completed, {"background": bg, "sources": sources, "markers": markers,
                       "source_only": source_only, "copy_only": copy_only, "copy_added": len(claimed),
                       "overlap": sum(len(m["pixels"] & source_pixels) for m in markers),
                       "new_overlap": sum(grid[p] == bg for m in markers
                                          for p in m["pixels"] & source_pixels)}


def _keep_masks(a):
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return rows, cols


def _crop(a, bg):
    points = np.argwhere(a != bg)
    _require(bool(len(points)), "empty mosaic")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _contract(a, bg, axes=(True, True)):
    """Masks are both computed on the SAME complete grid, before deletion."""
    rows, cols = _keep_masks(a)
    if not axes[0]:
        rows[:] = True
    if not axes[1]:
        cols[:] = True
    return _crop(a[np.ix_(rows, cols)], bg)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    completed, info = _complete(grid)
    return _contract(completed, info["background"])


def _global_unique(a, bg):
    # Deliberately wrong alternative: remove separated recurrences as well.
    rows, cols = [], []
    seen = set()
    for i, row in enumerate(a):
        key = row.tobytes()
        if key not in seen:
            rows.append(i)
            seen.add(key)
    seen.clear()
    for i, col in enumerate(a.T):
        key = col.tobytes()
        if key not in seen:
            cols.append(i)
            seen.add(key)
    return _crop(a[np.ix_(rows, cols)], bg)


def _all_alignment_completion(inp, info):
    """Wrong alternative: draw every eligible chord, including in companions."""
    bg = info["background"]
    sources = []
    for s in info["sources"]:
        points = s["points"]
        pixels = set().union(*(set(_line(points[i], points[j]))
                              for i, j in _candidate_edges(points)))
        sources.append(pixels)
    all_source = set().union(*sources)
    out = inp.copy()
    for m in info["markers"]:
        dr, dc = m["offset"]
        for r, c in sources[m["source"]]:
            p = r + dr, c + dc
            if inp[p] == bg and p not in all_source:
                out[p] = m["color"]
    for s, pixels in zip(info["sources"], sources):
        for p in pixels:
            if inp[p] == bg:
                out[p] = s["color"]
    return out


def _global_interaction(completed, info):
    """A repeated occupied local cross-section is distinct on the full canvas."""
    bg = info["background"]
    rows, cols = _keep_masks(completed)
    for index, s in enumerate(info["sources"]):
        local = np.full_like(completed, bg)
        for m in info["markers"]:
            if m["source"] == index:
                for p in m["pixels"] - s["pixels"]:
                    local[p] = m["color"]
        for p in s["pixels"]:
            local[p] = s["color"]
        lr, lc = _keep_masks(local)
        row_witness = ~lr[1:] & rows[1:] & np.any(local[1:] != bg, axis=1)
        col_witness = ~lc[1:] & cols[1:] & np.any(local[:, 1:] != bg, axis=0)
        if row_witness.any() or col_witness.any():
            return True
    return False


def _features(inp):
    completed, info = _complete(inp)
    bg = info["background"]
    out = _contract(completed, bg)
    cropped = _crop(completed, bg)
    rows, cols = _keep_masks(cropped)
    raw_rows, raw_cols = _keep_masks(inp)
    after_rows, after_cols = _keep_masks(completed)
    # Holding raw equality decisions fixed while applying the completed colors
    # tests the causal dependency, separately from merely comparing raw pixels.
    premature = _crop(completed[np.ix_(raw_rows, raw_cols)], bg)
    differs = lambda a: not np.array_equal(a, out)
    info.update({"completed": completed, "output": out,
                 "row_removed": cropped.shape[0] - int(rows.sum()),
                 "col_removed": cropped.shape[1] - int(cols.sum()),
                 "raw_diff": differs(_contract(inp, bg)),
                 "copy_diff": differs(_contract(info["source_only"], bg)),
                 "source_diff": differs(_contract(info["copy_only"], bg)),
                 "no_compression_diff": differs(cropped),
                 "row_only_diff": differs(_contract(completed, bg, (True, False))),
                 "col_only_diff": differs(_contract(completed, bg, (False, True))),
                 "global_diff": differs(_global_unique(completed, bg)),
                 "chord_diff": differs(_contract(_all_alignment_completion(inp, info), bg)),
                 "global_interaction": _global_interaction(completed, info),
                 "comparison_changed": not np.array_equal(raw_rows, after_rows) or
                                       not np.array_equal(raw_cols, after_cols),
                 "premature_diff": differs(premature),
                 "singleton_row": bool(np.any(np.r_[True, rows[1:]] & np.r_[rows[1:], True])),
                 "singleton_col": bool(np.any(np.r_[True, cols[1:]] & np.r_[cols[1:], True]))})
    return info


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _shape(rng, family, small=False, chord=False):
    """Procedural landmarks, checked against visible-input topology inference."""
    for _ in range(100):
        upper = 7 if small else 11
        h, w = (int(rng.integers(4, upper)) for _ in range(2))
        a, b, d = (int(rng.integers(2, 5)) for _ in range(3))
        if family == "rectangle":
            p, closed = [(0, 0), (0, w), (h, w), (h, 0)], True
            if chord:
                if rng.random() < .5:
                    w = h
                    p = [(0, 0), (0, w), (h, w), (h, 0)]
                else:
                    # Collinear waypoints expose an interior alignment at any
                    # aspect ratio; the unique perimeter must reject that chord.
                    k = int(rng.integers(2, h - 1))
                    p = [(0, 0), (0, w), (k, w), (h, w), (h, 0), (k, 0)]
        elif family == "bevel":
            p, closed = [(0, 0), (0, w), (d, w + d), (d + h, w + d)], False
        elif family == "stair":
            p, closed = [(0, 0), (0, w), (a, w), (a, w + b), (a + h, w + b)], False
        elif family == "notch":
            h, w = h + 4, w + 6
            p = [(0, 0), (0, w), (h, w), (h, w - 2),
                 (h - d, w - 2), (h - d, 2), (h, 2), (h, 0)]
            closed = True
        elif family == "clipped":
            p = [(0, 0), (0, w), (d, w + d), (h + d, w + d),
                 (h + d, b), (h + d - b, 0)]
            closed = True
        elif family == "L":
            p, closed = [(0, 0), (0, w), (h, w)], False
        elif family == "segment":
            p, closed = [(0, 0), (0, h + w)], False
        elif family == "diamond":
            k = a + 1
            p, closed = [(0, k), (k, 2 * k), (2 * k, k), (k, 0)], True
        else:
            raise ValueError("unknown shape family")
        try:
            info = _topology(p)
        except ValueError:
            continue
        intended = {(i, i + 1) for i in range(len(p) - 1)}
        if closed:
            intended.add((0, len(p) - 1))
        if set(info["edges"]) == intended and max(max(v[k] for v in p) - min(v[k] for v in p) for k in (0, 1)) <= 22:
            return p, closed
    # Preserve requested class; failure is explicit, not a silent shape fallback.
    raise ValueError("shape sampling exhausted: " + family)


def _spec(rng, family, marked=True, axial=False, small=False, chord=False):
    p, closed = _shape(rng, family, small, chord)
    edges = list(zip(p, p[1:])) + ([(p[-1], p[0])] if closed else [])
    source = set().union(*(set(_line(a, b)) for a, b in edges))
    choices = []
    if marked:
        for dr, dc in OFFSETS:
            if axial and dr * dc:
                continue
            copy = {(r + dr, c + dc) for r, c in source}
            for v in p:
                marker = v[0] + dr, v[1] + dc
                if marker not in source and sum(max(abs(marker[0] - q[0]), abs(marker[1] - q[1])) == 1 for q in p) == 1:
                    if len(copy - source - {marker}) >= 2:
                        choices.append((marker, copy))
        _require(bool(choices), "no visible companion anchor")
        marker, copy = choices[int(rng.integers(len(choices)))]
    else:
        marker, copy = None, set()
    code = int(rng.integers(8))
    return {"vertices": _transform(p, code), "source": set(_transform(source, code)),
            "copy": set(_transform(copy, code)),
            "marker": _transform([marker], code)[0] if marker is not None else None}


def _shift(spec, dr, dc):
    return {"vertices": [(r + dr, c + dc) for r, c in spec["vertices"]],
            "source": {(r + dr, c + dc) for r, c in spec["source"]},
            "copy": {(r + dr, c + dc) for r, c in spec["copy"]},
            "marker": (spec["marker"][0] + dr, spec["marker"][1] + dc)
                      if spec["marker"] is not None else None}


def _render(rng, specs):
    placed, blocked = [], set()
    for spec in specs:
        foot = spec["source"] | spec["copy"]
        r0, c0 = min(r for r, c in foot), min(c for r, c in foot)
        h, w = max(r for r, c in foot) - r0 + 1, max(c for r, c in foot) - c0 + 1
        _require(h <= 30 and w <= 30, "footprint too large")
        # Sparse random placement permits nesting whenever footprints do not touch.
        options = [(r - r0, c - c0) for r in range(31 - h) for c in range(31 - w)]
        rng.shuffle(options)
        for dr, dc in options:
            moved = {(r + dr, c + dc) for r, c in foot}
            if not moved.intersection(blocked):
                placed.append(_shift(spec, dr, dc))
                blocked.update((r + a, c + b) for r, c in moved for a in (-1, 0, 1) for b in (-1, 0, 1))
                break
        else:
            raise ValueError("cannot pack drawings")
    footprint = set().union(*(s["source"] | s["copy"] for s in placed))
    lo_r, lo_c = min(r for r, c in footprint), min(c for r, c in footprint)
    h, w = max(r for r, c in footprint) - lo_r + 1, max(c for r, c in footprint) - lo_c + 1
    pads = [int(rng.integers(min(6, 30 - length) + 1)) for length in (h, w)]
    before = [int(rng.integers(pad + 1)) for pad in pads]
    placed = [_shift(s, before[0] - lo_r, before[1] - lo_c) for s in placed]
    palette = [int(c) for c in rng.permutation(10)]
    bg = palette.pop()
    inp = np.full((h + pads[0], w + pads[1]), bg, dtype=np.int8)
    full = inp.copy()
    colors = []
    for s in placed:
        color, companion = palette.pop(), palette.pop() if s["marker"] is not None else None
        colors.append((color, companion))
        for p in s["vertices"]:
            inp[p] = color
        if companion is not None:
            inp[s["marker"]] = companion
    full[:] = inp
    all_source = set().union(*(s["source"] for s in placed))
    for s, (color, companion) in zip(placed, colors):
        if companion is not None:
            for p in s["copy"]:
                if inp[p] == bg and p not in all_source:
                    full[p] = companion
    for s, (color, _) in zip(placed, colors):
        for p in s["source"]:
            if inp[p] == bg:
                full[p] = color
    # Labels are rasterized from the constructed vertices, not obtained by solve.
    return {"input": inp, "output": _contract(full, bg)}, full


def _pair(rng, profile, seen):
    for _ in range(120):
        try:
            if profile == "recurrence_demo":
                specs = [_spec(rng, "rectangle", axial=True, chord=True)]
            elif profile == "diagonal_demo":
                specs = [_spec(rng, "bevel", small=True),
                         _spec(rng, str(rng.choice(("L", "segment"))), marked=False, small=True)]
            else:
                count = int(rng.integers(1, 4)) if profile != "test" else int(rng.integers(2, 4))
                families = ("rectangle", "bevel", "stair", "notch", "clipped", "L", "segment", "diamond")
                specs = [_spec(rng, str(rng.choice(families)), marked=i == 0 or rng.random() < .65,
                               small=count >= 2) for i in range(count)]
            pair, intended = _render(rng, specs)
            info = _features(pair["input"])
            if not np.array_equal(intended, info["completed"]):
                continue
            if not np.array_equal(pair["output"], info["output"]):
                continue
            if not all(info[k] for k in ("raw_diff", "source_diff", "copy_diff", "no_compression_diff", "comparison_changed", "premature_diff")):
                continue
            if profile == "recurrence_demo" and not all(info[k] for k in ("row_only_diff", "col_only_diff", "global_diff", "chord_diff")):
                continue
            if profile == "diagonal_demo" and not info["global_interaction"]:
                continue
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                return pair
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted: " + profile)


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    profiles = ["recurrence_demo", "diagonal_demo"] + ["varied"] * (int(num_train) - 2)
    rng.shuffle(profiles)
    seen = set()
    return {"train": [_pair(rng, p, seen) for p in profiles],
            "test": [_pair(rng, "test", seen) for _ in range(int(num_test))]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists required")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _features(inp)
            _require(np.array_equal(out, info["output"]), "incorrect completion/compression")
            _require(all(info[k] for k in ("raw_diff", "source_diff", "copy_diff", "no_compression_diff", "comparison_changed", "premature_diff")),
                     "each pair must use completion, visible copying and content contraction")
            _require(info["row_removed"] + info["col_removed"] > 0, "only cropped padding")
            # Contraction retains every color and the order of separated recurrences.
            _require(set(np.unique(out)) - {info["background"]} ==
                     set(np.unique(info["completed"])) - {info["background"]}, "lost foreground color")
            _require(all(np.any(out[edge] != info["background"]) for edge in (0, -1)), "uncropped rows")
            _require(all(np.any(out[:, edge] != info["background"]) for edge in (0, -1)), "uncropped columns")
            rr, cc = _keep_masks(out)
            _require(bool(rr.all() and cc.all()), "remaining adjacent duplicate vector")
            if split == "train":
                training.append(info)
    for feature in ("row_only_diff", "col_only_diff", "global_diff", "chord_diff", "global_interaction"):
        _require(any(i[feature] for i in training), "missing training evidence: " + feature)
    _require(any(s["closed"] for i in training for s in i["sources"]), "no contour demonstration")
    _require(any(not s["closed"] and s["diagonal"] for i in training for s in i["sources"]), "no bent diagonal demonstration")
    _require(any(i["new_overlap"] > 0 for i in training), "no newly drawn source-over-companion overlap evidence")
    _require(any(len(i["markers"]) < len(i["sources"]) for i in training), "no unmarked source demonstration")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded: one puzzle at a time across 42 host seeds; no artifact writes."""
    signatures, shapes, structures, offsets, source_counts, vertex_counts = set(), set(), set(), set(), set(), set()
    concave = diagonal = both_axes = no_marker_source = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _features(a)
                shapes.add(a.shape)
                palette = {}
                normalized = bytes(palette.setdefault(int(c), len(palette)) for c in a.flat)
                structures.add((a.shape, normalized))
                source_counts.add(len(info["sources"]))
                vertex_counts.update(len(s["points"]) for s in info["sources"])
                offsets.update(m["offset"] for m in info["markers"])
                concave += any(s["concave"] for s in info["sources"])
                diagonal += any(s["diagonal"] for s in info["sources"])
                both_axes += info["row_removed"] > 0 and info["col_removed"] > 0
                no_marker_source += len(info["markers"]) < len(info["sources"])
                assert np.array_equal(solve(a.tolist()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(structures) >= 150
    assert offsets == set(OFFSETS) and source_counts == {1, 2, 3}
    assert len(vertex_counts) >= 5 and concave and diagonal and both_axes and no_marker_source
    a = generate_puzzle(314159, 2, 1)
    assert _puzzle_digest(a) == _puzzle_digest(generate_puzzle(314159, 2, 1))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(10000 + 10 * nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Literal equality must retain order and separated repeats on BOTH axes.
    base = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int8)
    expanded = np.repeat(np.repeat(base, [2, 1, 3], axis=0), [1, 4, 2], axis=1)
    assert np.array_equal(_contract(expanded, 0), base)
    assert np.array_equal(_contract(expanded.T, 0), base.T)
    # A single differing pixel prevents a row or column contraction.
    exceptional = expanded.copy()
    exceptional[1, 2] = 5
    assert 5 in _contract(exceptional, 0)
    # Simultaneous masks commute; deleted duplicates cannot hide a distinction.
    rr, cc = _keep_masks(expanded)
    sequential = _contract(expanded[rr], 0, (False, True))
    assert np.array_equal(sequential, base)
    assert np.array_equal(_contract(base, 0), base)
    assert np.array_equal(_contract(np.array([[1, 1, 2, 1, 1]], dtype=np.int8), 0),
                          np.array([[1, 2, 1]], dtype=np.int8))
    # Hand-worked completion/copy/contraction regression, independent of sampling.
    fixture = np.zeros((9, 11), dtype=np.int8)
    for p in ((1, 2), (1, 8), (7, 8), (7, 2)):
        fixture[p] = 1
    fixture[0, 2] = 2
    expected = np.array([[2, 2, 2], [1, 1, 1], [1, 0, 1],
                         [1, 2, 1], [1, 1, 1]], dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(fixture, turns)), np.rot90(expected, turns))
    assert np.array_equal(solve((fixture + 7) % 10), (expected + 7) % 10)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
