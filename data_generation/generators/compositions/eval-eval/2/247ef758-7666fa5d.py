"""Rigid stencil threading: coordinate proposals subject to whole-mask clearance.

Only Python's standard library and NumPy are used. solve reads one input only.
No files, examples, seeds or sampler state are consulted by the solver. Tests
stream in memory; this module never writes grids, JSON, logs or traces to disk.
"""
from itertools import combinations, product

import numpy as np


_ALTERNATIVES = ("no_corridor", "all_centers", "center", "bbox", "clip",
                 "adjacent", "pixel_centers", "zip", "erase_complete")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _array(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("a grid must be an integer 1..30 by 1..30 array of colors 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in product((-1, 0, 1), repeat=2):
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _offsets(mask):
    return [(int(r - mask.shape[0] // 2), int(c - mask.shape[1] // 2))
            for r, c in np.argwhere(mask)]


def _recognize_strokes(cells):
    """Find the sole slope for which every maximal diagonal run has length >=2."""
    candidates = []
    for s in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(int(c - s * r), []).append(int(r))
        strokes = []
        valid = True
        for q, rows in lines.items():
            runs = []
            for r in sorted(rows):
                if not runs or r != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                if len(run) < 2:
                    valid = False
                    break
                strokes.append(dict(q=q, lo=2 * run[0] + s * q,
                                    hi=2 * run[-1] + s * q, length=len(run)))
        if valid and len(strokes) >= 2 and len({x["q"] for x in strokes}) >= 2:
            candidates.append((s, sorted(strokes, key=lambda x: (x["q"], x["lo"]))))
    if len(candidates) != 1:
        raise ValueError("the parallel strokes must have one recognizable slope")
    return candidates[0]


def _parse(value):
    inp = _array(value)
    h, w = inp.shape
    bg = int(inp[0, 0])
    candidates = []
    for d in range(1, w - 5):
        divider = int(inp[0, d])
        if divider == bg or not np.all(inp[:, d] == divider):
            continue
        frame = int(inp[0, d + 1])
        if frame in (bg, divider):
            continue
        if any(int(inp[r, c]) != frame for r, c in
               ((0, w - 1), (h - 1, d + 1), (h - 1, w - 1))):
            continue
        field = inp[1:-1, d + 2:-1]
        if not field.size:
            continue
        rails = set(map(int, np.unique(field))) - {bg}
        if len(rails) != 1:
            continue
        rail = next(iter(rails))
        if len({bg, divider, frame, rail}) != 4:
            continue
        stage = inp[:, :d]
        colors = set(map(int, np.unique(stage))) - {bg}
        if not colors or colors & {divider, frame, rail}:
            continue
        if np.any(stage[[0, -1], :] != bg):
            continue
        if not np.array_equal(inp[0, d + 2:-1], inp[-1, d + 2:-1]):
            continue
        if not np.array_equal(inp[1:-1, d + 1], inp[1:-1, -1]):
            continue
        edge = np.concatenate((inp[0, d + 2:-1], inp[1:-1, d + 1]))
        if any(int(x) in (bg, divider, rail) for x in edge):
            continue
        try:
            slope, strokes = _recognize_strokes(np.argwhere(field == rail))
            templates = []
            for color in sorted(colors):
                cells = np.argwhere(stage == color)
                lo, hi = cells.min(axis=0), cells.max(axis=0)
                bh, bw = map(int, hi - lo + 1)
                if bh % 2 != 1 or bw % 2 != 1:
                    raise ValueError("stencils require odd tight boxes")
                mask = stage[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] == color
                if not _connected(mask):
                    raise ValueError("one connected stencil per color is required")
                templates.append(dict(color=color, cells=cells, mask=mask,
                                      offsets=_offsets(mask), lo=tuple(map(int, lo)),
                                      rows=list(map(int, np.flatnonzero(inp[1:-1, d + 1] == color))),
                                      cols=list(map(int, np.flatnonzero(inp[0, d + 2:-1] == color)))))
            templates.sort(key=lambda x: x["lo"])
        except ValueError:
            continue
        candidates.append(dict(input=inp, d=d, bg=bg, divider=divider, frame=frame,
                               rail=rail, field=field, slope=slope, strokes=strokes,
                               templates=templates))
    if len(candidates) != 1:
        raise ValueError("staging region, divider and framed loom must be unambiguous")
    return candidates[0]


def _corridor(info, method="sweep"):
    field, s, strokes = info["field"], info["slope"], info["strokes"]
    rr, cc = np.indices(field.shape)
    q, t = cc - s * rr, rr + s * cc
    allowed = np.zeros(field.shape, dtype=bool)
    if method == "no_corridor":
        return field == info["bg"]
    if method == "sweep":
        for level in range(int(t.min()), int(t.max()) + 1):
            active = {a["q"] for a in strokes if a["lo"] <= level <= a["hi"]}
            if len(active) >= 2:
                allowed |= (t == level) & (q > min(active)) & (q < max(active))
    else:
        ranks = sorted({a["q"] for a in strokes})
        for a, b in combinations(strokes, 2):
            if a["q"] == b["q"]:
                continue
            if method == "adjacent" and abs(ranks.index(a["q"]) - ranks.index(b["q"])) != 1:
                continue
            mask = ((t >= max(a["lo"], b["lo"])) & (t <= min(a["hi"], b["hi"]))
                    & (q > min(a["q"], b["q"])) & (q < max(a["q"], b["q"])))
            if method == "pixel_centers":
                mask &= ((t - a["lo"]) % 2 == 0) & ((t - b["lo"]) % 2 == 0)
            allowed |= mask
    return allowed & (field == info["bg"])


def _shift(allowed, dr, dc):
    """result[r,c] = allowed[r+dr,c+dc], with false outside the field."""
    h, w = allowed.shape
    result = np.zeros((h, w), dtype=bool)
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    if r0 < r1 and c0 < c1:
        result[r0:r1, c0:c1] = allowed[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return result


def _fit_map(mask, allowed):
    fits = np.ones(allowed.shape, dtype=bool)
    for dr, dc in _offsets(mask):
        fits &= _shift(allowed, dr, dc)
    return fits


def _partial_map(mask, allowed):
    partial = np.zeros(allowed.shape, dtype=bool)
    for dr, dc in _offsets(mask):
        partial |= _shift(allowed, dr, dc)
    return partial


def _predict(info, alternative="", method="sweep", strict=True):
    """Input-only forward solver; alternate rules are used only for diagnostics."""
    geometry = alternative if alternative in ("no_corridor", "adjacent", "pixel_centers") else method
    allowed = _corridor(info, geometry)
    field = info["field"]
    h, w = field.shape
    out = info["input"].copy()
    target = out[1:-1, info["d"] + 2:-1]
    owners = {}
    decisions = []
    for template in info["templates"]:
        mask, offsets = template["mask"], template["offsets"]
        requested = list(product(template["rows"], template["cols"]))
        if alternative == "zip":
            requested = list(zip(template["rows"], template["cols"]))
        fit = _fit_map(np.ones(mask.shape, dtype=bool) if alternative == "bbox" else mask, allowed)
        if alternative == "center":
            # Only center clearance, but preserve rails and the canvas as an
            # otherwise plausible shortcut would do.
            fit = allowed & _fit_map(mask, field == info["bg"])
        if alternative == "all_centers":
            requested = list(map(tuple, np.argwhere(fit)))
        copies = []
        accepted = []
        for r, c in requested:
            cells = {(r + dr, c + dc) for dr, dc in offsets}
            if alternative == "clip":
                cells = {(y, x) for y, x in cells if 0 <= y < h and 0 <= x < w and allowed[y, x]}
                success = bool(cells)
            else:
                success = bool(fit[r, c])
            if success:
                accepted.append((int(r), int(c)))
                copies.append(cells)
        if accepted or (alternative == "erase_complete" and requested):
            cells = template["cells"]
            out[cells[:, 0], cells[:, 1]] = info["bg"]
        for cells in copies:
            for cell in cells:
                owner = owners.get(cell)
                if owner is not None and owner != template["color"]:
                    if strict:
                        raise ValueError("different-color accepted stencils must not intersect")
                    continue
                owners[cell] = template["color"]
                target[cell] = template["color"]
        decisions.append(dict(requested=requested, accepted=accepted, copies=copies))
    return out, decisions


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every decision from the supplied grid, never demonstrations or a seed."""
    return _predict(_parse(input_grid))[0]


def _differences(info, expected):
    return {name: int(np.count_nonzero(_predict(info, name, strict=False)[0] != expected))
            for name in _ALTERNATIVES}


def _sample_mask(rng, sw, thread_slope=None, small=False):
    if thread_slope is not None:
        size = 3 if sw < 5 or small or rng.random() < .65 else 5
        rr, cc = np.indices((size, size))
        mask = cc - size // 2 != thread_slope * (rr - size // 2)
        trial = mask & (rng.random(mask.shape) > .23)
        mid = size // 2
        trial[0, mid] = trial[-1, mid] = trial[mid, 0] = trial[mid, -1] = True
        return trial if _connected(trial) else mask
    sizes = [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (5, 5), (1, 5), (5, 1)]
    sizes = [(h, w) for h, w in sizes if w <= sw and (not small or max(h, w) <= 3)]
    bh, bw = sizes[int(rng.integers(len(sizes)))]
    mode = int(rng.integers(6))
    mask = np.zeros((bh, bw), dtype=bool)
    if min(bh, bw) == 1 or mode == 0:
        mask[:] = True
    elif mode == 1:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif mode == 2 and bh == bw:
        rr, cc = np.indices((bh, bw))
        mask = abs(rr - bh // 2) + abs(cc - bw // 2) == bh // 2
    else:
        # Connect extremal points with randomized monotone 8-neighbor paths.
        points = [(0, int(rng.integers(bw))), (bh - 1, int(rng.integers(bw))),
                  (int(rng.integers(bh)), 0), (int(rng.integers(bh)), bw - 1)]
        rng.shuffle(points)
        r, c = points[0]
        mask[r, c] = True
        for y, x in points[1:]:
            while (r, c) != (y, x):
                dr, dc = int(np.sign(y - r)), int(np.sign(x - c))
                if dr and dc and rng.random() < .5:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r + dr, c + dc
                mask[r, c] = True
        # A single bounded growth sweep enriches density without a retry loop.
        for index in rng.permutation(bh * bw):
            r, c = divmod(int(index), bw)
            if rng.random() < .45 and mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                mask[r, c] = True
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _sample_loom(rng, slope, kind, sw, fallback=False):
    if fallback:
        specs = [(0, -12, 12), (6, -2, 2), (12, -12, 12)]
    else:
        n = int(rng.integers(3 if kind != "free" else 2, 6 if kind != "free" else 7))
        qs = [0]
        for _ in range(n - 1):
            qs.append(qs[-1] + int(rng.integers(3, 8)))
        specs = []
        equal = kind == "free" and rng.random() < .18
        equal_length = int(rng.integers(5, 11))
        for i, q in enumerate(qs):
            if kind != "free" and i in (0, n - 1):
                length, core = int(rng.integers(9, 14)), 0
            elif kind == "bridge" and i == n // 2:
                length, core = int(rng.integers(2, 4)), 0
            elif kind == "thread" and i == n // 2:
                length, core = int(rng.integers(4, 9)), 0
            else:
                length = equal_length if equal else int(rng.integers(2, 14))
                core = 0 if equal else int(rng.integers(-7, 8))
            lo = core - length + 1
            lo -= (lo - q) % 2
            specs.append((q, lo, lo + 2 * (length - 1)))
    cells = []
    for q, lo, hi in specs:
        cells.extend(((t - slope * q) // 2, (q + slope * t) // 2)
                     for t in range(lo, hi + 1, 2))
    coords = np.asarray(cells, dtype=int)
    low, high = coords.min(axis=0), coords.max(axis=0)
    span_h, span_w = map(int, high - low + 1)
    min_h, min_w = max(11, span_h), max(8, span_w)
    max_w = 27 - sw
    if min_h > 28 or min_w > max_w:
        raise ValueError("loom does not fit the ARC canvas")
    h = min_h + int(rng.integers(0, min(6, 28 - min_h) + 1))
    w = min_w + int(rng.integers(0, min(5, max_w - min_w) + 1))
    rshift = int(rng.integers(h - span_h + 1)) - int(low[0])
    cshift = int(rng.integers(w - span_w + 1)) - int(low[1])
    rails = np.zeros((h, w), dtype=bool)
    rails[coords[:, 0] + rshift, coords[:, 1] + cshift] = True
    actual_slope, strokes = _recognize_strokes(np.argwhere(rails))
    if actual_slope != slope:
        raise ValueError("slope mismatch")
    return rails, dict(field=rails.astype(np.int8), bg=0, slope=slope, strokes=strokes)


def _choose_axes(rng, mask, allowed, core, counts, kind):
    h, w = allowed.shape
    fit = _fit_map(mask, allowed)
    free_fit = _fit_map(mask, np.ones_like(allowed))
    partial = _partial_map(mask, allowed)
    # Require an unclipped, nonbracketed but partially clear rejected copy in
    # teaching scenes. Input strokes are excluded separately by the caller.
    for _ in range(120):
        rows = sorted([core[0]] + list(map(int, rng.choice([r for r in range(h) if r != core[0]], counts[0] - 1, replace=False))))
        cols = sorted([core[1]] + list(map(int, rng.choice([c for c in range(w) if c != core[1]], counts[1] - 1, replace=False))))
        requested = list(product(rows, cols))
        if kind != "free":
            if not any(not fit[r, c] and partial[r, c] and free_fit[r, c] for r, c in requested):
                continue
        if kind == "bridge" and not any(fit[r, c] and (r, c) not in set(zip(rows, cols)) for r, c in requested):
            continue
        return rows, cols
    raise ValueError("no distinguishing coordinate product found")


def _stamp_cells(mask, centers):
    offsets = _offsets(mask)
    return {(r + dr, c + dc) for r, c in centers for dr, dc in offsets}


def _place_sources(rng, records, h, sw):
    occupied = np.zeros((h, sw), dtype=bool)
    for rec in records:
        mask = rec["mask"]
        bh, bw = mask.shape
        choices = [(r, c) for r in range(1, h - bh)
                   for c in range(sw - bw + 1)
                   if not (occupied[r:r + bh, c:c + bw] & mask).any()]
        if not choices:
            raise ValueError("source masks do not fit")
        r, c = choices[int(rng.integers(len(choices)))]
        rec["top"], rec["left"] = r, c
        occupied[r:r + bh, c:c + bw] |= mask


def _direct_labels(inp, rails, records, sw, slope, strokes, bg):
    """Construction labels use per-cell unions of boundary-pair parallelograms.

    No call to solve, its scanline envelope or its fit-map operation is made.
    This is a second self-authored calculation, not independent certification.
    """
    h, w = rails.shape
    out = inp.copy()
    claimed = set()
    for rec in records:
        accepted = []
        for r, c in product(rec["rows"], rec["cols"]):
            cells = _stamp_cells(rec["mask"], [(r, c)])
            success = True
            for y, x in cells:
                if not (0 <= y < h and 0 <= x < w) or rails[y, x]:
                    success = False
                    break
                q, t = x - slope * y, y + slope * x
                bracketed = any(max(a["lo"], b["lo"]) <= t <= min(a["hi"], b["hi"])
                                and min(a["q"], b["q"]) < q < max(a["q"], b["q"])
                                for a, b in combinations(strokes, 2))
                if not bracketed:
                    success = False
                    break
            if success:
                accepted.append(cells)
        if accepted:
            bh, bw = rec["mask"].shape
            view = out[rec["top"]:rec["top"] + bh, rec["left"]:rec["left"] + bw]
            view[rec["mask"]] = bg
        cells = set().union(*accepted) if accepted else set()
        if cells & claimed:
            raise ValueError("conflicting successful colors")
        claimed |= cells
        for r, c in cells:
            out[r + 1, sw + 2 + c] = rec["color"]
    return out


def _make_pair(rng, kind, slope, test=False):
    for attempt in range(80):
        fallback = attempt >= 56
        sw = 5 if fallback else int(rng.choice([3, 5, 7], p=[.2, .65, .15]))
        try:
            rails, geometry = _sample_loom(rng, slope, kind, sw, fallback)
            allowed = _corridor(geometry)
            adjacent = _corridor(geometry, "adjacent")
            pixels = _corridor(geometry, "pixel_centers")
            chosen = None
            for _ in range(18):
                mask = _sample_mask(rng, sw, slope if kind == "thread" else None,
                                    small=fallback or (kind == "bridge" and rng.random() < .6))
                fit = _fit_map(mask, allowed)
                wanted = fit.copy()
                if kind == "thread":
                    wanted &= ~allowed & ~_fit_map(np.ones_like(mask), allowed)
                elif kind == "bridge":
                    wanted &= ~_fit_map(mask, adjacent) & ~_fit_map(mask, pixels)
                choices = np.argwhere(wanted)
                if len(choices):
                    core = tuple(map(int, choices[int(rng.integers(len(choices)))]))
                    chosen = mask, fit, core
                    break
            if chosen is None:
                continue
            mask, fit, core = chosen
            if test:
                counts = [(3, 1), (1, 3), (3, 2), (2, 3), (3, 3)][int(rng.integers(5))]
            elif kind == "bridge":
                counts = (2, 2)
            else:
                counts = [(1, 2), (2, 1), (2, 2)][int(rng.integers(3))]
            rows, cols = _choose_axes(rng, mask, allowed, core, counts, kind)
            records = [dict(mask=mask, rows=rows, cols=cols)]
            claimed = _stamp_cells(mask, [(r, c) for r, c in product(rows, cols) if fit[r, c]])
            used_r, used_c = set(rows), set(cols)
            if kind == "thread":
                statuses = ["blocked", "N"]
            elif kind == "bridge":
                statuses = ["R", "C"]
            else:
                statuses = list(rng.choice(["success", "blocked", "N", "R", "C"],
                                           int(rng.integers(0, 4)), p=[.28, .28, .16, .14, .14]))
            for status in statuses:
                other = _sample_mask(rng, sw, small=fallback)
                fits = _fit_map(other, allowed)
                free_rows = [r for r in range(rails.shape[0]) if r not in used_r]
                free_cols = [c for c in range(rails.shape[1]) if c not in used_c]
                rr, cc = [], []
                if status in ("success", "blocked"):
                    choices = [(r, c) for r in free_rows for c in free_cols
                               if bool(fits[r, c]) == (status == "success")
                               and (status != "success" or not (_stamp_cells(other, [(r, c)]) & claimed))]
                    if not choices and status == "success":
                        status = "blocked"
                        choices = [(r, c) for r in free_rows for c in free_cols if not fits[r, c]]
                    if not choices:
                        raise ValueError("insufficient free coordinates")
                    r, c = choices[int(rng.integers(len(choices)))]
                    rr, cc = [r], [c]
                    if fits[r, c]:
                        claimed |= _stamp_cells(other, [(r, c)])
                elif status == "R":
                    rr = [int(rng.choice(free_rows))]
                elif status == "C":
                    cc = [int(rng.choice(free_cols))]
                used_r.update(rr)
                used_c.update(cc)
                records.append(dict(mask=other, rows=rr, cols=cc))
            h, w = rails.shape
            _place_sources(rng, records, h + 2, sw)
            palette = list(map(int, rng.permutation(10)))
            bg, divider, frame, rail = palette[:4]
            inp = np.full((h + 2, sw + w + 3), bg, dtype=np.int8)
            inp[:, sw] = divider
            inp[[0, -1], sw + 1:] = frame
            inp[:, [sw + 1, -1]] = frame
            inp[1:-1, sw + 2:-1][rails] = rail
            for rec, color in zip(records, palette[4:]):
                rec["color"] = color
                bh, bw = rec["mask"].shape
                inp[rec["top"]:rec["top"] + bh, rec["left"]:rec["left"] + bw][rec["mask"]] = color
                for r in rec["rows"]:
                    inp[r + 1, sw + 1] = inp[r + 1, -1] = color
                for c in rec["cols"]:
                    inp[0, sw + 2 + c] = inp[-1, sw + 2 + c] = color
            out = _direct_labels(inp, rails, records, sw, slope, geometry["strokes"], bg)
            info = _parse(inp)
            if not np.array_equal(_predict(info)[0], out):
                raise AssertionError("construction and visible-input solution disagree")
            if kind in ("thread", "bridge"):
                ds = _differences(info, out)
                needs = ("no_corridor", "all_centers", "center", "bbox", "clip", "erase_complete") if kind == "thread" else ("adjacent", "pixel_centers", "zip")
                if not all(ds[name] > 0 for name in needs):
                    continue
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded stencil/loom construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    first_slope = int(rng.choice([-1, 1]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            kind = ("thread" if index == 0 else "bridge") if split == "train" and index < 2 else "free"
            slope = first_slope * (1 if index % 2 == 0 else -1) if split == "train" else int(rng.choice([-1, 1]))
            for _ in range(12):
                pair = _make_pair(rng, kind, slope, test=split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair, alternatives=False):
    info = _parse(pair["input"])
    out, decisions = _predict(info)
    allowed = _corridor(info)
    source_states, products, masks = [], [], []
    successes = failures = hole_fits = merges = 0
    for template, decision in zip(info["templates"], decisions):
        rs, cs = template["rows"], template["cols"]
        source_states.append("A" if decision["accepted"] else "B" if rs and cs else "R" if rs else "C" if cs else "N")
        products.append((len(rs), len(cs)))
        masks.append((template["mask"].shape, template["mask"].tobytes()))
        successes += len(decision["accepted"])
        failures += len(decision["requested"]) - len(decision["accepted"])
        box_fit = _fit_map(np.ones_like(template["mask"]), allowed)
        hole_fits += sum(not box_fit[r, c] for r, c in decision["accepted"])
        if decision["copies"]:
            merges += sum(map(len, decision["copies"])) - len(set().union(*decision["copies"]))
    result = dict(shape=info["input"].shape, slope=info["slope"], rails=len(info["strokes"]),
                  lengths=tuple(a["length"] for a in info["strokes"]),
                  gaps=tuple(map(int, np.diff(sorted({a["q"] for a in info["strokes"]})))),
                  sources=len(masks), masks=tuple(masks), products=tuple(products),
                  states="".join(source_states), successes=successes, failures=failures,
                  hole_fits=hole_fits, merges=merges,
                  roles=(info["bg"], info["divider"], info["frame"], info["rail"]))
    if alternatives:
        result["differences"] = _differences(info, out)
    return result


def validate_generated_puzzle(puzzle):
    """Check visible semantics and teaching evidence, not hidden generation tags."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        _counts(len(puzzle["train"]), len(puzzle["test"]))
        seen = set()
        taught = set()
        slopes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                inp, out = _array(pair["input"]), _array(pair["output"])
                assert inp.shape == out.shape
                assert _key(inp) not in seen, "duplicate input"
                seen.add(_key(inp))
                info = _parse(inp)
                expected, decisions = _predict(info)
                assert np.array_equal(out, expected), "wrong whole-stencil placement or source consumption"
                assert np.array_equal(out, _predict(info, method="pairwise")[0]), "interval geometry mismatch"
                d = info["d"]
                assert np.array_equal(out[:, d:d + 2], inp[:, d:d + 2])
                assert np.array_equal(out[[0, -1]], inp[[0, -1]])
                assert np.array_equal(out[:, -1], inp[:, -1])
                rail_cells = info["field"] == info["rail"]
                assert np.all(out[1:-1, d + 2:-1][rail_cells] == info["rail"])
                assert any(x["accepted"] for x in decisions), "at least one stencil must succeed"
                allowed = _corridor(info)
                for template, decision in zip(info["templates"], decisions):
                    cells = template["cells"]
                    want = info["bg"] if decision["accepted"] else template["color"]
                    assert np.all(out[cells[:, 0], cells[:, 1]] == want)
                    for copy in decision["copies"]:
                        assert len(copy) == int(template["mask"].sum()), "partial copy"
                        assert all(allowed[r, c] for r, c in copy)
                    if split == "train":
                        assert len(template["rows"]) <= 2 and len(template["cols"]) <= 2
                if split == "train":
                    slopes.add(info["slope"])
                    taught.update(k for k, value in _differences(info, out).items() if value)
                else:
                    assert any(len(t["rows"]) == 3 or len(t["cols"]) == 3 for t in info["templates"]), "missing count shift"
        assert slopes == {-1, 1}, "both orientations must be demonstrated"
        assert taught == set(_ALTERNATIVES), "missing distinguishing training witnesses: " + str(set(_ALTERNATIVES) - taught)
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise AssertionError("malformed or unsupported family puzzle") from exc


def _fingerprint(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _analytical_fixture():
    """Hand-specified 11x11 interior, not a random generator case.

    s=-1, outer rails q=6 and q=14 have t=-4..4. At (5,5),
    q=10,t=0; the short q=10 rail passes through the empty center
    of a 3x3 cardinal diamond. Its four cells (4,5),(5,4),(5,6),
    (6,5) have q=9 or 11, t=-1 or 1, so all are clear. A copy
    at (1,1) fails wholly. Exactly one diamond is moved.
    """
    inp = np.zeros((13, 18), dtype=np.int8)
    sw = 4
    inp[:, sw] = 8
    inp[[0, -1], sw + 1:] = 7
    inp[:, [sw + 1, -1]] = 7
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    inp[1:4, :3][mask] = 3
    for q, lo, hi in ((6, -4, 4), (10, 0, 2), (14, -4, 4)):
        for t in range(lo, hi + 1, 2):
            r, c = (t + q) // 2, (q - t) // 2
            inp[r + 1, sw + 2 + c] = 9
    for r in (1, 5):
        inp[r + 1, sw + 1] = inp[r + 1, -1] = 3
    for c in (1, 5):
        inp[0, sw + 2 + c] = inp[-1, sw + 2 + c] = 3
    expected = inp.copy()
    expected[1:4, :3][mask] = 0
    for r, c in ((4, 5), (5, 4), (5, 6), (6, 5)):
        expected[r + 1, sw + 2 + c] = 3
    return inp, expected


def self_test():
    inp, expected = _analytical_fixture()
    assert np.array_equal(solve(inp), expected), "hand-calculated diamond threading fixture"
    # Bijective recoloring and reflection preserve the semantic roles.
    for delta in (1, 5, 9):
        assert np.array_equal(solve((inp + delta) % 10), (expected + delta) % 10)
    reflected = inp.copy()
    reflected[:, 5:] = reflected[:, 5:][:, ::-1]
    reflected_out = expected.copy()
    reflected_out[:, 5:] = reflected_out[:, 5:][:, ::-1]
    assert np.array_equal(solve(reflected), reflected_out)
    signatures, dimensions, masks, rails, source_counts, slopes, products_seen = set(), set(), set(), set(), set(), set(), set()
    roles, lengths, gaps = set(), set(), set()
    holes = merges = blocked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                dimensions.add(f["shape"])
                masks.update(f["masks"])
                rails.add(f["rails"])
                source_counts.add(f["sources"])
                slopes.add(f["slope"])
                products_seen.update(f["products"])
                roles.add(f["roles"])
                lengths.update(f["lengths"])
                gaps.update(f["gaps"])
                holes += f["hole_fits"]
                merges += f["merges"]
                blocked += "B" in f["states"]
                assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(signatures) == 42 and len(dimensions) >= 40 and len(masks) >= 60
    assert {2, 3, 4, 5} <= rails and {1, 2, 3, 4} <= source_counts
    assert slopes == {-1, 1} and len(roles) >= 100
    assert {(1, 2), (2, 1), (2, 2)} <= products_seen
    assert any(a == 3 or b == 3 for a, b in products_seen)
    assert min(lengths) == 2 and max(lengths) >= 10 and len(gaps) >= 4
    assert holes >= 42 and blocked >= 42 and merges > 0
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        assert len(generate_puzzle(42, a, b)["train"]) == a
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(91, 2, 1)
    # Every shortcut must actually change a training label and be rejected.
    for alternative in _ALTERNATIVES:
        for index, pair in enumerate(p["train"]):
            wrong = _predict(_parse(pair["input"]), alternative, strict=False)[0]
            if not np.array_equal(wrong, pair["output"]):
                bad = {s: [dict(x) for x in p[s]] for s in ("train", "test")}
                bad["train"][index] = {"input": pair["input"], "output": wrong}
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    break
                raise AssertionError("shortcut label accepted: " + alternative)
        else:
            raise AssertionError("missing shortcut witness: " + alternative)
    bad = {s: [dict(x) for x in p[s]] for s in ("train", "test")}
    corrupt = p["test"][0]["output"].copy()
    corrupt[0, 0] = (int(corrupt[0, 0]) + 1) % 10
    bad["test"][0] = {"input": p["test"][0]["input"], "output": corrupt}
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("corrupt output accepted")
