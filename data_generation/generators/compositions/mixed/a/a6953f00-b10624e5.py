"""Parity elects one of two prototypes; anchor geometry transfers its pattern.

Generation paints lattice rectangles forward. The input-only solver reconstructs
exact rational rasters from the actual pixels; it never reads generation state.
No files, caches, saved traces, or network operations are used by this module.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "integer cells required")
    _require(bool(np.all((array >= 0) & (array <= 9))), "colors must be 0..9")
    return array.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bounds(n, row, col):
    return ((0, row, 0, col), (0, row, col + 1, n),
            (row + 1, n, 0, col), (row + 1, n, col + 1, n))


def _connected(motif):
    remaining = set(motif)
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in remaining:
                remaining.remove(point)
                stack.append(point)
    return not remaining


def _flip(motif, side, vertical, horizontal):
    return {(side - 1 - r if vertical else r,
             side - 1 - c if horizontal else c): color
            for (r, c), color in motif.items()}


def _asymmetric(motif, side):
    return all(_flip(motif, side, v, h) != motif
               for v, h in ((False, True), (True, False), (True, True)))


def _square(panel, color):
    positions = np.argwhere(panel == color)
    _require(len(positions) > 0, "missing anchor")
    top, left = map(int, positions.min(0))
    bottom, right = map(int, positions.max(0) + 1)
    side = bottom - top
    _require(side == right - left and len(positions) == side * side,
             "anchor color must form exactly one solid square per panel")
    return top, left, side


def _decode(input_grid):
    grid = _grid(input_grid)
    n = grid.shape[0]
    _require(grid.shape == (n, n), "canvas must be square")
    rows = [r for r in range(1, n - 1) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(1, n - 1) if np.all(grid[:, c] == grid[0, c])]
    crosses = [(r, c, int(grid[r, c])) for r in rows for c in cols
               if grid[r, 0] == grid[0, c]]
    _require(len(crosses) == 1, "one single-cell full cross required")
    row, col, divider = crosses[0]
    bounds = _bounds(n, row, col)
    panels = [grid[r0:r1, c0:c1] for r0, r1, c0, c1 in bounds]
    counts = np.bincount(np.concatenate([p.ravel() for p in panels]), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] * 2 > sum(p.size for p in panels) and bg != divider,
             "background must be a strict majority outside divider")
    colors = [set(map(int, np.unique(p))) for p in panels]
    common = set.intersection(*colors) - {bg}
    _require(len(common) == 1, "unique shared anchor identity required")
    anchor = next(iter(common))
    _require(anchor != divider, "anchor and divider must differ")
    _require(all(colors[p] == {bg, anchor} for p in (2, 3)),
             "bottom panels must contain only background and anchor")
    anchors = [_square(p, anchor) for p in panels]
    motifs = []
    for p in (0, 1):
        r, c, side = anchors[p]
        motif = {(int(y) - r, int(x) - c): int(panels[p][y, x])
                 for y, x in np.argwhere(panels[p] != bg)}
        _require(len(set(motif.values())) >= 2 and _connected(motif),
                 "both upper panels must hold complete connected prototypes")
        motifs.append(motif)
    return {"grid": grid, "n": n, "row": row, "col": col, "bounds": bounds,
            "panels": panels, "background": bg, "divider": divider,
            "anchor": anchor, "anchors": anchors, "motifs": motifs,
            "reference": 0 if n % 2 else 1}


def _scaled(motif, source_side, dest_side, bg, vertical=False, horizontal=False):
    """Refine actual source pixels, then require every rational cell to agree.

    Anchor-multiple padding supplies background around sparse shapes. A reduction
    may merge same-color cells, but a cell straddling different colors is illegal.
    Check the entire transformed shape before clipping, so clipping cannot hide
    an ambiguous scale. Even the most permissive 30-cell structural input keeps
    the transient raster below 8 MB; sampled anchor sides <=6 use far less.
    """
    rr, cc = zip(*motif)
    top = min(rr) // source_side * source_side
    left = min(cc) // source_side * source_side
    bottom = ((max(rr) + source_side) // source_side) * source_side
    right = ((max(cc) + source_side) // source_side) * source_side
    raw = np.full((bottom - top, right - left), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r - top, c - left] = color
    expanded = raw.repeat(dest_side, 0).repeat(dest_side, 1)
    h, w = expanded.shape[0] // source_side, expanded.shape[1] // source_side
    blocks = expanded.reshape(h, source_side, w, source_side)
    raster = blocks.min(axis=(1, 3))
    _require(np.array_equal(raster, blocks.max(axis=(1, 3))),
             "nonintegral color boundary; rounding is not part of this family")
    ar, ac = -top * dest_side // source_side, -left * dest_side // source_side
    if vertical:
        raster = raster[::-1]
        ar = h - ar - dest_side
    if horizontal:
        raster = raster[:, ::-1]
        ac = w - ac - dest_side
    return raster, (ar, ac)


def _paste(panel, raster, top, left, bg):
    r0, c0 = max(0, top), max(0, left)
    r1 = min(panel.shape[0], top + raster.shape[0])
    c1 = min(panel.shape[1], left + raster.shape[1])
    if r0 < r1 and c0 < c1:
        tile = raster[r0 - top:r1 - top, c0 - left:c1 - left]
        target = panel[r0:r1, c0:c1]
        target[tile != bg] = tile[tile != bg]


def _prediction(data, reference=None, reflect=True, scale=True, keep_decoy=False):
    ref = data["reference"] if reference is None else reference
    bg = data["background"]
    source_side = data["anchors"][ref][2]
    out = data["grid"].copy()
    records = []
    for p, (r0, r1, c0, c1) in enumerate(data["bounds"]):
        if p == ref:
            continue
        r, c, side = data["anchors"][p]
        raster, (ar, ac) = _scaled(data["motifs"][ref], source_side,
                                 side if scale else source_side, bg,
                                 reflect and p // 2 != ref // 2,
                                 reflect and p % 2 != ref % 2)
        panel = out[r0:r1, c0:c1]
        if not keep_decoy:
            panel[:] = bg
        _paste(panel, raster, r - ar, c - ac, bg)
        # Correct copies already map the anchor exactly. For the no-scale
        # counterfactual retain the observed anchor to avoid a trivial deletion.
        panel[r:r + side, c:c + side] = data["anchor"]
        total = int(np.count_nonzero(raster != bg))
        visible = int(np.count_nonzero(panel != bg))
        records.append({"panel": p, "total": total, "visible": visible,
                        "lost": total - visible, "side": side})
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Elect TL on odd canvas sides, TR on even, then replace all other motifs."""
    return _prediction(_decode(input_grid))[0]


def _motif(rng, a, anchor, details, simple=False):
    motif = {(r, c): anchor for r in range(a) for c in range(a)}
    style = 0 if simple else int(rng.integers(4))

    def put(region, color):
        for point in region:
            if motif.get(point) != anchor:
                motif[point] = color

    if style == 0:
        up = 1 if simple else int(rng.integers(1, 4))
        right = 1 if simple else int(rng.integers(1, 4))
        put(((r, c) for r in range(-up, 0) for c in range(a)), details[0])
        put(((r, c) for r in range(a) for c in range(a, a + right)), details[-1])
    elif style == 1:
        length = int(rng.integers(3, 6))
        c = int(rng.integers(a))
        put(((r, c) for r in range(-length, 0)), details[0])
        for i, level in enumerate(range(1, length + 1, 2)):
            sign = int(rng.choice([-1, 1]))
            extent = int(rng.integers(1, 3))
            put(((-level, c + sign * d) for d in range(1, extent + 1)),
                details[(i + 1) % len(details)])
    else:
        for step in range(int(rng.integers(3, 10))):
            cells = sorted(motif)
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            length = int(rng.integers(1, 4 if style == 3 else 2))
            width = int(rng.integers(1, a + 1))
            region = {(r + dr * d + (j if dc else 0),
                       c + dc * d + (j if dr else 0))
                      for d in range(1, length + 1) for j in range(width)}
            combined = set(motif) | region
            rr, cc = zip(*combined)
            if max(rr) - min(rr) < 6 and max(cc) - min(cc) < 6:
                put(region, details[step % len(details)])
    for _ in range(int(rng.integers(4))):
        motif = {(c, a - 1 - r): color for (r, c), color in motif.items()}
    return _flip(motif, a, False, bool(rng.integers(2)))


def _forward_raster(motif, a, unit, bg, reference, panel):
    """Construction renderer works directly from latent lattice rectangles."""
    transformed = _flip(motif, a, reference // 2 != panel // 2,
                        reference % 2 != panel % 2)
    rr, cc = zip(*transformed)
    top, left = min(rr), min(cc)
    raster = np.full(((max(rr) - top + 1) * unit,
                      (max(cc) - left + 1) * unit), bg, dtype=np.int8)
    for (r, c), color in transformed.items():
        raster[(r - top) * unit:(r - top + 1) * unit,
               (c - left) * unit:(c - left + 1) * unit] = color
    return raster, (-top * unit, -left * unit)


def _placement(rng, shape, side, rasters, bg, clipped):
    # Both candidate elections must be possible at the same observed anchors.
    packed = []
    for raster, offset in rasters:
        mask = (raster != bg).astype(np.int32)
        prefix = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        packed.append((raster.shape, offset, prefix, int(mask.sum())))
    candidates = []
    for r in range(shape[0] - side + 1):
        for c in range(shape[1] - side + 1):
            valid = True
            for (h, w), (ar, ac), prefix, total in packed:
                top, left = r - ar, c - ac
                y0, x0 = max(0, -top), max(0, -left)
                y1, x1 = min(h, shape[0] - top), min(w, shape[1] - left)
                visible = int(prefix[y1, x1] - prefix[y0, x1]
                              - prefix[y1, x0] + prefix[y0, x0])
                if (visible < total) != clipped or visible <= side * side:
                    valid = False
                    break
            if valid:
                candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _build(rng, n, test=False, simple=False):
    colors = [int(c) for c in rng.permutation(10)]
    bg, divider, anchor = colors[:3]
    details = colors[3:3 + int(rng.integers(2, 5))]
    if not simple and rng.random() < .35:
        details[-1] = divider
    a = 1 if simple else int(rng.choice([1, 1, 2]))
    motifs = [_motif(rng, a, anchor, details, simple) for _ in range(2)]
    if (motifs[0] == motifs[1] or
            not all(_connected(m) and _asymmetric(m, a) for m in motifs)):
        return None
    units = [int(x) for x in rng.permutation([1, 2])]
    units += [int(rng.integers(1, 3)), 3 if test and not simple else int(rng.integers(1, 3))]
    if not simple and test and rng.random() < .25:
        units[int(rng.integers(2))] = 3
    if simple:
        units = [1, 2, 1, 2]
    # Off-center divider positions vary independently of canvas parity.
    min_span = 5 if test else 7
    row = int(rng.integers(min_span, n - min_span))
    col = int(rng.integers(min_span, n - min_span))
    bounds = _bounds(n, row, col)
    inp = np.full((n, n), bg, dtype=np.int8)
    inp[row] = divider
    inp[:, col] = divider
    positions = []
    for p, (r0, r1, c0, c1) in enumerate(bounds):
        rasters = [_forward_raster(motifs[q], a, units[p], bg, q, p) for q in (0, 1)]
        clipped = p == 3 or (test and p == 2 and rng.random() < .55)
        pos = _placement(rng, (r1 - r0, c1 - c0), a * units[p], rasters, bg, clipped)
        if pos is None:
            return None
        positions.append(pos)
        r, c = pos
        panel = inp[r0:r1, c0:c1]
        side = a * units[p]
        panel[r:r + side, c:c + side] = anchor
        if p < 2:
            raster, (ar, ac) = rasters[p]
            _paste(panel, raster, r - ar, c - ac, bg)
    # Enforce role visibility, exact-raster compatibility and necessity on pixels.
    try:
        data = _decode(inp)
        for reference in (0, 1):
            out, records = _prediction(data, reference)
            if any(r["visible"] <= r["side"] ** 2 for r in records):
                return None
        # Construction labels are produced independently from solver scaling.
        def output(grid):
            ref = 0 if len(grid) % 2 else 1
            out = grid.copy()
            for p, (r0, r1, c0, c1) in enumerate(_bounds(len(grid), row, col)):
                if p == ref:
                    continue
                panel = out[r0:r1, c0:c1]
                panel[:] = bg
                raster, (ar, ac) = _forward_raster(motifs[ref], a, units[p], bg, ref, p)
                r, c = positions[p]
                _paste(panel, raster, r - ar, c - ac, bg)
            return {"input": grid, "output": out}

        pair = output(inp)
        _analyze(pair)
        if test:
            return [pair]
        extended = np.full((n + 1, n + 1), bg, dtype=np.int8)
        extended[:n, :n] = inp
        extended[row, :] = divider
        extended[:, col] = divider
        second = output(extended)
        _analyze(second)
        return [pair, second]
    except AssertionError:
        return None


def _fallback(n, test, seen):
    """Guaranteed compact scene; 90 distinct palettes exceed 14 retained pairs.

    For n>=17, centered panels have sides >=8. Each full motif needs at most
    4 rows and 6 columns. Bottom-right loses its downward arm but retains its
    horizontal arm for either election. Palette enumeration cannot exhaust.
    """
    row = col = n // 2
    units = [1, 2, 1, 2]
    bounds = _bounds(n, row, col)
    for bg in range(10):
        for anchor in range(10):
            if bg == anchor:
                continue
            divider, first, second = [c for c in range(10) if c not in (bg, anchor)][:3]
            motifs = [{(0, 0): anchor, (-1, 0): first, (0, 1): second},
                      {(0, 0): anchor, (-1, 0): second, (0, 1): first}]
            inp = np.full((n, n), bg, dtype=np.int8)
            inp[row] = divider
            inp[:, col] = divider
            positions = []
            for p, (r0, r1, c0, c1) in enumerate(bounds):
                unit = units[p]
                r = r1 - r0 - unit if p == 3 else unit
                c = unit
                positions.append((r, c))
                panel = inp[r0:r1, c0:c1]
                panel[r:r + unit, c:c + unit] = anchor
                if p < 2:
                    raster, (ar, ac) = _forward_raster(motifs[p], 1, unit, bg, p, p)
                    _paste(panel, raster, r - ar, c - ac, bg)
            batch = []
            for size in ([n] if test else [n, n + 1]):
                grid = np.full((size, size), bg, dtype=np.int8)
                grid[:n, :n] = inp
                grid[row] = divider
                grid[:, col] = divider
                ref = 0 if size % 2 else 1
                out = grid.copy()
                for p, (r0, r1, c0, c1) in enumerate(_bounds(size, row, col)):
                    if p == ref:
                        continue
                    panel = out[r0:r1, c0:c1]
                    panel[:] = bg
                    raster, (ar, ac) = _forward_raster(motifs[ref], 1, units[p], bg, ref, p)
                    r, c = positions[p]
                    _paste(panel, raster, r - ar, c - ac, bg)
                pair = {"input": grid, "output": out}
                _analyze(pair)
                batch.append(pair)
            if all(_key(p["input"]) not in seen for p in batch):
                return batch
    raise AssertionError("fallback palette space unexpectedly exhausted")


def _fresh(rng, n, test, seen):
    # Finite attempts followed by a constructive fallback; no disk diagnostics.
    for attempt in range(160):
        batch = _build(rng, n, test, simple=attempt >= 128)
        if batch is not None and all(_key(p["input"]) not in seen for p in batch):
            return batch
    return _fallback(n, test, seen)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first = int(rng.choice([19, 21, 23]))
    while len(puzzle["train"]) < num_train:
        index = len(puzzle["train"]) // 2
        if index == 0:
            n = first
        elif index == 1:
            n = int(rng.choice([26, 28]))
        else:
            n = int(rng.integers(19, 29))
        batch = _fresh(rng, n, False, seen)
        for pair in batch[:num_train - len(puzzle["train"])]:
            seen.add(_key(pair["input"]))
            puzzle["train"].append(pair)
    for _ in range(num_test):
        n = int(rng.integers(17, 31))
        pair = _fresh(rng, n, True, seen)[0]
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    data = _decode(inp)
    expected, records = _prediction(data)
    _require(out.shape == inp.shape and np.array_equal(expected, out),
             "wrong election, scaling, reflection, replacement or clipping")
    ref = data["reference"]
    source = data["bounds"][ref]
    r0, r1, c0, c1 = source
    _require(np.array_equal(inp[r0:r1, c0:c1], out[r0:r1, c0:c1]),
             "selected source changed")
    _require(np.array_equal(inp[data["row"]], out[data["row"]]) and
             np.array_equal(inp[:, data["col"]], out[:, data["col"]]), "divider changed")
    _require(all(r["visible"] > r["side"] ** 2 for r in records),
             "every destination must retain visible detail")
    other = _prediction(data, 1 - ref)[0]
    bottom = data["row"] + 1
    _require(not np.array_equal(other[bottom:], out[bottom:]),
             "parity election must affect bottom completions")
    _require(not np.array_equal(_prediction(data, reflect=False)[0], out),
             "reflection must affect answer")
    _require(not np.array_equal(_prediction(data, scale=False)[0], out),
             "scaling must affect answer")
    _require(not np.array_equal(_prediction(data, keep_decoy=True)[0], out),
             "non-selected prototype must really be replaced")
    data["copies"] = records
    return data


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, parities, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            data = _analyze(pair)
            key = _key(data["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                parities.add(data["n"] % 2)
                training.append(data)
    _require(parities == {0, 1}, "both parity branches must be demonstrated")
    # A matched pair isolates parity from palette, anchors, positions and detail.
    twins = False
    for small in training:
        for large in training:
            n = small["n"]
            if (large["n"] == n + 1 and small["row"] == large["row"]
                    and small["col"] == large["col"]
                    and np.array_equal(small["grid"], large["grid"][:n, :n])):
                twins = True
    _require(twins, "training needs a content-matched parity counterfactual")
    if a >= 4:
        sizes = {p: {d["n"] for d in training if d["n"] % 2 == p} for p in (0, 1)}
        _require(all(len(v) >= 2 for v in sizes.values()), "need two sizes of each parity")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _features(pair):
    data = _decode(pair["input"])
    _, records = _prediction(data)
    ref = data["reference"]
    side = data["anchors"][ref][2]
    motif = data["motifs"][ref]
    # Normalize uniform lattice enlargement and role names, retaining anchor.
    boundary = [side]
    for (r, c), color in motif.items():
        for dr, dc, offset in ((-1, 0, r), (1, 0, r + 1),
                               (0, -1, c), (0, 1, c + 1)):
            if motif.get((r + dr, c + dc), data["background"]) != color:
                boundary.append(offset)
    quantum = int(np.gcd.reduce(boundary))
    roles = {data["anchor"]: 0}
    normalized = sorted({(r // quantum, c // quantum, color)
                         for (r, c), color in motif.items()})
    structure = tuple((r, c, roles.setdefault(color, len(roles))) for r, c, color in normalized)
    return {"side": data["n"], "divider": (data["row"], data["col"]),
            "roles": (data["background"], data["divider"], data["anchor"]),
            "source": ref, "source_side": side,
            "anchor_sides": tuple(x[2] for x in data["anchors"]),
            "ratios": tuple((r["side"] // int(np.gcd(r["side"], side)),
                              side // int(np.gcd(r["side"], side))) for r in records),
            "clipped": sum(r["lost"] > 0 for r in records),
            "fine": quantum < side, "divider_detail": data["divider"] in motif.values(),
            "motif_structure": structure}


def self_test():
    """42 streaming seeds, mutation checks and all legal counts; writes nothing."""
    digests, shapes, dividers, roles, ratios, motifs, anchors = (set() for _ in range(7))
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        if seed < 3:
            _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f["side"])
                dividers.add((f["side"], *f["divider"]))
                roles.add(f["roles"])
                ratios.update(f["ratios"])
                motifs.add(f["motif_structure"])
                anchors.update(f["anchor_sides"])
                totals["pairs"] += 1
                totals["fine"] += f["fine"]
                totals["divider_detail"] += f["divider_detail"]
                totals[(split, f["source"])] += 1
                totals[("clipped", f["clipped"])] += 1
                original = pair["input"].copy()
                _require(np.array_equal(solve(original), pair["output"]), "solver disagreement")
                _require(np.array_equal(original, pair["input"]), "solver mutated input")
    _require(len(digests) == 42 and len(shapes) >= 10, "size or seed diversity collapsed")
    _require(len(dividers) >= 70 and len(roles) >= 60, "layout or palette diversity collapsed")
    _require(len(motifs) >= 45 and len(ratios) >= 5 and len(anchors) >= 4, "geometry diversity collapsed")
    _require(totals["fine"] >= 10 and totals["divider_detail"] >= 10, "fine or shared-color cases missing")
    _require(all(totals[(s, r)] for s in ("train", "test") for r in (0, 1)), "parity absent")
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    fallback_seen = set()
    for n, test in ((17, True), (18, True), (19, False), (28, False), (30, True)):
        batch = _fallback(n, test, fallback_seen)
        fallback_seen.update(_key(p["input"]) for p in batch)
        # Exercise palette collision handling on the very same geometry.
        again = _fallback(n, test, fallback_seen)
        _require(all(_key(p["input"]) not in fallback_seen for p in again), "fallback collision")
    # Exact-raster regression includes fine details and a forbidden half-cell.
    fine = {(0, 0): 2, (0, 1): 2, (1, 0): 2, (1, 1): 2, (-1, 0): 3}
    _scaled(fine, 2, 4, 0, True, True)
    try:
        _scaled(fine, 2, 3, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous scale accepted")
    broken = generate_puzzle(91, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "pairs": totals["pairs"],
            "canvas_sides": sorted(shapes), "divider_layouts": len(dividers),
            "role_assignments": len(roles), "normalized_source_motifs": len(motifs),
            "anchor_sides": sorted(anchors), "scale_ratios": sorted(ratios),
            "fine_pairs": totals["fine"], "divider_detail_pairs": totals["divider_detail"]}
