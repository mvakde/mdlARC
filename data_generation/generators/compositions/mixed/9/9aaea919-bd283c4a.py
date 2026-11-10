"""Counted frequency raster: stack addition changes the inventory to be sorted.

Control bars: 2 = source, 3 = destination; processed sources become 5.
Each lane has its own repeated token, with exactly one blank row between tokens.
No parent modules, examples, random state, or puzzle-level state are used by solve.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


SOURCE, DESTINATION, HIGHLIGHT = 2, 3, 5
BACKGROUNDS = (0, 1, 4, 6, 7, 8, 9)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be rectangles with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid cells must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _runs(bits):
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _connected(mask):
    points = set(zip(*np.nonzero(mask)))
    if not points:
        return False
    pending = [points.pop()]
    while pending:
        r, c = pending.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in points:
                points.remove(p)
                pending.append(p)
    return not points


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(bg in BACKGROUNDS and np.all(a[0] == bg),
             "the first row must be blank; background cannot be a control/highlight")
    _require(not np.any(a == HIGHLIGHT), "highlight is reserved for processed sources")
    lanes = _runs(np.any(a != bg, axis=0))
    _require(len(lanes) >= 2, "at least a source and a destination are required")
    stacks, baselines, control_rows = [], set(), set()
    for x, end in lanes:
        strip = a[:, x:end]
        groups = _runs(np.any(strip != bg, axis=1))
        marker = None
        if groups[-1][1] - groups[-1][0] == 1:
            cy, _ = groups.pop()
            marker = int(strip[cy, 0])
            _require(marker in (SOURCE, DESTINATION) and np.all(strip[cy] == marker),
                     "controls must be full-width source/destination bars")
            control_rows.add(cy)
        _require(bool(groups), "every lane must contain tokens")
        first, stop = groups[0]
        mask = strip[first:stop] != bg
        h, w = mask.shape
        _require(2 <= h <= 4 and 2 <= w <= 4 and _connected(mask),
                 "token masks must be connected and 2..4 cells in each dimension")
        _require(np.all(np.any(mask, axis=0)) and np.all(np.any(mask, axis=1)),
                 "token masks must span their bounding boxes")
        color = int(strip[first:stop][mask][0])
        for top, bottom in groups:
            block = strip[top:bottom]
            _require(block.shape == mask.shape and np.array_equal(block != bg, mask),
                     "a lane must repeat one silhouette")
            _require(np.all(block[mask] == color), "each stack must be monochrome")
        ys = [top for top, _ in groups]
        _require(all(v - u == h + 1 for u, v in zip(ys, ys[1:])),
                 "tokens must have exactly one blank row between them")
        baselines.add(groups[-1][1])
        stacks.append({"x": x, "mask": mask.copy(), "color": color,
                       "count": len(groups), "ys": ys, "marker": marker})
    _require(len(baselines) == len(control_rows) == 1,
             "tokens must share a bottom baseline and controls a lower row")
    baseline, cy = baselines.pop(), control_rows.pop()
    _require(1 <= cy - baseline <= 3 and np.all(a[baseline:cy] == bg),
             "controls must be separated from tokens by 1..3 blank rows")
    _require(np.all(a[cy + 1:] == bg), "only blank margin may follow controls")
    sources = [s for s in stacks if s["marker"] == SOURCE]
    destinations = [s for s in stacks if s["marker"] == DESTINATION]
    _require(1 <= len(sources) <= 4 and len(destinations) == 1,
             "one destination and one to four sources are required")
    dst = destinations[0]
    total = sum(s["count"] for s in sources)
    h = dst["mask"].shape[0]
    _require(dst["ys"][-1] - (dst["count"] + total - 1) * (h + 1) >= 1,
             "the complete extension must fit below the blank first row")
    return {"grid": a, "background": bg, "stacks": stacks, "sources": sources,
            "destination": dst, "total": total, "baseline": baseline, "control_row": cy}


def _draw(a, top, x, color, mask):
    h, w = mask.shape
    a[top:top + h, x:x + w][mask] = color


def _transfer(parsed):
    """Materialize A before B counts any colors."""
    a = parsed["grid"].copy()
    a[parsed["control_row"], :] = parsed["background"]
    for s in parsed["sources"]:
        for y in s["ys"]:
            _draw(a, y, s["x"], HIGHLIGHT, s["mask"])
    dst = parsed["destination"]
    pitch = dst["mask"].shape[0] + 1
    for k in range(dst["count"], dst["count"] + parsed["total"]):
        y = dst["ys"][-1] - k * pitch
        h, w = dst["mask"].shape
        _require(np.all(a[y:y + h, dst["x"]:dst["x"] + w][dst["mask"]] == parsed["background"]),
                 "appended token would collide with existing cells")
        _draw(a, y, dst["x"], dst["color"], dst["mask"])
    return a


def _hist(a):
    return np.bincount(a.ravel(), minlength=10).astype(np.int64)


def _pack_counts(counts, shape):
    colors = np.flatnonzero(counts)
    _require(len(set(counts[colors].tolist())) == len(colors),
             "positive post-transfer frequencies must be distinct; ties are outside the family")
    _require(np.all(counts >= 0) and int(counts.sum()) == int(np.prod(shape)),
             "inventory must exactly fill the original canvas")
    order = colors[np.argsort(-counts[colors])]
    stream = np.repeat(order, counts[order]).astype(np.int8)
    h, w = shape
    return stream.reshape(w, h)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible lanes, fixed controls, and the resulting inventory."""
    p = _parse(input_grid)
    transformed = _transfer(p)
    return _pack_counts(_hist(transformed), transformed.shape)


def _inventory(p, *, copies=None, recolor=True, erase=True, global_recolor=False,
               move=False, extra_cells=None):
    """Accounting check, independent of painting the intermediate grid."""
    counts = _hist(p["grid"])
    bg = p["background"]
    if erase:
        for s in p["stacks"]:
            if s["marker"] is not None:
                width = s["mask"].shape[1]
                counts[s["marker"]] -= width
                counts[bg] += width
    if recolor:
        selected = p["sources"]
        if global_recolor:
            colors = {s["color"] for s in selected}
            selected = [s for s in p["stacks"] if s["color"] in colors]
        for s in selected:
            cells = s["count"] * int(s["mask"].sum())
            counts[s["color"]] -= cells
            counts[bg if move else HIGHLIGHT] += cells
    dst = p["destination"]
    n = p["total"] if copies is None else copies
    cells = n * int(dst["mask"].sum()) if extra_cells is None else extra_cells
    counts[bg] -= cells
    counts[dst["color"]] += cells
    return counts


def _mask(rng, h=None, w=None):
    h = int(rng.integers(2, 5)) if h is None else h
    w = int(rng.integers(2, 5)) if w is None else w
    kind = int(rng.integers(4))
    if kind == 0:
        mask = np.ones((h, w), dtype=bool)
    elif kind == 1:
        mask = np.zeros((h, w), dtype=bool)
        mask[:, int(rng.integers(w))] = True
        mask[int(rng.integers(h)), :] = True
    else:
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1] = True
        for r in range(h - 1):
            mask[r, :int(rng.integers(1, w + 1))] = True
        if rng.integers(2):
            mask = mask[::-1]
        if rng.integers(2):
            mask = mask[:, ::-1]
    return mask.copy()


def _partition(rng, total, n):
    if n == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), n - 1, replace=False).tolist())
    values = np.diff([0] + cuts + [total]).tolist()
    rng.shuffle(values)
    return values


def _reversal(p, counts):
    old = _hist(p["grid"])
    colors = np.flatnonzero((counts > 0) & (old > 0))
    return any((int(old[a]) - int(old[b])) * (int(counts[a]) - int(counts[b])) < 0
               for i, a in enumerate(colors) for b in colors[i + 1:])


def _partial(counts, height):
    ranked = sorted(counts[counts > 0].tolist(), reverse=True)
    return any(2 <= int(x) % height <= height - 2 for x in np.cumsum(ranked)[:-1])


def _make_pair(rng, nsrc, *, anchor=None, dense=False, seen=()):
    """Sample one bounded scene, then draw its answer directly from descriptors."""
    for _ in range(600):
        compact = not dense and bool(rng.integers(2))
        nu = 5 if dense else (int(rng.integers(1, 4)) if anchor else int(rng.integers(0, min(3, 6 - nsrc) + 1)))
        n = nsrc + 1 + nu
        if n > 7:
            continue
        masks = [_mask(rng) for _ in range(n)]
        if dense:
            masks = [np.ones((2, 2), dtype=bool), _mask(rng, 2, 2)] + [
                np.ones((4, int(rng.integers(3, 5))), dtype=bool) for _ in range(nu)]
        # One simple anchor shows that more source cells need not mean more tokens.
        if anchor == "single":
            masks[0] = _mask(rng, 2, int(rng.integers(2, 4)))
            masks[1] = _mask(rng, int(rng.integers(3, 5)), int(rng.integers(3, 5)))
        widths = [m.shape[1] for m in masks]
        minimum_width = sum(widths) + n - 1
        if minimum_width > 30:
            continue
        extent_limit = int(rng.integers(15, 25)) if compact else 27
        caps = [(extent_limit + 1) // (m.shape[0] + 1) for m in masks]
        min_total = nsrc + (1 if nsrc > 1 and anchor else 0)
        min_total = max(2, min_total) if anchor else min_total
        if caps[0] <= min_total:
            continue
        d = int(rng.integers(1, caps[0] - min_total + 1))
        total = int(rng.integers(min_total, caps[0] - d + 1))
        if dense and d + total < 5:
            continue
        sources = _partition(rng, total, nsrc)
        if any(c > cap for c, cap in zip(sources, caps[1:nsrc + 1])):
            continue
        if anchor == "multi" and len(set(sources)) < 2:
            continue
        counts = [d] + sources + [int(rng.integers(1, cap + 1)) for cap in caps[nsrc + 1:]]
        if dense:
            counts[2:] = caps[2:]
        bg = int(rng.choice(BACKGROUNDS))
        pool = [c for c in range(10) if c not in (bg, HIGHLIGHT)]
        colors = rng.choice(pool, n, replace=True).tolist()
        if dense:
            colors[2:] = [colors[0]] * nu
        if anchor == "multi":
            colors[-1] = colors[1]  # spatially distinct unmarked/source color match
        elif not dense and rng.random() < .40 and nu:
            colors[-1] = colors[1]
        if not anchor and not dense and rng.random() < .15 and nu >= 2:
            colors[nsrc + 1:] = [colors[-1]] * nu
        final_counts = counts.copy()
        final_counts[0] += total
        extents = [m.shape[0] + (c - 1) * (m.shape[0] + 1)
                   for m, c in zip(masks, final_counts)]
        extent = max(extents)
        if extent + 3 > 30:
            continue
        spare = 30 - extent - 3
        gap = 1 if dense else int(rng.integers(1, min(2 if compact else 3, spare + 1) + 1))
        spare -= gap - 1
        bottom_margin = 0 if dense else int(rng.integers(0, min(1 if compact else 3, spare) + 1))
        spare -= bottom_margin
        top_margin = 1 if dense else int(rng.integers(1, min(2, spare + 1) + 1) if compact else rng.integers(1, spare + 2))
        baseline = top_margin + extent
        cy = baseline + gap
        height = cy + 1 + bottom_margin
        width = minimum_width if dense else int(rng.integers(minimum_width, min(30, minimum_width + 4) + 1) if compact else rng.integers(minimum_width, 31))
        order = rng.permutation(n).tolist()
        extras = rng.multinomial(width - minimum_width, np.full(n + 1, 1 / (n + 1)))
        xs = [0] * n
        x = int(extras[0])
        for j, i in enumerate(order):
            xs[i] = x
            x += widths[i] + (1 + int(extras[j + 1]) if j < n - 1 else 0)
        before = np.full((height, width), bg, dtype=np.int8)
        intermediate = before.copy()
        for i, (mask, color, count) in enumerate(zip(masks, colors, counts)):
            h, w = mask.shape
            for k in range(count):
                _draw(before, baseline - h - k * (h + 1), xs[i], color, mask)
            for k in range(final_counts[i]):
                out_color = HIGHLIGHT if 1 <= i <= nsrc else color
                _draw(intermediate, baseline - h - k * (h + 1), xs[i], out_color, mask)
            if i <= nsrc:
                before[cy, xs[i]:xs[i] + w] = DESTINATION if i == 0 else SOURCE
        if _signature(before) in seen:
            continue
        hist = _hist(intermediate)
        positive = hist[hist > 0]
        if len(set(positive.tolist())) != len(positive):
            continue
        if anchor:
            if not _partial(hist, height):
                continue
            p = _parse(before)
            dst = p["destination"]
            dst_cells = int(hist[dst["color"]])
            if anchor == "single" and not hist[HIGHLIGHT] > dst_cells:
                continue
            if anchor == "multi" and not _reversal(p, hist):
                continue
        # This construction uses declared final stacks, not solve's transfer.
        return {"input": before, "output": _pack_counts(hist, before.shape)}
    raise RuntimeError("bounded scene sampler exhausted")


def _signature(a):
    return a.shape, a.tobytes()


def _mirror_pair(pair):
    # Horizontal reflection permutes whole lanes; its post-transfer inventory is identical.
    return {"input": pair["input"][:, ::-1].copy(), "output": pair["output"].copy()}


def _counts(num_train, num_test):
    for v, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    train_source_max = int(rng.integers(2, 4))
    single = _make_pair(rng, 1, anchor="single")
    seen = {_signature(single["input"])}
    p0 = _parse(single["input"])
    for _ in range(100):
        multi = _make_pair(rng, train_source_max, anchor="multi", seen=seen)
        p1 = _parse(multi["input"])
        s0, s1 = p0["total"], p1["total"]
        d0, d1 = p0["destination"]["count"], p1["destination"]["count"]
        if s0 != s1 and d0 != d1 and s0 * d1 != s1 * d0:
            break
    else:
        raise RuntimeError("bounded operand contrast sampler exhausted")
    anchors = [single, multi]
    rng.shuffle(anchors)
    puzzle = {"train": anchors, "test": []}
    seen.add(_signature(multi["input"]))
    for i in range(2, num_train):
        candidate = _mirror_pair(anchors[int(rng.integers(2))]) if i == 2 and rng.random() < .65 else None
        if candidate is None or _signature(candidate["input"]) in seen:
            dense = bool(rng.random() < .30)
            candidate = _make_pair(rng, 1 if dense else int(rng.integers(1, train_source_max + 1)), dense=dense, seen=seen)
        puzzle["train"].append(candidate)
        seen.add(_signature(candidate["input"]))
    # Unequal multi-source aggregation is already demonstrated. Tests extend it
    # to one more source lane, not to a new arithmetic or spacing convention.
    for _ in range(num_test):
        pair = _make_pair(rng, train_source_max + 1, seen=seen)
        puzzle["test"].append(pair)
        seen.add(_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _rival_histograms(p):
    s = p["total"]
    d = p["destination"]["count"]
    src = p["sources"]
    unmarked = [x for x in p["stacks"] if x["marker"] is None]
    return {
        "omit_A": _hist(p["grid"]),
        "double_destination": _inventory(p, copies=d),
        "count_bars": _inventory(p, copies=len(src)),
        "tallest_source": _inventory(p, copies=max(x["count"] for x in src)),
        "fixed_one_copy": _inventory(p, copies=1),
        "no_append": _inventory(p, copies=0),
        "count_unmarked_too": _inventory(p, copies=s + sum(x["count"] for x in unmarked)),
        "source_cells_as_tokens": _inventory(p, copies=sum(x["count"] * int(x["mask"].sum()) for x in src)),
        "copy_source_cell_area": _inventory(p, extra_cells=sum(x["count"] * int(x["mask"].sum()) for x in src)),
        "no_highlight": _inventory(p, recolor=False),
        "keep_controls": _inventory(p, erase=False),
        "move_sources": _inventory(p, move=True),
        "recolor_by_color_globally": _inventory(p, global_recolor=True),
    }


def _traversal(counts, shape, major, reverse_major, reverse_minor):
    colors = sorted(np.flatnonzero(counts).tolist(), key=lambda c: -int(counts[c]))
    stream = [c for c in colors for _ in range(int(counts[c]))]
    h, w = shape
    outer = list(range(w if major == "column" else h))
    inner = list(range(h if major == "column" else w))
    if reverse_major:
        outer.reverse()
    if reverse_minor:
        inner.reverse()
    out = np.empty(shape, dtype=np.int8)
    index = 0
    for u in outer:
        for v in inner:
            r, c = (v, u) if major == "column" else (u, v)
            out[r, c] = stream[index]
            index += 1
    return out


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train, tests = set(), [], []
    rival_witnesses = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions must be preserved")
            key = _signature(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            p = _parse(a)
            intermediate = _transfer(p)
            inventory = _inventory(p)
            _require(np.array_equal(inventory, _hist(intermediate)), "transfer inventory mismatch")
            _require(np.array_equal(inventory, _hist(b)), "output must conserve the post-transfer inventory")
            _require(np.array_equal(b, _pack_counts(inventory, a.shape)), "incorrect frequency raster")
            _require(not np.array_equal(b, intermediate), "B must change the transfer scene")
            _require(not np.array_equal(b, a), "identity pair")
            _require(np.array_equal(solve(a.copy()), b), "input-only solve disagrees")
            (train if split == "train" else tests).append(p)
            if split == "train":
                rival_witnesses.update(name for name, hist in _rival_histograms(p).items()
                                       if not np.array_equal(hist, inventory))
    expected_rivals = set(_rival_histograms(train[0]))
    _require(rival_witnesses == expected_rivals, "training leaves a listed inventory shortcut unresolved")
    totals = [p["total"] for p in train]
    dests = [p["destination"]["count"] for p in train]
    _require(len(set(totals)) > 1 and len(set(dests)) > 1 and
             any(totals[0] * d != s * dests[0] for s, d in zip(totals, dests)),
             "training must vary both operands and their ratio")
    _require(any(len(p["sources"]) > 1 and len({s["count"] for s in p["sources"]}) > 1 for p in train),
             "unequal multi-source aggregation must be demonstrated")
    _require(any(_reversal(p, _inventory(p)) for p in train), "training must reverse a surviving frequency rank")
    _require(any(_partial(_inventory(p), p["grid"].shape[0]) for p in train), "training must reveal partial-column direction")
    _require(any(_inventory(p)[HIGHLIGHT] > _inventory(p)[p["destination"]["color"]] for p in train),
             "training must distinguish cell frequency from final token counts")
    options = [(major, a, b) for major in ("column", "row") for a in (False, True) for b in (False, True)]
    matches = [o for o in options if all(np.array_equal(_traversal(_inventory(p), p["grid"].shape, *o), pair["output"])
                                      for p, pair in zip(train, puzzle["train"]))]
    _require(matches == [("column", False, True)], "training does not distinguish the eight basic traversals")
    max_train_sources = max(len(p["sources"]) for p in train)
    _require(all(len(p["sources"]) == max_train_sources + 1 for p in tests),
             "tests must extend the demonstrated source-lane count by one")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[r], y[r])
               for x, y in zip(a[s], b[s]) for r in ("input", "output")) for s in ("train", "test"))


def self_test():
    """42 sequential seeds, bounded memory, no files or execution traces saved."""
    shapes, masks, palettes, structures, fingerprints, source_counts = set(), set(), set(), set(), set(), set()
    rank_orders, remainders = set(), set()
    equal_output_contrasts = shared_color_pairs = mixed_shapes = 0
    records = 0
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if sample is None:
            sample = puzzle
        digest = hashlib.sha256()
        local_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                p = _parse(a)
                counts = _inventory(p)
                # Loop-based packing checks the reshape-based production code.
                direct = _traversal(counts, a.shape, "column", False, True)
                _require(np.array_equal(b, direct), "direct traversal check failed")
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                shapes.add(a.shape)
                local_shapes.add(a.shape)
                palettes.add(len(np.flatnonzero(counts)))
                source_counts.add(len(p["sources"]))
                rank_orders.add(tuple(sorted(np.flatnonzero(counts).tolist(), key=lambda c: -int(counts[c]))))
                remainders.update(int(x) % a.shape[0] for x in np.cumsum(sorted(counts[counts > 0].tolist(), reverse=True))[:-1])
                roles = {}
                normal = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(normal).hexdigest()))
                colors = [s["color"] for s in p["stacks"]]
                shared_color_pairs += len(set(colors)) < len(colors)
                for stack in p["stacks"]:
                    m = stack["mask"]
                    masks.add((m.shape, m.tobytes()))
                records += 1
        mixed_shapes += len(local_shapes) > 1
        fingerprints.add(digest.hexdigest())
        equal_output_contrasts += any(np.array_equal(x["output"], y["output"])
                                     for i, x in enumerate(puzzle["train"]) for y in puzzle["train"][i + 1:])
    _require(len(fingerprints) == 42, "seeds must vary whole puzzles")
    _require(len(shapes) >= 60 and len(masks) >= 35 and len(structures) >= 150, "insufficient structural variation")
    _require({1, 2, 3, 4} <= source_counts and len(palettes) >= 3, "counts/palettes did not vary")
    _require(len(rank_orders) >= 40 and len(remainders) >= 15, "frequency profiles did not vary")
    _require(mixed_shapes >= 38 and equal_output_contrasts >= 12 and shared_color_pairs >= 80,
             "missing cross-example structural/color contrasts")
    _require(_equal(generate_puzzle(73, 7, 3), generate_puzzle(73, 7, 3)), "nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "explicit counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Singleton geometry needs no earlier example: local pitch is height + 1.
    singleton = np.zeros((10, 6), dtype=np.int8)
    singleton[6:8, 0:2] = 7
    singleton[6:8, 3:6] = 9
    singleton[9, 0:2] = SOURCE
    singleton[9, 3:6] = DESTINATION
    target_hist = np.zeros(10, dtype=np.int64)
    target_hist[[0, 5, 9]] = [44, 4, 12]
    _require(np.array_equal(solve(singleton), _traversal(target_hist, singleton.shape, "column", False, True)),
             "singleton-only scene failed")
    tied = np.zeros((10, 7), dtype=np.int8)
    tied[6:8, 0:4] = 6
    tied[6:8, 5:7] = 7
    tied[9, 0:4] = SOURCE
    tied[9, 5:7] = DESTINATION
    try:
        solve(tied)  # 8 highlighted cells and 8 destination cells
    except ValueError as error:
        _require("frequencies must be distinct" in str(error), "wrong tied-inventory rejection")
    else:
        raise AssertionError("tied post-transfer inventory accepted")
    # Corrupt the output inventory and then the raster while retaining inventory.
    for preserve_histogram in (False, True):
        bad = {s: [{r: a.copy() for r, a in pair.items()} for pair in sample[s]] for s in ("train", "test")}
        out = bad["train"][0]["output"]
        if preserve_histogram:
            flat = out.ravel()
            k = int(np.flatnonzero(flat != flat[0])[0])
            flat[0], flat[k] = flat[k], flat[0]
        else:
            out[0, 0] = (int(out[0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("mutated output accepted")
    return {"seeds": 42, "pairs": records, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "token_masks": len(masks), "input_structures": len(structures),
            "palette_sizes": sorted(palettes), "rank_orders": len(rank_orders),
            "boundary_remainders": len(remainders), "mixed_shape_puzzles": mixed_shapes,
            "equal_output_contrast_puzzles": equal_output_contrasts, "shared_payload_pairs": shared_color_pairs}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
