"""Unary tokens program ranked fills inside an anchored assembly of bar glyphs.

Only NumPy and the standard library are used.  solve() reads pixels afresh;
sampling records never participate in decoding.
"""
from itertools import product

import numpy as np


PITCH = 5


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _runs(indices):
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)


def _gap(a, b):
    """Rectangles use inclusive extrema; require one blank separating band."""
    t, l, d, r = a
    u, v, e, s = b
    return d + 1 < u or e + 1 < t or r + 1 < v or s + 1 < l


def _parse(g, bg):
    bars, tokens = [], []
    for c in range(g.shape[1]):
        for run in _runs(np.flatnonzero(g[:, c] != bg)):
            t, b = int(run[0]), int(run[-1])
            vals = g[t:b + 1, c]
            if np.all(vals == vals[0]):
                tokens.append({"r": t, "c": c, "n": b - t + 1,
                               "color": int(vals[0])})
            else:
                _require(b - t >= 2, "bar has no interior")
                cap, ink = int(vals[0]), int(vals[1])
                _require(int(vals[-1]) == cap and cap != ink
                         and np.all(vals[1:-1] == ink), "not a capped bar")
                bars.append({"t": t, "b": b, "c": c, "length": b - t + 1,
                             "cap": cap, "ink": ink})
    _require(len(bars) >= 2 and len(tokens) >= 2, "missing reference or token map")
    cap, ink = bars[0]["cap"], bars[0]["ink"]
    _require(all(b["cap"] == cap and b["ink"] == ink for b in bars),
             "one reference family with uniform role colors is required")
    _require(len({b["b"] for b in bars}) == 1, "reference baseline is not common")
    _require(len({b["length"] for b in bars}) == len(bars), "tied reference lengths")
    cols = sorted(b["c"] for b in bars)
    _require(all(y - x >= 2 for x, y in zip(cols, cols[1:])),
             "reference bars need blank columns between them")
    t, l, b, r = min(x["t"] for x in bars), min(cols), bars[0]["b"], max(cols)
    h, w = b - t + 1, r - l + 1
    _require(all(1 <= x["n"] <= min(4, len(bars)) for x in tokens),
             "invalid unary ordinal")
    _require(all(x["color"] in (cap, ink) for x in tokens), "unrecognized token color")
    anchors = [x for x in tokens if x["color"] == cap]
    _require(len(anchors) == 1, "exactly one cap-colored token must anchor the map")
    anchor = anchors[0]
    mr, mc = min(x["r"] for x in tokens), min(x["c"] for x in tokens)
    _require(all((x["r"] - mr) % PITCH == 0 and (x["c"] - mc) % PITCH == 0
                 for x in tokens), "token starts do not lie on the pitch-five lattice")
    map_box = (mr, mc, max(x["r"] + x["n"] - 1 for x in tokens),
               max(x["c"] for x in tokens))
    _require(_gap((t, l, b, r), map_box), "reference and map are not separate")
    for x in tokens:
        x["site"] = ((x["r"] - mr) // PITCH, (x["c"] - mc) // PITCH)
        x["anchor"] = x is anchor
        x["out"] = (t + ((x["r"] - anchor["r"]) // PITCH) * h,
                      l + ((x["c"] - anchor["c"]) // PITCH) * w)
        a, z = x["out"]
        _require(0 <= a and a + h <= g.shape[0] and 0 <= z and z + w <= g.shape[1],
                 "expanded box leaves the canvas")
    _require(len({x["site"] for x in tokens}) == len(tokens), "duplicate map site")
    return {"bg": bg, "cap": cap, "ink": ink, "bars": bars, "tokens": tokens,
            "t": t, "l": l, "h": h, "w": w, "anchor": anchor["site"],
            "map_dims": (max(x["site"][0] for x in tokens) + 1,
                         max(x["site"][1] for x in tokens) + 1),
            "reference": g[t:b + 1, l:r + 1].copy(),
            "ordered": sorted(bars, key=lambda x: x["length"], reverse=True)}


def _decode(value):
    g = _grid(value)
    colors, counts = np.unique(g, return_counts=True)
    if len(colors) != 3 or np.sum(counts == counts.max()) != 1:
        raise ValueError("three local colors and a unique modal background are required")
    bg = int(colors[int(np.argmax(counts))])
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(g, -rotation)
        try:
            m = _parse(canonical, bg)
        except AssertionError:
            continue
        candidates.append((rotation, canonical, m))
    if len(candidates) != 1:
        raise ValueError("input has no unique reference axis and token decomposition")
    return candidates[0]


def _render(g, m, *, fixed_rank=None, invert=True, replicate=True):
    out = np.full(g.shape, m["bg"], dtype=np.int8)
    for token in m["tokens"]:
        if not replicate and not token["anchor"]:
            continue
        tile = m["reference"].copy()
        rank = token["n"] if fixed_rank is None else fixed_rank
        bar = m["ordered"][rank - 1]
        tile[bar["t"] - m["t"] + 1:m["h"] - 1,
             bar["c"] - m["l"]] = m["cap"]
        if token["anchor"] and invert:
            caps, inks = tile == m["cap"], tile == m["ink"]
            tile[caps], tile[inks] = m["ink"], m["cap"]
        r, c = token["out"]
        out[r:r + m["h"], c:c + m["w"]] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the full answer using only this input's geometry and colors."""
    rotation, g, m = _decode(input_grid)
    return np.ascontiguousarray(np.rot90(_render(g, m), rotation))


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _connected(points):
    points = set(points)
    todo = [next(iter(points))]
    seen = set(todo)
    while todo:
        r, c = todo.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in points and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == points


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas dimensions changed")
    rotation, g, m = _decode(inp)
    canonical_out = np.rot90(out, -rotation)
    expected = _render(g, m)
    _require(np.array_equal(canonical_out, expected), "incorrect programmed assembly")
    _require(not np.array_equal(inp, out), "identity example")
    population = sum(x["length"] for x in m["bars"])
    _require(np.count_nonzero(out != m["bg"]) == population * len(m["tokens"]),
             "template mask population was not preserved per occupied site")
    # Check each actual output tile's requested bar and all unselected bars.
    for token in m["tokens"]:
        r, c = token["out"]
        chosen = m["ordered"][token["n"] - 1]
        cap, ink = (m["ink"], m["cap"]) if token["anchor"] else (m["cap"], m["ink"])
        for bar in m["bars"]:
            col = c + bar["c"] - m["l"]
            top, bottom = r + bar["t"] - m["t"], r + m["h"] - 1
            _require(canonical_out[top, col] == cap and canonical_out[bottom, col] == cap,
                     "output cap color is wrong")
            wanted = cap if bar is chosen else ink
            _require(np.all(canonical_out[top + 1:bottom, col] == wanted),
                     "wrong selected or unselected interior")
    m["rotation"] = rotation
    return m


def _training_checks(metas, tests):
    ranks = {x["n"] for m in metas for x in m["tokens"] if not x["anchor"]}
    _require({1, 2} <= ranks, "training needs ordinary ranks one and two")
    _require(any({1, 2} <= {x["n"] for x in m["tokens"] if not x["anchor"]}
                 and len(m["bars"]) >= 3 for m in metas),
             "training needs contrasting local keys and a non-extreme second rank")
    _require(any(m["l"] < m["ordered"][0]["c"] < m["l"] + m["w"] - 1
                 and any(x["n"] == 1 and not x["anchor"] for x in m["tokens"])
                 for m in metas), "spatial ordinal shortcuts are not contradicted")
    _require(len({m["h"] for m in metas}) >= 2 and len({m["w"] for m in metas}) >= 2,
             "both reference pitches need a training contrast")
    _require(len({m["anchor"][0] for m in metas}) >= 2
             and len({m["anchor"][1] for m in metas}) >= 2, "anchor coordinates did not vary")
    _require(any(m["h"] != m["w"] for m in metas), "rectangular reference needed")
    _require(any(m["anchor"] != ((m["map_dims"][0] - 1) // 2,
                                  (m["map_dims"][1] - 1) // 2) for m in metas),
             "noncentral anchoring needs demonstration")
    _require(any(all(any((r + (axis == 0), c + (axis == 1)) in
                         {x["site"] for x in m["tokens"]}
                         for r, c in {x["site"] for x in m["tokens"]})
                     for axis in (0, 1)) for m in metas), "both lattice steps need witnesses")
    for m in tests:
        for x in m["tokens"]:
            n = x["n"]
            _require(n in ranks or set(range(1, n)) <= ranks,
                     "test ordinal has no taught progression")
        # More general gapped maps are supported by solve. Such test extensions
        # require a corresponding demonstration, rather than silent difficulty.
        for axis in (0, 1):
            missing = len({x["site"][axis] for x in m["tokens"]}) < m["map_dims"][axis]
            if missing:
                _require(any(len({x["site"][axis] for x in a["tokens"]}) < a["map_dims"][axis]
                             for a in metas), "an empty map band was not demonstrated")
        if not _connected(x["site"] for x in m["tokens"]):
            _require(any(not _connected(x["site"] for x in a["tokens"]) for a in metas),
                     "disconnected map extension was not demonstrated")


def validate_generated_puzzle(puzzle):
    """Validate pixels, per-tile semantics, and demonstration sufficiency."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad example counts")
    seen, meta = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            m = _check_pair(pair)
            signature = _key(pair["input"])
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            meta[split].append(m)
    _training_checks(meta["train"], meta["test"])


def _reference_options(target, teaching, taught, shift, previous):
    max_pop = 4 if shift == "population" and not target else 5
    min_pop = max(3, taught) if teaching else taught if target else 2
    if target and shift == "population":
        min_pop = max_pop = 5
    options = []
    for pop in range(min_pop, max_pop + 1):
        for gaps in product((1, 2), repeat=pop - 1):
            w = pop + sum(gaps)
            for h in range(pop + 2, 10):
                if previous is not None and (h == previous["h"] or w == previous["w"]):
                    continue
                if teaching and previous is None and h == w:
                    continue
                for mh in range(1, 5):
                    for mw in range(1, 5):
                        if mh * h > 28 or mw * w > 28 or mh * mw < 2:
                            continue
                        needed = taught + 1 if teaching else 2
                        if mh * mw < needed or (teaching and min(mh, mw) < 2):
                            continue
                        if shift in ("wide", "tall"):
                            want_wide = (shift == "wide") == target
                            if not ((mw > mh) if want_wide else (mh > mw)):
                                continue
                        options.append((pop, h, w, gaps, mh, mw))
    return options


def _layout(rng, mh, mw, anchor, minimum):
    """Connected masks from a spanning path plus random frontier growth."""
    mask = np.zeros((mh, mw), dtype=bool)
    flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
    ar = mh - 1 - anchor[0] if flip_r else anchor[0]
    ac = mw - 1 - anchor[1] if flip_c else anchor[1]
    r = c = 0
    mask[r, c] = True
    for er, ec in ((ar, ac), (mh - 1, mw - 1)):
        steps = [0] * (er - r) + [1] * (ec - c)
        rng.shuffle(steps)
        for axis in steps:
            r += axis == 0
            c += axis == 1
            mask[r, c] = True
    goal = int(rng.integers(max(minimum, int(mask.sum())), mh * mw + 1))
    while mask.sum() < goal:
        frontier = set()
        for r, c in np.argwhere(mask):
            for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= a < mh and 0 <= b < mw and not mask[a, b]:
                    frontier.add((int(a), int(b)))
        mask[_pick(rng, sorted(frontier))] = True
    if flip_r:
        mask = np.flipud(mask)
    if flip_c:
        mask = np.fliplr(mask)
    return mask


def _place(rng, reference, lengths, columns, layout, anchor, ranks, colors):
    """Fit the union of the expanded assembly and the separate input token map."""
    bg, cap, ink = colors
    h, w = reference.shape
    mh, mw = layout.shape
    ah, aw = mh * h, mw * w
    sites = [tuple(map(int, x)) for x in np.argwhere(layout)]
    ph = max(PITCH * r + ranks[(r, c)] for r, c in sites)
    pw = PITCH * (mw - 1) + 1
    tr, tc = anchor[0] * h, anchor[1] * w
    rr, cc = np.meshgrid(np.arange(-ph - 2, ah + 3),
                         np.arange(-pw - 2, aw + 3), indexing="ij")
    low_r, low_c = np.minimum(0, rr), np.minimum(0, cc)
    high_r, high_c = np.maximum(ah, rr + ph), np.maximum(aw, cc + pw)
    valid = ((rr + ph < tr) | (tr + h < rr) | (cc + pw < tc) | (tc + w < cc))
    valid &= (high_r - low_r <= 30) & (high_c - low_c <= 30)
    choices = np.argwhere(valid)
    if not len(choices):
        raise ValueError("geometry has no fitting placement")
    if rng.random() < 0.3:
        areas = (high_r - low_r)[valid] * (high_c - low_c)[valid]
        choices = choices[areas <= np.quantile(areas, 0.2)]
    i, j = _pick(rng, choices)
    pr, pc = int(rr[i, j]), int(cc[i, j])
    bh, bw = int(high_r[i, j] - low_r[i, j]), int(high_c[i, j] - low_c[i, j])
    gh = 30 if rng.random() < 0.1 else bh + int(rng.integers(min(4, 30 - bh) + 1))
    gw = 30 if rng.random() < 0.1 else bw + int(rng.integers(min(4, 30 - bw) + 1))
    top = int(rng.integers(gh - bh + 1)) - int(low_r[i, j])
    left = int(rng.integers(gw - bw + 1)) - int(low_c[i, j])
    inp = np.full((gh, gw), bg, dtype=np.int8)
    inp[top + tr:top + tr + h, left + tc:left + tc + w] = reference
    out = np.full_like(inp, bg)
    order = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
    for r, c in sites:
        n = ranks[(r, c)]
        special = (r, c) == anchor
        ir, ic = top + pr + PITCH * r, left + pc + PITCH * c
        inp[ir:ir + n, ic] = cap if special else ink
        tile = reference.copy()
        k = order[n - 1]
        tile[h - lengths[k] + 1:h - 1, columns[k]] = cap
        if special:
            caps, inks = tile == cap, tile == ink
            tile[caps], tile[inks] = ink, cap
        a, b = top + r * h, left + c * w
        out[a:a + h, b:b + w] = tile
    # This affects neither reference pitches nor output positions.
    while True:
        counts = np.bincount(inp.ravel().astype(int), minlength=10)
        if counts[bg] > max(counts[x] for x in range(10) if x != bg):
            break
        axis = 0 if inp.shape[0] < 30 else 1
        _require(inp.shape[axis] < 30, "could not expose modal background")
        pads = ((0, 1), (0, 0)) if axis == 0 else ((0, 0), (0, 1))
        inp = np.pad(inp, pads, constant_values=bg)
        out = np.pad(out, pads, constant_values=bg)
    return {"input": inp, "output": out}


def _sample_pair(rng, *, target, teaching, limit, shift, previous, palette, rotation):
    options = _reference_options(target, teaching, limit, shift, previous)
    _require(bool(options), "no compatible reference dimensions")
    pop, h, w, gaps, mh, mw = _pick(rng, options)
    anchor_options = [(r, c) for r in range(mh) for c in range(mw)
                      if previous is None or (r != previous["anchor"][0]
                                              and c != previous["anchor"][1])]
    if teaching and previous is None:
        anchor_options = [p for p in anchor_options if p != ((mh - 1) // 2, (mw - 1) // 2)]
    anchor = _pick(rng, anchor_options)
    layout = _layout(rng, mh, mw, anchor, limit + 1 if teaching else 2)
    lengths = [h] + [int(x) for x in rng.choice(np.arange(3, h), pop - 1, replace=False)]
    rng.shuffle(lengths)
    if teaching:
        k = int(np.argmax(lengths))
        j = int(rng.integers(1, pop - 1))
        lengths[k], lengths[j] = lengths[j], lengths[k]
    columns, col = [0], 0
    for gap in gaps:
        col += 1 + gap
        columns.append(col)
    bg, cap, ink = palette if palette is not None else tuple(map(int, rng.choice(10, 3, replace=False)))
    reference = np.full((h, w), bg, dtype=np.int8)
    for n, c in zip(lengths, columns):
        reference[h - n, c] = reference[h - 1, c] = cap
        reference[h - n + 1:h - 1, c] = ink
    upper = min(4, pop, limit)
    sites = [tuple(map(int, x)) for x in np.argwhere(layout)]
    ranks = {p: int(rng.integers(1, upper + 1)) for p in sites}
    if teaching:
        ordinary = [p for p in sites if p != anchor]
        rng.shuffle(ordinary)
        for p, n in zip(ordinary, range(1, limit + 1)):
            ranks[p] = n
    if target:
        ranks[_pick(rng, sites)] = limit
    pair = _place(rng, reference, lengths, columns, layout, anchor, ranks, (bg, cap, ink))
    reflect = bool(rng.integers(2))
    if teaching and reflect:
        reflected_anchor = (anchor[0], mw - 1 - anchor[1])
        if previous is None:
            reflect = reflected_anchor != ((mh - 1) // 2, (mw - 1) // 2)
        else:
            reflect = (reflected_anchor[0] != previous["anchor"][0]
                       and reflected_anchor[1] != previous["anchor"][1])
    if reflect:
        pair = {k: np.fliplr(v) for k, v in pair.items()}
    rot = int(rng.integers(4)) if rotation is None else rotation
    pair = {k: np.ascontiguousarray(np.rot90(v, rot)) for k, v in pair.items()}
    # Derive curriculum state from returned pixels, including reflections.
    _, _, m = _decode(pair["input"])
    return pair, {"h": m["h"], "w": m["w"], "anchor": m["anchor"]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    shift = _pick(rng, ("ordinal", "wide", "tall", "population"))
    taught = int(rng.integers(2, 4)) if shift == "ordinal" else int(rng.integers(2, 5))
    tested = taught + 1 if shift == "ordinal" else taught
    palette = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < 0.3 else None
    rotation = None if shift in ("ordinal", "population") and rng.random() < 0.4 else int(rng.integers(4))
    puzzle, used, previous = {"train": [], "test": []}, set(), None
    for split, count in (("train", nt), ("test", nq)):
        target = split == "test"
        for index in range(count):
            teaching = not target and index < 2
            for _ in range(80):
                # Retry finite placement/uniqueness choices, never a seed or label.
                try:
                    pair, info = _sample_pair(
                        rng, target=target, teaching=teaching,
                        limit=tested if target else taught, shift=shift,
                        previous=previous if teaching and index == 1 else None,
                        palette=palette, rotation=rotation)
                except ValueError:
                    continue
                if _key(pair["input"]) in used:
                    continue
                if teaching and index == 1 and (info["anchor"][0] == previous["anchor"][0]
                                                or info["anchor"][1] == previous["anchor"][1]):
                    continue
                used.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if teaching and index == 0:
                    previous = info
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((s, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]))
                 for s in ("train", "test"))


def self_test():
    """Bounded semantic, metamorphic, transfer, and 42-seed diversity checks."""
    puzzles, metas = [], []
    for seed in range(42):
        nq = 1 + seed % 4
        nt = nq + 1 + (seed // 4) % (10 - nq)
        p = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for s in ("train", "test"):
            metas.extend((_check_pair(pair), pair["input"].shape) for pair in p[s])
    _require(len({_signature(p) for p in puzzles}) == 42, "cross-seed identity")
    populations = {len(m["bars"]) for m, _ in metas}
    reference_shapes = {(m["h"], m["w"]) for m, _ in metas}
    map_shapes = {m["map_dims"] for m, _ in metas}
    anchors = {m["anchor"] for m, _ in metas}
    ordinal_pairs = {(len(m["bars"]), x["n"]) for m, _ in metas for x in m["tokens"]}
    _require(populations == {2, 3, 4, 5}, "reference population coverage")
    _require(len(reference_shapes) >= 15 and len(map_shapes) >= 10, "structural size variation")
    _require(len(anchors) >= 10, "anchor variation")
    _require({(2, 2), (3, 3), (4, 3), (5, 3), (4, 4), (5, 4)} <= ordinal_pairs,
             "population and rank must vary separately")
    _require({1, 2, 3, 4} == {x["n"] for m, _ in metas for x in m["tokens"]}, "ordinal coverage")
    _require(any(m["map_dims"][0] == 1 for m, _ in metas)
             and any(m["map_dims"][1] == 1 for m, _ in metas), "line maps missing")
    _require(len({shape for _, shape in metas}) >= 40, "canvas variation")
    _require(len({len(m["tokens"]) for m, _ in metas}) >= 7, "occupancy variation")
    _require({m["bg"] for m, _ in metas} == set(range(10)), "background role variation")
    _require({m["rotation"] for m, _ in metas} == set(range(4)), "global orientation variation")
    _require(all(_connected(x["site"] for x in m["tokens"]) for m, _ in metas), "sampler map connectivity")
    _require(_signature(puzzles[17]) == _signature(generate_puzzle(17, len(puzzles[17]["train"]),
                                                                len(puzzles[17]["test"]))), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1234 + nt, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    probe = puzzles[0]["train"][0]
    rot, canonical, m = _decode(probe["input"])
    correct = _render(canonical, m)
    for alternative in (_render(canonical, m, fixed_rank=1),
                        _render(canonical, m, invert=False),
                        _render(canonical, m, replicate=False)):
        _require(not np.array_equal(correct, alternative), "a parent ablation did not affect the answer")
    token = next(x for x in m["tokens"] if not x["anchor"] and x["n"] == 2)
    changed = canonical.copy()
    changed[token["r"] + 1, token["c"]] = m["bg"]
    altered = solve(changed)
    _require(not np.array_equal(altered, correct), "unary-rank counterfactual was ignored")
    # The rank counterfactual changes the fill of exactly one tile, not its site.
    r, c = token["out"]
    diff = altered != correct
    _require(not np.any(diff[:r]) and not np.any(diff[r + m["h"]:])
             and not np.any(diff[r:r + m["h"], :c])
             and not np.any(diff[r:r + m["h"], c + m["w"]:]), "local key had a nonlocal effect")
    for pair in puzzles[1]["train"][:2]:
        for rotation in range(4):
            for reflection in (False, True):
                a, b = pair["input"], pair["output"]
                if reflection:
                    a, b = np.fliplr(a), np.fliplr(b)
                a, b = np.rot90(a, rotation), np.rot90(b, rotation)
                palette = np.array([7, 2, 9, 1, 5, 8, 0, 4, 6, 3])
                _require(np.array_equal(solve(palette[a]), palette[b]), "geometric/color equivariance")
    # Valid semantic members outside the sampler's connected/small-rack range:
    # empty map bands, a ten-cell bar, and six competing bars. This tests that
    # solve has no dependency on the random curriculum or sampling bounds.
    for lengths, columns, sites, anchor in (
            ([3, 10], [0, 4], [(0, 0), (0, 2), (1, 0), (1, 2)], (0, 2)),
            ([8, 3, 6, 4, 7, 5], [0, 2, 4, 6, 8, 10], [(0, 0), (0, 1)], (0, 1))):
        height, width = max(lengths), max(columns) + 1
        reference = np.zeros((height, width), dtype=np.int8)
        for length, col in zip(lengths, columns):
            reference[height - length, col] = reference[height - 1, col] = 6
            reference[height - length + 1:height - 1, col] = 7
        layout = np.zeros((max(r for r, _ in sites) + 1, max(c for _, c in sites) + 1), dtype=bool)
        for p in sites:
            layout[p] = True
        ranks = {p: 1 + i % min(4, len(lengths)) for i, p in enumerate(sites)}
        case = _place(np.random.default_rng(987), reference, lengths, columns,
                      layout, anchor, ranks, (0, 6, 7))
        _check_pair(case)
        _require(np.array_equal(solve(case["input"]), case["output"]), "unsampled semantic member")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("damaged label accepted")
    return {"seeds": 42, "pairs": len(metas), "populations": sorted(populations),
            "reference_shapes": len(reference_shapes), "map_shapes": len(map_shapes),
            "anchor_positions": len(anchors), "canvas_shapes": len({s for _, s in metas}),
            "count_rank_combinations": sorted(ordinal_pairs)}
