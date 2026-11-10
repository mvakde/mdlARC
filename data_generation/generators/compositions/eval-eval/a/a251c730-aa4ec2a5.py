"""Sealed stencils: keyed motifs print only inside their request's own cavity.

Only 0 (paper) and 5 (walls) have fixed meanings. Every input supplies its
complete colored dictionary. No file access, global RNG, or saved examples.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from itertools import permutations

import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "illegal grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "illegal colors")
    return a


def _regions(mask, neighbors=D4):
    """Return components, their labels, and whether each meets the canvas edge."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    regions, edges = [], []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(regions)
        labels[r, c] = index
        todo, cells, edge = [(r, c)], [], False
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            edge |= y in (0, h - 1) or x in (0, w - 1)
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and labels[yy, xx] < 0:
                    labels[yy, xx] = index
                    todo.append((yy, xx))
        regions.append(cells)
        edges.append(edge)
    return regions, labels, edges


def _interpret(input_grid):
    g = _grid(input_grid)
    colored, _, _ = _regions((g != 0) & (g != 5))
    templates, references, anchors = {}, [], []
    for cells in colored:
        if len(cells) == 1:
            r, c = cells[0]
            anchors.append((r, c, int(g[r, c])))
            continue
        counts = Counter(int(g[r, c]) for r, c in cells)
        keys = [k for k, n in counts.items() if n == 1]
        _require(len(keys) == 1 and len(cells) >= 3, "a reference needs one singleton key")
        key = keys[0]
        ar, ac = next((r, c) for r, c in cells if g[r, c] == key)
        sig = tuple(sorted((r - ar, c - ac, int(g[r, c])) for r, c in cells))
        _require(key not in templates or templates[key] == sig, "conflicting dictionary entries")
        templates[key] = sig
        references.append(cells)
    _require(bool(templates) and bool(anchors), "dictionary and requests are required")
    _require(all(k in templates for _, _, k in anchors), "unknown request key")
    keys = set(templates)
    _require(all(color not in keys for sig in templates.values() for dr, dc, color in sig
                 if (dr, dc) != (0, 0)), "key used as ink")
    # ALL non-wall colors are passable, including markers and reference ink.
    regions, labels, edges = _regions(g != 5)
    for cells in references:
        _require(all(edges[int(labels[r, c])] for r, c in cells), "dictionary must be exterior")
    return {"templates": templates, "references": references, "anchors": anchors,
            "regions": regions, "labels": labels, "edges": edges}


def _paint(g, info, mode="correct"):
    out = g.copy()
    proposed = {}
    first = info["templates"][min(info["templates"])]
    h, w = g.shape
    for ar, ac, key in info["anchors"]:
        region = int(info["labels"][ar, ac])
        active = not info["edges"][region]
        if mode != "no_topology" and not active:
            continue
        sig = first if mode == "one_template" else info["templates"][key]
        if mode == "fill_cavity":
            ink = next(color for dr, dc, color in sig if dr or dc)
            candidates = [(r, c, ink) for r, c in info["regions"][region]]
        else:
            candidates = [(ar + dr, ac + dc, color) for dr, dc, color in sig if dr or dc]
        for r, c, color in candidates:
            if not (0 <= r < h and 0 <= c < w) or g[r, c] != 0:
                continue
            dest = int(info["labels"][r, c])
            permitted = (mode == "no_topology" or
                         (mode == "all_holes" and not info["edges"][dest]) or dest == region)
            if not permitted:
                continue
            if mode == "correct":
                _require((r, c) not in proposed or proposed[r, c] == color, "conflicting stamps")
            proposed[r, c] = color
    for (r, c), color in proposed.items():
        out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the dictionary and wall-bounded regions from this input alone."""
    g = _grid(input_grid)
    return _paint(g, _interpret(g))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _signature(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _template(rng, key, inks, bridge=False, multicolor=False):
    if bridge:
        # A path across a one-cell partition plus two surviving vertical cells.
        cells = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}
        h, w = 4, 4
    else:
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    n = int(rng.integers(max(5 if multicolor else 3, len(cells)), min(13, h * w + 1)))
    for _ in range(n - len(cells)):
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in D4
                           if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
        cells.add(_choose(rng, frontier))
    ar, ac = (0, 0) if bridge else _choose(rng, sorted(cells))
    offsets = sorted((r - ar, c - ac) for r, c in cells if (r, c) != (ar, ac))
    ncolors = min(len(inks), 2 if multicolor else (2 if len(offsets) >= 4 and rng.random() < .4 else 1))
    colors = list(map(int, rng.choice(inks, size=ncolors, replace=False)))
    assignment = colors * 2 + [_choose(rng, colors) for _ in range(len(offsets) - 2 * ncolors)]
    rng.shuffle(assignment)
    return tuple(sorted([(0, 0, key)] + [(r, c, int(v)) for (r, c), v in zip(offsets, assignment)]))


def _reference(template):
    rows, cols, _ = zip(*template)
    g = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=np.int8)
    for r, c, color in template:
        g[r - min(rows), c - min(cols)] = color
    return g, g.copy()


def _render_module(body, requests):
    """Generation labels use explicit carved cavity masks, not the global solver."""
    inp = np.where(body, 5, 0).astype(np.int8)
    for ar, ac, template, region in requests:
        key = next(v for r, c, v in template if (r, c) == (0, 0))
        inp[ar, ac] = key
    out = inp.copy()
    for ar, ac, template, region in requests:
        if region is None:
            continue
        for dr, dc, color in template:
            r, c = ar + dr, ac + dc
            if 0 <= r < len(body) and 0 <= c < body.shape[1] and region[r, c] and inp[r, c] == 0:
                out[r, c] = color
    return inp, out


def _candidate_anchors(pocket, template, full=False, clipped=False, forbidden=()):
    options = []
    h, w = pocket.shape
    n = len(template) - 1
    for ar, ac in np.argwhere(pocket):
        ar, ac = int(ar), int(ac)
        painted = {(ar + dr, ac + dc) for dr, dc, _ in template if (dr or dc)
                   and 0 <= ar + dr < h and 0 <= ac + dc < w and pocket[ar + dr, ac + dc]}
        if painted and not painted.intersection(forbidden) and (not full or len(painted) == n) and (not clipped or len(painted) < n):
            options.append((ar, ac))
    return options


def _chamber(rng, template, full=False, clipped=False, irregular=False):
    rows, cols, _ = zip(*template)
    if full:
        ph, pw = max(rows) - min(rows) + 3, max(cols) - min(cols) + 3
    else:
        ph, pw = int(rng.integers(3, 7)), int(rng.integers(3, 8))
    thickness = int(rng.integers(1, 3))
    pocket = np.zeros((ph + 2 * thickness, pw + 2 * thickness), dtype=bool)
    pocket[thickness:-thickness, thickness:-thickness] = True
    if irregular and not full and min(ph, pw) >= 4:
        # L-shaped cavity, with a broad orthogonal connection at its elbow.
        a, b = int(rng.integers(1, ph - 1)), int(rng.integers(1, pw - 1))
        pocket[thickness + a:thickness + ph, thickness + b:thickness + pw] = False
    options = _candidate_anchors(pocket, template, full=full, clipped=clipped)
    if not options:
        # The 3x3 mandatory bridge always has a clipped position in this pocket.
        pocket[thickness:-thickness, thickness:-thickness] = True
        options = _candidate_anchors(pocket, template, full=full, clipped=clipped)
    _require(bool(options), "no chamber anchor")
    ar, ac = _choose(rng, options)
    body = ~pocket
    # Optional asymmetric body appendage, independent of cavity topology.
    if rng.random() < .4:
        body = np.pad(body, ((0, 1), (0, 0)))
        pocket = np.pad(pocket, ((0, 1), (0, 0)))
        start = int(rng.integers(0, body.shape[1] - 1))
        body[-1, start:start + int(rng.integers(1, 3))] = True
    return body, pocket, ar, ac


def _open_module(body, pocket, ar, ac, template):
    # Open a straight orthogonal door from a pocket cell to the exterior.
    for r, c in np.argwhere(pocket):
        candidate = body.copy()
        candidate[:int(r) + 1, int(c)] = False
        if len(_regions(candidate)[0]) == 1:
            regs, labs, edges = _regions(~candidate)
            if edges[int(labs[ar, ac])]:
                return _render_module(candidate, [(ar, ac, template, None)])
    raise AssertionError("could not open chamber")


def _partitioned(rng, templates, irregular):
    """One connected wall body, but independent print domains in its pockets."""
    h = int(rng.integers(8, 11))
    widths = [int(rng.integers(3, 6)) for _ in templates]
    w = sum(widths) + len(widths) + 1
    body = np.ones((h, w), dtype=bool)
    pockets, start = [], 1
    for i, width in enumerate(widths):
        p = np.zeros_like(body)
        p[1:-1, start:start + width] = True
        if irregular and i > 0:
            p[1:3, start + width - 1] = False
        pockets.append(p)
        body[p] = False
        start += width + 1
    ar, ac = 2, widths[0]
    requests = [(ar, ac, templates[0], pockets[0])]
    # Keep at least this crossing cell empty: global-hole clipping would paint it.
    forbidden = {(ar, ac + 2)}
    for t, p in zip(templates[1:], pockets[1:]):
        options = _candidate_anchors(p, t, forbidden=forbidden)
        _require(bool(options), "partition anchor missing")
        # Prefer positions remote from the first stencil's crossing witness.
        options.sort(key=lambda q: -abs(q[0] - ar))
        r, c = _choose(rng, options[:max(1, len(options) // 3)])
        requests.append((r, c, t, p))
    return _render_module(body, requests)


def _orient_module(rng, module):
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    return tuple(np.fliplr(np.rot90(g, k)).copy() if flip else np.rot90(g, k).copy() for g in module)


def _pack(rng, modules):
    """Bounded random free placement, with a finite shelf fallback."""
    options = []
    margin = int(rng.integers(0, 3))
    for attempt in range(8):
        order = list(map(int, rng.permutation(len(modules))))
        if attempt == 0:
            order.sort(key=lambda i: -modules[i][0].shape[0])
        for width in range(max(m[0].shape[1] for m in modules) + 2 * margin, 31):
            y, x, row_h, places = margin, margin, 0, []
            for i in order:
                mh, mw = modules[i][0].shape
                if x + mw > width - margin:
                    y, x, row_h = y + row_h + 1, margin, 0
                places.append((i, y, x))
                x += mw + 1
                row_h = max(row_h, mh)
            height = y + row_h + margin
            if height <= 30:
                options.append((height, width, places))
    if not options and margin:
        # No retries or files: remove optional outer margin, keeping all modules.
        return _pack_with_zero_margin(rng, modules)
    _require(bool(options), "modules exceed finite canvas capacity")
    # Favor compact but retain tall, wide, and full-size arrangements.
    options.sort(key=lambda p: p[0] * p[1])
    chosen = _choose(rng, options[:max(1, len(options) // 2)])
    h, w, fallback = chosen
    h = int(rng.integers(h, min(30, h + 3) + 1))
    w = int(rng.integers(w, min(30, w + 3) + 1))
    order = sorted(range(len(modules)), key=lambda i: -modules[i][0].size)
    places = None
    for _ in range(3):
        occupied = np.zeros((h, w), dtype=bool)
        trial = []
        for i in order:
            mh, mw = modules[i][0].shape
            candidates = [(r, c) for r in range(margin, h - mh - margin + 1)
                          for c in range(margin, w - mw - margin + 1)
                          if not occupied[r:r + mh, c:c + mw].any()]
            if not candidates:
                break
            r, c = _choose(rng, candidates)
            trial.append((i, r, c))
            occupied[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)] = True
        else:
            places = trial
            break
    if places is None:
        dy = int(rng.integers(0, h - chosen[0] + 1))
        dx = int(rng.integers(0, w - chosen[1] + 1))
        places = [(i, r + dy, c + dx) for i, r, c in fallback]
    a, b = np.zeros((h, w), dtype=np.int8), np.zeros((h, w), dtype=np.int8)
    for i, r, c in places:
        inp, out = modules[i]
        mh, mw = inp.shape
        a[r:r + mh, c:c + mw] = inp
        b[r:r + mh, c:c + mw] = out
    # This moves dictionary and targets together: copying remains translation.
    return _orient_module(rng, (a, b))


def _pack_with_zero_margin(rng, modules):
    """Guaranteed fallback: enumerate finite shelf orders with no outer margin."""
    # Called only when the stricter 1/2-cell outer-margin search failed.
    # At most six modules: <=720 orders, with immediate exit on the first fit.
    # Exhaustion removes the chance that random order trials miss a shelf fit.
    for order in permutations(range(len(modules))):
        for width in range(max(m[0].shape[1] for m in modules), 31):
            y, x, rh, places = 0, 0, 0, []
            for i in order:
                mh, mw = modules[i][0].shape
                if x + mw > width:
                    y, x, rh = y + rh + 1, 0, 0
                places.append((i, y, x))
                x, rh = x + mw + 1, max(rh, mh)
            if y + rh <= 30:
                a, b = np.zeros((y + rh, width), np.int8), np.zeros((y + rh, width), np.int8)
                for i, r, c in places:
                    m, n = modules[i]
                    a[r:r + len(m), c:c + m.shape[1]] = m
                    b[r:r + len(n), c:c + n.shape[1]] = n
                return _orient_module(rng, (a, b))
    raise AssertionError("shelf capacity exhausted")


def _make_pair(rng, stage, test=False):
    nkeys = 3 if test else (1 if stage == 0 else 2 if stage == 1 else int(rng.integers(1, 3)))
    palette = list(map(int, rng.permutation([1, 2, 3, 4, 6, 7, 8, 9])))
    keys, inks = palette[:nkeys], palette[nkeys:nkeys + 3]
    templates = []
    for i, key in enumerate(keys):
        allowed = inks
        if stage == 1 and not test and i == 1:
            # A finite teaching guarantee: any surviving wrong-key print has
            # the wrong ink, even if two randomly grown fragments coincide.
            used = {color for dr, dc, color in templates[0] if dr or dc}
            allowed = [color for color in inks if color not in used]
        templates.append(_template(rng, key, allowed, bridge=i == 0,
                                   multicolor=(stage == 1 and i == 0)))
    modules = [_reference(t) for t in templates]
    if stage == 0 and not test:
        # Exact matched topology contrast, plus a complete transparent print.
        body, pocket, ar, ac = _chamber(rng, templates[0], clipped=True)
        modules.append(_render_module(body, [(ar, ac, templates[0], pocket)]))
        modules.append(_open_module(body, pocket, ar, ac, templates[0]))
        body, pocket, ar, ac = _chamber(rng, templates[0], full=True)
        modules.append(_render_module(body, [(ar, ac, templates[0], pocket)]))
        # Two completed occurrences show that the key denotes a repeatable stamp.
        modules.append(_reference(templates[0]))
    elif stage == 1 and not test:
        modules.append(_partitioned(rng, templates, irregular=True))
        body, pocket, ar, ac = _chamber(rng, templates[1], full=True)
        modules.append(_render_module(body, [(ar, ac, templates[1], pocket)]))
    elif test:
        if rng.random() < .5:
            modules.append(_partitioned(rng, templates, irregular=True))
        else:
            modules.append(_partitioned(rng, templates[:2], irregular=True))
            body, pocket, ar, ac = _chamber(rng, templates[2], irregular=True)
            modules.append(_render_module(body, [(ar, ac, templates[2], pocket)]))
        if rng.random() < .6:
            t = _choose(rng, templates)
            body, pocket, ar, ac = _chamber(rng, t, irregular=True)
            modules.append(_open_module(body, pocket, ar, ac, t))
    else:
        for i, t in enumerate(templates):
            body, pocket, ar, ac = _chamber(rng, t, irregular=True)
            modules.append(_render_module(body, [(ar, ac, t, pocket)]))
            if i == 0 and rng.random() < .65:
                modules.append(_open_module(body, pocket, ar, ac, t))
        if rng.random() < .3:
            modules.append(_reference(_choose(rng, templates)))
    # Keep each motif's original orientation while placing modules freely.
    inp, out = _pack(rng, modules)
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError("counts must be integers in train=2..10 and test=1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for attempt in range(16):
                pair = _make_pair(rng, i, split == "test")
                sig = _signature(pair["input"])
                if sig not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input attempts exhausted")
            seen.add(sig)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(g, info):
    active, inactive, clipped, full, cross = 0, 0, 0, 0, 0
    for ar, ac, key in info["anchors"]:
        reg = int(info["labels"][ar, ac])
        if info["edges"][reg]:
            inactive += 1
            continue
        active += 1
        kept = 0
        for dr, dc, color in info["templates"][key]:
            if not (dr or dc):
                continue
            r, c = ar + dr, ac + dc
            if 0 <= r < len(g) and 0 <= c < g.shape[1] and g[r, c] == 0:
                dest = int(info["labels"][r, c])
                kept += dest == reg
                cross += dest != reg and not info["edges"][dest]
        full += kept == len(info["templates"][key]) - 1
        clipped += 0 < kept < len(info["templates"][key]) - 1
    holes = [cells for cells, edge in zip(info["regions"], info["edges"]) if not edge]
    irregular = sum(len(cells) != (max(r for r, c in cells) - min(r for r, c in cells) + 1)
                    * (max(c for r, c in cells) - min(c for r, c in cells) + 1) for cells in holes)
    return {"keys": len(info["templates"]), "anchors": len(info["anchors"]), "active": active,
            "inactive": inactive, "clipped": clipped, "full": full, "cross": cross,
            "holes": len(holes), "irregular": irregular, "walls": len(_regions(g == 5)[0])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, teaching = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(g.shape == out.shape and _signature(g) not in seen, "duplicate input or wrong shape")
            seen.add(_signature(g))
            info = _interpret(g)
            expected = _paint(g, info)
            _require(np.array_equal(expected, out), "incorrect cavity-clipped stamp")
            _require(np.any(g != out), "pair must show printing")
            _require(np.all(out[g != 0] == g[g != 0]), "original colored cells changed")
            # Exclude diagonal-only topology and collective/nested enclosures.
            padded = np.pad(g == 5, 1)
            wall4 = _regions(padded)[0]
            _require(len(wall4) == len(_regions(padded, D8)[0]), "diagonally touching walls")
            regs8, labs8, edges8 = _regions(g != 5, D8)
            hole4 = np.zeros(g.shape, bool)
            hole8 = np.zeros(g.shape, bool)
            for cells, edge in zip(info["regions"], info["edges"]):
                if not edge:
                    rr, cc = zip(*cells)
                    hole4[rr, cc] = True
                    ids = {int(labs8[r, c]) for r, c in cells}
                    _require(len(ids) == 1, "diagonal cavity split")
            for cells, edge in zip(regs8, edges8):
                if not edge:
                    rr, cc = zip(*cells)
                    hole8[rr, cc] = True
            _require(np.array_equal(hole4, hole8), "diagonal exterior leak")
            _require(len([e for e in info["edges"] if not e]) == len([e for e in edges8 if not e]), "diagonal cavity merge")
            # Every sealed region is requested once; no hidden overlap rule.
            active_ids = [int(info["labels"][r, c]) for r, c, k in info["anchors"]
                          if not info["edges"][int(info["labels"][r, c])]]
            _require(len(active_ids) == len(set(active_ids)) == sum(not e for e in info["edges"]),
                     "one request per sealed region is required")
            owned = set()
            for cells in wall4:
                mask = np.zeros_like(padded)
                rr, cc = zip(*cells)
                mask[rr, cc] = True
                regs, labels, edges = _regions(~mask)
                local_holes = {(r - 1, c - 1) for region, edge in zip(regs, edges) if not edge for r, c in region}
                _require(all(hole4[r, c] for r, c in local_holes), "nested or unrelated enclosure")
                _require(not owned.intersection(local_holes), "shared cavity ownership")
                owned.update(local_holes)
            _require(owned == set(map(tuple, np.argwhere(hole4).tolist())), "collective enclosure")
            f = _features(g, info)
            _require(f["active"] >= 1, "no active request")
            if split == "train":
                teaching.append((g, out, info, f))
    _require(any(f["inactive"] and f["full"] and f["clipped"] for _, _, _, f in teaching), "missing open/full/clipped teaching contrast")
    _require(any(f["cross"] for _, _, _, f in teaching), "missing same-cavity teaching witness")
    for mode in ("no_topology", "all_holes", "fill_cavity", "one_template"):
        _require(any(not np.array_equal(_paint(g, info, mode), out) for g, out, info, f in teaching),
                 "training does not distinguish shortcut: " + mode)
    _require(any(f["irregular"] for _, _, _, f in teaching), "irregular cavities must be taught")


def _puzzle_hash(p):
    h = hashlib.sha256()
    for s in ("train", "test"):
        for pair in p[s]:
            for role in ("input", "output"):
                g = np.asarray(pair[role])
                h.update(bytes(g.shape))
                h.update(g.astype(np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """Bounded: 42 puzzles streamed through memory, no artifact writes."""
    dims, shapes, cavities, hashes, palettes, profiles = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        hashes.add(_puzzle_hash(p))
        for split in ("train", "test"):
            for pair in p[split]:
                g = pair["input"]
                _require(np.array_equal(solve(g.copy()), pair["output"]), "input-only solve mismatch")
                info = _interpret(g)
                f = _features(g, info)
                totals.update(f)
                profiles.add((f["keys"], f["anchors"], f["walls"], f["holes"], f["inactive"]))
                dims.add(g.shape)
                palettes.add(tuple(sorted(info["templates"])))
                for sig in info["templates"].values():
                    shapes.add(tuple((r, c) for r, c, color in sig))
                for cells, edge in zip(info["regions"], info["edges"]):
                    if not edge:
                        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
                        cavities.add(tuple(sorted((r - r0, c - c0) for r, c in cells)))
        if seed == 0:
            _require(_puzzle_hash(p) == _puzzle_hash(generate_puzzle(seed)), "nondeterminism")
    _require(len(hashes) == 42 and len(dims) >= 30, "puzzle/geometry variation missing")
    _require(len(shapes) >= 100 and len(cavities) >= 30, "motif/cavity variation missing")
    _require(len(palettes) >= 40 and len(profiles) >= 10, "palette/count variation missing")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(700 + a + b, a, b)
    for a, b in ((1, 1), (11, 1), (2, 2), (4, 0), (5, 5), (6, 5), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(311, 2, 1)
    for role in ("print", "wall", "reference", "empty"):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = q["train"][0]
        g, out = pair["input"], pair["output"]
        info = _interpret(g)
        if role == "print":
            r, c = np.argwhere(g != out)[0]
        elif role == "wall":
            r, c = np.argwhere(g == 5)[0]
        elif role == "reference":
            r, c = info["references"][0][0]
        else:
            r, c = np.argwhere((g == 0) & (out == 0))[0]
        out[r, c] = (int(out[r, c]) + 1) % 10
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupt " + role)
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes), "grid_shapes": len(dims),
            "motif_geometries": len(shapes), "cavity_geometries": len(cavities),
            "key_palettes": len(palettes), "count_profiles": len(profiles), "feature_totals": dict(totals)}
