"""Stencil pulse networks: simultaneous, scale-aware parity state exchange.

Only Python's standard library and NumPy are used. Generation writes no files.
The gray (5) stencil is a pulse map, not paint. Its source label selects the
emitting state; its target label identifies the invariant registration block.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np


NEUTRAL = 5
STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _bounds(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _components(points):
    left = set(points)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in left:
                        left.remove(p)
                        group.add(p)
                        stack.append(p)
        result.append(group)
    return result


def _signature(mapping):
    return tuple(sorted((r, c, int(v)) for (r, c), v in mapping.items()))


def _check_grid(grid):
    array = np.asarray(grid)
    _require(array.ndim == 2 and all(1 <= x <= 30 for x in array.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= array) & (array <= 9))), "colors must be 0..9")
    return array.astype(int, copy=False)


def _read_input(input_grid):
    """Decode all instructions, templates, scales and hits from input pixels."""
    grid = _check_grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(int(counts[bg]) * 2 > grid.size and bg != NEUTRAL,
             "background must be a strict majority and cannot be gray")
    mask = {tuple(map(int, p)) for p in np.argwhere(grid == NEUTRAL)}
    _require(mask and len(_components(mask)) == 1, "one connected gray stencil required")
    near = {(r + dr, c + dc) for r, c in mask
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    candidates = []
    for sr, sc in np.argwhere((grid != bg) & (grid != NEUTRAL)):
        sr, sc = int(sr), int(sc)
        for dr, dc in STEPS:
            tr, tc = sr + dr, sc + dc
            q = sr + 2 * dr, sc + 2 * dc
            if q not in near or not (0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1]):
                continue
            a, receptor = int(grid[sr, sc]), int(grid[tr, tc])
            if receptor in (bg, NEUTRAL, a) or int(grid[q]) not in (bg, NEUTRAL):
                continue
            candidates.append((a, receptor, (sr, sc), (tr, tc), q, (dr, dc)))
    _require(len(candidates) == 1, "stencil label ray must be unique")
    source, receptor, source_label, target_label, q, direction = candidates[0]
    instruction = mask | {source_label, target_label}
    offsets = frozenset((r - q[0], c - q[1]) for r, c in mask)
    foreground = {tuple(map(int, p)) for p in np.argwhere(grid != bg)} - instruction
    objects = []
    for group in _components(foreground):
        ports = {p for p in group if int(grid[p]) == receptor}
        _require(ports, "every motif needs a receptor")
        r0, r1, c0, c1 = _bounds(ports)
        scale = r1 - r0 + 1
        _require(c1 - c0 + 1 == scale and len(ports) == scale * scale,
                 "receptor must be one solid square")
        mapping = {}
        blocks = {}
        for r, c in group:
            logical = (r - r0) // scale, (c - c0) // scale
            blocks.setdefault(logical, []).append(int(grid[r, c]))
        for p, values in blocks.items():
            _require(len(values) == scale * scale and len(set(values)) == 1,
                     "motif is not uniformly pixel-scaled relative to its receptor")
            mapping[p] = values[0]
        _require(mapping.get((0, 0)) == receptor and
                 sum(v == receptor for v in mapping.values()) == 1,
                 "each logical motif must have exactly one receptor")
        objects.append({"anchor": (r0, c0), "scale": scale,
                        "signature": _signature(mapping)})
    _require(len(objects) >= 3, "at least three motifs required")
    signatures = sorted({o["signature"] for o in objects})
    _require(len(signatures) == 2, "exactly two visible normalized states required")
    templates = [{(r, c): v for r, c, v in sig} for sig in signatures]
    emitting = [i for i, t in enumerate(templates) if source in t.values()]
    _require(len(emitting) == 1, "source label must identify exactly one state")
    common = {p: v for p, v in templates[0].items() if templates[1].get(p) == v}
    _require(len(common) >= 3 and common.get((0, 0)) == receptor,
             "two states need a shared scaffold containing the receptor")
    minimum = min(o["scale"] for o in objects)
    _require({o["signature"] for o in objects if o["scale"] == minimum} == set(signatures),
             "both states must be visible at the minimum scale")
    union = set(templates[0]) | set(templates[1])
    rects = []
    for obj in objects:
        obj["state"] = signatures.index(obj["signature"])
        obj["active"] = obj["state"] == emitting[0]
        r0, r1, c0, c1 = _bounds(union)
        r, c = obj["anchor"]
        s = obj["scale"]
        rect = r + r0 * s, c + c0 * s, (r1 - r0 + 1) * s, (c1 - c0 + 1) * s
        y, x, h, w = rect
        _require(0 <= y and y + h <= grid.shape[0] and
                 0 <= x and x + w <= grid.shape[1], "replacement may not be clipped")
        _require(all(_separated(rect, old) for old in rects), "replacement rectangles must be separated")
        _require(not any(y <= r < y + h and x <= c < x + w for r, c in instruction),
                 "instruction intersects a replacement rectangle")
        rects.append(rect)
    hits = _count_hits(objects, offsets)
    return {"grid": grid, "background": bg, "source": source, "receptor": receptor,
            "instruction": instruction, "offsets": offsets, "q": q, "direction": direction,
            "templates": templates, "common": common, "objects": objects, "hits": hits}


def _count_hits(objects, offsets, scaled=True):
    hits = [0] * len(objects)
    for emitter in objects:
        if not emitter["active"]:
            continue
        s = emitter["scale"] if scaled else 1
        er, ec = emitter["anchor"]
        for j, receiver in enumerate(objects):
            # Equality of the complete receptor square, not partial overlap.
            if scaled and receiver["scale"] != s:
                continue
            rr, rc = receiver["anchor"]
            dr, dc = rr - er, rc - ec
            if dr % s == 0 and dc % s == 0 and (dr // s, dc // s) in offsets:
                hits[j] += 1
    return hits


def _render(grid, template, anchor, scale):
    for (dr, dc), color in template.items():
        r, c = anchor[0] + dr * scale, anchor[1] + dc * scale
        grid[r:r + scale, c:c + scale] = color


def _answer(info, hits=None, mode="parity"):
    out = info["grid"].copy()
    bg, templates = info["background"], info["templates"]
    for p in info["instruction"]:
        out[p] = bg
    union = set(templates[0]) | set(templates[1])
    hits = info["hits"] if hits is None else hits
    for obj, n in zip(info["objects"], hits):
        flip = n % 2 if mode == "parity" else n > 0 if mode == "any" else True
        for dr, dc in union:
            r, c = obj["anchor"][0] + dr * obj["scale"], obj["anchor"][1] + dc * obj["scale"]
            out[r:r + obj["scale"], c:c + obj["scale"]] = bg
        _render(out, templates[obj["state"] ^ int(flip)], obj["anchor"], obj["scale"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """One synchronous update; no training examples or generation state."""
    return _answer(_read_input(input_grid))


def _separated(a, b, gap=1):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h + gap <= yy or yy + hh + gap <= y or x + w + gap <= xx or xx + ww + gap <= x


def _design(rng, colors, change, compact=False):
    receptor, carrier, a, b, stable = colors
    if compact:
        core = {(0, 0), (0, 1), (1, 0), (1, 1)}
        port = (0, 0)
        moving = {(1, 1)} if change != "relocate" else set()
        deco = [{(-1, 0)}, {(0, -1)}] if change != "recolor" else [{(-1, 0)}] * 2
    else:
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        # Grow an irregular scaffold, always keeping a connected 2x2 seed.
        core = {(0, 0), (0, 1), (1, 0), (1, 1)}
        for _ in range(int(rng.integers(0, 6))):
            frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in STEPS
                               if 0 <= r + dr < h and 0 <= c + dc < w} - core)
            if frontier:
                core.add(_pick(rng, frontier))
        port = _pick(rng, sorted(core))
        interior = [p for p in sorted(core) if p != port]
        rng.shuffle(interior)
        moving = set(interior[:int(rng.integers(1, min(2, len(core) - 3) + 1))]) if change != "relocate" else set()
        sides = list(rng.permutation(4))
        deco = []
        for state in range(2):
            if state and change == "recolor":
                deco.append(deco[0])
                break
            dr, dc = STEPS[int(sides[state])]
            edge = max(r * dr + c * dc for r, c in core)
            roots = sorted(p for p in core if p[0] * dr + p[1] * dc == edge)
            root = _pick(rng, roots)
            patch = {(root[0] + dr, root[1] + dc)}
            if rng.random() < .45:
                lateral = (dc, -dr)
                p = next(iter(patch))
                patch.add((p[0] + lateral[0], p[1] + lateral[1]))
            deco.append(patch)
    common = {p: carrier for p in core - moving}
    common[port] = receptor
    stable_points = [p for p in sorted(common) if p != port]
    if rng.random() < .35 and len(stable_points) >= 3:
        common[_pick(rng, stable_points)] = stable
    result = []
    for state, accent in enumerate((a, b)):
        template = dict(common)
        template.update({p: accent for p in moving | deco[state]})
        template = {(r - port[0], c - port[1]): v for (r, c), v in template.items()}
        result.append(template)
    return result


def _pulse_mask(rng, u, v, neighbors, self_hit):
    """Join desired displacement pixels by varied thin lattice paths."""
    cells = {(0, 0)}
    for nr, nc in sorted(neighbors):
        target = nr * u[0] + nc * v[0], nr * u[1] + nc * v[1]
        r = c = 0
        style = int(rng.integers(3))
        while (r, c) != target:
            dr = int(np.sign(target[0] - r))
            dc = int(np.sign(target[1] - c))
            if dr and dc:
                if style == 0:
                    r += dr
                elif style == 1:
                    c += dc
                else:
                    r, c = r + dr, c + dc
            else:
                r, c = r + dr, c + dc
            cells.add((r, c))
    if not self_hit:
        cells.remove((0, 0))
    return frozenset(cells)


def _pack(rng, sizes):
    """Bounded shelf proposals followed by independent block moves."""
    options = []
    orders = [sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)]
    orders += [list(map(int, rng.permutation(len(sizes)))) for _ in range(3)]
    for width in range(max(w for h, w in sizes), 31):
        for order in orders:
            rects = [None] * len(sizes)
            y = x = shelf = 0
            for i in order:
                h, w = sizes[i]
                if x and x + w > width:
                    y, x, shelf = y + shelf + 1, 0, 0
                if y + h > 30:
                    break
                rects[i] = y, x, h, w
                x += w + 1
                shelf = max(shelf, h)
            else:
                options.append(rects)
    if not options:
        return None
    rects = _pick(rng, options)
    h0, w0 = max(y + h for y, x, h, w in rects), max(x + w for y, x, h, w in rects)
    height = int(rng.integers(h0, min(30, h0 + 6) + 1))
    width = int(rng.integers(w0, min(30, w0 + 6) + 1))
    for _ in range(40):
        i = int(rng.integers(len(rects)))
        _, _, h, w = rects[i]
        trial = int(rng.integers(height - h + 1)), int(rng.integers(width - w + 1)), h, w
        if all(_separated(trial, other) for j, other in enumerate(rects) if i != j):
            rects[i] = trial
    return (height, width), rects


def _construct(rng, mode, attempt):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != NEUTRAL])]
    bg, receptor, carrier, a, b, stable = palette[:6]
    main_scale = (1 if mode == "first" else 2 if mode == "second" else
                  _pick(rng, [1, 1, 1, 2, 2, 3] if mode == "test" else [1, 1, 2]))
    compact = mode == "second" or main_scale == 3 or attempt >= 30
    change = ("recolor" if mode == "first" else "combined" if mode == "second" else
              _pick(rng, ["recolor", "relocate", "combined", "combined"]))
    templates = _design(rng, (receptor, carrier, a, b, stable), change, compact)
    union = set(templates[0]) | set(templates[1])
    r0, r1, c0, c1 = _bounds(union)
    mh, mw = r1 - r0 + 1, c1 - c0 + 1
    py, px = mh + int(rng.integers(1, 3)), mw + int(rng.integers(1, 3))
    if main_scale == 3 or attempt >= 20:
        py, px = mh + 1, mw + 1
    shear = int(rng.integers(0, 3)) if main_scale == 1 else 0
    u, v = (py, 0), (shear, px)
    if mode == "second":
        rows, cols = 2, 3
    elif mode == "test" and main_scale < 3:
        rows, cols = ((2, 3) if attempt >= 15 else
                      _pick(rng, [(2, 3), (3, 2), (3, 3), (2, 4)]))
    elif mode == "first" or main_scale == 3 or attempt >= 15:
        rows = cols = 2
    elif main_scale == 2:
        rows, cols = _pick(rng, [(2, 2), (2, 3), (3, 2)])
    else:
        rows, cols = _pick(rng, [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)])
    sites = {(0, 0), (0, 1), (1, 0), (1, 1)}
    sites |= {(r, c) for r in range(rows) for c in range(cols) if rng.random() < .75}
    far = None
    if mode == "second":
        sites = {(r, c) for r in range(2) for c in range(3)}
    elif mode == "test" and main_scale < 3:
        far = (2, 0) if rows > 2 and (cols == 2 or rng.random() < .5) else (0, 2)
        sites.add(far)
    sites = sorted(sites)
    neighbors = {(0, 1), (1, 0)}
    if mode in ("test", "second"):
        neighbors.add((1, 1))
        if far is not None:
            neighbors.add(far)
    elif mode == "free":
        neighbors = set(_pick(rng, [[(0, 1)], [(1, 0)], [(1, 1)], [(0, 1), (1, 0)], [(0, 1), (1, 1)]]))
    if mode not in ("first", "second"):
        if cols > 2 and rng.random() < .6:
            neighbors.add((0, 2))
        if rows > 2 and rng.random() < .6:
            neighbors.add((2, 0))
    self_hit = mode == "first" or mode not in ("first", "second") and rng.random() < .4
    offsets = _pulse_mask(rng, u, v, neighbors, self_hit)
    if len(_components(offsets)) != 1:
        return None
    states = []
    for site in sites:
        if mode == "first":
            state = int(site != (0, 0))
        elif mode == "second":
            state = int(site[0] == 1 and site[1] >= 1)
        elif far is not None and site == (0, 0):
            state = 0
        else:
            state = int(rng.random() < float(_pick(rng, [.25, .5, .75])))
        states.append(state)
    if set(states) != {0, 1}:
        return None
    # Relative port positions for a sparse/sheared constellation, not tile cells.
    objects = [{"anchor": ((r * py + c * shear - r0) * main_scale,
                            (c * px - c0) * main_scale),
                "scale": main_scale, "state": state, "active": state == 0}
               for (r, c), state in zip(sites, states)]
    field_h = max(o["anchor"][0] + (r1 + 1) * main_scale for o in objects)
    field_w = max(o["anchor"][1] + (c1 + 1) * main_scale for o in objects)
    # Smaller reference copies show both states when the main network is enlarged.
    ref_scales = [1, 1] if main_scale > 1 else ([] if rng.random() < .5 else [int(_pick(rng, [1, 2]))] * 2)
    instruction_points = set(offsets) | {(0, -2), (0, -1), (0, 0)}
    ir0, ir1, ic0, ic1 = _bounds(instruction_points)
    sizes = [(field_h, field_w), (ir1 - ir0 + 1, ic1 - ic0 + 1)]
    sizes += [(mh * s, mw * s) for s in ref_scales]
    if any(h > 30 or w > 30 for h, w in sizes):
        return None
    packed = _pack(rng, sizes)
    if packed is None:
        return None
    shape, rects = packed
    fy, fx, _, _ = rects[0]
    for obj in objects:
        obj["anchor"] = obj["anchor"][0] + fy, obj["anchor"][1] + fx
    for i, (s, (y, x, _, _)) in enumerate(zip(ref_scales, rects[2:])):
        objects.append({"anchor": (y - r0 * s, x - c0 * s), "scale": s,
                        "state": i % 2, "active": i % 2 == 0})
    hits = _count_hits(objects, offsets)
    if not any(n % 2 for n in hits) or not any(n % 2 == 0 for n in hits):
        return None
    if mode == "first" and hits[:4] != [1, 1, 1, 0]:
        # Extra references may join the same network, but cannot obscure the
        # four-site demonstration of one synchronous wave.
        return None
    if mode == "first" and not (any(n == 0 for n in hits) and
                                   {o["state"] for o, n in zip(objects, hits) if n % 2} == {0, 1}):
        return None
    if mode == "second" and not (2 in hits and 3 in hits and
                                    any(n % 2 and o["scale"] == 2 for o, n in zip(objects, hits))):
        return None
    inp, out = np.full(shape, bg, dtype=int), np.full(shape, bg, dtype=int)
    for obj, n in zip(objects, hits):
        _render(inp, templates[obj["state"]], obj["anchor"], obj["scale"])
        _render(out, templates[obj["state"] ^ (n % 2)], obj["anchor"], obj["scale"])
    y, x, _, _ = rects[1]
    q = y - ir0, x - ic0
    for dr, dc in offsets:
        inp[q[0] + dr, q[1] + dc] = NEUTRAL
    inp[q[0], q[1] - 2], inp[q[0], q[1] - 1] = a, receptor
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    pair = {"input": inp.copy(), "output": out.copy()}
    try:
        info = _read_input(inp)
        if not np.array_equal(_answer(info), out):
            return None
    except AssertionError:
        return None
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            mode = ("first" if i == 0 else "second" if i == 1 else "free") if split == "train" else "test"
            for attempt in range(80):
                pair = _construct(rng, mode, attempt)
                if pair is None:
                    continue
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded pulse-network construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    templates, objects, hits = info["templates"], info["objects"], info["hits"]
    union = set(templates[0]) | set(templates[1])
    source = info["source"]
    # Color-normalized complete registered state vocabulary, independent of state order.
    normalized = []
    roles = {info["receptor"]: 0, source: 1}
    for template in sorted(templates, key=lambda t: source not in t.values()):
        normalized.append(tuple((r, c, roles.setdefault(v, len(roles)))
                                for r, c, v in _signature(template)))
    return {"shape": tuple(info["grid"].shape), "background": info["background"],
            "count": len(objects), "scales": tuple(sorted(o["scale"] for o in objects)),
            "hits": tuple(sorted(hits)), "mask": tuple(sorted(info["offsets"])),
            "hole": (0, 0) not in info["offsets"], "direction": info["direction"],
            "vocabulary": tuple(normalized), "scaffold": tuple(sorted(info["common"])),
            "relocate": set(templates[0]) != set(templates[1]),
            "recolor": any(templates[0][p] != templates[1][p] for p in set(templates[0]) & set(templates[1])),
            "union_size": len(union)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            key = inp.shape, inp.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _read_input(inp)
            _require(np.array_equal(_answer(info), out), "incorrect pulse-state output")
            _require(any(n % 2 for n in info["hits"]), "network must change a motif")
            _require(any(n % 2 == 0 for n in info["hits"]), "network must preserve a motif")
            # Every scaffold pixel stays, all instruction pixels disappear.
            for p in info["instruction"]:
                _require(out[p] == info["background"], "instruction not erased")
            for obj in info["objects"]:
                for (dr, dc), color in info["common"].items():
                    r, c = obj["anchor"][0] + dr * obj["scale"], obj["anchor"][1] + dc * obj["scale"]
                    _require(bool(np.all(out[r:r + obj["scale"], c:c + obj["scale"]] == color)),
                             "scaffold moved or recolored")
            if split == "train":
                training.append(info)
    _require(any((0, 0) in i["offsets"] for i in training) and
             any((0, 0) not in i["offsets"] for i in training), "training must teach occupied and hole registration")
    _require(any(any(n > 0 and n % 2 == 0 for n in i["hits"]) for i in training),
             "training must demonstrate positive even cancellation")
    _require(any(3 in i["hits"] for i in training), "training must distinguish odd parity from exactly one hit")
    _require({o["active"] for i in training for o, n in zip(i["objects"], i["hits"]) if n % 2} == {True, False},
             "training must demonstrate both exchange directions")
    _require(any(any(o["scale"] > 1 and n % 2 for o, n in zip(i["objects"], i["hits"])) for i in training),
             "training must demonstrate enlarged reception")
    _require(any(_features(i)["relocate"] and any(n % 2 for n in i["hits"]) for i in training),
             "training must demonstrate attachment relocation")


def _digest(puzzle):
    payload = {s: [{k: np.asarray(p[k]).tolist() for k in ("input", "output")} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """42 streamed seeds, semantic counterexamples and bounded mutation checks."""
    # Hand-built cancellation/recombination witnesses. Expected state lists
    # are specified directly, not obtained from the generator's hit counter.
    hand_templates = [
        {(0, 0): 6, (0, 1): 2, (1, 0): 2, (1, 1): 7, (-1, 0): 7},
        {(0, 0): 6, (0, 1): 2, (1, 0): 2, (1, 1): 3, (0, -1): 3}]
    hand_anchors = [(3, 3), (3, 8), (8, 3), (8, 8)]
    for diagonal, expected_states, expected_hits in (
            (False, [0, 1, 1, 1], [0, 1, 1, 2]),
            (True, [0, 1, 1, 0], [0, 1, 1, 3])):
        inp, expected = np.zeros((20, 20), dtype=int), np.zeros((20, 20), dtype=int)
        for anchor, state, result in zip(hand_anchors, [0, 0, 0, 1], expected_states):
            _render(inp, hand_templates[state], anchor, 1)
            _render(expected, hand_templates[result], anchor, 1)
        hand_mask = {(0, k) for k in range(1, 6)} | {(k, 0) for k in range(1, 6)}
        if diagonal:
            hand_mask |= {(k, k) for k in range(1, 6)}
        for dr, dc in hand_mask:
            inp[13 + dr, 13 + dc] = NEUTRAL
        inp[13, 11], inp[13, 12] = 7, 6
        info = _read_input(inp)
        _require(info["hits"] == expected_hits, "manual pulse counts disagree")
        _require(np.array_equal(solve(inp), expected), "manual state exchange disagrees")
        _require(np.any((inp != 0) & (expected == 0)) and np.any((inp == 0) & (expected != 0)),
                 "hand witness must erase and create endpoint support")
    hashes, masks, vocabularies, shapes, schedules, counts, backgrounds = (set() for _ in range(7))
    directions, hit_counts, changes = set(), set(), set()
    metrics = {"seeds": 42, "pairs": 0, "odd_three_pairs": 0, "mixed_scale_pairs": 0,
               "center_shortcut_rejected": 0, "scale_shortcut_rejected": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                feature = _features(info)
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only solve failed")
                masks.add(feature["mask"])
                vocabularies.add(feature["vocabulary"])
                shapes.add(feature["shape"])
                schedules.add(feature["scales"])
                counts.add(feature["count"])
                backgrounds.add(feature["background"])
                directions.add(feature["direction"])
                hit_counts.update(feature["hits"])
                changes.add((feature["recolor"], feature["relocate"]))
                metrics["pairs"] += 1
                metrics["odd_three_pairs"] += any(n >= 3 and n % 2 for n in info["hits"])
                metrics["mixed_scale_pairs"] += len(set(feature["scales"])) > 1
                # Omitting A by indiscriminately swapping every motif must fail.
                _require(not np.array_equal(_answer(info, mode="all"), pair["output"]), "all-swap shortcut works")
                r0, r1, c0, c1 = _bounds(info["offsets"])
                center = (r0 + r1) // 2, (c0 + c1) // 2
                shifted = {(r - center[0], c - center[1]) for r, c in info["offsets"]}
                metrics["center_shortcut_rejected"] += not np.array_equal(
                    _answer(info, _count_hits(info["objects"], shifted)), pair["output"])
                metrics["scale_shortcut_rejected"] += not np.array_equal(
                    _answer(info, _count_hits(info["objects"], info["offsets"], scaled=False)), pair["output"])
        second = _read_input(puzzle["train"][1]["input"])
        _require(not np.array_equal(_answer(second, mode="any"), puzzle["train"][1]["output"]),
                 "any-hit shortcut survived cancellation demonstration")
        _require(not np.array_equal(_answer(second, [int(n == 1) for n in second["hits"]]),
                                   puzzle["train"][1]["output"]), "exactly-one shortcut survived three-hit demonstration")
        first = _read_input(puzzle["train"][0]["input"])
        # Every original emitter is needed: letting the other state emit too
        # would give the originally unhit fourth site extra requests.
        all_emitters = [dict(o, active=True) for o in first["objects"]]
        _require(not np.array_equal(_answer(first, _count_hits(all_emitters, first["offsets"])),
                                   puzzle["train"][0]["output"]), "all-states-emit shortcut works")
        original, result = second["grid"], puzzle["train"][1]["output"]
        _require(np.any((original == second["background"]) & (result != second["background"])),
                 "relocation demonstration creates no new endpoint")
    _require(len(hashes) == 42 and len(masks) >= 60 and len(vocabularies) >= 80, "insufficient structural variation")
    _require(len(shapes) >= 40 and len(schedules) >= 10 and len(counts) >= 4 and len(backgrounds) >= 8,
             "insufficient geometry, count, scale or palette variation")
    _require(directions == set(STEPS) and {0, 1, 2, 3} <= hit_counts and len(changes) == 3,
             "missing pulse or state-change regimes")
    _require(metrics["odd_three_pairs"] >= 8 and metrics["scale_shortcut_rejected"] >= 42,
             "missing recombination or scale witnesses")
    # Exercise every legal explicit count combination, keeping only one puzzle at a time.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(9000 + nt * 10 + ns, nt, ns)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == ns, "count mismatch")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(51, 2, 1)
    # Break the exact receptor square in one enlarged occurrence. Segmentation
    # and block normalization must reject the malformed input before any label.
    bad_input = puzzle["train"][1]["input"].copy()
    info = _read_input(bad_input)
    large = next(o for o in info["objects"] if o["scale"] > 1)
    bad_input[large["anchor"]] = info["background"]
    try:
        solve(bad_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged receptor square accepted")
    puzzle["test"][0]["output"][0, 0] = (int(puzzle["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    metrics.update({"distinct_puzzles": len(hashes), "registered_masks": len(masks),
                    "state_vocabularies": len(vocabularies), "grid_shapes": len(shapes),
                    "scale_schedules": len(schedules), "motif_counts": sorted(counts),
                    "background_colors": sorted(backgrounds), "hit_counts": sorted(hit_counts)})
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
