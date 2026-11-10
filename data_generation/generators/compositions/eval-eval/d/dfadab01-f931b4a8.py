"""Registered cutout weave: input-only, color-matched multiscale stamping.

No parent module, external file, saved example, or generation state is used.
The optional command-line entry point runs a bounded in-memory self-test.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, bg):
    """Eight-neighbor components, including different colors in one object."""
    remaining = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = (r + dr, c + dc)
                    if point in remaining:
                        remaining.remove(point)
                        stack.append(point)
        result.append(cells)
    return result


def _members(tile, codes, bg):
    live, rejected, bare, painted = [], [], [], []
    for cells in _components(tile, bg):
        points = [(r, c, int(tile[r, c])) for r, c in cells
                  if int(tile[r, c]) in codes]
        origin = min(r for r, _ in cells), min(c for _, c in cells)
        if len(points) == 1 and points[0][:2] == origin:
            live.append(points[0])
            if len(cells) > 1:
                painted.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    return sorted(live), sorted(rejected), bare, sorted(painted)


def _decode(input_grid):
    g = _grid(input_grid)
    if min(g.shape) < 10:
        raise ValueError("missing loom frame")
    bg, frame = int(g[0, 0]), int(g[1, 1])
    if bg == frame:
        raise ValueError("frame and background must differ")
    if not (np.all(g[[0, -1], :] == bg) and np.all(g[:, [0, -1]] == bg)
            and np.all(g[[1, -2], 1:-1] == frame)
            and np.all(g[1:-1, [1, -2]] == frame)):
        raise ValueError("incomplete outer frame")
    bars = [j for j in range(3, g.shape[1] - 3)
            if np.all(g[2:-2, j] == frame)]
    if not bars:
        raise ValueError("missing vertical divider")
    divider = bars[0]
    left = g[2:-2, 2:divider]
    cuts = np.flatnonzero(np.all(left == frame, axis=1))
    if len(cuts) != 2:
        raise ValueError("expected swatch, height counter, width counter")
    first, second = map(int, cuts)
    carrier = left[:first].copy()
    height_counter = left[first + 1:second]
    width_counter = left[second + 1:]
    tile = g[2:-2, divider + 1:-2].copy()
    h, w = carrier.shape
    if not (3 <= h <= 5 and 3 <= w <= 5 and height_counter.size
            and width_counter.size and tile.shape[1] >= w):
        raise ValueError("invalid panel dimensions")
    codes = set(map(int, np.unique(carrier)))
    if len(codes) not in (2, 3) or codes & {bg, frame}:
        raise ValueError("swatch must have two or three payload colors")
    for code in codes:
        mask = carrier == code
        if int(mask.sum()) < 2 or len(_components(mask.astype(np.int8), 0)) != 1:
            raise ValueError("each swatch silhouette must be connected and nontrivial")
    for counter in (height_counter, width_counter):
        if not set(map(int, np.unique(counter))) <= codes | {bg}:
            raise ValueError("counter contains a non-payload ink")
    if not set(map(int, np.unique(tile))) <= codes | {bg, frame}:
        raise ValueError("tile contains an unexplained ink")
    H = int(np.count_nonzero(height_counter != bg))
    W = int(np.count_nonzero(width_counter != bg))
    if not (h <= H <= 30 and w <= W <= 30 and H % h == 0 and W % w == 0):
        raise ValueError("counter dimensions must permit integral replication")
    return dict(grid=g, bg=bg, frame=frame, carrier=carrier, tile=tile,
                counters=(height_counter, width_counter), H=H, W=W,
                codes=codes, scale=(H // h, W // w))


def _layers(carrier, shape, records, *, wrap=False, rectangle=False,
            cropped=False, point=False):
    p, q = shape
    layers = {int(c): np.zeros(shape, dtype=bool) for c in np.unique(carrier)}
    for r, c, code in records:
        mask = carrier == code
        if cropped:
            yy, xx = np.nonzero(mask)
            mask = mask[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        if rectangle:
            mask = np.ones(carrier.shape, dtype=bool)
        if point:
            mask = np.ones((1, 1), dtype=bool)
        for dy, dx in np.argwhere(mask):
            y, x = r + int(dy), c + int(dx)
            if wrap:
                layers[code][y % p, x % q] = True
            elif y < p and x < q:
                layers[code][y, x] = True
    return layers


def _render(d, records, *, mode="correct"):
    """Vectorized construction and counterfactuals; no output-label inference."""
    carrier, tile = d["carrier"], d["tile"]
    h, w = carrier.shape
    H, W, bg = d["H"], d["W"], d["bg"]
    p, q = tile.shape
    rr, cc = np.arange(H)[:, None], np.arange(W)[None, :]
    sr, sc = rr * h // H, cc * w // W
    if mode == "tile_carrier_rows":
        sr = rr % h
    if mode == "tile_carrier_columns":
        sc = cc % w
    base = carrier[sr, sc].copy()
    layers = _layers(carrier, tile.shape, records, wrap=mode == "wrap_stamps",
                     rectangle=mode == "rectangle", cropped=mode == "cropped",
                     point=mode == "point")
    tr, tc = rr % p, cc % q
    if mode == "resize_stencil":
        tr, tc = rr * p // H, cc * q // W
    elif mode == "phase_rows":
        tr = (rr + 1) % p
    elif mode == "phase_columns":
        tc = (cc + 1) % q
    erase = np.zeros((H, W), dtype=bool)
    for code, layer in layers.items():
        hit = layer[tr, tc]
        if mode == "no_repeat":
            hit = hit & (rr < p) & (cc < q)
        erase |= hit if mode == "unconditional" else hit & (base == code)
    if mode == "no_cutout":
        erase[:] = False
    base[erase] = bg
    return base


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the frame, counters, swatch and objects from this input alone."""
    d = _decode(input_grid)
    carrier, tile = d["carrier"], d["tile"]
    live = _members(tile, d["codes"], d["bg"])[0]
    # Separate cell-based implementation of the rule. No generation records or
    # precomputed dictionaries participate. A stencil coordinate is first
    # reduced modulo the TILE; stamp offsets are then tested without wrapping.
    H, W = d["H"], d["W"]
    h, w = carrier.shape
    p, q = tile.shape
    out = np.empty((H, W), dtype=np.int8)
    by_code = {code: [(r, c) for r, c, v in live if v == code] for code in d["codes"]}
    for r in range(H):
        for c in range(W):
            code = int(carrier[r * h // H, c * w // W])
            y, x = r % p, c % q
            cut = any(0 <= y - a < h and 0 <= x - b < w
                      and carrier[y - a, x - b] == code for a, b in by_code[code])
            out[r, c] = d["bg"] if cut else code
    return out


def _carrier(rng, h, w, codes):
    """Connected color partitions, including runs, checks and grown regions."""
    k = len(codes)
    for attempt in range(48):
        kind = int(rng.integers(5))
        if kind in (0, 1):
            length = w if kind == 0 else h
            cuts = sorted(map(int, rng.choice(np.arange(1, length), k - 1, replace=False)))
            line = np.searchsorted(cuts, np.arange(length), side="right")
            labels = np.tile(line, (h, 1)) if kind == 0 else np.tile(line[:, None], (1, w))
        elif kind == 2 and k == 2:
            rr, cc = np.indices((h, w))
            labels = (rr + cc + int(rng.integers(2))) % 2
        elif kind == 3:
            # Unequal staircase bands: every band appears in every row, so
            # each silhouette remains eight-connected after the cuts wander.
            cuts = sorted(map(int, rng.choice(np.arange(1, w), k - 1, replace=False)))
            labels = np.empty((h, w), dtype=int)
            for r in range(h):
                labels[r] = np.searchsorted(cuts, np.arange(w), side="right")
                j = int(rng.integers(k - 1))
                proposal = cuts[j] + int(rng.choice([-1, 0, 1]))
                low = 1 if j == 0 else cuts[j - 1] + 1
                high = w - 1 if j == k - 2 else cuts[j + 1] - 1
                cuts[j] = max(low, min(high, proposal))
        else:
            labels = np.full((h, w), -1, dtype=int)
            seeds = rng.choice(h * w, k, replace=False)
            frontiers = [set() for _ in range(k)]
            sizes = np.ones(k, dtype=int)
            for i, loc in enumerate(seeds):
                labels[int(loc) // w, int(loc) % w] = i
            def extend(i, r, c):
                for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= y < h and 0 <= x < w and labels[y, x] < 0:
                        frontiers[i].add((y, x))
            for i, loc in enumerate(seeds):
                extend(i, int(loc) // w, int(loc) % w)
            while np.any(labels < 0):
                for i in range(k):
                    frontiers[i] = {v for v in frontiers[i] if labels[v] < 0}
                choices = [i for i in range(k) if frontiers[i]]
                weights = np.array([1 / sizes[i] for i in choices])
                i = int(rng.choice(choices, p=weights / weights.sum()))
                cells = sorted(frontiers[i])
                r, c = cells[int(rng.integers(len(cells)))]
                labels[r, c] = i
                sizes[i] += 1
                extend(i, r, c)
        if rng.random() < .5:
            labels = labels[::-1]
        if rng.random() < .5:
            labels = labels[:, ::-1]
        if all(np.count_nonzero(labels == i) >= 2
               and len(_components((labels == i).astype(np.int8), 0)) == 1
               for i in range(k)):
            return np.asarray(rng.permutation(codes), dtype=np.int8)[labels].copy()
    # Guaranteed connected fallback with no extra colors or hidden convention.
    labels = np.minimum(np.arange(w) * k // w, k - 1)
    return np.tile(np.asarray(codes, dtype=np.int8)[labels], (h, 1))


def _counter(rng, height, width, count, codes, bg):
    mask = np.zeros((height, width), dtype=bool)
    style = int(rng.integers(4))
    if style == 0:
        ids = rng.permutation(height * width)
    elif style == 1:
        ids = np.arange(height * width).reshape(height, width)
        ids = ids[rng.permutation(height)][:, rng.permutation(width)].ravel()
    elif style == 2:
        rr, cc = np.indices(mask.shape)
        r, c = int(rng.integers(height)), int(rng.integers(width))
        ids = np.argsort((abs(rr - r) + abs(cc - c) + rng.random(mask.shape) * .8).ravel())
    else:
        ids = np.arange(height * width).reshape(height, width).T.ravel()
        if rng.random() < .5:
            ids = ids[::-1]
    mask.ravel()[ids[:count]] = True
    inks = rng.choice(codes, int(rng.integers(1, len(codes) + 1)), replace=False)
    out = np.full(mask.shape, bg, dtype=np.int8)
    out[mask] = rng.choice(inks, count, p=rng.dirichlet(np.ones(len(inks))))
    return out


def _halo(mask):
    h, w = mask.shape
    a = np.pad(mask, 1)
    return np.logical_or.reduce([a[r:r + h, c:c + w] for r in range(3) for c in range(3)])


def _put(tile, obj, bg, *, replace=None):
    fg = obj != bg
    old = tile != bg
    if replace is not None:
        old = old.copy()
        old[replace] = False
    if np.any(_halo(fg) & old):
        return False
    tile[fg] = obj[fg]
    return True


def _material(rng, tile, carrier, codes, bg, frame, kind, weights=None):
    h, w = carrier.shape
    p, q = tile.shape
    locations = [(r, c, int(code)) for r in range(p - h + 1)
                 for c in range(q - w + 1) for code in codes]
    rng.shuffle(locations)
    if kind == "attached" and weights is not None:
        # Prefer an attached marker whose erroneous activation would make a
        # new matching cut. Its attachment must still pass the same visible
        # component/spacing rules; no singleton is secretly declared invalid.
        offsets = {int(code): list(map(tuple, np.argwhere(carrier == code))) for code in codes}
        def score(location):
            r, c, code = location
            dy, dx = max(offsets[code])
            y, x = r + int(dy) + 1, c + int(dx) + 1
            return sum(int(weights[code][y + int(a), x + int(b)])
                       for a, b in offsets[code] if y + a < p and x + b < q)
        locations.sort(key=score, reverse=True)
    for r, c, code in locations:
        mask = carrier == code
        obj = np.full(tile.shape, bg, dtype=np.int8)
        obj[r:r + h, c:c + w][mask] = frame
        if kind == "attached":
            dy, dx = max(map(tuple, np.argwhere(mask)))
            y, x = r + int(dy) + 1, c + int(dx) + 1
            if y >= p or x >= q:
                continue
            obj[y, x] = code
        if _put(tile, obj, bg):
            return True
    return False


def _make_pair(rng, palette, scale, *, attached=False, bare=False, paint=True):
    bg, frame, codes = palette
    # Each attempt holds one <=30x30 pair and a few boolean layers. Failed
    # attempts are discarded, never logged or retained as grids.
    for attempt in range(160):
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        carrier = _carrier(rng, h, w, codes)
        H, W = h * scale[0], w * scale[1]
        a = max(2, (H + w - 1) // w) + int(rng.integers(3))
        b = max(2, scale[1]) + int(rng.integers(3))
        p = h + a + b + 2
        if p > 26:
            a = max(2, (H + w - 1) // w)
            b = max(2, scale[1])
            p = h + a + b + 2
        # Extra counter space makes a useful tile for separated input objects.
        while p < 13:
            a += 1
            p += 1
        q = int(rng.integers(w + 4, min(20, 25 - w) + 1))
        tile = np.full((p, q), bg, dtype=np.int8)
        base = np.repeat(np.repeat(carrier, scale[0], axis=0), scale[1], axis=1)
        yy, xx = np.indices(base.shape)
        weights = {}
        for code in codes:
            z = np.zeros(tile.shape, dtype=int)
            np.add.at(z, (yy.ravel() % p, xx.ravel() % q), (base == code).ravel())
            weights[int(code)] = z
        records = []
        desired = list(map(int, rng.permutation(codes)))
        desired += list(map(int, rng.choice(codes, int(rng.integers(1, 6)))))
        failed = False
        for index, code in enumerate(desired):
            offsets = np.argwhere(carrier == code)
            blocked = _halo(tile != bg)
            options, clipped = [], []
            for r in range(p):
                for c in range(q):
                    if blocked[r, c]:
                        continue
                    score = sum(int(weights[code][r + int(dy), c + int(dx)])
                                for dy, dx in offsets if r + dy < p and c + dx < q)
                    if score:
                        options.append((r, c))
                        if r + h > p or c + w > q:
                            clipped.append((r, c))
            if not options:
                if index < len(codes):
                    failed = True
                break
            pool = clipped if clipped and rng.random() < .38 else options
            r, c = pool[int(rng.integers(len(pool)))]
            tile[r, c] = code
            records.append((r, c, code))
        if failed:
            continue
        if paint and rng.random() < .55:
            for r, c, code in rng.permutation(np.array(records, dtype=int)):
                r, c, code = int(r), int(c), int(code)
                mask = carrier == code
                if r + h > p or c + w > q or mask[0, 0] or not mask[:2, :2].any():
                    continue
                obj = np.full(tile.shape, bg, dtype=np.int8)
                obj[r:r + h, c:c + w][mask] = frame
                obj[r, c] = code
                if len(_components(obj, bg)) == 1 and _put(tile, obj, bg, replace=(r, c)):
                    break
        if attached:
            live_layers = _layers(carrier, tile.shape, records)
            free_weights = {int(code): weights[int(code)] * ~live_layers[int(code)] for code in codes}
            if not _material(rng, tile, carrier, codes, bg, frame, "attached", free_weights):
                continue
        if bare and not _material(rng, tile, carrier, codes, bg, frame, "bare"):
            continue
        actual = _members(tile, set(map(int, codes)), bg)[0]
        if actual != sorted(records):
            continue
        left = np.vstack((carrier, np.full((1, w), frame, dtype=np.int8),
                          _counter(rng, a, w, H, codes, bg),
                          np.full((1, w), frame, dtype=np.int8),
                          _counter(rng, b, w, W, codes, bg)))
        g = np.full((p + 4, w + q + 5), bg, dtype=np.int8)
        g[1:-1, 1:-1] = frame
        g[2:-2, 2:2 + w] = left
        g[2:-2, 3 + w:-2] = tile
        d = _decode(g)
        out = _render(d, records)
        # Every code both cuts and leaves material. Thus shape/color matching
        # is not a merely latent permission in a generated example.
        if not all(np.any((base == code) & (out == bg)) and np.any(out == code)
                   for code in codes):
            continue
        return {"input": g, "output": out}
    raise RuntimeError("bounded placement budget exhausted")


_REQUIRED = frozenset(("point", "rectangle", "cropped", "all_markers",
                       "tile_carrier_rows", "tile_carrier_columns", "resize_stencil",
                       "no_repeat", "unconditional", "no_cutout", "phase_rows",
                       "phase_columns", "wrap_stamps", "counter_swap",
                       "counter_bbox", "counter_one_ink", "counter_colors"))


def _evidence(pair):
    d = _decode(pair["input"])
    live, rejected, _, _ = _members(d["tile"], d["codes"], d["bg"])
    result = set()
    for name in _REQUIRED - {"all_markers", "counter_swap", "counter_bbox",
                             "counter_one_ink", "counter_colors"}:
        if not np.array_equal(pair["output"], _render(d, live, mode=name)):
            result.add(name)
    if not np.array_equal(pair["output"], _render(d, live + rejected)):
        result.add("all_markers")
    if d["H"] != d["W"]:
        result.add("counter_swap")
    bbox, one_ink, color_counts = [], [], []
    for counter in d["counters"]:
        r, c = np.nonzero(counter != d["bg"])
        bbox.append(int((r.max() - r.min() + 1) * (c.max() - c.min() + 1)))
        _, counts = np.unique(counter[counter != d["bg"]], return_counts=True)
        one_ink.append(int(counts.max()))
        color_counts.append(len(counts))
    for name, values in (("counter_bbox", bbox), ("counter_one_ink", one_ink),
                         ("counter_colors", color_counts)):
        if tuple(values) != (d["H"], d["W"]):
            result.add(name)
    return result


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    palette = colors[0], colors[1], tuple(colors[2:2 + int(rng.integers(2, 4))])
    # The held-out scale pair recombines axes already demonstrated separately.
    ra, rb = map(int, rng.choice(np.arange(2, 7), 2, replace=False))
    ca, cb = map(int, rng.choice(np.arange(2, 7), 2, replace=False))
    if rng.random() < .5:
        ca, cb = cb, ca
    first_scale, second_scale, target_scale = (ra, ca), (rb, cb), (ra, cb)
    candidates = [[], []]
    train = None
    # At most 48 candidate pairs are ever considered, at most 16 retained at
    # once. Selection measures actual arrays and does not require fixed images.
    for serial in range(48):
        slot = serial % 2
        pair = _make_pair(rng, palette, (first_scale, second_scale)[slot],
                          attached=True, bare=serial % 3 != 0)
        cover = _evidence(pair)
        for other, other_cover in candidates[1 - slot]:
            proposed = [pair, other] if slot == 0 else [other, pair]
            if _REQUIRED <= cover | other_cover:
                # Both bare and attached material must be visibly taught.
                if any(_members(_decode(p["input"])["tile"], set(palette[2]), palette[0])[2]
                       for p in proposed):
                    train = proposed
                    break
        if train is not None:
            break
        candidates[slot].append((pair, cover))
        candidates[slot] = candidates[slot][-8:]
    if train is None:
        best = max((a | b for _, a in candidates[0] for _, b in candidates[1]),
                   key=len, default=set())
        raise RuntimeError(f"teaching budget exhausted at seed {seed}; missing {sorted(_REQUIRED - best)}")
    puzzle = {"train": train, "test": []}
    seen = {_key(pair["input"]) for pair in train}
    taught_paint = any(_members(_decode(p["input"])["tile"], set(palette[2]), palette[0])[3]
                       for p in train)
    scale_choices = [(r, c) for r in range(2, 7) for c in range(2, 7)
                     if (r, c) != target_scale]
    for split, total in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < total:
            for retry in range(24):
                scale = target_scale if split == "test" else scale_choices[int(rng.integers(len(scale_choices)))]
                pair = _make_pair(rng, palette, scale, attached=rng.random() < .60,
                                  bare=rng.random() < .45,
                                  paint=split == "train" or taught_paint)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input budget exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                taught_paint |= bool(_members(_decode(pair["input"])["tile"],
                                              set(palette[2]), palette[0])[3])
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_material(d):
    """Generated composites are full silhouettes, optionally with one marker."""
    tile, carrier, bg, frame = d["tile"], d["carrier"], d["bg"], d["frame"]
    h, w = carrier.shape
    for cells in _components(tile, bg):
        actual = {p: int(tile[p]) for p in cells}
        if len(cells) == 1 and actual[cells[0]] in d["codes"]:
            continue
        points = [p for p in cells if actual[p] in d["codes"]]
        assert len(points) <= 1, "unexplained multi-code composite"
        matched = False
        codes = [actual[points[0]]] if points else sorted(d["codes"])
        for code in codes:
            mask = carrier == code
            yy, xx = np.nonzero(mask)
            origin = min(r for r, _ in cells), min(c for _, c in cells)
            starts = {(origin[0] - int(yy.min()), origin[1] - int(xx.min()))}
            if points:
                starts.add(points[0])
            for r, c in starts:
                if not (0 <= r <= tile.shape[0] - h and 0 <= c <= tile.shape[1] - w):
                    continue
                expected = {(r + int(y), c + int(x)): frame for y, x in zip(yy, xx)}
                if points:
                    point = points[0]
                    last = max(map(tuple, np.argwhere(mask)))
                    valid_paint = (point == (r, c) and not mask[0, 0]
                                   and mask[:2, :2].any())
                    valid_attached = point == (r + int(last[0]) + 1, c + int(last[1]) + 1)
                    if not (valid_paint or valid_attached):
                        continue
                    expected[point] = code
                if expected == actual:
                    matched = True
        assert matched, "material does not match any visible swatch silhouette"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, evidence = set(), set(), set()
    features = {"train": set(), "test": set()}
    scales = {"train": set(), "test": set()}
    row_scales, column_scales = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            d = _decode(inp)
            conventions.add((d["bg"], d["frame"], tuple(sorted(d["codes"]))))
            live, rejected, bare, painted = _members(d["tile"], d["codes"], d["bg"])
            assert {code for _, _, code in live} == d["codes"], "a swatch code has no live seed"
            assert np.array_equal(solve(inp.copy()), out), "incorrect input-derived output"
            assert np.array_equal(_render(d, live), out), "construction disagrees with cell solver"
            assert out.shape == (d["H"], d["W"])
            assert set(map(int, np.unique(out))) <= d["codes"] | {d["bg"]}
            base = np.repeat(np.repeat(d["carrier"], d["scale"][0], axis=0), d["scale"][1], axis=1)
            assert all(np.any(out == c) and np.any((base == c) & (out == d["bg"]))
                       for c in d["codes"]), "every material must be cut and remain visible"
            assert np.all((out == base) | (out == d["bg"])), "cutting must not recolor material"
            _check_material(d)
            for name, values in (("attached", rejected), ("bare", bare), ("painted", painted)):
                if values:
                    features[split].add(name)
            scales[split].add(d["scale"])
            if split == "train":
                evidence |= _evidence(pair)
                row_scales.add(d["scale"][0])
                column_scales.add(d["scale"][1])
    assert len(conventions) == 1, "palette roles change within a puzzle"
    assert _REQUIRED <= evidence, "unresolved training alternatives: " + str(sorted(_REQUIRED - evidence))
    assert features["test"] <= features["train"], "test uses an untaught object category"
    assert {"bare", "attached"} <= features["train"]
    assert scales["test"].isdisjoint(scales["train"]), "test scale combination was not held out"
    assert all(r in row_scales and c in column_scales for r, c in scales["test"]), "new test axis factor"


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streaming metrics, mutation checks; no disk artifacts."""
    stats = dict(seeds=42, pairs=0)
    sets = {name: set() for name in ("signatures", "input_shapes", "output_shapes", "swatches",
                                    "tile_shapes", "live_counts", "palettes", "scales", "test_outputs")}
    counters = {name: 0 for name in ("clipped", "overlap", "bare", "attached", "painted",
                                    "repeat_rows", "repeat_columns", "black_payload")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        sets["signatures"].add(_signature(puzzle))
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                live, rejected, bare, painted = _members(d["tile"], d["codes"], d["bg"])
                h, w = d["carrier"].shape
                p, q = d["tile"].shape
                sets["input_shapes"].add(pair["input"].shape)
                sets["output_shapes"].add(pair["output"].shape)
                sets["swatches"].add(_normalized(d["carrier"]))
                sets["tile_shapes"].add(d["tile"].shape)
                sets["live_counts"].add(len(live))
                sets["palettes"].add((d["bg"], d["frame"], tuple(sorted(d["codes"]))))
                sets["scales"].add(d["scale"])
                if split == "test":
                    sets["test_outputs"].add(_normalized(pair["output"]))
                counters["clipped"] += any(r + h > p or c + w > q for r, c, _ in live)
                layers = _layers(d["carrier"], d["tile"].shape, live)
                counters["overlap"] += bool(np.any(sum(layer.astype(int) for layer in layers.values()) > 1))
                for name, values in (("bare", bare), ("attached", rejected), ("painted", painted)):
                    counters[name] += bool(values)
                counters["repeat_rows"] += d["H"] > p
                counters["repeat_columns"] += d["W"] > q
                counters["black_payload"] += 0 in d["codes"]
                stats["pairs"] += 1
    for name, minimum in (("signatures", 42), ("input_shapes", 30), ("output_shapes", 30),
                           ("swatches", 60), ("tile_shapes", 30), ("live_counts", 4),
                           ("palettes", 35), ("scales", 15), ("test_outputs", 40)):
        assert len(sets[name]) >= minimum, (name, len(sets[name]))
    assert all(value > 0 for value in counters.values()), counters
    # Additional explicit counts are exercised by verify.py. Include the
    # minimal teaching boundary and largest request here without a large pool.
    for counts in ((2, 1), (10, 4)):
        p = generate_puzzle(8675309, *counts)
        assert tuple(map(len, (p["train"], p["test"]))) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    out = damaged["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("changed output was accepted")
    stats.update({name: len(values) for name, values in sets.items()})
    stats.update(counters)
    return stats


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
