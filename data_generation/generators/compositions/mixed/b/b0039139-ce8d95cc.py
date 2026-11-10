"""Counted repetition of a tightly cropped, recolored, run-normalized stencil.

Only NumPy and the standard library are used. No files, examples, or seed state
are consulted by solve. Generation and validation retain no execution traces.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= x <= 30 for x in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _digest(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.int8).tobytes()).hexdigest()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _compress(tile):
    # Equality is of FULL vectors, separately on each axis of the same tile.
    rows = np.r_[True, np.any(tile[1:] != tile[:-1], axis=1)]
    cols = np.r_[True, np.any(tile[:, 1:] != tile[:, :-1], axis=0)]
    return tile[np.ix_(rows, cols)].copy()


def _runs(tile, axis):
    vectors = tile if axis == 0 else tile.T
    result = [1]
    for previous, current in zip(vectors, vectors[1:]):
        if np.array_equal(previous, current):
            result[-1] += 1
        else:
            result.append(1)
    return tuple(result)


def _decode(grid):
    """Find exactly one positional four-panel interpretation, input only.

    For vertical strips transpose into cross-axis by strip-axis coordinates;
    this is a coordinate change, never a rotation of the tile in the answer.
    """
    candidates = []
    for vertical in (False, True):
        g = grid.T if vertical else grid
        for divider in map(int, np.unique(g)):
            cuts = np.flatnonzero(np.all(g == divider, axis=0))
            if len(cuts) != 3 or np.count_nonzero(g == divider) != 3 * g.shape[0]:
                continue
            a, b, c = map(int, cuts)
            if min(a, b-a-1, c-b-1, g.shape[1]-c-1) < 1:
                continue
            stencil, counter, primary, secondary = (
                g[:, :a], g[:, a+1:b], g[:, b+1:c], g[:, c+1:])
            p, s = int(primary[0, 0]), int(secondary[0, 0])
            if p == s or not np.all(primary == p) or not np.all(secondary == s):
                continue
            sc, cc = set(map(int, np.unique(stencil))), set(map(int, np.unique(counter)))
            common = sc & cc
            if len(sc) != 2 or len(cc) != 2 or len(common) != 1:
                continue
            bg = next(iter(common))
            ink, counter_ink = next(iter(sc - {bg})), next(iter(cc - {bg}))
            if len({bg, ink, counter_ink, divider}) != 4:
                continue
            mask, cm = stencil == ink, counter == counter_ink
            points = np.argwhere(mask)
            lo, hi = points.min(axis=0), points.max(axis=0) + 1
            candidates.append({
                "vertical": vertical, "palette": (bg, divider, ink, counter_ink),
                "primary": p, "secondary": s, "raw": _crop(mask),
                "counter": cm, "count": len(_components(cm)),
                "margins": (int(lo[0]), int(mask.shape[0]-hi[0]),
                            int(lo[1]), int(mask.shape[1]-hi[1])),
                "lengths": (a, b-a-1, c-b-1, g.shape[1]-c-1),
            })
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous four-panel parse")
    return candidates[0]


def _repeat(tile, count, secondary):
    separator = np.full((tile.shape[0], 1), secondary, dtype=np.int8)
    parts = []
    for i in range(count):
        if i:
            parts.append(separator)
        parts.append(tile)
    return np.concatenate(parts, axis=1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(_grid(input_grid))
    colored = np.where(d["raw"], d["primary"], d["secondary"]).astype(np.int8)
    tile = _compress(colored)
    out = _repeat(tile, d["count"], d["secondary"])
    if max(out.shape) > 30:
        raise ValueError("the encoded output exceeds ARC bounds")
    return out.T.copy() if d["vertical"] else out


def _transform(mask, index):
    return np.rot90(mask[:, ::-1] if index >= 4 else mask, index % 4).copy()


def _motif(h, w, rng):
    """Connected stencils, without mandatory corner paths or line templates."""
    m = np.zeros((h, w), dtype=bool)
    if min(h, w) == 1:
        m[:] = True
    else:
        mode = int(rng.integers(5))
        if mode == 0:
            m[:] = True
            target = int(rng.integers(max(h, w), h*w))
            for flat in rng.permutation(h*w):
                if int(m.sum()) <= target:
                    break
                r, c = divmod(int(flat), w)
                m[r, c] = False
                if not m.any() or _crop(m).shape != (h, w) or len(_components(m)) != 1:
                    m[r, c] = True
        elif mode == 1:
            center = int(rng.integers(h)), int(rng.integers(w))
            ends = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                    (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
            for r, c in ends:
                m[r, c] = True
                while (r, c) != center:
                    moves = []
                    if r != center[0]:
                        moves.append((r + (1 if r < center[0] else -1), c))
                    if c != center[1]:
                        moves.append((r, c + (1 if c < center[1] else -1)))
                    r, c = moves[int(rng.integers(len(moves)))]
                    m[r, c] = True
        elif mode == 2:
            m[[0, -1], :] = True
            m[:, [0, -1]] = True
            if h > 2 and w > 2:
                for r in range(1, h-1):
                    if rng.random() < .5:
                        m[r, 1:int(rng.integers(1, w))] = True
        elif mode == 3:
            m[::2] = True
            for r in range(1, h, 2):
                m[r, 0 if (r//2) % 2 else -1] = True
        else:
            m[:] = True
    return _compress(_crop(m))


def _frame(rng):
    # Repeated separated border vectors on both axes teach RUN, not set,
    # compression. Interior cells vary; this need not be an empty ring.
    h, w = int(rng.integers(3, 6)), int(rng.integers(3, 5))
    m = np.ones((h, w), dtype=bool)
    m[1:-1, 1:-1] = rng.random((h-2, w-2)) < .45
    m[int(rng.integers(1, h-1)), int(rng.integers(1, w-1))] = False
    return _compress(m)


def _asymmetric(rng):
    for _ in range(12):
        m = _motif(int(rng.integers(3, 6)), int(rng.integers(3, 5)), rng)
        if all(not np.array_equal(m, _transform(m, k)) for k in range(1, 8)):
            return m
    return np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool)


def _tall_motif(h, w, rng):
    # An exact-height connected fallback: alternate distinct signatures.
    m = np.ones((h, w), dtype=bool)
    for r in range(1, h, 2):
        m[r] = False
        m[r, 0 if (r//2) % 2 else -1] = True
    if rng.random() < .5:
        m = m[:, ::-1].copy()
    return _compress(m)


def _multiplicities(size, limit, rng, kind="normal", strong=None):
    """Positive run lengths; anchor has both length one and a repeated run."""
    result = np.ones(size, dtype=int)
    if strong is not None:
        result[int(rng.integers(size))] = strong
    elif kind != "none":
        result = rng.integers(1, 5, size=size)
        if kind == "anchor":
            order = rng.permutation(size)
            result[order[0]] = 1
            result[order[1]] = int(rng.integers(2, 5))
    # Limits are allocated BEFORE expansion; no oversized temporary grid.
    while int(result.sum()) > limit:
        choices = np.flatnonzero(result > 1)
        result[int(rng.choice(choices))] -= 1
    return result


def _segments(length, pieces, rng, randomize):
    extras = length - (2*pieces - 1)
    if randomize:
        allocation = rng.multinomial(extras, np.ones(2*pieces+1)/(2*pieces+1))
        sizes = 1 + allocation[:pieces]
        gaps = np.r_[0, np.ones(pieces-1, dtype=int), 0] + allocation[pieces:]
    else:
        sizes = np.full(pieces, 1 + extras//pieces, dtype=int)
        sizes[rng.permutation(pieces)[:extras % pieces]] += 1
        gaps = np.r_[0, np.ones(pieces-1, dtype=int), 0]
    result, offset = [], int(gaps[0])
    for i, size in enumerate(sizes):
        result.append((offset, offset + int(size)))
        offset += int(size) + int(gaps[i+1])
    return result


def _grow(h, w, area, rng):
    m = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        m[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < h and 0 <= y < w and not m[x, y]:
                frontier.add((x, y))
    return m


def _counter(h, w, n, rng, area=None):
    layouts = []
    for nr in range(1, min(n, (h+1)//2)+1):
        for nc in range(1, min(n, (w+1)//2)+1):
            if not n <= nr*nc <= n+2:
                continue
            rows, cols = _segments(h, nr, rng, False), _segments(w, nc, rng, False)
            boxes = [(a, b, c, d) for a, b in rows for c, d in cols]
            capacities = sorted(((b-a)*(d-c) for a, b, c, d in boxes), reverse=True)
            if area is None or sum(capacities[:n]) >= area:
                layouts.append((nr, nc, boxes))
    _require(bool(layouts), "counter capacity allocation failed")
    nr, nc, boxes = layouts[int(rng.integers(len(layouts)))]
    rows, cols = _segments(h, nr, rng, True), _segments(w, nc, rng, True)
    alternative = [(a, b, c, d) for a, b in rows for c, d in cols]
    if area is None or sum(sorted(((b-a)*(d-c) for a, b, c, d in alternative), reverse=True)[:n]) >= area:
        boxes = alternative
    rng.shuffle(boxes)
    if area is not None:
        boxes.sort(key=lambda box: (box[1]-box[0])*(box[3]-box[2]), reverse=True)
    boxes = boxes[:n]
    capacities = [(b-a)*(d-c) for a, b, c, d in boxes]
    areas = [1]*n
    if area is not None:
        for _ in range(area-n):
            i = int(rng.choice([j for j in range(n) if areas[j] < capacities[j]]))
            areas[i] += 1
    all_single = area is None and rng.random() < .18
    mask = np.zeros((h, w), dtype=bool)
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        bh, bw = r1-r0, c1-c0
        if area is not None or all_single:
            obj = _grow(bh, bw, areas[i], rng)
        else:
            oh, ow = int(rng.integers(1, bh+1)), int(rng.integers(1, bw+1))
            mode = int(rng.integers(5))
            obj = np.ones((oh, ow), dtype=bool)
            if mode == 0:
                obj = _grow(oh, ow, int(rng.integers(1, oh*ow+1)), rng)
            elif mode == 1:
                obj[1:] = False  # bar, with orientation depending on box geometry
            elif mode == 2:
                obj[:, 1:] = False
            elif mode == 3 and min(oh, ow) > 2:
                obj[1:-1, 1:-1] = False
            r0 += int(rng.integers(bh-oh+1))
            c0 += int(rng.integers(bw-ow+1))
        mask[r0:r0+obj.shape[0], c0:c0+obj.shape[1]] |= obj
    if mask.all():
        # Only possible for one solid rectangular object. Removing a corner
        # exposes background while preserving its connectivity and count.
        mask[0, 0] = False
    _require(len(_components(mask)) == n == len(_components(mask, True)), "counter objects touch")
    return mask


def _make_pair(m, n, vertical, palette, swatches, rng, anchor=False,
               counter_area=None, strong_run=None):
    h, a = m.shape
    _require(n*(a+1)-1 <= 30, "normalized tile exceeds output budget")
    mode = int(rng.integers(5))
    row_kind = "anchor" if anchor else ("none" if mode == 0 else "normal")
    col_kind = "anchor" if anchor else ("none" if mode == 1 else "normal")
    rr = _multiplicities(h, 26, rng, row_kind, strong_run)
    cr = _multiplicities(a, 16 if anchor else 18, rng, col_kind)
    raw = np.repeat(np.repeat(m, rr, axis=0), cr, axis=1)
    _require(np.array_equal(_compress(raw), m), "motif must be run-canonical")
    # A visible blank margin in the frame witness distinguishes crop scope.
    min_h = max(raw.shape[0] + int(anchor), 7 if anchor else 3,
                2*((n+3)//4)-1)
    canvas_h = int(rng.integers(min_h, min(30, min_h+5)+1))
    max_stencil_pad = min(4, 30-3-raw.shape[1]-(7 if anchor else 3)-2)
    along_pad = int(rng.integers(int(anchor), max_stencil_pad+1))
    sw = raw.shape[1] + along_pad
    top = int(rng.integers(canvas_h-raw.shape[0]+1))
    left = int(rng.integers(along_pad+1))
    bg, divider, ink, ci = palette
    stencil = np.full((canvas_h, sw), bg, dtype=np.int8)
    stencil[top:top+raw.shape[0], left:left+raw.shape[1]][raw] = ink
    if np.all(stencil == ink):
        # Solid one-cell canonical motifs still need background for parsing.
        stencil = np.pad(stencil, ((0, 0), (0, 1)), constant_values=bg)
        sw += 1
    capacity_widths = [w for w in range(7 if anchor else 3, 30-3-sw-2+1)
                       if ((canvas_h+1)//2)*((w+1)//2) >= n]
    _require(bool(capacity_widths), "input capacity allocation failed")
    cw = int(rng.choice(capacity_widths))
    cm = _counter(canvas_h, cw, n, rng, counter_area)
    budget = 30-3-sw-cw
    swatch_sizes = [(x, y) for x in range(1, budget) for y in range(1, budget-x+1)
                    if not anchor or (x != n and y != n)]
    _require(bool(swatch_sizes), "swatch budget failed")
    pw, zw = swatch_sizes[int(rng.integers(len(swatch_sizes)))]
    p, s = swatches
    line = np.full((canvas_h, 1), divider, dtype=np.int8)
    inp = np.concatenate((stencil, line, np.where(cm, ci, bg).astype(np.int8), line,
                          np.full((canvas_h, pw), p, dtype=np.int8), line,
                          np.full((canvas_h, zw), s, dtype=np.int8)), axis=1)
    # Labels use the canonical motif, whereas solve contracts the observed crop.
    out = _repeat(np.where(m, p, s).astype(np.int8), n, s)
    return {"input": inp.T.copy() if vertical else inp,
            "output": out.T.copy() if vertical else out}


def _checked_int(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _checked_int(num_train, "num_train", 2, 10)
    nq = _checked_int(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = tuple(map(int, rng.choice(10, 4, replace=False)))
    shift = int(rng.integers(1, 10))
    palettes = [base, tuple((c+shift) % 10 for c in base)]
    palettes.extend(tuple(map(int, rng.choice(10, 4, replace=False))) for _ in range(2))
    counts = list(map(int, rng.permutation([2, 3])))
    same_area = int(rng.integers(12, 19))
    first_vertical = bool(rng.integers(2))
    mixed_orientations = nt >= 3 and rng.random() < .75
    puzzle, used, training = {"train": [], "test": []}, set(), []
    test_modes = list(map(int, rng.permutation(4)))
    for split, amount in (("train", nt), ("test", nq)):
        for index in range(amount):
            anchor = split == "train" and index < 2
            strong = None
            if anchor:
                n = counts[index]
                m = _frame(rng) if index == 0 else _asymmetric(rng)
                vertical = first_vertical
            elif split == "train":
                n = counts[0] if mixed_orientations and index == 2 else int(rng.integers(1, 5))
                m = _motif(int(rng.integers(2, 8)),
                           int(rng.integers(2, min(7, 31//n-1)+1)), rng)
                vertical = (not first_vertical if index == 2 else bool(rng.integers(2))) if mixed_orientations else first_vertical
            else:
                max_n = max(d["count"] for d in training)
                max_h = max(_compress(d["raw"]).shape[0] for d in training)
                mode = test_modes[index]
                n = (int(rng.integers(max_n+1, min(7 if mode == 3 else 10, max_n+5)+1))
                     if mode in (0, 3) else int(rng.choice([d["count"] for d in training])))
                if mode in (1, 3):
                    target_h = int(rng.integers(max_h+1, min(12, max_h+4)+1))
                    target_w = int(rng.integers(3, min(5, 31//n-1)+1))
                    m = _motif(target_h, target_w, rng)
                    if m.shape[0] <= max_h:
                        m = _tall_motif(target_h, target_w, rng)
                elif mode == 2:
                    reference = training[int(rng.integers(len(training)))]
                    m, n = _compress(reference["raw"]), reference["count"]
                    strong = max(max(_runs(d["raw"], ax)) for d in training for ax in (0, 1)) + int(rng.integers(1, 4))
                else:
                    m = _motif(int(rng.integers(2, 8)),
                               int(rng.integers(2, min(6, 31//n-1)+1)), rng)
                vertical = bool(rng.integers(2)) if mixed_orientations else first_vertical
            palette = palettes[index if anchor else int(rng.integers(len(palettes)))]
            if anchor and index == 0:
                swatches = tuple(map(int, rng.choice([c for c in range(10) if c not in palette], 2, replace=False)))
            elif anchor:
                swatches = palette[0], palette[3]
            else:
                swatches = tuple(map(int, rng.choice([c for c in range(10) if c != palette[1]], 2, replace=False)))
            pair = _make_pair(m, n, vertical, palette, swatches, rng, anchor,
                              same_area if anchor else None, strong)
            # At most fourteen inputs are retained. Ten cyclic relabelings of
            # an existing geometry suffice in normal use; color swaps extend
            # the fallback without retries or extra stored grids.
            if _key(pair["input"]) in used:
                original = pair
                found = False
                for offset in range(10):
                    for swap in range(10):
                        mapping = (np.arange(10)+offset) % 10
                        mapping[0], mapping[swap] = mapping[swap], mapping[0]
                        candidate = {k: mapping[v].astype(np.int8) for k, v in original.items()}
                        if _key(candidate["input"]) not in used:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                _require(found, "uniqueness fallback exhausted")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                training.append(_decode(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    d = _decode(inp)
    raw = d["raw"]
    _require(len(_components(raw)) == 1, "supported stencils are connected")
    _require(d["count"] == len(_components(d["counter"], True)), "ambiguous counter connectivity")
    # Explicit first-of-run indices independently check the vectorized solver.
    rows = [i for i in range(raw.shape[0]) if i == 0 or not np.array_equal(raw[i], raw[i-1])]
    cols = [j for j in range(raw.shape[1]) if j == 0 or not np.array_equal(raw[:, j], raw[:, j-1])]
    canonical = raw[np.ix_(rows, cols)]
    rr, cr = _runs(raw, 0), _runs(raw, 1)
    _require(np.array_equal(np.repeat(np.repeat(canonical, rr, axis=0), cr, axis=1), raw),
             "run compression is not lossless given its run lengths")
    h, a = canonical.shape
    n = d["count"]
    normalized_output = out.T if d["vertical"] else out
    _require(normalized_output.shape == (h, n*(a+1)-1), "incorrect output dimensions")
    for r in range(h):
        for c in range(normalized_output.shape[1]):
            j = c % (a+1)
            expected = d["primary"] if j < a and canonical[r, j] else d["secondary"]
            _require(int(normalized_output[r, c]) == expected, "incorrect retained value, copy or separator")
    d.update({"canonical": canonical, "row_runs": rr, "col_runs": cr,
              "area": int(d["counter"].sum()), "input_shape": inp.shape,
              "output_shape": out.shape})
    return d


def _separated_repeat(tile, axis):
    vectors = tile if axis == 0 else tile.T
    return any(np.array_equal(vectors[i], vectors[j])
               for i in range(len(vectors)) for j in range(i+2, len(vectors)))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            d = _analyze(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(d)
    train = analyses["train"]
    _require(len({d["count"] for d in train}) >= 2, "need multiple training counts")
    _require(any(a["vertical"] == b["vertical"] and a["count"] != b["count"]
                 for a in train for b in train), "count must vary within a familiar orientation")
    _require({d["vertical"] for d in analyses["test"]} <= {d["vertical"] for d in train},
             "test orientation must be demonstrated")
    _require(any(a["area"] == b["area"] and a["count"] != b["count"]
                 for a in train for b in train), "area/count confounding")
    _require(any(d["area"] > d["count"] for d in train), "pixel-count shortcut")
    _require(any(d["count"] > 1 and sum(d["margins"]) > 0 and not d["raw"].all()
                 for d in train), "need cropping, recoloring, repetition witness")
    for ax, label in ((0, "row_runs"), (1, "col_runs")):
        _require(any(min(d[label]) == 1 and max(d[label]) > 1 for d in train),
                 "unequal run lengths must distinguish uniform scaling")
        _require(any(_separated_repeat(d["canonical"], ax) for d in train),
                 "separated equal vectors must survive")
    for transform in range(1, 8):
        _require(any(not np.array_equal(d["canonical"], _transform(d["canonical"], transform))
                     for d in train), "stencil orientation is not established")
    for panel in (2, 3):
        _require(any(d["lengths"][panel] != d["count"] for d in train), "swatch-size shortcut")
    _require(len({d["palette"] for d in train}) >= 2, "need role-based palettes")
    max_n = max(d["count"] for d in train)
    max_h = max(d["canonical"].shape[0] for d in train)
    max_run = max(max(d[label]) for d in train for label in ("row_runs", "col_runs"))
    for d in analyses["test"]:
        _require(d["count"] > max_n or d["canonical"].shape[0] > max_h
                 or max(d["row_runs"] + d["col_runs"]) > max_run,
                 "test needs a measured count, shape or run-length shift")


def _signature(puzzle):
    return tuple(_digest(_grid(pair[k])) for split in ("train", "test")
                 for pair in puzzle[split] for k in ("input", "output"))


def self_test():
    """42 streamed seeds, bounded metrics, legal counts and corruption checks."""
    puzzles, shapes, output_shapes, motifs, layouts, palettes, counts = (set() for _ in range(7))
    axes, no_op, two_axis, pairs, maximum_run = set(), 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(hashlib.sha256("".join(_signature(p)).encode()).hexdigest())
        if seed in (0, 17, 41):
            _require(_signature(p) == _signature(generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in p[split]:
                d = _analyze(pair)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagreement")
                shapes.add(d["input_shape"])
                output_shapes.add(d["output_shape"])
                motifs.add(_digest(d["canonical"]))
                layouts.add((d["vertical"], d["canonical"].shape, d["row_runs"], d["col_runs"], d["lengths"]))
                palettes.add(d["palette"])
                counts.add(d["count"])
                axes.add(d["vertical"])
                r, c = max(d["row_runs"]) > 1, max(d["col_runs"]) > 1
                two_axis += r and c
                no_op += r != c
                maximum_run = max(maximum_run, max(d["row_runs"] + d["col_runs"]))
                pairs += 1
    _require(len(puzzles) == 42 and len(shapes) >= 60 and len(output_shapes) >= 30, "insufficient geometric variation")
    _require(len(motifs) >= 50 and len(layouts) >= 160 and len(palettes) >= 60, "insufficient structural variation")
    _require(axes == {False, True} and 1 in counts and max(counts) >= 7, "count or axis coverage")
    _require(two_axis >= 80 and no_op >= 10 and maximum_run >= 5, "compression regime coverage")
    legal_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(9137, nt, nq)
            legal_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    p["test"][0]["output"].flat[0] = (int(p["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupt label")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "canonical_motifs": len(motifs), "layouts": len(layouts),
            "structural_palettes": len(palettes), "counts": sorted(counts),
            "two_axis_compression": int(two_axis), "one_axis_compression": int(no_op),
            "maximum_run": maximum_run, "legal_count_calls": legal_calls}
