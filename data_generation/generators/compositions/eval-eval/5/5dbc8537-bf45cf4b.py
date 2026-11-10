"""Reconstruct an unseen unit cell by inventory-constrained periodic packing.

Only NumPy and the standard library are used. The public solver reads no
generation state. Source rectangles can bridge the inferred coarse-cell seams.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


SEARCH_BUDGET = 40000


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(binary, diagonal=False):
    left = set(map(tuple, np.argwhere(binary)))
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((int(y), int(x)))
            for dy, dx in offsets:
                p = (y + dy, x + dx)
                if p in left:
                    left.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    yy, xx = zip(*cells)
    return min(yy), min(xx), max(yy) + 1, max(xx) + 1


def _read_input(grid):
    g = _grid(grid)
    bg = int(g[0, 0])
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    if not np.all(border == bg):
        raise ValueError("The outside border must identify the background")
    comps = _components(g != bg)
    candidates = []
    for cells in comps:
        y0, x0, y1, x1 = _bbox(cells)
        h, w = y1 - y0, x1 - x0
        if h < 5 or w < 5 or len(cells) != 2 * h + 2 * w - 4:
            continue
        frame = int(g[y0, x0])
        crop = g[y0:y1, x0:x1]
        if (frame == bg or any(int(g[y, x]) != frame for y, x in cells)
                or not all(np.all(side == frame) for side in
                           (crop[0], crop[-1], crop[:, 0], crop[:, -1]))):
            continue
        inner = crop[1:-1, 1:-1]
        colors = set(map(int, np.unique(inner))) - {bg}
        if len(colors) != 1 or frame in colors:
            continue
        signal = next(iter(colors))
        if not all(np.all(side == bg) for side in
                   (inner[0], inner[-1], inner[:, 0], inner[:, -1])):
            continue
        sy, sx, ey, ex = _bbox(list(map(tuple, np.argwhere(inner == signal))))
        mask = (inner[sy:ey, sx:ex] == signal).copy()
        if int(mask.sum()) < 2:
            continue
        outside = g != bg
        outside[y0:y1, x0:x1] = False
        tiles, locations = [], []
        valid = True
        for tc in _components(outside):
            ty, tx, te, tr = _bbox(tc)
            tile = g[ty:te, tx:tr].copy()
            if tile.size != len(tc) or np.any(np.isin(tile, (bg, signal, frame))):
                valid = False
                break
            tiles.append(tile)
            locations.append((ty, tx))
        if (valid and len(tiles) >= 2
                and len(set(int(c) for t in tiles for c in t.flat)) >= 2):
            candidates.append(dict(background=bg, frame_color=frame,
                                   signal=signal, frame_box=(y0, x0, y1, x1),
                                   mask=mask, tiles=tiles, locations=locations))
    if len(candidates) != 1:
        raise ValueError("Expected one framed mask and a rectangular inventory")
    return candidates[0]


def _render(mask, motif, bg):
    h, w = motif.shape
    yy, xx = np.indices((mask.shape[0] * h, mask.shape[1] * w))
    return np.where(mask[yy // h, xx // w], motif[yy % h, xx % w], bg).astype(np.int8)


def _pack(mask, unit, tiles, budget=SEARCH_BUDGET):
    """Enumerate up to two motifs for a scale; exhaustively prove a singleton.

    Placements carry both a geometric bitset and unit-coordinate color claims.
    Equal patterns are grouped with physical multiplicity, never made reusable.
    """
    sh, sw = unit
    silhouette = np.repeat(np.repeat(mask, sh, axis=0), sw, axis=1)
    oh, ow = silhouette.shape
    active = [int(i) for i in np.flatnonzero(silhouette)]
    full = sum(1 << i for i in active)
    groups, lookup = [], {}
    for t in tiles:
        key = _key(t)
        if key in lookup:
            groups[lookup[key]]["count"] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(pattern=t, count=1))
    placements, by_cell = [], {i: [] for i in active}
    geometry_cache = {}
    for gi, group in enumerate(groups):
        t = group["pattern"]
        th, tw = t.shape
        # Even before positioning, one rectangle must be self-consistent with
        # a proposed period. The coordinate shift does not change these ties.
        residues = {}
        for y in range(th):
            for x in range(tw):
                r = (y % sh, x % sw)
                c = int(t[y, x])
                if r in residues and residues[r] != c:
                    return {}, groups
                residues[r] = c
        if t.shape not in geometry_cache:
            options = []
            for y in range(oh - th + 1):
                for x in range(ow - tw + 1):
                    if np.all(silhouette[y:y + th, x:x + tw]):
                        ids = tuple((y + dy) * ow + x + dx
                                    for dy in range(th) for dx in range(tw))
                        options.append((y, x, sum(1 << i for i in ids), ids))
            geometry_cache[t.shape] = options
        if not geometry_cache[t.shape]:
            return {}, groups
        for y, x, bits, ids in geometry_cache[t.shape]:
            claims = tuple((((dy + y) % sh) * sw + (dx + x) % sw, c)
                           for (dy, dx), c in sorted(residues.items()))
            pi = len(placements)
            placements.append((gi, y, x, bits, claims))
            for i in ids:
                by_cell[i].append(pi)
    remaining = [q["count"] for q in groups]
    motif = [-1] * (sh * sw)
    outcomes, path, dead = {}, [], set()
    nodes = 0

    def compatible(claims):
        return all(motif[i] == -1 or motif[i] == c for i, c in claims)

    def visit(covered):
        nonlocal nodes
        # Once every motif coordinate is known, all continuations render that
        # same image. A previously witnessed image needs no further tiling-count
        # enumeration; physical-copy feasibility was proved by its first path.
        if -1 not in motif and tuple(motif) in outcomes:
            return True
        nodes += 1
        if nodes > budget:
            raise ValueError("Periodic exact-cover search exceeded its node budget")
        if covered == full:
            if any(remaining) or -1 in motif:
                return False
            outcomes.setdefault(tuple(motif), dict(path=tuple(path), nodes=nodes))
            return True
        state = (covered, tuple(remaining), tuple(motif))
        if state in dead:
            return False
        best = None
        for i in active:
            if covered & (1 << i):
                continue
            options = [pi for pi in by_cell[i]
                       if remaining[placements[pi][0]]
                       and not covered & placements[pi][3]
                       and compatible(placements[pi][4])]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for pi in best:
            gi, y, x, bits, claims = placements[pi]
            writes = [i for i, c in claims if motif[i] == -1]
            for i, c in claims:
                motif[i] = c
            remaining[gi] -= 1
            path.append((gi, y, x))
            found = visit(covered | bits) or found
            path.pop()
            remaining[gi] += 1
            for i in writes:
                motif[i] = -1
            if len(outcomes) >= 2:
                return True
        if not found:
            dead.add(state)
        return found

    visit(0)
    for rec in outcomes.values():
        rec["nodes"] = nodes
    return outcomes, groups


def _analyze(input_grid, budget=SEARCH_BUDGET):
    info = _read_input(input_grid)
    mask, tiles = info["mask"], info["tiles"]
    total = sum(t.size for t in tiles)
    k = int(mask.sum())
    if total % k:
        raise ValueError("Inventory area must be an integer number of unit cells")
    area = total // k
    answers = {}
    for h in range(1, 30 // mask.shape[0] + 1):
        if area % h:
            continue
        w = area // h
        if not 1 <= w <= 30 // mask.shape[1]:
            continue
        motifs, groups = _pack(mask, (h, w), tiles, budget=budget)
        for values, rec in motifs.items():
            motif = np.array(values, dtype=np.int8).reshape(h, w)
            output = _render(mask, motif, info["background"])
            answers[_key(output)] = dict(info, motif=motif, unit=(h, w),
                                          output=output, groups=groups,
                                          path=rec["path"], nodes=rec["nodes"])
            if len(answers) > 1:
                raise ValueError("Input admits more than one rendered answer")
    if len(answers) != 1:
        raise ValueError("Inventory has no periodic exact-cover completion")
    return next(iter(answers.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the entire output from the visible input; no training or seed needed."""
    return _analyze(input_grid)["output"].copy()


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in _transforms(a)[1:])


def _blocks_agree(out, mask, unit):
    h, w = unit
    blocks = [out[y*h:(y+1)*h, x*w:(x+1)*w] for y, x in np.argwhere(mask)]
    return all(np.array_equal(blocks[0], b) for b in blocks[1:])


def _swap_witness(info):
    """An explicit valid geometric cover rejected only by periodic agreement."""
    path, groups = info["path"], info["groups"]
    for i, (gi, y, x) in enumerate(path):
        a = groups[gi]["pattern"]
        for gj, yy, xx in path[i+1:]:
            b = groups[gj]["pattern"]
            if a.shape != b.shape or np.array_equal(a, b):
                continue
            h, w = a.shape
            swapped = info["output"].copy()
            swapped[y:y+h, x:x+w] = b
            swapped[yy:yy+h, xx:xx+w] = a
            if not _blocks_agree(swapped, info["mask"], info["unit"]):
                return True
    return False


def _crossing_count(info):
    sh, sw = info["unit"]
    return sum(y // sh != (y + info["groups"][gi]["pattern"].shape[0] - 1) // sh
               or x // sw != (x + info["groups"][gi]["pattern"].shape[1] - 1) // sw
               for gi, y, x in info["path"])


def _sample_mask(rng, shape, witness=False, islands=False):
    h, w = shape
    yy, xx = np.indices(shape)
    for attempt in range(250):
        family = str(rng.choice(("random", "random", "diagonal", "ring", "path", "dense", "full", "islands")))
        if family == "ring":
            m = (yy == 0) | (xx == 0) | (yy == h-1) | (xx == w-1)
        elif family == "path":
            m = np.zeros(shape, bool)
            y = x = 0
            m[y, x] = True
            while y < h-1 or x < w-1:
                if x < w-1 and (y == h-1 or rng.random() < .5):
                    x += 1
                else:
                    y += 1
                m[y, x] = True
        elif family == "diagonal":
            m = abs(yy * (w-1) - xx * (h-1)) <= int(rng.integers(1, max(h, w)))
            m |= rng.random(shape) < .15
        elif family == "full":
            m = np.ones(shape, bool)
        elif family == "dense":
            m = rng.random(shape) < .8
        elif family == "islands":
            m = (yy % 2 == 0) & (xx % 2 == 0)
            m |= rng.random(shape) < .18
        else:
            m = rng.random(shape) < float(rng.uniform(.25, .75))
        m = np.flip(m, axis=int(rng.integers(2))).copy() if rng.random() < .5 else m
        if not all(np.any(s) for s in (m[0], m[-1], m[:, 0], m[:, -1])):
            continue
        k = int(m.sum())
        adjacent = np.any(m[:-1] & m[1:]) or np.any(m[:, :-1] & m[:, 1:])
        if not (3 <= k <= 10 and adjacent):
            continue
        if witness and (np.all(m) or not _asymmetric(m)):
            continue
        if islands and len(_components(m, True)) < 2:
            continue
        return m
    # Constructive island-capable asymmetric fallback for the witness envelope.
    m = np.zeros(shape, bool)
    m[0, 0] = m[1, 0] = m[0, -1] = True
    if h > 2:
        m[-1, -1] = True
        m[0, -1] = False
    if islands and shape == (3, 2):
        m[:] = False
        m[0, :] = True
        m[-1, 0] = True
    if witness and not _asymmetric(m):
        raise RuntimeError("Failed to construct an asymmetric coarse mask")
    if islands and len(_components(m, True)) < 2:
        raise RuntimeError("Failed to construct separated coarse islands")
    return m


def _motif(rng, shape, palette, witness):
    yy, xx = np.indices(shape)
    for attempt in range(80):
        family = "random" if witness else str(rng.choice(("random", "bands", "checker", "mark", "diagonal")))
        if family == "bands":
            labels = yy if rng.random() < .5 else xx
        elif family == "checker":
            labels = yy + xx
        elif family == "diagonal":
            labels = yy * int(rng.integers(1, len(palette)+1)) + xx
        elif family == "mark":
            labels = np.zeros(shape, int)
            labels[int(rng.integers(shape[0])), int(rng.integers(shape[1]))] = 1
        else:
            labels = rng.integers(len(palette), size=shape)
        p = np.asarray(palette, np.int8)[labels % len(palette)]
        if len(np.unique(p)) >= 2 and (not witness or _asymmetric(p)):
            return p
    p = np.full(shape, palette[0], np.int8)
    p[0, 0] = palette[1]
    p[-1, 0] = palette[-1]
    return p


def _split_rect(rect, axis, cut):
    y, x, h, w = rect
    if axis == 0:
        return [(y, x, cut, w), (y + cut, x, h - cut, w)]
    return [(y, x, h, cut), (y, x + cut, h, w - cut)]


def _partition(rng, mask, unit, witness, partial_witness=False):
    sh, sw = unit
    occupied = np.zeros(mask.shape, bool)
    bridges = []
    options = []
    for h in range(1, min(3, mask.shape[0]) + 1):
        for w in range(1, min(3, mask.shape[1]) + 1):
            if not 2 <= h*w <= (2 if witness else 4) or h*w >= mask.sum():
                continue
            if partial_witness and not (sw % 2 == 0 if w > 1 else sh % 2 == 0):
                continue
            for y in range(mask.shape[0] - h + 1):
                for x in range(mask.shape[1] - w + 1):
                    if np.all(mask[y:y+h, x:x+w]):
                        options.append((y, x, h, w))
    if not options:
        return None
    first = options[int(rng.integers(len(options)))]
    chosen = [first]
    if not witness and rng.random() < .3:
        y, x, h, w = first
        occupied[y:y+h, x:x+w] = True
        more = [q for q in options if not np.any(occupied[q[0]:q[0]+q[2], q[1]:q[1]+q[3]])]
        if more:
            chosen.append(more[int(rng.integers(len(more)))])
    for y, x, h, w in chosen:
        occupied[y:y+h, x:x+w] = True
        rect = (y*sh, x*sw, h*sh, w*sw)
        if partial_witness or (not witness and rng.random() < .45):
            axis = 0 if w > 1 and h == 1 else 1 if h > 1 and w == 1 else int(rng.integers(2))
            extent = rect[2+axis]
            bridges.extend(_split_rect(rect, axis, int(rng.integers(1, extent))))
        else:
            bridges.append(rect)
    rects = list(bridges)
    rest = list(map(tuple, np.argwhere(mask & ~occupied)))
    rng.shuffle(rest)
    if not rest and witness:
        return None
    for i, (y, x) in enumerate(rest):
        rect = (int(y)*sh, int(x)*sw, sh, sw)
        if witness and i == 0:
            axes = [1 if chosen[0][3] > 1 else 0] if partial_witness else [a for a, n in enumerate(unit) if n % 2 == 0]
            axis = axes[int(rng.integers(len(axes)))]
            cut = unit[axis] // 2
        else:
            axis = int(rng.integers(2))
            cut = int(rng.integers(1, unit[axis]))
        rects.extend(_split_rect(rect, axis, cut))
    if len(rects) < 3 or len(rects) > 17:
        return None
    # Occasionally subdivide one fragment again, retaining solid rectangles.
    if not witness and len(rects) < 17 and rng.random() < .25:
        choices = [(i, a) for i, q in enumerate(rects) for a in (0, 1) if q[2+a] > 1]
        if choices:
            i, a = choices[int(rng.integers(len(choices)))]
            rects[i:i+1] = _split_rect(rects[i], a, int(rng.integers(1, rects[i][2+a])))
    return rects


def _scatter(rng, mask, tiles, bg, frame, signal):
    pad = int(rng.integers(1, 3))
    window = np.full((mask.shape[0]+2*pad+2, mask.shape[1]+2*pad+2), bg, np.int8)
    window[0] = window[-1] = frame
    window[:, 0] = window[:, -1] = frame
    window[pad+1:pad+1+mask.shape[0], pad+1:pad+1+mask.shape[1]] = np.where(mask, signal, bg)
    objects = [window] + tiles
    min_h = max(t.shape[0] for t in objects) + 2
    min_w = max(t.shape[1] for t in objects) + 2
    for attempt in range(32):
        h = int(rng.integers(max(17, min_h), 31)) if attempt < 24 else 30
        w = int(rng.integers(max(17, min_w), 31)) if attempt < 24 else 30
        occupied = np.zeros((h, w), bool)
        positions = {}
        jitter = rng.random(len(objects))
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(jitter[i])))
        for i in order:
            th, tw = objects[i].shape
            # All one-cell moats at once. argwhere retains the previous row-major
            # candidate order, so this acceleration consumes the same RNG draws.
            integral = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            rh, rw = th + 2, tw + 2
            totals = (integral[rh:, rw:] - integral[:-rh, rw:]
                      - integral[rh:, :-rw] + integral[:-rh, :-rw])
            legal = np.argwhere(totals == 0) + 1
            if not len(legal):
                break
            y, x = map(int, legal[int(rng.integers(len(legal)))])
            positions[i] = (y, x)
            occupied[y:y+th, x:x+tw] = True
        if len(positions) == len(objects):
            g = np.full((h, w), bg, np.int8)
            for i, (y, x) in positions.items():
                th, tw = objects[i].shape
                g[y:y+th, x:x+tw] = objects[i]
            return g
    return None


def _checked_integer(value, name, lo, hi=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < lo or (hi is not None and int(value) > hi)):
        raise ValueError(f"Illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_integer(seed, "seed", 0)
    nt = _checked_integer(num_train, "num_train", 2, 10)
    ne = _checked_integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .2)
    base_palette = list(map(int, rng.permutation(10)))
    long_axis = int(rng.integers(2))
    shapes = [(h, w) for h in range(2, 6) for w in range(2, 6) if h*w <= 20]
    puzzle, seen = {"train": [], "test": []}, set()
    witness_mask_shapes, witness_units = [], []
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = split == "train" and index < 2
            for attempt in range(180):
                if split == "test":
                    dims = [int(rng.integers(2, 4)), int(rng.integers(2, 4))]
                    dims[long_axis] = int(rng.integers(4, 6))
                    mshape = tuple(dims)
                elif witness:
                    choices = [q for q in ((2, 3), (3, 2), (3, 3)) if q not in witness_mask_shapes]
                    mshape = choices[int(rng.integers(len(choices)))]
                else:
                    mshape = (int(rng.integers(2, 4)), int(rng.integers(2, 4)))
                mask = _sample_mask(rng, mshape, witness, witness and index == 1)
                choices = [q for q in shapes if not witness or (
                    q not in witness_units and any(n % 2 == 0 for n in q)
                    and q != (2, 2) and (index != 0 or q[0] != q[1]))]
                unit = choices[int(rng.integers(len(choices)))]
                palette = base_palette if stable else list(map(int, rng.permutation(10)))
                bg, frame, signal = palette[:3]
                payload = palette[3:3+int(rng.integers(2, 6))]
                motif = _motif(rng, unit, payload, witness)
                if witness and not _asymmetric(motif):
                    continue
                rects = _partition(rng, mask, unit, witness,
                                   partial_witness=witness and index == 1)
                if rects is None:
                    continue
                out = _render(mask, motif, bg)
                tiles = [out[y:y+h, x:x+w].copy() for y, x, h, w in rects]
                if any(np.array_equal(t, motif) for t in tiles):
                    continue
                g = _scatter(rng, mask, tiles, bg, frame, signal)
                if g is None or _key(g) in seen:
                    continue
                try:
                    # Reject expensive construction candidates conservatively.
                    # The public solver retains the larger family-wide budget.
                    info = _analyze(g, budget=2000)
                except ValueError:
                    continue
                if (not np.array_equal(info["output"], out) or not _crossing_count(info)
                        or (witness and not _swap_witness(info))):
                    continue
                puzzle[split].append({"input": g, "output": out})
                seen.add(_key(g))
                if witness:
                    witness_mask_shapes.append(mshape)
                    witness_units.append(unit)
                break
            else:
                raise RuntimeError("Could not construct an unambiguous periodic inventory")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(g) not in seen
            seen.add(_key(g))
            a = _analyze(g)
            assert np.array_equal(out, a["output"]), "Wrong joint reconstruction"
            mask, motif = a["mask"], a["motif"]
            h, w = a["unit"]
            assert sum(t.size for t in a["tiles"]) == int(mask.sum()) * h * w
            assert _blocks_agree(out, mask, (h, w))
            inactive = ~np.repeat(np.repeat(mask, h, axis=0), w, axis=1)
            assert np.all(out[inactive] == a["background"])
            assert not np.any(np.isin(motif, (a["background"], a["frame_color"], a["signal"])))
            assert not any(np.array_equal(t, motif) for t in a["tiles"]), "A complete stamp must not be supplied"
            assert _crossing_count(a) > 0, "Demonstrate seams that source rectangles cross"
            used = Counter(gi for gi, y, x in a["path"])
            assert all(used[i] == q["count"] for i, q in enumerate(a["groups"]))
            # Repaint using physical placements, rather than modular substitution.
            painted = np.full(out.shape, a["background"], np.int8)
            occupancy = np.zeros(out.shape, int)
            for gi, y, x in a["path"]:
                t = a["groups"][gi]["pattern"]
                th, tw = t.shape
                painted[y:y+th, x:x+tw] = t
                occupancy[y:y+th, x:x+tw] += 1
            assert np.array_equal(painted, out)
            assert np.all(occupancy[~inactive] == 1) and np.all(occupancy[inactive] == 0)
            infos[split].append(a)
    tr, te = infos["train"], infos["test"]
    witnesses = [a for a in tr if _swap_witness(a) and _asymmetric(a["mask"])
                 and _asymmetric(a["motif"]) and np.any(~a["mask"])]
    assert len(witnesses) >= 2, "Need two demonstrations distinguishing both contributions"
    assert len({a["unit"] for a in witnesses}) >= 2
    assert any(a["unit"][0] != a["unit"][1] for a in witnesses)
    assert len({a["mask"].shape for a in witnesses}) >= 2
    assert any(all(t.shape[0] < a["unit"][0] or t.shape[1] < a["unit"][1]
                   for t in a["tiles"]) for a in witnesses), "Need a motif that cannot be cropped from any source"
    assert any(len(_components(a["mask"], True)) > 1 for a in tr)
    assert any(all(a["mask"].shape[axis] > max(b["mask"].shape[axis] for b in tr)
                   for a in te) for axis in (0, 1)), "Missing coarse-extent shift"


def _puzzle_signature(p):
    return tuple((_key(q["input"]), _key(q["output"])) for s in ("train", "test") for q in p[s])


def self_test():
    """Bounded semantic checks and observable variation over 42 fixed seeds."""
    fingerprints, masks, units, layouts, patterns, outputs, roles, counts = (set() for _ in range(8))
    crossing, islands, duplicates = [], [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(_puzzle_signature(p))
        for s in ("train", "test"):
            for pair in p[s]:
                a = _analyze(pair["input"])
                masks.add((a["mask"].shape, a["mask"].tobytes()))
                units.add(a["unit"])
                layouts.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                mapping = {}
                patterns.add((a["motif"].shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in a["motif"].flat)))
                roles.add((a["background"], a["frame_color"], a["signal"]))
                counts.add(len(a["tiles"]))
                crossing.append(_crossing_count(a))
                islands.append(len(_components(a["mask"], True)))
                duplicates.append(any(q["count"] > 1 for q in a["groups"]))
    assert len(fingerprints) == 42
    assert len(masks) >= 60 and len(patterns) >= 80
    assert len(units) >= 12 and len(layouts) >= 30 and len(outputs) >= 40
    assert len(roles) >= 30 and len(counts) >= 8
    assert max(crossing) >= 2 and max(islands) >= 2 and any(duplicates)
    assert _puzzle_signature(generate_puzzle(63)) == _puzzle_signature(generate_puzzle(63))
    # The host verifier exercises all 30 legal combinations. Keep this local
    # self-test bounded while independently covering the count boundaries.
    for nt, ne in ((2, 1), (3, 2), (10, 1), (10, 4)):
        p = generate_puzzle(400+10*nt+ne, nt, ne)
        assert (len(p["train"]), len(p["test"])) == (nt, ne)
        validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # A missing physical copy cannot fill both coarse blocks, even when identical.
    m = np.ones((1, 2), bool)
    t = np.array([[2, 3]], np.int8)
    assert not _pack(m, (1, 2), [t])[0]
    assert len(_pack(m, (1, 2), [t, t])[0]) == 1
    # Geometry permits the pieces, but repetition rejects conflicting full blocks.
    assert not _pack(m, (1, 2), [t, np.array([[4, 5]], np.int8)])[0]
    # Two compatible phase choices must be recognized as different motifs.
    motifs, _ = _pack(m, (1, 2), [np.array([[2]], np.int8), np.array([[2]], np.int8),
                                np.array([[3]], np.int8), np.array([[3]], np.int8)])
    assert len(motifs) == 2
    # Each individual scale has one image, but the inventory can support two
    # different scales. The public input-only interpretation must reject it.
    strips = [t.copy() for _ in range(4)]
    assert len(_pack(m, (1, 4), strips)[0]) == 1
    assert len(_pack(m, (2, 2), strips)[0]) == 1
    ambiguous = _scatter(np.random.default_rng(808), m, strips, 0, 9, 8)
    assert ambiguous is not None
    try:
        solve(ambiguous)
    except ValueError as error:
        assert "more than one" in str(error)
    else:
        raise AssertionError("Scale ambiguity was silently resolved")
    p = generate_puzzle(91, 2, 1)
    damaged = {s: [{k: np.array(v, copy=True) for k, v in q.items()} for q in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "mask_patterns": len(masks),
            "unit_shapes": len(units), "input_shapes": len(layouts), "output_shapes": len(outputs),
            "normalized_motifs": len(patterns), "role_assignments": len(roles),
            "piece_counts": sorted(counts), "max_crossing_pieces": max(crossing),
            "duplicate_bearing_pairs": sum(duplicates)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
