"""Fusion-and-echo docks: routing, independent docking, union and conserved count.

No parent code, files, generation metadata, or random state is used by solve.
Only standard-library modules and NumPy are needed.
"""
from __future__ import annotations

import copy
import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((g >= 0) & (g <= 9))), "colors outside 0..9")
    return g.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = EIGHT if diagonal else FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        groups.append(tuple(sorted(cells)))
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _shape_signature(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _prototype(components):
    """Preserve shared lateral coordinates, normalize each bottom separately."""
    height = max(max(r for r, c in x) - min(r for r, c in x) + 1 for x in components)
    left = min(c for x in components for r, c in x)
    right = max(c for x in components for r, c in x)
    mask = np.zeros((height, right-left+1), dtype=bool)
    for cells in components:
        bottom = max(r for r, c in cells)
        for r, c in cells:
            mask[height-1+r-bottom, c-left] = True
    return mask


def _row_band(row, bg):
    nonblank = np.flatnonzero(row != bg)
    if len(nonblank) < 6:
        return None
    left, right = int(nonblank[0]), int(nonblank[-1])
    rail = int(row[left])
    if int(row[right]) != rail:
        return None
    for anchor, lth, rth in (("left", 2, 1), ("right", 1, 2)):
        lo, hi = left+lth, right-rth+1
        if hi-lo < 3:
            continue
        route = int(row[lo])
        if route in (bg, rail):
            continue
        if (np.all(row[left:lo] == rail) and np.all(row[lo:hi] == route)
                and np.all(row[hi:right+1] == rail)):
            return left, right, lo, hi, route, rail, anchor
    return None


def _canonical_candidates(g, bg):
    rows = [_row_band(row, bg) if np.any(row != bg) else "blank" for row in g]
    for first in range(2, g.shape[0]):
        if rows[first] in (None, "blank") or rows[first-1] != "blank":
            continue
        if any(x is None for x in rows[first:]):
            continue
        bands, cursor = [], first
        while cursor < g.shape[0]:
            if rows[cursor] == "blank":
                cursor += 1
                continue
            info, top = rows[cursor], cursor
            while cursor+1 < g.shape[0] and rows[cursor+1] == info:
                cursor += 1
            left, right, lo, hi, route, rail, anchor = info
            bands.append(dict(top=top, bottom=cursor, left=left, right=right,
                              lo=lo, hi=hi, route=route, rail=rail, anchor=anchor))
            cursor += 1
        if len({b["route"] for b in bands}) != len(bands):
            continue
        if any(b["top"] < a["bottom"]+2 for a, b in zip(bands, bands[1:])):
            continue
        stage = g[:first]
        if (not np.any(stage != bg) or np.any(stage[:, 0] != bg)
                or np.any(stage[:, -1] != bg)):
            continue
        routes = {b["route"] for b in bands}
        if not set(map(int, np.unique(stage))) <= routes | {bg}:
            continue
        groups, tiles, valid = {}, {}, True
        for b in bands:
            color = b["route"]
            parts = _components(stage == color)
            # Avoid introducing an undemonstrated diagonal-contact convention.
            if len(parts) != len(_components(stage == color, True)):
                valid = False
                break
            groups[color] = parts
            if not parts:
                continue
            tile = _prototype(parts)
            h, w = tile.shape
            if h > b["bottom"]-b["top"]+1 or len(parts)*(w+1)-1 > b["hi"]-b["lo"]:
                valid = False
                break
            tiles[color] = tile
        if valid:
            yield dict(canonical=g, bg=bg, first=first, bands=bands, groups=groups, tiles=tiles)


def _analyze(input_grid):
    g = _grid(input_grid)
    candidates = []
    backgrounds = sorted({int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])})
    for turn in range(4):
        canonical = np.rot90(g, turn)
        for bg in backgrounds:
            for a in _canonical_candidates(canonical, bg):
                a["turn"] = turn
                candidates.append(a)
    _require(len(candidates) == 1, "input needs exactly one staging/dock interpretation")
    return candidates[0]


def _render_analysis(a):
    out = a["canonical"].copy()
    out[:a["first"]] = a["bg"]
    for b in a["bands"]:
        parts = a["groups"][b["route"]]
        if not parts:
            continue
        tile = a["tiles"][b["route"]]
        h, w = tile.shape
        length = len(parts)*(w+1)-1
        start = b["lo"] if b["anchor"] == "left" else b["hi"]-length
        top = b["bottom"]-h+1
        for r in range(h):
            for c in range(length):
                x = c % (w+1)
                if x < w and tile[r, x]:
                    out[top+r, start+c] = b["rail"]
    return np.rot90(out, -a["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all quantities, masks, colors and endpoints from input pixels."""
    return _render_analysis(_analyze(input_grid))


def _grow(h, w, area, rng, start=None):
    cells = {start if start is not None else (int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in FOUR
                           if 0 <= r+dr < h and 0 <= c+dc < w and (r+dr, c+dc) not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    mask = np.zeros((h, w), dtype=bool)
    for p in cells:
        mask[p] = True
    return mask


def _mask(h, w, rng):
    """Connected masks with tight extents: solids, frames, bends and pruning."""
    mode = int(rng.integers(5))
    m = np.ones((h, w), dtype=bool)
    if mode == 1 and min(h, w) >= 3:
        m[1:-1, 1:-1] = False
        if h >= 5 and w >= 5 and rng.random() < .5:
            m[:, int(rng.integers(2, w-2))] = True
    elif mode == 2:
        m[:] = False
        spine = int(rng.integers(w))
        m[:, spine] = True
        for row, end in ((int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)):
            m[row, min(spine, end):max(spine, end)+1] = True
    elif mode == 3:
        m[:] = False
        m[::2] = True
        side = int(rng.integers(2))
        for r in range(1, h, 2):
            m[r, -1 if side else 0] = True
            side = 1-side
    elif mode == 4 and min(h, w) > 1:
        goal = int(rng.integers(max(h, w), h*w+1))
        for flat in rng.permutation(h*w):
            if int(m.sum()) <= goal:
                break
            r, c = divmod(int(flat), w)
            m[r, c] = False
            if not m.any() or _crop(m).shape != (h, w) or len(_components(m)) != 1:
                m[r, c] = True
    return m


def _roles(rng, n):
    colors = list(map(int, rng.permutation(10)))
    bg, routes = colors[0], colors[1:n+1]
    external = colors[n+1:]
    rails = []
    common = int(rng.choice(external))
    for route in routes:
        choices = [c for c in colors[1:] if c != route]
        rails.append(common if rng.random() < .35 else int(rng.choice(choices)))
    return bg, routes, rails


def _layout(rng, stage, depths, min_widths, anchors=None):
    n = len(depths)
    bg, routes, rails = _roles(rng, n)
    width = int(rng.integers(max(17, max(min_widths)+3), 31))
    gaps = [int(rng.integers(1, 3)) for _ in depths]
    margin = int(rng.integers(0, 3))
    while stage+sum(depths)+sum(gaps)+margin > 30:
        if margin:
            margin -= 1
        elif max(gaps) > 1:
            gaps[gaps.index(max(gaps))] -= 1
        else:
            return None
    grid = np.full((stage+sum(depths)+sum(gaps)+margin, width), bg, dtype=np.int8)
    bands, cursor = [None]*n, stage
    # Logical source roles do not determine a dock's position in the array.
    for raw_i in rng.permutation(n):
        i = int(raw_i)
        depth = depths[i]
        cursor += gaps[i]
        max_inset = min(3, width-3-min_widths[i])
        total_inset = int(rng.integers(max_inset+1))
        left = int(rng.integers(total_inset+1))
        right = width-1-(total_inset-left)
        anchor = anchors[i] if anchors else ("left" if rng.integers(2) else "right")
        lth, rth = (2, 1) if anchor == "left" else (1, 2)
        b = dict(top=cursor, bottom=cursor+depth-1, left=left, right=right,
                 lo=left+lth, hi=right-rth+1, route=routes[i], rail=rails[i], anchor=anchor)
        bands[i] = b
        grid[cursor:cursor+depth, left:right+1] = rails[i]
        grid[cursor:cursor+depth, b["lo"]:b["hi"]] = routes[i]
        cursor += depth
    return grid, bands, bg


def _put(grid, bg, color, mask, top, left, stage):
    cells = tuple((top+int(r), left+int(c)) for r, c in np.argwhere(mask))
    if not all(0 <= r < stage and 1 <= c < grid.shape[1]-1 and grid[r, c] == bg for r, c in cells):
        return None
    if any(0 <= r+dr < stage and 0 <= c+dc < grid.shape[1] and grid[r+dr, c+dc] == color
           for r, c in cells for dr, dc in EIGHT):
        return None
    for p in cells:
        grid[p] = color
    return cells


def _finish(grid, bands, bg, placed, rotation):
    """Construction labels: concatenate known unions; solve uses modular pixels."""
    out = grid.copy()
    for parts in placed.values():
        for cells in parts:
            for p in cells:
                out[p] = bg
    for b in bands:
        parts = placed.get(b["route"], [])
        if not parts:
            continue
        # Build the union directly from known source coordinates.
        bottom = max(max(r for r, c in x)-min(r for r, c in x)+1 for x in parts)-1
        left = min(c for x in parts for r, c in x)
        right = max(c for x in parts for r, c in x)
        union = np.zeros((bottom+1, right-left+1), dtype=bool)
        for cells in parts:
            dy = bottom-max(r for r, c in cells)
            for r, c in cells:
                union[r+dy, c-left] = True
        tile = np.where(union, b["rail"], b["route"]).astype(np.int8)
        separator = np.full((tile.shape[0], 1), b["route"], dtype=np.int8)
        train = np.concatenate([part for i in range(len(parts))
                                for part in ((tile,) if i == len(parts)-1 else (tile, separator))], axis=1)
        start = b["lo"] if b["anchor"] == "left" else b["hi"]-train.shape[1]
        _require(start >= b["lo"] and start+train.shape[1] <= b["hi"], "echo train does not fit")
        top = b["bottom"]-train.shape[0]+1
        _require(top >= b["top"], "prototype is too tall")
        out[top:b["bottom"]+1, start:start+train.shape[1]] = train
    return {"input": np.rot90(grid, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _witness_one(rng, rotation):
    h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
    bar_width = int(rng.integers(1, w+1))
    frame = np.ones((h, w), dtype=bool)
    frame[1:-1, 1:-1] = False
    inner = _grow(h-2, w-2, int(rng.integers(1, (h-2)*(w-2)+1)), rng)
    stage = int(rng.integers(h+4, h+8))
    layout = _layout(rng, stage, [h+int(rng.integers(0, 3)), h-2+int(rng.integers(0, 2)),
                                 int(rng.integers(1, 4))], [2*w+1, w-2, 3], ["right", "left", "right"])
    if layout is None:
        return None
    g, bands, bg = layout
    x = int(rng.integers(1, g.shape[1]-w))
    y = int(rng.integers(0, stage-h-2))
    a, b = bands[:2]
    outer = _put(g, bg, a["route"], frame, y, x, stage)
    fill = _put(g, bg, b["route"], inner, y+1, x+1, stage)
    # The second A piece overlaps the frame's bottom after docking.
    bar = _put(g, bg, a["route"], np.ones((1, bar_width), bool),
               int(rng.integers(y+h+1, stage)), x+int(rng.integers(w-bar_width+1)), stage)
    if any(c is None for c in (outer, fill, bar)):
        return None
    pair = _finish(g, bands, bg, {a["route"]: [outer, bar], b["route"]: [fill]}, rotation)
    return pair, int(frame.sum())+bar_width


def _witness_two(rng, rotation, area):
    w = int(rng.integers(4, 6))
    sizes = np.ones(3, dtype=int)
    for _ in range(area-3):
        eligible = np.flatnonzero(sizes < 3*w)
        sizes[int(rng.choice(eligible))] += 1
    masks = [_grow(3, w, int(size), rng, (2, w//2)) for size in sizes]
    # Keep global column offsets, but remove unused source rows individually.
    masks = [m[np.flatnonzero(m.any(axis=1))[0]:] for m in masks]
    union = np.zeros((3, w), dtype=bool)
    for m in masks:
        union[3-m.shape[0]:] |= m
    union = _crop(union)
    transforms = [np.rot90(union[:, ::-1] if k >= 4 else union, k % 4) for k in range(1, 8)]
    if union.all() or any(np.array_equal(union, t) for t in transforms):
        return None
    stage = sum(m.shape[0] for m in masks)+int(rng.integers(3, 6))
    layout = _layout(rng, stage, [3+int(rng.integers(0, 2)), 3+int(rng.integers(0, 2))],
                     [3*(w+1)-1, 5], ["left", "right"])
    if layout is None:
        return None
    g, bands, bg = layout
    x = int(rng.integers(1, g.shape[1]-w))
    heights = sum(m.shape[0] for m in masks)+2
    y = int(rng.integers(stage-heights+1))
    a, b = bands
    pieces = []
    for m in masks:
        cells = _put(g, bg, a["route"], m, y, x, stage)
        if cells is None:
            return None
        pieces.append(cells)
        y += m.shape[0]+1
    # An exact copy in a second color defeats shape/area-based destination rules.
    other = _crop(masks[int(rng.integers(len(masks)))])
    options = list(range(1, g.shape[1]-other.shape[1]))
    rng.shuffle(options)
    placed_other = None
    for col in options:
        for row in rng.permutation(stage-other.shape[0]+1):
            placed_other = _put(g, bg, b["route"], other, int(row), col, stage)
            if placed_other:
                break
        if placed_other:
            break
    if not placed_other:
        return None
    return _finish(g, bands, bg, {a["route"]: pieces, b["route"]: [placed_other]}, rotation)


def _general_pair(rng, rotation, shift=None):
    nband = int(rng.choice([1, 2, 2, 3, 3, 4]))
    if shift and shift[0] == "size":
        nband = min(nband, 3)
    counts = [int(rng.integers(0, 5)) for _ in range(nband)]
    target = int(rng.integers(nband))
    counts[target] = int(shift[1]) if shift and shift[0] == "count" else max(1, counts[target])
    depths = [int(rng.integers(1, 6)) for _ in counts]
    if shift and shift[0] == "size":
        depths[target] = int(shift[1])
        counts[target] = int(rng.integers(1, 3))
    stage = int(rng.integers(9, 16))
    if stage+sum(depths)+nband > 30:
        return None
    # A narrow tile permits more echoes; geometry and count are budget-coupled.
    min_widths = [max(3, 2*k-1) for k in counts]
    layout = _layout(rng, stage, depths, min_widths)
    if layout is None:
        return None
    g, bands, bg = layout
    placed = {}
    order = list(map(int, rng.permutation(nband)))
    for i in order:
        k, b = counts[i], bands[i]
        if not k:
            continue
        max_w = (b["hi"]-b["lo"]+1)//k-1
        window = int(rng.integers(1, min(9, max_w)+1))
        left = int(rng.integers(1, g.shape[1]-window))
        masks = []
        for j in range(k):
            max_h = min(depths[i], max(1, (stage-k+1)//k+1))
            h = int(rng.integers(1, max_h+1))
            if shift and shift[0] == "size" and i == target and j == 0:
                h = depths[i]
            mw = int(rng.integers(1, window+1))
            masks.append(_mask(h, mw, rng))
        masks.sort(key=lambda m: m.size, reverse=True)
        pieces = []
        for mask in masks:
            cells = None
            positions = [(r, c) for r in range(stage-mask.shape[0]+1)
                         for c in range(left, left+window-mask.shape[1]+1)]
            for pos in rng.permutation(len(positions)):
                row, col = positions[int(pos)]
                cells = _put(g, bg, b["route"], mask, row, col, stage)
                if cells is not None:
                    break
            if cells is None:
                return None
            pieces.append(cells)
        placed[b["route"]] = pieces
    return _finish(g, bands, bg, placed, rotation)


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = int(rng.integers(4))
    mixed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    witness_area = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            rotation = (base+int(rng.integers(4))) % 4 if mixed else base
            # Test directions have been demonstrated: the shift is count/size.
            if split == "test":
                rotation = int(rng.choice([(-_analyze(p["input"])["turn"]) % 4 for p in puzzle["train"]]))
            shift = None
            if split == "test":
                train = [_analyze(p["input"]) for p in puzzle["train"]]
                max_k = max(len(parts) for a in train for parts in a["groups"].values())
                max_h = max(t.shape[0] for a in train for t in a["tiles"].values())
                shift = (("count", max_k+int(rng.integers(1, 3))) if rng.integers(2)
                         else ("size", max_h+1))
            pair = None
            for _ in range(200):
                area = None
                if split == "train" and index == 0:
                    result = _witness_one(rng, rotation)
                    if result is None:
                        continue
                    candidate, area = result
                elif split == "train" and index == 1:
                    candidate = _witness_two(rng, rotation, witness_area)
                else:
                    candidate = _general_pair(rng, rotation, shift)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                try:
                    a = _analyze(candidate["input"])
                except AssertionError:
                    continue
                if split == "test":
                    if shift[0] == "count" and max(map(len, a["groups"].values())) <= max_k:
                        continue
                    if shift[0] == "size" and max(t.shape[0] for t in a["tiles"].values()) <= max_h:
                        continue
                pair = candidate
                if area is not None:
                    witness_area = area
                break
            _require(pair is not None, "bounded scene construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    g, bg, first = a["canonical"], a["bg"], a["first"]
    stage = g[:first]
    touch = any(stage[r, c] != bg and stage[r+dr, c+dc] != bg
                and stage[r, c] != stage[r+dr, c+dc]
                for r in range(first) for c in range(g.shape[1]) for dr, dc in ((1, 0), (0, 1))
                if r+dr < first and c+dc < g.shape[1])
    independent, overlap, merged = False, False, False
    records = []
    for b in a["bands"]:
        parts = a["groups"][b["route"]]
        if not parts:
            records.append((0, 0, None))
            continue
        tile = a["tiles"][b["route"]]
        independent |= len({max(r for r, c in p) for p in parts}) > 1
        overlap |= sum(map(len, parts)) > int(tile.sum())
        merged |= len(_components(tile)) < len(parts)
        records.append((len(parts), sum(map(len, parts)), tile.shape))
    return dict(touch=touch, independent=independent, overlap=overlap, merged=merged, records=records,
                unused=any(not p for p in a["groups"].values()))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, analyses, features = set(), {}, {}
    for split in ("train", "test"):
        analyses[split], features[split] = [], []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must be preserved")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(out, _render_analysis(a)), "incorrect fusion/echo output")
            canonical_out = np.rot90(out, a["turn"])
            _require(np.all(canonical_out[:a["first"]] == a["bg"]), "source not cleared")
            for b in a["bands"]:
                rows = slice(b["top"], b["bottom"]+1)
                _require(np.array_equal(canonical_out[rows, b["left"]:b["lo"]],
                                        a["canonical"][rows, b["left"]:b["lo"]]), "near rail changed")
                _require(np.array_equal(canonical_out[rows, b["hi"]:b["right"]+1],
                                        a["canonical"][rows, b["hi"]:b["right"]+1]), "far rail changed")
                if not a["groups"][b["route"]]:
                    _require(np.array_equal(canonical_out[rows], a["canonical"][rows]), "unused dock changed")
            analyses[split].append(a)
            features[split].append(_features(a))
    train = features["train"]
    for feature in ("touch", "independent", "overlap", "merged", "unused"):
        _require(any(f[feature] for f in train), "missing training witness: "+feature)
    records = [r for f in train for r in f["records"] if r[0]]
    _require(any(x[1] == y[1] and x[0] != y[0] for x in records for y in records),
             "training confounds area and component count")
    _require(any(any(_shape_signature(x) == _shape_signature(y)
                     for x in a["groups"][c] for y in a["groups"][d])
                 for a in analyses["train"] for c in a["groups"] for d in a["groups"] if c != d),
             "training lacks the same-shape/different-route witness")
    _require(any(not t.all() and len(a["groups"][c]) > 1 for a in analyses["train"]
                 for c, t in a["tiles"].items()), "missing blank-cell and separator witness")
    for k in range(1, 8):
        _require(any(not np.array_equal(t, np.rot90(t[:, ::-1] if k >= 4 else t, k % 4))
                     for a in analyses["train"] for t in a["tiles"].values()),
                 "training does not establish stencil orientation")
    # Explicitly ablate B's count and the bridge's count-before-fusion rule.
    for count_mode in ("one", "fused"):
        changed = False
        for a in analyses["train"]:
            alternative = dict(a)
            alternative["groups"] = {
                c: [()] * (1 if count_mode == "one" else len(_components(a["tiles"][c])))
                if parts else [] for c, parts in a["groups"].items()}
            changed |= not np.array_equal(_render_analysis(a), _render_analysis(alternative))
        _require(changed, "training fails to distinguish count shortcut: "+count_mode)
    _require({b["anchor"] for a in analyses["train"] for b in a["bands"]
              if a["groups"][b["route"]]} == {"left", "right"}, "packing endpoint not demonstrated")
    max_k = max(r[0] for r in records)
    max_h = max(t.shape[0] for a in analyses["train"] for t in a["tiles"].values())
    turns = {a["turn"] for a in analyses["train"]}
    for a in analyses["test"]:
        _require(a["turn"] in turns, "undemonstrated global orientation")
        _require(max(map(len, a["groups"].values())) > max_k
                 or max(t.shape[0] for t in a["tiles"].values()) > max_h, "missing count/size test shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded deterministic suite with structural metrics and exact regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed puzzle duplication")
    shapes, masks, counts, dock_counts, endpoints, turns, palettes, depths = (set() for _ in range(8))
    overlaps = contacts = unused = disconnected = 0
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                f = _features(a)
                shapes.add(pair["input"].shape)
                dock_counts.add(len(a["bands"]))
                turns.add(a["turn"])
                palettes.add((a["bg"], tuple((b["route"], b["rail"]) for b in a["bands"])))
                overlaps += f["overlap"]
                contacts += f["touch"]
                unused += f["unused"]
                for b in a["bands"]:
                    endpoints.add(b["anchor"])
                    depths.add(b["bottom"]-b["top"]+1)
                    counts.add(len(a["groups"][b["route"]]))
                for t in a["tiles"].values():
                    masks.add(_key(t))
                    disconnected += len(_components(t)) > 1
    _require(len(shapes) >= 25 and len(masks) >= 60 and len(palettes) >= 40, "structural variation collapsed")
    _require(dock_counts >= {1, 2, 3, 4} and counts >= {0, 1, 2, 3, 4, 5}, "count variation collapsed")
    _require(turns == set(range(4)) and endpoints == {"left", "right"}, "orientation variation collapsed")
    _require(max(depths) >= 6 and min(depths) == 1, "depth variation collapsed")
    _require(overlaps >= 42 and contacts >= 42 and unused >= 42 and disconnected > 0, "fusion variation collapsed")
    for seed in (0, 17, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(700+nt*10+nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact collision regression: L plus overlapping bar makes an L, twice.
    g = np.zeros((9, 16), dtype=np.int8)
    g[7:9, 0] = g[7:9, 14:16] = 2
    g[7:9, 1:14] = 1
    g[0, 4] = g[1, 4] = g[1, 5] = g[4, 4] = g[4, 5] = 1
    expected = g.copy()
    expected[:7] = 0
    expected[7, 9] = expected[8, 9] = expected[8, 10] = 2
    expected[7, 12] = expected[8, 12] = expected[8, 13] = 2
    _require(np.array_equal(solve(g), expected), "collision multiplicity or separator regression")
    for rotation in range(4):
        _require(np.array_equal(solve(np.rot90(g, rotation)), np.rot90(expected, rotation)), "rotation regression")
    # Changing depth alone preserves docking; changing lateral relations does not.
    deeper = g.copy()
    deeper[4, 4:6] = 0
    deeper[5, 4:6] = 1
    _require(np.array_equal(solve(deeper), expected), "independent-depth invariance")
    sideways = g.copy()
    sideways[4, 4:6] = 0
    sideways[4, 6:8] = 1
    _require(not np.array_equal(solve(sideways), expected), "lateral relationships ignored")
    all_cells = tuple((int(r), int(c)) for r, c in np.argwhere(g[:7] == 1))
    _require(_prototype([all_cells]).shape != (2, 2), "whole-color alignment shortcut not distinguished")
    # Recolor source pieces only: color-based routing must change both destinations.
    routed = np.zeros((13, 16), dtype=np.int8)
    routed[:9] = g
    routed[2:4, 9] = 3
    routed[10:12, :2] = routed[10:12, 15] = 4
    routed[10:12, 2:15] = 3
    swapped = routed.copy()
    swapped[:7][routed[:7] == 1] = 3
    swapped[:7][routed[:7] == 3] = 1
    _require(not np.array_equal(solve(routed), solve(swapped)), "routing-color ablation had no effect")
    # Mutation checks: generation metadata is unavailable and labels can be wrong.
    broken = copy.deepcopy(puzzles[0])
    broken["test"][0]["output"].flat[0] = (int(broken["test"][0]["output"].flat[0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes), "prototype_masks": len(masks),
            "palettes": len(palettes), "dock_counts": sorted(dock_counts), "component_counts": sorted(counts),
            "directions": sorted(turns), "depths": sorted(depths), "overlap_pairs": overlaps,
            "multicolor_contact_pairs": contacts, "unused_dock_pairs": unused,
            "disconnected_prototypes": disconnected}


if __name__ == "__main__":
    print(self_test())
