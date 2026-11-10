"""Cooperative stencil carving. Only Python's standard library and NumPy.

A common framed register supplies mixed-color ink. Detect cardinal five-cell
pluses on joint occupancy, then route participating cells to same-color tiles.
There is no learned color, orientation, or generation parameter in solve().
"""
from functools import lru_cache
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ALTERNATIVES = (
    "all_ink", "monochrome", "centers_only", "union_everywhere",
    "center_owned", "discard_unmatched", "destructive", "components",
    "frame_support", "positive_copy",
)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _points(r, c):
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _plus(mask):
    """Production Boolean detector: return centers and simultaneous union."""
    centers = np.zeros_like(mask, dtype=bool)
    centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
        & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:])
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return centers, union


def _scan(mask):
    """Coordinate-set check, also used for constructing planned labels."""
    centers, cells = set(), set()
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if all(mask[p] for p in _points(r, c)):
                centers.add((r, c))
                cells.update(_points(r, c))
    union = np.zeros_like(mask, dtype=bool)
    for p in cells:
        union[p] = True
    return centers, union


def _bbox(mask):
    pts = np.argwhere(mask)
    assert len(pts)
    y, x = map(int, pts.min(axis=0))
    b, r = map(int, pts.max(axis=0) + 1)
    return y, x, b, r


def _halo(g, box, bg):
    y, x, b, r = box
    if not (y > 0 and x > 0 and b < g.shape[0] and r < g.shape[1]):
        return False
    return bool(np.all(g[y-1, x-1:r+1] == bg)
        and np.all(g[b, x-1:r+1] == bg)
        and np.all(g[y:b, x-1] == bg) and np.all(g[y:b, r] == bg))


def _decode(input_grid):
    """Recover frame, register and targets solely from an input."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)), "background border"
    candidates = []
    for marker in map(int, np.unique(g)):
        if marker == bg:
            continue
        y, x, b, r = _bbox(g == marker)
        th, tw = b-y, r-x
        if min(th, tw) < 5:
            continue
        perimeter = np.ones((th, tw), dtype=bool)
        perimeter[1:-1, 1:-1] = False
        if not np.array_equal(g[y:b, x:r] == marker, perimeter):
            continue
        if not _halo(g, (y, x, b, r), bg):
            continue
        exterior = np.ones(g.shape, dtype=bool)
        exterior[y:b, x:r] = False
        colors = [int(v) for v in np.unique(g[exterior]) if v != bg]
        if not 1 <= len(colors) <= 8:
            continue
        sheet = g[y+1:b-1, x+1:r-1]
        tiles, valid = [], True
        for color in colors:
            ty, tx, tb, tr = _bbox(exterior & (g == color))
            if ((tb-ty, tr-tx) != (th, tw)
                    or not np.all(g[ty:tb, tx:tr] == color)
                    or not _halo(g, (ty, tx, tb, tr), bg)
                    or not np.any(sheet == color)):
                valid = False
                break
            tiles.append((ty, tx, color))
        if not valid:
            continue
        cy = min(t[0] for t in tiles)-1
        cx = min(t[1] for t in tiles)-1
        cb = max(t[0]+th for t in tiles)+1
        cr = max(t[1]+tw for t in tiles)+1
        if not (b <= cy or cb <= y or r <= cx or cr <= x):
            continue
        candidates.append(dict(grid=g, bg=bg, marker=marker,
            frame=(y, x, b, r), sheet=sheet, tiles=sorted(tiles),
            tile_shape=(th, tw), crop=(cy, cx, cb, cr)))
    assert len(candidates) == 1, "one unambiguous framed register required"
    return candidates[0]


def _render(info, channels):
    cy, cx, cb, cr = info["crop"]
    out = info["grid"][cy:cb, cx:cr].copy()
    th, tw = info["tile_shape"]
    for y, x, color in info["tiles"]:
        inset = out[y-cy+1:y-cy+th-1, x-cx+1:x-cx+tw-1]
        inset[channels[color]] = info["bg"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution, including palette roles, registration and crop."""
    info = _decode(input_grid)
    _, active = _plus(info["sheet"] != info["bg"])
    return _render(info, {c: active & (info["sheet"] == c)
                          for _, _, c in info["tiles"]})


def _channels(sheet, bg, colors, alternative=None):
    mask = sheet != bg
    centers, active = _scan(mask)
    if alternative == "all_ink":
        active = mask
    elif alternative == "centers_only":
        active = np.zeros_like(mask)
        for p in centers:
            active[p] = True
    elif alternative == "discard_unmatched":
        _, active = _scan(np.isin(sheet, colors))
    elif alternative == "destructive":
        remaining, active = mask.copy(), np.zeros_like(mask)
        for r in range(1, mask.shape[0]-1):
            for c in range(1, mask.shape[1]-1):
                if all(remaining[p] for p in _points(r, c)):
                    for p in _points(r, c):
                        remaining[p] = False
                        active[p] = True
    elif alternative == "components":
        pending = set(map(tuple, np.argwhere(mask)))
        grown = np.zeros_like(mask)
        while pending:
            root = pending.pop()
            stack, component = [root], [root]
            while stack:
                r, c = stack.pop()
                for dr, dc in _DIRS:
                    q = (r+dr, c+dc)
                    if q in pending:
                        pending.remove(q)
                        component.append(q)
                        stack.append(q)
            if any(active[p] for p in component):
                for p in component:
                    grown[p] = True
        active = grown
    elif alternative == "frame_support":
        _, padded = _scan(np.pad(mask, 1, constant_values=True))
        active = padded[1:-1, 1:-1]
    if alternative == "monochrome":
        return {c: _scan(sheet == c)[1] for c in colors}
    if alternative == "union_everywhere":
        return {c: active.copy() for c in colors}
    if alternative == "center_owned":
        result = {c: np.zeros_like(mask) for c in colors}
        for r, c in centers:
            owner = int(sheet[r, c])
            if owner in result:
                for p in _points(r, c):
                    result[owner][p] = True
        return result
    result = {c: active & (sheet == c) for c in colors}
    if alternative == "positive_copy":
        return {c: ~m for c, m in result.items()}
    return result


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [
        np.rot90(np.fliplr(mask), k) for k in range(4)]


def _evidence(sheet, bg, colors):
    correct = _channels(sheet, bg, colors)
    flags = {name: any(not np.array_equal(correct[c], other)
                      for c, other in _channels(sheet, bg, colors, name).items())
             for name in _ALTERNATIVES}
    orientations = set(range(8))
    for mask in correct.values():
        orientations &= {i for i, m in enumerate(_transforms(mask))
                         if np.array_equal(mask, m)}
    return flags, orientations


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


@lru_cache(maxsize=None)
def _layout_options(h, w, n):
    """Minimal feasible target slots and a side for the separate sheet."""
    th, tw = h+2, w+2
    result = []
    for rows in range(1, n+1):
        for cols in range(1, n+1):
            if not n <= rows*cols <= n+2:
                continue
            ht, wt = rows*th+rows-1, cols*tw+cols-1
            for side in range(4):
                hh, ww = ((max(ht, th), wt+1+tw) if side < 2
                          else (ht+1+th, max(wt, tw)))
                if max(hh, ww) <= 28:
                    result.append((rows, cols, side))
    return tuple(result)


def _fits(ht, wt, th, tw, side, gap):
    shape = ((max(ht, th), wt+gap+tw) if side < 2
             else (ht+gap+th, max(wt, tw)))
    return max(shape) <= 28


def _layout(rng, h, w, n):
    th, tw = h+2, w+2
    rows, cols, side = _pick(rng, _layout_options(h, w, n))
    # Complete/partial arrays, unequal gaps, and jitter use the same geometry.
    best = None
    for _ in range(20):
        jitter = int(rng.integers(3)) if rng.random() < .55 else 0
        ys, xs = [0], [0]
        for _ in range(rows-1):
            ys.append(ys[-1]+th+int(rng.integers(1, 4))+jitter)
        for _ in range(cols-1):
            xs.append(xs[-1]+tw+int(rng.integers(1, 4))+jitter)
        slots = [(y+int(rng.integers(jitter+1)), x+int(rng.integers(jitter+1)))
                 for y in ys for x in xs]
        positions = [slots[int(i)] for i in rng.choice(len(slots), n, replace=False)]
        ht, wt = max(y+th for y, x in positions), max(x+tw for y, x in positions)
        if _fits(ht, wt, th, tw, side, 1):
            best = positions
            break
    if best is None:
        slots = [(i*(th+1), j*(tw+1)) for i in range(rows) for j in range(cols)]
        best = [slots[int(i)] for i in rng.choice(len(slots), n, replace=False)]
    # Freely scattered targets sometimes replace the slot layout, bounded by a
    # separate sheet strip. Failure keeps the feasible slot layout.
    if rng.random() < .28:
        zh, zw = (28, 27-tw) if side < 2 else (27-th, 28)
        occupied = np.zeros((zh, zw), dtype=bool)
        candidates = [(y, x) for y in range(zh-th+1) for x in range(zw-tw+1)]
        scattered = []
        for index in rng.permutation(len(candidates)):
            y, x = candidates[int(index)]
            if not occupied[y:y+th, x:x+tw].any():
                scattered.append((y, x))
                occupied[max(0,y-1):y+th+1, max(0,x-1):x+tw+1] = True
                if len(scattered) == n:
                    best = scattered
                    break
    ymin, xmin = min(y for y, x in best), min(x for y, x in best)
    positions = [(y-ymin, x-xmin) for y, x in best]
    ht, wt = max(y+th for y, x in positions), max(x+tw for y, x in positions)
    possible_gaps = [gap for gap in range(1, 5) if _fits(ht, wt, th, tw, side, gap)]
    gap = _pick(rng, possible_gaps)
    if side < 2:
        hh, ww = max(ht, th), wt+gap+tw
        ty = int(rng.integers(hh-ht+1))
        sy = int(rng.integers(hh-th+1))
        tx, sx = (0, wt+gap) if side == 0 else (tw+gap, 0)
    else:
        hh, ww = ht+gap+th, max(wt, tw)
        tx = int(rng.integers(ww-wt+1))
        sx = int(rng.integers(ww-tw+1))
        ty, sy = (0, ht+gap) if side == 2 else (th+gap, 0)
    # Mandatory border plus independent visible blank padding.
    padding = []
    for d in (hh+2, ww+2):
        extra = int(rng.integers(min(4, 30-d)+1))
        if rng.random() < .15:
            extra = 30-d
        before = int(rng.integers(extra+1))
        padding.append((1+before, 1+extra-before))
    py, px = padding[0][0], padding[1][0]
    positions = [(y+ty+py, x+tx+px) for y, x in positions]
    shape = (hh+sum(padding[0]), ww+sum(padding[1]))
    return shape, (sy+py, sx+px), positions


def _paint(mask, pts):
    for r, c in pts:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            mask[r, c] = True


def _occupancy(rng, h, w, kind, n):
    m = np.zeros((h, w), dtype=bool)
    if kind == "single":
        choices = [(r, c, dr, dc) for r in range(1, h-1) for c in range(1, w-1)
                   for dr, dc in _DIRS if 0 <= r+2*dr < h and 0 <= c+2*dc < w]
        r, c, dr, dc = _pick(rng, choices)
        _paint(m, _points(r, c))
        length = 2
        while (0 <= r+(length+1)*dr < h and 0 <= c+(length+1)*dc < w
               and rng.random() < .5):
            length += 1
        _paint(m, [(r+k*dr, c+k*dc) for k in range(2, length+1)])
        if rng.random() < .4:
            a, b = _pick(rng, ((-1,-1), (-1,1), (1,-1), (1,1)))
            m[r+a, c+b] = True
    elif kind == "recombine":
        m[:3, :4] = True
        _paint(m, _points(h-2, w-2))
        if rng.random() < .5:
            m[h-2, w-4] = True
        if rng.random() < .5:
            m = np.flipud(m)
        if rng.random() < .5:
            m = np.fliplr(m)
        return m.copy()
    else:
        style = _pick(rng, ("pluses", "block", "walk", "ring", "near", "pixels", "none"))
        if style in ("pluses", "block", "walk", "ring"):
            for _ in range(int(rng.integers(1, 5))):
                _paint(m, _points(int(rng.integers(1, h-1)), int(rng.integers(1, w-1))))
        if style == "block":
            a, b = int(rng.integers(2, h+1)), int(rng.integers(2, w+1))
            y, x = int(rng.integers(h-a+1)), int(rng.integers(w-b+1))
            m[y:y+a, x:x+b] = True
        elif style == "walk":
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(4, 3*(h+w)))):
                m[r, c] = True
                dr, dc = _pick(rng, _DIRS)
                r, c = min(h-1, max(0, r+dr)), min(w-1, max(0, c+dc))
        elif style == "ring":
            m[0] = m[-1] = True
            m[:, 0] = m[:, -1] = True
        elif style == "near":
            stamps = (((0,0),(0,1),(0,2),(1,1)),
                      ((0,0),(0,2),(1,1),(2,0),(2,2)),
                      ((0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)))
            y, x = int(rng.integers(h-2)), int(rng.integers(w-2))
            _paint(m, [(r+y, c+x) for r, c in _pick(rng, stamps)])
            if rng.random() < .5:
                _paint(m, _points(0, int(rng.integers(1, w-1))))
        elif style in ("pixels", "none"):
            m = rng.random((h, w)) < rng.uniform(.15, .9)
        if m.sum() < n:
            empty = np.argwhere(~m)
            for i in rng.choice(len(empty), n-int(m.sum()), replace=False):
                m[tuple(empty[int(i)])] = True
        if style == "none":
            # An uncut answer is legal. A hollow border provides enough ink
            # for all target identities without any complete pluses.
            m[_plus(m)[0]] = False
            if m.sum() < n:
                m[:] = False
                m[0] = m[-1] = True
                m[:, 0] = m[:, -1] = True
    return m


def _sheet(rng, h, w, colors, unmatched, bg, kind):
    if kind == "application":
        # Finite test fallback: a mixed, full sheet with one protected edge
        # witness. (0,1) can only be activated by center (1,1), whose right
        # neighbor has an unmatched color. Corners remain unselected.
        result = np.full((h,w),colors[0],dtype=np.int8)
        protected = {(0,0),(0,1),(1,1),(1,2)}
        result[1,2] = unmatched[0]
        slots = [(r,c) for r in range(h) for c in range(w) if (r,c) not in protected]
        for color,index in zip(colors[1:],rng.choice(len(slots),len(colors)-1,replace=False)):
            result[slots[int(index)]] = color
        return result
    if kind == "overlap":
        # Adjacent centers, one essential unmatched center, matched corners.
        # A fixed labeled fallback guarantees all named witnesses, while most
        # cases use freshly randomized cell ownership.
        canonical = np.array([[0,1,0,1], [1,0,-1,0], [0,0,1,1]])
        for attempt in range(25):
            labels = rng.integers(len(colors), size=(3,4)) if attempt < 24 else canonical.copy()
            if attempt == 24 and len(colors) == 3:
                labels[1,3] = 2
            labels[1,2] = -1
            block = np.empty((3,4), dtype=np.int8)
            for i, color in enumerate(colors):
                block[labels == i] = color
            block[labels == -1] = unmatched[0]
            if any(not np.any(block == c) for c in colors):
                continue
            turns = int(rng.integers(4))
            block = np.rot90(block, turns)
            if block.shape[0] > h or block.shape[1] > w:
                block = block.T
            bh, bw = block.shape
            y, x = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
            edge = int(rng.integers(4))
            if edge < 2:
                y = 0 if edge == 0 else h-bh
            else:
                x = 0 if edge == 2 else w-bw
            result = np.full((h,w), bg, dtype=np.int8)
            result[y:y+bh, x:x+bw] = block
            flags, orientation = _evidence(result, bg, colors)
            if all(flags.values()) and orientation == {0}:
                return result
        raise AssertionError("constructive overlap witness failed")
    mask = _occupancy(rng, h, w, kind, len(colors))
    result = np.full((h,w), bg, dtype=np.int8)
    ink = colors + unmatched
    result[mask] = rng.choice(ink, int(mask.sum()))
    active = _scan(mask)[1]
    # Simple training shows multiple ownership channels within one plus.
    available = np.argwhere(active if kind == "single" else mask)
    for i, index in enumerate(rng.choice(len(available), len(colors), replace=False)):
        result[tuple(available[int(index)])] = colors[i]
    return result


def _make_pair(rng, config, bg, marker, kind):
    h, w, n = config
    u = 1 if kind in ("overlap","application") else (0 if kind == "single" else int(rng.integers(min(3,8-n)+1)))
    palette = list(map(int, rng.choice([c for c in range(10) if c not in (bg,marker)], n+u, replace=False)))
    colors, unmatched = palette[:n], palette[n:]
    sheet = _sheet(rng, h, w, colors, unmatched, bg, kind)
    shape, (sy, sx), positions = _layout(rng, h, w, n)
    inp = np.full(shape, bg, dtype=np.int8)
    inp[sy:sy+h+2, sx:sx+w+2] = marker
    inp[sy+1:sy+h+1, sx+1:sx+w+1] = sheet
    tiles = [(y, x, color) for (y, x), color in zip(positions, colors)]
    for y, x, color in tiles:
        inp[y:y+h+2, x:x+w+2] = color
    # Ground truth uses planned target positions, not solve's geometry decoder.
    crop = (min(y for y,x,c in tiles)-1, min(x for y,x,c in tiles)-1,
            max(y+h+2 for y,x,c in tiles)+1, max(x+w+2 for y,x,c in tiles)+1)
    planned = dict(grid=inp, bg=bg, tiles=tiles, tile_shape=(h+2,w+2), crop=crop)
    return {"input": inp, "output": _render(planned, _channels(sheet,bg,colors))}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train",num_train,2,10), ("num_test",num_test,1,4)):
        if (isinstance(value, (bool,np.bool_)) or not isinstance(value,(int,np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = _pick(rng, ("more_targets", "larger_register", "recombine"))
    stable = rng.random() < .25
    fixed_bg, fixed_marker = map(int, rng.choice(10, 2, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            kind = ("single" if index == 0 else "overlap") if split == "train" and index < 2 else "random"
            if split == "train":
                sizes = range(3,5) if regime == "larger_register" else range(3,6)
                counts = range(2,4) if kind != "random" else range(1,4)
            elif regime == "more_targets":
                sizes, counts = range(3,6), range(4,7)
            else:
                sizes = range(5,8) if regime == "larger_register" else range(6,8)
                counts = range(1,4)
                if regime == "recombine":
                    kind = "recombine"
            configs = [(h,w,n) for h in sizes for w in sizes for n in counts
                       if (kind not in ("single","overlap") or max(h,w) >= 4)
                       and _layout_options(h,w,n)]
            chosen_count = _pick(rng, sorted({q[2] for q in configs}))
            configs = [q for q in configs if q[2] == chosen_count]
            for attempt in range(40):
                config = _pick(rng, configs)
                bg, marker = (fixed_bg, fixed_marker) if stable else map(int, rng.choice(10,2,replace=False))
                scene_kind = "application" if split == "test" and index == 0 and attempt >= 30 else kind
                pair = _make_pair(rng, config, bg, marker, scene_kind)
                if split == "test" and index == 0:
                    # The held-out application must exercise both mechanisms.
                    # Later tests may include a supported zero-match boundary.
                    info = _decode(pair["input"])
                    flags, _ = _evidence(info["sheet"], bg, [c for y,x,c in info["tiles"]])
                    if not all(flags[k] for k in ("all_ink","union_everywhere","monochrome")):
                        continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct scene sampling exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate actual grids and training evidence, without sampling metadata."""
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, flags, orientations = set(), dict.fromkeys(_ALTERNATIVES,False), set(range(8))
    simple = overlap = False
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(inp)
            sheet, bg = info["sheet"], info["bg"]
            colors = [c for y,x,c in info["tiles"]]
            centers, direct = _scan(sheet != bg)
            cm, vectorized = _plus(sheet != bg)
            assert np.array_equal(direct,vectorized) and len(centers) == int(cm.sum())
            channels = {c: direct & (sheet == c) for c in colors}
            expected = _render(info,channels)
            assert np.array_equal(out, expected), "incorrect holes, perimeter, positions or crop"
            assert np.all(out[0] == bg) and np.all(out[-1] == bg)
            assert np.all(out[:,0] == bg) and np.all(out[:,-1] == bg)
            assert np.array_equal(solve(inp),out)
            # Every changed crop cell is an inset cell of exactly one owner.
            cy,cx,cb,cr = info["crop"]
            changed = inp[cy:cb,cx:cr] != out
            assert int(changed.sum()) == sum(int(m.sum()) for m in channels.values())
            if split == "train":
                evidence, candidates = _evidence(sheet,bg,colors)
                flags = {k: flags[k] or evidence[k] for k in flags}
                orientations &= candidates
                simple |= len(centers) == 1 and len({int(sheet[p]) for p in _points(*next(iter(centers)))}) > 1
                overlap |= 5*len(centers) > int(direct.sum())
            elif index == 0:
                evidence, _ = _evidence(sheet,bg,colors)
                assert all(evidence[k] for k in ("all_ink","union_everywhere","monochrome")), "first test must exercise both parents"
    assert simple and overlap, "need mixed-color simple and overlapping demonstrations"
    assert all(flags.values()), f"missing training evidence: {[k for k,v in flags.items() if not v]}"
    assert orientations == {0}, "training leaves an alternative orientation"


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p["input"])),_key(np.asarray(p["output"])))
                 for s in ("train","test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed coverage, count contracts and semantic regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    dimensions, registers, counts, backgrounds, markers = set(), set(), set(), set(), set()
    center_counts, layouts, structures = set(), set(), set()
    no_plus = mixed_roles = rectangles = irregular = 0
    for puzzle in puzzles:
        roles = set()
        for split in ("train","test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                sheet = info["sheet"]
                colors = [c for y,x,c in info["tiles"]]
                centers, active = _scan(sheet != info["bg"])
                dimensions.add(pair["input"].shape)
                registers.add(sheet.shape)
                counts.add(len(colors))
                backgrounds.add(info["bg"]); markers.add(info["marker"])
                roles.add((info["bg"],info["marker"]))
                center_counts.add(len(centers))
                no_plus += not centers
                rectangles += sheet.shape[0] != sheet.shape[1]
                cy,cx,cb,cr = info["crop"]
                positions = tuple(sorted((y-cy,x-cx) for y,x,c in info["tiles"]))
                layouts.add((sheet.shape,positions))
                yy,xx = {y for y,x in positions},{x for y,x in positions}
                irregular += len(yy)*len(xx) > len(positions)
                roles_map = {}
                structures.add(tuple(roles_map.setdefault(int(v),len(roles_map)) for v in sheet.flat))
        mixed_roles += len(roles) > 1
    assert len(dimensions) >= 35 and len(registers) >= 12
    assert {1,2,3,4} <= counts and len(counts) >= 5
    assert backgrounds == markers == set(range(10))
    assert len(center_counts) >= 6 and len(layouts) >= 70 and len(structures) >= 100
    assert no_plus >= 3 and rectangles >= 40 and irregular >= 15
    assert 5 <= mixed_roles <= 40
    # Solve accepts fitting family members outside the sampler's preferred
    # register/count range; these are ordinary applications, not exceptions.
    for config in ((9,3,2),(3,9,2),(3,3,5),(3,3,8)):
        pair = _make_pair(np.random.default_rng(91),config,0,9,"random")
        assert np.array_equal(solve(pair["input"]),pair["output"])
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(100+nt*10+ne,nt,ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    for bits in range(512):
        mask = np.array([(bits >> i)&1 for i in range(9)],dtype=bool).reshape(3,3)
        assert np.array_equal(_scan(mask)[1],_plus(mask)[1])
    block = np.ones((3,4),dtype=bool)
    assert len(_scan(block)[0]) == 2 and _scan(block)[1].sum() == 8
    clipped = np.zeros((4,5),dtype=bool)
    _paint(clipped,[(0,1),(0,2),(0,3),(1,2),(3,2)])
    assert not _plus(clipped)[1].any(), "no wrapped or frame-supplied neighbor"
    # Metamorphic checks: palette, translation, owner routing, support deletion.
    p = puzzles[0]
    inp = p["train"][0]["input"]
    perm = np.array([7,2,8,0,5,9,1,4,6,3],dtype=np.int8)
    assert np.array_equal(solve(perm[inp]),perm[solve(inp)])
    compact = None
    for puzzle in puzzles:
        for pair in puzzle["train"]:
            if max(pair["input"].shape) < 30:
                compact = pair
                break
        if compact is not None:
            break
    assert compact is not None
    g = compact["input"]
    assert np.array_equal(solve(np.pad(g,((1,0),(0,1)),constant_values=int(g[0,0]))),solve(g))
    checked = False
    for pair in p["train"]:
        info = _decode(pair["input"])
        sheet,bg = info["sheet"],info["bg"]
        colors = [c for y,x,c in info["tiles"]]
        sy,sx,b,r = info["frame"]
        active = _scan(sheet != bg)[1]
        support = (sheet != bg) & ~np.isin(sheet,colors)
        if not support.any() or len(colors) < 2:
            continue
        reduced = pair["input"].copy()
        reduced[sy+1:b-1,sx+1:r-1][support] = bg
        assert np.array_equal(np.isin(sheet,colors), np.isin(_decode(reduced)["sheet"],colors))
        if np.array_equal(solve(reduced),pair["output"]):
            continue
        owner = next(c for c in colors if np.count_nonzero(sheet == c) > 1 and np.any(active & (sheet == c)))
        receiver = next(c for c in colors if c != owner)
        point = tuple(np.argwhere(active & (sheet == owner))[0])
        recolored = pair["input"].copy()
        recolored[sy+1+point[0],sx+1+point[1]] = receiver
        assert np.count_nonzero(solve(recolored) != pair["output"]) == 2
        swapped = pair["input"].copy()
        th,tw = info["tile_shape"]
        y0,x0,c0 = info["tiles"][0]
        y1,x1,c1 = info["tiles"][1]
        swapped[y0:y0+th,x0:x0+tw] = c1
        swapped[y1:y1+th,x1:x1+tw] = c0
        after = solve(swapped)
        cy,cx,cb,cr = info["crop"]
        before = pair["output"]
        assert np.array_equal(after[y0-cy:y0-cy+th,x0-cx:x0-cx+tw],
                              before[y1-cy:y1-cy+th,x1-cx:x1-cx+tw])
        checked = True
        break
    assert checked
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0,0] = (int(bad["test"][0]["output"][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted wrong labels")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, semantic regressions)")
