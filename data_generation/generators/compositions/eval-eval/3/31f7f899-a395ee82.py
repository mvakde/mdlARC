"""Sorted calibration of an anchored, nonuniform stencil lattice.

No parent generator is imported. Generation, input decoding, copy-loop solving,
and inverse-pixel validation use separate representations of the rule.
"""
from itertools import combinations_with_replacement, permutations

import numpy as np


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= s <= 30 for s in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(p):
    return tuple((_key(q["input"]), _key(q["output"]))
                 for s in ("train", "test") for q in p[s])


def _components(mask):
    todo = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while todo:
        start = min(todo)
        todo.remove(start)
        seen, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    seen.append(q)
                    stack.append(q)
        groups.append(seen)
    return groups


def _gap(a, b):
    # Half-open rectangles with at least one intervening blank row/column.
    t, l, h, w = a
    u, v, x, y = b
    return t+h < u or u+x < t or l+w < v or v+y < l


def _asymmetric(mask):
    others = (np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2),
              mask.T, np.flipud(mask).T, np.fliplr(mask).T,
              np.rot90(mask, 2).T)
    return all(not np.array_equal(mask, q) for q in others)


def _decode(value):
    """Derive every semantic parameter from input pixels alone."""
    g = _grid(value)
    histogram = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(histogram))
    assert np.count_nonzero(histogram == histogram[bg]) == 1
    rows = np.flatnonzero(np.all(g != bg, axis=1))
    cols = np.flatnonzero(np.all(g != bg, axis=0))
    assert len(rows) + len(cols) == 1, "carrier must be unique"
    vertical = bool(len(cols))
    g = g.T if vertical else g
    row = int(cols[0] if vertical else rows[0])
    line = g[row]
    counts = np.bincount(line, minlength=10)
    carrier = int(np.argmax(counts))
    assert carrier != bg and np.count_nonzero(counts == counts[carrier]) == 1
    assert counts[carrier] >= 3
    assert np.count_nonzero(g == carrier) == counts[carrier]
    remaining = g.copy()
    remaining[row] = bg
    rack = np.full_like(g, bg)
    rack[row] = carrier
    records = []
    x = 0
    while x < g.shape[1]:
        color, end = int(line[x]), x+1
        while end < len(line) and line[end] == color:
            end += 1
        if color != carrier:
            assert color != bg
            top = bottom = row
            while top and np.all(g[top-1, x:end] == color):
                top -= 1
            while bottom+1 < len(g) and np.all(g[bottom+1, x:end] == color):
                bottom += 1
            assert 1 <= end-x <= 3 and 1 <= bottom-top+1 <= 5
            rack[top:bottom+1, x:end] = color
            remaining[top:bottom+1, x:end] = bg
            records.append((x, end-x, color, bottom-top+1, top, bottom))
        x = end
    n = len(records)
    assert 2 <= n <= 5 and len({q[2] for q in records}) == n
    assert len({q[3] for q in records}) >= 2
    extras = set(map(int, np.unique(g))) - {bg, carrier} - {q[2] for q in records}
    assert len(extras) == 1
    ink = extras.pop()
    groups = _components(remaining == ink)
    assert len(groups) == 2 and sorted(map(len, groups))[0] == 1
    glyph = next(q for q in groups if len(q) > 1)
    anchor_cell = next(q[0] for q in groups if len(q) == 1)
    coords = np.array(glyph)
    tr, tc = map(int, coords.min(axis=0))
    br, bc = map(int, coords.max(axis=0))
    h, w = br-tr+1, bc-tc+1
    assert 2 <= h <= 5 and 2 <= w <= 5 and len(glyph) >= 3
    mask = remaining[tr:br+1, tc:bc+1] == ink
    remaining[tr:br+1, tc:bc+1][mask] = bg
    points = [tuple(map(int, p)) for p in np.argwhere(remaining != bg)]
    assert len(points) >= 2 and anchor_cell in points
    assert all(len(q) == 1 for q in _components(remaining != bg)), "map points must be isolated"
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    assert (ec-pc)//2+1 == n and (ec-pc) % 2 == 0
    mh = (er-pr)//2+1
    assert (er-pr) % 2 == 0 and 1 <= mh <= 5
    assert _gap((tr, tc, h, w), (pr, pc, er-pr+1, ec-pc+1))
    layout = np.zeros((mh, n), dtype=bool)
    anchor = ((anchor_cell[0]-pr)//2, (anchor_cell[1]-pc)//2)
    expected = rack.copy()
    expected[tr:tr+h, tc:tc+w][mask] = ink
    for r, c in points:
        assert (r-pr) % 2 == 0 and (c-pc) % 2 == 0
        a, b = (r-pr)//2, (c-pc)//2
        is_anchor = (r, c) == anchor_cell
        assert int(remaining[r, c]) == (ink if is_anchor else records[b][2])
        layout[a, b] = True
        expected[r, c] = ink if is_anchor else records[b][2]
    assert np.array_equal(expected, g), "unexplained cells or nonsolid rack"
    assert layout[0].any() and layout[-1].any() and layout[:, 0].any() and layout[:, -1].any()
    info = {"shape": g.shape, "vertical": vertical, "background": bg,
            "carrier": carrier, "ink": ink, "records": records,
            "mask": mask, "glyph_origin": (tr, tc), "layout": layout,
            "anchor": anchor, "map_origin": (pr, pc), "carrier_row": row}
    assert _bounds(info, sorted(q[3] for q in records)) is not None
    return info


def _bounds(info, factors, anchor=None):
    h, w = info["mask"].shape
    mh, n = info["layout"].shape
    a, b = info["anchor"] if anchor is None else anchor
    prefix = np.concatenate(([0], np.cumsum(factors)))
    tr, tc = info["glyph_origin"]
    top, left = tr-a*h, tc-int(prefix[b])*w
    height, width = mh*h, int(prefix[-1])*w
    gh, gw = info["shape"]
    if top < 0 or left < 0 or top+height > gh or left+width > gw:
        return None
    return top, left, height, width, prefix


def _copy_prediction(info, factors=None, mask=None, invert=True, anchor=None):
    """Copy-loop solver; also used for explicit training alternatives."""
    factors = sorted(q[3] for q in info["records"]) if factors is None else list(factors)
    box = _bounds(info, factors, anchor)
    if box is None:
        return None
    top, left, _, _, prefix = box
    mask = info["mask"] if mask is None else mask
    h, w = mask.shape
    anchor = info["anchor"] if anchor is None else anchor
    out = np.full(info["shape"], info["background"], dtype=np.int8)
    for r, c in np.argwhere(info["layout"]):
        repeated = np.repeat(mask, factors[c], axis=1)
        target = out[top+r*h:top+(r+1)*h,
                     left+int(prefix[c])*w:left+int(prefix[c+1])*w]
        color = info["records"][c][2] if invert and (r, c) == anchor else info["ink"]
        target[repeated] = color
    return out.T.copy() if info["vertical"] else out


def _compressed_prediction(info, axis):
    """A specific incorrect hypothesis: discard entirely empty lattice bands."""
    used = np.flatnonzero(info["layout"].any(axis=1-axis))
    alternate = dict(info)
    alternate["layout"] = np.take(info["layout"], used, axis=axis)
    anchor = list(info["anchor"])
    anchor[axis] = list(used).index(anchor[axis])
    alternate["anchor"] = tuple(anchor)
    factors = sorted(q[3] for q in info["records"])
    if axis == 1:
        alternate["records"] = [info["records"][j] for j in used]
        factors = [factors[j] for j in used]
    return _copy_prediction(alternate, factors=factors)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only inference, with no seed, examples, or generation state."""
    try:
        return _copy_prediction(_decode(input_grid))
    except (AssertionError, IndexError, TypeError) as exc:
        raise ValueError("input is outside the elastic stencil family") from exc


def _inverse_check(inp, out):
    """Map each output pixel back to its band, map site, and glyph pixel."""
    info = _decode(inp)
    actual = _grid(out)
    assert actual.shape == np.asarray(inp).shape
    if info["vertical"]:
        actual = actual.T
    factors = np.array(sorted(q[3] for q in info["records"]), dtype=int)
    top, left, height, width, prefix = _bounds(info, factors)
    h, w = info["mask"].shape
    map_row, glyph_row = np.divmod(np.arange(height), h)
    columns = np.arange(width)
    band = np.searchsorted(prefix*w, columns, side="right")-1
    glyph_col = (columns-prefix[band]*w)//factors[band]
    support = (info["layout"][map_row[:, None], band[None, :]] &
               info["mask"][glyph_row[:, None], glyph_col[None, :]])
    a, b = info["anchor"]
    exceptional = (map_row[:, None] == a) & (band[None, :] == b)
    colors = np.where(exceptional, info["records"][b][2], info["ink"])
    expected = np.full(info["shape"], info["background"], dtype=np.int8)
    tile = expected[top:top+height, left:left+width]
    tile[support] = np.broadcast_to(colors, support.shape)[support]
    assert np.array_equal(expected, actual), "incorrect mask, calibration, anchor, or color"
    mass = int(info["mask"].sum()) * int(np.sum(info["layout"] * factors[None, :]))
    assert np.count_nonzero(actual != info["background"]) == mass
    assert set(map(int, np.unique(actual))) == {info["background"], info["ink"], info["records"][b][2]}
    assert np.count_nonzero(actual == info["records"][b][2]) == int(info["mask"].sum())*factors[b]
    assert not np.array_equal(np.asarray(inp), np.asarray(out))
    return info


def _glyph(rng, h, w, informative=False):
    if informative:
        mask = np.ones((h, w), dtype=bool)
        mask[_pick(rng, [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)])] = False
        return mask
    # A random corner-to-corner path gives tight bounds and connectedness.
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    steps = [0]*(h-1) + [1]*(w-1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        mask[r, c] = True
    goal = max(int(mask.sum()), int(round(h*w*float(_pick(rng, [0.0, 0.5, 0.7, 0.9, 1.0])))))
    while mask.sum() < goal:
        frontier = set()
        for r, c in np.argwhere(mask):
            for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= a < h and 0 <= b < w and not mask[a, b]:
                    frontier.add((int(a), int(b)))
        mask[_pick(rng, sorted(frontier))] = True
    return np.fliplr(mask).copy() if rng.integers(2) else mask


def _calibration(rng, n, witness, previous_width):
    multisets = [p for p in combinations_with_replacement(range(1, 6), n)
                 if len(set(p)) >= 2 and (witness != 1 or {1, 2, 3} <= set(p))]
    widths = [w for w in range(2, 6) if (witness != 2 or w != previous_width)
              and any(w*sum(p) <= 26 for p in multisets)]
    glyph_width = _pick(rng, widths)
    values = _pick(rng, [p for p in multisets if glyph_width*sum(p) <= 26])
    if witness:
        choices = sorted({p for p in permutations(values)
                          if list(p) != sorted(values) and list(p)[::-1] != sorted(values)})
        lengths = list(_pick(rng, choices))
    else:
        lengths = list(values)
        mode = int(rng.integers(5))
        if mode <= 1:
            rng.shuffle(lengths)
        elif mode == 2:
            lengths.reverse()
        elif mode == 3:
            i = _pick(rng, [i for i in range(n-1) if lengths[i] != lengths[i+1]])
            lengths[i], lengths[i+1] = lengths[i+1], lengths[i]
    return lengths, glyph_width


def _map(rng, mh, n, gap, witness, previous_anchor, reflected, density):
    rows, cols = list(range(mh)), list(range(n))
    if gap:
        rows.remove(2)
        cols.remove(int(rng.integers(1, n-1)))
    if witness == 1:
        anchor = (_pick(rng, [0, mh-1]), 0)
    else:
        choices = [(r, c) for r in rows for c in cols
                   if witness != 2 or (c != previous_anchor[1] and
                       (mh-1-r if reflected else r) != previous_anchor[0])]
        anchor = _pick(rng, choices)
    layout = np.zeros((mh, n), dtype=bool)
    layout[anchor] = True
    # Every supported band and both row extremes are visibly represented.
    for c in cols:
        layout[_pick(rng, rows), c] = True
    for r in (rows[0], rows[-1]):
        layout[r, _pick(rng, cols)] = True
    if witness:
        adjacent = _pick(rng, [c for c in cols if c+1 in cols])
        layout[rows[0], adjacent:adjacent+2] = True
        layout[0:2, cols[0]] = True
    free = [(r, c) for r in rows for c in cols if not layout[r, c]]
    rng.shuffle(free)
    goal = int(np.ceil(len(rows)*len(cols)*density))
    for p in free[:max(0, goal-int(layout.sum()))]:
        layout[p] = True
    return layout, anchor


def _construct(rng, lengths, widths, mask, layout, anchor, colors, reflected, vertical):
    """Construct input and assemble output with per-band Kronecker products."""
    bg, carrier, ink, *keys = colors
    h, w = mask.shape
    mh, n = layout.shape
    factors = sorted(lengths)
    prefix = np.concatenate(([0], np.cumsum(factors)))
    ah, aw = mh*h, sum(factors)*w
    upper = max(length//2 for length in lengths)
    lower = max((length-1)//2 for length in lengths)
    bh, ph, pw = upper+lower+1, 2*mh-1, 2*n-1
    base_h = max(max(0, bh+1-anchor[0]*h)+ah, bh+1+max(h, ph))
    rack_top = int(rng.integers(min(3, 30-base_h)+1))
    free_start = rack_top+bh+1
    rmin = max(0, free_start-anchor[0]*h)
    min_h = max(rmin+ah, free_start+max(h, ph))
    min_w = max(aw, sum(widths)+max(3, max(widths)+1), w+pw+1)
    gh = 30 if rng.random() < 0.13 else int(rng.integers(min_h, min(30, min_h+7)+1))
    gw = 30 if rng.random() < 0.13 else int(rng.integers(min_w, min(30, min_w+7)+1))
    top = int(rng.integers(rmin, gh-ah+1))
    left = int(rng.integers(gw-aw+1))
    tr, tc = top+anchor[0]*h, left+int(prefix[anchor[1]])*w

    def map_positions():
        return [(r, c) for r in range(free_start, gh-ph+1) for c in range(gw-pw+1)
                if _gap((tr, tc, h, w), (r, c, ph, pw))]

    positions = map_positions()
    if not positions:
        # Width 30 always leaves a horizontal map placement, regardless of anchor.
        gw = 30
        left = int(rng.integers(gw-aw+1))
        tc = left+int(prefix[anchor[1]])*w
        positions = map_positions()
    assert positions
    pr, pc = _pick(rng, positions)
    extra = gw-sum(widths)
    gaps = [1] + [0]*(n-1) + [1]
    weights = rng.dirichlet(np.ones(n+1))
    for i, amount in enumerate(rng.multinomial(extra-2, weights)):
        gaps[i] += int(amount)
    inp = np.full((gh, gw), bg, dtype=np.int8)
    row = rack_top+upper
    inp[row] = carrier
    start = gaps[0]
    for j, (length, width) in enumerate(zip(lengths, widths)):
        lo, hi = row-length//2, row+(length-1)//2+1
        inp[lo:hi, start:start+width] = keys[j]
        start += width+gaps[j+1]
    inp[tr:tr+h, tc:tc+w][mask] = ink
    for r, c in np.argwhere(layout):
        inp[pr+2*r, pc+2*c] = ink if (r, c) == anchor else keys[c]
    bands = []
    for j, factor in enumerate(factors):
        stretched = np.kron(mask, np.ones((1, factor), dtype=bool))
        occupied = np.kron(layout[:, j:j+1], stretched)
        band = np.full(occupied.shape, bg, dtype=np.int8)
        band[occupied] = ink
        if j == anchor[1]:
            band[anchor[0]*h:(anchor[0]+1)*h][stretched] = keys[j]
        bands.append(band)
    carpet = np.concatenate(bands, axis=1)
    out = np.full_like(inp, bg)
    out[top:top+ah, left:left+aw] = carpet
    if reflected:
        inp, out = np.flipud(inp).copy(), np.flipud(out).copy()
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _sample_pair(rng, config, split, n, vertical, witness=0, previous=None):
    lengths, w = _calibration(rng, n, witness, None if previous is None else previous["mask"].shape[1])
    if witness == 2:
        dims = [(h, mh) for h in range(2, 6) for mh in (4, 5)
                if h != previous["mask"].shape[0] and h*mh <= 24]
    else:
        row_options = (3, 4, 5) if config["regime"] == 2 else (2, 3, 4) if witness else (1, 2, 3, 4, 5)
        dims = [(h, mh) for h in range(2, 6) for mh in row_options if h*mh <= 24
                and (witness != 1 or h != w)]
    h, mh = _pick(rng, dims)
    mask = _glyph(rng, h, w, informative=witness == 1)
    widths = [int(x) for x in rng.integers(1, 4, size=n)]
    if not witness and rng.random() < 0.25:
        widths = [_pick(rng, [1, 2, 3])]*n
    if witness == 1:
        widths[lengths.index(2)] = 3
        widths[lengths.index(3)] = 1
    reflected = bool(rng.integers(2))
    gap = witness == 2 or (not witness and n >= 4 and mh >= 4 and rng.random() < 0.30)
    density = float(rng.uniform(0.85, 1.0) if split == "test" and config["regime"] == 2
                    else rng.uniform(0.30, 0.60) if config["regime"] == 2 else rng.uniform(0.2, 0.95))
    layout, anchor = _map(rng, mh, n, gap, witness,
                          None if previous is None else previous["anchor"], reflected, density)
    palette = config["palette"] if config["fixed_palette"] else list(map(int, rng.permutation(10)))
    colors = palette[:n+3]
    return _construct(rng, lengths, widths, mask, layout, anchor, colors, reflected, vertical)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _novel_pair(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # Forty bijective affine palettes, each different on >=5 visible colors,
    # outnumber the at most thirteen earlier inputs. No retry exhaustion.
    for multiplier in (1, 3, 7, 9):
        for offset in range(10):
            palette = ((multiplier*np.arange(10)+offset) % 10).astype(np.int8)
            candidate = {role: palette[pair[role]] for role in ("input", "output")}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("finite palette bound violated")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    config = {"regime": int(rng.integers(3)), "fixed_palette": bool(rng.random() < 0.25),
              "palette": list(map(int, rng.permutation(10)))}
    orientation = int(rng.integers(3))
    axes = [orientation == 1]*nt if orientation < 2 else [False, True]+[bool(rng.integers(2)) for _ in range(nt-2)]
    train_range = [4, 5] if config["regime"] == 1 else [3, 4]
    test_range = [5] if config["regime"] == 0 else [2, 3] if config["regime"] == 1 else [3, 4]
    p, seen, previous = {"train": [], "test": []}, set(), None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i+1 if split == "train" and i < 2 else 0
            n = train_range[i] if witness else _pick(rng, train_range if split == "train" else test_range)
            axis = axes[i] if split == "train" else (orientation == 1 if orientation < 2 else bool(rng.integers(2)))
            pair = _novel_pair(_sample_pair(rng, config, split, n, axis, witness, previous), seen)
            seen.add(_key(pair["input"]))
            p[split].append(pair)
            if witness == 1:
                previous = _decode(pair["input"])
    rng.shuffle(p["train"])
    validate_generated_puzzle(p)
    return p


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    metas, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            metas[split].append(_inverse_check(inp, out))
    train = metas["train"]
    assert len({len(m["records"]) for m in train}) >= 2
    assert len({m["mask"].shape[0] for m in train}) >= 2
    assert len({m["mask"].shape[1] for m in train}) >= 2
    assert len({m["anchor"][0] for m in train}) >= 2
    assert len({m["anchor"][1] for m in train}) >= 2
    assert any(_asymmetric(m["mask"]) and not m["mask"].all() for m in train)
    assert any(np.any(m["layout"][:-1] & m["layout"][1:]) and
               np.any(m["layout"][:, :-1] & m["layout"][:, 1:]) for m in train)
    assert sum([q[3] for q in m["records"]] != sorted(q[3] for q in m["records"]) and
               [q[3] for q in m["records"]][::-1] != sorted(q[3] for q in m["records"])
               for m in train) >= 2
    for m in metas["test"]:
        assert m["vertical"] in {q["vertical"] for q in train}
        for axis in (0, 1):
            if np.any(~np.any(m["layout"], axis=1-axis)):
                assert any(np.any(~np.any(q["layout"], axis=1-axis)) for q in train)
    rejected = {name: False for name in ("unsorted", "area", "reversed", "uniform", "solid", "reflected", "no_inversion", "compress_rows", "compress_bands")}
    for pair, m in zip(puzzle["train"], train):
        lengths = [q[3] for q in m["records"]]
        alternatives = {
            "unsorted": {"factors": lengths},
            "area": {"factors": [q[3] for q in sorted(m["records"], key=lambda q: q[1]*q[3])]},
            "reversed": {"factors": lengths[::-1]},
            "uniform": {"factors": [1]*len(lengths)},
            "solid": {"mask": np.ones_like(m["mask"])},
            "reflected": {"mask": np.fliplr(m["mask"])},
            "no_inversion": {"invert": False}}
        for name, arguments in alternatives.items():
            candidate = _copy_prediction(m, **arguments)
            rejected[name] |= candidate is not None and not np.array_equal(candidate, pair["output"])
        for axis, name in ((0, "compress_rows"), (1, "compress_bands")):
            candidate = _compressed_prediction(m, axis)
            rejected[name] |= candidate is not None and not np.array_equal(candidate, pair["output"])
    assert all(rejected.values()), f"insufficient training alternatives: {rejected}"
    return metas


def _features(info, pair):
    layout = info["layout"]
    lengths = tuple(q[3] for q in info["records"])
    widths = tuple(q[1] for q in info["records"])
    mask = info["mask"]
    n = len(lengths)
    return {"shape": tuple(np.asarray(pair["input"]).shape), "vertical": info["vertical"],
            "bands": n, "map_shape": layout.shape, "sites": int(layout.sum()),
            "anchor": info["anchor"], "glyph_shape": mask.shape,
            "glyph": (mask.shape, mask.tobytes()), "layout": (layout.shape, layout.tobytes()),
            "lengths": lengths, "widths": widths, "ties": len(set(lengths)) < n,
            "mixed_widths": len(set(widths)) > 1, "empty_rows": bool(np.any(~layout.any(axis=1))),
            "empty_bands": bool(np.any(~layout.any(axis=0))), "density": float(layout.mean()),
            "roles": (info["background"], info["carrier"], info["ink"]),
            "touching": any(b[0] == a[0]+a[1] for a, b in zip(info["records"], info["records"][1:])),
            "solid_glyph": bool(mask.all())}


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    rows = []
    anchor_changes = 0
    for p in puzzles:
        metas = validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair, m in zip(p[split], metas[split]):
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                row = _features(m, pair)
                row["split"] = split
                rows.append(row)
                if anchor_changes < 12:
                    pr, pc = m["map_origin"]
                    a, b = m["anchor"]
                    canonical = pair["input"].T.copy() if m["vertical"] else pair["input"].copy()
                    for r, c in np.argwhere(m["layout"]):
                        if (r, c) == (a, b):
                            continue
                        altered = canonical.copy()
                        altered[pr+2*a, pc+2*b] = m["records"][b][2]
                        altered[pr+2*r, pc+2*c] = m["ink"]
                        if m["vertical"]:
                            altered = altered.T.copy()
                        try:
                            changed = solve(altered)
                        except ValueError:
                            continue
                        assert not np.array_equal(changed, pair["output"])
                        _inverse_check(altered, changed)
                        anchor_changes += 1
                        break
        # These transformations preserve the rule even outside sampled policies.
        pair = p["train"][0]
        assert np.array_equal(solve(pair["input"].T), pair["output"].T)
        m = metas["train"][0]
        flip_axis = 1 if m["vertical"] else 0
        assert np.array_equal(solve(np.flip(pair["input"], axis=flip_axis)),
                              np.flip(pair["output"], axis=flip_axis))
        palette = np.arange(9, -1, -1, dtype=np.int8)
        assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    assert anchor_changes >= 12
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(7000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    assert len({r["shape"] for r in rows}) >= 40
    assert len({r["glyph"] for r in rows}) >= 30
    assert len({r["layout"] for r in rows}) >= 50
    assert len({r["anchor"] for r in rows}) >= 12
    assert {r["bands"] for r in rows} == {2, 3, 4, 5}
    assert {r["vertical"] for r in rows} == {False, True}
    assert {r["glyph_shape"][0] for r in rows} == {2, 3, 4, 5}
    assert {r["glyph_shape"][1] for r in rows} == {2, 3, 4, 5}
    assert {l for r in rows for l in r["lengths"]} == {1, 2, 3, 4, 5}
    assert {w for r in rows for w in r["widths"]} == {1, 2, 3}
    assert {r["solid_glyph"] for r in rows} == {False, True}
    assert any(r["map_shape"][0] == 1 for r in rows)
    assert any(r["sites"] >= 15 for r in rows)
    for property_name in ("ties", "mixed_widths", "empty_rows", "empty_bands", "touching"):
        assert {r[property_name] for r in rows} == {False, True}
    assert {c for r in rows for c in r["roles"]} == set(range(10))
    broken = {s: [{k: np.asarray(g).copy() for k, g in pair.items()} for pair in puzzles[0][s]]
              for s in ("train", "test")}
    broken["train"][0]["output"][0, 0] = (int(broken["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(rows), "shapes": len({r["shape"] for r in rows}),
            "glyph_masks": len({r["glyph"] for r in rows}),
            "map_layouts": len({r["layout"] for r in rows}),
            "anchors": len({r["anchor"] for r in rows}), "anchor_counterfactuals": anchor_changes}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
