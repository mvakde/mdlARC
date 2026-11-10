"""Address-space growth and line fusion. Standard library and NumPy only.

Labels are rendered from sampled tokens and explicit affine rectangles. solve
recovers the rectangles from unordered glyph keys in the supplied pixels.
"""
from itertools import product
from functools import lru_cache

import numpy as np


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside its permitted range")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _undo(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_marker(a):
    found = []
    for color in np.unique(a):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(0))
        r1, c1 = map(int, cells.max(0))
        if min(r1 - r0, c1 - c0) < 1 or r1 - r0 + c1 - c0 < 4:
            continue
        bb = r0, r1, c0, c1
        if set(map(tuple, cells)) == _corners(bb):
            found.append((int(color), bb))
    assert len(found) == 1, "ambiguous or malformed L annotations"
    return found[0]


def _trim(a, bg):
    cells = np.argwhere(a != bg)
    assert len(cells)
    lo, hi = cells.min(0), cells.max(0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]]


def _selection(rect, n):
    r, c, h, w = map(int, rect)
    if min(h, w) <= 0:
        return set()
    return {(rr, cc) for rr in range(max(0, r), min(n, r + h))
            for cc in range(max(0, c), min(n, c + w))}


def _rectangle(cells):
    assert cells, "observed snapshots must be nonempty"
    rr, cc = zip(*cells)
    rect = min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1
    r, c, h, w = rect
    assert cells == {(r + i, c + j) for i in range(h) for j in range(w)}, "not a filled rectangle"
    return rect


def _future(rectangles):
    x = np.asarray(rectangles, dtype=int)
    delta = x[1] - x[0]
    assert np.array_equal(x[2] - x[1], delta), "non-affine address trajectory"
    return tuple(map(int, x[2] + delta)), tuple(map(int, delta))


def _lines(selected, n):
    rows = {r for r in range(n) if all((r, c) in selected for c in range(n))}
    cols = {c for c in range(n) if all((r, c) in selected for r in range(n))}
    return rows, cols


def _paint(info, selected, fuse=True):
    """Input-side pixel predicates; not the construction renderer."""
    field = info["board_field"].copy()
    gh, gw, n = info["gh"], info["gw"], info["n"]
    sr, sc = gh + 2, gw + 2
    rr, cc = np.indices(field.shape)
    tile_r, tile_c = rr // sr, cc // sc
    local_r, local_c = rr % sr, cc % sc
    footprint = (local_r < gh + 1) & (local_c < gw + 1)
    chosen = np.zeros(field.shape, bool)
    for r, c in selected:
        chosen |= (tile_r == r) & (tile_c == c) & footprint
    if fuse:
        rows, cols = _lines(selected, n)
        for r in rows:
            chosen |= (tile_r == r) & (local_r < gh + 1)
        for c in cols:
            chosen |= (tile_c == c) & (local_c < gw + 1)
    field[chosen & (field == info["bg"])] = info["paint"]
    return field


def _crop(field, info):
    r0, r1, c0, c1 = info["local_bounds"]
    return field[r0:r1 + 1, c0:c1 + 1].copy()


def _parse_canonical(a, marker_mask, bg):
    bars = [r for r in range(len(a)) if a[r, 0] != bg and np.all(a[r] == a[r, 0])]
    assert len(bars) == 4
    hb = bars[0]
    step = bars[1] - bars[0]
    gh = step - 1
    assert gh in (2, 3) and bars == [hb + k * step for k in range(4)]
    assert len(a) == bars[-1] + 1
    paint = int(a[hb, 0])
    assert all(np.all(a[r] == paint) for r in bars)
    assert (hb + 1) % (gh + 2) == 0
    n = (hb + 1) // (gh + 2)
    assert 3 <= n <= 5
    dividers = [c for c in range(a.shape[1]) if np.all(a[:hb, c] == paint)]
    assert len(dividers) == 1
    wb = dividers[0]
    assert (wb + 1) % n == 0
    gw = (wb + 1) // n - 2
    assert gw in (2, 3)
    cells = np.argwhere(marker_mask)
    assert len(cells) == 6
    r0, c0 = map(int, cells.min(0))
    r1, c1 = map(int, cells.max(0))
    assert 0 <= r0 <= r1 < hb and wb + 1 <= c0 <= c1 < a.shape[1]
    local = r0, r1, c0 - wb - 1, c1 - wb - 1
    assert local[3] < wb
    assert np.all(a[:hb, wb + 1:] == bg), "unexpected evidence in future field"
    expected = np.full(a.shape, bg, np.int8)
    expected[bars] = paint
    expected[:hb, wb] = paint
    board, lookup, trimmed = {}, {}, set()
    for r, c in product(range(n), repeat=2):
        y, x = r * (gh + 2), c * (gw + 2)
        token = a[y:y + gh, x:x + gw].copy()
        colors = np.unique(token[token != bg])
        assert len(colors) == 1 and int(colors[0]) != paint
        trim = _trim(token, bg)
        pos = np.argwhere(token != bg).min(0)
        assert tuple(pos) == ((gh - len(trim)) // 2, (gw - trim.shape[1]) // 2)
        key, tk = _key(token), _key(trim)
        assert key not in lookup and tk not in trimmed, "duplicate colored glyph"
        lookup[key] = r, c
        trimmed.add(tk)
        board[r, c] = token
        expected[y:y + gh, x:x + gw] = token
    capacity = (a.shape[1] + 1) // (gw + 1)
    snapshots, orders, rectangles = [], [], []
    for i in range(3):
        y = hb + 1 + i * step
        order = []
        for q in range(capacity):
            x = q * (gw + 1)
            token = a[y:y + gh, x:x + gw]
            if np.all(token == bg):
                continue
            assert _key(token) in lookup, "query key has no exact board copy"
            order.append(lookup[_key(token)])
            expected[y:y + gh, x:x + gw] = token
        assert len(order) >= 1 and len(set(order)) == len(order)
        selected = set(order)
        rectangles.append(_rectangle(selected))
        snapshots.append(selected)
        orders.append(order)
    assert np.array_equal(expected, a), "ink outside the declared lattice and strips"
    future, delta = _future(rectangles)
    return dict(bg=bg, paint=paint, n=n, gh=gh, gw=gw, hb=hb, wb=wb,
                board=board, board_field=a[:hb, :wb].copy(), snapshots=snapshots,
                orders=orders, rectangles=rectangles, future=future, delta=delta,
                selected=_selection(future, n), local_bounds=local)


def _analyze(input_grid):
    a = _grid(input_grid)
    marker, bounds = _find_marker(a)
    vals, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "background frequency tie"
    bg = int(vals[counts.argmax()])
    assert bg != marker
    clean = a.copy()
    clean[clean == marker] = bg
    marker_mask = a == marker
    candidates = []
    for code in range(8):
        try:
            info = _parse_canonical(_orient(clean, code), _orient(marker_mask, code), bg)
        except AssertionError:
            continue
        info.update(code=code, marker=marker, bounds=bounds)
        result = _undo(_crop(_paint(info, info["selected"]), info), code)
        candidates.append((result, info))
    assert candidates, "no supported address-space scene"
    assert all(np.array_equal(candidates[0][0], p) for p, _ in candidates), "ambiguous scene orientation"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer only from the supplied input pixels."""
    return _analyze(input_grid)[0]


@lru_cache(maxsize=4)
def _mask_bank(gh, gw):
    bank = {}
    for bits in range(1, 1 << (gh * gw)):
        mask = np.array([(bits >> k) & 1 for k in range(gh * gw)], bool).reshape(gh, gw)
        trim = _trim(mask, False)
        centered = np.zeros((gh, gw), bool)
        y, x = (gh - len(trim)) // 2, (gw - trim.shape[1]) // 2
        centered[y:y + len(trim), x:x + trim.shape[1]] = trim
        bank[_key(centered)] = centered
    return list(bank.values())


def _make_tokens(rng, n, gh, gw, bg, inks, snapshots):
    bank = _mask_bank(gh, gw)
    # A selected key has both kinds of reduced-key aliases outside every query.
    used = set.union(*snapshots)
    outside = sorted(set(product(range(n), repeat=2)) - used)
    assert len(outside) >= 2
    anchor = sorted(snapshots[0])[0]
    a, b = list(map(int, rng.choice(len(bank), 2, replace=False)))
    specs = {anchor: (a, inks[0]), outside[0]: (a, inks[1]), outside[1]: (b, inks[0])}
    available = [(s, ink) for s in range(len(bank)) for ink in inks if (s, ink) not in specs.values()]
    # Vary mask reuse instead of forcing a fixed shape/ink multiplicity.
    if rng.random() < .5:
        preferred = list(map(int, rng.choice(len(bank), min(len(bank), n + 2), replace=False)))
        subset = [p for p in available if p[0] in preferred]
        if len(subset) >= n * n - 3:
            available = subset
    rng.shuffle(available)
    for pos in product(range(n), repeat=2):
        if pos not in specs:
            specs[pos] = available.pop()
    tokens = {}
    for pos, (s, ink) in specs.items():
        token = np.full((gh, gw), bg, np.int8)
        token[bank[s]] = ink
        tokens[pos] = token
    return tokens


def _board_canvas(tokens, n, gh, gw, bg):
    field = np.full((n * (gh + 2) - 1, n * (gw + 2) - 1), bg, np.int8)
    for (r, c), token in tokens.items():
        y, x = r * (gh + 2), c * (gw + 2)
        field[y:y + gh, x:x + gw] = token
    return field


def _render_future(tokens, n, gh, gw, bg, paint, rect, fuse=True):
    """Construction path: explicit footprints and bridges on a latent board."""
    field = _board_canvas(tokens, n, gh, gw, bg)
    selected = _selection(rect, n)
    for r, c in selected:
        y, x = r * (gh + 2), c * (gw + 2)
        block = field[y:y + gh + 1, x:x + gw + 1]
        block[block == bg] = paint
    if fuse:
        for r in range(n):
            if {(r, c) for c in range(n)} <= selected:
                y = r * (gh + 2)
                band = field[y:y + gh + 1]
                band[band == bg] = paint
        for c in range(n):
            if {(r, c) for r in range(n)} <= selected:
                x = c * (gw + 2)
                band = field[:, x:x + gw + 1]
                band[band == bg] = paint
    return field


def _rects(start, delta):
    return [tuple(a + i * b for a, b in zip(start, delta)) for i in range(4)]


def _inside(rect, n):
    r, c, h, w = rect
    return min(h, w) > 0 and min(r, c) >= 0 and r + h <= n and c + w <= n


def _sample_rectangles(rng, n, kind, capacity, allow_clip=False, horizontal_only=False):
    if kind == "row":
        height = int(rng.integers(1, 3))
        return _rects((int(rng.integers(n - height + 1)), 0, height, 1), (0, 0, 0, 1))
    if kind == "col":
        return _rects((0, int(rng.integers(2)), 1, int(rng.integers(1, 3))), (0, 0, 1, 0))
    if kind == "drift":
        down = bool(rng.integers(2))
        return _rects((0 if down else n - 1, 0, 1, 2), (1 if down else -1, 0, 0, 0))
    if kind == "boundary":
        return _rects((0, 0, 2, int(rng.integers(1, 3))), (1, 0, 0, 0))
    for _ in range(400):
        start = (int(rng.integers(n)), int(rng.integers(n)),
                 int(rng.integers(1, n + 1)), int(rng.integers(1, n + 1)))
        if kind == "size":
            delta = (0, 0, int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
        elif kind == "move":
            delta = (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)), 0, 0)
        else:
            delta = tuple(map(int, rng.integers(-1, 2, 4)))
        motion, growth = any(delta[:2]), any(delta[2:])
        if kind == "size" and not growth or kind == "move" and not motion:
            continue
        if kind == "combined" and not (motion and growth):
            continue
        if horizontal_only and delta[2] != 0:
            continue
        rects = _rects(start, delta)
        if not all(_inside(r, n) and r[2] * r[3] <= capacity for r in rects[:3]):
            continue
        if not allow_clip and not _inside(rects[3], n):
            continue
        if not _selection(rects[3], n):
            continue
        if horizontal_only and _lines(_selection(rects[3], n), n)[1]:
            continue
        if len(set.union(*[_selection(r, n) for r in rects[:3]])) > n * n - 2:
            continue
        return rects
    # Constructive, nonrandom fallback meets the same lesson/shift requirements.
    if kind == "size":
        return _rects((0, 0, 1, 3), (0, 0, 0, -1))
    if kind == "move":
        return _rects((0, 0, 1, 2), (1, 0, 0, 0))
    return _rects((0, 0, 1, 1), (1, 0, 0, 1))


def _choose_crop(rng, field, last, unfused, n, gh, gw, max_width, kind, selected, bg, paint):
    hb = len(field)
    ink = (field != bg) & (field != paint)
    selected_ink = np.zeros(field.shape, bool)
    for r, c in selected:
        y, x = r * (gh + 2), c * (gw + 2)
        selected_ink[y:y + gh, x:x + gw] = ink[y:y + gh, x:x + gw]
    for attempt in range(160):
        if attempt < 120:
            r0 = int(rng.integers(hb - 1))
            r1 = int(rng.integers(r0 + 1, hb))
            c0 = int(rng.integers(max_width - 1))
            c1 = int(rng.integers(c0 + 1, max_width))
        else:
            # Whole available projection always witnesses a relevant difference.
            r0, r1, c0, c1 = 0, hb - 1, 0, max_width - 1
        if kind == "boundary":
            # Show both the surviving edge and the opposite edge that would
            # incorrectly be painted under a wraparound interpretation.
            r0, r1 = 0, hb - 1
            if c0 > gw:
                continue
        if r1 - r0 + c1 - c0 < 4:
            continue
        crop = field[r0:r1 + 1, c0:c1 + 1]
        if len(np.unique(crop)) < 2 or np.array_equal(crop, last[r0:r1 + 1, c0:c1 + 1]):
            continue
        if not np.any(ink[r0:r1 + 1, c0:c1 + 1]):
            continue
        if kind in ("row", "col") and np.array_equal(crop, unfused[r0:r1 + 1, c0:c1 + 1]):
            continue
        if kind in ("row", "col", "drift", "boundary") and not np.any(selected_ink[r0:r1 + 1, c0:c1 + 1]):
            continue
        if kind == "drift":
            r = next(iter(selected))[0]
            # Include the unpainted gap between two adjacent selected slots.
            if not (r0 <= r * (gh + 2) + gh <= r1 and c0 <= gw and gw + 1 <= c1):
                continue
            if not np.any((ink & ~selected_ink)[r0:r1 + 1, c0:c1 + 1]):
                continue
        return r0, r1, c0, c1
    return None


def _make_pair(rng, roles, kind, code, allow_clip=False, horizontal_only=False, constructive=False):
    bg, paint, marker, pool = roles
    n = 4 if constructive or kind in ("row", "col", "boundary") else int(rng.integers(4, 6))
    gh, gw = 2, 2 if constructive else int(rng.integers(2, 4))
    hb, wb = n * (gh + 2) - 1, n * (gw + 2) - 1
    # At least five future columns are retained, with a glyph and first gap.
    max_width = min(wb, 29 - wb) if constructive else int(rng.integers(5, min(wb, 29 - wb) + 1))
    width = wb + 1 + max_width
    capacity = (width + 1) // (gw + 1)
    if constructive:
        recipes = {
            "row": ((0, 0, 2, 1), (0, 0, 0, 1)),
            "col": ((0, 0, 1, 2), (0, 0, 1, 0)),
            "drift": ((0, 0, 1, 2), (1, 0, 0, 0)),
            "boundary": ((0, 0, 2, 2), (1, 0, 0, 0)),
            "size": ((0, 0, 1, 3), (0, 0, 0, -1)),
            "move": ((0, 0, 1, 2), (1, 0, 0, 0)),
            "combined": ((0, 0, 1, 1), (1, 0, 0, 1)),
        }
        rects = _rects(*recipes[kind])
    else:
        rects = _sample_rectangles(rng, n, kind, capacity, allow_clip, horizontal_only)
    snapshots = [_selection(rect, n) for rect in rects[:3]]
    min_inks = 3 if n == 5 and gw == 2 else 2
    ink_count = int(rng.integers(min_inks, min(6, len(pool)) + 1))
    inks = list(map(int, rng.choice(pool, ink_count, replace=False)))
    tokens = _make_tokens(rng, n, gh, gw, bg, inks, snapshots)
    field = _render_future(tokens, n, gh, gw, bg, paint, rects[3])
    last = _render_future(tokens, n, gh, gw, bg, paint, rects[2])
    unfused = _render_future(tokens, n, gh, gw, bg, paint, rects[3], False)
    selected = _selection(rects[3], n)
    crop = _choose_crop(rng, field, last, unfused, n, gh, gw, max_width, kind, selected, bg, paint)
    if crop is None:
        return None
    a = np.full((hb + 1 + 3 * (gh + 1), width), bg, np.int8)
    a[:hb, :wb] = _board_canvas(tokens, n, gh, gw, bg)
    a[:hb, wb] = paint
    for i in range(4):
        a[hb + i * (gh + 1)] = paint
    for i, positions in enumerate(snapshots):
        order = sorted(positions)
        rng.shuffle(order)
        slots = sorted(map(int, rng.choice(capacity, len(order), replace=False)))
        for q, pos in zip(slots, order):
            y, x = hb + 1 + i * (gh + 1), q * (gw + 1)
            a[y:y + gh, x:x + gw] = tokens[pos]
    r0, r1, c0, c1 = crop
    answer = field[r0:r1 + 1, c0:c1 + 1].copy()
    mask = np.zeros(a.shape, bool)
    mask[r0:r1 + 1, wb + 1 + c0:wb + 2 + c1] = True
    a, mask, answer = _orient(a, code), _orient(mask, code), _orient(answer, code)
    cells = np.argwhere(mask)
    lo, hi = cells.min(0), cells.max(0)
    bounds = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
    for y, x in _corners(bounds):
        a[y, x] = marker
    return {"input": a, "output": answer}


def _fallback_candidates(rng, roles, kind, code):
    """32 distinct valid encodings of one constructive member of this rule.

    There can be at most 13 prior inputs, so this finite pool cannot be exhausted.
    Only query order/blank positions vary here; ordinary examples use fresh scenes.
    """
    pair = _make_pair(rng, roles, kind, code, constructive=True)
    assert pair is not None
    canonical = _undo(pair["input"], code)
    bg = roles[0]
    keys = []
    for j in range(2):
        strip = canonical[16 + 3 * j:18 + 3 * j]
        keys.append([strip[:, 3 * q:3 * q + 2].copy() for q in range(10)
                     if np.any(strip[:, 3 * q:3 * q + 2] != bg)])
    assert 1 <= len(keys[0]) <= 4 and 1 <= len(keys[1]) <= 6
    for variant in range(32):
        a = canonical.copy()
        for j, order in enumerate(keys):
            y = 16 + 3 * j
            a[y:y + 2] = bg
            first = variant % 7 if j == 0 else variant // 7
            slots = [first] + list(range(11 - len(order), 10))
            for q, token in zip(slots, order):
                a[y:y + 2, 3 * q:3 * q + 2] = token
        yield {"input": _orient(a, code), "output": pair["output"].copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    roles = colors[0], colors[1], colors[2], colors[3:]
    code = int(rng.integers(8))
    mixed = nt >= 4 and rng.random() < .4
    schedule = ["row", "drift"]
    if nt >= 3:
        schedule.append("col")
    if nt >= 4:
        schedule.append("boundary")
    while len(schedule) < nt:
        schedule.append(str(rng.choice(["size", "move"])))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, kinds in (("train", schedule), ("test", ["combined"] * ne)):
        for kind in kinds:
            pair = None
            for _ in range(120):
                orientation = int(rng.integers(8)) if mixed else code
                candidate = _make_pair(rng, roles, kind, orientation,
                                       allow_clip=nt >= 4 and kind == "combined",
                                       horizontal_only=nt == 2 and kind == "combined")
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                pair = candidate
                break
            if pair is None:
                pair = next(p for p in _fallback_candidates(rng, roles, kind, code)
                            if _key(p["input"]) not in seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    # Examples retain chronology internally, but lesson order is not a signal.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(ne)]
    return puzzle


def _reduced_key_evidence(info):
    union = set.union(*info["snapshots"])
    queried = [info["board"][p] for p in union]
    others = [v for p, v in info["board"].items() if p not in union]
    shape = lambda t: _key(_trim(t != info["bg"], False))
    ink = lambda t: int(t[t != info["bg"]][0])
    shape_alias = any(shape(a) == shape(b) and ink(a) != ink(b) for a in queried for b in others)
    color_alias = any(ink(a) == ink(b) and shape(a) != shape(b) for a in queried for b in others)
    return shape_alias and color_alias


def _negative_gap(info, output):
    bg, paint, gh, gw = (info[k] for k in ("bg", "paint", "gh", "gw"))
    r0, r1, c0, c1 = info["local_bounds"]
    for r, c in info["selected"]:
        y, x = r * (gh + 2) + gh, c * (gw + 2) + gw + 1
        if ((r, c + 1) in info["selected"] and r0 <= y <= r1 and c0 <= x <= c1
                and output[y - r0, x - c0] == bg
                and x - 1 >= c0 and output[y - r0, x - 1 - c0] == paint):
            return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos, conventions = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            predicted, info = _analyze(inp)
            assert np.array_equal(predicted, out), "wrong future label"
            conventions.add((info["bg"], info["paint"], info["marker"]))
            assert info["marker"] not in np.unique(out)
            assert _reduced_key_evidence(info), "shape/color aliases not exposed"
            canonical_out = _orient(out, info["code"])
            frozen = _crop(_paint(info, info["snapshots"][-1]), info)
            assert not np.array_equal(frozen, canonical_out), "A extrapolation can be omitted"
            info["fusion_relevant"] = not np.array_equal(
                canonical_out, _crop(_paint(info, info["selected"], False), info))
            info["negative_gap"] = _negative_gap(info, canonical_out)
            r, c, h, w = info["future"]
            wrapped = {(rr % info["n"], cc % info["n"])
                       for rr in range(r, r + max(0, h)) for cc in range(c, c + max(0, w))}
            info["boundary_relevant"] = not np.array_equal(
                canonical_out, _crop(_paint(info, wrapped), info))
            # Ink survives even in selected slots, and every unselected pixel
            # outside a completed band retains its dictionary value.
            src = _crop(info["board_field"], info)
            assert np.array_equal(src[src != info["bg"]], canonical_out[src != info["bg"]])
            motion, growth = any(info["delta"][:2]), any(info["delta"][2:])
            assert (motion and growth) == (split == "test"), "missing combination shift"
            if split == "test" and nt == 2:
                assert info["delta"][2] == 0 and not _lines(info["selected"], info["n"])[1]
            if split == "test" and nt < 4:
                assert _inside(info["future"], info["n"]), "boundary case lacks lesson"
            infos[split].append(info)
    assert len(conventions) == 1, "role conventions vary within puzzle"
    train = infos["train"]
    assert any(i["fusion_relevant"] and _lines(i["selected"], i["n"])[0] for i in train)
    assert any(i["negative_gap"] and not any(_lines(i["selected"], i["n"])) for i in train)
    assert any(any(i["delta"][2:]) for i in train) and any(any(i["delta"][:2]) for i in train)
    if nt >= 3:
        assert any(i["fusion_relevant"] and _lines(i["selected"], i["n"])[1] for i in train)
    if nt >= 4:
        assert any(not _inside(i["future"], i["n"]) and i["boundary_relevant"] for i in train), "missing clipping lesson"


def self_test():
    """Bounded structural and semantic checks over 42 seeds and all counts."""
    signatures, sizes, deltas, palettes, selections, crop_sizes = set(), set(), set(), set(), set(), set()
    positive = negative = clipped = combined = disconnected = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        signature = tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s])
        assert signature == tuple(_key(pair["input"]) for s in ("train", "test") for pair in q[s])
        signatures.add(signature)
        for s in ("train", "test"):
            for pair in p[s]:
                _, i = _analyze(pair["input"])
                sizes.add((i["n"], i["gh"], i["gw"]))
                deltas.add(i["delta"])
                palettes.add((i["bg"], i["paint"], i["marker"]))
                selections.add((i["n"], tuple(sorted(i["selected"]))))
                crop_sizes.add(pair["output"].shape)
                rows, cols = _lines(i["selected"], i["n"])
                positive += bool(rows or cols)
                negative += not (rows or cols)
                clipped += not _inside(i["future"], i["n"])
                combined += any(i["delta"][:2]) and any(i["delta"][2:])
                disconnected += any(np.array_equal(t != i["bg"], np.eye(2, dtype=bool))
                                    for t in i["board"].values() if t.shape == (2, 2))
    assert len(signatures) == 42 and len(palettes) >= 30
    assert sizes == {(4, 2, 2), (4, 2, 3), (5, 2, 2), (5, 2, 3)}
    assert len(deltas) >= 12 and len(selections) >= 35 and len(crop_sizes) >= 30
    assert min(positive, negative, clipped, combined) >= 42 and disconnected >= 10
    roles = (0, 8, 9, [1, 2, 3, 4, 5, 6, 7])
    for kind in ("row", "col", "drift", "boundary", "size", "move", "combined"):
        pool = list(_fallback_candidates(np.random.default_rng(812), roles, kind, 3))
        assert len({_key(p["input"]) for p in pool}) == 32
        assert all(np.array_equal(solve(p["input"]), p["output"]) for p in pool)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Scene equivariance with canonical markers redrawn in the displayed frame.
    baseline = generate_puzzle(81)
    for code in range(8):
        transformed = {"train": [], "test": []}
        for s in transformed:
            for pair in baseline[s]:
                _, i = _analyze(pair["input"])
                clean = pair["input"].copy()
                clean[clean == i["marker"]] = i["bg"]
                r0, r1, c0, c1 = i["bounds"]
                mask = np.zeros(clean.shape, bool)
                mask[r0:r1 + 1, c0:c1 + 1] = True
                a, mask = _orient(clean, code), _orient(mask, code)
                lo, hi = np.argwhere(mask).min(0), np.argwhere(mask).max(0)
                bb = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
                for r, c in _corners(bb):
                    a[r, c] = i["marker"]
                transformed[s].append({"input": a, "output": _orient(pair["output"], code)})
        validate_generated_puzzle(transformed)
    # Fixed small semantic fixtures: clipping is intersection, never wrapping;
    # a nonpositive future extent vanishes; extras retain complete-line fusion.
    assert _selection((-1, 1, 3, 2), 4) == {(r, c) for r in (0, 1) for c in (1, 2)}
    assert _selection((1, 1, 0, 2), 4) == set()
    assert _future([(0, 0, 3, 1), (0, 0, 2, 1), (0, 0, 1, 1)])[0] == (0, 0, 0, 1)
    assert _lines({(r, c) for r in (0, 1) for c in range(4)}, 4) == ({0, 1}, set())
    # End-to-end fixtures outside the sampled templates. Expected painted
    # regions are specified directly, without using solve or _render_future.
    for n, gh, gw, rects, painted in (
        (4, 2, 2, _rects((0, 0, 1, 3), (0, 0, 0, -1)), []),
        (4, 2, 3, _rects((0, 0, 2, 2), (1, 0, 0, 0)),
         [(12, 15, 0, 4), (12, 15, 5, 9)]),
        (3, 3, 3, _rects((0, 0, 1, 1), (0, 0, 0, 1)), [(0, 4, 0, 14)]),
        (4, 2, 2, _rects((2, 0, 2, 2), (-1, 0, 0, 0)), [(0, 3, 0, 3), (0, 3, 4, 7)]),
    ):
        # All observed rectangles fit; the last fixture extrapolates a
        # negative top coordinate and retains just its in-board final row.
        snapshots = [_selection(r, n) for r in rects[:3]]
        if not all(snapshots):
            raise AssertionError("empty fixture observation")
        tokens = _make_tokens(np.random.default_rng(710 + n + gh + gw), n, gh, gw, 0, [1, 2, 3], snapshots)
        board = _board_canvas(tokens, n, gh, gw, 0)
        hb, wb = board.shape
        target_width = min(wb, 29 - wb)
        a = np.zeros((hb + 1 + 3 * (gh + 1), wb + 1 + target_width), np.int8)
        a[:hb, :wb] = board
        a[:hb, wb] = 8
        for j in range(4):
            a[hb + j * (gh + 1)] = 8
        for j, positions in enumerate(snapshots):
            for q, pos in enumerate(sorted(positions, reverse=True)):
                y, x = hb + 1 + j * (gh + 1), q * (gw + 1)
                a[y:y + gh, x:x + gw] = tokens[pos]
        expected = board[:, :target_width].copy()
        for r0, r1, c0, c1 in painted:
            block = expected[r0:r1, c0:c1]
            block[block == 0] = 8
        bounds = 0, hb - 1, wb + 1, wb + target_width
        for r, c in _corners(bounds):
            a[r, c] = 9
        assert np.array_equal(solve(a), expected), "end-to-end boundary/extent fixture failed"
    for defect in ("label", "duplicate"):
        p = generate_puzzle(93)
        if defect == "label":
            a = p["test"][0]["output"]
            a[0, 0] = (int(a[0, 0]) + 1) % 10
        else:
            p["train"][1] = {k: v.copy() for k, v in p["train"][0].items()}
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("corruption accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, count boundaries, semantic invariants, and scene symmetries")
