"""Port-gated overlapping route windows. Only NumPy and the standard library.

Frames and their tabs are instructions for the enclosed picture, not picture
material. Port geometry and literal landmark identity jointly select each join.
The landmark coordinates determine its translation, including transverse drift.
"""
from itertools import combinations
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, tuple(map(int, a.ravel()))


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _bbox(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points)+1, max(c for r, c in points)+1)


def _glyph(cells):
    r0, c0, r1, c1 = _bbox(cells)
    return tuple((r-r0, c-c0, int(v)) for (r, c), v in sorted(cells.items()))


def _glyph_shape(glyph):
    return max(p[0] for p in glyph)+1, max(p[1] for p in glyph)+1


def _read_windows(grid):
    """Read visible enclosing rectangles; no size or placement catalog."""
    bg = int(grid[0, 0])
    _require(all(int(grid[r, c]) == bg for r, c in
                 ((0, 0), (0, -1), (-1, 0), (-1, -1))), "unclear exterior")
    boxes = [_bbox(group) for group in _components(grid != bg)]
    # The connected non-background border encloses any detached interior cells.
    boxes = [b for b in boxes if not any(b != q and q[0] <= b[0] and q[1] <= b[1]
             and b[2] <= q[2] and b[3] <= q[3] for q in boxes)]
    _require(2 <= len(boxes) <= 6, "need two to six windows")
    windows, axes, frames = [], [], set()
    for r0, c0, r1, c1 in sorted(boxes):
        _require(0 < r0 < r1 < grid.shape[0] and 0 < c0 < c1 < grid.shape[1],
                 "windows need an exterior margin")
        tile = grid[r0:r1, c0:c1]
        _require(min(tile.shape) >= 4, "window too small")
        frame = int(tile[0, 0])
        _require(frame != bg and all(int(tile[r, c]) == frame for r, c in
                 ((0, 0), (0, -1), (-1, 0), (-1, -1))), "invalid frame corners")
        border = np.concatenate((tile[0], tile[-1], tile[1:-1, 0], tile[1:-1, -1]))
        _require(not np.any(border == bg), "broken frame")
        inks = set(map(int, border)) - {frame}
        _require(len(inks) == 1, "each window has one tab/route ink")
        ink = next(iter(inks))
        ports = [tuple(map(int, np.flatnonzero(edge == ink))) for edge in
                 (tile[1:-1, 0], tile[1:-1, -1], tile[0, 1:-1], tile[-1, 1:-1])]
        horizontal = bool(ports[0] or ports[1]) and not (ports[2] or ports[3])
        vertical = bool(ports[2] or ports[3]) and not (ports[0] or ports[1])
        _require(horizontal != vertical, "opposite port sides must specify an axis")
        axis = "H" if horizontal else "V"
        payload = tile[1:-1, 1:-1].copy()
        _require(not np.any(payload == frame), "frame color is reserved")
        if axis == "V":
            payload = payload.T.copy()
            ports = ports[2:4]
        else:
            ports = ports[:2]
        landmarks = []
        for group in _components((payload != bg) & (payload != ink)):
            colors = {int(payload[p]) for p in group}
            _require(len(colors) >= 2, "landmarks must be multicolored")
            box = _bbox(group)
            landmarks.append(dict(glyph=_glyph({p: payload[p] for p in group}),
                                  anchor=box[:2], box=box))
        incoming = [g for g in landmarks if g["box"][1] == 0]
        outgoing = [g for g in landmarks if g["box"][3] == payload.shape[1]]
        _require(len(incoming) == bool(ports[0]) and len(outgoing) == bool(ports[1])
                 and len(landmarks) == len(incoming)+len(outgoing),
                 "every tab side needs its edge landmark")
        _require(bool(np.any(payload == ink)), "route ink must be visible inside")
        _require(len(_components(payload != bg)) == 1, "one connected route per window")
        windows.append(dict(payload=payload, ink=ink, ports=tuple(ports),
                            incoming=incoming[0] if incoming else None,
                            outgoing=outgoing[0] if outgoing else None, box=(r0, c0, r1, c1)))
        frames.add(frame)
        axes.append(axis)
    _require(len(set(axes)) == 1 and len(frames) == 1, "one axis and frame role per input")
    _require(len({w["payload"].shape[0] for w in windows}) == 1, "common transverse span")
    # No unframed material or touching windows is admitted.
    covered = np.zeros(grid.shape, dtype=bool)
    for w in windows:
        r0, c0, r1, c1 = w["box"]
        _require(not covered[max(0, r0-1):r1+1, max(0, c0-1):c1+1].any(),
                 "windows must be separated")
        covered[r0:r1, c0:c1] = True
    _require(bool(np.all(grid[~covered] == bg)), "unframed material")
    return windows, bg, axes[0], next(iter(frames))


def _compatible(a, apos, b, bpos):
    ar, ac = apos
    br, bc = bpos
    r0, c0 = max(ar, br), max(ac, bc)
    r1, c1 = min(ar+a.shape[0], br+b.shape[0]), min(ac+a.shape[1], bc+b.shape[1])
    if r0 >= r1 or c0 >= c1:
        return True
    return np.array_equal(a[r0-ar:r1-ar, c0-ac:c1-ac],
                          b[r0-br:r1-br, c0-bc:c1-bc])


def _join(a, b, use_ports=True, use_landmarks=True):
    if not a["outgoing"] or not b["incoming"]:
        return None
    if use_ports and a["ports"][1] != b["ports"][0]:
        return None
    ga, gb = a["outgoing"], b["incoming"]
    if use_landmarks and ga["glyph"] != gb["glyph"]:
        return None
    return (ga["anchor"][0]-gb["anchor"][0], ga["anchor"][1]-gb["anchor"][1])


def _arrangements(windows, *, use_ports=True, use_landmarks=True, limit=2):
    """Joint search. A repeated port or glyph is not itself an adjacency label."""
    starts = [i for i, w in enumerate(windows) if not w["ports"][0] and w["ports"][1]]
    ends = [i for i, w in enumerate(windows) if w["ports"][0] and not w["ports"][1]]
    if len(starts) != 1 or len(ends) != 1:
        return []
    results = []

    def visit(order, positions):
        if len(results) >= limit:
            return
        if len(order) == len(windows):
            if order[-1] == ends[0]:
                results.append((tuple(order), tuple(positions)))
            return
        a = windows[order[-1]]
        for j, b in enumerate(windows):
            if j in order or (j == ends[0] and len(order) != len(windows)-1):
                continue
            delta = _join(a, b, use_ports, use_landmarks)
            if delta is None or delta[1] <= 0:
                continue
            pos = positions[-1][0]+delta[0], positions[-1][1]+delta[1]
            # Pixel agreement is part of the full rule, but excluded from the
            # port-only ablation: that ablation deliberately ignores A.
            if use_landmarks and any(not _compatible(windows[i]["payload"], p,
                                                     b["payload"], pos)
                                     for i, p in zip(order, positions)):
                continue
            visit(order+[j], positions+[pos])

    visit([starts[0]], [(0, 0)])
    return results


def _cardinality_arrangements(windows):
    simplified = [dict(w, ports=tuple(tuple(range(len(p))) for p in w["ports"]))
                  for w in windows]
    return _arrangements(simplified)


def _render(windows, order, positions, bg, axis, *, butt=False):
    cells = {}
    cursor = 0
    for i, (dr, dc) in zip(order, positions):
        a = windows[i]["payload"]
        if butt:
            dr, dc = 0, cursor
            cursor += a.shape[1]
        for r, c in np.argwhere(a != bg):
            p, color = (int(r)+dr, int(c)+dc), int(a[r, c])
            _require(p not in cells or cells[p] == color, "foreground collision")
            cells[p] = color
    r0, c0, r1, c1 = _bbox(cells)
    result = np.full((r1-r0, c1-c0), bg, dtype=int)
    for (r, c), color in cells.items():
        result[r-r0, c-c0] = color
    return result if axis == "H" else result.T.copy()


def _decode(input_grid):
    windows, bg, axis, frame = _read_windows(_grid(input_grid))
    answers = _arrangements(windows)
    _require(len(answers) == 1, "the complete overlapping chain must be unique")
    order, positions = answers[0]
    result = _render(windows, order, positions, bg, axis)
    _require(max(result.shape) <= 30, "assembled picture exceeds ARC bounds")
    return result, windows, order, positions, bg, axis, frame


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete cropped route from the supplied pixels alone."""
    return _decode(input_grid)[0]


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _new_glyph(rng, colors, mask=None):
    if mask is None:
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        if h*w == 1:
            h = 2
        style = int(rng.integers(4))
        mask = {(r, c) for r in range(h) for c in range(w)
                if style == 0 or r in (0, h-1) or c == 0
                or (style == 1 and c == w-1)}
        if style == 3 and h == 3 and w == 3:
            mask = {(r, c) for r in range(h) for c in range(w) if r in (0, 2) or c in (0, 2)}
    points = sorted(mask)
    labels = list(map(int, rng.choice(colors, len(points))))
    if len(set(labels)) == 1:
        labels[-1] = _choose(rng, [x for x in colors if x != labels[0]])
    return tuple((r, c, v) for (r, c), v in zip(points, labels))


def _landmark_pool(rng, colors, number, same_shape=False):
    result = []
    mask = None
    for _ in range(200):
        g = _new_glyph(rng, colors, mask)
        if g not in result:
            result.append(g)
            if same_shape:
                mask = {(r, c) for r, c, v in g}
        if len(result) == number:
            return result
    raise RuntimeError("could not sample distinct landmarks")


def _make_payload(rng, h, w, incoming, outgoing, ink, bg):
    a = np.full((h, w), bg, dtype=int)
    anchors = []
    for glyph, side in ((incoming, "L"), (outgoing, "R")):
        if glyph is None:
            anchors.append(None)
            continue
        gh, gw = _glyph_shape(glyph)
        r0 = int(rng.integers(h-gh+1))
        c0 = 0 if side == "L" else w-gw
        for r, c, color in glyph:
            a[r+r0, c+c0] = color
        anchors.append((r0, c0))
    il = _glyph_shape(incoming)[1] if incoming else 0
    ol = _glyph_shape(outgoing)[1] if outgoing else 0
    lo, hi = il, w-ol-1
    spine = int(rng.integers(lo, hi+1))
    roots = []
    for g, pos, side in ((incoming, anchors[0], "L"), (outgoing, anchors[1], "R")):
        if g is None:
            continue
        edge = max(c for r, c, v in g) if side == "L" else 0
        r = _choose(rng, [r for r, c, v in g if c == edge])+pos[0]
        roots.append(r)
        if side == "L":
            a[r, lo:spine+1] = ink
        else:
            a[r, spine:hi+1] = ink
    a[min(roots):max(roots)+1, spine] = ink
    if incoming is None:
        a[roots[0], :spine+1] = ink
    if outgoing is None:
        a[roots[-1], spine:] = ink
    # Local route branches and thickness vary independently of either join key.
    for _ in range(int(rng.integers(1, 5))):
        source = _choose(rng, [tuple(map(int, p)) for p in np.argwhere(a == ink)])
        r, c = int(rng.integers(h)), int(rng.integers(lo, hi+1))
        sr, sc = source
        a[min(sr, r):max(sr, r)+1, sc] = ink
        a[r, min(sc, c):max(sc, c)+1] = ink
    if hi-lo >= 2 and h >= 4 and rng.random() < .35:
        # A little circuit, attached to the route, retains its enclosed hole.
        r0 = int(rng.integers(h-2))
        c0 = int(rng.integers(lo, hi-1))
        a[r0:r0+3, c0] = ink
        a[r0:r0+3, c0+2] = ink
        a[r0, c0:c0+3] = ink
        a[r0+2, c0:c0+3] = ink
        a[r0:r0+3, spine] = ink
    return a, anchors


def _framed(payload, incoming, outgoing, ink, frame, axis):
    h, w = payload.shape
    a = np.full((h+2, w+2), frame, dtype=int)
    a[1:-1, 1:-1] = payload
    for r in incoming:
        a[r+1, 0] = ink
    for r in outgoing:
        a[r+1, -1] = ink
    return a if axis == "H" else a.T.copy()


def _pack(rng, tiles, bg):
    """Unequal rectangles in shuffled, jittered slots; vacancies are allowed."""
    n = len(tiles)
    mh, mw = max(a.shape[0] for a in tiles), max(a.shape[1] for a in tiles)
    layouts = [(nr, nc) for nr in range(1, n+1) for nc in range(1, n+1)
               if n <= nr*nc <= n+2 and nr*mh+nr+1 <= 30 and nc*mw+nc+1 <= 30]
    _require(bool(layouts), "frames do not fit")
    nr, nc = _choose(rng, layouts)
    free_r, free_c = 30-(nr*mh+nr+1), 30-(nc*mw+nc+1)
    extra_r, extra_c = int(rng.integers(min(3, free_r)+1)), int(rng.integers(min(3, free_c)+1))
    h, w = nr*mh+nr+1+extra_r, nc*mw+nc+1+extra_c
    a = np.full((h, w), bg, dtype=int)
    slots = list(map(int, rng.choice(nr*nc, n, replace=False)))
    tiles = [tiles[i] for i in rng.permutation(n)]
    for tile, slot in zip(tiles, slots):
        rr, cc = divmod(slot, nc)
        r0 = 1+rr*(mh+1)+int(rng.integers(mh-tile.shape[0]+1))
        c0 = 1+cc*(mw+1)+int(rng.integers(mw-tile.shape[1]+1))
        a[r0:r0+tile.shape[0], c0:c0+tile.shape[1]] = tile
    return a


def _generate_pair(rng, mode, axis):
    for attempt in range(240):
        n = 4 if mode in ("A_witness", "B_witness") else int(rng.integers(5, 7) if mode == "test" else rng.integers(2, 6))
        h = int(rng.integers(4, 7))
        palette = list(map(int, rng.permutation(10)))
        bg, frame = palette[:2]
        k = int(rng.integers(3, 5))
        key_colors, inks = palette[2:2+k], palette[2+k:]
        signature_pool = [p for size in (1, 2, 3) for p in combinations(range(h), size)]
        if mode == "A_witness":
            glyphs = _landmark_pool(rng, key_colors, 1)
            profiles = [_choose(rng, [p for p in signature_pool if len(p) == 2 and p[1] == p[0]+1]),
                        _choose(rng, [p for p in signature_pool if len(p) == 2 and p[1] > p[0]+1])]
            profiles.append(_choose(rng, [p for p in signature_pool if len(p) == 2 and p not in profiles]))
            rng.shuffle(profiles)
            links = [(p, glyphs[0]) for p in profiles]
        elif mode == "B_witness":
            glyphs = _landmark_pool(rng, key_colors, n-1, same_shape=True)
            profile = _choose(rng, [p for p in signature_pool if len(p) == 1])
            links = [(profile, g) for g in glyphs]
        else:
            count = 3 if n >= 5 else 2
            glyphs = _landmark_pool(rng, key_colors, count)
            profiles = [signature_pool[i] for i in rng.choice(len(signature_pool), count, replace=False)]
            pool = [(p, g) for p in profiles for g in glyphs]
            links = [pool[i] for i in rng.choice(len(pool), n-1, replace=False)]
            # A repeated joint label is allowed as one forced straight-through
            # window. Unique full ordering is still checked from the input.
            if n >= 4 and rng.random() < .4:
                j = int(rng.integers(n-2))
                links[j+1] = links[j]
        windows, tiles, offsets = [], [], [(0, 0)]
        for i in range(n):
            left = None if i == 0 else links[i-1]
            right = None if i == n-1 else links[i]
            gl, gr = left[1] if left else None, right[1] if right else None
            kl, kr = _glyph_shape(gl)[1] if gl else 0, _glyph_shape(gr)[1] if gr else 0
            lower = max(5, kl+kr+1)
            w = int(rng.integers(lower, 10))
            # Alternating ink in the witnesses explicitly refutes color matching.
            ink = inks[i % len(inks)] if mode.endswith("witness") else _choose(rng, inks)
            payload, anchors = _make_payload(rng, h, w, gl, gr, ink, bg)
            ports = (left[0] if left else (), right[0] if right else ())
            windows.append(dict(payload=payload, ink=ink, ports=ports,
                                incoming=dict(glyph=gl, anchor=anchors[0]) if gl else None,
                                outgoing=dict(glyph=gr, anchor=anchors[1]) if gr else None))
            tiles.append(_framed(payload, *ports, ink, frame, axis))
            if i:
                delta = _join(windows[i-1], windows[i])
                offsets.append((offsets[-1][0]+delta[0], offsets[-1][1]+delta[1]))
        out = _render(windows, tuple(range(n)), offsets, bg, axis)
        if max(out.shape) > 30:
            continue
        if any(not _compatible(a["payload"], p, b["payload"], q)
               for i, (a, p) in enumerate(zip(windows, offsets))
               for b, q in list(zip(windows, offsets))[i+1:]):
            continue
        try:
            inp = _pack(rng, tiles, bg)
            decoded = _decode(inp)
        except ValueError:
            continue
        if not np.array_equal(out, decoded[0]):
            continue
        # These witness tests use returned pixels, not construction labels.
        if mode == "A_witness" and len(_arrangements(decoded[1], use_ports=False)) < 2:
            continue
        if mode == "A_witness" and not any(
                not np.array_equal(_render(decoded[1], *answer, bg, axis), out)
                for answer in _arrangements(decoded[1], use_ports=False)):
            continue
        if mode == "B_witness" and len(_arrangements(decoded[1], use_landmarks=False)) < 2:
            continue
        if mode.endswith("witness") and len({p[0] for p in decoded[3]}) < 2:
            continue
        if mode.endswith("witness") and decoded[2] == tuple(range(n)):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)), "integer seed required")
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    first_axis = _choose(rng, ("H", "V"))
    plans = [("A_witness", first_axis), ("B_witness", "V" if first_axis == "H" else "H")]
    plans += [("train", _choose(rng, ("H", "V"))) for _ in range(num_train-2)]
    rng.shuffle(plans)
    test_plans = [("test", _choose(rng, ("H", "V"))) for _ in range(num_test)]
    result, seen = {"train": [], "test": []}, set()
    for split, local_plans in (("train", plans), ("test", test_plans)):
        for mode, axis in local_plans:
            for _ in range(20):
                pair = _generate_pair(rng, mode, axis)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, teaching = set(), {"A": False, "B": False}
    train_axes, train_counts, test_counts = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            predicted, windows, order, positions, bg, axis, frame = _decode(inp)
            _require(np.array_equal(predicted, out), "wrong registered union or crop")
            _require(frame not in set(map(int, out.ravel())), "frames are not picture material")
            _require(set(map(int, out.ravel()))-{bg} ==
                     set().union(*(set(map(int, w["payload"].ravel()))-{bg} for w in windows)),
                     "payload colors must survive")
            _require(len(_components(out != bg)) == 1, "the assembled route must be connected")
            # Every literal payload cell is conserved at its derived location;
            # only repeated equal landmarks are represented once.
            normalized = out if axis == "H" else out.T
            cells = {(r+dr, c+dc) for i, (dr, dc) in zip(order, positions)
                     for r, c in np.argwhere(windows[i]["payload"] != bg)}
            r0, c0, _, _ = _bbox(cells)
            for i, (dr, dc) in zip(order, positions):
                a = windows[i]["payload"]
                for r, c in np.argwhere(a != bg):
                    _require(normalized[r+dr-r0, c+dc-c0] == a[r, c], "lost payload cell")
            _require(all(np.any(normalized[edge] != bg) for edge in (0, -1))
                     and all(np.any(normalized[:, edge] != bg) for edge in (0, -1)), "crop must be tight")
            if split == "train":
                train_axes.add(axis)
                train_counts.append(len(windows))
                alternatives = _arrangements(windows, use_ports=False)
                if len(alternatives) >= 2:
                    teaching["B"] |= any(not np.array_equal(_render(windows, *answer, bg, axis), out)
                                         for answer in alternatives)
                teaching["A"] |= len(_arrangements(windows, use_landmarks=False)) >= 2
                _require(not np.array_equal(_render(windows, order, positions, bg, axis, butt=True), out),
                         "abutting panels must be distinguishable from overlap")
            else:
                test_counts.append(len(windows))
    _require(all(teaching.values()), "demonstrate necessity of both gates")
    _require(train_axes == {"H", "V"}, "demonstrate both input-visible axes")
    _require(all(n >= 5 for n in test_counts) and max(train_counts) <= 5, "longer test routes")


def self_test():
    """Bounded, pixel-based family checks on 42 seeds and focused negatives."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        assert any(len(_cardinality_arrangements(_decode(p["input"])[1])) >= 2
                   for p in puzzle["train"]), "tab cardinality shortcut not distinguished"
    fingerprints = {tuple(_key(p[role]) for s in ("train", "test") for p in puzzle[s]
                          for role in ("input", "output")) for puzzle in puzzles}
    assert len(fingerprints) == 42
    records, masks, input_shapes, output_shapes = [], set(), set(), set()
    for puzzle in puzzles:
        for pair in puzzle["train"]+puzzle["test"]:
            out, windows, order, positions, bg, axis, frame = _decode(pair["input"])
            input_shapes.add(pair["input"].shape)
            output_shapes.add(out.shape)
            joins = [windows[i]["ports"][1] for i in order[:-1]]
            glyphs = [windows[i]["outgoing"]["glyph"] for i in order[:-1]]
            records.append((len(windows), axis, bg, frame, tuple(joins), tuple(glyphs), tuple(positions)))
            for w in windows:
                masks.add(_key((w["payload"] != bg).astype(int)))
    assert {r[0] for r in records} >= {2, 3, 4, 5, 6}
    assert {r[1] for r in records} == {"H", "V"}
    assert {r[2] for r in records} == set(range(10))
    assert {r[3] for r in records} == set(range(10))
    assert len(input_shapes) >= 30 and len(output_shapes) >= 30 and len(masks) >= 200
    assert any(len(set(r[4])) < len(r[4]) for r in records)
    assert any(len(set(r[5])) < len(r[5]) for r in records)
    assert any(len({p[0] for p in r[6]}) >= 3 for r in records)
    assert any(len({b[1]-a[1] for a, b in zip(r[6], r[6][1:])}) > 1 for r in records)
    assert any(len(p) > 1 and max(p)-min(p)+1 > len(p) for r in records for p in r[4])
    assert any(min(_glyph_shape(g)) == 1 for r in records for g in r[5])
    assert any(len(g) < np.prod(_glyph_shape(g)) for r in records for g in r[5])
    # Determinism, extreme legal counts, and invalid count types.
    assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
               for a, b in zip(puzzles[17][s], generate_puzzle(17)[s]) for k in ("input", "output"))
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+sum(counts), *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Whole-example transpose and arbitrary bijective palette permutations.
    p = puzzles[0]["train"][0]
    assert np.array_equal(solve(p["input"].T), p["output"].T)
    mapping = np.array([7, 4, 0, 9, 2, 5, 1, 8, 6, 3])
    assert np.array_equal(solve(mapping[p["input"]]), mapping[p["output"]])
    # Make both intermediates interchangeable under BOTH gates. The input-only
    # solver must reject the ambiguity rather than pick reading order.
    for p in puzzles[0]["train"]:
        out, windows, order, positions, bg, axis, frame = _decode(p["input"])
        if len(_arrangements(windows, use_ports=False)) >= 2:
            break
    profile = (0,)
    ordered = [windows[i] for i in order]
    tiles = [_framed(w["payload"], () if i == 0 else profile,
                     () if i == len(ordered)-1 else profile, w["ink"], frame, axis)
             for i, w in enumerate(ordered)]
    ambiguous = _pack(np.random.default_rng(19), tiles, bg)
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("interchangeable overlapping windows accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    out = bad["test"][0]["output"]
    bg = int(bad["test"][0]["input"][0, 0])
    point = tuple(np.argwhere(out != bg)[0])
    out[point] = bg
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("deleted route cell accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds")
