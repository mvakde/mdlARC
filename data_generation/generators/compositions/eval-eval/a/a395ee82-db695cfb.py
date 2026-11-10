"""Signal-routed glyph planting: an input-only ARC family.

The frame identifies endpoint color and the pitch-two board. The one connected
glyph identifies template color; its unique singleton on the board is the anchor.
All other nodes share one color. Only original nodes inside finite bridges launch
signals. Signals select existing nodes; they are never rasterized in the output.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _components(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _segment(a, b):
    if a[0] > b[0]:
        a, b = b, a
    d = b[0] - a[0]
    _require(d >= 2 and d == abs(b[1] - a[1]), "invalid primary bridge")
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + i, a[1] + slope * i) for i in range(d + 1))


def _slope(segment):
    return 1 if segment[-1][1] > segment[0][1] else -1


def _ray(point, slope, shape):
    r, c = point
    return {(a, c + slope * (a - r)) for a in range(shape[0])
            if 0 <= c + slope * (a - r) < shape[1]}


def _routing(endpoints, nodes, shape):
    endpoints, nodes = set(endpoints), set(nodes)
    bridges = []
    degrees = {p: 0 for p in endpoints}
    for a, b in combinations(sorted(endpoints), 2):
        if abs(a[0] - b[0]) == abs(a[1] - b[1]) and a[0] != b[0]:
            bridges.append(_segment(a, b))
            degrees[a] += 1
            degrees[b] += 1
    _require(bool(bridges) and max(degrees.values()) <= 1, "ambiguous endpoint pairing")
    launches = []
    for p in sorted(nodes):
        containing = [s for s in bridges if p in s[1:-1]]
        _require(len(containing) <= 1, "node on more than one bridge")
        if containing:
            launches.append((p, -_slope(containing[0])))
    beams = set().union(*(_ray(p, d, shape) for p, d in launches))
    selected = nodes & beams
    return {"bridges": tuple(bridges), "launches": tuple(launches),
            "beams": beams, "selected": selected,
            "bridge_sites": set().union(*(set(s) for s in bridges)),
            "unpaired": {p for p in endpoints if degrees[p] == 0}}


def _decode(input_grid):
    g = _grid(input_grid)
    values, counts = np.unique(g, return_counts=True)
    _require(len(values) == 4, "expected background, frame, glyph, and node colors")
    _require(np.count_nonzero(counts == counts.max()) == 1, "background not unique")
    bg = int(values[int(np.argmax(counts))])
    frames = []
    for color in values:
        color = int(color)
        if color == bg:
            continue
        coordinates = np.argwhere(g == color)
        top, left = map(int, coordinates.min(axis=0))
        bottom, right = map(int, coordinates.max(axis=0))
        h, w = bottom - top + 1, right - left + 1
        if (h >= 7 and w >= 7 and h % 2 == w % 2 == 1
                and np.all(g[top, left:right + 1] == color)
                and np.all(g[bottom, left:right + 1] == color)
                and np.all(g[top:bottom + 1, left] == color)
                and np.all(g[top:bottom + 1, right] == color)):
            frames.append((color, top, left, bottom, right))
    _require(len(frames) == 1, "expected one rectangular frame")
    frame, ft, fl, fb, fr = frames[0]
    board_shape = ((fb - ft) // 2, (fr - fl) // 2)

    glyph_candidates = []
    for color in values:
        color = int(color)
        if color in (bg, frame):
            continue
        groups = _components(tuple(map(int, p)) for p in np.argwhere(g == color))
        large = [p for p in groups if len(p) > 1]
        single = [next(iter(p)) for p in groups if len(p) == 1]
        if len(large) == len(single) == 1:
            coordinates = np.array(sorted(large[0]))
            tr, tc = map(int, coordinates.min(axis=0))
            br, bc = map(int, coordinates.max(axis=0))
            separated = br + 1 < ft or fb + 1 < tr or bc + 1 < fl or fr + 1 < tc
            if separated and ft < single[0][0] < fb and fl < single[0][1] < fr:
                glyph_candidates.append((color, tr, tc, br, bc, single[0]))
    _require(len(glyph_candidates) == 1, "glyph or anchor is ambiguous")
    ink, tr, tc, br, bc, anchor_pixel = glyph_candidates[0]
    marker = next(int(v) for v in values if v not in (bg, frame, ink))
    mask = g[tr:br + 1, tc:bc + 1] == ink
    h, w = mask.shape
    _require(h >= 2 and w >= 2, "glyph must be two-dimensional")
    _require(len(_components(map(tuple, np.argwhere(mask)))) == 1, "disconnected glyph")
    _require(np.count_nonzero(g[tr:br + 1, tc:bc + 1] != bg) == int(mask.sum()),
             "foreign color in glyph box")

    endpoints, nodes = set(), set()
    for r, c in np.argwhere(g != bg):
        r, c = int(r), int(c)
        v = int(g[r, c])
        if tr <= r <= br and tc <= c <= bc:
            continue
        if (r in (ft, fb) and fl <= c <= fr) or (c in (fl, fr) and ft <= r <= fb):
            _require(v == frame, "broken frame")
            continue
        _require(ft < r < fb and fl < c < fr, "foreground outside glyph and board")
        _require((r - ft - 1) % 2 == (c - fl - 1) % 2 == 0, "off-lattice point")
        p = ((r - ft - 1) // 2, (c - fl - 1) // 2)
        (endpoints if v == frame else nodes).add(p)
        _require(v in (frame, marker) or (r, c) == anchor_pixel, "extra glyph-colored node")
    anchor = ((anchor_pixel[0] - ft - 1) // 2, (anchor_pixel[1] - fl - 1) // 2)
    _require(anchor in nodes and bool(nodes - {anchor}), "missing ordinary nodes")
    top, left = tr - anchor[0] * h, tc - anchor[1] * w
    _require(0 <= top and top + board_shape[0] * h <= g.shape[0]
             and 0 <= left and left + board_shape[1] * w <= g.shape[1],
             "anchored board expansion does not fit")
    info = {"grid": g, "roles": (bg, frame, ink, marker), "frame": (ft, fl, fb, fr),
            "shape": board_shape, "mask": mask, "glyph_top": (tr, tc),
            "anchor": anchor, "top": top, "left": left,
            "endpoints": endpoints, "nodes": nodes}
    info.update(_routing(endpoints, nodes, board_shape))
    _require(bool(info["launches"]), "at least one transmitter is required")
    return info


def _stamp(info, selected=None, mask=None, invert=True):
    selected = info["selected"] if selected is None else set(selected)
    mask = info["mask"] if mask is None else mask
    h, w = info["mask"].shape
    bg, _, ink, marker = info["roles"]
    output = np.full_like(info["grid"], bg)
    rr, cc = np.nonzero(mask)
    # Coordinates use the source glyph's position, not a crop of selected sites.
    tr, tc = info["glyph_top"]
    for r, c in sorted(selected):
        y, x = tr + (r - info["anchor"][0]) * h, tc + (c - info["anchor"][1]) * w
        valid = (y + rr >= 0) & (y + rr < output.shape[0]) & (x + cc >= 0) & (x + cc < output.shape[1])
        _require(bool(np.all(valid)), "glyph copy would clip")
        color = marker if invert and (r, c) == info["anchor"] else ink
        if not invert:
            color = ink if (r, c) == info["anchor"] else marker
        output[y + rr, x + cc] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this grid alone; no training data or RNG is used."""
    return _stamp(_decode(input_grid))


def _mask(rng, h, w, informative=False):
    # A monotone spanning walk touches all four bounds and stays connected.
    result = np.zeros((h, w), dtype=bool)
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    r = c = 0
    result[r, c] = True
    for axis in steps:
        r += axis == 0
        c += axis == 1
        result[r, c] = True
    target = int(rng.integers(int(result.sum()), h * w + 1))
    while int(result.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(result):
            for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= a < h and 0 <= b < w and not result[a, b]:
                    frontier.add((int(a), int(b)))
        result[_pick(rng, sorted(frontier))] = True
    if rng.integers(2):
        result = np.fliplr(result).copy()
    if rng.integers(2):
        result = np.flipud(result).copy()
    if informative and not _asymmetric(result):
        result[:] = True
        result[_pick(rng, ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))] = False
    return result


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in
               (np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2)))


def _candidate_bridges(shape, slope):
    h, w = shape
    result = []
    for length in range(3, min(h, w) + 1):
        for r in range(h - length + 1):
            for c in range(w - length + 1):
                start_c = c if slope == 1 else c + length - 1
                result.append(tuple((r + i, start_c + slope * i) for i in range(length)))
    return result


def _geometry(rng, shape, count, witness):
    bridges, endpoints = [], set()
    parallel = int(rng.choice((-1, 1))) if rng.random() < .3 else None
    slopes = [witness.get("slope", parallel or int(rng.choice((-1, 1))))]
    slopes += [parallel or int(rng.choice((-1, 1))) for _ in range(count - 1)]
    for slope in slopes:
        choices = []
        for s in _candidate_bridges(shape, slope):
            ends = (s[0], s[-1])
            if not any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for p in ends for q in endpoints):
                choices.append(s)
        if not choices:
            return None
        chosen = _pick(rng, choices)
        bridges.append(chosen)
        endpoints.update((chosen[0], chosen[-1]))
    bridge_union = set().union(*(set(s) for s in bridges))
    sources = set()
    launches = []
    for i, s in enumerate(bridges):
        eligible = [p for p in s[1:-1] if sum(p in other for other in bridges) == 1]
        floor = (2 if witness.get("multiple") else 1) if i == 0 else 0
        cap = min(3, len(eligible))
        if floor > cap:
            return None
        n = 0 if i == 1 and witness.get("zero") else int(rng.integers(floor, cap + 1))
        if n:
            chosen = [eligible[int(j)] for j in rng.permutation(len(eligible))[:n]]
            sources.update(chosen)
            launches.extend((p, -_slope(s)) for p in chosen)
    if not sources:
        return None
    beams = set().union(*(_ray(p, d, shape) for p, d in launches))
    all_sites = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    receiver_pool = sorted(beams - bridge_union - endpoints)
    inert_pool = sorted(all_sites - beams - bridge_union - endpoints)
    if not receiver_pool or not inert_pool:
        return None
    receivers, inert = set(), set()
    if witness.get("echo"):
        echo_choices = []
        for p in receiver_pool:
            slopes_here = {d for q, d in launches if p in _ray(q, d, shape)}
            choices = set().union(*(_ray(p, -d, shape) for d in slopes_here)) & set(inert_pool)
            if choices:
                echo_choices.append((p, sorted(choices)))
        if not echo_choices:
            return None
        p, choices = _pick(rng, echo_choices)
        receivers.add(p)
        inert.add(_pick(rng, choices))
    if witness.get("beyond"):
        beyond = [p for p in inert_pool if any(
            p[1] - s[0][1] == _slope(s) * (p[0] - s[0][0]) for s in bridges)]
        if not beyond:
            return None
        inert.add(_pick(rng, beyond))
    nr = max(len(receivers), int(rng.integers(1 if witness else 0, min(5, len(receiver_pool)) + 1)))
    ni = max(len(inert), int(rng.integers(1 if witness else 0, min(4, len(inert_pool)) + 1)))
    for i in rng.permutation(len(receiver_pool)):
        if len(receivers) >= nr:
            break
        receivers.add(receiver_pool[int(i)])
    for i in rng.permutation(len(inert_pool)):
        if len(inert) >= ni:
            break
        inert.add(inert_pool[int(i)])
    nodes = sources | receivers | inert
    if len(nodes) < 2:
        nodes.add(_pick(rng, receiver_pool + inert_pool))
    # Optional primary distractors never acquire an accidental diagonal partner.
    extra = int(rng.choice((0, 0, 1, 2)))
    for i in rng.permutation(shape[0] * shape[1]):
        if extra == 0:
            break
        p = (int(i) // shape[1], int(i) % shape[1])
        if p not in nodes and not any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in endpoints):
            endpoints.add(p)
            extra -= 1
    return endpoints, nodes, beams & nodes


def _placement_options(dims, shape, anchor):
    h, w = dims
    rows, cols = shape
    ah, aw = rows * h, cols * w
    ph, pw = 2 * rows + 1, 2 * cols + 1
    tr, tc = anchor[0] * h, anchor[1] * w
    rr, cc = np.meshgrid(np.arange(-ph - 1, ah + 2), np.arange(-pw - 1, aw + 2), indexing="ij")
    minr, minc = np.minimum(0, rr), np.minimum(0, cc)
    maxr, maxc = np.maximum(ah, rr + ph), np.maximum(aw, cc + pw)
    valid = ((rr + ph < tr) | (tr + h < rr) | (cc + pw < tc) | (tc + w < cc))
    valid &= (maxr - minr <= 30) & (maxc - minc <= 30)
    positions = np.argwhere(valid)
    return [(int(rr[i, j]), int(cc[i, j]), int(minr[i, j]), int(minc[i, j]),
             int(maxr[i, j]), int(maxc[i, j])) for i, j in positions]


def _place(rng, mask, shape, anchor, endpoints, nodes, selected, colors, options):
    # Fit the union of the latent assembly and the visible framed board.
    areas = [(p[4] - p[2]) * (p[5] - p[3]) for p in options]
    if rng.random() < .65:
        cutoff = float(np.quantile(areas, .2))
        options = [p for p, area in zip(options, areas) if area <= cutoff]
    pr, pc, minr, minc, maxr, maxc = _pick(rng, options)
    base_h, base_w = maxr - minr, maxc - minc
    if rng.random() < .12:
        gh = gw = 30
    else:
        gh = int(rng.integers(base_h, min(30, base_h + 5) + 1))
        gw = int(rng.integers(base_w, min(30, base_w + 5) + 1))
    top, left = int(rng.integers(gh - base_h + 1)) - minr, int(rng.integers(gw - base_w + 1)) - minc
    bg, frame, ink, marker = colors
    g = np.full((gh, gw), bg, dtype=np.int8)
    ft, fl = top + pr, left + pc
    fb, fr = ft + 2 * shape[0], fl + 2 * shape[1]
    g[ft, fl:fr + 1] = g[fb, fl:fr + 1] = frame
    g[ft:fb + 1, fl] = g[ft:fb + 1, fr] = frame
    for r, c in sorted(endpoints):
        g[ft + 1 + 2 * r, fl + 1 + 2 * c] = frame
    for r, c in sorted(nodes):
        g[ft + 1 + 2 * r, fl + 1 + 2 * c] = ink if (r, c) == anchor else marker
    h, w = mask.shape
    tr, tc = top + anchor[0] * h, left + anchor[1] * w
    rr, cc = np.nonzero(mask)
    g[tr + rr, tc + cc] = ink
    info = {"grid": g, "mask": mask, "roles": colors, "glyph_top": (tr, tc),
            "anchor": anchor, "selected": selected}
    return {"input": g, "output": _stamp(info)}


def _shape_options(regime, split, direction):
    if regime == 0:  # Strict coarse-span shift.
        lo, hi = (4, 6) if split == "train" else (7, 10)
        return [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    if regime == 1:  # Aspect reversal without changing the routing rule.
        sign = direction if split == "train" else -direction
        return [(r, c) for r in range(4, 10) for c in range(4, 10) if int(np.sign(c - r)) == sign]
    return [(r, c) for r in range(5 if split == "test" else 4, 10)
            for c in range(5 if split == "test" else 4, 10)]


def _make_pair(rng, regime, split, direction, colors, witness, previous=None):
    shapes = _shape_options(regime, split, direction)
    count = int(rng.integers(3, 5)) if regime == 2 and split == "test" else int(rng.integers(1, 3))
    if witness.get("zero"):
        count = max(2, count)
    dimension_choices = [(h, w) for h in range(2, 6) for w in range(2, 6)
                         if (not witness.get("asymmetric") or h != w)
                         and (previous is None or h != previous["mask"].shape[0]
                              and w != previous["mask"].shape[1])
                         and any(h * r <= 28 and w * c <= 28 for r, c in shapes)]
    # A bounded search over freely composed geometry; no library of solved pairs.
    for _ in range(240):
        dims = _pick(rng, dimension_choices)
        legal_shapes = [s for s in shapes if dims[0] * s[0] <= 28 and dims[1] * s[1] <= 28]
        shape = _pick(rng, legal_shapes)
        geometry = _geometry(rng, shape, count, witness)
        if geometry is None:
            continue
        endpoints, nodes, selected = geometry
        state = witness.get("anchor_selected")
        anchors = sorted(nodes if state is None else selected if state else nodes - selected)
        if previous is not None:
            anchors = [p for p in anchors if p[0] != previous["anchor"][0] and p[1] != previous["anchor"][1]]
        rng.shuffle(anchors)
        for anchor in anchors:
            options = _placement_options(dims, shape, anchor)
            if options:
                mask = _mask(rng, *dims, informative=witness.get("asymmetric", False))
                return _place(rng, mask, shape, anchor, endpoints, nodes, selected, colors, options)
    raise RuntimeError("bounded geometry search exhausted")


def _check_count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _check_count("num_train", num_train, 2, 10), _check_count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime, direction = int(rng.integers(3)), int(rng.choice((-1, 1)))
    colors = tuple(map(int, rng.choice(10, 4, replace=False)))
    constant_palette = rng.random() < .25
    slopes = rng.permutation((-1, 1))
    witnesses = [{"slope": int(slopes[0]), "anchor_selected": True, "asymmetric": True},
                 {"slope": int(slopes[1]), "anchor_selected": False, "beyond": True}]
    witnesses[int(rng.integers(2))]["multiple"] = True
    witnesses[int(rng.integers(2))]["zero"] = True
    witnesses[int(rng.integers(2))]["echo"] = True
    witnesses += [{} for _ in range(nt - 2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = None
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = witnesses[i] if split == "train" else {}
            local_colors = colors if constant_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            # Structural/color role assignment is visible in every input.
            for _ in range(12):
                pair = _make_pair(rng, regime, split, direction, local_colors, witness,
                                  previous if split == "train" and i == 1 else None)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct-input search exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train" and i == 0:
                previous = _decode(pair["input"])
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(info):
    bridge_sources = [sum(p in s[1:-1] for p, _ in info["launches"]) for s in info["bridges"]]
    receivers = info["selected"] - info["bridge_sites"]
    inert = info["nodes"] - info["selected"]
    beyond = {p for p in inert if any(
        p[1] - s[0][1] == _slope(s) * (p[0] - s[0][0]) for s in info["bridges"])}
    echo = any(q in _ray(p, -d, info["shape"]) for p in receivers
               for source, d in info["launches"] if p in _ray(source, d, info["shape"]) for q in inert)
    return {"counts": tuple(bridge_sources), "receivers": receivers, "inert": inert,
            "beyond": beyond, "echo": echo,
            "active_slopes": {-d for _, d in info["launches"]}}


def _validate(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check_count("training count", nt, 2, 10)
    _check_count("test count", nq, 1, 4)
    _require(nt > nq, "training must outnumber tests")
    seen, groups = set(), {"train": [], "test": []}
    for split in groups:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(out, _stamp(info)), "incorrect planted assembly")
            population = int(info["mask"].sum())
            _require(np.count_nonzero(out != info["roles"][0]) == population * len(info["selected"]),
                     "glyph copies overlap, clip, or change their mask")
            expected_marker = population if info["anchor"] in info["selected"] else 0
            _require(np.count_nonzero(out == info["roles"][3]) == expected_marker, "wrong anchor inversion")
            _require(not np.any(out == info["roles"][1]), "frame or endpoint survived")
            _require(not np.array_equal(inp, out), "identity pair")
            # Analytic incidence checks the selection without rasterizing beams.
            analytic = {p for p in info["nodes"] if any(
                p[1] - q[1] == d * (p[0] - q[0]) for q, d in info["launches"])}
            _require(analytic == info["selected"], "wrong signal incidence")
            groups[split].append(info)
    train = groups["train"]
    facts = [_facts(i) for i in train]
    _require(set().union(*(f["active_slopes"] for f in facts)) == {-1, 1}, "missing active slope demonstration")
    _require(any(max(f["counts"]) >= 2 for f in facts), "missing multi-source demonstration")
    _require(any(0 in f["counts"] for f in facts), "missing untriggered bridge")
    _require(any(f["receivers"] for f in facts) and any(f["inert"] for f in facts), "selection not demonstrated")
    _require(any(f["beyond"] for f in facts), "finite span not demonstrated")
    _require(any(f["echo"] for f in facts), "nonrecursive reception not demonstrated")
    _require({i["anchor"] in i["selected"] for i in train} == {True, False}, "missing anchor states")
    _require(len({i["mask"].shape[0] for i in train}) >= 2 and len({i["mask"].shape[1] for i in train}) >= 2,
             "both glyph dimensions must vary in training")
    _require(len({i["anchor"][0] for i in train}) >= 2 and len({i["anchor"][1] for i in train}) >= 2,
             "anchor coordinates must vary")
    _require(any(i["mask"].shape[0] != i["mask"].shape[1] and _asymmetric(i["mask"]) for i in train),
             "missing exact-mask/aspect witness")
    # Check observable counterfactuals for both inherited contributions.
    alternatives = {
        "omit_B": lambda i: _stamp(i, i["nodes"]),
        "omit_A": lambda i: _stamp(i, mask=np.ones((1, 1), dtype=bool)),
        "no_inversion": lambda i: _stamp(i, invert=False),
        "solid_glyph": lambda i: _stamp(i, mask=np.ones_like(i["mask"])),
        "plant_empty_beam_sites": lambda i: _stamp(i, i["beams"] - i["endpoints"]),
        "only_transmitters": lambda i: _stamp(i, [p for p, _ in i["launches"]]),
    }
    for name, alternative in alternatives.items():
        _require(any(not np.array_equal(_stamp(i), alternative(i)) for i in train), f"undistinguished shortcut: {name}")
    # These are generator distribution checks, not new solving conventions.
    test = groups["test"]
    span_shift = all(max(i["shape"]) <= 6 for i in train) and all(min(i["shape"]) >= 7 for i in test)
    tr_signs = {int(np.sign(i["shape"][1] - i["shape"][0])) for i in train}
    te_signs = {int(np.sign(i["shape"][1] - i["shape"][0])) for i in test}
    aspect_shift = tr_signs in ({-1}, {1}) and te_signs == {-v for v in tr_signs}
    route_shift = all(len(i["bridges"]) <= 2 for i in train) and all(len(i["bridges"]) >= 3 for i in test)
    _require(span_shift or aspect_shift or route_shift, "missing generated train/test shift")
    return groups


def validate_generated_puzzle(puzzle) -> None:
    """Validate pixels, registration, routing, exact copies, and teaching evidence."""
    try:
        _validate(puzzle)
    except (ValueError, TypeError, KeyError, StopIteration) as exc:
        raise AssertionError(str(exc)) from exc


def _signature(puzzle):
    return tuple((split, _key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def _semantic_checks():
    """Hand-built family case with a cellwise expected answer, not RNG metadata."""
    g = np.zeros((30, 20), dtype=np.int8)
    # Six rows/seven columns on the board. Endpoint hue matches the frame.
    g[0, :15] = g[12, :15] = 1
    g[:13, 0] = g[:13, 14] = 1
    for r, c in ((1, 1), (4, 4)):
        g[1 + 2 * r, 1 + 2 * c] = 1
    for r, c in ((2, 2), (1, 3), (0, 4), (2, 4), (5, 5)):
        g[1 + 2 * r, 1 + 2 * c] = 3
    g[3, 7] = 2  # Selected anchor at (1,3).
    glyph_pixels = ((0, 0), (0, 1), (1, 0), (2, 0))
    for dr, dc in glyph_pixels:
        g[15 + dr, 10 + dc] = 2
    # Source (2,2) emits r+c=4. Only its two marked receivers join it.
    expected = np.zeros_like(g)
    for r, c, color in ((18, 8, 2), (15, 10, 3), (12, 12, 2)):
        for dr, dc in glyph_pixels:
            expected[r + dr, c + dc] = color
    assert np.array_equal(solve(g), expected)
    i = _decode(g)
    assert i["selected"] == {(2, 2), (1, 3), (0, 4)}
    assert (5, 5) in _facts(i)["beyond"] and _facts(i)["echo"]
    # The beyond-endpoint anchor registers copies without itself being selected.
    swapped = g.copy()
    swapped[3, 7], swapped[11, 11] = 3, 2
    expected_swapped = np.zeros_like(g)
    for r, c in ((6, 4), (3, 6), (0, 8)):
        for dr, dc in glyph_pixels:
            expected_swapped[r + dr, c + dc] = 2
    assert np.array_equal(solve(swapped), expected_swapped)
    assert not np.array_equal(expected, expected_swapped)
    # Same input relationships survive a larger blank border, with coordinates
    # translated rather than inferred from canvas center.
    translated = np.full((30, 22), 0, dtype=np.int8)
    translated[:, 2:] = g
    translated_expected = np.zeros_like(translated)
    translated_expected[:, 2:] = expected
    assert np.array_equal(solve(translated), translated_expected)


def self_test():
    """Bounded 42-seed coverage suite and targeted semantic counterfactuals."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    infos = [i for p in puzzles for group in _validate(p).values() for i in group]
    assert len({i["grid"].shape for i in infos}) >= 35
    assert len({i["roles"] for i in infos}) >= 25
    assert len({(i["mask"].shape, i["mask"].tobytes()) for i in infos}) >= 35
    assert len({i["shape"] for i in infos}) >= 15
    assert len({i["anchor"] for i in infos}) >= 20
    assert len({len(i["selected"]) for i in infos}) >= 7
    assert {2, 3, 4, 5} <= {i["mask"].shape[0] for i in infos}
    assert {2, 3, 4, 5} <= {i["mask"].shape[1] for i in infos}
    assert {1, 2, 3, 4} <= {len(i["bridges"]) for i in infos}
    assert any(i["unpaired"] for i in infos)
    assert any(all(i["mask"].flat) for i in infos)
    assert any(i["anchor"] not in i["selected"] for i in infos)
    assert any(i["anchor"] in _facts(i)["receivers"] for i in infos)
    _semantic_checks()
    for nt, nq in ((2, 1), (10, 4), (5, 3)):
        p = generate_puzzle(1700 + nt, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Color equivariance: all semantic roles are encoded by geometry.
    perm = np.random.default_rng(912).permutation(10).astype(np.int8)
    pair = puzzles[0]["train"][0]
    assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    # Moving the matching-colored anchor among existing nodes must
    # alter registration even when signal selection is unchanged.
    witnessed = False
    for info in infos:
        if info["anchor"] not in info["selected"]:
            for replacement in sorted(info["nodes"] - info["selected"] - {info["anchor"]}):
                h, w = info["mask"].shape
                tr, tc = info["glyph_top"]
                top, left = tr - replacement[0] * h, tc - replacement[1] * w
                if (0 <= top and top + info["shape"][0] * h <= info["grid"].shape[0]
                        and 0 <= left and left + info["shape"][1] * w <= info["grid"].shape[1]):
                    changed = info["grid"].copy()
                    ft, fl, _, _ = info["frame"]
                    a, b = info["anchor"], replacement
                    changed[ft + 1 + 2 * a[0], fl + 1 + 2 * a[1]] = info["roles"][3]
                    changed[ft + 1 + 2 * b[0], fl + 1 + 2 * b[1]] = info["roles"][2]
                    assert not np.array_equal(solve(changed), _stamp(info))
                    witnessed = True
                    break
        if witnessed:
            break
    assert witnessed, "inactive-anchor counterfactual not exercised"
    corrupt = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
