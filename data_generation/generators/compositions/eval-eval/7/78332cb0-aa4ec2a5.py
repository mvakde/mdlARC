"""Paint an inferred joined structure, then return paint to the shuffled cuts.

No parent code, examples, hidden palette map, or generation state is used by
solve. Colors 6 (cavity) and 8 (hollow body) are fixed family conventions.
"""
from __future__ import annotations

from itertools import permutations, product
import numpy as np

FILL = 6
HOLLOW = 8
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _regions(mask, neighbors=N4):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    regions = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, region = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    region.add(q)
                    stack.append(q)
        regions.append(region)
    return regions


def _holes(body, neighbors=N4):
    h, w = body.shape
    holes = np.zeros_like(body, dtype=bool)
    regions = []
    for cells in _regions(~body, neighbors):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            rr, cc = np.array(sorted(cells)).T
            holes[rr, cc] = True
            regions.append(cells)
    return holes, regions


def _unambiguous(body):
    a, b = body[:-1, :-1], body[:-1, 1:]
    c, d = body[1:, :-1], body[1:, 1:]
    return not bool(np.any((a == d) & (b == c) & (a != b)))


def _dilate(body):
    h, w = body.shape
    p = np.pad(body, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                                 for dr, dc in N8])


def _zones(joined, bg, strict=False):
    body = joined != bg
    components = _regions(body)
    labels = np.full(body.shape, -1, dtype=int)
    for i, cells in enumerate(components):
        rr, cc = np.array(sorted(cells)).T
        labels[rr, cc] = i
    holes, cavities = _holes(body)
    owners = set()
    for cells in cavities:
        adjacent = {int(labels[r + dr, c + dc]) for r, c in cells for dr, dc in N4
                    if labels[r + dr, c + dc] >= 0}
        _require(len(adjacent) == 1, "cavity must have one connected wall owner")
        owners.update(adjacent)
    hollow = np.isin(labels, sorted(owners)) if owners else np.zeros_like(body)
    outline = _dilate(body) & ~body & ~holes
    if strict:
        _require(_unambiguous(body), "diagonal-only contact or leak")
        _require({frozenset(c) for c in components}
                 == {frozenset(c) for c in _regions(body, N8)}, "ambiguous wall connectivity")
        h8, c8 = _holes(body, N8)
        _require(np.array_equal(holes, h8)
                 and {frozenset(c) for c in cavities} == {frozenset(c) for c in c8},
                 "ambiguous background connectivity")
        # A global cavity must also be a cavity of its one owner's wall alone.
        # This excludes disconnected enclosed islands/nested objects.
        for owner in owners:
            local_holes, _ = _holes(labels == owner)
            _require(not bool(np.any(local_holes & body)), "nested object in cavity")
    return dict(body=body, labels=labels, components=components, cavities=cavities,
                holes=holes, hollow=hollow, solid=body & ~hollow, outline=outline,
                far=~body & ~holes & ~outline)


def _paint(joined, bg, divider):
    z = _zones(joined, bg)
    out = joined.copy()
    out[z["hollow"]] = HOLLOW
    out[z["holes"]] = FILL
    out[z["outline"]] = divider
    return out


def _layout(panels, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    result = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=int)
    for slot, p in enumerate(panels):
        r, c = divmod(slot, nc)
        result[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = p
    return result


def _chains(signatures):
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in signatures):
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) >= 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = signatures[order[-1]][1]
        if outgoing:
            for i in remaining:
                if signatures[i][0] == outgoing:
                    visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _decode(input_grid):
    a = _grid(input_grid)
    _require(not bool(np.any((a == FILL) | (a == HOLLOW))), "reserved output colors in input")
    partitions = []
    for divider in map(int, np.unique(a)):
        rows = list(map(int, np.flatnonzero(np.all(a == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(a == divider, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [a.shape[0]], [-1] + cols + [a.shape[1]]
        hh = [y - x - 1 for x, y in zip(rr, rr[1:])]
        ww = [y - x - 1 for x, y in zip(cc, cc[1:])]
        if min(hh + ww) < 3 or len(set(hh)) != 1 or len(set(ww)) != 1:
            continue
        panels = [a[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == divider) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        if any(not np.any(p != bg) for p in panels):
            continue
        partitions.append(dict(panels=panels, layout=(len(hh), len(ww)), bg=bg, divider=divider))
    _require(len(partitions) == 1, "need one equal-panel separator partition")
    info = partitions[0]
    axes = []
    for axis in (1, 0):
        oriented = [p if axis == 1 else p.T for p in info["panels"]]
        if any(np.any(p[0] != info["bg"]) or np.any(p[-1] != info["bg"]) for p in oriented):
            continue
        signatures = [(tuple(map(int, np.flatnonzero(p[:, 0] != info["bg"]))),
                       tuple(map(int, np.flatnonzero(p[:, -1] != info["bg"])))) for p in oriented]
        chains = _chains(signatures)
        if chains:
            _require(len(chains) == 1, "ambiguous complete chain")
            axes.append((axis, chains[0], signatures))
    _require(len(axes) == 1, "need one chain axis")
    info["axis"], info["order"], info["signatures"] = axes[0]
    return info


def _joined(info, order=None):
    order = info["order"] if order is None else order
    return np.concatenate([info["panels"][i] for i in order], axis=info["axis"])


def _restore(painted, info, order=None):
    order = info["order"] if order is None else order
    pieces = np.split(painted, len(order), axis=info["axis"])
    slots = [None] * len(order)
    for index, p in zip(order, pieces):
        slots[index] = p
    return _layout(slots, info["layout"], info["divider"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the input: virtual joins determine paint, not output placement."""
    info = _decode(input_grid)
    return _restore(_paint(_joined(info), info["bg"], info["divider"]), info)


def _bands(h, k):
    sizes = [len(a) for a in np.array_split(np.arange(h - k - 1), k)]
    bands, row = [], 1
    for size in sizes:
        bands.append(tuple(range(row, row + size)))
        row += size + 1
    return bands


def _links(rng, n, bands, distinct=False):
    pool = list(product(*bands))
    rng.shuffle(pool)
    count = n - 1
    repeat = count >= 3 and (len(pool) < count or rng.random() < .35)
    unique = count - int(repeat)
    _require(len(pool) >= unique, "insufficient geometric signatures")
    links = pool[:unique]
    if repeat:
        at = int(rng.integers(unique))
        links.insert(at, links[at])
    if distinct:
        _require(len(set(links)) >= 2, "teaching chain needs different joins")
    _require(len(_chains(list(zip([()] + links, links + [()])))) == 1, "construction chain ambiguity")
    return links


def _rails(rng, h, w, incoming, outgoing, groups=None):
    mask = np.zeros((h, w), dtype=bool)
    port = incoming or outgoing
    groups = groups or [tuple(range(len(port)))]
    if not incoming:
        for r in outgoing:
            mask[r, 1:] = True
        for group in groups:
            rows = [outgoing[j] for j in group]
            mask[min(rows):max(rows) + 1, 1] = True
    elif not outgoing:
        for r in incoming:
            mask[r, :w - 1] = True
        for group in groups:
            rows = [incoming[j] for j in group]
            mask[min(rows):max(rows) + 1, w - 2] = True
    else:
        for left, right in zip(incoming, outgoing):
            hub = int(rng.integers(1, w - 1))
            mask[left, :hub + 1] = True
            mask[right, hub:] = True
            mask[min(left, right):max(left, right) + 1, hub] = True
    return mask


def _gap(mask, band, col):
    mask[min(band):max(band) + 1, col] = False


def _baffle(mask, upper, lower, col):
    above = np.flatnonzero(mask[min(upper):max(upper) + 1, col]) + min(upper)
    below = np.flatnonzero(mask[min(lower):max(lower) + 1, col]) + min(lower)
    if len(above) and len(below):
        mask[int(above.max()):int(below.min()) + 1, col] = True


def _mutate(rng, body, w):
    """Grow wall detail without introducing diagonal ambiguity or new holes."""
    old_holes, old_cavities = _holes(body)
    old_components = len(_regions(body))
    h, length = body.shape
    for _ in range(int(rng.integers(2, 24))):
        candidates = []
        for r, c in np.argwhere(~body):
            r, c = int(r), int(c)
            if r in (0, h - 1) or c % w in (0, w - 1):
                continue
            if any(0 <= r + dr < h and 0 <= c + dc < length and body[r + dr, c + dc]
                   for dr, dc in N4):
                candidates.append((r, c))
        if not candidates:
            break
        q = _choose(rng, candidates)
        trial = body.copy()
        trial[q] = True
        if not _unambiguous(trial) or len(_regions(trial)) != old_components:
            continue
        holes, cavities = _holes(trial)
        if len(cavities) == len(old_cavities) and not bool(np.any(holes & ~old_holes)):
            body = trial
    # Detached exterior dots or short lines create preserved bodies beside
    # hollow spanning structures, never islands inside their cavities.
    for _ in range(int(rng.integers(3))):
        holes, _ = _holes(body)
        free = ~body & ~_dilate(body) & ~holes
        free[0] = free[-1] = False
        for c in range(length):
            if c % w in (0, w - 1):
                free[:, c] = False
        points = list(map(tuple, np.argwhere(free).tolist()))
        if points:
            r, c = _choose(rng, points)
            body[r, c] = True
            if rng.random() < .5 and c + 1 < length and free[r, c + 1]:
                body[r, c + 1] = True
    return body


def _make_pair(rng, mode, axis, color_mode=None):
    if mode == "open_anchor":
        n, k = 3, 4
    elif mode == "closed_anchor":
        n, k = 4, 3
    else:
        n = int(rng.integers(5, 7)) if mode == "test" else int(rng.integers(2, 5))
        k = int(rng.integers(2, 5))
    minimum_height = {2: 7, 3: 9, 4: 13}[k]
    dimensions = [(h, w) for h in range(minimum_height, 15)
                  for w in range(4, min(10, 30 // n) + 1)
                  if any(n % nr == 0 and nr * (h + 1) - 1 <= 30
                         and (n // nr) * (w + 1) - 1 <= 30 for nr in range(1, n + 1))]
    h, w = _choose(rng, dimensions)
    bands = _bands(h, k)
    links = _links(rng, n, bands, distinct=mode.endswith("anchor"))
    groups = None
    if mode == "open_anchor" or (not mode.endswith("anchor") and k >= 3 and rng.random() < .35):
        groups = [(0, 1), tuple(range(2, k))]
    masks = [_rails(rng, h, w, () if i == 0 else links[i - 1],
                    () if i == n - 1 else links[i], groups) for i in range(n)]
    if mode == "open_anchor":
        # A one-cell-wide external passage; no fragment contains a local hole.
        i, c = int(rng.integers(n)), int(rng.integers(1, w - 1))
        _gap(masks[i], bands[0], c)
    elif mode == "closed_anchor":
        # Top bay open; lower bay divided into two seam-spanning rooms.
        i, c = int(rng.integers(n)), int(rng.integers(1, w - 1))
        _gap(masks[i], bands[0], c)
        i = int(rng.integers(1, n - 1))
        _baffle(masks[i], bands[1], bands[2], int(rng.integers(1, w - 1)))
    else:
        # Independently varied exterior vents and internal merges. The actual
        # topology, not the sampled style, determines the ground truth.
        for rail in range(k):
            if rng.random() < (.43 if rail in (0, k - 1) else .30):
                i, c = int(rng.integers(n)), int(rng.integers(1, w - 1))
                _gap(masks[i], bands[rail], c)
        for _ in range(int(rng.integers(3))):
            i, between = int(rng.integers(n)), int(rng.integers(k - 1))
            _baffle(masks[i], bands[between], bands[between + 1], int(rng.integers(1, w - 1)))
    body = _mutate(rng, np.concatenate(masks, axis=1), w)
    palette = [int(c) for c in rng.permutation(10) if c not in (FILL, HOLLOW)]
    bg, divider = palette[:2]
    available = palette[2:]
    color_mode = color_mode or _choose(rng, ("shared", "partial", "distinct"))
    if color_mode == "shared":
        colors = [available[0]] * n
    elif color_mode == "distinct":
        colors = available[:n]
    else:
        m = int(rng.integers(2, min(n, 4) + 1))
        labels = list(range(m)) + list(map(int, rng.integers(m, size=n - m)))
        rng.shuffle(labels)
        colors = [available[i] for i in labels]
    panels = [np.where(body[:, i * w:(i + 1) * w], colors[i], bg).astype(int) for i in range(n)]
    if axis == 0:
        panels = [p.T.copy() for p in panels]
    ph, pw = panels[0].shape
    layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
               and nr * (ph + 1) - 1 <= 30 and (n // nr) * (pw + 1) - 1 <= 30]
    layout = _choose(rng, layouts)
    order = list(map(int, rng.permutation(n)))
    if mode.endswith("anchor") and order == list(range(n)):
        order[0], order[-1] = order[-1], order[0]
    inp = _layout([panels[i] for i in order], layout, divider)
    # Labels use the known constructed geometry and inverse shuffle; solve
    # independently recovers those relations from the returned input.
    joined = np.concatenate(panels, axis=axis)
    painted = np.split(_paint(joined, bg, divider), n, axis=axis)
    out = _layout([painted[i] for i in order], layout, divider)
    return {"input": inp, "output": out}


def _fallback_pair(rng, mode, axis, seen):
    """Constructive completion if random attempts fail; at least 336 palettes.

    Only 13 previous inputs can exist. The fixed geometric construction is
    unique, unambiguous, and preserves the same teaching/count-shift roles.
    """
    n = 3 if mode == "open_anchor" else 4 if mode == "closed_anchor" else 5 if mode == "test" else 2
    k = 4 if mode == "open_anchor" else 3 if mode in ("closed_anchor", "test") else 2
    h, w = 14, 5
    bands = _bands(h, k)
    links = list(product(*bands))[:n - 1]
    groups = [(0, 1), (2, 3)] if mode == "open_anchor" else None
    masks = [_rails(rng, h, w, () if i == 0 else links[i - 1],
                    () if i == n - 1 else links[i], groups) for i in range(n)]
    if mode in ("open_anchor", "closed_anchor", "test"):
        # Cut the starting cap between its first two rails: the external
        # passage is visible, while boundary port sets remain unchanged.
        masks[0][(links[0][0] + links[0][1]) // 2, 1] = False
    if mode in ("closed_anchor", "test"):
        _baffle(masks[1], bands[1], bands[2], 2)
    order = list(range(n - 1, -1, -1))
    layout = (1, n) if axis == 1 else (n, 1)
    for bg, divider, color in permutations([c for c in range(10) if c not in (FILL, HOLLOW)], 3):
        panels = [np.where(mask, color, bg).astype(int) for mask in masks]
        if axis == 0:
            panels = [p.T.copy() for p in panels]
        inp = _layout([panels[i] for i in order], layout, divider)
        if _key(inp) in seen:
            continue
        painted = np.split(_paint(np.concatenate(panels, axis=axis), bg, divider), n, axis=axis)
        pair = {"input": inp, "output": _layout([painted[i] for i in order], layout, divider)}
        # Verify the fallback's substantive witnesses, not merely its schema.
        f = _features(pair)
        if mode == "open_anchor":
            _require(f["solid_bay"] and f["solid"] and f["hollow"]
                     and f["emergent"] and f["spanning"] and f["order_diff"], "open fallback witness")
        if mode == "closed_anchor":
            _require(f["holes"] >= 2 and f["spanning"] and f["emergent"]
                     and f["mixed_bays"] and f["whole_body_far"] and f["order_diff"], "closed fallback witness")
        return pair
    raise AssertionError("more previous inputs than legal puzzle capacity")


def _features(pair, strict=True):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    joined = _joined(info)
    z = _zones(joined, info["bg"], strict=strict)
    _require(inp.shape == out.shape, "output must retain source layout dimensions")
    expected = _restore(_paint(joined, info["bg"], info["divider"]), info)
    _require(np.array_equal(out, expected), "wrong wall, cavity, outline, or source slot")
    _require(np.all(out[inp == info["divider"]] == info["divider"]), "separator changed")
    oriented_holes = z["holes"] if info["axis"] == 1 else z["holes"].T
    oriented_hollow = z["hollow"] if info["axis"] == 1 else z["hollow"].T
    oriented_solid = z["solid"] if info["axis"] == 1 else z["solid"].T
    wall = z["body"] if info["axis"] == 1 else z["body"].T
    w = info["panels"][0].shape[info["axis"]]
    seams = [i * w for i in range(1, len(info["panels"]))]
    spanning = any(np.any(oriented_holes[:, c - 1] & oriented_holes[:, c]) for c in seams)
    local = _layout([_paint(p, info["bg"], info["divider"]) for p in info["panels"]],
                    info["layout"], info["divider"])
    natural = tuple(range(len(info["panels"])))
    try:
        wrong_order = _restore(_paint(_joined(info, natural), info["bg"], info["divider"]), info, natural)
        order_diff = not np.array_equal(wrong_order, out)
    except AssertionError:
        # An invalid ordering can create collective/nested enclosure. Such an
        # ablation is already outside the rule and cannot reproduce the answer.
        order_diff = True
    hollow_bay, solid_bay = False, False
    for cells in z["components"]:
        rr, cc = np.array(sorted(cells)).T
        bbox = (slice(int(rr.min()), int(rr.max()) + 1), slice(int(cc.min()), int(cc.max()) + 1))
        if bool(np.any((~z["body"] & ~z["holes"])[bbox])):
            if z["hollow"][rr[0], cc[0]]:
                hollow_bay = True
            else:
                solid_bay = True
    # A stronger local/global witness is a hole in the join that is exterior
    # in its own isolated source fragment.
    local_holes = np.concatenate([_holes(p != info["bg"])[0] for p in
                                 (info["panels"][i] for i in info["order"])], axis=info["axis"])
    return dict(info=info, zones=z, n=len(info["panels"]), axis=info["axis"],
                panel_shape=info["panels"][0].shape, layout=info["layout"],
                signatures=tuple(info["signatures"]), bg=info["bg"], divider=info["divider"],
                holes=len(z["cavities"]), solid=bool(z["solid"].any()),
                hollow=bool(z["hollow"].any()), spanning=spanning,
                emergent=bool(np.any(z["holes"] & ~local_holes)),
                local_diff=not np.array_equal(local, out),
                order_diff=order_diff, mixed_bays=hollow_bay, solid_bay=solid_bay,
                whole_body_far=bool(np.any(z["hollow"] & ~_dilate(z["holes"]))),
                rail_count=max(len(a or b) for a, b in info["signatures"]),
                repeated=len({b for a, b in info["signatures"] if b}) < len(info["panels"]) - 1,
                mask_key=_key(wall), solid_pixels=int(oriented_solid.sum()),
                hollow_pixels=int(oriented_hollow.sum()))


def validate_generated_puzzle(puzzle) -> None:
    """Inspect returned grids and family evidence; use no construction metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            k = _key(_grid(pair["input"]))
            _require(k not in seen, "duplicate input")
            seen.add(k)
            records[split].append(_features(pair))
    tr, te = records["train"], records["test"]
    _require({r["axis"] for r in tr} == {0, 1}, "teach both virtual axes")
    _require(any(r["solid"] and r["solid_bay"] and r["hollow"] and r["spanning"]
                 and r["emergent"] and r["order_diff"] for r in tr), "missing open/sealed body contrast")
    _require(any(r["holes"] >= 2 and r["spanning"] and r["emergent"] and r["mixed_bays"]
                 and r["whole_body_far"]
                 for r in tr), "missing sealed rooms beside exterior bay")
    _require(any(r["local_diff"] for r in tr), "local-panel shortcut survives")
    _require(any(r["order_diff"] for r in tr), "reading-order shortcut survives")
    _require(max(r["n"] for r in tr) < min(r["n"] for r in te), "missing longer-chain test shift")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible puzzle, including two guaranteed teaching scenes."""
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first_axis = int(rng.integers(2))
    plans = [("open_anchor", first_axis, "shared"), ("closed_anchor", 1 - first_axis, None)]
    plans += [("train", int(rng.integers(2)), None) for _ in range(int(num_train) - 2)]
    rng.shuffle(plans)
    plans += [("test", int(rng.integers(2)), None) for _ in range(int(num_test))]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, (mode, axis, colors) in enumerate(plans):
        # Finite rejection checks real topology and the mandatory teaching
        # evidence, retaining the same rule under every count request.
        for _ in range(160):
            try:
                pair = _make_pair(rng, mode, axis, colors)
                f = _features(pair)
                if mode == "open_anchor" and not (f["solid"] and f["solid_bay"] and f["hollow"]
                                                    and f["spanning"] and f["emergent"] and f["order_diff"]):
                    continue
                if mode == "closed_anchor" and not (f["holes"] >= 2 and f["spanning"]
                                                      and f["emergent"] and f["mixed_bays"]
                                                      and f["whole_body_far"] and f["order_diff"]):
                    continue
                if _key(pair["input"]) not in seen:
                    break
            except AssertionError:
                continue
        else:
            pair = _fallback_pair(rng, mode, axis, seen)
        seen.add(_key(pair["input"]))
        puzzle["train" if index < num_train else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(p):
    return tuple(tuple((_key(q["input"]), _key(q["output"])) for q in p[s]) for s in ("train", "test"))


def _regressions():
    # Two U fragments: isolated pieces have no holes, the joined room has 18.
    rng = np.random.default_rng(100)
    a = _rails(rng, 7, 5, (), (1, 5))
    b = _rails(rng, 7, 5, (1, 5), ())
    panels = [np.where(m, 3, 0) for m in (b, a)]
    inp = _layout(panels, (1, 2), 2)
    info = _decode(inp)
    z = _zones(_joined(info), 0, strict=True)
    _require(z["holes"].sum() == 18 and z["solid"].sum() == 0, "emergent-room regression")
    _require(all(not _holes(p != 0)[0].any() for p in panels), "fragments must be open")
    closed = solve(inp)
    # Open a one-pixel doorway remote from the rightmost fragment.
    a[1, 3] = False
    opened_inp = _layout([np.where(b, 3, 0), np.where(a, 3, 0)], (1, 2), 2)
    open_info = _decode(opened_inp)
    oz = _zones(_joined(open_info), 0, strict=True)
    _require(not oz["holes"].any() and not oz["hollow"].any(), "vent must open the whole room")
    painted = _paint(_joined(open_info), 0, 2)
    _require(painted[1, 3] == 2 and painted[3, 6] == 0, "outline must not trigger cavity filling")
    _require(np.all(painted[oz["body"]] == 3), "remote solid body must retain original color")
    _require(not np.array_equal(closed, solve(opened_inp)), "vent must change answer")
    # Repeated signatures may be forced, but interchangeable excursions fail.
    s, t = (1, 5), (2, 5)
    _require(len(_chains([((), s), (s, s), (s, t), (t, ())])) == 1, "forced repeat")
    _require(len(_chains([((), s), (s, s), (s, s), (s, ())])) == 2, "ambiguous repeat")
    # A singleton at a diagonal exterior neighbor requires eight-neighbor outline.
    dot = np.zeros((5, 5), int)
    dot[2, 2] = 3
    _require(_paint(dot, 0, 2)[1, 1] == 2, "diagonal outline regression")


def self_test() -> dict:
    """Bounded tests over 42 seeds, geometric metamorphisms, and legal counts."""
    _regressions()
    # Exercise constructive completion, including multiple distinct palettes
    # for identical fallback geometry and the maximum requested pair count.
    fallback_puzzle, fallback_seen = {"train": [], "test": []}, set()
    fallback_rng = np.random.default_rng(914)
    fallback_plans = [("open_anchor", 1), ("closed_anchor", 0)] + [("train", i % 2) for i in range(8)]
    fallback_plans += [("test", i % 2) for i in range(4)]
    for i, (mode, axis) in enumerate(fallback_plans):
        pair = _fallback_pair(fallback_rng, mode, axis, fallback_seen)
        fallback_seen.add(_key(pair["input"]))
        fallback_puzzle["train" if i < 10 else "test"].append(pair)
    validate_generated_puzzle(fallback_puzzle)
    puzzles, records, shapes, masks, signatures = [], [], set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(seed)), "nondeterministic seed")
        puzzles.append(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair)
                records.append(f)
                shapes.add(pair["input"].shape)
                masks.add(f["mask_key"])
                signatures.update(s for a, b in f["signatures"] for s in (a, b) if s)
                inp = pair["input"].copy()
                _require(np.array_equal(solve(inp), pair["output"]), "input-only solve mismatch")
                _require(np.array_equal(inp, pair["input"]), "solver mutated input")
                _require(np.array_equal(solve(inp.T), pair["output"].T), "transpose equivariance")
        # Moving source panels permutes paint correspondingly, without changing
        # their virtual relations or the fixed fill/highlight colors.
        pair = p["train"][0]
        info = _decode(pair["input"])
        moved = _layout(info["panels"][::-1], info["layout"], info["divider"])
        expected_panels = []
        h, w = info["panels"][0].shape
        for slot in range(len(info["panels"])):
            r, c = divmod(slot, info["layout"][1])
            expected_panels.append(pair["output"][r * (h + 1):r * (h + 1) + h,
                                                  c * (w + 1):c * (w + 1) + w])
        expected = _layout(expected_panels[::-1], info["layout"], info["divider"])
        _require(np.array_equal(solve(moved), expected), "source-slot permutation equivariance")
        mapping = np.arange(10)
        available = [c for c in range(10) if c not in (FILL, HOLLOW)]
        mapping[available] = available[1:] + available[:1]
        _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "visible-role palette equivariance")
    _require(len(set(puzzles)) == 42, "repeated whole puzzles")
    _require(len(shapes) >= 25 and len(masks) >= 150 and len(signatures) >= 20, "insufficient geometric variation")
    _require({f["n"] for f in records} == set(range(2, 7)), "panel count range")
    _require(len({f["panel_shape"] for f in records}) >= 25, "panel shape variation")
    _require({f["axis"] for f in records} == {0, 1}, "axis variation")
    _require({f["rail_count"] for f in records} == {2, 3, 4}, "edge signature cardinality variation")
    _require((2, 2) in {f["layout"] for f in records}
             and bool({(2, 3), (3, 2)} & {f["layout"] for f in records}), "array layout variation")
    _require(any(f["repeated"] for f in records), "repeated port variation")
    _require(any(f["solid"] and f["hollow"] for f in records), "mixed wall-class variation")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7000 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(811, 2, 1)
    # Mutate one represented pixel from each semantic zone; every corruption
    # must fail validation even though other correct evidence is retained.
    for role in ("solid", "hollow", "holes", "outline", "far"):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        changed = False
        for pair in q["train"]:
            info = _decode(pair["input"])
            z = _zones(_joined(info), info["bg"])
            if z[role].any():
                painted = _paint(_joined(info), info["bg"], info["divider"])
                r, c = np.argwhere(z[role])[0]
                painted[r, c] = (int(painted[r, c]) + 1) % 10
                pair["output"] = _restore(painted, info)
                changed = True
                break
        _require(changed, "missing corruption-test zone")
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted output corruption: " + role)
    return {"seeds": 42, "pairs": len(records), "input_shapes": len(shapes),
            "panel_shapes": len({f["panel_shape"] for f in records}),
            "joined_wall_masks": len(masks), "port_signatures": len(signatures),
            "layouts": sorted({f["layout"] for f in records}),
            "repeated_join_pairs": sum(f["repeated"] for f in records),
            "seam_spanning_pairs": sum(f["spanning"] for f in records),
            "emergent_cavity_pairs": sum(f["emergent"] for f in records),
            "mixed_wall_class_pairs": sum(f["solid"] and f["hollow"] for f in records)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
