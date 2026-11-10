"""Solved ARC puzzles with halos centered on markers in latent rectangles."""

import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    ((20, 20), "6dd4e2480fc37766abe7d2ffe7ebe8d68955a6efd1ccde2a9685e1367f48746c"),
    ((20, 20), "c3d7f829ef614ce510191f847403e7a852f4bb870bf723debbc543a1df032bc5"),
    ((10, 17), "ac4e50e080cba1858fd327cd6938127863f4645333d145201c641079b606d024"),
    ((16, 27), "63f29cf14131d085d19b12fba5099109d743f7d9c499bdfed1833806717ade53"),
}


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _rand(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _inside(rect, p):
    return rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _key(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def _connected_carrier(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    visited = {next(iter(points))}
    stack = list(visited)
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in points and p not in visited:
                visited.add(p)
                stack.append(p)
    return len(visited) == len(points)


def _carrier_ok(rect, markers):
    r0, c0, r1, c1 = rect
    mask = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in markers:
        if _inside(rect, (r, c)):
            mask[r - r0, c - c0] = False
    return (mask[0].any() and mask[-1].any() and mask[:, 0].any()
            and mask[:, -1].any() and _connected_carrier(mask))


def _find_rectangles(grid, background, carrier, marker):
    _check(set(map(int, np.unique(grid))) == {background, carrier, marker},
           "input must show three distinct roles")
    h, w = grid.shape
    cells = set(map(tuple, np.argwhere(grid == carrier)))
    rectangles = []
    while cells:
        start = next(iter(cells))
        cells.remove(start)
        component = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in cells:
                    cells.remove(p)
                    component.add(p)
                    stack.append(p)
        rows, cols = zip(*component)
        rect = (min(rows), min(cols), max(rows), max(cols))
        r0, c0, r1, c1 = rect
        _check(r1 - r0 >= 3 and c1 - c0 >= 3, "carrier field is too small")
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        _check(np.all((sub == carrier) | (sub == marker)),
               "carrier field is not a filled rectangle")
        _check(int(np.count_nonzero(sub == carrier)) == len(component),
               "carrier component does not establish its complete footprint")
        _check(_carrier_ok(rect, set(map(tuple, np.argwhere(grid == marker)))),
               "marker removal obscures a rectangle")
        rectangles.append(rect)
    _check(1 <= len(rectangles) <= 4, "invalid field count")
    rectangles.sort()
    outside = np.ones((h, w), dtype=bool)
    for r0, c0, r1, c1 in rectangles:
        outside[r0:r1 + 1, c0:c1 + 1] = False
    _check(int(np.count_nonzero(outside & (grid == background))) >
           int(np.count_nonzero(outside & (grid == marker))),
           "background must dominate the space outside carrier fields")
    return rectangles


def _active(grid, rectangles, marker):
    return sorted((int(r), int(c)) for r, c in np.argwhere(grid == marker)
                  if any(_inside(rect, (int(r), int(c))) for rect in rectangles))


def _predict(grid, rectangles, marker, halo):
    out = grid.copy()
    active = _active(grid, rectangles, marker)
    h, w = grid.shape
    for r, c in active:
        out[max(0, r - 1):min(h, r + 2),
            max(0, c - 1):min(w, c + 2)] = halo
    for r, c in active:
        out[r, c] = marker
    return out


def _role_candidates(grid, halo):
    colors = set(map(int, np.unique(grid)))
    if len(colors) != 3 or halo in colors:
        return []
    candidates = []
    for background in colors:
        for carrier in colors - {background}:
            marker = next(iter(colors - {background, carrier}))
            try:
                rectangles = _find_rectangles(grid, background, carrier, marker)
            except AssertionError:
                continue
            candidates.append(((background, carrier, marker, halo), rectangles))
    return candidates


def _separated(a, b):
    # At least one entire background row or column separates carrier footprints.
    return (a[2] + 1 < b[0] or b[2] + 1 < a[0]
            or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _free_layout(rng, count, border):
    for _ in range(250):
        h = _rand(rng, 10 if count == 2 else 13, 30)
        w = _rand(rng, 11 if count == 2 else 14, 30)
        rects = []
        for i in range(count):
            placed = False
            for _ in range(250):
                rh = _rand(rng, 4, min(h - 1, 12 if count == 2 else 10))
                rw = _rand(rng, 4, min(w - 1, 13 if count == 2 else 10))
                r = _rand(rng, 0, h - rh)
                c = _rand(rng, 0, w - rw)
                if i == 0 and border:
                    side = _rand(rng, 0, 3)
                    if side == 0:
                        r = 0
                    elif side == 1:
                        r = h - rh
                    elif side == 2:
                        c = 0
                    else:
                        c = w - rw
                rect = (r, c, r + rh - 1, c + rw - 1)
                if all(_separated(rect, other) for other in rects):
                    rects.append(rect)
                    placed = True
                    break
            if not placed:
                break
        if len(rects) == count and sum((a[2]-a[0]+1)*(a[3]-a[1]+1)
                                       for a in rects) < .57*h*w:
            return h, w, sorted(rects)
    raise RuntimeError("could not place free rectangles")


def _gap_layout(rng, count, border):
    for _ in range(180):
        h, w = _rand(rng, 14, 30), _rand(rng, 16, 30)
        rh1, rh2 = _rand(rng, 4, min(10, h - 2)), _rand(rng, 4, min(10, h - 2))
        rw1 = _rand(rng, 4, min(9, w - 8))
        rw2 = _rand(rng, 4, min(9, w - rw1 - 2))
        horizontal = bool(_rand(rng, 0, 1))
        if horizontal:
            c1 = _rand(rng, 0, w - rw1 - rw2 - 1)
            c2 = c1 + rw1 + 1
            r1 = 0 if border else _rand(rng, 0, h - rh1)
            low, high = max(0, r1 - rh2 + 2), min(h - rh2, r1 + rh1 - 2)
            if low > high:
                continue
            r2 = _rand(rng, low, high)
        else:
            if h - rh1 - rh2 - 1 < 0:
                continue
            r1 = 0 if border else _rand(rng, 0, h - rh1 - rh2 - 1)
            r2 = r1 + rh1 + 1
            c1 = _rand(rng, 0, w - rw1)
            low, high = max(0, c1 - rw2 + 2), min(w - rw2, c1 + rw1 - 2)
            if low > high:
                continue
            c2 = _rand(rng, low, high)
        rects = [(r1, c1, r1 + rh1 - 1, c1 + rw1 - 1),
                 (r2, c2, r2 + rh2 - 1, c2 + rw2 - 1)]
        if count == 3:
            for _ in range(160):
                rh = _rand(rng, 4, min(8, h - 1))
                rw = _rand(rng, 4, min(8, w - 1))
                r, c = _rand(rng, 0, h - rh), _rand(rng, 0, w - rw)
                third = (r, c, r + rh - 1, c + rw - 1)
                if all(_separated(third, x) for x in rects):
                    rects.append(third)
                    break
        if len(rects) == count and sum((a[2]-a[0]+1)*(a[3]-a[1]+1)
                                       for a in rects) < .57*h*w:
            return h, w, sorted(rects)
    raise RuntimeError("could not place gap rectangles")


def _gap_centers(rects):
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            if a[3] + 2 == b[1] or b[3] + 2 == a[1]:
                left, right = (a, b) if a[3] < b[1] else (b, a)
                for r in range(max(left[0], right[0]) + 1,
                               min(left[2], right[2])):
                    yield (r, left[3]), (r, right[1])
            if a[2] + 2 == b[0] or b[2] + 2 == a[0]:
                top, bottom = (a, b) if a[2] < b[0] else (b, a)
                for c in range(max(top[1], bottom[1]) + 1,
                               min(top[3], bottom[3])):
                    yield (top[2], c), (bottom[0], c)


def _add_markers(rng, rects, active, desired, isolated=False, allowed=None):
    candidates = [p for rect in rects for p in
                  ((r, c) for r in range(rect[0], rect[2] + 1)
                   for c in range(rect[1], rect[3] + 1))
                  if allowed is None or allowed(p)]
    order = rng.permutation(len(candidates))
    for i in order:
        if len(active) >= desired:
            break
        p = candidates[int(i)]
        if p in active or (isolated and any(_distance(p, q) <= 2 for q in active)):
            continue
        trial = active | {p}
        if all(_carrier_ok(rect, trial) for rect in rects):
            active.add(p)


def _marker_layout(rng, h, w, rects, mode, split, gap):
    active = set()
    crossing = None
    if mode == "crossing":
        options = []
        for rect in rects:
            r0, c0, r1, c1 = rect
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if r not in (r0, r1) and c not in (c0, c1):
                        continue
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        q = (r + dr, c + dc)
                        if (0 <= q[0] < h and 0 <= q[1] < w
                                and not any(_inside(x, q) for x in rects)):
                            options.append(((r, c), q))
        if not options:
            raise RuntimeError("no marker crossing")
        p, crossing = options[_rand(rng, 0, len(options) - 1)]
        active.add(p)
        # A separate, isolated active stamp makes the 3x3 footprint legible.
        other = [q for rect in rects for q in
                 ((r, c) for r in range(rect[0], rect[2] + 1)
                  for c in range(rect[1], rect[3] + 1))
                 if _distance(q, p) > 3]
        if not other:
            raise RuntimeError("no isolated stamp")
        active.add(other[_rand(rng, 0, len(other) - 1)])
    elif mode == "clipping":
        choices = [p for rect in rects for p in
                   ((r, c) for r in range(rect[0], rect[2] + 1)
                    for c in range(rect[1], rect[3] + 1))
                   if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
        if not choices:
            raise RuntimeError("no canvas edge marker")
        p = choices[_rand(rng, 0, len(choices) - 1)]
        active.add(p)
        near = [q for rect in rects for q in
                ((r, c) for r in range(rect[0], rect[2] + 1)
                 for c in range(rect[1], rect[3] + 1))
                if _distance(q, p) == 1 and q != p]
        if not near:
            raise RuntimeError("no adjacent marker")
        active.add(near[_rand(rng, 0, len(near) - 1)])
    elif mode == "corner":
        corners = [(r, c) for a, b, d, e in rects
                   for r in (a, d) for c in (b, e)]
        active.add(corners[_rand(rng, 0, len(corners) - 1)])

    if gap:
        options = list(_gap_centers(rects))
        if not options:
            raise RuntimeError("no facing gap")
        a, b = options[_rand(rng, 0, len(options) - 1)]
        active.update((a, b))

    if mode == "inert":
        active.clear()
    elif mode == "sparse":
        desired = max(len(active), _rand(rng, 1, 3))
        _add_markers(rng, rects, active, desired)
    elif mode == "empty_field":
        excluded = rects[_rand(rng, 0, len(rects) - 1)]
        _add_markers(rng, rects, active, max(len(active), _rand(rng, 2, 5)),
                     allowed=lambda p: not _inside(excluded, p))
    elif mode == "isolated":
        _add_markers(rng, rects, active, max(len(active), _rand(rng, 1, 4)),
                     isolated=True)
    elif mode == "dense":
        target = max(len(active), _rand(rng, 8, 20))
        _add_markers(rng, rects, active, target)
    else:
        target = max(len(active), _rand(rng, 2, 8))
        _add_markers(rng, rects, active, target)

    if not all(_carrier_ok(rect, active) for rect in rects):
        raise RuntimeError("markers obscure carrier")
    exterior = [(r, c) for r in range(h) for c in range(w)
                if not any(_inside(x, (r, c)) for x in rects)]
    if mode == "no_distractor":
        distractors = set()
    elif mode == "inert":
        distractors = set()
    else:
        distractors = {crossing} if crossing is not None else set()
    if mode == "inert":
        count = _rand(rng, 1, 6)
    elif mode == "crossing":
        count = _rand(rng, 2, 5)
    elif split == "test" and mode == "dense":
        count = _rand(rng, 3, 8)
    elif mode == "no_distractor":
        count = 0
    else:
        count = _rand(rng, 0, 5)
    order = rng.permutation(len(exterior))
    for i in order:
        if len(distractors) >= count:
            break
        q = exterior[int(i)]
        if q in distractors:
            continue
        if mode == "crossing" and not any(_distance(q, p) >= 3 for p in active):
            continue
        distractors.add(q)
    if mode == "crossing":
        far = [q for q in exterior if min(_distance(q, p) for p in active) >= 3]
        if not far:
            raise RuntimeError("no far distractor")
        distractors.add(far[_rand(rng, 0, len(far) - 1)])
    return active, distractors


def _make_pair(rng, roles, mode, split):
    background, carrier, marker, halo = roles
    for _ in range(80):
        count = (_rand(rng, 2, 3) if split == "train"
                 else _rand(rng, 1, 3))
        gap = count >= 2 and (mode == "gap" or
                              (mode == "clipping" and _rand(rng, 0, 1) == 1))
        border = mode == "clipping" or (mode not in ("crossing", "inert")
                                       and _rand(rng, 0, 3) == 0)
        try:
            h, w, rects = (_gap_layout(rng, count, border) if gap
                           else _free_layout(rng, count, border))
            active, distracting = _marker_layout(rng, h, w, rects, mode,
                                                  split, gap)
            grid = np.full((h, w), background, dtype=np.int8)
            for r0, c0, r1, c1 in rects:
                grid[r0:r1 + 1, c0:c1 + 1] = carrier
            for r, c in active | distracting:
                grid[r, c] = marker
            if len(_role_candidates(grid, halo)) != 1:
                continue
            out = _predict(grid, rects, marker, halo)
            if mode == "crossing":
                if (not any(out[r, c] == halo for r, c in distracting) or
                        not any(out[r, c] == marker
                                and min(_distance((r, c), p) for p in active) >= 3
                                for r, c in distracting) or
                        not any(all(a == b or _distance(a, b) > 2 for b in active)
                                for a in active)):
                    continue
            return {"input": grid, "output": out}
        except (RuntimeError, AssertionError):
            continue
    raise RuntimeError("could not make a distinct valid pair")


def _counts(seed, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= value <= hi):
            raise ValueError(name + " is outside its legal range")
    a, b = num_train, num_test
    if a is None and b is None:
        b = 1 + seed % 4
        a = max(2, b + 1) + (seed // 4) % (11 - max(2, b + 1))
    elif a is None:
        lo = max(2, b + 1)
        a = lo + seed % (11 - lo)
    elif b is None:
        b = 1 + seed % min(4, a - 1)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    value = int(seed) % (1 << 64)
    ntrain, ntest = _counts(value, num_train, num_test)
    rng = np.random.default_rng(value)
    halo = _rand(rng, 0, 9)
    colors = [x for x in range(10) if x != halo]
    first = tuple(int(x) for x in rng.choice(colors, 3, replace=False))
    varied = bool(_rand(rng, 0, 1))
    child_seeds = rng.integers(0, 1 << 63, size=ntrain + ntest, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    layouts = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(child_seeds[len(seen)]))
            if split == "train" and index == 0:
                mode = "crossing"
            elif split == "train" and index == 1:
                mode = "clipping"
            else:
                modes = (("sparse", "empty_field", "isolated", "dense",
                          "no_distractor", "corner", "gap")
                         if split == "train" else
                         ("sparse", "empty_field", "isolated", "dense",
                          "no_distractor", "corner", "gap", "inert"))
                mode = modes[_rand(pair_rng, 0, len(modes) - 1)]
            if varied and (split == "test" or index > 0):
                palette = tuple(int(x) for x in pair_rng.choice(colors, 3, replace=False))
                if split == "train" and index == 1 and palette == first:
                    palette = palette[1:] + palette[:1]
            else:
                palette = first
            roles = palette + (halo,)
            for _ in range(80):
                pair = _make_pair(pair_rng, roles, mode, split)
                key = _key(pair["input"])
                candidate = _role_candidates(pair["input"], halo)
                rects = candidate[0][1]
                layout = (pair["input"].shape, tuple(rects))
                if key not in seen and layout not in layouts:
                    break
            else:
                raise RuntimeError("could not produce unique pairs")
            puzzle[split].append(pair)
            seen.add(key)
            layouts.add(layout)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair, halo):
    grid = pair["input"]
    candidates = _role_candidates(grid, halo)
    _check(len(candidates) == 1, "input must uniquely reveal semantic roles")
    roles, rects = candidates[0]
    marker = roles[2]
    active = _active(grid, rects, marker)
    distracting = [(int(r), int(c)) for r, c in np.argwhere(grid == marker)
                   if (int(r), int(c)) not in active]
    h, w = grid.shape
    crossing = any(pair["output"][r, c] == halo for r, c in distracting)
    far = any(all(_distance(d, a) >= 3 for a in active) for d in distracting)
    isolated = any(all(a == b or _distance(a, b) > 2 for b in active)
                   for a in active)
    clipped = any(r in (0, h - 1) or c in (0, w - 1) for r, c in active)
    overlap = any(_distance(a, b) <= 2 for i, a in enumerate(active)
                  for b in active[i + 1:])
    spill = bool(np.any((grid == roles[0]) & (pair["output"] == halo)))
    gap = any(_distance(a, b) == 2 and
              any(_inside(x, a) and _inside(y, b) for x in rects for y in rects
                  if x != y) for i, a in enumerate(active) for b in active[i + 1:])
    return {"roles": roles, "rects": rects, "active": active,
            "distracting": distracting, "crossing": crossing, "far": far,
            "isolated": isolated, "clipped": clipped, "overlap": overlap,
            "spill": spill, "gap": gap}


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must contain train and test")
    _check(all(isinstance(puzzle[x], list) for x in ("train", "test")),
           "splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
           "invalid pair counts")
    all_pairs = puzzle["train"] + puzzle["test"]
    seen = set()
    layouts = set()
    for pair in all_pairs:
        _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
               "pair schema is invalid")
        for grid in pair.values():
            _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                   "grid must be a 2D array")
            _check(all(1 <= x <= 30 for x in grid.shape)
                   and np.issubdtype(grid.dtype, np.integer)
                   and bool(np.all((0 <= grid) & (grid <= 9))),
                   "grid size, dtype, or colors invalid")
        _check(pair["input"].shape == pair["output"].shape,
               "output must preserve input size")
        key = _key(pair["input"])
        _check(key not in seen, "duplicate input")
        seen.add(key)
        digest = hashlib.sha256(pair["input"].astype(np.int8).tobytes() +
                                pair["output"].astype(np.int8).tobytes()).hexdigest()
        _check((pair["input"].shape, digest) not in _SOURCE_DIGESTS, "source pair copied")
    halos = []
    for pair in puzzle["train"]:
        extra = set(map(int, np.unique(pair["output"]))) - set(map(int, np.unique(pair["input"])))
        _check(len(extra) == 1, "training must reveal the halo color")
        halos.append(next(iter(extra)))
    _check(len(set(halos)) == 1, "halo must be stable within puzzle")
    halo = halos[0]
    data = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _features(pair, halo)
            expected = _predict(pair["input"], info["rects"], info["roles"][2], halo)
            _check(np.array_equal(expected, pair["output"]),
                   "pair violates the embedded-marker 3x3 union")
            _check(1 <= len(info["rects"]) <= (3 if split == "test" else 3),
                   "field count outside regime")
            layout = (pair["input"].shape, tuple(info["rects"]))
            _check(layout not in layouts, "duplicate rectangle layout")
            layouts.add(layout)
            if split == "train":
                _check(len(info["rects"]) >= 2 and len(info["active"]) >= 1,
                       "training must have at least two fields and an active stamp")
            data[split].append(info)
    train = data["train"]
    _check(any(x["crossing"] and x["far"] for x in train),
           "training must distinguish crossing from inert exterior markers")
    _check(any(x["clipped"] for x in train), "training must demonstrate clipping")
    _check(any(x["overlap"] for x in train), "training must demonstrate overlap")
    _check(any(x["isolated"] for x in train), "training must reveal isolated stamp size")
    _check(any(x["spill"] for x in train), "training must demonstrate background spill")
    palettes = {x["roles"][:3] for x in train}
    if len(palettes) > 1:
        _check(all(x["roles"][3] == halo for x in train),
               "varying input palettes must share halo color")


def _equal(a, b):
    return (len(a["train"]) == len(b["train"]) and len(a["test"]) == len(b["test"])
            and all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                    for x, y in zip(a[s], b[s]) for k in ("input", "output")))


def _cellwise_prediction(grid, rects, marker, halo):
    centers = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])
               if grid[r, c] == marker and any(_inside(x, (r, c)) for x in rects)]
    result = grid.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) not in centers and any(_distance((r, c), p) <= 1 for p in centers):
                result[r, c] = halo
    return result


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    features = {"crossing": 0, "clipped": 0, "gap": 0, "empty": 0,
                "no_distractor": 0, "sparse": 0, "dense": 0, "inert_test": 0,
                "interior": 0, "four_square": 0, "corner": 0,
                "edge_cluster": 0, "varied_palette": 0, "small_test": 0,
                "one_field_test": 0, "sparse_no_distractor": 0}
    aspects = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        halo = next(iter(set(map(int, np.unique(puzzle["train"][0]["output"]))) -
                         set(map(int, np.unique(puzzle["train"][0]["input"])))))
        pair_palettes = set()
        train_min_area = min(x["input"].size for x in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _features(pair, halo)
                grid = pair["input"]
                h, w = grid.shape
                _check(np.array_equal(
                    _cellwise_prediction(grid, info["rects"], info["roles"][2], halo),
                    pair["output"]), "independent cellwise transform failed")
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, tuple(info["rects"])))
                palettes.add(info["roles"])
                pair_palettes.add(info["roles"][:3])
                aspects.add("wide" if w >= 1.25*h else
                            "tall" if h >= 1.25*w else "balanced")
                features["crossing"] += info["crossing"]
                features["clipped"] += info["clipped"]
                features["gap"] += info["gap"]
                features["empty"] += any(not any(_inside(r, a) for a in info["active"])
                                         for r in info["rects"])
                features["no_distractor"] += not info["distracting"]
                features["sparse"] += 1 <= len(info["active"]) <= 2
                features["dense"] += len(info["active"]) >= 9
                features["inert_test"] += split == "test" and not info["active"]
                features["interior"] += (not info["clipped"] and
                    all(r0 > 0 and c0 > 0 and r1 < h-1 and c1 < w-1
                        for r0, c0, r1, c1 in info["rects"]))
                features["four_square"] += any(r1-r0+1 == 4 and c1-c0+1 == 4
                    for r0, c0, r1, c1 in info["rects"])
                features["corner"] += any(p in ((r0,c0),(r0,c1),(r1,c0),(r1,c1))
                    for p in info["active"] for r0,c0,r1,c1 in info["rects"])
                features["edge_cluster"] += any(
                    _distance(a,b) == 1 and any(
                        _inside(rect,a) and
                        (a[0] in (rect[0],rect[2]) or a[1] in (rect[1],rect[3]))
                        for rect in info["rects"])
                    for i,a in enumerate(info["active"]) for b in info["active"][i+1:])
                features["small_test"] += split == "test" and grid.size < train_min_area
                features["one_field_test"] += split == "test" and len(info["rects"]) == 1
                features["sparse_no_distractor"] += (
                    1 <= len(info["active"]) <= 2 and not info["distracting"])
        features["varied_palette"] += len(pair_palettes) > 1
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _check(len(shapes) >= 30 and len(layouts) >= 100 and len(palettes) >= 30,
           "shapes, layouts, or palettes lack diversity")
    _check(aspects == {"wide", "tall", "balanced"}, "aspect ratios lack diversity")
    _check(all(v > 0 for v in features.values()), "reviewed feature coverage missing")
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4),
                                 (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + i, a, b)
        validate_generated_puzzle(p)
        _check((a is None or len(p["train"]) == a) and
               (b is None or len(p["test"]) == b), "requested counts ignored")
        _check(_equal(p, generate_puzzle(1000 + i, a, b)), "generation is nondeterministic")
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(7, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
