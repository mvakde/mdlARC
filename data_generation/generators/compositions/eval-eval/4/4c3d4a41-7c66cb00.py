"""Skyline docking: color-routed rigid masks supported by toothed floors.

Only stdlib and NumPy. No parent imports, file reads, persistent state or writes.
The direct constructor and the input-only stepwise solver use different movement
calculations. All diagnostics are bounded and returned in memory.
"""
from __future__ import annotations

import hashlib
import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
ABLATIONS = ("flat", "highest", "warp", "group", "wrong_route", "no_recolor", "bbox", "no_clip", "stack")


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must be integer")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _counts(a, b):
    for name, n, lo, hi in (("num_train", a, 2, 10), ("num_test", b, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return int(a), int(b)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, stop, bg, diagonal=False):
    unseen = {tuple(map(int, p)) for p in np.argwhere(a[:stop] != bg)}
    result = []
    neighbors = EIGHT if diagonal else FOUR
    while unseen:
        start = min(unseen)
        color = int(a[start])
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in unseen and int(a[p]) == color:
                    unseen.remove(p)
                    todo.append(p)
        result.append((color, tuple(sorted(cells))))
    return result


def _read_band(a, top, bottom, bg):
    ys, xs = np.where(a[top:bottom + 1] != bg)
    left, right = int(xs.min()), int(xs.max())
    _require(bottom - top >= 4 and right - left >= 6, "small receiver")
    block = a[top:bottom + 1, left:right + 1]
    _require(np.all(block != bg) and len(xs) == block.size, "receiver must be a filled rectangle")
    rail, route = int(block[0, 0]), int(block[1, 1])
    _require(route not in (rail, bg), "receiver roles")
    _require(np.all(block[0] == rail) and np.all(block[-1] == rail), "roof/base")
    _require(np.all(block[:, 0] == rail) and np.all(block[:, -1] == rail), "side rails")
    floor = {}
    for c in range(left + 1, right):
        col = a[top + 1:bottom, c]
        _require(col[0] == route and col[-1] == rail, "visible cavity and tooth required")
        teeth = np.flatnonzero(col == rail)
        first = int(teeth[0])
        _require(np.all(col[:first] == route) and np.all(col[first:] == rail), "solid skyline floor")
        floor[c] = top + 1 + first
    return dict(top=top, bottom=bottom, left=left, right=right, route=route, rail=rail, floor=floor)


def _canonical_candidates(a, bg):
    occupied = np.any(a != bg, axis=1)
    starts = [r for r in range(1, len(a)) if occupied[r] and not occupied[r - 1]]
    for cut in starts:
        bands = []
        r = cut
        try:
            while r < len(a):
                if not occupied[r]:
                    r += 1
                    continue
                top = r
                while r + 1 < len(a) and occupied[r + 1]:
                    r += 1
                bands.append(_read_band(a, top, r, bg))
                r += 1
            routes = [b["route"] for b in bands]
            _require(len(set(routes)) == len(routes), "unique routing colors")
            comps = _components(a, cut, bg)
            _require(bool(comps) and all(color in routes for color, _ in comps), "staging colors")
            lookup = {b["route"]: b for b in bands}
            _require(all(all(lookup[color]["left"] < c < lookup[color]["right"] for _, c in cells)
                         for color, cells in comps), "lateral fit")
            yield dict(bg=bg, cut=cut, bands=bands, comps=comps)
        except AssertionError:
            continue


def _analyze(value):
    a = _grid(value)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg) and
             np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "one-cell background border")
    found = []
    for turn in range(4):
        canonical = np.rot90(a, turn)
        for scene in _canonical_candidates(canonical, bg):
            scene.update(turn=turn, a=canonical)
            found.append(scene)
    _require(len(found) == 1, "input must have one staging/receiver interpretation")
    return found[0]


def _direct(a, scene, mode=None):
    """Closed-form placement, also used for explicit counterfactual checks."""
    out = a.copy()
    out[:scene["cut"]] = scene["bg"]
    bands = scene["bands"]
    lookup = {b["route"]: b for b in bands}
    comps = scene["comps"]
    stack_floors = {b["route"]: dict(b["floor"]) for b in bands}
    if mode == "group":
        comps = [(color, tuple(p for col, cells in comps if col == color for p in cells))
                 for color in sorted({color for color, _ in comps})]
    elif mode == "stack":
        comps = sorted(comps, key=lambda item: (-max(r for r, c in item[1]), item[1][0], item[0]))
    for color, cells in comps:
        band = lookup[color]
        if mode == "wrong_route":
            band = bands[(bands.index(band) + 1) % len(bands)]
        if mode == "bbox":
            cells = tuple((r, c) for r in range(min(r for r, c in cells), max(r for r, c in cells) + 1)
                          for c in range(min(c for r, c in cells), max(c for r, c in cells) + 1))
        floor = stack_floors[color] if mode == "stack" else band["floor"]
        shift = min(floor[c] - 1 - r for r, c in cells)
        if mode == "flat":
            shift = band["bottom"] - 2 - max(r for r, c in cells)
        elif mode == "highest":
            shift = min(floor[c] for r, c in cells) - 1 - max(r for r, c in cells)
        by_col = {}
        if mode == "warp":
            by_col = {c: floor[c] - 1 - max(r for r, x in cells if x == c) for r, c in cells}
        for r, c in cells:
            dest = r + by_col.get(c, shift)
            if (band["top"] < dest < band["bottom"]) or (mode == "no_clip" and 0 <= dest < len(a)):
                out[dest, c] = color if mode == "no_recolor" else band["rail"]
                if mode == "stack":
                    floor[c] = min(floor[c], dest)
    return out


def _stepwise(a, scene):
    """Input pixels only: start low, push upward until the whole mask clears."""
    out = a.copy()
    out[:scene["cut"]] = scene["bg"]
    lookup = {b["route"]: b for b in scene["bands"]}
    for color, cells in scene["comps"]:
        b = lookup[color]
        shift = b["bottom"] - 1 - max(r for r, c in cells)
        while any(r + shift >= b["floor"][c] for r, c in cells):
            shift -= 1
        for r, c in cells:
            dest = r + shift
            if dest > b["top"]:
                _require(dest < b["floor"][c], "floor penetration")
                out[dest, c] = b["rail"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _analyze(input_grid)
    return np.rot90(_stepwise(scene["a"], scene), -scene["turn"]).copy()


def _mask(rng, cap_h, cap_w, force_height=False):
    h = cap_h if force_height else int(rng.integers(1, cap_h + 1))
    w = int(rng.integers(1, cap_w + 1))
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, 0) for r in range(h)} | {(0, c) for c in range(w)}
    elif kind == 2:
        cells = {(r, w // 2) for r in range(h)}
        for r in range(h):
            if rng.random() < .65:
                endpoint = int(rng.integers(w))
                cells.update((r, c) for c in range(min(endpoint, w // 2), max(endpoint, w // 2) + 1))
    elif kind == 3 and min(h, w) >= 3:
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 4:
        cells = {(0, 0)}
        r = c = 0
        while r < h - 1 or c < w - 1:
            if r < h - 1 and (c == w - 1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            cells.add((r, c))
    else:
        # A spine guarantees the requested height; growth supplies irregularity.
        cells = {(r, int(w // 2)) for r in range(h)}
        goal = int(rng.integers(h, h * w + 1))
        while len(cells) < goal:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = {(r, w - 1 - c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _place(a, group, stage, left, right, bg, rng):
    h = 1 + max(r for color, mask in group for r, c in mask)
    w = 1 + max(c for color, mask in group for r, c in mask)
    if h > stage or w > right - left + 1:
        return False
    for top in rng.permutation(np.arange(1, stage - h + 2)):
        for start in rng.permutation(np.arange(left, right - w + 2)):
            colored = {(int(top) + r, int(start) + c): color for color, mask in group for r, c in mask}
            if any(a[p] != bg for p in colored):
                continue
            bad = False
            for (r, c), color in colored.items():
                for dr, dc in EIGHT:
                    p = r + dr, c + dc
                    if p not in colored and a[p] == color:
                        bad = True
            if bad:
                continue
            for p, color in colored.items():
                a[p] = color
            return True
    return False


def _palette(rng, n):
    palette = list(map(int, rng.permutation(10)))
    bg, routes = palette[0], palette[1:n + 1]
    outside = palette[n + 1:]
    rails = [int(rng.choice(outside))]
    for route in routes[1:]:
        choices = [c for c in palette[1:] if c != route]
        rails.append(rails[0] if rng.random() < .35 else int(rng.choice(choices)))
    return bg, routes, rails


def _draw_band(a, top, depth, left, right, route, rail, heights):
    bottom = top + depth - 1
    a[top:bottom + 1, left:right + 1] = rail
    floor = {}
    for c, height in zip(range(left + 1, right), heights):
        floor[c] = bottom - int(height)
        a[top + 1:floor[c], c] = route
    return dict(top=top, bottom=bottom, left=left, right=right, route=route, rail=rail, floor=floor)


def _attempt(rng, n, cap, lesson, roughness, tall):
    stage = int(rng.integers(max(6, cap), max(6, cap) + 3))
    width = int(rng.integers(17, 31))
    budget = 30 - (stage + 3 + n - 1)
    if budget < 5 * n:
        return None
    depths = [5] * n
    for _ in range(int(rng.integers(budget - 5 * n + 1))):
        depths[int(rng.integers(n))] += 1
    bg, routes, rails = _palette(rng, n)
    height = stage + 3 + sum(depths) + n - 1
    a = np.full((height, width), bg, dtype=np.int8)
    bands, top = [], stage + 2
    for depth, route, rail in zip(depths, routes, rails):
        left = int(rng.integers(1, 4))
        right = width - 1 - int(rng.integers(1, 4))
        count = right - left - 1
        lo, hi = {"low": (1, 3), "high": (6, 8), "normal": (1, 5)}[roughness]
        run_count = int(rng.integers(lo, min(count, hi) + 1))
        cuts = sorted(map(int, rng.choice(np.arange(1, count), run_count - 1, replace=False)))
        heights = []
        previous = None
        for start, stop in zip([0] + cuts, cuts + [count]):
            value = int(rng.choice([x for x in range(1, depth - 2) if x != previous]))
            heights.extend([value] * (stop - start))
            previous = value
        bands.append(_draw_band(a, top, depth, left, right, route, rail, heights))
        top += depth + 1
    left = max(b["left"] for b in bands) + 1
    right = min(b["right"] for b in bands) - 1
    maxw = max(2, min(right - left + 1, width // 2))
    if lesson == "split":
        h, w = int(rng.integers(3, min(cap, stage) + 1)), int(rng.integers(3, maxw + 1))
        outer, inner = map(int, rng.choice(routes, 2, replace=False))
        rim = tuple((r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1))
        fill = tuple((r, c) for r in range(1, h - 1) for c in range(1, w - 1))
        if not _place(a, [(outer, rim), (inner, fill)], stage, left, right, bg, rng):
            return None
    if lesson == "independent":
        repeated = int(rng.choice(routes))
        for _ in range(2):
            if not _place(a, [(repeated, _mask(rng, cap, max(2, maxw // 2)))], stage, left, right, bg, rng):
                return None
    if tall:
        if not _place(a, [(int(rng.choice(routes)), _mask(rng, cap, maxw, True))], stage, left, right, bg, rng):
            return None
    for route in rng.permutation(routes):
        if not np.any(a[:stage + 1] == route):
            if not _place(a, [(int(route), _mask(rng, cap, maxw))], stage, left, right, bg, rng):
                return None
    for _ in range(int(rng.integers(1, 6))):
        _place(a, [(int(rng.choice(routes)), _mask(rng, cap, maxw))], stage, left, right, bg, rng)
    scene = dict(bg=bg, bands=bands, cut=stage + 2, comps=_components(a, stage + 2, bg))
    return a, scene


def _reserve(rng, n, cap, roughness="normal"):
    """Finite reserve and literal regression: irregular support, repeats, a frame."""
    h = max(4, min(cap, 6)) if n < 3 else 4
    roof = h + 4
    depths = [7, 7] + ([5] if n == 3 else [])
    bg, routes, rails = _palette(rng, len(depths))
    height = roof + sum(depths) + len(depths)
    a = np.full((height, 22), bg, dtype=np.int8)
    bands, top = [], roof
    for i, depth in enumerate(depths):
        heights = [1] * 18
        if i == 0:
            heights[:3], heights[8:] = [3] * 3, [4] * 10
        if roughness == "high" and i == 1:
            heights = [1 + c % 2 for c in range(18)]
        bands.append(_draw_band(a, top, depth, 1, 20, routes[i], rails[i], heights))
        top += depth + 1
    a[1, 4:6] = routes[0]
    a[1:h + 1, 5] = routes[0]
    a[1:4, 10] = routes[0]
    a[1:3, 8] = routes[0]
    a[5:7, 8] = routes[0]
    a[1:4, 14:19] = routes[1]
    a[2, 15:18] = routes[0]
    if n == 3:
        a[5, 18] = routes[2]
    scene = dict(bg=bg, bands=bands, cut=roof, comps=_components(a, roof, bg))
    return a, scene


def _features(scene, expected, ablations=True):
    a, comps, bands = scene["a"], scene["comps"], scene["bands"]
    points = {p for _, cells in comps for p in cells}
    contact = any((r + dr, c + dc) in points and a[r, c] != a[r + dr, c + dc]
                  for r, c in points for dr, dc in FOUR)
    repeated = any(x == y and max(r for r, c in p) != max(r for r, c in q)
                   for i, (x, p) in enumerate(comps) for y, q in comps[i + 1:])
    killed = {mode for mode in ABLATIONS if not np.array_equal(_direct(a, scene, mode), expected)} if ablations else set()
    lookup, seen_targets, overlap = {b["route"]: b for b in bands}, set(), 0
    for color, cells in comps:
        b = lookup[color]
        shift = min(b["floor"][c] - 1 - r for r, c in cells)
        targets = {(r + shift, c) for r, c in cells if r + shift > b["top"]}
        overlap += len(targets & seen_targets)
        seen_targets.update(targets)
    return dict(contact=contact, repeated=repeated, killed=killed,
                overlap=overlap,
                max_height=max(max(r for r, c in cells) - min(r for r, c in cells) + 1 for _, cells in comps),
                count=len(comps), bands=len(bands), turn=scene["turn"],
                floors=tuple(tuple(b["bottom"] - b["floor"][c] for c in sorted(b["floor"])) for b in bands))


def _runs(profile):
    return 1 + sum(a != b for a, b in zip(profile, profile[1:]))


def _qualifies(scene, out, lesson):
    if lesson is None:
        return True
    f = _features(scene, out, ablations=False)
    if lesson == "split":
        if not f["contact"]:
            return False
        required = ("wrong_route", "no_recolor", "bbox", "flat")
    else:
        if not f["repeated"] or not f["overlap"]:
            return False
        required = ("highest", "group", "warp", "no_clip", "flat", "stack")
    return all(not np.array_equal(_direct(scene["a"], scene, mode), out) for mode in required)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a_count, b_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("tall", "count", "relief", "direction")))
    turns = list(map(int, rng.permutation(4)))
    mixed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a_count), ("test", b_count)):
        for index in range(count):
            training = split == "train"
            lesson = ("split", "independent")[index] if training and index < 2 else None
            n = 2 if lesson == "split" else int(rng.integers(1, 3 if training or regime != "count" else 4))
            if not training and regime == "count":
                n = 3
            cap = 4 if regime == "tall" and training else int(rng.integers(3, 6))
            tall = not training and regime == "tall"
            if tall:
                cap = int(rng.integers(6, 9))
            if regime == "direction":
                turn = turns[index % 2] if training else turns[2 + index % 2]
            else:
                turn = turns[index % 4] if mixed else turns[0]
            pair = None
            roughness = ("low" if training else "high") if regime == "relief" else "normal"
            for attempt in range(160):
                candidate = _attempt(rng, n, cap, lesson, roughness, tall)
                if candidate is None:
                    continue
                inp, constructed = candidate
                out = _direct(inp, constructed)
                if not _qualifies(dict(constructed, a=inp, turn=0), out, lesson):
                    continue
                try:
                    _analyze(inp)
                except AssertionError:
                    continue
                rotated = np.rot90(inp, turn).copy()
                if _key(rotated) in seen:
                    continue
                pair = {"input": rotated, "output": np.rot90(out, turn).copy()}
                break
            if pair is None:
                # Forty distinct bijections, more than the maximum 14 pairs.
                inp, constructed = _reserve(rng, 3 if n == 3 else 2, cap, roughness)
                out = _direct(inp, constructed)
                scene = _analyze(inp)
                _require(_qualifies(scene, out, lesson), "reserve witness")
                for multiplier in (1, 3, 7, 9):
                    for offset in range(10):
                        mapping = np.array([(c * multiplier + offset) % 10 for c in range(10)], dtype=np.int8)
                        rotated = np.rot90(mapping[inp], turn).copy()
                        if _key(rotated) not in seen:
                            pair = {"input": rotated, "output": np.rot90(mapping[out], turn).copy()}
                            break
                    if pair is not None:
                        break
            _require(pair is not None, "bounded uniqueness reserve exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features, test_features = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape/duplicate input")
            seen.add(_key(inp))
            scene = _analyze(inp)
            expected = _stepwise(scene["a"], scene)
            _require(np.array_equal(np.rot90(out, scene["turn"]), expected), "wrong output")
            _require(not np.array_equal(inp, out), "no movement")
            _require(np.array_equal(_direct(scene["a"], scene), expected), "formula/simulation disagreement")
            comps = scene["comps"]
            eight = _components(scene["a"], scene["cut"], scene["bg"], True)
            _require(len(eight) == len(comps), "generated diagonal-only ambiguity")
            _require({col for col, _ in comps} == {b["route"] for b in scene["bands"]}, "generated unused route")
            # Every unclipped cell lands in the original cavity; every stamp has
            # at least one surviving support contact, before union with others.
            lookup = {b["route"]: b for b in scene["bands"]}
            for color, cells in comps:
                b = lookup[color]
                shift = min(b["floor"][c] - 1 - r for r, c in cells)
                kept = [(r + shift, c) for r, c in cells if r + shift > b["top"]]
                _require(bool(kept) and any(r == b["floor"][c] - 1 for r, c in kept), "support witness")
                _require(all(scene["a"][r, c] == color for r, c in kept), "stamp outside cavity")
            feature = _features(scene, expected)
            (train_features if split == "train" else test_features).append(feature)
    _require(any(f["contact"] for f in train_features), "training color split witness")
    _require(any(f["repeated"] for f in train_features), "training independent same-color witness")
    _require(any(f["overlap"] for f in train_features), "training stamp-union witness")
    killed = set().union(*(f["killed"] for f in train_features))
    _require(set(ABLATIONS) <= killed, "training permits a tested shortcut")
    train_turns = {f["turn"] for f in train_features}
    if any(f["turn"] not in train_turns for f in test_features):
        _require(len(train_turns) >= 2, "direction transfer must be demonstrated")


def _regressions():
    # Literal coordinates, not just constructor-versus-solver equality.
    a, s = _reserve(np.random.default_rng(3), 2, 4)
    actual = solve(a)
    b = s["bands"][0]
    t, rail, route = b["top"], b["rail"], b["route"]
    _require(actual[t + 1, 4] == rail and actual[t + 1, 5] == rail, "rigid overhang")
    _require(actual[t + 2, 4] == route, "no columnwise warp")
    _require(actual[t + 1, 10] == rail and actual[t, 10] == rail, "clipped repeated rod")
    _require(np.all(actual[:t] == s["bg"]), "source erasure / roof clipping")
    f = _features(_analyze(a), actual)
    _require(set(ABLATIONS) <= f["killed"], "literal fixture must expose every shortcut")
    for n in (2, 3):
        for cap in (3, 4, 6, 8):
            for roughness in ("low", "high"):
                fixture, built = _reserve(np.random.default_rng(n + cap), n, cap, roughness)
                sc = _analyze(fixture)
                label = _direct(fixture, built)
                _require(np.array_equal(solve(fixture), label), "reserve semantics")
                _require(_qualifies(sc, label, "split") and _qualifies(sc, label, "independent"), "reserve lessons")
    for turn in range(4):
        # Bijection over all ten colors; no literal color has a semantic role.
        mapping = np.array([(c * 3 + turn) % 10 for c in range(10)], dtype=np.int8)
        _require(np.array_equal(solve(np.rot90(mapping[a], turn)), np.rot90(mapping[actual], turn)), "equivariance")
    _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(actual)), "reflection equivariance")
    # Two independent identical stamps can coincide; they do not stack.
    x = a.copy()
    x[5:7, 8] = s["bg"]
    _require(np.array_equal(solve(x), actual), "independent union, not stacking")
    # Semantic solver also supports diagonal-only same-color separation, even
    # though the sampler avoids that otherwise-undemonstrated ambiguity.
    y = a.copy()
    y[5, 7] = route
    y[6, 8] = route
    scene = _analyze(y)
    _require(np.array_equal(solve(y), _direct(y, scene)), "four-connected semantics")
    p = generate_puzzle(991, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")


def self_test():
    """42 seeds, <=14 pairs live at once; returns small measured diversity data."""
    _regressions()
    shapes, floors, masks, fingerprints, directions, counts, backgrounds = (set() for _ in range(7))
    clipping = touching = repeated = overlapping = pairs = 0
    killed_counts = {mode: 0 for mode in ABLATIONS}
    shifts = {name: 0 for name in ("tall", "count", "relief", "direction")}
    for seed in range(42):
        nt = 2 + seed % 9
        ne = 1 + seed % min(4, nt - 1)
        p = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(p)
        per_split = {"train": [], "test": []}
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                scene = _analyze(pair["input"])
                pred = solve(pair["input"])
                _require(np.array_equal(pred, pair["output"]), "self-test solver")
                digest.update(pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                counts.add(len(scene["bands"]))
                backgrounds.add(scene["bg"])
                directions.add(scene["turn"])
                f = _features(scene, np.rot90(pred, scene["turn"]))
                per_split[split].append(f)
                floors.update(f["floors"])
                touching += f["contact"]
                repeated += f["repeated"]
                overlapping += f["overlap"] > 0
                clipping += "no_clip" in f["killed"]
                for mode in f["killed"]:
                    killed_counts[mode] += 1
                for _, cells in scene["comps"]:
                    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
                    masks.add(tuple((r - r0, c - c0) for r, c in cells))
                pairs += 1
        fingerprints.add(digest.hexdigest())
        train, test = per_split["train"], per_split["test"]
        regime = str(np.random.default_rng(seed).choice(tuple(shifts)))
        if regime == "tall":
            _require(min(f["max_height"] for f in test) > max(f["max_height"] for f in train), "height shift")
        elif regime == "count":
            _require(min(f["bands"] for f in test) > max(f["bands"] for f in train), "chamber-count shift")
        elif regime == "relief":
            _require(min(max(map(_runs, f["floors"])) for f in test) >
                     max(max(map(_runs, f["floors"])) for f in train), "floor-run shift")
        else:
            _require(len({f["turn"] for f in train}) >= 2 and
                     not ({f["turn"] for f in train} & {f["turn"] for f in test}), "direction shift")
        shifts[regime] += 1
    _require(len(fingerprints) == 42 and len(shapes) >= 30, "puzzle/shape diversity")
    _require(len(floors) >= 80 and len(masks) >= 100, "structural diversity")
    _require(directions == set(range(4)) and counts == {1, 2, 3}, "layout diversity")
    _require(backgrounds == set(range(10)), "palette role diversity")
    _require(min(killed_counts.values()) >= 20, "shortcut rejection coverage")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(682420444, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p, q = generate_puzzle(731), generate_puzzle(731)
    _require(all(np.array_equal(x[k], y[k]) for split in p for x, y in zip(p[split], q[split])
                 for k in ("input", "output")), "nondeterminism")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), floor_profiles=len(floors),
                component_masks=len(masks), directions=sorted(directions), chamber_counts=sorted(counts),
                clipping_pairs=clipping, touching_pairs=touching, repeated_depth_pairs=repeated,
                overlapping_stamp_pairs=overlapping,
                shortcut_differences=killed_counts, shift_puzzles=shifts)
