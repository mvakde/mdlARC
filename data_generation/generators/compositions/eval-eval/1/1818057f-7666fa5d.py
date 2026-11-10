"""Valve-controlled diagonal bands. All conventions are visible in each input.

Three colors have geometric roles: a majority field, diagonal rails, and ink
crosses. Complete five-cell ink pluses toggle a rail, top to bottom. Fill field
between the outermost currently active rails with the input's ink color.
"""

import numpy as np


CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALTERNATIVES = ("ungated", "signal_sites", "first_tail", "first_pair",
                "consecutive", "pixel_centers", "horizontal", "reverse",
                "discard_odd", "one_per_component")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _centers(mask):
    result = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                    & mask[2:, 1:-1] & mask[1:-1, :-2]
                                    & mask[1:-1, 2:])
    return result


def _inspect(a):
    """Decode roles and rails without output, RNG, or puzzle-level state.

Rail endpoints retain their rail color. Each transverse coordinate has one
rail only, with at least three rail-color pixels; ink can replace interior
pixels. The correct rail color/slope must be uniquely recognizable.
"""
    a = _grid(a)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or counts.max() <= a.size / 2:
        raise ValueError("three colors and a strict majority field required")
    bg = int(colors[np.argmax(counts)])
    other = [int(c) for c in colors if c != bg]
    candidates = []
    for rail_color, ink in (other, other[::-1]):
        centers = {(int(r), int(c)) for r, c in np.argwhere(_centers(a == ink))}
        if not centers:
            continue
        for slope in (-1, 1):
            lines = {}
            for r, c in np.argwhere(a == rail_color):
                r, c = int(r), int(c)
                lines.setdefault(c - slope * r, []).append(r)
            if len(lines) < 2:
                continue
            rails, used, valid = [], set(), True
            for q, rs in sorted(lines.items()):
                lo_r, hi_r = min(rs), max(rs)
                if len(rs) < 3 or hi_r - lo_r < 3:
                    valid = False
                    break
                path = [(r, q + slope * r) for r in range(lo_r, hi_r + 1)]
                if any(not (0 <= c < a.shape[1]) or int(a[r, c]) == bg for r, c in path):
                    valid = False
                    break
                valves = sorted(r + slope*c for r, c in path if (r, c) in centers)
                used.update(p for p in path if p in centers)
                rails.append(dict(q=q, lo=lo_r + slope*path[0][1],
                                  hi=hi_r + slope*path[-1][1], valves=valves,
                                  path=path))
            if valid and used == centers:
                candidates.append(dict(bg=bg, rail=rail_color, ink=ink,
                                       slope=slope, rails=rails, centers=centers))
    if len(candidates) != 1:
        raise ValueError("rail color and parallel slope must be unique")
    return candidates[0]


def _windows(rail, events=None):
    events = rail["valves"] if events is None else events
    return [(events[i], events[i+1] if i+1 < len(events) else rail["hi"])
            for i in range(0, len(events), 2)]


def _fill_mask(a, info, alternative=""):
    rr, cc = np.indices(a.shape)
    q = cc - info["slope"] * rr
    t = rr + info["slope"] * cc
    rails = info["rails"]
    labels = np.full(a.shape, -1, dtype=int)
    if alternative == "one_per_component":
        component = 0
        for r, c in np.argwhere(a == info["ink"]):
            r, c = int(r), int(c)
            if labels[r, c] >= 0:
                continue
            stack = [(r, c)]
            labels[r, c] = component
            while stack:
                x, y = stack.pop()
                for dr, dc in CARDINALS:
                    u, v = x+dr, y+dc
                    if (0 <= u < a.shape[0] and 0 <= v < a.shape[1]
                            and int(a[u, v]) == info["ink"] and labels[u, v] < 0):
                        labels[u, v] = component
                        stack.append((u, v))
            component += 1
    windows = []
    for rail in rails:
        events = rail["valves"]
        if alternative == "ungated":
            current = [(rail["lo"], rail["hi"])]
        elif alternative == "signal_sites":
            events = [r + info["slope"]*c for r, c in rail["path"]
                      if int(a[r, c]) == info["ink"]]
            current = _windows(rail, events)
        elif alternative == "first_tail":
            current = [(events[0], rail["hi"])] if events else []
        elif alternative == "first_pair":
            current = _windows(rail)[:1]
        elif alternative == "reverse":
            backwards = events[::-1]
            current = [(backwards[i+1] if i+1 < len(backwards) else rail["lo"], backwards[i])
                       for i in range(0, len(backwards), 2)]
        elif alternative == "discard_odd":
            current = [(events[i], events[i+1]) for i in range(0, len(events)-1, 2)]
        elif alternative == "one_per_component":
            components, events = set(), []
            for r, c in rail["path"]:
                if (r, c) in info["centers"] and labels[r, c] not in components:
                    components.add(labels[r, c])
                    events.append(r + info["slope"]*c)
            current = _windows(rail, events)
        else:
            current = _windows(rail)
        windows.append(current)
    fill = np.zeros(a.shape, dtype=bool)
    if alternative == "horizontal":
        # Deliberately wrong: compare rail states at a common row rather than t.
        for r in range(a.shape[0]):
            active = []
            for rail, spans in zip(rails, windows):
                tr = 2*r + info["slope"]*rail["q"]
                if any(lo <= tr <= hi for lo, hi in spans):
                    active.append(rail["q"] + info["slope"]*r)
            if len(active) >= 2:
                fill[r] = (cc[r] > min(active)) & (cc[r] < max(active))
    elif alternative == "consecutive":
        for i in range(len(rails)-1):
            for (lo1, hi1), (lo2, hi2) in ((x, y) for x in windows[i] for y in windows[i+1]):
                fill |= ((t >= max(lo1, lo2)) & (t <= min(hi1, hi2))
                         & (q > rails[i]["q"]) & (q < rails[i+1]["q"]))
    else:
        for level in range(int(t.min()), int(t.max())+1):
            active = [rail["q"] for rail, spans in zip(rails, windows)
                      if any(lo <= level <= hi for lo, hi in spans)
                      and (alternative != "pixel_centers" or (level-rail["q"]) % 2 == 0)]
            if len(active) >= 2:
                fill |= (t == level) & (q > min(active)) & (q < max(active))
    return fill & (a == info["bg"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read valve centers and visible rail endpoints, then fill active brackets."""
    a = _grid(input_grid)
    info = _inspect(a)
    out = a.copy()
    out[_fill_mask(a, info)] = info["ink"]
    return out


def _audit_mask(a, info):
    """Cellwise interval-pair check, separate from the production scanline sweep.

This is generator-owned consistency evidence, not an independent ARC solver.
"""
    intervals = [(rail["q"], lo, hi) for rail in info["rails"]
                 for lo, hi in _windows(rail)]
    mask = np.zeros(a.shape, dtype=bool)
    for r, c in np.argwhere(a == info["bg"]):
        q, t = int(c - info["slope"]*r), int(r + info["slope"]*c)
        left = any(q0 < q and lo <= t <= hi for q0, lo, hi in intervals)
        right = any(q0 > q and lo <= t <= hi for q0, lo, hi in intervals)
        mask[r, c] = left and right
    return mask


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _witnesses(pair, info=None):
    a, b = np.asarray(pair["input"]), np.asarray(pair["output"])
    info = _inspect(a) if info is None else info
    actual = a != b
    return {name for name in ALTERNATIVES
            if not np.array_equal(actual, _fill_mask(a, info, name))}


def _render(rng, specs, roles, slope, extra=True, decoy=False, canvas_min=0):
    """Specs contain (q, lo, hi, valve t coordinates); all are visible pixels."""
    bg, rail_color, ink = roles
    rail_cells, valves = set(), set()
    endpoints = set()
    paths = []
    for q, lo, hi, events in specs:
        path = [((t-q)//2, (t+q)//2) for t in range(lo, hi+1, 2)]
        paths.append(path)
        rail_cells.update(path)
        endpoints.update((path[0], path[-1]))
        valves.update(((t-q)//2, (t+q)//2) for t in events)
    ink_cells = set()
    for r, c in valves:
        ink_cells.add((r, c))
        ink_cells.update((r+dr, c+dc) for dr, dc in CARDINALS)
    if decoy:
        # A T on one outer rail, before its first real valve. Its signal-colored
        # center is on the line, but the absent cardinal arm makes it inert.
        choices = []
        for index in (0, len(specs)-1):
            q, lo, hi, events = specs[index]
            if events and events[0]-lo >= 6:
                choices.append(((lo+2-q)//2, (lo+2+q)//2))
        if not choices:
            raise ValueError("no room for a pre-valve near miss")
        r, c = choices[int(rng.integers(len(choices)))]
        missing = int(rng.integers(4))
        ink_cells.add((r, c))
        ink_cells.update((r+dr, c+dc) for i, (dr, dc) in enumerate(CARDINALS) if i != missing)
    if extra:
        for r, c in valves:
            if rng.random() < .40:
                # Extensions and diagonals do not invalidate a complete plus.
                dr, dc = CARDINALS[int(rng.integers(4))]
                for k in range(2, int(rng.integers(3, 6))):
                    p = r+k*dr, c+k*dc
                    if p not in rail_cells:
                        ink_cells.add(p)
            if rng.random() < .32:
                dr, dc = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
                p = r+dr, c+dc
                if p not in endpoints:
                    ink_cells.add(p)
        # Detached T, L, X or hollow ring: fewer than four cardinal neighbors.
        if rng.random() < .5:
            kind = int(rng.integers(4))
            path = paths[int(rng.integers(len(paths)))]
            r, c = path[int(rng.integers(len(path)))]
            r += int(rng.choice((-4, 4)))
            c += int(rng.choice((-4, 4)))
            offsets = (((0, 0), (0, 1), (1, 0)),
                       ((0, -1), (0, 0), (0, 1), (1, 0)),
                       ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)),
                       tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)))[kind]
            ink_cells.update((r+i, c+j) for i, j in offsets if (r+i, c+j) not in rail_cells)
    if ink_cells & endpoints:
        raise ValueError("rail endpoints must retain their rail color")
    all_cells = rail_cells | ink_cells
    r0, c0 = min(r for r, c in all_cells), min(c for r, c in all_cells)
    h = max(r for r, c in all_cells)-r0+1
    w = max(c for r, c in all_cells)-c0+1
    if h > 30 or w > 30:
        raise ValueError("scene too large")
    dims = [int(rng.integers(max(span, canvas_min), min(30, max(span, canvas_min)+6)+1))
            for span in (h, w)]
    if rng.random() < .2:
        dims[int(rng.integers(2))] = 30
    offsets = [int(rng.integers(size-span+1)) for size, span in zip(dims, (h, w))]
    if rng.random() < .45:
        axis = int(rng.integers(2))
        offsets[axis] = int(rng.choice((0, dims[axis]-(h, w)[axis])))
    a = np.full(dims, bg, dtype=np.int8)
    for r, c in rail_cells:
        a[r-r0+offsets[0], c-c0+offsets[1]] = rail_color
    for r, c in ink_cells:
        a[r-r0+offsets[0], c-c0+offsets[1]] = ink
    planted = {(r-r0+offsets[0], c-c0+offsets[1]) for r, c in valves}
    if slope == -1:
        a = a[:, ::-1].copy()
        planted = {(r, a.shape[1]-1-c) for r, c in planted}
    info = _inspect(a)
    if info["centers"] != planted:
        raise ValueError("clutter created an unintended complete valve")
    b = a.copy()
    b[_fill_mask(a, info)] = ink
    return {"input": a, "output": b}


def _aligned(q, t):
    return t + ((q-t) % 2)


def _specs(rng, profile, shift):
    if profile == "simple":
        qs = [0, int(rng.integers(5, 11))]
        result = []
        for q in qs:
            lo = _aligned(q, -4 + int(rng.integers(-1, 2)))
            hi = lo + 2*int(rng.integers(4, 9))
            events = [lo+2]
            if rng.random() < .4:
                events.append(hi-2)
            result.append((q, lo, hi, events))
        return result
    if profile == "closed":
        count = int(rng.integers(3, 5))
    elif profile == "reopen":
        count = int(rng.integers(2, 5))
    elif profile == "test" and shift == "rails":
        count = int(rng.integers(5, 7))
    else:
        count = int(rng.integers(2, 5))
    gap_max = 12 if count <= 3 else (8 if count == 4 else 6)
    qs = [0]
    for _ in range(count-1):
        qs.append(qs[-1]+int(rng.integers(5, gap_max+1)))
    result = []
    # Control phases, lengths and spacings are sampled afresh in every scene.
    for i, q in enumerate(qs):
        if profile == "closed":
            events = [_aligned(q, int(rng.integers(-6, -2))),
                      _aligned(q, int(rng.integers(3, 8)))] if i in (0, count-1) else []
        elif profile == "reopen" and i in (0, count-1):
            events = [_aligned(q, t+int(rng.integers(-1, 2))) for t in (-8, -2, 4)]
            if rng.random() < .65:
                events[1] = events[0]+2  # Two adjacent centers share ink arms.
            if rng.random() < .35:
                events.append(_aligned(q, int(rng.integers(9, 12))))
        else:
            maximum = 5 if profile == "test" and shift == "switches" else 4
            m = maximum if profile == "test" and shift == "switches" and i in (0, count-1) else int(rng.integers(maximum+1))
            if m:
                step = int(rng.integers(3, 6)) if m <= 3 else int(rng.integers(3, 5))
                phase = int(rng.integers(-8, 2))
                events = sorted({_aligned(q, phase+step*k) for k in range(m)})
            else:
                events = []
        if events:
            lo = events[0] - 2*int(rng.integers(2, 5))
            hi = events[-1] + 2*int(rng.integers(2, 5))
        else:
            lo = _aligned(q, int(rng.integers(-16, -7)))
            hi = _aligned(q, int(rng.integers(8, 17)))
        # Some rails have short margins; all endpoints are visible and unchanged.
        if profile not in ("closed", "reopen") and events and rng.random() < .3:
            lo = events[0]-2
        result.append((q, lo, hi, events))
    return result


def _make_pair(rng, profile, shift, roles, slope, seen, required=()):
    for _ in range(160):
        try:
            p = _render(rng, _specs(rng, profile, shift), roles, slope,
                        extra=bool(rng.random() < (.35 if profile == "simple" else .8)),
                        decoy=profile == "closed")
        except ValueError:
            continue
        if _key(p["input"]) in seen or (profile != "free" and not np.any(p["input"] != p["output"])):
            continue
        if not set(required) <= _witnesses(p):
            continue
        return p
    # A bounded fallback preserves the requested evidence. Deterministic
    # placement enumeration supplies more choices than the fourteen pair slots.
    if profile == "closed":
        specs = [(0, -12, 12, [-4, 4]), (6, -12, 12, []), (12, -12, 12, [-4, 4])]
    elif profile == "reopen":
        specs = [(0, -14, 16, [-8, -6, 4]), (8, -14, 16, [-8, -6, 4])]
    elif profile == "simple":
        specs = [(0, -4, 8, [-2]), (6, -4, 8, [-2])]
    elif shift == "switches" and profile == "test":
        specs = [(0, -14, 20, [-10, -4, 2, 8, 14]), (8, -14, 20, [-10, -4, 2, 8, 14])]
    else:
        count = 5 if profile == "test" and shift == "rails" else 3
        specs = [(6*i, -12, 12, [-4, 4] if i in (0, count-1) else []) for i in range(count)]
    p = _render(rng, specs, roles, slope, extra=False,
                decoy=profile == "closed", canvas_min=28)
    bg, _, ink = roles
    rr, cc = np.where(p["input"] != bg)
    cropped = p["input"][rr.min():rr.max()+1, cc.min():cc.max()+1]
    for top in range(31-cropped.shape[0]):
        for left in range(31-cropped.shape[1]):
            a = np.full((30, 30), bg, dtype=np.int8)
            a[top:top+cropped.shape[0], left:left+cropped.shape[1]] = cropped
            if _key(a) in seen:
                continue
            info = _inspect(a)
            b = a.copy()
            b[_fill_mask(a, info)] = ink
            pair = {"input": a, "output": b}
            if set(required) <= _witnesses(pair, info):
                return pair
    raise RuntimeError("bounded witness construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return distinct examples of one fixed, input-decodable rule."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    base_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    palette_mode = str(rng.choice(("stable", "rotate", "free")))
    mixed_slopes = bool(rng.random() < .65)
    slope = int(rng.choice((-1, 1)))
    shift = str(rng.choice(("rails", "switches")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if palette_mode == "stable":
                roles = base_roles
            elif palette_mode == "rotate":
                k = i % 3 if split == "train" else int(rng.integers(3))
                roles = base_roles[k:] + base_roles[:k]
            else:
                if split == "train" and i == 1:
                    first_ink = _inspect(puzzle["train"][0]["input"])["ink"]
                    ink = int(rng.choice([c for c in range(10) if c != first_ink]))
                    bg, rail = map(int, rng.choice([c for c in range(10) if c != ink], 2, replace=False))
                    roles = bg, rail, ink
                else:
                    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
            direction = slope if not mixed_slopes else (slope*(-1 if i % 2 else 1) if split == "train" else int(rng.choice((-1, 1))))
            if split == "test":
                profile, required = "test", ()
            elif i == 0:
                profile = "closed"
                required = ("ungated", "signal_sites", "first_tail", "consecutive", "pixel_centers", "horizontal")
            elif i == 1:
                profile, required = "reopen", ("first_pair", "reverse", "discard_odd", "one_per_component")
            else:
                profile, required = str(rng.choice(("simple", "closed", "reopen", "free"),
                                                   p=(.35, .20, .15, .30))), ()
            pair = _make_pair(rng, profile, shift, roles, direction, seen, required)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    witnesses = {"train": set(), "test": set()}
    palettes = set()
    positive_inks = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _inspect(a)
            expected = _audit_mask(a, info)
            assert np.array_equal(a != b, expected), "wrong valve-controlled fill mask"
            assert np.all(b[expected] == info["ink"]), "fill must copy the valve ink"
            assert np.array_equal(a[~expected], b[~expected]), "preserve all other cells"
            assert np.array_equal(solve(a), b)
            palettes.add((info["bg"], info["rail"], info["ink"]))
            witnesses[split].update(_witnesses(pair, info))
            if split == "train" and expected.any():
                positive_inks.add(info["ink"])
    assert positive_inks, "positive training is required"
    assert set(ALTERNATIVES) <= witnesses["train"], "missing rule-disambiguating training evidence"
    assert witnesses["test"] <= witnesses["train"], "test relies on an untaught distinction"
    if len(palettes) > 1:
        assert len(positive_inks) >= 2, "changing palettes need two demonstrated ink colors"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded geometric regressions and observed variation over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, structures, colors, counts, valves, slopes, lengths, gaps = (set() for _ in range(8))
    edge, near_miss, reopened, odd_tail = 0, 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _inspect(a)
                shapes.add(a.shape)
                role_grid = np.where(a == info["bg"], 0, np.where(a == info["rail"], 1, 2))
                structures.add(_key(role_grid))
                colors.add((info["bg"], info["rail"], info["ink"]))
                counts.add(len(info["rails"]))
                slopes.add(info["slope"])
                near_miss += "signal_sites" in _witnesses(pair, info)
                edge += bool(np.any(a[0] != info["bg"]) or np.any(a[-1] != info["bg"])
                             or np.any(a[:, 0] != info["bg"]) or np.any(a[:, -1] != info["bg"]))
                for rail in info["rails"]:
                    valves.add(len(rail["valves"]))
                    lengths.add((rail["hi"]-rail["lo"])//2+1)
                    reopened += len(rail["valves"]) >= 3
                    odd_tail += len(rail["valves"]) % 2
                gaps.update(y["q"]-x["q"] for x, y in zip(info["rails"], info["rails"][1:]))
    assert len(shapes) >= 45 and any(30 in s for s in shapes)
    assert len(structures) >= 180 and len(colors) >= 30
    assert {2, 3, 4, 5, 6} <= counts and {0, 1, 2, 3, 4, 5} <= valves
    assert slopes == {-1, 1} and len(lengths) >= 12 and len(gaps) >= 5
    assert min(edge, near_miss, reopened, odd_tail) >= 20
    # Exhaust all 3x3 ink neighborhoods: only the five required bits matter.
    for bits in range(512):
        mask = np.array([(bits >> j) & 1 for j in range(9)], dtype=bool).reshape(3, 3)
        assert bool(_centers(mask)[1, 1]) == all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
    # Exact switch/end behavior, including odd tails and inclusive closure.
    assert _windows(dict(valves=[], hi=20)) == []
    assert _windows(dict(valves=[2, 8, 12], hi=20)) == [(2, 8), (12, 20)]
    assert _windows(dict(valves=[2, 8, 12, 16], hi=20)) == [(2, 8), (12, 16)]
    # A pure geometric fixture has ink valves at t=-4 and t=4 and inert middle
    # rail. Verify specific between-pixel and bypass cells analytically.
    rng = np.random.default_rng(123)
    fixture = _render(rng, [(0, -10, 10, [-4, 4]), (6, -10, 10, []),
                            (12, -10, 10, [-4, 4])], (0, 5, 2), 1, extra=False)
    a, info = fixture["input"], _inspect(fixture["input"])
    rr, cc = np.indices(a.shape)
    q, t = cc-rr, rr+cc
    outer = (info["rails"][0], info["rails"][-1])
    lo, hi = outer[0]["valves"]
    expected = (a == 0) & (lo <= t) & (t <= hi) & (q > outer[0]["q"]) & (q < outer[1]["q"])
    assert np.array_equal(expected, a != fixture["output"])
    assert np.any(expected & ((t-outer[0]["q"]) % 2 == 1))
    assert "consecutive" in _witnesses(fixture)
    # The input-only solver also supports simple, dense and unchanged members
    # outside the two required teaching profiles.
    simple = _render(rng, [(0, -4, 4, [-2]), (6, -4, 4, [-2])],
                     (7, 0, 9), -1, extra=False)
    assert np.array_equal(solve(simple["input"]), simple["output"])
    dense = _render(rng, [(0, -8, 8, [-4, -2, 0]), (8, -8, 8, [-4, -2, 0])],
                    (3, 7, 0), 1, extra=False)
    assert all(len(r["valves"]) == 3 for r in _inspect(dense["input"])["rails"])
    assert np.array_equal(solve(dense["input"]), dense["output"])
    inactive = _render(rng, [(0, -12, 12, [-8, -4]), (8, -12, 12, [4, 8])],
                       (0, 5, 2), 1, extra=False)
    assert np.array_equal(inactive["input"], inactive["output"])
    assert np.array_equal(solve(inactive["input"]), inactive["output"])
    clipped = np.zeros((4, 5), dtype=bool)
    clipped[0, 1:4] = clipped[1, 2] = clipped[3, 2] = True
    assert not _centers(clipped).any(), "do not wrap across the canvas boundary"
    # All count combinations are cheap enough to exercise here as well.
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(90817, n, m)
            assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.5, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(7, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Wrong labels and inadequate demonstrations must not pass validation.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    test = bad["test"][0]
    r, c = next((int(r), int(c)) for r, c in np.argwhere(test["input"] != _inspect(test["input"])["bg"]))
    test["output"][r, c] = (int(test["output"][r, c])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted a changed input marker")
    mapping = {0: 7, 5: 3, 2: 4}
    second = {key: np.vectorize(mapping.__getitem__, otypes=[np.int8])(value)
              for key, value in fixture.items()}
    inadequate = {"train": [fixture, second], "test": [puzzles[0]["test"][0]]}
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as error:
        assert "training evidence" in str(error)
    else:
        raise AssertionError("accepted training without the switch witnesses")
