"""Resonant diagonal bridges: whole-word phase gates perpendicular broadcasts.

Only Python/NumPy are used. Generation and tests perform no filesystem I/O.
The solver reconstructs geometry and roles solely from its argument.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must be integer")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _add(p, d, t=1):
    return p[0] + t*d[0], p[1] + t*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _diagonal(a, b):
    return a != b and abs(a[0]-b[0]) == abs(a[1]-b[1])


def _line(p, d, shape):
    while _inside(_add(p, d, -1), shape):
        p = _add(p, d, -1)
    cells = []
    while _inside(p, shape):
        cells.append(p)
        p = _add(p, d)
    return cells


def _plus(center, vr, hr):
    return ({_add(center, (1, 0), t) for t in range(-vr, vr+1)} |
            {_add(center, (0, 1), t) for t in range(-hr, hr+1)})


def _components(cells):
    remaining, groups = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            q = stack.pop()
            for d in CARDINALS:
                nxt = _add(q, d)
                if nxt in remaining:
                    remaining.remove(nxt)
                    group.add(nxt)
                    stack.append(nxt)
        groups.append(group)
    return groups


def _plus_parameters(cells):
    rows, cols = zip(*cells)
    if (max(rows)-min(rows)) % 2 or (max(cols)-min(cols)) % 2:
        return None
    vr, hr = (max(rows)-min(rows))//2, (max(cols)-min(cols))//2
    center = ((min(rows)+max(rows))//2, (min(cols)+max(cols))//2)
    if vr in (1, 2) and hr in (1, 2) and cells == _plus(center, vr, hr):
        return center, vr, hr
    return None


def _merge(writes, p, color):
    color = int(color)
    _require(p not in writes or writes[p] == color, "conflicting constructions")
    writes[p] = color


def _parse_marker(grid, bg, marker):
    groups = _components(map(tuple, np.argwhere(grid == marker)))
    sources, targets = [], []
    for cells in groups:
        if len(cells) == 1:
            targets.append(next(iter(cells)))
        else:
            params = _plus_parameters(cells)
            _require(params is not None, "marker components must be pluses or points")
            sources.append((params, cells))
    _require(1 <= len(sources) <= 3 and targets, "missing sources or endpoints")
    carriers, claimed_words, target_degrees = [], set(), {p: 0 for p in targets}
    all_hubs = set().union(*(cells for _, cells in sources))
    interior_union = set()
    for (center, vr, hr), hub in sources:
        partners = [q for q in targets if _diagonal(center, q)]
        _require(len(partners) == 1, "source must have a unique endpoint")
        endpoint = partners[0]
        target_degrees[endpoint] += 1
        length = abs(endpoint[0]-center[0])
        _require(length >= 3, "degenerate bridge")
        d = tuple(1 if endpoint[j] > center[j] else -1 for j in range(2))
        attachments = []
        for wd in CARDINALS:
            arm = vr if wd[0] else hr
            p = _add(center, wd, arm+1)
            if not _inside(p, grid.shape) or int(grid[p]) in (bg, marker):
                continue
            word, coords = [], []
            while _inside(p, grid.shape) and int(grid[p]) not in (bg, marker):
                word.append(int(grid[p]))
                coords.append(p)
                p = _add(p, wd)
            if _inside(p, grid.shape):
                _require(int(grid[p]) == bg, "word ends at a structural marker")
            attachments.append((wd, tuple(word), set(coords)))
        _require(len(attachments) == 1, "source needs one word")
        wd, word, word_cells = attachments[0]
        _require(1 <= len(word) <= 5, "word length outside family")
        _require(wd[0]*d[0]+wd[1]*d[1] < 0, "word must point away from endpoint")
        _require(not claimed_words & word_cells, "shared seed word")
        claimed_words.update(word_cells)
        interior = tuple(_add(center, d, t) for t in range(1, length))
        _require(not interior_union.intersection(interior), "carrier interiors intersect")
        interior_union.update(interior)
        carriers.append(dict(center=center, vr=vr, hr=hr, hub=hub, endpoint=endpoint,
                             length=length, direction=d, word_direction=wd, word=word,
                             word_cells=word_cells, interior=interior))
    _require(all(n <= 1 for n in target_degrees.values()), "endpoint has two partners")
    _require(not interior_union & (all_hubs | set(targets) | claimed_words), "bridge hits original structure")
    foreground = {tuple(map(int, p)): int(grid[tuple(p)]) for p in np.argwhere(grid != bg)}
    probes = {p: c for p, c in foreground.items() if c != marker and p not in claimed_words}
    for carrier in carriers:
        active, mismatches = [], []
        for t, p in enumerate(carrier["interior"], 1):
            if p in probes:
                (active if probes[p] == carrier["word"][(t-1) % len(carrier["word"])]
                 else mismatches).append((t, p, probes[p]))
        carrier["active"], carrier["mismatches"] = active, mismatches
    return dict(background=bg, marker=marker, carriers=carriers, probes=probes,
                foreground=foreground, interior_union=interior_union,
                unpaired={p for p, n in target_degrees.items() if n == 0})


def _predict_analysis(grid, a):
    writes = {}
    for c in a["carriers"]:
        word = c["word"]
        for t, p in enumerate(c["interior"], 1):
            if p not in a["foreground"]:
                _merge(writes, p, word[(t-1) % len(word)])
        dr, dc = c["direction"]
        for _, p, color in c["active"]:
            for q in _line(p, (dr, -dc), grid.shape):
                _require(q not in a["foreground"] or a["foreground"][q] == color,
                         "broadcast conflicts with original material")
                _merge(writes, q, color)
    out = grid.copy()
    for p, color in writes.items():
        out[p] = color
    a["writes"] = writes
    return out


def _analyze(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(2*int(counts[bg]) > grid.size, "background must be a strict majority")
    interpretations = []
    for marker in sorted(set(map(int, np.unique(grid))) - {bg}):
        try:
            a = _parse_marker(grid, bg, marker)
            out = _predict_analysis(grid, a)
            interpretations.append((out, a))
        except ValueError:
            continue
    _require(len(interpretations) == 1, "ambiguous or invalid geometric roles")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Evaluate the fixed family rule using only the input's visible cells."""
    return _analyze(input_grid)[0]


def _word(rng, palette, size, style=None):
    a, b = [int(x) for x in rng.choice(palette, size=2, replace=False)]
    if style == "whole":
        return (a, b, a)
    if style == "phase":
        return (a, b)
    values = tuple(int(x) for x in rng.choice(palette, size=size))
    if size > 1 and rng.random() < .35:
        values = tuple(a if j % 2 == 0 else b for j in range(size))
    if style == "primitive" and _period(values) != size:
        values = (a,)*(size-1)+(b,)
    return values


def _period(word):
    return next(p for p in range(1, len(word)+1)
                if len(word) % p == 0 and all(c == word[j % p] for j, c in enumerate(word)))


def _make_spec(shape, center, vr, hr, direction, wd, word, length, on_probes, beyond=False):
    hub = _plus(center, vr, hr)
    arm = vr if wd[0] else hr
    word_map = {_add(center, wd, arm+1+j): int(c) for j, c in enumerate(word)}
    endpoint = _add(center, direction, length)
    interior = {_add(center, direction, t): int(word[(t-1) % len(word)])
                for t in range(1, length)}
    probes = {_add(center, direction, t): int(c) for t, c in on_probes.items()}
    if beyond:
        probes[_add(center, direction, length+1)] = int(word[length % len(word)])
    points = hub | set(word_map) | {endpoint} | set(interior) | set(probes)
    _require(all(_inside(p, shape) for p in points), "construction outside grid")
    _require(not (set(word_map) | hub | {endpoint}) & set(probes), "probe overlaps structure")
    return dict(center=center, endpoint=endpoint, hub=hub, word_map=word_map,
                direction=direction, word=word, length=length, interior=interior,
                probes=probes)


def _render(shape, specs, bg, marker, extra=None):
    """Construction renderer; unlike solve it receives sampled geometric objects."""
    original, writes, bridges = {}, {}, set()
    centers, targets = [s["center"] for s in specs], [s["endpoint"] for s in specs]
    for i, center in enumerate(centers):
        _require([j for j, p in enumerate(targets) if _diagonal(center, p)] == [i],
                 "accidental diagonal endpoint partner")
    for s in specs:
        _require(not bridges & set(s["interior"]), "intersecting bridges")
        bridges.update(s["interior"])
        cells = {p: marker for p in s["hub"] | {s["endpoint"]}}
        cells.update(s["word_map"])
        cells.update(s["probes"])
        _require(not set(cells) & set(original), "overlapping original objects")
        original.update(cells)
    if extra:
        _require(not set(original) & set(extra), "extra dot overlaps original")
        original.update(extra)
    structures = set().union(*(s["hub"] | {s["endpoint"]} | set(s["word_map"]) for s in specs))
    _require(not bridges & structures, "bridge hits structure")
    for s in specs:
        for p, color in s["interior"].items():
            if p not in original:
                _merge(writes, p, color)
        dr, dc = s["direction"]
        for p, color in s["probes"].items():
            if p in s["interior"] and color == s["interior"][p]:
                for q in _line(p, (dr, -dc), shape):
                    _require(q not in original or original[q] == color, "line hits conflicting original")
                    _merge(writes, q, color)
    grid = np.full(shape, bg, dtype=np.int8)
    for p, color in original.items():
        grid[p] = color
    output = grid.copy()
    for p, color in writes.items():
        output[p] = color
    return {"input": grid, "output": output}, writes, structures, bridges


def _candidate(rng, shape, palette, style, forced_size, bounds, slope, beyond, quiet):
    h, w = shape
    vr, hr = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    dr = int(rng.choice((-1, 1)))
    dc = dr*int(slope if slope is not None else rng.choice((-1, 1)))
    direction = (dr, dc)
    wd = (-dr, 0) if rng.random() < .5 else (0, -dc)
    size = forced_size if forced_size else int(rng.integers(1, 4))
    word = _word(rng, palette, size, style or ("primitive" if forced_size and size >= 4 else None))
    low, high = bounds
    length = int(rng.integers(low, high+1))
    # Sample length before placement: free centers otherwise bias long bridges
    # toward the shortest feasible tail of their requested range.
    lower, upper = [vr, hr], [h-1-vr, w-1-hr]
    for j, step in enumerate(direction):
        if step > 0:
            upper[j] = min(upper[j], shape[j]-1-length-int(beyond))
        else:
            lower[j] = max(lower[j], length+int(beyond))
    for j, step in enumerate(wd):
        radius = vr if j == 0 else hr
        if step < 0:
            lower[j] = max(lower[j], radius+len(word))
        elif step > 0:
            upper[j] = min(upper[j], shape[j]-1-radius-len(word))
    _require(all(lower[j] <= upper[j] for j in range(2)), "insufficient bridge space")
    center = tuple(int(rng.integers(lower[j], upper[j]+1)) for j in range(2))
    if style == "whole":
        on = {2: word[1], 4: word[0], 5: word[0]}
    elif style == "phase":
        on = {3: word[0], 4: word[0]}
    else:
        positions = [int(t) for t in rng.permutation(np.arange(2, length-1))]
        active_n = 0 if quiet else min(len(positions), int(rng.choice((0, 1, 1, 2, 3))))
        mismatch_n = min(len(positions)-active_n, int(rng.choice((0, 0, 1, 2))))
        on = {t: word[(t-1) % len(word)] for t in positions[:active_n]}
        for t in positions[active_n:active_n+mismatch_n]:
            on[t] = int(rng.choice([c for c in palette if c != word[(t-1) % len(word)]]))
    return _make_spec(shape, center, vr, hr, direction, wd, word, length, on, beyond)


def _add_extras(rng, pair, writes, structures, bridges, specs, bg, marker, palette):
    extra = {}
    # A collar prevents probes from impersonating an attachment or extending a word.
    collar = {_add(p, d) for p in structures for d in CARDINALS} | structures
    count = int(rng.choice((0, 0, 1, 2, 3, 4)))
    want_marker = bool(rng.random() < .3)
    h, w = pair["input"].shape
    for index in rng.permutation(h*w):
        p = (int(index)//w, int(index) % w)
        if p in collar or p in bridges or int(pair["input"][p]) != bg:
            continue
        if any(_add(p, d) in extra for d in CARDINALS):
            continue
        if count:
            color = int(rng.choice(palette))
            if p in writes and writes[p] != color:
                continue
            extra[p] = color
            count -= 1
        elif want_marker:
            if p in writes or any(_diagonal(p, s["center"]) for s in specs):
                continue
            extra[p] = marker
            want_marker = False
        else:
            break
    return extra


def _fallback(rng, bg, marker, palette, count, style, forced_size, bounds, slope):
    """Finite construction with nonintersecting carriers and agreeing broadcasts."""
    word = _word(rng, palette, forced_size or 3,
                 style or ("primitive" if forced_size and forced_size >= 4 else None))
    length = 18 if bounds[0] >= 14 else 8
    placements = (((6, 6), (-1, 0)), ((21, 8), (0, -1)), ((8, 21), (-1, 0)))
    specs = []
    for center, wd in placements[:count]:
        on = {2: word[1 % len(word)], 4: word[3 % len(word)]}
        expected = word[4 % len(word)]
        on[5] = next(int(c) for c in palette if c != expected)
        if style == "whole":
            on = {2: word[1], 4: word[0], 5: word[0]}
        specs.append(_make_spec((30, 30), center, 1, 1, (1, 1), wd, word,
                                length, on, beyond=not specs and style == "whole"))
    pair = _render((30, 30), specs, bg, marker)[0]
    turns = int(rng.integers(4))
    mirror = bool(rng.integers(2)) if slope is None else ((-1)**turns != slope)
    result = {}
    for k, g in pair.items():
        a = np.rot90(g, turns)
        result[k] = np.ascontiguousarray(np.fliplr(a) if mirror else a)
    return result


def _pair(rng, count, style, forced_size, bounds, slope, bg_fixed=None):
    colors = [int(c) for c in rng.permutation(10)]
    if bg_fixed is not None:
        colors.remove(bg_fixed)
        colors.insert(0, bg_fixed)
    bg, marker = colors[:2]
    palette = colors[2:2+int(rng.integers(2, 6))]
    quiet = style is None and rng.random() < .23
    for scene_attempt in range(16):
        low = 23 if bounds[0] >= 14 else 20 if count == 3 else 14
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        if scene_attempt >= 10:
            shape = (30, 30)
        specs = []
        for i in range(count):
            for _ in range(100):
                try:
                    candidate = _candidate(rng, shape, palette, style if i == 0 else None,
                                           forced_size if i == 0 else None, bounds,
                                           slope if i == 0 else None,
                                           style == "whole" and i == 0, quiet)
                    _render(shape, specs+[candidate], bg, marker)
                    specs.append(candidate)
                    break
                except ValueError:
                    continue
            else:
                break
        if len(specs) != count:
            continue
        try:
            pair, writes, structures, bridges = _render(shape, specs, bg, marker)
            extra = _add_extras(rng, pair, writes, structures, bridges, specs, bg, marker, palette)
            pair = _render(shape, specs, bg, marker, extra)[0]
            expected, analysis = _analyze(pair["input"])
            _require(np.array_equal(expected, pair["output"]), "sampled construction differs from input")
            if style == "whole":
                evidence = _features(pair["input"], analysis)
                _require(evidence["phase_witness"] and evidence["beyond"], "occluded teaching witness")
            return pair
        except ValueError:
            continue
    return _fallback(rng, bg, marker, palette, count, style, forced_size, bounds, slope)


def _key(grid):
    return tuple(grid.shape), np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    # At most thirteen exclusions, but fifteen distinct injective recolorings.
    visible = sorted(map(int, np.unique(pair["input"])))
    for values in itertools.islice(itertools.permutations(range(10), len(visible)), 15):
        lut = np.arange(10, dtype=np.int8)
        lut[visible] = values
        candidate = {k: lut[g] for k, g in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise AssertionError("finite uniqueness repair exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Create a deterministic puzzle; explicit counts are strict integers."""
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(3))  # count, word, length
    slopes = [int(x) for x in rng.permutation((-1, 1))]
    bg_fixed = int(rng.integers(10)) if rng.random() < .25 else None
    puzzle, used = {"train": [], "test": []}, set()
    for split, n in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(n):
            style = ("whole" if i == 0 else "phase" if i == 1 else None) if split == "train" else None
            forced_size = 3 if style == "whole" else 2 if style == "phase" else None
            count = int(rng.choice((1, 1, 2)))
            bounds = (8, 11) if style else (6, 11)
            slope = slopes[i] if style else None
            if split == "test":
                if regime == 0:
                    count = 3
                    bounds = (6, 13)
                elif regime == 1:
                    forced_size = int(rng.integers(4, 6))
                else:
                    count, bounds = 1, (14, 24)
            pair = _unique(_pair(rng, count, style, forced_size, bounds, slope, bg_fixed), used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid, a):
    carriers = a["carriers"]
    active = {p for c in carriers for _, p, _ in c["active"]}
    on = a["interior_union"]
    off = set(a["probes"]) - on
    beyond = False
    phase_witness = False
    nonconstant_fill = False
    branch_lines = [(p, set(_line(p, (c["direction"][0], -c["direction"][1]), grid.shape)))
                    for c in carriers for _, p, _ in c["active"]]

    def extra_line_changes(point, direction, color):
        return any(int(grid[q]) == a["background"] and a["writes"].get(q, a["background"]) != color
                   for q in _line(point, (direction[0], -direction[1]), grid.shape))

    def exclusive_branch(point):
        own = next(cells for p, cells in branch_lines if p == point)
        others = set().union(*(cells for p, cells in branch_lines if p != point))
        return any(int(grid[q]) == a["background"] and q not in on and q not in others for q in own)

    for c in carriers:
        word, center, d, length = c["word"], c["center"], c["direction"], c["length"]
        p = _add(center, d, length+1)
        beyond |= (p in off and a["probes"][p] == word[length % len(word)]
                   and p not in a["writes"] and extra_line_changes(p, d, a["probes"][p]))
        for t, q in enumerate(c["interior"], 1):
            if q not in a["foreground"] and word[(t-1) % len(word)] != word[0]:
                nonconstant_fill = True
        if len(word) == 3 and word[0] == word[2] != word[1]:
            decisions = {t: color == word[(t-1) % 3]
                         for t, q in enumerate(c["interior"], 1)
                         if (color := a["probes"].get(q)) is not None}
            phase_witness |= (decisions.get(2) is True and decisions.get(4) is True
                              and decisions.get(5) is False
                              and exclusive_branch(_add(center, d, 4))
                              and extra_line_changes(_add(center, d, 5), d, word[0]))
    return dict(shape=tuple(grid.shape), count=len(carriers),
                word_lengths=tuple(len(c["word"]) for c in carriers),
                periods=tuple(_period(c["word"]) for c in carriers),
                words=tuple(c["word"] for c in carriers),
                lengths=tuple(c["length"] for c in carriers),
                slopes={c["direction"][0]*c["direction"][1] for c in carriers if c["active"]},
                directions={c["direction"] for c in carriers},
                attachments={c["word_direction"] for c in carriers},
                arms={(c["vr"], c["hr"]) for c in carriers},
                active=len(active), mismatch=sum(len(c["mismatches"]) for c in carriers),
                off=len(off), unpaired=len(a["unpaired"]), beyond=beyond,
                phase_witness=phase_witness, nonconstant_fill=nonconstant_fill,
                background=a["background"], marker=a["marker"])


def validate_generated_puzzle(puzzle) -> None:
    """Reject malformed geometry, wrong outputs, and inadequate demonstrations."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, analysis = _analyze(inp)
            _require(np.array_equal(expected, out), "incorrect output")
            _require(not np.array_equal(inp, out), "no visible construction")
            _require(np.array_equal(out[inp != analysis["background"]], inp[inp != analysis["background"]]),
                     "original material changed")
            if split == "train":
                train.append(_features(inp, analysis))
    _require(set().union(*(f["slopes"] for f in train)) == {-1, 1}, "teach active bridges of both slopes")
    _require(any(f["phase_witness"] for f in train), "teach whole-word matching and mismatching")
    _require(any(f["beyond"] for f in train), "teach finite eligibility")
    _require(any(f["nonconstant_fill"] for f in train), "teach periodic bridge colors")


def self_test() -> None:
    """Sequential 42-seed variation checks; no stored puzzles, files, or traces."""
    assert _period((2, 3, 2)) == 3 and _period((2, 3, 2, 3)) == 2
    # Hand-specified semantic anchors distinguish resonance, finite eligibility,
    # complete ABA words, and activation from original cells only.
    anchor = np.zeros((22, 24), dtype=np.int8)
    for p in ((6, 6), (5, 6), (7, 6), (6, 5), (6, 7), (14, 14)):
        anchor[p] = 1
    for p, color in { (4, 6): 2, (3, 6): 3, (2, 6): 2,
                      (8, 8): 3, (10, 10): 2, (11, 11): 2, (15, 15): 2}.items():
        anchor[p] = color
    predicted = solve(anchor)
    for p, color in {(0, 16): 3, (0, 20): 2, (0, 22): 0, (7, 23): 0,
                     (0, 18): 0, (1, 6): 0, (7, 7): 2, (11, 11): 2,
                     (14, 14): 1, (15, 15): 2}.items():
        assert int(predicted[p]) == color
    for turns in range(4):
        for mirror in (False, True):
            def transformed(g):
                q = np.rot90(g, turns)
                return np.fliplr(q) if mirror else q
            assert np.array_equal(solve(transformed(anchor)), transformed(predicted))
    lut = np.array([8, 0, 7, 4, 9, 3, 2, 1, 6, 5], dtype=np.int8)
    assert np.array_equal(solve(lut[anchor]), lut[predicted])
    for point, color in (((16, 16), 1), ((0, 20), 4)):
        invalid = anchor.copy()
        invalid[point] = color
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous pairing or conflicting original accepted")
    shapes, counts, sizes, lengths, directions, attachments, arms, roles = (set() for _ in range(8))
    signatures, flags, shifts = set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        records = {"train": [], "test": []}
        for split in records:
            for pair in p[split]:
                for g in pair.values():
                    digest.update(bytes(g.shape))
                    digest.update(g.tobytes())
                _, a = _analyze(pair["input"])
                f = _features(pair["input"], a)
                records[split].append(f)
                shapes.add(f["shape"])
                counts.add(f["count"])
                sizes.update(f["word_lengths"])
                lengths.update(f["lengths"])
                directions.update(f["directions"])
                attachments.update(f["attachments"])
                arms.update(f["arms"])
                roles.add((f["background"], f["marker"]))
                for flag in ("active", "mismatch", "off", "unpaired", "beyond", "phase_witness"):
                    if f[flag]:
                        flags.add(flag)
                if f["active"] == 0:
                    flags.add("untriggered")
        signatures.add(digest.digest())
        tr, te = records["train"], records["test"]
        if max(f["count"] for f in te) > max(f["count"] for f in tr):
            shifts.add("count")
        if max(max(f["periods"]) for f in te) > max(max(f["periods"]) for f in tr):
            shifts.add("word")
        if min(min(f["lengths"]) for f in te) > max(max(f["lengths"]) for f in tr):
            shifts.add("length")
        again = generate_puzzle(seed)
        assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], again[s]) for k in x)
    assert len(signatures) == 42 and len(shapes) >= 45 and len(roles) >= 25
    assert counts == {1, 2, 3} and sizes == {1, 2, 3, 4, 5}
    assert len(lengths) >= 12 and max(lengths) >= 20
    assert len(directions) == len(attachments) == len(arms) == 4
    assert flags == {"active", "mismatch", "off", "unpaired", "beyond", "phase_witness", "untriggered"}
    assert shifts == {"count", "word", "length"}
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(20261023 + train*10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3, 1.5), (None, 1), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise the deterministic escape construction, including all long words.
    for size in range(1, 6):
        for count in (1, 2, 3):
            pair = _fallback(np.random.default_rng(700+size+count), 0, 1, [2, 3, 4],
                             count, None, size, (6, 11), None)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    fallback_puzzle = {
        "train": [_fallback(np.random.default_rng(901), 0, 1, [2, 3, 4], 2, "whole", 3, (8, 11), -1),
                  _fallback(np.random.default_rng(902), 5, 6, [7, 8, 9], 1, "phase", 2, (8, 11), 1)],
        "test": [_fallback(np.random.default_rng(903), 3, 4, [0, 1, 2], 1, None, 5, (14, 24), None)]}
    validate_generated_puzzle(fallback_puzzle)
    p = generate_puzzle(0, 2, 1)
    g = p["test"][0]["output"]
    g[0, 0] = (int(g[0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, bounded fallback checks)")
