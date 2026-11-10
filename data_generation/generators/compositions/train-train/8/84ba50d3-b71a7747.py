"""Tile seams define independent, capacity-one membrane pores.

Only NumPy and the standard library are used.  Neither solving nor validation
reads parent files, examples, a seed, or generator metadata.
"""
from itertools import combinations
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad color")
    return a.astype(np.int16, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=np.int16))
    return result


def _groups(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [(int(g[0]), int(g[-1]) + 1)
            for g in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)]


def _scene_info(g, tile_h, tile_w):
    colors, frequencies = np.unique(g, return_counts=True)
    _require(len(colors) == 3, "scene needs three colors")
    bg = int(colors[np.argmax(frequencies)])
    _require(int(frequencies.max()) * 2 > g.size, "scene background not a majority")
    candidates = []
    for axis in (0, 1):
        for i in range(g.shape[axis]):
            line = g[i, :] if axis == 0 else g[:, i]
            if int(line[0]) != bg and np.all(line == line[0]):
                candidates.append((axis, i, int(line[0])))
    _require(len(candidates) == 1, "ambiguous membrane")
    axis, membrane, wall = candidates[0]
    _require(np.count_nonzero(g == wall) == g.shape[1 - axis], "wall color outside membrane")
    ink = next(int(c) for c in colors if c not in (bg, wall))
    cells = np.argwhere(g == ink)
    if np.all(cells[:, axis] < membrane):
        sign = 1
    elif np.all(cells[:, axis] > membrane):
        sign = -1
    else:
        raise AssertionError("payload must be on one side")
    _require(0 < membrane < g.shape[axis] - 1, "empty region missing")
    return dict(bg=bg, wall=wall, ink=ink, axis=axis, sign=sign,
                membrane=membrane, pore_size=tile_w if axis == 0 else tile_h)


def _decode(input_grid):
    """Find the unique Cartesian tile array, without literal color assumptions."""
    grid = _grid(input_grid)
    colors = [int(c) for c in np.unique(grid)]
    found = []
    if not 4 <= len(colors) <= 5:
        raise ValueError("expected three scene colors and one or two scaffold colors")
    for palette in combinations(colors, 3):
        mask = np.isin(grid, palette)
        rows, cols = mask.any(axis=1), mask.any(axis=0)
        if not np.array_equal(mask, rows[:, None] & cols[None, :]):
            continue
        rg, cg = _groups(rows), _groups(cols)
        if not (2 <= len(rg) <= 5 and 2 <= len(cg) <= 5):
            continue
        rh, cw = {b - a for a, b in rg}, {b - a for a, b in cg}
        if len(rh) != 1 or len(cw) != 1:
            continue
        th, tw = next(iter(rh)), next(iter(cw))
        if not (2 <= th <= 6 and 2 <= tw <= 6):
            continue
        margins = (rg[0][0], grid.shape[0] - rg[-1][1],
                   cg[0][0], grid.shape[1] - cg[-1][1])
        gaps = ([rg[i + 1][0] - rg[i][1] for i in range(len(rg) - 1)],
                [cg[i + 1][0] - cg[i][1] for i in range(len(cg) - 1)])
        if not all(1 <= n <= 3 for n in (*margins, *gaps[0], *gaps[1])):
            continue
        packed = grid[np.ix_(rows, cols)].copy()
        try:
            info = _scene_info(packed, th, tw)
        except AssertionError:
            continue
        found.append(dict(grid=grid, packed=packed, info=info, rows=rows, cols=cols,
                          tile=(th, tw), groups=(rg, cg), margins=margins, gaps=gaps,
                          palette=palette))
    if len(found) != 1:
        raise ValueError("input must have one unique tile-and-membrane interpretation")
    return found[0]


def _profile(cells, info, global_capacity=False):
    axis, sign, size = info["axis"], info["sign"], info["pore_size"]
    levels = sorted(map(int, np.unique(cells[:, axis])), reverse=sign > 0)
    loads = []
    for level in levels:
        transverse = cells[cells[:, axis] == level, 1 - axis]
        loads.append(len(transverse) if global_capacity else
                     int(np.bincount(transverse // size).max()))
    bad = next((i for i, load in enumerate(loads) if load > 1), None)
    return dict(levels=levels, loads=loads, first_bad=bad, length=len(levels),
                outcome="pass" if bad is None else "blocked" if bad == 0 else "partial")


def _motion(packed, info, components=None, global_capacity=False):
    """Move each rigid component; record actual traversals, including punctures."""
    if components is None:
        components = _components(packed == info["ink"])
    axis, sign, wall = info["axis"], info["sign"], info["membrane"]
    delta = np.zeros(2, dtype=np.int16)
    delta[axis] = sign
    records = []
    for cells in components:
        current = cells.copy()
        holes, swept = set(), set(map(tuple, cells))
        distance = 0
        for _ in range(packed.shape[axis] + 1):
            candidate = current + delta
            if np.any(candidate < 0) or any(np.any(candidate[:, a] >= packed.shape[a])
                                           for a in (0, 1)):
                reason = "boundary"
                break
            crossing = candidate[candidate[:, axis] == wall]
            load = (len(crossing) if global_capacity else
                    int(np.bincount(crossing[:, 1 - axis] // info["pore_size"]).max())
                    if len(crossing) else 0)
            if load > 1:
                reason = "pore"
                break
            current = candidate
            distance += 1
            holes.update(map(tuple, crossing))
            swept.update(map(tuple, current))
        else:
            raise AssertionError("motion failed to terminate")
        records.append(dict(original=cells, final=current, holes=holes, swept=swept,
                            distance=distance, reason=reason,
                            **_profile(cells, info, global_capacity)))
    output = packed.copy()
    output[packed == info["ink"]] = info["bg"]
    for rec in records:
        for p in rec["holes"]:
            output[p] = info["bg"]
    for rec in records:
        output[tuple(rec["final"].T)] = info["ink"]
    return output, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    decoded = _decode(input_grid)
    return _motion(decoded["packed"], decoded["info"])[0]


def _fragment_prediction(decoded):
    """A deliberately wrong rule: never join payload across scaffold seams."""
    ri = {int(v): i for i, v in enumerate(np.flatnonzero(decoded["rows"]))}
    ci = {int(v): i for i, v in enumerate(np.flatnonzero(decoded["cols"]))}
    fragments = []
    for cells in _components(decoded["grid"] == decoded["info"]["ink"]):
        fragments.append(np.array([(ri[int(r)], ci[int(c)]) for r, c in cells], dtype=np.int16))
    return _motion(decoded["packed"], decoded["info"], fragments)[0]


def _randint(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _axis_mask(rng, count, size):
    lengths = [1] * (count + 1)
    budget = 30 - count * size - sum(lengths)
    _require(budget >= 0, "axis cannot fit")
    for _ in range(_randint(rng, 0, min(budget, 2 * len(lengths)))):
        options = [i for i, value in enumerate(lengths) if value < 3]
        lengths[int(rng.choice(options))] += 1
    length = count * size + sum(lengths)
    mask = np.zeros(length, dtype=bool)
    position = lengths[0]
    for i in range(count):
        mask[position:position + size] = True
        position += size + lengths[i + 1]
    return mask


def _encode(rng, packed, tile_h, tile_w, colors, style):
    rows = _axis_mask(rng, packed.shape[0] // tile_h, tile_h)
    cols = _axis_mask(rng, packed.shape[1] // tile_w, tile_w)
    grid = np.full((len(rows), len(cols)), colors[3], dtype=np.int16)
    rr, cc = np.indices(grid.shape)
    if style == "stripes":
        grid[np.flatnonzero(~rows)[::2], :] = colors[4]
        grid[:, np.flatnonzero(~cols)[1::2]] = colors[4]
    elif style == "markers":
        grid[(~rows[:, None] & ~cols[None, :]) & ((rr + 2 * cc) % 3 == 0)] = colors[4]
        grid[np.flatnonzero(~rows)[0], np.flatnonzero(~cols)[0]] = colors[4]
    elif style == "texture":
        grid[(rr + cc) % 3 == 0] = colors[4]
    grid[np.ix_(rows, cols)] = packed
    return grid


def _shape(rng, left, width, band, outcome, length, force_tip=False):
    """Ordered slices from the leading end, in a disjoint transverse slot."""
    right = left + width
    seams = [s for s in range(band, right, band) if left < s < right]
    if outcome == "pass":
        if seams and bool(rng.integers(2)):
            seam = int(rng.choice(seams))
            states = ((seam - 1,), (seam,), (seam - 1, seam))
            slices = [states[_randint(rng, 0, 2)]]
            for _ in range(length - 1):
                choices = [s for s in states if set(s) & set(slices[-1])]
                slices.append(choices[_randint(rng, 0, len(choices) - 1)])
            slices[_randint(rng, 0, length - 1)] = states[2]
            return slices
        x = _randint(rng, left, right - 1)
        return [(x,)] * length
    k = 0 if outcome == "blocked" else _randint(rng, 2 if force_tip else 1, length - 1)
    intervals = [tuple(range(a, b)) for a in range(left, right)
                 for b in range(a + 2, right + 1)
                 if max(sum(x // band == gate for x in range(a, b))
                        for gate in range(left // band, (right - 1) // band + 1)) >= 2]
    broad = intervals[_randint(rng, 0, len(intervals) - 1)]
    tip = int(rng.choice(broad))
    slices = [(tip,)] * k + [broad]
    for _ in range(k + 1, length):
        intervals = [tuple(range(a, b)) for a in range(left, right)
                     for b in range(a + 1, right + 1)
                     if set(range(a, b)) & set(slices[-1])]
        slices.append(intervals[_randint(rng, 0, len(intervals) - 1)])
    return slices


def _make_pair(rng, role_colors, rotation, style, special, is_test):
    max_length = 7 if is_test else 4
    if special == "matched":
        length = _randint(rng, 2, 4)
        lengths, outcomes = [length, length], ["pass", "blocked"]
    else:
        count = _randint(rng, 1, 4 if is_test else 3)
        outcomes = [str(rng.choice(("pass", "blocked", "partial"))) for _ in range(count)]
        if special == "tip":
            outcomes[0] = "partial"
        lengths = [_randint(rng, 2 if o == "partial" else 1, max_length) for o in outcomes]
        if special == "tip":
            lengths[0] = _randint(rng, 3, 4)
        if is_test:
            lengths[0] = _randint(rng, 5, 7)
    longest = max(lengths)
    for _ in range(160):
        normal_layouts = [(n, size) for n in range(2, 6) for size in range(2, 7)
                          if max(9, 2 * longest + 2) <= n * size <= (24 if is_test else 20)
                          and n * size + n + 1 <= 30]
        nr, th = normal_layouts[_randint(rng, 0, len(normal_layouts) - 1)]
        height = nr * th
        if special == "matched":
            widths = [2, 2]
            minimum_width = 9
        else:
            widths = [_randint(rng, 3, 5) for _ in outcomes]
            minimum_width = sum(widths) + len(widths) - 1
        transverse_layouts = [(n, size) for n in range(2, 6) for size in range(2, 7)
                              if max(8, minimum_width) <= n * size <= 24
                              and n * size + n + 1 <= 30
                              and (special != "matched" or (size >= 3 and n >= 3))]
        if not transverse_layouts:
            continue
        nc, tw = transverse_layouts[_randint(rng, 0, len(transverse_layouts) - 1)]
        width = nc * tw
        if special == "matched":
            locations = [(a, b) for a in range(tw - 1, width - 1, tw)
                         for b in range(width - 1) if b // tw == (b + 1) // tw
                         and (a + 2 < b or b + 2 < a)]
            a, b = locations[_randint(rng, 0, len(locations) - 1)]
            shapes = [[(a, a + 1)] * length, [(b, b + 1)] * length]
        else:
            extra = width - minimum_width
            allocation = rng.multinomial(extra, np.ones(len(widths) + 1) / (len(widths) + 1))
            left, shapes = int(allocation[0]), []
            for j, (span, outcome, length) in enumerate(zip(widths, outcomes, lengths)):
                shapes.append(_shape(rng, left, span, tw, outcome, length,
                                     special == "tip" and j == 0))
                left += span + 1 + int(allocation[j + 1])
        membrane = _randint(rng, longest + 1, height - longest - 1)
        leading = []
        for j, (shape, outcome) in enumerate(zip(shapes, outcomes)):
            lo, hi = len(shape) - 1, membrane - 1 - (outcome == "blocked")
            options = list(range(lo, hi + 1))
            if special == "matched":
                options = [x for x in options if x <= membrane - 2
                           and x // th != (x - len(shape) + 1) // th]
            elif special == "tip" and j == 0:
                bad = next(i for i, sl in enumerate(shape)
                           if np.bincount(np.array(sl) // tw).max() > 1)
                options = [x for x in options if (x - bad + 1) % th == 0]
            if not options:
                break
            leading.append(int(rng.choice(options)))
        if len(leading) != len(shapes):
            continue
        if special == "matched":
            leading[1] = leading[0]
        packed = np.full((height, width), role_colors[0], dtype=np.int16)
        packed[membrane, :] = role_colors[1]
        for shape, lead in zip(shapes, leading):
            for i, sl in enumerate(shape):
                packed[lead - i, list(sl)] = role_colors[2]
        info = _scene_info(packed, th, tw)
        output, _ = _motion(packed, info)
        grid = _encode(rng, packed, th, tw, role_colors, style)
        grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
        try:
            decoded = _decode(grid)
        except ValueError:
            continue
        # The tip demonstration must expose the danger of moving visible pieces independently.
        if special == "tip" and np.array_equal(_fragment_prediction(decoded), output):
            continue
        return dict(input=grid, output=output)
    raise RuntimeError("could not construct a scene")


def _counts(num_train, num_test):
    for n, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.choice(10, size=5, replace=False)]
    rotation = _randint(rng, 0, 3)
    vary_colors, vary_rotation = bool(rng.integers(2)), bool(rng.integers(2))
    styles = ("flat", "stripes", "markers", "texture")
    first_style = str(rng.choice(styles))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            special = "matched" if split == "train" and index == 0 else (
                      "tip" if split == "train" and index == 1 else "free")
            if split == "train" and index == 0:
                palette, turn, style = colors, rotation, first_style
            else:
                if vary_colors and split == "train" and index == 1:
                    palette = [colors[i] for i in (3, 4, 0, 1, 2)]
                elif vary_colors:
                    palette = [int(c) for c in rng.choice(10, size=5, replace=False)]
                else:
                    palette = colors
                turn = ((rotation + _randint(rng, 1, 3)) % 4 if split == "train" and index == 1
                        else _randint(rng, 0, 3)) if vary_rotation else rotation
                choices = [s for s in styles if s != first_style] if split == "train" and index == 1 else styles
                style = str(rng.choice(choices))
            for _ in range(40):
                pair = _make_pair(rng, palette, turn, style, special, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_semantics(decoded, output):
    packed, info = decoded["packed"], decoded["info"]
    expected, records = _motion(packed, info)
    _require(np.array_equal(expected, output), "wrong output")
    _require(output.shape == packed.shape, "wrong packed dimensions")
    _require(np.count_nonzero(output == info["ink"]) == np.count_nonzero(packed == info["ink"]),
             "payload cells lost or duplicated")
    _require(set(map(int, np.unique(output))) <= set(decoded["palette"]), "scaffold retained")
    axis, sign, wall = info["axis"], info["sign"], info["membrane"]
    swept, final, holes = set(), set(), set()
    for rec in records:
        _require(swept.isdisjoint(rec["swept"]), "object trajectories interact")
        swept.update(rec["swept"])
        final.update(map(tuple, rec["final"]))
        holes.update(rec["holes"])
        _require(rec["distance"] > 0, "object did not move")
        lead, bad = rec["levels"][0], rec["first_bad"]
        if bad is None:
            boundary = packed.shape[axis] - 1 if sign > 0 else 0
            distance = (boundary - lead) * sign
            _require(np.all((rec["final"][:, axis] - wall) * sign > 0), "passer failed to clear wall")
            _require(rec["reason"] == "boundary" and rec["holes"], "passing semantics")
        else:
            distance = (wall - sign - (lead - sign * bad)) * sign
            _require(rec["reason"] == "pore", "stop reason")
            _require((rec["final"][:, axis].max() if sign > 0 else rec["final"][:, axis].min())
                     == wall + sign * (bad - 1), "first overloaded slice not caught")
        _require(distance == rec["distance"], "analytic displacement mismatch")
        delta = np.zeros(2, dtype=int)
        delta[axis] = distance * sign
        _require(np.array_equal(rec["original"] + delta, rec["final"]), "object deformed")
    # Check every untouched and traversed membrane cell directly.
    for t in range(packed.shape[1 - axis]):
        p = (wall, t) if axis == 0 else (t, wall)
        color = info["ink"] if p in final else info["bg"] if p in holes else info["wall"]
        _require(output[p] == color, "puncture did not persist")
    return records


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "bad split")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, outcomes = set(), set()
    pore_witness = reconnect_witness = alignment_witness = False
    train_max, test_max = 0, 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            decoded = _decode(grid)
            records = _check_semantics(decoded, output)
            if split == "train":
                outcomes.update(r["outcome"] for r in records)
                train_max = max(train_max, *(r["length"] for r in records))
                pore_witness |= not np.array_equal(_motion(decoded["packed"], decoded["info"],
                                                          global_capacity=True)[0], output)
                reconnect_witness |= not np.array_equal(_fragment_prediction(decoded), output)
                axis = decoded["info"]["axis"]
                for a in records:
                    for b in records:
                        if a["outcome"] != "pass" or b["outcome"] != "blocked":
                            continue
                        ac, bc = a["original"], b["original"]
                        alignment_witness |= (np.array_equal(ac - ac.min(axis=0), bc - bc.min(axis=0))
                                              and np.array_equal(ac[:, axis], bc[:, axis]))
                _require(not np.array_equal(decoded["packed"], output), "A has no effect")
            else:
                _require(any(r["length"] >= 5 for r in records), "test pair lacks the length shift")
                test_max = max(test_max, *(r["length"] for r in records))
    _require(outcomes == {"pass", "blocked", "partial"}, "demonstrations miss an outcome")
    _require(pore_witness, "training does not teach per-pore capacity")
    _require(alignment_witness, "training lacks the identical-shape alignment contrast")
    _require(reconnect_witness, "training does not teach seam reconnection")
    _require(train_max <= 4 and test_max >= 5, "length shift missing")


def _fingerprint(puzzle):
    return tuple(_key(np.asarray(p[k])) for s in ("train", "test") for p in puzzle[s]
                 for k in ("input", "output"))


def _hand_checked_cases():
    """Explicit answers: local overload, bent passage and non-contiguous slices."""
    rng = np.random.default_rng(2718)
    cases = []
    g = np.zeros((16, 16), dtype=np.int16)
    g[8, :] = 2
    g[3:6, 3:5] = 1  # same shape, different pore alignment
    g[3:6, 6:8] = 1
    g[5:7, 11] = 1
    g[4, 10:12] = 1
    out = np.zeros_like(g)
    out[8, :] = 2
    out[8, 3:5] = 0
    out[13:16, 3:5] = 1
    out[5:8, 6:8] = 1
    out[8:10, 11] = 1
    out[7, 10:12] = 1
    cases.append((g, out))
    # A fork's two front cells occupy different pores but belong to one rigid body.
    g = np.zeros((16, 16), dtype=np.int16)
    g[8, :] = 2
    g[3:5, 3] = g[3:5, 7] = 1
    g[2, 3:8] = 1
    out = np.zeros_like(g)
    out[8, :] = 2
    out[7, 3:8] = 1
    out[8:10, 3] = out[8:10, 7] = 1
    cases.append((g, out))
    # A bent four-cell object can pass, although its bounding box is two wide.
    g = np.zeros((16, 16), dtype=np.int16)
    g[8, :] = 2
    g[5, 4] = g[4, 3] = g[4, 4] = g[3, 3] = 1
    out = np.zeros_like(g)
    out[8, :] = 2
    out[8, 3:5] = 0
    out[15, 4] = out[14, 3] = out[14, 4] = out[13, 3] = 1
    cases.append((g, out))
    for packed, expected in cases:
        encoded = _encode(rng, packed, 4, 4, (0, 2, 1, 8, 9), "markers")
        for turn in range(4):
            inp, answer = np.rot90(encoded, turn), np.rot90(expected, turn)
            _require(np.array_equal(solve(inp), answer), "hand-checked geometry failed")
            permutation = rng.permutation(10)
            _require(np.array_equal(solve(permutation[inp]), permutation[answer]),
                     "color relabeling changed the rule")


def self_test():
    """Bounded regression, ablation, legal-count and 42-seed diversity checks."""
    _hand_checked_cases()
    _require(_fingerprint(generate_puzzle(917, 5, 3)) == _fingerprint(generate_puzzle(917, 5, 3)),
             "seed is nondeterministic")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(1000 + 10 * n + m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "count ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    shapes, layouts, normalized, palettes, directions = set(), set(), set(), set(), set()
    pore_counts, tile_shapes, object_counts, first_bad = set(), set(), set(), set()
    fingerprints = set()
    fragments = multi_pore_passers = unequal_gaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                decoded = _decode(pair["input"])
                info, packed = decoded["info"], decoded["packed"]
                records = _check_semantics(decoded, pair["output"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                shapes.add((pair["input"].shape, pair["output"].shape))
                tile_shapes.add(decoded["tile"])
                directions.add((info["axis"], info["sign"]))
                palettes.add((info["bg"], info["wall"], info["ink"]))
                pore_counts.add(packed.shape[1 - info["axis"]] // info["pore_size"])
                object_counts.add(len(records))
                layouts.add((decoded["tile"], tuple(map(tuple, decoded["groups"][0])),
                             tuple(map(tuple, decoded["groups"][1]))))
                norm = np.zeros_like(packed)
                norm[packed == info["wall"]] = 1
                norm[packed == info["ink"]] = 2
                normalized.add(_key(norm))
                unequal_gaps += any(len(set(g)) > 1 for g in decoded["gaps"])
                fragments += len(_components(pair["input"] == info["ink"])) > len(records)
                for rec in records:
                    first_bad.add(rec["first_bad"])
                    multi_pore_passers += rec["outcome"] == "pass" and any(
                        np.count_nonzero(rec["original"][:, info["axis"]] == level) > 1
                        for level in rec["levels"])
    _require(len(fingerprints) == 42 and len(normalized) >= 120, "insufficient structural diversity")
    _require(len(shapes) >= 70 and len(layouts) >= 100, "insufficient layout diversity")
    _require(directions == {(0, -1), (0, 1), (1, -1), (1, 1)}, "missing directions")
    _require(len(palettes) >= 30 and len(tile_shapes) >= 12, "limited roles or tiles")
    _require(pore_counts == {2, 3, 4, 5} and object_counts == {1, 2, 3, 4}, "limited counts")
    _require({None, 0, 1, 2, 3}.issubset(first_bad), "limited penetration depths")
    _require(fragments >= 80 and multi_pore_passers >= 42 and unequal_gaps >= 40,
             "missing interaction or gap diversity")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shape_pairs=len(shapes),
                layouts=len(layouts), normalized_scenes=len(normalized),
                palettes=len(palettes), tile_shapes=len(tile_shapes),
                fragmented_inputs=fragments, multi_pore_passers=multi_pore_passers)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
