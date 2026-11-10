"""Threaded sockets: rewired paths zip independent hollow/solid queues.

No parent code, examples, seed state, or training outputs are used by solve.
Only Python's standard library and NumPy are required.
"""
from collections import deque
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), []
        while queue:
            r, c = queue.popleft()
            cells.append((int(r), int(c)))
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(cells)
    return result


def _neighbors(mask, cell):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIRS
            if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
            and mask[r + dr, c + dc]]


def _paths(mask):
    result = []
    for component in _components(mask):
        endpoints = [p for p in component if len(_neighbors(mask, p)) == 1]
        assert len(endpoints) == 2, "every thread must have two endpoints"
        assert all(len(_neighbors(mask, p)) in (1, 2) for p in component), "branch"
        start, end = sorted(endpoints)
        path, previous, current = [start], None, start
        while current != end:
            options = [p for p in _neighbors(mask, current) if p != previous]
            assert len(options) == 1
            previous, current = current, options[0]
            path.append(current)
        assert len(path) == len(component)
        result.append(path)
    return sorted(result, key=lambda p: (p[0], p[-1]))


def _tile(size, color, bg, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == "H":
        a[1:-1, 1:-1] = bg
    return a


def _turn(base, bars, bg, track, divider, spacing):
    """A scaled switch has two side rails and a rotated central wall."""
    out, writes = base.copy(), {}
    radius = spacing // 2
    for r, c in bars:
        for y in range(r - radius, r + radius + 1):
            for x in range(c - radius, c + radius + 1):
                color = track if abs(x - c) == radius else (divider if x == c else bg)
                assert (y, x) not in writes or writes[y, x] == color, "conflicting switches"
                writes[y, x] = color
    for p, color in writes.items():
        out[p] = color
    return out, set(writes)


def _parse_horizontal(g):
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1]))
    foreground = np.argwhere(g != bg)
    assert len(foreground)
    lo, hi = int(foreground[:, 1].min()), int(foreground[:, 1].max())
    side = g[:, lo]
    side_colors = set(map(int, side[side != bg]))
    assert len(side_colors) == 1
    track = next(iter(side_colors))
    lanes = list(map(int, np.flatnonzero(side == track)))
    assert len(lanes) >= 2
    spacings = set(np.diff(lanes).tolist())
    assert len(spacings) == 1 and next(iter(spacings)) in (4, 6)
    spacing = next(iter(spacings))
    size, half = spacing - 1, (spacing - 2) // 2
    assert lanes[0] - half >= 1 and lanes[-1] + half < g.shape[0] - 1
    assert np.array_equal(g[:, lo], g[:, hi]), "exposed track ends"
    bars, objects, bar_colors = [], [], set()
    for color in sorted(set(map(int, np.unique(g))) - {bg, track}):
        for component in _components(g == color):
            rows, cols = zip(*component)
            top, left, bottom, right = min(rows), min(cols), max(rows) + 1, max(cols) + 1
            patch = g[top:bottom, left:right]
            if patch.shape == (1, spacing + 1) and np.all(patch == color):
                center = left + spacing // 2
                assert top in [r + spacing // 2 for r in lanes[:-1]]
                assert lo + spacing // 2 + 1 <= center <= hi - spacing // 2 - 1
                bars.append((top, center))
                bar_colors.add(color)
            else:
                assert patch.shape == (size, size)
                row, col = top + half, left + half
                assert row in lanes and lo < left and right - 1 < hi
                if np.all(patch == color):
                    kind = "S"
                else:
                    kind = "H"
                    assert np.array_equal(patch, _tile(size, color, bg, kind))
                objects.append(dict(center=(row, col), top=top, left=left,
                                    kind=kind, color=color, tile=patch.copy()))
    assert len(bar_colors) <= 1
    divider = next(iter(bar_colors)) if bar_colors else None
    assert all(o["color"] != divider for o in objects)
    assert {o["kind"] for o in objects} == {"H", "S"}
    # Reconstruct the track under each glyph, then check every input pixel.
    clean = g.copy()
    for o in objects:
        r, c = o["center"]
        for rr, cc in bars:
            if abs(r - rr) == spacing // 2:
                assert abs(c - cc) >= spacing + 1, "glyph touches a switch"
        clean[o["top"]:o["top"] + size, o["left"]:o["left"] + size] = bg
        clean[r, c - half:c + half + 1] = track
    base = np.full(g.shape, bg, dtype=np.int8)
    breaks = []
    for r in lanes:
        assert np.all(np.isin(clean[r, lo:hi + 1], [bg, track]))
        base[r, lo:hi + 1] = clean[r, lo:hi + 1]
        breaks += [(r, int(c)) for c in np.flatnonzero(clean[r, lo:hi + 1] == bg) + lo]
    for r, c in bars:
        base[r, c - spacing // 2:c + spacing // 2 + 1] = divider
    assert np.array_equal(base, clean), "unexpected foreground"
    for o in objects:
        r, c = o["center"]
        assert np.all(base[r, c - half - 1:c + half + 2] == track)
        for other in objects:
            if other is not o and other["center"][0] == r:
                assert abs(c - other["center"][1]) >= size + 1
    turned, protected = _turn(base, bars, bg, track, divider, spacing)
    assert not set(breaks) & protected, "a switch must not erase a break"
    paths = _paths(turned == track)
    path_ids = {p: i for i, path in enumerate(paths) for p in path}
    for path in paths:
        for r, c in (path[0], path[-1]):
            assert r in lanes
            assert c in (lo, hi) or (r, c - 1) in breaks or (r, c + 1) in breaks
    assert all(o["center"] in path_ids for o in objects)
    return dict(grid=g, bg=bg, track=track, divider=divider, size=size,
                spacing=spacing, lanes=lanes, lo=lo, hi=hi, bars=sorted(bars),
                breaks=breaks, base=base, structural=turned, paths=paths,
                path_ids=path_ids, objects=objects)


def _analyze(value):
    g = _grid(value)
    candidates = []
    for transpose in (False, True):
        try:
            a = _parse_horizontal(g.T if transpose else g)
            a["transpose"] = transpose
            candidates.append(a)
        except (AssertionError, IndexError):
            pass
    if len(candidates) != 1:
        raise ValueError("input must have one valid threaded-socket interpretation")
    return candidates[0]


def _matches(a, mode="rule"):
    objects = a["objects"]
    if mode == "global":
        sequences = [sorted(objects, key=lambda o: o["center"])]
    else:
        paths = a["paths"]
        if mode == "no_rewire":
            mask = a["base"] == a["track"]
            paths = _paths(mask)
        sequences = []
        for path in paths:
            positions = {p: i for i, p in enumerate(path)}
            seq = [o for o in objects if o["center"] in positions]
            if mode == "x_ascending":
                key = lambda o: (o["center"][1], o["center"][0])
            elif mode == "x_descending":
                key = lambda o: (-o["center"][1], -o["center"][0])
            elif mode == "row_order":
                key = lambda o: o["center"]
            else:
                key = lambda o: positions[o["center"]]
            sequences.append(sorted(seq, key=key, reverse=mode == "reverse"))
    matches = []
    for seq in sequences:
        hollows = [o for o in seq if o["kind"] == "H"]
        solids = [o for o in seq if o["kind"] == "S"]
        if mode == "greedy":
            slots, rows = {}, []
            for o in seq:
                if o["kind"] in slots:
                    rows.append(slots)
                    slots = {}
                slots[o["kind"]] = o
            rows.append(slots)
            matches.extend((row["H"], row["S"]) for row in rows if set(row) == {"H", "S"})
        elif mode == "nearest":
            available = solids.copy()
            index = {id(o): i for i, o in enumerate(seq)}
            for h in hollows:
                if available:
                    donor = min(available, key=lambda o: (abs(index[id(o)] - index[id(h)]), index[id(o)]))
                    matches.append((h, donor))
                    available = [o for o in available if o is not donor]
        else:
            matches.extend(zip(hollows, solids))
    return matches


def _render(a, matches):
    out = a["structural"].copy()
    size, half = a["size"], a["size"] // 2
    for o in a["objects"]:
        out[o["top"]:o["top"] + size, o["left"]:o["left"] + size] = o["tile"]
    for hollow, solid in matches:
        r, c = solid["center"]
        out[solid["top"]:solid["top"] + size, solid["left"]:solid["left"] + size] = a["bg"]
        out[r, c - half:c + half + 1] = a["track"]
        out[hollow["top"] + 1:hollow["top"] + size - 1,
            hollow["left"] + 1:hollow["left"] + size - 1] = solid["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive geometry, role colors, traversal, matches and output from input."""
    a = _analyze(input_grid)
    out = _render(a, _matches(a))
    return out.T.copy() if a["transpose"] else out.copy()


def _base(lanes, width, size, bars, breaks=()):
    spacing = size + 1
    rows = [size // 2 + 1 + spacing * i for i in range(lanes)]
    g = np.zeros((spacing * (lanes - 1) + size + 2, width), dtype=np.int8)
    g[rows, 1:width - 1] = 1
    for r, c in bars:
        g[r, c - spacing // 2:c + spacing // 2 + 1] = 2
    for p in breaks:
        g[p] = 0
    return g, rows


def _layout(rng, size, role):
    """Sample switches, open cycles visibly, and optionally split long paths."""
    spacing, radius = size + 1, (size + 1) // 2
    maximum = 6 if size == 3 else 4
    minimum = 3 if role == "loop" else (4 if role == "test" else 2)
    lanes = int(rng.integers(minimum, maximum + 1))
    minimum_width = (29 if size == 5 and role == "first" else
                     27 if role == "loop" or (role == "test" and size == 5) else
                     20 if role == "test" else 24 if role == "first" else 14)
    width = int(rng.integers(minimum_width, 31))
    rows = [size // 2 + 1 + spacing * i for i in range(lanes)]
    sep_rows = [r + radius for r in rows[:-1]]
    low, high = 1 + radius + 1, width - 2 - radius - 1
    if role == "first":
        # Bias toward a long two-lane arm, with left/right and separator free.
        options = [c for c in range(low, high + 1)
                   if c <= low + 2 or c >= high - 2]
        bars = [(int(rng.choice(sep_rows)), int(rng.choice(options)))]
    elif role == "loop":
        row = int(rng.choice(sep_rows))
        bars = [(row, low), (row, high)]
    else:
        bars = []
        wanted = int(rng.integers(1, min(9, 2 * lanes) + 1))
        sites = [(r, c) for r in sep_rows for c in range(low, high + 1)]
        # Include agreeing, one-cell switch-neighborhood overlaps sometimes.
        if lanes >= 3 and rng.random() < .35 and high - low >= spacing:
            j = int(rng.integers(lanes - 2))
            c = int(rng.integers(low, high - spacing + 1))
            bars = [(sep_rows[j], c), (sep_rows[j + 1], c + spacing)]
        for idx in rng.permutation(len(sites)):
            r, c = sites[int(idx)]
            if len(bars) >= wanted:
                break
            if all(r != rr or abs(c - cc) >= spacing + 2 for rr, cc in bars) and all(
                    abs(r - rr) != spacing or abs(c - cc) >= spacing for rr, cc in bars):
                bars.append((r, c))
    base, rows = _base(lanes, width, size, bars)
    turned, protected = _turn(base, bars, 0, 1, 2, spacing)
    breaks = []
    mask = turned == 1
    for component in _components(mask):
        degrees = [len(_neighbors(mask, p)) for p in component]
        assert all(d in (1, 2) for d in degrees)
        if all(d == 2 for d in degrees):
            pool = [p for p in component if p[0] in rows and p not in protected]
            assert pool
            # Central cuts retain usable glyph sites on both sides of a loop.
            middle = .5 * (min(c for _, c in pool) + max(c for _, c in pool))
            near = [p for p in pool if abs(p[1] - middle) <= 1]
            cut = near[int(rng.integers(len(near)))] if role == "loop" else pool[int(rng.integers(len(pool)))]
            breaks.append(cut)
            base[cut] = turned[cut] = 0
    if role not in ("first", "loop"):
        for _ in range(int(rng.choice([0, 0, 1, 2, 3]))):
            paths = _paths(turned == 1)
            pool = [p for path in paths for p in path[2:-2]
                    if p[0] in rows and p not in protected
                    and len(_neighbors(turned == 1, p)) == 2]
            if pool:
                cut = pool[int(rng.integers(len(pool)))]
                breaks.append(cut)
                base[cut] = turned[cut] = 0
    paths = _paths(turned == 1)
    return dict(base=base, structural=turned, paths=paths, bars=bars,
                breaks=breaks, lanes=rows, size=size, spacing=spacing,
                lo=1, hi=width - 2)


def _packed_sites(layout, rng):
    s, half, d = layout["size"], layout["size"] // 2, layout["spacing"]
    result = []
    for path in layout["paths"]:
        indices = {p: i for i, p in enumerate(path)}
        candidates = []
        for r in layout["lanes"]:
            xs = []
            for c in range(layout["lo"] + half + 1, layout["hi"] - half):
                if (r, c) not in indices:
                    continue
                if not np.all(layout["base"][r, c - half - 1:c + half + 2] == 1):
                    continue
                if any(abs(r - rr) == d // 2 and abs(c - cc) < d + 1 for rr, cc in layout["bars"]):
                    continue
                xs.append(c)
            # Either extremal packing is maximal; their different phases vary sites.
            reverse = bool(rng.integers(2))
            chosen = []
            for c in sorted(xs, reverse=reverse):
                if all(abs(c - old) >= s + 1 for old in chosen):
                    chosen.append(c)
            candidates.extend((r, c) for c in chosen)
        result.append(sorted(candidates, key=indices.__getitem__))
    return result


def _pick(rng, sites, count):
    return [sites[int(i)] for i in sorted(rng.choice(len(sites), count, replace=False).tolist())]


def _glyphs(rng, layout, role):
    packs = _packed_sites(layout, rng)
    selected = []
    if role == "first":
        choices = [i for i, p in enumerate(packs) if len(p) >= 5
                   and layout["paths"][i][0][0] != layout["paths"][i][-1][0]]
        if not choices:
            return None
        i = int(rng.choice(choices))
        selected.append((_pick(rng, packs[i], 5), list("SSHHH")))
        others = [j for j, p in enumerate(packs) if j != i and len(p) >= 2]
        if others and rng.random() < .6:
            j = int(rng.choice(others))
            selected.append((_pick(rng, packs[j], 2), list(rng.permutation(list("HS")))))
    elif role == "loop":
        choices = [i for i, p in enumerate(packs) if len(p) >= 4 and
                   {int(b[0] - a[0]) for a, b in zip(layout["paths"][i], layout["paths"][i][1:])} >= {-1, 1}]
        if not choices:
            return None
        i = int(rng.choice(choices))
        selected.append((_pick(rng, packs[i], 4), list("HHSS" if rng.integers(2) else "SSHH")))
        others = [j for j, p in enumerate(packs) if j != i and len(p) >= 3]
        if not others:
            return None
        j = int(rng.choice(others))
        selected.append((_pick(rng, packs[j], 3), list(rng.permutation(list("HSS")))))
    else:
        available = [i for i, p in enumerate(packs) if len(p) >= 2]
        minimum = 3 if role == "test" else 1
        if len(available) < minimum:
            return None
        count = int(rng.integers(minimum, min(len(available), 4 if role == "test" else 2) + 1))
        for j in rng.choice(available, count, replace=False):
            j = int(j)
            n = int(rng.integers(2, min(len(packs[j]), 7) + 1))
            h = int(rng.integers(1, n))
            kinds = ["H"] * h + ["S"] * (n - h)
            mode = int(rng.integers(3))
            if mode == 0:
                rng.shuffle(kinds)
            elif mode == 1:
                kinds.reverse()
            selected.append((_pick(rng, packs[j], n), kinds))
    return selected


def _pair(rng, size, role, palette, transpose):
    for _ in range(160):
        layout = _layout(rng, size, role)
        selections = _glyphs(rng, layout, role)
        if selections is None:
            continue
        # Palettes encode all structural roles visibly; data colors may repeat.
        bg, track, divider = palette[:3]
        colors = palette[3:]
        inp = np.asarray(palette[:3], dtype=np.int8)[layout["base"]]
        for sites, kinds in selections:
            h_rank = s_rank = 0
            a, b, c = map(int, rng.choice(colors, 3, replace=False))
            for site, kind in zip(sites, kinds):
                if role in ("first", "loop"):
                    # Crossed colors make rank observable and refute color matching.
                    sequence = (a, b, c) if kind == "H" else (b, a, c)
                    rank = h_rank if kind == "H" else s_rank
                    color = sequence[rank % 3]
                else:
                    color = int(rng.choice(colors))
                h_rank += kind == "H"
                s_rank += kind == "S"
                r, x = site
                half = size // 2
                inp[r - half:r + half + 1, x - half:x + half + 1] = _tile(size, color, bg, kind)
        margins = []
        for extent in inp.shape:
            total = int(rng.integers(min(3, 30 - extent) + 1))
            before = int(rng.integers(total + 1))
            margins.append((before, total - before))
        inp = np.pad(inp, margins, constant_values=bg)
        if rng.integers(2):
            inp = np.fliplr(inp)
        if rng.integers(2):
            inp = np.flipud(inp)
        if transpose:
            inp = inp.T
        inp = inp.copy()
        return dict(input=inp, output=solve(inp))
    raise RuntimeError("bounded sampler could not place the required glyphs")


_ALTERNATIVES = ("no_rewire", "greedy", "global", "nearest", "reverse",
                 "x_ascending", "x_descending", "row_order")


def _features(pair):
    a = _analyze(pair["input"])
    matches = _matches(a)
    donors, sockets = {id(s) for _, s in matches}, {id(h) for h, _ in matches}
    groups = {a["path_ids"][h["center"]] for h, _ in matches}
    expected = _render(a, matches)
    alternatives = {m: not np.array_equal(expected, _render(a, _matches(a, m))) for m in _ALTERNATIVES}
    return dict(analysis=a, matches=matches, groups=len(groups),
                unmatched_h=any(o["kind"] == "H" and id(o) not in sockets for o in a["objects"]),
                unmatched_s=any(o["kind"] == "S" and id(o) not in donors for o in a["objects"]),
                alternatives=alternatives)


def _training_evidence(pairs):
    fs = [_features(p) for p in pairs]
    return (all(any(f["alternatives"][m] for f in fs) for m in _ALTERNATIVES)
            and all(any(f[k] for f in fs) for k in ("unmatched_h", "unmatched_s"))
            and any(f["groups"] == 2 for f in fs))


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    transpose = bool(rng.integers(2))
    variable_scale = bool(rng.integers(2))
    varying_palette = bool(rng.integers(2))
    fixed_palette = list(map(int, rng.permutation(10)))

    def palette():
        return list(map(int, rng.permutation(10))) if varying_palette else fixed_palette

    # The two witnesses remain sufficient at the minimum legal train count.
    for _ in range(32):
        training = [_pair(rng, 5 if variable_scale else 3, "first", palette(), transpose),
                    _pair(rng, 3, "loop", palette(), transpose)]
        if _training_evidence(training):
            break
    else:
        raise RuntimeError("could not expose all documented alternative rules")
    seen = {_key(p["input"]) for p in training}
    assert len(seen) == 2
    puzzle = dict(train=training, test=[])
    for split, count in (("train", nt - 2), ("test", nq)):
        for _ in range(count):
            for attempt in range(32):
                size = int(rng.choice([3, 5])) if variable_scale else 3
                pair = _pair(rng, size, "test" if split == "test" else "ordinary", palette(), transpose)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, axes, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _features(pair)
            a, matches = f["analysis"], f["matches"]
            axes.add(a["transpose"])
            expected = _render(a, matches)
            assert np.array_equal(out, expected.T if a["transpose"] else expected)
            assert matches and a["bars"] and not np.array_equal(inp, out)
            # One donor per socket, local to a rewired component, exact rim retention.
            assert len({id(s) for _, s in matches}) == len(matches)
            assert len({id(h) for h, _ in matches}) == len(matches)
            for h, s in matches:
                assert a["path_ids"][h["center"]] == a["path_ids"][s["center"]]
                tile = expected[h["top"]:h["top"] + a["size"], h["left"]:h["left"] + a["size"]]
                assert np.all(tile[1:-1, 1:-1] == s["color"])
                assert np.all(np.concatenate((tile[0], tile[-1], tile[:, 0], tile[:, -1])) == h["color"])
            # Excess donors and sockets retain their complete original glyph.
            used = {id(o) for match in matches for o in match}
            for o in a["objects"]:
                if id(o) not in used:
                    assert np.array_equal(expected[o["top"]:o["top"] + a["size"],
                                                   o["left"]:o["left"] + a["size"]], o["tile"])
            if split == "train":
                assert f["groups"] <= 2
                training.append(pair)
            else:
                assert f["groups"] >= 3, "test shift is more simultaneous active threads"
    assert len(axes) == 1
    assert _training_evidence(training), "missing a required instructional witness"
    sizes = { _analyze(p["input"])["size"] for p in training }
    assert all(_analyze(p["input"])["size"] in sizes for p in puzzle["test"])


def self_test():
    """Bounded family invariants, negative controls, and 42-seed diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    structures, shapes, palettes, bar_counts, lane_counts, break_counts = set(), set(), set(), set(), set(), set()
    sizes, axes, object_counts, overlaps, nonmonotone = set(), set(), set(), 0, 0
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                saved = inp.copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, saved), "solve must not mutate its argument"
                a = _analyze(inp)
                sizes.add(a["size"])
                axes.add(a["transpose"])
                shapes.add(inp.shape)
                palettes.add((a["bg"], a["track"], a["divider"]))
                bar_counts.add(len(a["bars"]))
                lane_counts.add(len(a["lanes"]))
                break_counts.add(len(a["breaks"]))
                object_counts.add(len(a["objects"]))
                structures.add((a["grid"].shape, tuple(a["lanes"]), tuple(a["bars"]),
                                tuple(a["breaks"]), tuple(sorted((o["center"], o["kind"]) for o in a["objects"]))))
                overlaps += sum(abs(r - rr) == a["spacing"] and abs(c - cc) == a["spacing"]
                                for i, (r, c) in enumerate(a["bars"]) for rr, cc in a["bars"][i + 1:])
                nonmonotone += sum({b[0] - p[0] for p, b in zip(path, path[1:])} >= {-1, 1} for path in a["paths"])
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 35 and len(palettes) >= 35
    assert sizes == {3, 5} and axes == {False, True}
    assert lane_counts >= {2, 3, 4, 5, 6}
    assert bar_counts >= {1, 2, 3, 4} and len(break_counts) >= 4
    assert len(object_counts) >= 8 and overlaps and nonmonotone >= 42
    # A concrete five-glyph regression, with named donor-to-socket mapping.
    base, rows = _base(2, 25, 3, [(4, 16)])
    for r, c, color, kind in ((2, 3, 4, "S"), (2, 7, 5, "S"), (2, 11, 6, "H"),
                               (6, 11, 7, "H"), (6, 7, 8, "H")):
        base[r - 1:r + 2, c - 1:c + 2] = _tile(3, color, 0, kind)
    result = solve(base)
    assert result[2, 11] == 4 and result[6, 11] == 5 and result[6, 7] == 0
    assert np.all(result[2, 2:5] == 1) and np.all(result[2, 6:9] == 1)
    assert np.all(result[2:7, 16] == 2) and np.all(result[2:7, 14] == 1)
    assert np.array_equal(solve(base.T), result.T)
    # Border translation and arbitrary role recoloring cannot affect matching.
    mapping = np.array([8, 0, 9, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
    assert np.array_equal(solve(mapping[base]), mapping[result])
    assert np.array_equal(solve(np.pad(base, 1)), np.pad(result, 1))
    # The solver also handles zero transfers, outside the generated teaching subset.
    isolated, _ = _base(2, 25, 3, [(4, 12)])
    isolated[1:4, 2:5] = _tile(3, 3, 0, "H")
    isolated[1:4, 20:23] = _tile(3, 4, 0, "S")
    isolated_out = solve(isolated)
    assert np.array_equal(isolated_out[1:4, 2:5], isolated[1:4, 2:5])
    assert np.array_equal(isolated_out[1:4, 20:23], isolated[1:4, 20:23])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert tuple(_key(p["input"]) for p in generate_puzzle(17)["train"]) == tuple(
        _key(p["input"]) for p in generate_puzzle(17)["train"])
    damaged = {s: [{k: np.array(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Closed threads are rejected: the traversal start would be undefined.
    cycle, _ = _base(2, 30, 3, [(4, 5), (4, 24)])
    for r, c, color, kind in ((6, 10, 3, "H"), (6, 14, 4, "S")):
        cycle[r - 1:r + 2, c - 1:c + 2] = _tile(3, color, 0, kind)
    try:
        solve(cycle)
    except ValueError:
        pass
    else:
        raise AssertionError("closed thread accepted")
    return dict(seeds=42, puzzles=len(signatures), structures=len(structures),
                shapes=len(shapes), structural_palettes=len(palettes),
                sizes=sorted(sizes), lanes=sorted(lane_counts),
                bars=sorted(bar_counts), breaks=sorted(break_counts),
                glyph_counts=sorted(object_counts), compatible_overlaps=overlaps,
                paths_with_vertical_reversal=nonmonotone)
