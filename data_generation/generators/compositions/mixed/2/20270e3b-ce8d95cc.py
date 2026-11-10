"""Register two silhouettes, union their pixels, then contract equal axis runs.

Only NumPy and the standard library are used. Generation builds a joint mask
and partitions it into connected fragments; solve reconstructs it from pixels.
There are no parent imports, stored pairs, caches, or file-writing side effects.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("grids must be integer rectangles, sides 1..30, colors 0..9")
    return g.astype(np.int8, copy=True)


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r or c) and (diagonal or not (r and c))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in offsets:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr=0, dc=0):
    return {(r + dr, c + dc) for r, c in cells}


def _raster(cells, background=0, foreground=1, box=None):
    if not cells:
        return np.full((1, 1), background, dtype=np.int8)
    t, b, l, r = _bbox(cells) if box is None else box
    g = np.full((b - t + 1, r - l + 1), background, dtype=np.int8)
    for y, x in cells:
        g[y - t, x - l] = foreground
    return g


def _tight(g, background):
    positions = np.argwhere(g != background)
    if not len(positions):
        return np.full((1, 1), background, dtype=np.int8)
    t, l = positions.min(axis=0)
    b, r = positions.max(axis=0)
    return g[t:b + 1, l:r + 1].copy()


def _compress(g, axes=(0, 1), global_unique=False):
    result = g.copy()
    for axis in axes:
        vectors = result if axis == 0 else result.T
        keep, seen = [], set()
        for i, vector in enumerate(vectors):
            if global_unique:
                key = vector.tobytes()
                accept = key not in seen
                seen.add(key)
            else:
                accept = i == 0 or not np.array_equal(vector, vectors[i - 1])
            if accept:
                keep.append(i)
        result = np.take(result, keep, axis=axis)
    return result.copy()


def _parse(input_grid):
    """Input-only role inference; bar backing, never canvas order, picks roles."""
    g = _grid(input_grid)
    background = int(g[0, 0])
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    colors = [int(x) for x in np.unique(g)]
    if len(colors) != 3 or np.any(border != background):
        raise ValueError("expected three roles and a uniform background border")
    points = {c: {tuple(map(int, q)) for q in np.argwhere(g == c)} for c in colors}
    fragments = _components(set.union(*(points[c] for c in colors if c != background)))
    if len(fragments) != 2:
        raise ValueError("expected exactly two eight-connected fragments")
    choices = []
    for marker in colors:
        if marker == background:
            continue
        foreground = next(c for c in colors if c not in (background, marker))
        bars = _components(points[marker], diagonal=False)
        if len(bars) != 2 or not 2 <= len(bars[0]) == len(bars[1]) <= 6:
            continue
        length = len(bars[0])
        for axis in (0, 1):
            oriented = [{(q[axis], q[1 - axis]) for q in bar} for bar in bars]
            if any(len({y for y, x in bar}) != 1
                   or _bbox(bar)[3] - _bbox(bar)[2] + 1 != length for bar in oriented):
                continue
            records = []
            for fragment in fragments:
                owned = [bar for bar in bars if bar <= fragment]
                if len(owned) != 1:
                    break
                bar = {(q[axis], q[1 - axis]) for q in owned[0]}
                y0, _, x0, _ = _bbox(bar)
                fg = {(q[axis] - y0, q[1 - axis] - x0)
                      for q in fragment & points[foreground]}
                negative = any((-1, c) in fg for c in range(length))
                positive = any((1, c) in fg for c in range(length))
                if negative == positive:
                    break
                records.append((negative, fg))
            if len(records) != 2 or records[0][0] == records[1][0]:
                continue
            records.sort(key=lambda item: not item[0])
            choices.append({"background": background, "foreground": foreground,
                            "marker": marker, "axis": axis, "length": length,
                            "a": records[0][1], "b": records[1][1]})
    if len(choices) != 1:
        raise ValueError("input does not uniquely identify the marker and mating axis")
    return choices[0]


def _predict(parsed, variant="correct"):
    """Variants are finite negative checks, never options selected by solve."""
    p = parsed
    seam = {"seam0": 0, "seam1": 1}.get(variant, -1)
    a, b = p["a"], _shift(p["b"], seam)
    cells = a | b
    guides = {(y, x) for y in (0, seam) for x in range(p["length"])}
    if variant == "erase":
        cells -= guides
    elif variant == "paint":
        cells |= guides
    elif variant == "xor":
        cells = a ^ b
    elif variant in ("anchor", "mover"):
        cells = a if variant == "anchor" else b
    elif variant in ("opaque_a", "opaque_b"):
        front, back = (a, b) if variant == "opaque_a" else (b, a)
        t, bot, l, r = _bbox(front)
        cells = front | {q for q in back if not (t <= q[0] <= bot and l <= q[1] <= r)}
    raw = _raster(cells, p["background"], p["foreground"])
    if variant == "union_only":
        result = raw
    else:
        axes = {"rows_only": (0,), "columns_only": (1,)}.get(variant, (0, 1))
        result = _compress(raw, axes, global_unique=variant == "global")
    result = _tight(result, p["background"])
    return result.T.copy() if p["axis"] else result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, demonstrations, generator state, or output is consulted."""
    return _predict(_parse(input_grid))


def _equal_runs(g, axis):
    vectors = g if axis == 0 else g.T
    runs = [1]
    for i in range(1, len(vectors)):
        if np.array_equal(vectors[i], vectors[i - 1]):
            runs[-1] += 1
        else:
            runs.append(1)
    return tuple(runs)


def _separated_repeat(g, axis):
    vectors = g if axis == 0 else g.T
    seen, previous = set(), None
    for v in vectors:
        key = v.tobytes()
        if key != previous and key in seen:
            return True
        seen.add(key)
        previous = key
    return False


def _facts(p):
    a, b = p["a"], _shift(p["b"], -1)
    union = a | b
    box = _bbox(union)
    aa, bb = _raster(a, box=box), _raster(b, box=box)
    raw = aa | bb
    witnesses = []
    for axis in (0, 1):
        u, x, y = (v if axis == 0 else v.T for v in (raw, aa, bb))
        witnesses.append(sum(np.array_equal(u[i], u[i - 1])
                             and not np.array_equal(x[i], x[i - 1])
                             and not np.array_equal(y[i], y[i - 1])
                             for i in range(1, len(u))))
    guides = {(r, c) for r in (-1, 0) for c in range(p["length"])}
    return {"raw": raw, "row_runs": _equal_runs(raw, 0),
            "column_runs": _equal_runs(raw, 1), "causal": tuple(witnesses),
            "separated": (_separated_repeat(raw, 0), _separated_repeat(raw, 1)),
            "overlap": len(a & b), "covered": len(guides & union),
            "uncovered": len(guides - union)}


_MUTANTS = ("union_only", "rows_only", "columns_only", "global", "seam0", "seam1",
            "erase", "paint", "xor", "anchor", "mover", "opaque_a", "opaque_b")
_FIRST_TEACHES = tuple(m for m in _MUTANTS if m not in ("paint", "xor"))


def _motif(rng, h, w, left, right, style):
    """One connected silhouette attached along its final row; no saved masks."""
    m = np.zeros((h, w), dtype=bool)
    m[-1, left:right + 1] = True
    if style == "ring":
        m[0, left:right + 1] = True
        m[:, left] = m[:, right] = True
        if right - left >= 4 and rng.random() < .5:
            m[:, int(rng.integers(left + 2, right - 1))] = True
    elif style == "mass":
        m[:, left:right + 1] = True
        if h >= 3 and right - left >= 2 and rng.random() < .8:
            m[int(rng.integers(1, h - 1)), int(rng.integers(left + 1, right))] = False
    elif style == "fork":
        m[:, left] = m[:, right] = True
        m[:int(rng.integers(1, h)), int(rng.integers(left, right + 1))] = True
        # Connect a shorter tine downward, rather than adding a floating pixel.
        for c in range(left, right + 1):
            rows = np.flatnonzero(m[:, c])
            if len(rows):
                m[rows[0]:, c] = True
    elif style == "branch":
        spine = int(rng.integers(left, right + 1))
        m[:, spine] = True
        for y in range(h - 1):
            lo = int(rng.integers(left, spine + 1))
            hi = int(rng.integers(spine, right + 1))
            m[y, lo:hi + 1] = True
    elif style == "stair":
        for y in range(h):
            x = left + ((right - left) * y // max(1, h - 1))
            next_x = left + ((right - left) * min(y + 1, h - 1) // max(1, h - 1))
            m[y, x:next_x + 1] = True
    else:  # A hook has independently chosen upper and lower arms.
        spine = left if rng.integers(2) else right
        m[:, spine] = True
        end = int(rng.integers(left, right + 1))
        m[0, min(spine, end):max(spine, end) + 1] = True
    return m


def _geometry(rng, seam_size, teaching, collision, gap):
    w = int(rng.integers(6, 10))
    ht, hb = (int(rng.integers(3, 5)) for _ in range(2))
    # A connected rail ties the two independently shaped halves together.
    # A gap is bridged in each fragment by the neighboring attachment row.
    rail_l = int(rng.integers(0, 2))
    rail_r = w - 1 - int(rng.integers(0, 2))
    rail = np.zeros(w, dtype=bool)
    rail[rail_l:rail_r + 1] = True
    gap_column = int(rng.integers(rail_l + 2, rail_r)) if gap else None
    if gap:
        rail[gap_column] = False
    motifs = []
    for h in (ht, hb):
        left = int(rng.integers(0, rail_l + 2))
        right = int(rng.integers(max(left + 2, rail_r - 1), w))
        style = "ring" if teaching == 0 else str(rng.choice(
            ["ring", "mass", "fork", "branch", "stair", "hook"]))
        motif = _motif(rng, h, w, left, right, style)
        motif[-1, min(left, rail_l):max(right, rail_r) + 1] = True
        motifs.append(motif)
    canonical = np.vstack((motifs[0], rail[None, :], motifs[1][::-1]))
    row_runs = np.ones(ht + hb + 1, dtype=int)
    for indices in (list(range(ht)), list(range(ht + 1, ht + hb + 1))):
        # At most six rows per half, so even the longer test seam can be staged.
        budget = int(rng.integers(0, 7 - len(indices)))
        for _ in range(budget):
            row_runs[int(rng.choice(indices))] += 1
    row_runs[ht] = seam_size
    col_runs = np.ones(w, dtype=int)
    col_runs[rail_l] = 2  # Protected column-equality witness outside the bar.
    budget = int(rng.integers(0, 14 - int(col_runs.sum())))
    candidates = [x for x in range(w) if x != gap_column]
    for _ in range(budget):
        col_runs[int(rng.choice(candidates))] += 1
    target = np.repeat(np.repeat(canonical, row_runs, axis=0), col_runs, axis=1)
    starts = np.concatenate(([0], np.cumsum(col_runs)))
    witness = (int(starts[rail_l]), int(starts[rail_l] + 1))
    lo, hi = int(starts[rail_l] + 2), int(starts[rail_r + 1])
    lengths = list(range(2, min(6, hi - lo) + 1))
    if not lengths:
        return None
    if gap:
        gx = int(starts[gap_column])
        possibilities = [(x, n) for n in lengths for x in range(lo, hi - n + 1)
                         if x <= gx < x + n and any(rail[gap_column + d]
                            for d in (-1, 1))]
    else:
        possibilities = [(x, n) for n in lengths for x in range(lo, hi - n + 1)]
    if not possibilities:
        return None
    bar_x, length = possibilities[int(rng.integers(len(possibilities)))]
    lead = int(rng.integers(2, seam_size - 1))
    origin = int(row_runs[:ht].sum()) + lead
    frontiers = rng.integers(-2, 1, size=target.shape[1])
    frontiers[bar_x:bar_x + length] = -1
    frontiers[witness[0]], frontiers[witness[1]] = -2, 0
    a, b = set(), set()
    for r, c in np.argwhere(target):
        y, x = int(r) - origin, int(c)
        (a if y <= frontiers[x] else b).add((y, x))
    if collision:
        options = sorted(q for q in a if q[1] not in witness
                         and not bar_x <= q[1] < bar_x + length
                         and any((q[0] + dy, q[1] + dx) in b
                                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)))
        if not options:
            return None
        for i in rng.choice(len(options), min(len(options), int(rng.integers(1, 5))), replace=False):
            b.add(options[int(i)])
    # Convert both fragments to their OWN bar origin. In the registered frame
    # the mover's bar is at -1, so its ordinary coordinates are shifted by +1.
    aa, bb = _shift(a, dc=-bar_x), _shift(b, 1, -bar_x)
    bar = {(0, c) for c in range(length)}
    if any(len(_components(part | bar)) != 1 or part & bar for part in (aa, bb)):
        return None
    return aa, bb, length, target


def _stage(rng, a, b, length, palette, mode):
    bar = {(0, c) for c in range(length)}
    groups = []
    for part in (a, b):
        t, bot, l, r = _bbox(part | bar)
        groups.append((_shift(part, -t, -l), _shift(bar, -t, -l), bot - t + 1, r - l + 1))
    if rng.integers(2):
        groups.reverse()
    af, am, ah, aw = groups[0]
    bf, bm, bh, bw = groups[1]
    gap = int(rng.integers(1, 5))
    if mode == "horizontal":
        dy, dx = int(rng.integers(-bh + 1, ah)), aw + gap
    elif mode == "vertical":
        dy, dx = ah + gap, int(rng.integers(-bw + 1, aw))
    else:
        dy, dx = ah + gap, aw + gap
    fg, markers = af | _shift(bf, dy, dx), am | _shift(bm, dy, dx)
    t, bot, l, r = _bbox(fg | markers)
    h, w = bot - t + 1, r - l + 1
    if h > 28 or w > 28:
        return None
    pads = [int(rng.integers(1, 3)) for _ in range(4)]
    pads[1] = min(pads[1], 30 - h - pads[0])
    pads[3] = min(pads[3], 30 - w - pads[2])
    bg, fc, mc = palette
    g = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), bg, dtype=np.int8)
    for points, color in ((fg, fc), (markers, mc)):
        for y, x in points:
            g[y - t + pads[0], x - l + pads[2]] = color
    return g


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, profile, used):
    for _ in range(160):
        geometry = _geometry(rng, profile["seam"], profile["teaching"],
                             profile["collision"], profile["gap"])
        if geometry is None:
            continue
        a, b, length, target = geometry
        # Random staging is independent of the assembly translation.
        g = _stage(rng, a, b, length, profile["palette"], str(rng.choice(
            ["horizontal", "vertical", "diagonal"])))
        if g is None:
            continue
        bg, fg, _ = profile["palette"]
        output = _tight(_compress(np.where(target, fg, bg).astype(np.int8)), bg)
        k = profile["rotation"]
        g, output = np.rot90(g, k).copy(), np.rot90(output, k).copy()
        if _key(g) in used:
            continue
        try:
            p = _parse(g)
        except ValueError:
            continue
        facts = _facts(p)
        if not all(facts["causal"]) or not np.array_equal(_predict(p), output):
            continue
        if profile["collision"] != bool(facts["overlap"]):
            continue
        if profile["gap"] != bool(facts["uncovered"]):
            continue
        required = _FIRST_TEACHES if profile["teaching"] == 0 else (
            ("paint", "xor") if profile["teaching"] == 1 else ())
        if any(np.array_equal(_predict(p, mutant), output) for mutant in required):
            continue
        if profile["teaching"] == 0 and not all(facts["separated"]):
            continue
        return {"input": g, "output": output}
    raise RuntimeError("bounded construction exhausted: no valid registered silhouette")


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    changing_palette = bool(rng.integers(2))
    mixed_axis = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    # Only the thickness of the registered seam is held out. All other rules,
    # geometry ranges, palettes and orientation policies are shared.
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            teaching = i if split == "train" and i < 2 else -1
            colors = tuple(map(int, rng.choice(10, 3, replace=False))) if changing_palette else palette
            k = (rotation + i % 2 + 2 * int(rng.integers(2))) % 4 if mixed_axis else rotation
            profile = {"teaching": teaching, "palette": colors, "rotation": k,
                       "seam": int(rng.integers(4, 6) if split == "train" else rng.integers(6, 9)),
                       "collision": teaching == 1 or (teaching < 0 and bool(rng.integers(2))),
                       "gap": teaching == 1 or (teaching < 0 and bool(rng.integers(2)))}
            pair = _make_pair(rng, profile, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Pixel-derived checks, including both-parent causal witnesses and mutants."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, training, test_axes, train_axes = set(), [], set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(g) not in used
            used.add(_key(g))
            p, facts = _parse(g), None
            facts = _facts(p)
            raw = np.where(facts["raw"], p["foreground"], p["background"]).astype(np.int8)
            # Literal reconstruction: deleting runs must retain every chosen
            # crossing value; expanding those runs must exactly recover union.
            expected = _compress(raw)
            reconstructed = np.repeat(np.repeat(expected, facts["row_runs"], axis=0),
                                      facts["column_runs"], axis=1)
            assert np.array_equal(reconstructed, raw)
            assert np.array_equal(_compress(raw, (1, 0)), expected)
            if p["axis"]:
                expected = expected.T.copy()
            assert np.array_equal(expected, out)
            assert np.array_equal(solve(g), out)
            assert p["marker"] not in out
            assert all(facts["causal"]), "union must create new equal vectors on BOTH axes"
            assert facts["covered"] > 0
            assert max(facts["row_runs"]) > 1 and max(facts["column_runs"]) > 1
            assert not np.array_equal(g, out)
            if split == "train":
                training.append((p, out, facts))
                train_axes.add(p["axis"])
            else:
                test_axes.add(p["axis"])
    assert test_axes <= train_axes
    assert all(training[0][2]["separated"]), "teach retention of separated repeats"
    assert training[0][2]["overlap"] == training[0][2]["uncovered"] == 0
    assert training[1][2]["overlap"] > 0 and training[1][2]["uncovered"] > 0
    for mutant in _MUTANTS:
        assert any(not np.array_equal(_predict(p, mutant), out) for p, out, _ in training), mutant
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, bounded in-memory diagnostics and no disk writes."""
    shapes, outputs, palettes, structures, hashes = set(), set(), set(), set(), set()
    lengths, axes, causal, overlaps, uncovered = set(), set(), [0, 0], 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                facts = _facts(p)
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                palettes.add((p["background"], p["foreground"], p["marker"]))
                structures.add(_key(pair["input"] == p["foreground"]))
                lengths.add(p["length"])
                axes.add(p["axis"])
                causal[0] += int(facts["causal"][0])
                causal[1] += int(facts["causal"][1])
                overlaps += facts["overlap"] > 0
                uncovered += facts["uncovered"] > 0
        # Compare compact hashes so the deterministic repeat does not retain a
        # second complete puzzle. The loop's last pair is at most 1,800 bytes.
        fingerprint = _fingerprint(puzzle)
        del puzzle
        assert _fingerprint(generate_puzzle(seed)) == fingerprint
    assert len(hashes) == 42 and len(shapes) >= 35 and len(outputs) >= 12
    assert len(structures) >= 170 and len(palettes) >= 25
    assert axes == {0, 1} and len(lengths) >= 4
    assert min(causal) >= 210 and overlaps >= 42 and uncovered >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(73, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Small hand-computed operation regression: separated repeats survive,
    # and columns with a single different crossing are never collapsed.
    probe = np.array([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0],
                      [1, 1, 0, 1]], dtype=np.int8)
    assert _compress(probe).tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "input_shapes": len(shapes), "output_shapes": len(outputs),
            "foreground_structures": len(structures), "role_palettes": len(palettes),
            "bar_lengths": sorted(lengths), "axes": sorted(axes),
            "union_created_row_equalities": causal[0], "union_created_column_equalities": causal[1],
            "pairs_with_collisions": overlaps, "pairs_with_uncovered_guides": uncovered,
            "semantic_scope": "authored pixel invariants and finite alternatives; no independent solver"}
