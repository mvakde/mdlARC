"""A half-specified sheet, a reflected cut, and the marked remaining piece.

Only standard-library modules and NumPy are used. Geometry is constructed
directly; solve() recovers all conventions from the input and the fixed rule.
"""
from __future__ import annotations

from collections import deque
import json

import numpy as np


HOST_SEEDS = (
    0, 789475835, 811046817, 862949392, 602453027, 685932530,
    292163842, 768934851, 256740210, 739953205, 90475857, 402963383,
    11263175, 290960200, 376367728, 735941594, 28239631, 232925504,
    210969417, 108215954, 231724027, 572621662, 527428108, 63244804,
    828989328, 759700202, 763702199, 768492776, 71605960, 86644250,
    80834447, 249280486, 981390798, 826571627, 279270596, 223345694,
    847865022, 117917387, 394348397, 921701297, 942290289, 925174924,
)
_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
               if (r, c) != (0, 0))
_KINDS = ("rectangle", "taper", "stepped", "asymmetric", "waist")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid of size 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a)
    return (a.shape, a.tobytes())


def _components(mask, diagonal=False):
    """Return masks, in row-major order, of all connected components."""
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    result = []
    h, w = mask.shape
    directions = _EIGHT if diagonal else _FOUR
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        part = np.zeros_like(mask)
        pending = [(r, c)]
        seen[r, c] = True
        while pending:
            y, x = pending.pop()
            part[y, x] = True
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    pending.append((ny, nx))
        result.append(part)
    return result


def _reach(mask, start):
    h, w = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    if not mask[start]:
        return result
    result[start] = True
    todo = deque([start])
    while todo:
        r, c = todo.popleft()
        for dr, dc in _FOUR:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w
                    and mask[nr, nc] and not result[nr, nc]):
                result[nr, nc] = True
                todo.append((nr, nc))
    return result


def _boundary(filled):
    p = np.pad(filled, 1, constant_values=False)
    inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
    inner &= p[1:-1, :-2] & p[1:-1, 2:]
    return filled & ~inner


def _interior_by_flood(boundary):
    """Flood on a padded canvas, including when an outline touches an edge."""
    blocked = np.pad(boundary, 1, constant_values=False)
    outside = _reach(~blocked, (0, 0))[1:-1, 1:-1]
    return ~boundary & ~outside


def _near(mask):
    """Chebyshev dilation: seed must be disconnected from its half-outline."""
    h, w = mask.shape
    p = np.pad(mask, 1, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            result |= p[dy:dy + h, dx:dx + w]
    return result


def _normalize(a):
    candidates = []
    for k in range(4):
        g = np.rot90(a, k)
        color = int(g[-1, 0])
        if not np.all(g[-1] == color):
            continue
        t = 0
        while t < g.shape[0] and np.all(g[-1 - t] == color):
            t += 1
        expected = np.zeros(g.shape, dtype=bool)
        expected[g.shape[0] - t:] = True
        if 0 < t < g.shape[0] and np.array_equal(g == color, expected):
            candidates.append((g, k, color, t))
    if len(candidates) != 1:
        raise ValueError("The edge mirror must be unique")
    return candidates[0]


def _segment(mask):
    cells = np.argwhere(mask)
    if not 2 <= len(cells) <= 4:
        return None
    dif = np.diff(cells, axis=0)
    if (not np.all(dif[:, 0] == 1) or abs(int(dif[0, 1])) != 1
            or not np.all(dif[:, 1] == dif[0, 1])):
        return None
    return cells, int(dif[0, 1])


def _trace(shape, cue_cells, slope, bounce_row):
    """Input-only stepwise A-style ray tracing; includes the visible cue."""
    h, w = shape
    incoming = np.zeros(shape, dtype=bool)
    outgoing = np.zeros(shape, dtype=bool)
    for r, c in cue_cells:
        incoming[int(r), int(c)] = True
    r, c = (int(x) for x in cue_cells[-1])
    if r >= bounce_row:
        raise ValueError("The cue must precede the mirror")
    while r < bounce_row:
        r, c = r + 1, c + slope
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError("The incoming ray misses the mirror")
        incoming[r, c] = True
    bounce_col = c
    r, c = r - 1, c + slope
    while 0 <= r < h and 0 <= c < w:
        outgoing[r, c] = True
        r, c = r - 1, c + slope
    return incoming, outgoing, bounce_col


def _analytic_cut(shape, start_row, bounce_row, bounce_col, slope):
    """Closed coordinate formulas, used for construction and cross-checking."""
    yy, xx = np.indices(shape)
    incoming = ((yy >= start_row) & (yy <= bounce_row)
                & (xx == bounce_col - slope * (bounce_row - yy)))
    outgoing = ((yy < bounce_row)
                & (xx == bounce_col + slope * (bounce_row - yy)))
    return incoming, outgoing


def _decode(input_grid):
    a = _grid(input_grid)
    g, k, mirror, t = _normalize(a)
    values, counts = np.unique(g, return_counts=True)
    if len(values) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("Four colors and a uniquely dominant background required")
    bg = int(values[np.argmax(counts)])
    if bg == mirror:
        raise ValueError("Mirror cannot be the background")
    options = []
    for color in values:
        color = int(color)
        if color not in (bg, mirror):
            segment = _segment(g == color)
            if segment is not None:
                options.append((color, segment))
    if len(options) != 1:
        raise ValueError("Cue must be one uniquely identifiable diagonal segment")
    cue, (cue_cells, slope) = options[0]
    fg = next(int(c) for c in values if int(c) not in (bg, mirror, cue))
    parts = _components(g == fg, diagonal=True)
    parts.sort(key=lambda p: int(p.sum()))
    if len(parts) != 2 or parts[0].sum() != 1 or parts[1].sum() < 6:
        raise ValueError("Contour color needs a half-outline and an isolated dot")
    dot, half = parts
    seed = tuple(int(x) for x in np.argwhere(dot)[0])
    rows, cols = np.nonzero(half)
    seam = int(cols.max())
    reflected = 2 * seam + 1 - cols
    if reflected.max() >= g.shape[1] or reflected.min() < 0:
        raise ValueError("Completed outline does not fit")
    perimeter = half.copy()
    perimeter[rows, reflected] = True
    interior = _interior_by_flood(perimeter)
    if not interior.any() or not interior[seed]:
        raise ValueError("The dot must lie in the hidden strict interior")
    b = g.shape[0] - t - 1
    incoming, outgoing, bc = _trace(g.shape, cue_cells, slope, b)
    cut = incoming | outgoing
    if cut[seed]:
        raise ValueError("Dot cannot be on the cut")
    kept = _reach(interior & ~cut, seed)
    return dict(g=g, rotation=k, bg=bg, fg=fg, mirror=mirror, cue=cue,
                thickness=t, half=half, seed=seed, seam=seam,
                perimeter=perimeter, interior=interior, incoming=incoming,
                outgoing=outgoing, cut=cut, kept=kept, slope=slope,
                cue_cells=cue_cells, bounce_row=b, bounce_col=bc,
                delta=2 * bc - 2 * seam - 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the marked cut component from this input only."""
    d = _decode(input_grid)
    result = np.full(d["g"].shape, d["bg"], dtype=np.int8)
    result[d["kept"]] = d["fg"]
    return np.rot90(result, -d["rotation"]).copy()


def _row_sheet(half, seam):
    filled = np.zeros_like(half)
    for r in np.flatnonzero(half.any(axis=1)):
        c = int(np.flatnonzero(half[r])[0])
        filled[r, c:2 * seam + 2 - c] = True
    return filled


def _piece_kind(part, bounce_row, bounce_col, slope):
    y, x = np.nonzero(part)
    side = slope * (x - bounce_col) - (bounce_row - y)
    if np.all(side > 0):
        return "outer"
    other = slope * (x - bounce_col) + (bounce_row - y)
    if np.all(side < 0) and np.all(other > 0):
        return "middle"
    return "other"


def _bbox_interior(filled):
    y, x = np.nonzero(filled)
    box = np.zeros_like(filled)
    box[int(y.min()):int(y.max()) + 1, int(x.min()):int(x.max()) + 1] = True
    return box & ~_boundary(box)


def _profile(rng, kind, radius, height):
    if kind == "rectangle":
        return [radius] * height
    if kind == "waist":
        anchors = [0, height // 4, height // 2, 3 * height // 4, height - 1]
        radii = [int(rng.integers(1, radius - 1)), radius,
                 max(2, radius - int(rng.integers(2, 5))),
                 radius - int(rng.integers(0, 2)),
                 int(rng.integers(1, radius - 1))]
    else:
        peak = int(rng.integers(2, height - 2))
        anchors = [0, peak, height - 1]
        radii = [int(rng.integers(1, radius - 1)), radius,
                 int(rng.integers(1, radius - 1))]
    p = np.rint(np.interp(np.arange(height), anchors, radii)).astype(int)
    if kind == "stepped":
        p = 1 + 2 * ((p - 1) // 2)
        p[int(np.argmax(p))] = radius
    elif kind == "asymmetric":
        p[1] = max(1, int(p[1]) - 1)
        p[-2] = min(radius, int(p[-2]) + 1)
    return [int(x) for x in p]


def _make_pair(rng, *, kind, selection, rotation, displacement, seen,
               rank_largest=None):
    """Construct material and analytic cuts independently of solve()."""
    for _ in range(1800):
        radius = int(rng.integers(3 if displacement <= 3 else 5, 10))
        height = int(rng.integers(6 if kind != "waist" else 8, 13))
        profile = _profile(rng, kind, radius, height)
        radius = max(profile)
        left = int(rng.integers(0, 7))
        right = int(rng.integers(0, 7))
        width = 2 * radius + 2 + left + right
        top = int(rng.integers(4, 11))
        gap = int(rng.integers(0, 4))
        thickness = int(rng.integers(1, 5))
        bounce_row = top + height - 1 + gap
        total_height = bounce_row + thickness + 1
        if width > 30 or total_height > 30:
            continue
        seam = left + radius
        delta = displacement * int(rng.choice((-1, 1)))
        bc = seam + (delta + 1) // 2
        if not 0 <= bc < width:
            continue
        slope = int(rng.choice((-1, 1)))
        cue_length = int(rng.integers(2, 5))
        starts = []
        for y in range(top - cue_length + 1):
            x = bc - slope * (bounce_row - y)
            end_x = x + slope * (cue_length - 1)
            if 0 <= x < width and 0 <= end_x < width:
                starts.append((y, x))
        if not starts:
            continue
        sy, sx = starts[int(rng.integers(len(starts)))]
        filled = np.zeros((total_height, width), dtype=bool)
        for j, rad in enumerate(profile):
            filled[top + j, seam - rad:seam + rad + 2] = True
        boundary = _boundary(filled)
        half = boundary.copy()
        half[:, seam + 1:] = False
        interior = filled & ~boundary
        incoming, outgoing = _analytic_cut(filled.shape, sy, bounce_row, bc, slope)
        cut = incoming | outgoing
        parts = _components(interior & ~cut)
        if len(parts) < 3 or not (interior & incoming).any() or not (interior & outgoing).any():
            continue
        far_from_half = ~_near(half)
        eligible = {"middle": [], "outer": []}
        bbox = _bbox_interior(filled)
        for part in parts:
            category = _piece_kind(part, bounce_row, bc, slope)
            if category not in eligible or part.sum() < 4:
                continue
            sites = np.argwhere(part & far_from_half)
            if not len(sites):
                continue
            seed = tuple(int(x) for x in sites[0])
            if np.array_equal(_reach(interior & ~incoming, seed), part):
                continue
            if kind != "rectangle" and np.array_equal(_reach(bbox & ~cut, seed), part):
                continue
            eligible[category].append((part, sites))
        if not eligible["middle"] or not eligible["outer"]:
            continue
        part, sites = eligible[selection][int(rng.integers(len(eligible[selection])))]
        if rank_largest is not None:
            other_sizes = [int(p.sum()) for p in parts if not np.array_equal(p, part)]
            if rank_largest and int(part.sum()) <= max(other_sizes):
                continue
            if not rank_largest and int(part.sum()) >= max(other_sizes):
                continue
        # The center demonstration must show material on the missing side.
        if selection == "middle" and part[:, seam + 1:].sum() < 2:
            continue
        seed = tuple(int(x) for x in sites[int(rng.integers(len(sites)))])
        bg, mirror, fg, cue = (int(x) for x in rng.choice(10, 4, replace=False))
        g = np.full(filled.shape, bg, dtype=np.int8)
        g[bounce_row + 1:] = mirror
        g[half] = fg
        g[seed] = fg
        for j in range(cue_length):
            g[sy + j, sx + slope * j] = cue
        if np.count_nonzero(g == bg) <= max(np.count_nonzero(g == c) for c in (mirror, fg, cue)):
            continue
        out = np.full_like(g, bg)
        out[part] = fg
        pair = {"input": np.rot90(g, rotation).copy(),
                "output": np.rot90(out, rotation).copy()}
        if _signature(pair["input"]) in seen:
            continue
        seen.add(_signature(pair["input"]))
        return pair
    raise RuntimeError("Could not construct a clear sheet with two effective cut arms")


def _move_dot(pair, rng, seen):
    d = _decode(pair["input"])
    current_kind = _piece_kind(d["kept"], d["bounce_row"], d["bounce_col"], d["slope"])
    target = "outer" if current_kind == "middle" else "middle"
    filled = d["interior"] | d["perimeter"]
    is_rect = len(set(_profile_of(d))) == 1
    for part in _components(d["interior"] & ~d["cut"]):
        if _piece_kind(part, d["bounce_row"], d["bounce_col"], d["slope"]) != target:
            continue
        sites = np.argwhere(part & ~_near(d["half"]))
        if part.sum() < 4 or not len(sites):
            continue
        site = tuple(int(x) for x in sites[int(rng.integers(len(sites)))])
        if np.array_equal(part, _reach(d["interior"] & ~d["incoming"], site)):
            continue
        if not is_rect and np.array_equal(part, _reach(_bbox_interior(filled) & ~d["cut"], site)):
            continue
        g = d["g"].copy()
        g[d["seed"]] = d["bg"]
        g[site] = d["fg"]
        out = np.full_like(g, d["bg"])
        out[part] = d["fg"]
        result = {"input": np.rot90(g, -d["rotation"]).copy(),
                  "output": np.rot90(out, -d["rotation"]).copy()}
        sig = _signature(result["input"])
        if sig not in seen:
            seen.add(sig)
            return result
    raise RuntimeError("Constructed scaffold lost its alternate dot site")


def _checked_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _checked_count(num_train, 2, 10, "num_train")
    m = _checked_count(num_test, 1, 4, "num_test")
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_rotation = int(rng.integers(4))
    kinds = ["rectangle", str(rng.choice(_KINDS[1:]))]
    rng.shuffle(kinds)
    selections = ["middle", "outer"]
    rng.shuffle(selections)
    displacements = [1, 3]
    rng.shuffle(displacements)
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(2):
        puzzle["train"].append(_make_pair(
            rng, kind=kinds[i], selection=selections[i],
            rotation=(base_rotation + i) % 4,
            displacement=displacements[i], seen=seen,
            rank_largest=selections[i] == "middle"))
    if n >= 3:
        puzzle["train"].append(_move_dot(puzzle["train"][0], rng, seen))
    while len(puzzle["train"]) < n:
        puzzle["train"].append(_make_pair(
            rng, kind=str(rng.choice(_KINDS)),
            selection=str(rng.choice(("middle", "outer"))),
            rotation=int(rng.integers(4)), displacement=int(rng.choice((1, 3))), seen=seen))
    profiles = {tuple(_profile_of(_decode(p["input"]))) for p in puzzle["train"]}
    for _ in range(m):
        for attempt in range(100):
            pair = _make_pair(
                rng, kind=str(rng.choice(_KINDS[1:])),
                selection=str(rng.choice(("middle", "outer"))),
                rotation=int(rng.integers(4)), displacement=int(rng.choice((5, 7))), seen=seen)
            profile = tuple(_profile_of(_decode(pair["input"])))
            if profile not in profiles:
                profiles.add(profile)
                puzzle["test"].append(pair)
                break
        else:
            raise RuntimeError("Could not find a new test contour")
    validate_generated_puzzle(puzzle)
    return puzzle


def _profile_of(d):
    return tuple(d["seam"] - int(np.flatnonzero(d["half"][r])[0])
                 for r in np.flatnonzero(d["half"].any(axis=1)))


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, out = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == out.shape
    d = _decode(a)
    half, seam, interior = d["half"], d["seam"], d["interior"]
    rows = np.flatnonzero(half.any(axis=1))
    assert len(rows) >= 3 and np.all(np.diff(rows) == 1)
    assert half[rows[0], seam] and half[rows[-1], seam]
    for r in rows:
        columns = np.flatnonzero(half[r])
        assert np.all(np.diff(columns) == 1)
    filled = _row_sheet(half, seam)
    boundary = _boundary(filled)
    assert np.array_equal(boundary, d["perimeter"])
    assert np.array_equal(filled & ~boundary, interior)
    assert len(_components(half, True)) == 1
    assert len(_components(boundary, True)) == 1
    assert len(_components(interior)) == 1
    assert not np.any(filled[d["bounce_row"] + 1:])
    assert int(d["cue_cells"][-1, 0]) < int(rows[0])
    analytic_in, analytic_out = _analytic_cut(
        d["g"].shape,
        int(d["cue_cells"][0, 0]), d["bounce_row"], d["bounce_col"], d["slope"])
    assert np.array_equal(analytic_in, d["incoming"])
    assert np.array_equal(analytic_out, d["outgoing"])
    assert (interior & d["incoming"]).any() and (interior & d["outgoing"]).any()
    assert len(_components(interior & ~d["cut"])) >= 3
    kept = d["kept"]
    assert kept.sum() >= 4 and kept[d["seed"]]
    assert len(_components(kept)) == 1
    assert not np.any(kept & (boundary | d["cut"]))
    assert not np.array_equal(kept, _reach(interior & ~d["incoming"], d["seed"]))
    profile = _profile_of(d)
    if len(set(profile)) > 1:
        assert not np.array_equal(kept, _reach(_bbox_interior(filled) & ~d["cut"], d["seed"]))
    normalized_out = np.rot90(out, d["rotation"])
    expected = np.full_like(d["g"], d["bg"])
    expected[kept] = d["fg"]
    assert np.array_equal(expected, normalized_out)
    assert np.array_equal(out, solve(a))
    assert set(np.unique(out)) == {d["bg"], d["fg"]}
    assert not np.array_equal(a, out)
    d["profile"] = profile
    d["selection"] = _piece_kind(kept, d["bounce_row"], d["bounce_col"], d["slope"])
    assert d["selection"] in ("middle", "outer")
    return d


def validate_generated_puzzle(puzzle):
    """Check returned grids and family-specific invariants; no latent metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    analyses = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            sig = _signature(_grid(pair["input"]))
            assert sig not in seen
            seen.add(sig)
            d = _analyze_pair(pair)
            assert 1 <= d["thickness"] <= 4
            assert abs(d["delta"]) in ((1, 3) if split == "train" else (5, 7))
            analyses[split].append(d)
    train, test = analyses["train"], analyses["test"]
    assert {len(set(d["profile"])) == 1 for d in train[:2]} == {False, True}
    assert {d["selection"] for d in train[:2]} == {"middle", "outer"}
    assert {abs(d["delta"]) for d in train[:2]} == {1, 3}
    assert train[0]["rotation"] % 2 != train[1]["rotation"] % 2
    for d in train[:2]:
        sizes = [int(p.sum()) for p in _components(d["interior"] & ~d["cut"])
                 if not np.array_equal(p, d["kept"])]
        if d["selection"] == "middle":
            assert int(d["kept"].sum()) > max(sizes)
        else:
            assert int(d["kept"].sum()) < max(sizes)
    assert any(d["kept"][:, d["seam"] + 1:].sum() >= 2 for d in train)
    assert all(d["profile"] not in {a["profile"] for a in train} for d in test)
    if n >= 3:
        a, b = train[0], train[2]
        assert a["rotation"] == b["rotation"] and a["seed"] != b["seed"]
        ga, gb = a["g"].copy(), b["g"].copy()
        ga[a["seed"]] = a["bg"]
        gb[b["seed"]] = b["bg"]
        assert np.array_equal(ga, gb)
        assert a["selection"] != b["selection"]
        assert not np.any(a["kept"] & b["kept"])


def _encode(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                           for p in puzzle[s]] for s in ("train", "test")},
                      separators=(",", ":"))


def self_test():
    """Bounded reproducibility, counterexample, count, and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_encode(p) for p in puzzles}) == 42
    assert _encode(puzzles[0]) == _encode(generate_puzzle(0))
    assert _encode(generate_puzzle(811046817, 7, 3)) == _encode(generate_puzzle(811046817, 7, 3))
    shapes, profiles, palettes, rotations, slopes, deltas, lengths, thicknesses = (set() for _ in range(8))
    selections, structures = set(), set()
    asymmetric = stepped = waist = 0
    for p in puzzles:
        for split in ("train", "test"):
            for pair in p[split]:
                d = _analyze_pair(pair)
                g = d["g"]
                shapes.add(g.shape)
                profiles.add(d["profile"])
                palettes.add((d["bg"], d["mirror"], d["fg"], d["cue"]))
                rotations.add(d["rotation"])
                slopes.add(d["slope"])
                deltas.add(d["delta"])
                lengths.add(len(d["cue_cells"]))
                thicknesses.add(d["thickness"])
                selections.add(d["selection"])
                profile = d["profile"]
                asymmetric += profile != profile[::-1]
                stepped += any(abs(x - y) >= 2 for x, y in zip(profile, profile[1:]))
                waist += any(profile[i] < profile[i - 1] and profile[i] < profile[i + 1]
                             for i in range(1, len(profile) - 1))
                normalized = np.zeros_like(g)
                for role, color in enumerate((d["bg"], d["mirror"], d["fg"], d["cue"])):
                    normalized[g == color] = role
                structures.add(_signature(normalized))
    assert len(shapes) >= 35 and len(profiles) >= 65
    assert len(palettes) >= 80 and len(structures) >= 140
    assert rotations == {0, 1, 2, 3} and slopes == {-1, 1}
    assert deltas == {-7, -5, -3, -1, 1, 3, 5, 7}
    assert lengths == {2, 3, 4} and thicknesses == {1, 2, 3, 4}
    assert selections == {"middle", "outer"}
    assert asymmetric >= 25 and stepped >= 15 and waist >= 3
    # The solver must recognize roles and the mirror's frame after recoloring
    # and rotation, without being given the original example's metadata.
    permutation = np.array([5, 8, 0, 6, 9, 2, 7, 1, 4, 3], dtype=np.int8)
    for pair in puzzles[0]["train"][:2]:
        for k in range(4):
            transformed = permutation[np.rot90(pair["input"], k)]
            expected = permutation[np.rot90(pair["output"], k)]
            assert np.array_equal(solve(transformed), expected)
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(1000 + 10 * n + m, n, m)
            assert (len(p["train"]), len(p["test"])) == (n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(9, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # Confirm that the validator rejects a plausible wrong output, not just bad shapes.
    wrong = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
             for s in ("train", "test")}
    d = _decode(wrong["train"][0]["input"])
    bad = np.full_like(d["g"], d["bg"])
    bad[d["interior"]] = d["fg"]
    wrong["train"][0]["output"] = np.rot90(bad, -d["rotation"]).copy()
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("Whole-interior shortcut was not rejected")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 host seeds, all legal counts)")
