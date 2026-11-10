"""Registered coincidence loom. No parent imports or hidden solver state.

Generation builds a canonical code; solve reconstructs it in the orientation
encoded by two outer identities. Only standard-library Python and NumPy.
"""

from collections import Counter
import numpy as np


SIDES = ("T", "R", "B", "L")
ADJACENT = {frozenset((SIDES[i], SIDES[(i + 1) % 4])) for i in range(4)}


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _side(a, s):
    return {"T": a[0, :], "R": a[:, -1], "B": a[-1, :], "L": a[:, 0]}[s]


def _side_maps():
    cross = np.array([[0, 1, 0], [4, 0, 2], [0, 3, 0]])
    return [{s: next(target for target in SIDES
                     if int(_side(_orient(cross, t), target)[1]) == i + 1)
             for i, s in enumerate(SIDES)} for t in range(8)]


MAPS = _side_maps()


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid extent"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=False)


def _components(a, color):
    remaining = set(map(tuple, np.argwhere(a == color).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for pos in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if pos in remaining:
                    remaining.remove(pos)
                    stack.append(pos)
        result.append(cells)
    return result


def _frame(a):
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), "corner convention"
    controls = {}
    for s in SIDES:
        core = _side(a, s)[1:-1]
        assert core.size >= 3
        values = np.unique(core)
        assert len(values) == 1, "outer side must be clean"
        if int(values[0]) != bg:
            controls[s] = int(values[0])
    assert frozenset(controls) in ADJACENT and len(set(controls.values())) == 2
    return bg, controls


def _bars(a, markers):
    bars = []
    H, W = a.shape
    for color in sorted(markers):
        for cells in _components(a, color):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells):
                continue
            rs, cs = zip(*cells)
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            assert 3 <= len(cells) <= 8
            assert (r0 == r1 or c0 == c1) and len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
            bars.append(dict(r0=r0, r1=r1, c0=c0, c1=c1, color=color,
                             axis="H" if r0 == r1 else "V"))
    return bars


def _partner(hbar, vbar):
    """Recover a rectangle from two diagonally adjoining guide endpoints."""
    rr, c0, c1 = hbar["r0"], hbar["c0"], hbar["c1"]
    cc, r0, r1 = vbar["c0"], vbar["r0"], vbar["r1"]
    hs = "T" if r0 == rr + 1 else "B" if r1 == rr - 1 else None
    vs = "L" if cc == c0 - 1 else "R" if cc == c1 + 1 else None
    if hs is None or vs is None or hbar["color"] == vbar["color"]:
        return None
    return (r0, c0, r1 - r0 + 1, c1 - c0 + 1), {hs: hbar["color"], vs: vbar["color"]}


def _weave(shape, bg, rows, cols):
    result = np.full(shape, bg, dtype=np.int8)
    for r, x in rows.items():
        for c, y in cols.items():
            if x == y:
                result[r, c] = x
    return result


def _decode(input_grid):
    """Input-only extraction, registration, and agreement checks."""
    a = _grid(input_grid)
    H, W = a.shape
    bg, controls = _frame(a)
    markers = set(controls.values())
    bars = _bars(a, markers)
    assert 4 <= len(bars) <= 10 and len(bars) % 2 == 0
    registrations, uses = [], Counter()
    for hi, hb in enumerate(bars):
        if hb["axis"] != "H":
            continue
        for vi, vb in enumerate(bars):
            if vb["axis"] != "V":
                continue
            candidate = _partner(hb, vb)
            if candidate is not None:
                bounds, guides = candidate
                registrations.append((bounds, guides))
                uses.update((hi, vi))
    assert len(registrations) * 2 == len(bars) and all(uses[i] == 1 for i in range(len(bars))), "ambiguous guide partners"
    accounted = np.zeros(a.shape, dtype=bool)
    accounted[[0, -1], :] = True
    accounted[:, [0, -1]] = True
    footprints = np.zeros(a.shape, dtype=bool)
    rows, cols, station_info = {}, {}, []
    shape = None
    for (r, c, h, w), guides in sorted(registrations, key=lambda item: item[0]):
        assert 3 <= h <= 8 and 3 <= w <= 8
        assert r >= 3 and c >= 3 and r + h <= H - 3 and c + w <= W - 3
        assert not footprints[r - 2:r + h + 2, c - 2:c + w + 2].any(), "registrations need a gap"
        footprints[r - 1:r + h + 1, c - 1:c + w + 1] = True
        accounted[r - 1:r + h + 1, c - 1:c + w + 1] = True
        patch = a[r - 1:r + h + 1, c - 1:c + w + 1].copy()
        assert np.all(patch[1:-1, 1:-1] == bg), "interiors must be empty"
        assert all(int(patch[p]) == bg for p in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
        for s in SIDES:
            line = _side(patch, s)[1:-1]
            if s in guides:
                assert np.all(line == guides[s])
            else:
                assert not set(map(int, line)) & markers, "marker on signal side"
        transformations = [t for t, mapping in enumerate(MAPS)
                           if all(controls.get(mapping[s]) == color for s, color in guides.items())]
        assert len(transformations) == 1, "nonunique registration"
        t = transformations[0]
        aligned = _orient(patch, t)
        current_shape = (aligned.shape[0] - 2, aligned.shape[1] - 2)
        if shape is None:
            shape = current_shape
        assert shape == current_shape, "aligned extents differ"
        own_rows, own_cols = {}, {}
        for s in SIDES:
            if s in controls:
                continue
            own = own_rows if s in ("L", "R") else own_cols
            pooled = rows if s in ("L", "R") else cols
            for slot, value in enumerate(_side(aligned, s)[1:-1]):
                color = int(value)
                if color != bg:
                    assert slot not in pooled or pooled[slot] == color, "contradictory signal"
                    pooled[slot] = own[slot] = color
        station_info.append(dict(bounds=(r, c, h, w), guides=guides, transform=t,
                                 rows=own_rows, cols=own_cols))
    assert np.all(a[~accounted] == bg), "unaccounted input content"
    assert rows and cols and set(rows.values()) & set(cols.values()), "empty coincidence image"
    assert 1 <= len(set(rows.values()) | set(cols.values())) <= 4
    result = _weave(shape, bg, rows, cols)
    return result, dict(background=bg, controls=controls, shape=shape,
                        rows=rows, cols=cols, stations=station_info)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _decode(input_grid)[0]


def _paint_frame(a, controls):
    for s, color in controls.items():
        _side(a, s)[1:-1] = color


def _pack(rng, patches, bg, controls):
    """Free-position packing first; finite spacious placement fallback."""
    longest = max(max(p.shape) for p in patches)
    low = max(12, longest + 5, 18 if len(patches) >= 3 else 12,
              24 if len(patches) == 5 else 12)
    for _ in range(12):
        H, W = map(int, rng.integers(low, 31, size=2))
        if rng.random() < .25:
            W = H
        occupied = np.zeros((H, W), dtype=bool)
        placements = []
        for patch in patches:
            ph, pw = patch.shape
            candidates = [(r, c) for r in range(2, H - ph - 1)
                          for c in range(2, W - pw - 1)]
            rng.shuffle(candidates)
            for r, c in candidates:
                if not occupied[r - 1:r + ph + 1, c - 1:c + pw + 1].any():
                    occupied[r:r + ph, c:c + pw] = True
                    placements.append((r, c))
                    break
            else:
                break
        if len(placements) == len(patches):
            break
    else:
        H = W = 30
        stride = longest + 1
        ncols = 3 if len(patches) == 5 else 2
        positions = [(2 + (i // ncols) * stride, 2 + (i % ncols) * stride)
                     for i in range(len(patches))]
        rng.shuffle(positions)
        placements = positions
    a = np.full((H, W), bg, dtype=np.int8)
    _paint_frame(a, controls)
    for patch, (r, c) in zip(patches, placements):
        ph, pw = patch.shape
        assert r + ph <= H - 2 and c + pw <= W - 2
        a[r:r + ph, c:c + pw] = patch
    return a


def _codes(rng, h, w, signal_colors, teaching):
    def slots(n):
        available = list(range(n))
        if teaching or rng.random() < .45:
            available.remove(int(rng.choice([0, n - 1])))
        count = int(rng.integers(2 if teaching else 1, min(6, len(available)) + 1))
        return sorted(map(int, rng.choice(available, count, replace=False)))
    rr, cc = slots(h), slots(w)
    rows = {r: int(rng.choice(signal_colors)) for r in rr}
    cols = {c: int(rng.choice(signal_colors)) for c in cc}
    # Ensure an observable answer. Teaching has two colors on both axes,
    # which exposes equality and unlike-color suppression in the same picture.
    rows[rr[0]] = cols[cc[0]] = signal_colors[0]
    if teaching:
        rows[rr[1]] = cols[cc[1]] = signal_colors[1]
    return rows, cols


def _make_pair(rng, teaching=False, forbidden_corner=None, forbidden_shapes=()):
    count = 3 if teaching else int(rng.integers(2, 6))
    max_side = 6 if count == 5 else 8
    for _ in range(64):
        h, w = map(int, rng.integers(4 if teaching else 3, max_side + 1, size=2))
        if teaching and h == w:
            continue
        key_choices = [t for t in range(8)
                       if forbidden_corner is None or
                       frozenset((MAPS[t]["T"], MAPS[t]["L"])) != forbidden_corner]
        kt = int(rng.choice(key_choices))
        out_shape = _orient(np.empty((h, w), dtype=np.int8), kt).shape
        if out_shape not in forbidden_shapes:
            break
    else:
        raise AssertionError("shape sampling exhausted")
    palette = list(map(int, rng.permutation(10)))
    bg, ma, mb = palette[:3]
    signals = palette[3:3 + (2 if teaching else int(rng.integers(1, 5)))]
    for _ in range(64):
        rows, cols = _codes(rng, h, w, signals, teaching)
        image = _weave((h, w), bg, rows, cols)
        if not teaching or len({_key(_orient(image, t)) for t in range(8)}) == 8:
            break
    else:
        # Deterministic asymmetric two-color evidence with blank far edges.
        rows = {0: signals[0], 1: signals[1]}
        cols = {0: signals[0], 1: signals[1]}
    assignments = [[] for _ in range(count)]
    observations = [("row", s, x) for s, x in rows.items()] + [("col", s, x) for s, x in cols.items()]
    rng.shuffle(observations)
    mode = "split" if teaching else str(rng.choice(["split", "full", "spread"]))
    full_owner = int(rng.integers(count))
    for i, obs in enumerate(observations):
        owner = i % (3 if teaching else 2) if mode == "split" else full_owner if mode == "full" else int(rng.integers(count))
        assignments[owner].append(obs)
        if not teaching and rng.random() < .2:
            other = int(rng.integers(count))
            if obs not in assignments[other]:
                assignments[other].append(obs)
    poses = [0, 1, 4] if teaching else list(map(int, rng.integers(8, size=count)))
    rng.shuffle(poses)
    patches = []
    for pose, observed in zip(poses, assignments):
        patch = np.full((h + 2, w + 2), bg, dtype=np.int8)
        patch[0, 1:-1] = ma
        patch[1:-1, 0] = mb
        for axis, slot, color in observed:
            if axis == "row":
                patch[slot + 1, -1] = color
            else:
                patch[-1, slot + 1] = color
        patches.append(_orient(patch, pose))
    controls = {MAPS[kt]["T"]: ma, MAPS[kt]["L"]: mb}
    inp = _pack(rng, patches, bg, controls)
    out = _orient(_weave((h, w), bg, rows, cols), kt)
    return {"input": inp, "output": out}


def _contrast(pair):
    _, info = _decode(pair["input"])
    a = pair["input"].copy()
    swapped = dict(zip(info["controls"], reversed(list(info["controls"].values()))))
    _paint_frame(a, swapped)
    t = next(t for t, m in enumerate(MAPS)
             if all(swapped.get(m[s]) == x for s, x in info["controls"].items()))
    return {"input": a, "output": _orient(pair["output"], t)}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first = _make_pair(rng, teaching=True)
    puzzle = {"train": [first, _contrast(first)], "test": []}
    _, initial = _decode(first["input"])
    used = {_key(p["input"]) for p in puzzle["train"]}
    previous_bg = initial["background"]
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            for attempt in range(64):
                pair = _make_pair(rng,
                                  forbidden_corner=frozenset(initial["controls"]) if split == "test" else None,
                                  forbidden_shapes=(initial["shape"], initial["shape"][::-1]) if split == "test" else ())
                _, info = _decode(pair["input"])
                if (_key(pair["input"]) not in used and info["background"] != previous_bg
                        and (split != "test" or info["background"] != initial["background"])):
                    break
            else:
                raise AssertionError("distinct example sampling exhausted")
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            previous_bg = info["background"]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, teaching = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            expected, info = _decode(a)
            assert np.array_equal(output, expected), "incorrect coincidence output"
            bg, rows, cols = info["background"], info["rows"], info["cols"]
            row_counts, col_counts = Counter(rows.values()), Counter(cols.values())
            for x in set(row_counts) | set(col_counts):
                assert int(np.count_nonzero(output == x)) == row_counts[x] * col_counts[x]
            assert np.count_nonzero(output != bg) > 0
            if split == "train":
                teaching.append((a, output, info))
    # Evidence constraints are checked separately from the input-only rule.
    assert any(all(not np.array_equal(_weave(i["shape"], i["background"], s["rows"], s["cols"]), out)
                   for s in i["stations"]) for _, out, i in teaching), "pooling witness absent"
    assert any(len(set(i["rows"].values()) & set(i["cols"].values())) >= 2
               and any(x != y for x in i["rows"].values() for y in i["cols"].values())
               for _, _, i in teaching), "equality witness absent"
    assert any(np.all(_side(out, s) == i["background"]) for _, out, i in teaching for s in SIDES), "extent witness absent"
    pose_witness = False
    for _, _, i in teaching:
        ts = [s["transform"] for s in i["stations"] if s["rows"] or s["cols"]]
        pose_witness |= (len({t >= 4 for t in ts}) == 2 and
                         any((a >= 4) == (b >= 4) and (a - b) % 4 in (1, 3)
                             for a in ts for b in ts))
    assert pose_witness, "clues must demonstrate relative quarter-turn and reflection"
    contrasts = False
    for n, (a, out, i) in enumerate(teaching):
        for b, out2, j in teaching[n + 1:]:
            if a.shape != b.shape or set(i["controls"]) != set(j["controls"]):
                continue
            if np.array_equal(a[1:-1, 1:-1], b[1:-1, 1:-1]) and i["controls"] != j["controls"]:
                contrasts |= (not np.array_equal(out, out2)
                              and len({_key(_orient(out, t)) for t in range(8)}) == 8
                              and len({_key(_orient(out2, t)) for t in range(8)}) == 8)
    assert contrasts, "identity-changing controlled contrast absent"


def _features(pair):
    _, i = _decode(pair["input"])
    return dict(canvas=pair["input"].shape, shape=i["shape"], count=len(i["stations"]),
                corner=tuple(sorted(i["controls"])), background=i["background"],
                markers=tuple(sorted(i["controls"].values())),
                slots=(tuple(sorted(i["rows"])), tuple(sorted(i["cols"]))),
                active=(len(i["rows"]), len(i["cols"])),
                signals=len(set(i["rows"].values()) | set(i["cols"].values())),
                transforms=tuple(s["transform"] for s in i["stations"]),
                allocations=tuple(len(s["rows"]) + len(s["cols"]) for s in i["stations"]),
                repeated=any(len(set(d.values())) < len(d) for d in (i["rows"], i["cols"])),
                redundant=sum(len(s["rows"]) + len(s["cols"]) for s in i["stations"]) > len(i["rows"]) + len(i["cols"]),
                exclusive=bool(set(i["rows"].values()) ^ set(i["cols"].values())),
                blank=any(np.all(_side(pair["output"], s) == i["background"]) for s in SIDES))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_features(p) for puzzle in puzzles for split in ("train", "test") for p in puzzle[split]]
    assert len({tuple(_key(p["input"]) for p in pu["train"] + pu["test"]) for pu in puzzles}) == 42
    assert {f["count"] for f in features} == {2, 3, 4, 5}
    assert {t for f in features for t in f["transforms"]} == set(range(8))
    assert {frozenset(f["corner"]) for f in features} == ADJACENT
    assert len({f["shape"] for f in features}) >= 20
    assert len({f["canvas"] for f in features}) >= 35
    assert len({f["slots"] for f in features}) >= 60
    assert {f["background"] for f in features} == set(range(10))
    assert {f["signals"] for f in features} >= {1, 2, 3, 4}
    assert all(any(f[prop] for f in features) for prop in ("repeated", "redundant", "exclusive", "blank"))
    original = puzzles[0]
    assert all(np.array_equal(p[role], q[role]) for split in original
               for p, q in zip(original[split], generate_puzzle(0)[split]) for role in p)
    # The entire input key, guides, and observations can be transformed/recolored.
    for t in range(8):
        permutation = np.roll(np.arange(10, dtype=np.int8), t + 1)
        transformed = {split: [{role: _orient(permutation[value], t) for role, value in pair.items()}
                               for pair in original[split]] for split in original}
        validate_generated_puzzle(transformed)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            pu = generate_puzzle(10000 + nt * 10 + ne, nt, ne)
            assert len(pu["train"]) == nt and len(pu["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for kind in ("output", "guide", "crop"):
        bad = {split: [{role: grid.copy() for role, grid in pair.items()} for pair in original[split]]
               for split in original}
        pair = bad["test"][0]
        if kind == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "crop":
            pair["output"] = pair["output"][1:, :].copy()
        else:
            _, i = _decode(pair["input"])
            station = i["stations"][0]
            r, c, h, w = station["bounds"]
            s = next(iter(station["guides"]))
            pos = {"T": (r - 1, c + w // 2), "B": (r + h, c + w // 2),
                   "L": (r + h // 2, c - 1), "R": (r + h // 2, c + w)}[s]
            pair["input"][pos] = i["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("mutation accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations, equivariance, mutations)")
