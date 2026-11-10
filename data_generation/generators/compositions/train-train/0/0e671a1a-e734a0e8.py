"""Exclusive delivery: parity-gated template replication on two ordered routes.

All semantic roles are visible in each input. No training examples, palettes,
random state, or generation metadata are consulted by solve().
"""
from itertools import combinations

import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not 2 <= a <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(b) or not 1 <= b <= 4 or a <= b:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _slice(point, height, width):
    r, c = point
    return (slice(r * (height + 1), r * (height + 1) + height),
            slice(c * (width + 1), c * (width + 1) + width))


def _routes(p, h, v, h_first=True, v_first=False):
    def route(q, horizontal_first):
        bend = (p[0], q[1]) if horizontal_first else (q[0], p[1])
        result = set()
        for a, b in ((p, bend), (bend, q)):
            for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
                for c in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
                    result.add((r, c))
        return result
    return route(h, h_first), route(v, v_first)


def _decode(value):
    """Parse lattice, source tile, center candidates and axis-shaped terminals."""
    a = _grid(value)
    candidates = []
    for color in np.unique(a):
        rr = np.flatnonzero(np.all(a == color, axis=1))
        cc = np.flatnonzero(np.all(a == color, axis=0))
        if len(rr) and len(cc):
            candidates.append((int(color), rr, cc))
    if len(candidates) != 1:
        raise ValueError("expected one complete lattice")
    lattice, rr, cc = candidates[0]
    height, width = int(rr[0]), int(cc[0])
    if (height < 3 or width < 3 or height % 2 != 1 or width % 2 != 1
            or not np.array_equal(rr, np.arange(height, a.shape[0], height + 1))
            or not np.array_equal(cc, np.arange(width, a.shape[1], width + 1))
            or (a.shape[0] + 1) % (height + 1)
            or (a.shape[1] + 1) % (width + 1)):
        raise ValueError("expected equal odd rectangular compartments")
    rows, cols = len(rr) + 1, len(cc) + 1
    others = [int(x) for x in np.unique(a) if x != lattice]
    if len(others) != 2:
        raise ValueError("expected lattice, carrier and source-accent colors")
    carrier, accent = sorted(others, key=lambda x: int(np.count_nonzero(a == x)), reverse=True)
    if np.count_nonzero(a == carrier) <= np.count_nonzero(a == accent):
        raise ValueError("carrier must dominate the accent")
    cy, cx = height // 2, width // 2
    blank = np.full((height, width), carrier, dtype=np.int8)
    center = blank.copy()
    center[cy, cx] = lattice
    horizontal = blank.copy()
    horizontal[cy, cx - 1:cx + 2] = lattice
    vertical = blank.copy()
    vertical[cy - 1:cy + 2, cx] = lattice
    source = h = v = motif = None
    marks, blanks = set(), set()
    for r in range(rows):
        for c in range(cols):
            point = (r, c)
            tile = a[_slice(point, height, width)]
            if np.any(tile == accent):
                if source is not None or not np.all((tile == carrier) | (tile == accent)):
                    raise ValueError("expected exactly one accent-bearing source tile")
                source, motif = point, tile.copy()
            elif np.array_equal(tile, center):
                marks.add(point)
            elif np.array_equal(tile, horizontal):
                if h is not None:
                    raise ValueError("duplicate horizontal terminal")
                h = point
            elif np.array_equal(tile, vertical):
                if v is not None:
                    raise ValueError("duplicate vertical terminal")
                v = point
            elif np.array_equal(tile, blank):
                blanks.add(point)
            else:
                raise ValueError("unrecognized compartment content")
    if source is None or h is None or v is None or not marks:
        raise ValueError("missing source, terminal or candidate marker")
    if not 2 <= np.count_nonzero(motif == accent) <= motif.size - 2:
        raise ValueError("source must have at least two accent and two carrier cells")
    return dict(grid=a, layout=(rows, cols, height, width), lattice=lattice,
                carrier=carrier, accent=accent, source=source, h=h, v=v,
                motif=motif, marks=marks, blanks=blanks)


def _stamp(a, layout, motif, positions):
    out = a.copy()
    for point in positions:
        out[_slice(point, *layout[2:])] = motif
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    rh, rv = _routes(d["source"], d["h"], d["v"])
    return _stamp(d["grid"], d["layout"], d["motif"], d["marks"] & (rh ^ rv))


_LAYOUTS = tuple((r, c, h, w) for h in (3, 5, 7) for w in (3, 5, 7)
                 for r in range(3, 8) for c in range(3, 8)
                 if r * (h + 1) - 1 <= 30 and c * (w + 1) - 1 <= 30)


def _pick(rng, values):
    values = sorted(values)
    return values[int(rng.integers(len(values)))]


def _positions(rng, rows, cols, mode):
    if mode == "shared":
        options = []
        if rows >= 5 and cols >= 4:
            options.append("vertical")
        if cols >= 5 and rows >= 4:
            options.append("horizontal")
        axis = _pick(rng, options)
        if axis == "vertical":
            yy = _pick(rng, [q for q in combinations(range(rows), 3)
                             if q[1] - q[0] >= 2 and q[2] - q[1] >= 2])
            xx = _pick(rng, [q for q in combinations(range(cols), 2) if q[1] - q[0] >= 3])
            points = ((yy[0], xx[0]), (yy[2], xx[0]), (yy[1], xx[1]))
        else:
            xx = _pick(rng, [q for q in combinations(range(cols), 3)
                             if q[1] - q[0] >= 2 and q[2] - q[1] >= 2])
            yy = _pick(rng, [q for q in combinations(range(rows), 2) if q[1] - q[0] >= 3])
            points = ((yy[0], xx[0]), (yy[1], xx[1]), (yy[0], xx[2]))
    else:
        yy = sorted(int(x) for x in rng.choice(rows, 3, replace=False))
        xx = sorted(int(x) for x in rng.choice(cols, 3, replace=False))
        if mode == "cross":
            points = ((yy[0], xx[0]), (yy[2], xx[1]), (yy[1], xx[2]))
        else:
            yy, xx = rng.permutation(yy), rng.permutation(xx)
            points = tuple((int(yy[i]), int(xx[i])) for i in range(3))
    flip_r, flip_c = rng.integers(2, size=2)
    return tuple((rows - 1 - r if flip_r else r, cols - 1 - c if flip_c else c)
                 for r, c in points)


def _asymmetric(mask):
    transforms = [np.rot90(mask, k) for k in (1, 2, 3)]
    transforms += [np.fliplr(np.rot90(mask, k)) for k in range(4)]
    return all(x.shape != mask.shape or not np.array_equal(mask, x) for x in transforms)


def _motif(rng, height, width, center_value, require_asym):
    for _ in range(300):
        mask = np.zeros((height, width), dtype=bool)
        kind = int(rng.integers(8))
        y, x = int(rng.integers(height)), int(rng.integers(width))
        if kind == 0:
            mask[y, :] = True
        elif kind == 1:
            mask[y:, x] = True
            mask[y, x:] = True
        elif kind == 2:
            mask[y, :] = True
            mask[:, x] = True
        elif kind == 3:
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
        elif kind == 4:
            for r in range(height):
                mask[r, round(r * (width - 1) / (height - 1))] = True
        elif kind == 5:
            mask[::2, ::2] = True
        elif kind == 6:
            mask.flat[rng.choice(mask.size, int(rng.integers(2, min(7, mask.size - 2) + 1)), replace=False)] = True
        else:
            mask = rng.random((height, width)) < float(rng.choice((0.25, 0.5, 0.8)))
        if rng.random() < 0.3:
            mask |= np.fliplr(mask)
        if rng.random() < 0.3:
            mask |= np.flipud(mask)
        if rng.integers(2):
            mask = np.fliplr(mask).copy()
        if rng.integers(2):
            mask = np.flipud(mask).copy()
        mask[height // 2, width // 2] = center_value
        if 2 <= mask.sum() <= mask.size - 2 and (not require_asym or _asymmetric(mask)):
            return mask
    raise RuntimeError("could not sample an informative source motif")


def _pair(rng, mode, colors, center_value, first=False, forbidden_layout=None, tile_regime=None):
    layouts = [x for x in _LAYOUTS if x != forbidden_layout
               and (tile_regime is None or (x[2:] == (3, 3)) == (tile_regime == "small")) and (
        mode != "shared" or (x[0] >= 5 and x[1] >= 4) or (x[1] >= 5 and x[0] >= 4))]
    for _ in range(1200):
        layout = _pick(rng, layouts)
        rows, cols, height, width = layout
        p, h, v = _positions(rng, rows, cols, mode)
        rh, rv = _routes(p, h, v)
        controls = {p, h, v}
        eh, ev, common = rh - rv - controls, rv - rh - controls, (rh & rv) - controls
        outside = {(r, c) for r in range(rows) for c in range(cols)} - rh - rv - controls
        if not eh or not ev or not outside or len(eh | ev) < 3:
            continue
        if (mode == "separate" and common) or (mode != "separate" and not common):
            continue
        # Reserve one unmarked visited compartment to teach that routes alone
        # do not create copies. Both branches must retain an active candidate.
        reserve_pool = (eh if len(eh) > 1 else set()) | (ev if len(ev) > 1 else set())
        reserve = _pick(rng, reserve_pool)
        available = {(r, c) for r in range(rows) for c in range(cols)} - controls - {reserve}
        masks = []
        active = rh ^ rv
        possible = True
        for hf, vf in ((False, True), (True, True), (False, False)):
            ah, av = _routes(p, h, v, hf, vf)
            witness = available & (active ^ (ah ^ av))
            if not witness:
                if mode != "shared":
                    possible = False
                    break
                continue  # a straight route has no distinguishable bend order
            masks.append(witness)
        if not possible or not masks:
            continue
        density = float(rng.choice((0.12, 0.3, 0.55, 0.8, 1.0)))
        marks = {q for q in sorted(available) if rng.random() < density}
        for pool in (eh - {reserve}, ev - {reserve}, outside, *masks):
            marks.add(_pick(rng, pool))
        if common:
            marks.add(_pick(rng, common))
        lattice, carrier, accent = colors
        mask = _motif(rng, height, width, center_value, first)
        motif = np.where(mask, accent, carrier).astype(np.int8)
        a = np.full((rows * (height + 1) - 1, cols * (width + 1) - 1), carrier, dtype=np.int8)
        a[height::height + 1, :] = lattice
        a[:, width::width + 1] = lattice
        a[_slice(p, height, width)] = motif
        for r, c in marks:
            a[r * (height + 1) + height // 2, c * (width + 1) + width // 2] = lattice
        hy, hx = h[0] * (height + 1) + height // 2, h[1] * (width + 1) + width // 2
        vy, vx = v[0] * (height + 1) + height // 2, v[1] * (width + 1) + width // 2
        a[hy, hx - 1:hx + 2] = lattice
        a[vy - 1:vy + 2, vx] = lattice
        out = _stamp(a, layout, motif, marks & active)
        return {"input": a, "output": out}, layout
    raise RuntimeError("bounded geometry search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(x) for x in rng.permutation(10))
    color_mode = int(rng.integers(4))
    scale_shift = bool(rng.random() < 0.3)
    collision = str(rng.choice(("cross", "shared")))
    held_out = "shared" if collision == "cross" else "cross"
    puzzle = {"train": [], "test": []}
    seen, first_layout = set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            global_index = index if split == "train" else num_train + index
            if color_mode == 0:
                colors = palette[:3]
            elif color_mode == 1:
                colors = (palette[0], palette[1], palette[2 + global_index % 4])
            elif color_mode == 2:
                colors = tuple(palette[(global_index + k) % 3] for k in range(3))
            else:
                colors = tuple(int(x) for x in rng.choice(10, 3, replace=False))
            if split == "train":
                modes = ("separate", "cross", "shared") if scale_shift else ("separate", collision)
                mode = "separate" if index == 0 else collision if index == 1 else str(rng.choice(modes))
            else:
                mode = ("separate" if scale_shift else held_out) if index == 0 else str(rng.choice(("separate", "cross", "shared")))
            center_value = bool(index == 1) if split == "train" and index < 2 else bool(rng.integers(2))
            for _ in range(50):
                pair, layout = _pair(rng, mode, colors, center_value,
                                     first=split == "train" and index == 0,
                                     forbidden_layout=first_layout if split == "train" and index == 1 else None,
                                     tile_regime=("small" if split == "train" else "large") if scale_shift else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_layout = layout
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(point, p, h, v):
    """Direct segment predicates, separate from the solver's route enumerator."""
    r, c = point
    between = lambda x, a, b: min(a, b) <= x <= max(a, b)
    on_h = ((r == p[0] and between(c, p[1], h[1]))
            or (c == h[1] and between(r, p[0], h[0])))
    on_v = ((c == p[1] and between(r, p[0], v[0]))
            or (r == v[0] and between(c, p[1], v[1])))
    return bool(on_h), bool(on_v)


def _analyze(d):
    p, h, v = d["source"], d["h"], d["v"]
    all_points = {(r, c) for r in range(d["layout"][0]) for c in range(d["layout"][1])}
    controls = {p, h, v}
    bins = {q: set() for q in ((False, False), (True, False), (False, True), (True, True))}
    for point in all_points - controls:
        bins[_coverage(point, p, h, v)].add(point)
    common = bins[True, True]
    # A shared run contains adjacent double-covered compartments; for the
    # smallest shared layout, the bend and its neighbor are both candidates.
    topology = "separate" if not common else "shared" if any(
        (r + 1, c) in common or (r, c + 1) in common for r, c in common) else "cross"
    active = d["marks"] & (bins[True, False] | bins[False, True])
    return bins, active, topology


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    teaching_witnesses = [False, False, False]
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            d, out = _decode(pair["input"]), _grid(pair["output"])
            a = d["grid"]
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            bins, active, topology = _analyze(d)
            assert d["marks"] & bins[True, False] and d["marks"] & bins[False, True]
            assert d["marks"] & bins[False, False]  # copy-all counterexample
            assert d["blanks"] & (bins[True, False] | bins[False, True])
            if topology != "separate":
                assert d["marks"] & bins[True, True]  # union counterexample
            expected = a.copy()
            for point in active:
                expected[_slice(point, *d["layout"][2:])] = d["motif"]
            assert np.array_equal(out, expected) and not np.array_equal(a, out)
            assert np.array_equal(solve(a), out)
            # Straight shared-run examples cannot identify one bend order;
            # the separate nonaligned training example must expose all three.
            for index, (hf, vf) in enumerate(((False, True), (True, True), (False, False))):
                ah, av = _routes(d["source"], d["h"], d["v"], hf, vf)
                if split == "train":
                    teaching_witnesses[index] |= active != d["marks"] & (ah ^ av)
            # Dropping either route changes the selected stamps.
            assert active != d["marks"] & (bins[True, False] | bins[True, True])
            assert active != d["marks"] & (bins[False, True] | bins[True, True])
            analyses[split].append((d, topology))
    train = analyses["train"]
    assert all(teaching_witnesses)
    assert train[0][1] == "separate"
    assert train[1][1] in ("cross", "shared")
    assert len({d["layout"] for d, _ in train}) >= 2
    assert {bool(d["motif"][d["layout"][2] // 2, d["layout"][3] // 2] == d["accent"])
            for d, _ in train} == {False, True}
    assert _asymmetric(train[0][0]["motif"] == train[0][0]["accent"])
    test_d, test_topology = analyses["test"][0]
    scale_shift = (all(d["layout"][2:] == (3, 3) for d, _ in train)
                   and all(d["layout"][2:] != (3, 3) for d, _ in analyses["test"]))
    topology_shift = (test_topology in ("cross", "shared")
                      and test_topology not in {topology for _, topology in train})
    assert scale_shift or topology_shift
    return True


def self_test():
    """Bounded 42-seed semantic/diversity audit plus interface boundaries."""
    shapes, layouts, palettes, motifs, counts, sources, fingerprints = (set() for _ in range(7))
    topologies = {s: {k: 0 for k in ("separate", "cross", "shared")} for s in ("train", "test")}
    symmetric = asymmetric = rectangles = seven = 0
    directions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                bins, active, topology = _analyze(d)
                topologies[split][topology] += 1
                shapes.add(d["grid"].shape)
                layouts.add(d["layout"])
                palettes.add((d["lattice"], d["carrier"], d["accent"]))
                mask = d["motif"] == d["accent"]
                motifs.add((mask.shape, mask.tobytes()))
                counts.add((len(d["marks"]), len(active)))
                sources.add(d["source"])
                asymmetric += _asymmetric(mask)
                symmetric += not _asymmetric(mask)
                rectangles += mask.shape[0] != mask.shape[1]
                seven += 7 in mask.shape
                p, h, v = d["source"], d["h"], d["v"]
                directions.add(tuple(int(np.sign(q[k] - p[k])) for q in (h, v) for k in (0, 1)))
                fingerprint.append(_key(d["grid"]))
                # Unmarked and doubly visited cells must remain exactly intact.
                for point in d["blanks"] | (d["marks"] & bins[True, True]):
                    sl = _slice(point, *d["layout"][2:])
                    assert np.array_equal(pair["output"][sl], pair["input"][sl])
        fingerprints.add(tuple(fingerprint))
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(layouts) >= 45 and len(palettes) >= 35
    assert len(motifs) >= 130 and len(counts) >= 35 and len(sources) >= 16
    assert len(directions) >= 20
    assert symmetric >= 15 and asymmetric >= 60 and rectangles >= 50 and seven >= 10
    assert all(topologies[s][k] >= 5 for s in topologies for k in topologies[s]), topologies
    # Explicit expectations chosen by hand: isolated crossing, shared run,
    # collinearity through another terminal, and single-route elbows. These
    # fixtures do not get their active positions from either routing algorithm.
    fixtures = (
        (5, 5, (0, 0), (4, 2), (2, 4),
         {(0, 1), (1, 2), (3, 2), (1, 0), (2, 1), (2, 3), (2, 2), (4, 4)},
         {(0, 1), (1, 2), (3, 2), (1, 0), (2, 1), (2, 3)}),
        (5, 5, (0, 0), (4, 0), (2, 4),
         {(1, 0), (2, 0), (3, 0), (2, 1), (2, 2), (4, 4)},
         {(3, 0), (2, 1), (2, 2)}),
        (3, 6, (0, 0), (0, 2), (0, 5),
         {(0, 1), (0, 3), (0, 4), (2, 4)}, {(0, 3), (0, 4)}),
        (5, 5, (2, 2), (0, 0), (4, 4),
         {(2, 0), (4, 2), (1, 0), (3, 2), (0, 4)},
         {(2, 0), (4, 2), (1, 0), (3, 2)}),
    )
    for rows, cols, p, h, v, marks, active in fixtures:
        a = np.full((4 * rows - 1, 4 * cols - 1), 7, dtype=np.int8)
        a[3::4, :] = 0
        a[:, 3::4] = 0
        motif = np.array([[2, 2, 7], [7, 7, 7], [7, 7, 2]], dtype=np.int8)
        a[_slice(p, 3, 3)] = motif
        for r, c in marks:
            a[4 * r + 1, 4 * c + 1] = 0
        a[4 * h[0] + 1, 4 * h[1]:4 * h[1] + 3] = 0
        a[4 * v[0]:4 * v[0] + 3, 4 * v[1] + 1] = 0
        expected = _stamp(a, (rows, cols, 3, 3), motif, active)
        assert np.array_equal(solve(a), expected)
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            p = generate_puzzle(6000 + a * 10 + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(739, 5, 2), generate_puzzle(739, 5, 2)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    return dict(seeds=42, pairs=210, shapes=len(shapes), layouts=len(layouts),
                palettes=len(palettes), motifs=len(motifs), mark_copy_counts=len(counts),
                source_positions=len(sources), directions=len(directions),
                symmetric=symmetric, asymmetric=asymmetric, rectangular_tiles=rectangles,
                side_seven=seven, topologies=topologies, hand_cases=len(fixtures))


if __name__ == "__main__":
    print(self_test())
