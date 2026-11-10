"""Corner-locked interference cloth. Standard library and NumPy only.

The solver parses only pixels. No source fixtures, seed state, or sampled
metadata participate in solving. Background holes are part of every tile.
"""

from collections import deque
import numpy as np


HOST_SEEDS = (0, 597681402, 135683155, 13458662, 197849088, 596776355,
              945026191, 913293617, 843369808, 847932522, 84186249,
              149970430, 865795262, 587355144, 285945335, 383054073,
              666937149, 912412178, 733563920, 822125789, 297995984,
              243233847, 200267376, 945979104, 175198384, 638359691,
              896835334, 734934614, 378022345, 836777499, 923556091,
              970760313, 299128960, 791493053, 393619462, 790741581,
              990453280, 789472670, 354396135, 962337942, 609888222,
              979015689)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=False)


def _components(cells):
    remaining = set(cells)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, queue = {start}, deque([start])
        while queue:
            r, c = queue.popleft()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _crop(cells):
    rs, cs = zip(*cells)
    r0, c0 = min(rs), min(cs)
    tile = np.zeros((max(rs)-r0+1, max(cs)-c0+1), dtype=bool)
    for r, c in cells:
        tile[r-r0, c-c0] = True
    return tile


def _analyze(value):
    g = _grid(value)
    h, w = g.shape
    if min(h, w) < 3 or len(np.unique(g)) != 3:
        raise ValueError("family inputs have three roles and an interior")
    rim = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    colors, counts = np.unique(rim, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("ambiguous border background")
    border = np.zeros(g.shape, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    marks = [tuple(map(int, p)) for p in np.argwhere(border & (g != bg))]
    if len(marks) not in (1, 2) or len({int(g[p]) for p in marks}) != 1:
        raise ValueError("expected one or two same-colored border markers")
    marker = int(g[marks[0]])
    if np.count_nonzero(g == marker) != len(marks):
        raise ValueError("marker color is reserved for border markers in inputs")
    payload = next(int(c) for c in np.unique(g) if int(c) not in (bg, marker))
    cells = set(tuple(map(int, p)) for p in np.argwhere(g == payload))
    if not cells or any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
        raise ValueError("payload must be strictly interior")
    if len(_components(cells)) != 1:
        raise ValueError("source payload must be four-connected")
    sides, ar, ac = set(), None, None
    for r, c in marks:
        options = [s for s, yes in (("top", r == 0), ("bottom", r == h-1),
                                    ("left", c == 0), ("right", c == w-1)) if yes]
        if len(options) != 1 or options[0] in sides:
            raise ValueError("markers cannot occupy corners or repeat a side")
        side = options[0]
        sides.add(side)
        if side in ("top", "bottom"):
            ac = c
        else:
            ar = r
    if len(sides) == 2 and (ar is None or ac is None):
        raise ValueError("two markers must occupy adjacent sides")
    if ac is not None and not any(c == ac for r, c in cells):
        raise ValueError("vertical cut must intersect the source")
    if ar is not None and not any(r == ar for r, c in cells):
        raise ValueError("horizontal cut must intersect the source")
    groups = {}
    for r, c in sorted(cells):
        if r == ar or c == ac:
            continue
        v = ("top" if r < ar else "bottom") if ar is not None else next(iter(sides))
        s = ("left" if c < ac else "right") if ac is not None else next(iter(sides))
        groups.setdefault((v, s), set()).add((r, c))
    if len(groups) != (2 if len(sides) == 1 else 4):
        raise ValueError("every region of the cut must contain payload")
    retained = {corner: group for corner, group in groups.items() if set(corner) & sides}
    return dict(grid=g, background=bg, payload=payload, marker=marker,
                cells=cells, sides=sides, axes=(ar, ac), groups=groups,
                retained=retained)


def _origin(shape, tile, corner):
    return (0 if corner[0] == "top" else shape[0]-tile.shape[0],
            0 if corner[1] == "left" else shape[1]-tile.shape[1])


def _prefix_tile(tile):
    """Deliberately wrong shortened-prefix alternative for diagnostic ablation."""
    result = tile
    for axis in (0, 1):
        slices = np.moveaxis(result, axis, 0)
        length = len(slices)
        p = next(p for p in range(1, length+1)
                 if all(np.array_equal(slices[i], slices[i % p]) for i in range(length)))
        result = np.moveaxis(slices[:p], 0, axis)
    return result


def _coverage(info, *, finite=False, all_regions=False, fragments=False,
              top_left=False, prefix=False, uncut=False):
    h, w = info["grid"].shape
    rr, cc = np.indices((h, w))
    coverage = np.zeros((h, w), dtype=np.int8)
    if uncut:
        groups = {("top", "left"): info["cells"]}
    else:
        groups = info["groups"] if all_regions else info["retained"]
    for corner, group in groups.items():
        # Components of a single cut are still ONE contributing stencil. The
        # fragments ablation wrongly gives them individual crops and clocks.
        group_field = np.zeros((h, w), dtype=bool)
        for cells in (_components(group) if fragments else [group]):
            tile = _crop(cells)
            original_origin = _origin((h, w), tile, corner)
            r0, c0 = (0, 0) if top_left else original_origin
            if prefix:
                tile = _prefix_tile(tile)
            if finite:
                th, tw = tile.shape
                group_field[r0:r0+th, c0:c0+tw] |= tile
            else:
                group_field |= tile[(rr-r0) % tile.shape[0], (cc-c0) % tile.shape[1]]
        coverage += group_field
    return coverage


def _colorize(info, coverage):
    out = np.full(info["grid"].shape, info["background"], dtype=np.int8)
    out[coverage == 1] = info["payload"]
    out[coverage >= 2] = info["marker"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Parse the visible cuts, repeat whole stencils, and count their overlap."""
    info = _analyze(input_grid)
    return _colorize(info, _coverage(info))


def _stamp_output(info):
    """Construction check by explicit translated copies, including blank holes.

    This is an author-owned invariant, not an independent reference solver.
    Each lattice copy is disjoint from other copies of the SAME stencil.
    """
    h, w = info["grid"].shape
    coverage = np.zeros((h, w), dtype=np.int8)
    for corner, cells in info["retained"].items():
        tile = _crop(cells)
        th, tw = tile.shape
        r0, c0 = _origin((h, w), tile, corner)
        for r in range(r0-((r0+th-1)//th)*th, h, th):
            for c in range(c0-((c0+tw-1)//tw)*tw, w, tw):
                ra, rb, ca, cb = max(r, 0), min(r+th, h), max(c, 0), min(c+tw, w)
                coverage[ra:rb, ca:cb] += tile[ra-r:rb-r, ca-c:cb-c]
    return _colorize(info, coverage)


def pair_features(value):
    info = _analyze(value)
    coverage = _coverage(info)
    out = _colorize(info, coverage)
    tiles = [_crop(c) for c in info["retained"].values()]
    differences = {name: not np.array_equal(out, _colorize(info, _coverage(info, **args)))
                   for name, args in (("omit_A", {"uncut": True}),
                                      ("omit_B", {"finite": True}),
                                      ("fragments", {"fragments": True}),
                                      ("phase", {"top_left": True}),
                                      ("prefix", {"prefix": True}),
                                      ("deletion", {"all_regions": True}))}
    return dict(shape=info["grid"].shape, sides=tuple(sorted(info["sides"])),
                roles=(info["background"], info["payload"], info["marker"]),
                tile_shapes=tuple(t.shape for t in tiles),
                tile_sizes=tuple(int(t.sum()) for t in tiles),
                densities=tuple(float(t.mean()) for t in tiles),
                components=tuple(len(_components(c)) for c in info["retained"].values()),
                coverage_counts=tuple(int(np.count_nonzero(coverage == n)) for n in range(4)),
                **differences)


def _teeth(rng, rlo, rhi, clo, chi, axis, width_cap, height_cap):
    """Unequal thin arms plus occasional attached growth, linked by the cut axis."""
    band_h, band_w = rhi-rlo+1, chi-clo+1
    th = int(rng.integers(3, min(band_h, height_cap)+1))
    tw = int(rng.integers(1, min(band_w, width_cap)+1))
    r0 = int(rng.integers(rlo, rhi-th+2))
    rows = {0, th-1}
    if th >= 5 and rng.random() < .5:
        rows.add(int(rng.integers(2, th-2)))
    lengths = {r: int(rng.integers(1, tw+1)) for r in sorted(rows)}
    lengths[int(rng.choice(sorted(rows)))] = tw
    cells = {(r0+r, axis+d if clo > axis else axis-d)
             for r in rows for d in range(1, lengths[r]+1)}
    # Shape diversity beyond disconnected horizontal teeth. Growth never
    # makes the whole source disconnected, because it attaches to these arms.
    for _ in range(int(rng.integers(0, max(2, th*tw//5)))):
        frontier = sorted({q for r, c in cells for q in
                           ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if r0 <= q[0] < r0+th and clo <= q[1] <= chi
                           and abs(q[1]-axis) <= tw and q not in cells})
        if frontier and rng.random() < .65:
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _make_input(rng, double, witness, shape, roles, turn, enlarged=False):
    h, w = shape
    ph = int(rng.integers(13, min(18, h-1))) if enlarged else int(rng.integers(9, min(13, h-1)))
    pw = int(rng.integers(13, min(19, w-1))) if enlarged else int(rng.integers(9, min(14, w-1)))
    ar = int(rng.integers(3, ph-3)) if double else None
    ac = int(rng.integers(3, pw-3))
    cells = {(r, ac) for r in range(ph)}
    if double:
        cells |= {(ar, c) for c in range(pw)}
    if witness:
        # A blank middle slice is retained in the left collective cut. Right
        # is the complete word ABA, where B is a narrow near-axis slice.
        left_w = int(rng.integers(2, min(5, ac+1)))
        right_w = int(rng.integers(2, min(5, pw-ac)))
        lr, rr = int(rng.integers(0, ph-2)), int(rng.integers(0, ph-2))
        cells |= {(lr+r, ac-d) for r in (0, 2)
                  for d in range(1, (left_w if r == 0 else left_w-1)+1)}
        cells |= {(rr+r, ac+d) for r in (0, 2) for d in range(1, right_w+1)}
        cells.add((rr+1, ac+1))
    else:
        for rlo, rhi in ([(0, ph-1)] if not double else [(0, ar-1), (ar+1, ph-1)]):
            for clo, chi in ((0, ac-1), (ac+1, pw-1)):
                cells |= _teeth(rng, rlo, rhi, clo, chi, ac,
                                8 if enlarged else 5, 9 if enlarged else 7)
    r0 = int(rng.integers(1, h-ph))
    c0 = int(rng.integers(1, w-pw))
    bg, payload, marker = roles
    g = np.full(shape, bg, dtype=np.int8)
    for r, c in cells:
        g[r+r0, c+c0] = payload
    g[0, ac+c0] = marker
    if double:
        g[ar+r0, -1] = marker
    return np.rot90(g, turn).copy()


def _counts(num_train, num_test):
    for v, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError("counts must be integers in train 2..10 and test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    regime, base_turn = int(rng.integers(3)), int(rng.integers(4))
    varied_palette = bool(rng.integers(2))
    fixed_roles = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            training = split == "train"
            witness = training and index == 0
            double = (index == 1 if training and index < 2 else bool(rng.integers(2)))
            turn = ((base_turn + (int(rng.integers(2)) if training else 2+int(rng.integers(2)))) % 4
                    if regime == 1 else int(rng.integers(4)))
            roles = tuple(int(c) for c in rng.choice(10, 3, replace=False)) if varied_palette else fixed_roles
            enlarged = regime == 0 and not training
            for attempt in range(192):
                if enlarged:
                    shape = tuple(int(v) for v in rng.integers(24, 31, size=2))
                elif regime == 2 and not training:
                    shape = (int(rng.integers(14, 19)), int(rng.integers(26, 31)))
                else:
                    shape = tuple(int(v) for v in rng.integers(14, 24 if regime == 0 else 25, size=2))
                g = _make_input(rng, double, witness, shape, roles, turn, enlarged)
                info = _analyze(g)
                f = pair_features(g)
                key = (g.shape, g.tobytes())
                # Keep every coverage class legible. Triple coverage is taught
                # by the double-marker demonstration, along with deletion.
                acceptable = min(f["coverage_counts"][:3]) >= 3 and f["omit_A"] and f["omit_B"]
                if witness:
                    acceptable &= f["fragments"] and f["prefix"] and f["phase"]
                if training and index == 1:
                    acceptable &= f["deletion"] and f["coverage_counts"][3] >= 2
                if acceptable and key not in seen:
                    seen.add(key)
                    puzzle[split].append({"input": g, "output": _stamp_output(info)})
                    break
            else:
                raise RuntimeError("bounded stencil construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            key = (g.shape, g.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(g)
            coverage = _coverage(info)
            f = pair_features(g)
            assert np.array_equal(solve(g), out), "wrong interference cloth"
            assert np.array_equal(_stamp_output(info), out), "translation/modulo mismatch"
            # The entire canvas is replaced: source scaffold and marker cells
            # receive exactly the same counting rule as every other location.
            assert np.all(out[coverage == 0] == info["background"])
            assert np.all(out[coverage == 1] == info["payload"])
            assert np.all(out[coverage >= 2] == info["marker"])
            assert sum(f["coverage_counts"]) == g.size
            assert min(f["coverage_counts"][:3]) >= 3
            assert f["omit_A"] and f["omit_B"], "both parents must change the answer"
            features[split].append(f)
    tr = features["train"]
    assert {len(f["sides"]) for f in tr} == {1, 2}
    assert any(f["fragments"] and f["prefix"] and f["phase"] for f in tr), "missing crop/word/phase witness"
    assert any(f["deletion"] and f["coverage_counts"][3] >= 2 for f in tr), "missing deletion/saturation witness"


def _puzzle_key(p):
    return tuple((s, tuple((np.asarray(x["input"]).shape, np.asarray(x["input"]).tobytes(),
                           np.asarray(x["output"]).tobytes()) for x in p[s])) for s in ("train", "test"))


def self_test():
    """Bounded checks of 42 supplied seeds, all counts, and visible invariants."""
    puzzles = [generate_puzzle(s) for s in HOST_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    records = [pair_features(pair["input"]) for p in puzzles
               for split in ("train", "test") for pair in p[split]]
    assert len({f["shape"] for f in records}) >= 60
    assert len({shape for f in records for shape in f["tile_shapes"]}) >= 20
    assert len({f["tile_sizes"] for f in records}) >= 50
    assert len({f["roles"] for f in records}) >= 20
    assert len({f["sides"] for f in records}) == 8
    assert max(max(f["shape"]) for f in records) == 30
    assert min(min(f["shape"]) for f in records) == 14
    assert max(n for f in records for n in f["components"]) >= 3
    assert any(1 in shape for f in records for shape in f["tile_shapes"])
    assert sum(f["deletion"] for f in records) >= 30
    assert all(f["omit_A"] and f["omit_B"] for f in records)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(17+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(HOST_SEEDS[0]))
    # Explicit whole-rectangle boundary and negative-space word regressions.
    tile = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool)
    assert _prefix_tile(tile).shape[0] == 2
    assert not np.array_equal(tile[np.arange(9) % 3], _prefix_tile(tile)[np.arange(9) % 2])
    for p in puzzles:
        for s in ("train", "test"):
            for pair in p[s]:
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal counts)")
