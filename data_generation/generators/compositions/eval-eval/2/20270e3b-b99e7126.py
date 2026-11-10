"""Two incomplete macro blueprints choose the join of their micro fragments.

The solver reads pixels only. Generation builds an oracle from sampled sets,
then checks that the input supplies exactly one cross-scale registration.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _points(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _shift(points, delta):
    return {(r + delta[0], c + delta[1]) for r, c in points}


def _box(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and i == result[-1][1] + 1:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _lattice(a):
    """There are interior dividers, with no outer divider frame."""
    candidates = []
    for color in map(int, np.unique(a)):
        rr = _runs(np.flatnonzero(np.all(a == color, axis=1)))
        cc = _runs(np.flatnonzero(np.all(a == color, axis=0)))
        if len(rr) < 2 or len(cc) < 2:
            continue
        thickness = {b - x + 1 for x, b in rr + cc}
        if len(thickness) != 1:
            continue
        d = next(iter(thickness))
        row_intervals = [(0, rr[0][0])] + [(rr[i][1] + 1, rr[i+1][0])
                         for i in range(len(rr)-1)] + [(rr[-1][1]+1, a.shape[0])]
        col_intervals = [(0, cc[0][0])] + [(cc[i][1] + 1, cc[i+1][0])
                         for i in range(len(cc)-1)] + [(cc[-1][1]+1, a.shape[1])]
        heights = {b-x for x, b in row_intervals}
        widths = {b-x for x, b in col_intervals}
        if len(heights) != 1 or len(widths) != 1:
            continue
        h, w = next(iter(heights)), next(iter(widths))
        if min(h, w) < 3 or max(h, w) > 5 or d not in (1, 2):
            continue
        geometry = (h, w, d, len(row_intervals), len(col_intervals))
        if geometry[3] < h or geometry[4] < w:
            continue
        if any(color in a[y:z, x:v] for y, z in row_intervals for x, v in col_intervals):
            continue
        candidates.append((geometry, color))
    if len(candidates) != 1:
        raise ValueError("the regular interior divider lattice must be unique")
    return candidates[0]


def _bar(points):
    if len(points) < 2:
        return None
    t, b, l, r = _box(points)
    if t == b and r-l+1 == len(points):
        return 0, (t, l), len(points)
    if l == r and b-t+1 == len(points):
        return 1, (t, l), len(points)
    return None


def _joins(masks, bars, shape):
    """Both immediately adjacent seam orders; neither fragment is rotated."""
    axis, a, length = bars[0]
    _, b, _ = bars[1]
    joins = []
    for sign in (-1, 1):
        delta = [a[0]-b[0], a[1]-b[1]]
        delta[axis] += sign
        moved = _shift(masks[1], delta)
        union = masks[0] | moved
        top, bottom, left, right = _box(union)
        if (bottom-top+1, right-left+1) != shape:
            continue
        normalization = (-top, -left)
        registered = [_shift(masks[0], normalization), _shift(moved, normalization)]
        guides = _shift(_points_from_bar(bars[0]), normalization)
        guides |= _shift(_shift(_points_from_bar(bars[1]), delta), normalization)
        joins.append({"sign": sign, "delta": tuple(delta), "normalization": normalization,
                      "masks": registered, "union": set.union(*registered), "guides": guides})
    return joins


def _points_from_bar(bar):
    axis, (r, c), length = bar
    return {(r + (i if axis else 0), c + (0 if axis else i)) for i in range(length)}


def _origins(masks, positions, geometry):
    h, w, _, rows, cols = geometry
    first = next(iter(positions[0]))
    possibilities = {(first[0]-r, first[1]-c) for r, c in masks[0]}
    return sorted((y, x) for y, x in possibilities
                  if 0 <= y <= rows-h and 0 <= x <= cols-w
                  and all((r-y, c-x) in masks[k] for k in (0, 1) for r, c in positions[k]))


def _solutions(joins, positions, geometry):
    result = []
    for join in joins:
        for origin in _origins(join["masks"], positions, geometry):
            active = _shift(join["union"], origin)
            if (all(len(positions[k]) < len(join["masks"][k]) for k in (0, 1))
                    and set.union(*positions) < active):
                result.append((join, origin))
    return result


def _read(a):
    geometry, divider = _lattice(a)
    h, w, d, rows, cols = geometry
    records = {}
    for r in range(rows):
        for c in range(cols):
            y, x = r*(h+d), c*(w+d)
            tile = a[y:y+h, x:x+w]
            records.setdefault(_key(tile), [tile.copy(), set()])[1].add((r, c))
    if len(records) != 3:
        raise ValueError("expected one ordinary tile and two fragment tile designs")
    ordered = sorted(records.values(), key=lambda x: (-len(x[1]), tuple(map(int, x[0].flat))))
    ordinary, normal_positions = ordered[0]
    fragments, positions = [x[0] for x in ordered[1:]], [x[1] for x in ordered[1:]]
    if len(normal_positions) <= sum(map(len, positions)) or min(map(len, positions)) < 2:
        raise ValueError("ordinary tiles need a strict majority; each fragment needs two clues")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    if len(ordinary_colors) not in (2, 3):
        raise ValueError("ordinary tiles use two or three colors")
    extras = [set(map(int, np.unique(f))) - ordinary_colors for f in fragments]
    if extras[0] != extras[1] or len(extras[0]) != 2:
        raise ValueError("the two fragments share exactly two novel colors")
    interpretations = []
    for marker in sorted(extras[0]):
        ink = next(iter(extras[0] - {marker}))
        bars = [_bar(_points(f == marker)) for f in fragments]
        if (any(b is None for b in bars) or bars[0][0] != bars[1][0]
                or bars[0][2] != bars[1][2]):
            continue
        backgrounds = [set(map(int, np.unique(f[(f != ink) & (f != marker)]))) for f in fragments]
        if backgrounds[0] != backgrounds[1] or len(backgrounds[0]) != 1:
            continue
        background = next(iter(backgrounds[0]))
        if background not in ordinary_colors:
            continue
        masks = [_points(f == ink) for f in fragments]
        if min(map(len, masks)) < 3:
            continue
        joins = _joins(masks, bars, (h, w))
        # The intended family deliberately retains both geometrically viable orders.
        if len(joins) != 2 or joins[0]["union"] == joins[1]["union"]:
            continue
        solutions = _solutions(joins, positions, geometry)
        if len(solutions) == 1:
            interpretations.append({"geometry": geometry, "divider": divider,
                                    "ordinary": ordinary, "fragments": fragments,
                                    "positions": positions, "bars": bars, "raw_masks": masks,
                                    "background": background, "ink": ink, "marker": marker,
                                    "joins": joins, "join": solutions[0][0], "origin": solutions[0][1]})
    if len(interpretations) != 1:
        raise ValueError("the micro seam and macro translation must be jointly unique")
    return interpretations[0]


def _fused_tile(info, union=None):
    h, w = info["geometry"][:2]
    tile = np.full((h, w), info["background"], dtype=np.int8)
    for r, c in info["join"]["union"] if union is None else union:
        tile[r, c] = info["ink"]
    return tile


def _completion(a, info, union=None):
    h, w, d, _, _ = info["geometry"]
    union = info["join"]["union"] if union is None else union
    tile = _fused_tile(info, union)
    result = a.copy()
    for r, c in _shift(union, info["origin"]):
        y, x = r*(h+d), c*(w+d)
        result[y:y+h, x:x+w] = tile
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer color roles, both seam candidates, and their macro registrations."""
    a = _grid(input_grid)
    return _completion(a, _read(a))


def _paint(geometry, divider, ordinary, fragments, positions):
    h, w, d, rows, cols = geometry
    grid = np.full((rows*(h+d)-d, cols*(w+d)-d), divider, dtype=np.int8)
    lookup = {p: k for k in (0, 1) for p in positions[k]}
    for r in range(rows):
        for c in range(cols):
            y, x = r*(h+d), c*(w+d)
            grid[y:y+h, x:x+w] = fragments[lookup[(r, c)]] if (r, c) in lookup else ordinary
    return grid


def _geometry(rng, shape=None):
    h, w = shape or (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    d = 1 if max(h, w) == 5 else int(rng.choice((1, 1, 2)))
    rows = int(rng.integers(h, (30+d)//(h+d)+1))
    cols = int(rng.integers(w, (30+d)//(w+d)+1))
    return h, w, d, rows, cols


def _design(rng, shape):
    h, w = shape
    axis = int(rng.integers(2))
    normal, tangent = shape[axis], shape[1-axis]
    length = int(rng.integers(2, tangent))
    a_start = (int(rng.integers(normal)), int(rng.integers(tangent-length+1)))
    b_start = (int(rng.integers(normal)), int(rng.integers(tangent-length+1)))
    convert = lambda p: p if axis == 0 else (p[1], p[0])
    bars = [(axis, convert(a_start), length), (axis, convert(b_start), length)]
    am, bm = map(_points_from_bar, bars)
    difference = (a_start[0]-b_start[0], a_start[1]-b_start[1])
    available = []
    for r in range(normal):
        for c in range(tangent):
            p = convert((r, c))
            if (p not in bm and all(0 <= r+difference[0]+s < normal
                                     and 0 <= c+difference[1] < tangent for s in (-1, 1))):
                available.append(p)
    if len(available) < 3:
        return None
    mode = int(rng.integers(5))
    density = float(rng.uniform(.12, .65))
    a = {(r, c) for r in range(h) for c in range(w) if (r, c) not in am
         and (rng.random() < density or (mode == 0 and (r in (0, h-1) or c in (0, w-1)))
              or (mode == 1 and (r == c or r+c == w-1)))}
    # A spans the tile box, so both cropped joins have exactly the original tile dimensions.
    for edge in ([p for p in [(0, c) for c in range(w)] if p not in am],
                 [p for p in [(h-1, c) for c in range(w)] if p not in am],
                 [p for p in [(r, 0) for r in range(h)] if p not in am],
                 [p for p in [(r, w-1) for r in range(h)] if p not in am]):
        if not any(p in a for p in edge):
            a.add(edge[int(rng.integers(len(edge)))])
    count = int(rng.integers(3, len(available)+1))
    b = {available[int(i)] for i in rng.choice(len(available), count, replace=False)}
    joins = _joins([a, b], bars, shape)
    if len(a) < 3 or len(joins) != 2 or joins[0]["union"] == joins[1]["union"]:
        return None
    if any(len(j["union"]) == h*w or not (j["masks"][0]-j["masks"][1])
           or not (j["masks"][1]-j["masks"][0]) for j in joins):
        return None
    return {"masks": [a, b], "bars": bars, "joins": joins, "axis": axis, "mode": mode}


def _clues(rng, design, geometry, sign, total, excluded=()):
    join = next(j for j in design["joins"] if j["sign"] == sign)
    h, w, _, rows, cols = geometry
    masks = join["masks"]
    if total > min((rows*cols-1)//2, len(join["union"]), sum(map(len, masks))-2):
        return None
    possible_counts = [(n, total-n) for n in range(2, total-1)
                       if n < len(masks[0]) and total-n < len(masks[1])]
    if not possible_counts:
        return None
    active = [sorted(m) for m in masks]
    for _ in range(160):
        origin = (int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1)))
        n0, n1 = possible_counts[int(rng.integers(len(possible_counts)))]
        p0 = {active[0][int(i)] for i in rng.choice(len(active[0]), n0, replace=False)}
        free = [p for p in active[1] if p not in p0]
        if len(free) < n1:
            continue
        p1 = {free[int(i)] for i in rng.choice(len(free), n1, replace=False)}
        fragment_key = (tuple(sorted(p0)), tuple(sorted(p1)))
        if fragment_key in excluded:
            continue
        positions = [_shift(p0, origin), _shift(p1, origin)]
        solutions = _solutions(design["joins"], positions, geometry)
        if len(solutions) == 1 and solutions[0][0]["sign"] == sign and solutions[0][1] == origin:
            return positions, origin, fragment_key
    return None


def _make_pair(rng, palette, sign, total, *, familiar=None, overlap=False, mixed_guides=False, shape=None):
    for _ in range(256):
        geometry = _geometry(rng, shape if familiar is None else familiar["shape"])
        design = _design(rng, geometry[:2]) if familiar is None else familiar["design"]
        if design is None:
            continue
        join = next(j for j in design["joins"] if j["sign"] == sign)
        if overlap and not (join["masks"][0] & join["masks"][1]):
            continue
        visible_guides = {p for p in join["guides"] if 0 <= p[0] < geometry[0] and 0 <= p[1] < geometry[1]}
        if mixed_guides and not (visible_guides & join["union"] and visible_guides - join["union"]):
            continue
        unshifted_union = set.union(*design["masks"])
        if unshifted_union == join["union"]:
            continue
        clues = _clues(rng, design, geometry, sign, total,
                        () if familiar is None else familiar["excluded"])
        if clues is None:
            continue
        # Six-clue designs teach the mechanism, then reappear with the OTHER
        # seam in test. Certify sparse evidence for both orders in advance.
        if familiar is None and total > 4:
            if any(j["union"] == unshifted_union for j in design["joins"]):
                continue
            if any(_clues(rng, design, geometry, s, 4) is None for s in (-1, 1)):
                continue
        positions, origin, fragment = clues
        divider, background, auxiliary, ink, marker, auxiliary2 = palette
        colors = (background, auxiliary, auxiliary2) if rng.random() < .25 else (background, auxiliary)
        h, w = geometry[:2]
        labels = np.concatenate((np.arange(len(colors)), rng.integers(len(colors), size=h*w-len(colors))))
        rng.shuffle(labels)
        ordinary = np.array(colors, dtype=np.int8)[labels.reshape(h, w)]
        inactive = [(r, c) for r in range(h) for c in range(w) if (r, c) not in join["union"]]
        if all(ordinary[p] == background for p in inactive):
            ordinary[inactive[0]] = auxiliary
        if len(np.unique(ordinary)) < len(colors):
            continue
        tiles = []
        for mask, bar in zip(design["masks"], design["bars"]):
            tile = np.full((h, w), background, dtype=np.int8)
            for p in mask:
                tile[p] = ink
            for p in _points_from_bar(bar):
                tile[p] = marker
            tiles.append(tile)
        input_grid = _paint(geometry, divider, ordinary, tiles, positions)
        # Construct the expected grid directly, without calling solve/_read.
        fused = np.full((h, w), background, dtype=np.int8)
        for p in join["union"]:
            fused[p] = ink
        output_grid = _paint(geometry, divider, ordinary, [fused, fused],
                             [_shift(join["union"], origin), set()])
        try:
            info = _read(input_grid)
        except ValueError:
            continue
        if not np.array_equal(_completion(input_grid, info), output_grid):
            continue
        return {"input": input_grid, "output": output_grid}, {
            "shape": (h, w), "design": design, "sign": sign, "excluded": {fragment}}
    raise RuntimeError("bounded cross-scale construction exhausted")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = rng.random() < .4
    base_palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    first_sign = int(rng.choice((-1, 1)))
    puzzle, used, familiar = {"train": [], "test": []}, set(), []
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            old = None
            if split == "test" and (i == 0 or rng.random() < .6):
                # The two six-clue demonstrations are certified to admit sparse clues.
                old = familiar[int(rng.integers(2))]
            sign = -old["sign"] if old else (first_sign if i == 0 else -first_sign if i == 1 else int(rng.choice((-1, 1))))
            total = 4 if split == "test" else 6 if i < 2 else int(rng.integers(4, 8))
            palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 6, replace=False)))
            for _ in range(32):
                try:
                    pair, metadata = _make_pair(rng, palette, sign, total, familiar=old,
                                               overlap=split == "train" and i == 0,
                                               mixed_guides=split == "train" and i == 1)
                except RuntimeError:
                    if split != "test" or i == 0 or old is None:
                        raise
                    # Later tests can use a fresh design if a familiar design's
                    # finite catalog of different four-clue subsets is exhausted.
                    old = None
                    continue
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train":
                familiar.append(metadata)
            elif old is not None:
                old["excluded"].update(metadata["excluded"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    a, output = _grid(pair["input"]), _grid(pair["output"])
    info = _read(a)
    expected = _completion(a, info)
    assert np.array_equal(output, expected), "incorrect cross-scale completion"
    h, w, d, rows, cols = info["geometry"]
    join, origin = info["join"], info["origin"]
    masks, union = join["masks"], join["union"]
    assert len(union) == sum(map(len, masks))-len(masks[0] & masks[1])
    assert masks[0]-masks[1] and masks[1]-masks[0], "both fragments must contribute exclusive ink"
    assert int(np.sum(output == info["ink"])) == len(union)**2, "the two scales must agree"
    assert info["marker"] not in output, "all microscopic guide ink disappears"
    assert output.shape == a.shape
    visible = set.union(*info["positions"])
    active = _shift(union, origin)
    assert visible < active
    assert all(len(info["positions"][k]) < len(masks[k]) for k in (0, 1))
    fused = _fused_tile(info)
    for r in range(rows):
        for c in range(cols):
            y, x = r*(h+d), c*(w+d)
            tile = output[y:y+h, x:x+w]
            assert np.array_equal(tile, fused if (r, c) in active else info["ordinary"])
    assert any(info["ordinary"][r, c] != info["background"]
               for r in range(h) for c in range(w) if (r, c) not in union)
    # Frame-spanning fragment is a measurement role, never a hidden solving convention.
    spans = [_box(m) == (0, h-1, 0, w-1) for m in info["raw_masks"]]
    assert sum(spans) == 1
    frame_index = spans.index(True)
    info["frame_sign"] = join["sign"] if frame_index == 0 else -join["sign"]
    info["visible_count"] = len(visible)
    info["missing_count"] = len(active-visible)
    info["overlap"] = len(masks[0] & masks[1])
    visible_guides = {p for p in join["guides"] if 0 <= p[0] < h and 0 <= p[1] < w}
    info["covered_guides"] = len(visible_guides & union)
    info["uncovered_guides"] = len(visible_guides-union)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for a in pair.values():
                _grid(a)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            information[split].append(_analyze_pair(pair))
    first, second = information["train"][:2]
    assert first["frame_sign"] != second["frame_sign"], "both mating orders must be demonstrated"
    assert first["overlap"] > 0, "union must be distinguishable from XOR"
    assert second["covered_guides"] > 0 and second["uncovered_guides"] > 0
    assert first["visible_count"] == second["visible_count"] == 6
    assert all(i["visible_count"] == 4 for i in information["test"])
    # The first test reuses a teaching design but asks for the opposite join.
    test = information["test"][0]
    def design_signature(i):
        # Palette- and presentation-independent microscopic identities.
        return (i["geometry"][:2], tuple(sorted((tuple(sorted(m)), b)
                                                for m, b in zip(i["raw_masks"], i["bars"]))))
    matches = [i for i in information["train"][:2] if design_signature(i) == design_signature(test)]
    assert matches and any(i["frame_sign"] != test["frame_sign"] for i in matches), (
        "a familiar test must use a newly inferred seam")
    for index, i in enumerate(information["train"][:2]):
        zero_union = set.union(*i["raw_masks"])
        assert zero_union != i["join"]["union"], "ignoring registration must change the blueprint"
        alternative = next(j for j in i["joins"] if j["sign"] != i["join"]["sign"])
        assert alternative["union"] != i["join"]["union"]
        assert not _origins(alternative["masks"], i["positions"], i["geometry"]), (
            "macro clues must eliminate the other geometrically viable seam")
        if i["overlap"]:
            pair = puzzle["train"][index]
            assert not np.array_equal(_completion(_grid(pair["input"]), i,
                                                  i["join"]["masks"][0] ^ i["join"]["masks"][1]),
                                      pair["output"])


def self_test():
    """Bounded checks on 42 seeds, interfaces, semantics, and structural diversity."""
    shapes, geometry, designs, unions, palettes, axes, overlaps, signs = (set() for _ in range(8))
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = _grid(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve must not mutate input"
                i = _analyze_pair(pair)
                shapes.add(before.shape)
                geometry.add(i["geometry"])
                designs.add(tuple(sorted(tuple(sorted(m)) for m in i["raw_masks"])))
                unions.add((i["geometry"][:2], tuple(sorted(i["join"]["union"]))))
                palettes.add((i["divider"], i["background"], i["ink"], i["marker"]))
                axes.add(i["bars"][0][0])
                overlaps.add(i["overlap"])
                signs.add(i["frame_sign"])
    assert len(shapes) >= 15 and len(geometry) >= 25
    assert len(designs) >= 100 and len(unions) >= 75 and len(palettes) >= 30
    assert axes == {0, 1} and signs == {-1, 1}
    assert 0 in overlaps and max(overlaps) >= 3
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    assert len({i["geometry"][:2] for p in puzzles for s in ("train", "test")
                for pair in p[s] for i in [_read(pair["input"])]}) >= 8
    # The smallest tile is relatively rare under six-clue teaching constraints.
    small, _ = _make_pair(np.random.default_rng(317), tuple(range(6)), 1, 4, shape=(3, 3))
    assert _analyze_pair(small)["geometry"][:2] == (3, 3)
    repeated = generate_puzzle(17)
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(puzzles[17][s], repeated[s]) for k in ("input", "output"))
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4), (3, 2)):
        p = generate_puzzle(920+sum(counts), *counts)
        assert (len(p["train"]), len(p["test"])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.array(a, copy=True) for k, a in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("a damaged output was accepted")
    # An ink-only paste leaves a non-ink ordinary pixel behind in a new tile.
    pair = puzzles[0]["train"][0]
    info = _read(pair["input"])
    h, w, d, _, _ = info["geometry"]
    new_position = min(_shift(info["join"]["union"], info["origin"])
                       - set.union(*info["positions"]))
    pixel = next((r, c) for r in range(h) for c in range(w)
                 if (r, c) not in info["join"]["union"]
                 and info["ordinary"][r, c] != info["background"])
    bad = {"input": pair["input"].copy(), "output": pair["output"].copy()}
    bad["output"][new_position[0]*(h+d)+pixel[0], new_position[1]*(w+d)+pixel[1]] = info["ordinary"][pixel]
    try:
        _analyze_pair(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("an ink-only paste was accepted")
    # Interior placements of the documented blueprint, in both directions.
    # These recorded pixel expectations do not invoke solve to build the oracle.
    ordinary = np.array([[0,0,0,0], [0,0,1,0], [0,0,0,0], [0,0,0,0]], dtype=np.int8)
    fragments = [np.array([[int(c) for c in row] for row in digits], dtype=np.int8)
                 for digits in (("2002", "2880", "2000", "0022"),
                                ("0000", "0880", "2202", "0000"))]
    for sign, row, digits in ((-1, 1, ("2002", "2202", "2000", "0022")),
                              (1, 3, ("2002", "2000", "2000", "2222"))):
        fused = np.array([[int(c) for c in line] for line in digits], dtype=np.int8)
        positions = [{(1,1), (4,3)}, {(row+1,1), (row+1,2)}]
        a = _paint((4,4,1,6,6), 9, ordinary, fragments, positions)
        expected = _paint((4,4,1,6,6), 9, ordinary, [fused, fused],
                          [_shift(_points(fused == 2), (1,1)), set()])
        assert np.array_equal(solve(a), expected)
        info = _analyze_pair({"input": a, "output": expected})
        assert info["origin"] == (1,1) and info["frame_sign"] == sign
    # Make a true full-subset tie at the two scales; do not break it by ordering.
    tie_checked = False
    for p in puzzles:
        for pair in p["train"]:
            info = _read(pair["input"])
            left, right = info["joins"]
            common = [left["masks"][k] & right["masks"][k] for k in (0, 1)]
            if min(map(len, common)) < 2:
                continue
            p0 = set(sorted(common[0])[:2])
            free = sorted(common[1]-p0)
            if len(free) < 2:
                continue
            positions = [_shift(p0, info["origin"]), _shift(set(free[:2]), info["origin"])]
            ambiguous = _paint(info["geometry"], info["divider"], info["ordinary"],
                               info["fragments"], positions)
            try:
                solve(ambiguous)
            except ValueError:
                tie_checked = True
                break
            raise AssertionError("an ambiguous seam was accepted")
        if tie_checked:
            break
    assert tie_checked, "the finite seed suite must exercise a rejected seam tie"
