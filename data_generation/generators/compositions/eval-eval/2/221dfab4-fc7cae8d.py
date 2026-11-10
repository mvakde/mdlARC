"""Color-keyed periodic cross-section sampling with guide-relative coordinates.

No parent generator is imported.  All solving conventions are visible in the
input: two clocks, their identities, their common corner, and two crop guides.
"""

from collections import Counter

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}


def _matrix(k, mirror=False):
    rotation = np.array([[0, -1], [1, 0]], dtype=int)
    result = np.linalg.matrix_power(rotation, k)
    return result @ np.diag([1, -1]) if mirror else result


TRANSFORMS = tuple((k, mirror) for mirror in (False, True) for k in range(4))


def _transform(grid, transform):
    k, mirror = transform
    return np.rot90(np.fliplr(grid) if mirror else grid, k).copy()


def _side_map(transform):
    matrix = _matrix(*transform)
    inverse = {v: k for k, v in NORMALS.items()}
    return {s: inverse[tuple(matrix @ n)] for s, n in NORMALS.items()}


def _adjacent(a, b):
    return (SIDES.index(a) - SIDES.index(b)) % 2 == 1


def _edge(grid, side):
    return {"T": grid[0], "R": grid[:, -1],
            "B": grid[-1], "L": grid[:, 0]}[side]


def _clock_coordinates(shape, side, other):
    """Walk along a ruler away from the corner shared with the other ruler."""
    h, w = shape
    if side in ("T", "B"):
        row = 0 if side == "T" else h - 1
        cols = range(w) if other == "L" else range(w - 1, -1, -1)
        return [(row, c) for c in cols]
    col = 0 if side == "L" else w - 1
    rows = range(h) if other == "T" else range(h - 1, -1, -1)
    return [(r, col) for r in rows]


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not np.issubdtype(array.dtype, np.integer)
            or any(n < 1 or n > 30 for n in array.shape)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("grids must be integer matrices, sizes 1..30, colors 0..9")
    return array.astype(np.int8, copy=False)


def _guide_positions(bounds, side):
    r, c, h, w = bounds
    if side == "T":
        return [(r - 1, x) for x in range(c, c + w)]
    if side == "B":
        return [(r + h, x) for x in range(c, c + w)]
    if side == "L":
        return [(x, c - 1) for x in range(r, r + h)]
    return [(x, c + w) for x in range(r, r + h)]


def _decode(input_grid):
    grid = _grid(input_grid)
    H, W = grid.shape
    if min(H, W) < 11:
        raise ValueError("canvas is too small for the clock and guide grammar")
    background = int(grid[0, 0])
    if any(int(grid[p]) != background for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("all four ruler end caps must be background")
    outer_colors = {}
    for side in SIDES:
        colors = set(map(int, np.unique(_edge(grid, side)))) - {background}
        if colors:
            outer_colors[side] = colors
    if len(outer_colors) != 2:
        raise ValueError("exactly two outer rulers are required")
    outer_sides = list(outer_colors)
    if not _adjacent(*outer_sides) or any(len(s) != 2 for s in outer_colors.values()):
        raise ValueError("rulers must be adjacent, each with ordinary and special ticks")
    common = set.intersection(*outer_colors.values())
    if len(common) != 1:
        raise ValueError("the two rulers must share exactly the special tick color")
    special = common.pop()
    outer = {side: next(iter(colors - {special}))
             for side, colors in outer_colors.items()}
    if len(set(outer.values())) != 2:
        raise ValueError("ordinary tick identities must be distinct")
    clocks = {}
    for side, identity in outer.items():
        other = next(s for s in outer if s != side)
        points = _clock_coordinates(grid.shape, side, other)
        values = np.array([grid[p] for p in points], dtype=int)
        ticks = np.flatnonzero(values[1:-1] != background) + 1
        if len(ticks) < 3:
            raise ValueError("each clock needs a complete ordinary/ordinary/special cycle")
        q, spacing = int(ticks[0]), int(ticks[1] - ticks[0])
        if not (1 <= q <= 3 and 1 <= spacing <= 3):
            raise ValueError("clock offset and tick spacing must be in 1..3")
        expected = np.full(len(values), background, dtype=int)
        for tick, d in enumerate(range(q, len(values) - 1, spacing)):
            expected[d] = special if tick % 3 == 2 else identity
        if not np.array_equal(values, expected):
            raise ValueError("broken periodic ruler")
        clocks[identity] = (q, spacing)

    descriptions = []
    for identity in outer.values():
        positions = np.argwhere(grid[1:-1, 1:-1] == identity) + 1
        if len(positions) < 6:
            raise ValueError("missing interior guide")
        rows, cols = positions.T
        if np.all(rows == rows[0]) and len(positions) == int(cols.max() - cols.min() + 1):
            descriptions.append(("H", int(rows[0]), int(cols.min()), len(positions), identity))
        elif np.all(cols == cols[0]) and len(positions) == int(rows.max() - rows.min() + 1):
            descriptions.append(("V", int(cols[0]), int(rows.min()), len(positions), identity))
        else:
            raise ValueError("each interior identity must form exactly one straight guide")
    horizontal = [d for d in descriptions if d[0] == "H"]
    vertical = [d for d in descriptions if d[0] == "V"]
    if len(horizontal) != 1 or len(vertical) != 1:
        raise ValueError("guides must be perpendicular")
    _, hr, c, w, hc = horizontal[0]
    _, vc, r, h, vcolor = vertical[0]
    hs = "T" if hr == r - 1 else "B" if hr == r + h else None
    vs = "L" if vc == c - 1 else "R" if vc == c + w else None
    if hs is None or vs is None or not (6 <= h <= 24 and 6 <= w <= 24):
        raise ValueError("guide endpoints must define one legal rectangle")
    guides = {hs: hc, vs: vcolor}
    bounds = (r, c, h, w)
    if r < 2 or c < 2 or r + h > H - 2 or c + w > W - 2:
        raise ValueError("payload must be separated from the outer rulers")
    payload = grid[r:r + h, c:c + w].copy()
    if set(map(int, np.unique(payload))) & (set(outer.values()) | {special}):
        raise ValueError("clock colors are reserved outside the payload")

    # Reconstruct the complete exterior. This also checks clean separation and
    # rejects extra bars, guide protrusions, or a second competing payload.
    exterior = np.full(grid.shape, background, dtype=np.int8)
    for side in outer:
        _edge(exterior, side)[:] = _edge(grid, side)
    for side, identity in guides.items():
        positions = _guide_positions(bounds, side)
        if any(min(rr, cc, H - 1 - rr, W - 1 - cc) < 2 for rr, cc in positions):
            raise ValueError("guides need one blank cell between them and the ruler")
        for p in positions:
            exterior[p] = identity
    exterior[r:r + h, c:c + w] = payload
    if not np.array_equal(exterior, grid):
        raise ValueError("unrecognized material outside the guided payload")
    transforms = [t for t in TRANSFORMS
                  if all(outer.get(_side_map(t)[s]) == color for s, color in guides.items())]
    if len(transforms) != 1:
        raise ValueError("correspondence must specify a unique symmetry")
    info = dict(background=background, special=special, outer=outer, guides=guides,
                clocks=clocks, bounds=bounds, payload=payload, transform=transforms[0],
                canvas_shape=grid.shape)
    rows, cols = _indices(info)
    if not len(rows) or not len(cols):
        raise ValueError("each inward sweep must have at least one special cross-section")
    return info


def _indices(info, mode="special"):
    h, w = info["payload"].shape
    rows, cols = np.arange(h), np.arange(w)
    identities = list(info["guides"].values())
    for side, identity in info["guides"].items():
        if mode == "swap_clocks":
            identity = next(color for color in identities if color != identity)
        q, spacing = info["clocks"][identity]
        if side in ("T", "B"):
            size, coordinates = h, rows
        else:
            size, coordinates = w, cols
        distances = coordinates if side in ("T", "L") else size - 1 - coordinates
        if mode == "one_based":
            distances = distances + 1
        if mode == "canvas_phase":
            r, c, _, _ = info["bounds"]
            H, W = info["canvas_shape"]
            distances = distances + {"T": r, "B": H - r - h,
                                     "L": c, "R": W - c - w}[side]
        if mode == "q_zero":
            q = 0
        tick = (distances - q) // spacing
        keep = (distances >= q) & ((distances - q) % spacing == 0)
        if mode != "every_tick":
            keep &= tick % 3 != 2 if mode == "ordinary" else tick % 3 == 2
        if side in ("T", "B"):
            rows = coordinates[keep]
        else:
            cols = coordinates[keep]
    return rows, cols


def _prediction(info, mode="special", oriented=True):
    if mode == "all":
        result = info["payload"].copy()
    else:
        rows, cols = _indices(info, mode)
        result = info["payload"][np.ix_(rows, cols)]
    return _transform(result, info["transform"]) if oriented else result.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the supplied grid; sample intersecting slices and align them."""
    return _prediction(_decode(input_grid))


def _coordinate_oracle(info):
    """Generator-owned validation by individual source cells and signed axes.

    This deliberately avoids the solver's indexing and array symmetry helpers.
    It is a corroborating implementation, not an independent semantic review.
    """
    payload = info["payload"]
    h, w = payload.shape
    selected = []
    for r in range(h):
        for c in range(w):
            good = True
            for side, identity in info["guides"].items():
                d = {"T": r, "B": h - 1 - r, "L": c, "R": w - 1 - c}[side]
                q, spacing = info["clocks"][identity]
                first = q + 2 * spacing
                good &= d >= first and (d - first) % (3 * spacing) == 0
            if good:
                selected.append((r, c))
    row_ranks = {r: i for i, r in enumerate(sorted({r for r, _ in selected}))}
    col_ranks = {c: i for i, c in enumerate(sorted({c for _, c in selected}))}
    matrix = np.zeros((2, 2), dtype=int)
    for side, identity in info["guides"].items():
        target = next(s for s, color in info["outer"].items() if color == identity)
        matrix += np.outer(NORMALS[target], NORMALS[side])
    cells = [matrix @ (row_ranks[r], col_ranks[c]) for r, c in selected]
    coordinates = np.array(cells)
    coordinates -= coordinates.min(axis=0)
    result = np.empty(tuple(coordinates.max(axis=0) + 1), dtype=np.int8)
    for (r, c), (rr, cc) in zip(selected, coordinates):
        result[rr, cc] = payload[r, c]
    return result


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _asymmetric(grid):
    return grid.size > 0 and len({_key(_transform(grid, t)) for t in TRANSFORMS}) == 8


def _tight_prediction(info):
    positions = np.argwhere(info["payload"] != info["background"])
    if not len(positions):
        return np.empty((0, 0), dtype=np.int8)
    lo, hi = positions.min(axis=0), positions.max(axis=0) + 1
    altered = dict(info)
    altered["payload"] = info["payload"][lo[0]:hi[0], lo[1]:hi[1]]
    return _prediction(altered)


def _teaches(info):
    output = _prediction(info)
    sample = _prediction(info, oriented=False)
    bg = info["background"]
    return (min(sample.shape) >= 2 and np.count_nonzero(sample != bg) >= 3
            and bg in sample and len(np.unique(sample[sample != bg])) >= 2
            and (np.any(np.all(sample == bg, axis=0)) or np.any(np.all(sample == bg, axis=1)))
            and _asymmetric(sample)
            and all(not np.array_equal(output, _prediction(info, mode))
                    for mode in ("all", "swap_clocks", "q_zero", "one_based", "every_tick", "ordinary", "canvas_phase"))
            and not np.array_equal(output, sample)
            and not np.array_equal(output, _tight_prediction(info)))


def _paint_rulers(grid, outer, clocks, background, special):
    for side, identity in outer.items():
        other = next(s for s in outer if s != side)
        points = _clock_coordinates(grid.shape, side, other)
        q, spacing = clocks[identity]
        for tick, d in enumerate(range(q, len(points) - 1, spacing)):
            grid[points[d]] = special if tick % 3 == 2 else identity


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    return dict(background=colors[0], identities=colors[1:3], special=colors[3],
                objects=colors[4:4 + int(rng.integers(2, 6))])


def _motif(rng, shape, palette, blank=False):
    """Fresh rectangles, paths, teeth, patches, outlines and sparse fragments."""
    h, w = shape
    bg, colors = palette["background"], palette["objects"]
    grid = np.full(shape, bg if rng.random() < .8 else int(rng.choice(colors)), dtype=np.int8)
    for _ in range(int(rng.integers(3, 10))):
        color = int(rng.choice(colors))
        r, c = int(rng.integers(h)), int(rng.integers(w))
        kind = int(rng.integers(6))
        if kind == 0:
            hh = int(rng.integers(1, max(2, h // 2 + 1)))
            ww = int(rng.integers(1, max(2, w // 2 + 1)))
            grid[r:min(h, r + hh), c:min(w, c + ww)] = color
        elif kind == 1:
            for _ in range(int(rng.integers(5, max(6, h * w // 2)))):
                grid[r, c] = color
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                r, c = int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))
        elif kind == 2:
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            a, b = sorted((r, rr)); x, y = sorted((c, cc))
            grid[a:b + 1, c] = color
            grid[rr, x:y + 1] = color
        elif kind == 3:
            step = int(rng.integers(2, 6))
            if rng.integers(2):
                for x in range(c % step, w, step):
                    grid[r:min(h, r + int(rng.integers(1, 4))), x] = color
            else:
                for y in range(r % step, h, step):
                    grid[y, c:min(w, c + int(rng.integers(1, 4)))] = color
        elif kind == 4:
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            a, b = sorted((r, rr)); x, y = sorted((c, cc))
            grid[a, x:y + 1] = color
            grid[a:b + 1, y] = color
            grid[b, x:y] = color
        else:
            rr = rng.integers(h, size=int(rng.integers(3, max(4, h * w // 6))))
            cc = rng.integers(w, size=len(rr))
            grid[rr, cc] = color
    if rng.random() < .35:
        grid[rng.random(shape) < float(rng.uniform(.02, .15))] = bg
    if blank or rng.random() < .35:
        side = str(rng.choice(SIDES))
        thickness = int(rng.integers(1, 3))
        if side == "T":
            grid[:thickness] = bg
        elif side == "B":
            grid[-thickness:] = bg
        elif side == "L":
            grid[:, :thickness] = bg
        else:
            grid[:, -thickness:] = bg
    return grid


def _construct(rng, shape, inner_corner, transform, palette, teaching=False):
    h, w = shape
    inner = (SIDES[inner_corner], SIDES[(inner_corner + 1) % 4])
    guides = dict(zip(inner, palette["identities"]))
    mapping = _side_map(transform)
    outer = {mapping[s]: color for s, color in guides.items()}
    H, W = int(rng.integers(h + 5, 31)), int(rng.integers(w + 5, 31))
    if rng.random() < .12:
        H = W = 30
    elif rng.random() < .2:
        H = W = max(H, W)
    r = int(rng.integers(2 + ("T" in guides), H - h - 1 - ("B" in guides)))
    c = int(rng.integers(2 + ("L" in guides), W - w - 1 - ("R" in guides)))
    clocks = {}
    for side, identity in guides.items():
        depth = min(h, w) if teaching else h if side in ("T", "B") else w
        target = mapping[side]
        rail_length = W if target in ("T", "B") else H
        choices = [(q, s) for q in range(1, 4) for s in range(1, 4)
                   if q + (5 if teaching else 2) * s < depth
                   and q + 2 * s < rail_length - 1]
        if teaching and clocks:
            choices = [clock for clock in choices if clock != next(iter(clocks.values()))]
        clocks[identity] = choices[int(rng.integers(len(choices)))]
    payload = _motif(rng, shape, palette, blank=teaching)
    grid = np.full((H, W), palette["background"], dtype=np.int8)
    grid[r:r + h, c:c + w] = payload
    for side, identity in guides.items():
        for p in _guide_positions((r, c, h, w), side):
            grid[p] = identity
    _paint_rulers(grid, outer, clocks, palette["background"], palette["special"])
    return grid


def _swap_outer_identities(grid, info):
    result = grid.copy()
    a, b = info["outer"].values()
    border = np.zeros(grid.shape, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    result[border & (grid == a)] = b
    result[border & (grid == b)] = a
    return result


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    held_corner = int(rng.integers(4))
    train_corners = [i for i in range(4) if i != held_corner]
    stable = bool(rng.random() < .25)
    palette = _palette(rng)
    inner = train_corners[int(rng.integers(3))]
    first_transform = (int(rng.choice([1, 3])), False)
    first = second = None
    for _ in range(512):
        shape = tuple(map(int, rng.integers(17, 25, size=2)))
        candidate = _construct(rng, shape, inner, first_transform, palette, teaching=True)
        info = _decode(candidate)
        if not _teaches(info):
            continue
        reversed_input = _swap_outer_identities(candidate, info)
        reversed_info = _decode(reversed_input)
        if (not _asymmetric(_prediction(reversed_info, oriented=False))
                or np.array_equal(_prediction(info), _prediction(reversed_info))
                or np.array_equal(_prediction(reversed_info), _prediction(reversed_info, "swap_clocks"))):
            continue
        first = dict(input=candidate, output=_prediction(info))
        second = dict(input=reversed_input, output=_prediction(reversed_info))
        break
    if first is None:
        raise RuntimeError("bounded natural teaching-scene sampling was exhausted")
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            for attempt in range(64):
                local_palette = palette if stable else _palette(rng)
                corner = int(rng.choice(train_corners)) if split == "train" else held_corner
                transform = TRANSFORMS[int(rng.integers(8))]
                low = 9 if split == "train" else 6
                h, w = map(int, rng.integers(low, 25, size=2))
                if rng.random() < .18:
                    w = h
                inp = _construct(rng, (h, w), corner, transform, local_palette)
                if _key(inp) not in seen:
                    out = solve(inp)
                    pair = dict(input=inp, output=out)
                    seen.add(_key(inp))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling was exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _contrast(a, b):
    return (a["bounds"] == b["bounds"] and a["canvas_shape"] == b["canvas_shape"]
            and a["guides"] == b["guides"] and set(a["outer"]) == set(b["outer"])
            and a["outer"] != b["outer"]
            and set(a["outer"].values()) == set(b["outer"].values())
            and np.array_equal(a["payload"], b["payload"])
            and all(a["clocks"][a["outer"][s]] == b["clocks"][b["outer"][s]] for s in a["outer"])
            and not np.array_equal(_prediction(a), _prediction(b)))


def validate_generated_puzzle(puzzle):
    """Check schema, grammar, cell provenance, training evidence and the shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            expected = _coordinate_oracle(info)
            assert np.array_equal(out, expected), "incorrect sample, orientation, or extent"
            rows, cols = _indices(info)
            retained = info["payload"][np.ix_(rows, cols)]
            assert np.array_equal(np.sort(out.ravel()), np.sort(retained.ravel())), "selected cell colors changed"
            assert not (set(map(int, np.unique(out))) & (set(info["outer"].values()) | {info["special"]})), "clock ink leaked"
            assert inp.shape != out.shape, "the canvas must be removed"
            decoded[split].append(info)
    train, test = decoded["train"], decoded["test"]
    assert any(_teaches(info) for info in train), "missing discriminating sampling demonstration"
    assert any(_contrast(a, b) for i, a in enumerate(train) for b in train[i + 1:]), "missing identity-only clock contrast"
    assert any(t["transform"][0] in (1, 3) and not t["transform"][1] for t in train), "missing quarter-turn"
    assert any(t["transform"][1] for t in train), "missing reflection"
    known_corners = {frozenset(t["guides"]) for t in train}
    assert all(frozenset(t["guides"]) not in known_corners for t in test), "test must shift the guide origin corner"


def _measure(info):
    rows, cols = _indices(info)
    payload, bg = info["payload"], info["background"]
    h, w = payload.shape
    output = _prediction(info)
    return dict(transform=info["transform"], inner=tuple(sorted(info["guides"])),
                outer=tuple(sorted(info["outer"])), input_shape=info["canvas_shape"],
                payload_shape=payload.shape, output_shape=output.shape,
                counts=(len(rows), len(cols)),
                clocks=tuple(sorted(info["clocks"].values())),
                colors=len(np.unique(payload[payload != bg])),
                density=float(np.mean(payload != bg)),
                aspect=int(np.sign(h - w)),
                blank_output=bool(np.any(np.all(output == bg, axis=0)) or np.any(np.all(output == bg, axis=1))),
                border_ink=int(np.count_nonzero(payload[0] != bg) + np.count_nonzero(payload[-1] != bg)
                               + np.count_nonzero(payload[:, 0] != bg) + np.count_nonzero(payload[:, -1] != bg)))


def self_test():
    """Bounded 42-seed audit, equivariance, legal counts and rejected mutations."""
    records, puzzles = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                record = _measure(info)
                record["split"] = split
                records.append(record)
        if seed < 3:
            repeated = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for split in puzzle
                       for a, b in zip(puzzle[split], repeated[split]) for k in ("input", "output"))
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    assert {clock[1] for m in records for clock in m["clocks"]} == {1, 2, 3}
    assert {clock[0] for m in records for clock in m["clocks"]} == {1, 2, 3}
    assert len({m["input_shape"] for m in records}) >= 35
    assert len({m["payload_shape"] for m in records}) >= 50
    assert len({m["output_shape"] for m in records}) >= 20
    assert {m["aspect"] for m in records} == {-1, 0, 1}
    for split in ("train", "test"):
        subset = [m for m in records if m["split"] == split]
        assert {m["inner"] for m in subset} == {tuple(sorted((SIDES[i], SIDES[(i + 1) % 4]))) for i in range(4)}
        assert {m["transform"] for m in subset} == set(TRANSFORMS)
    assert any(1 in m["counts"] for m in records if m["split"] == "test")
    assert max(max(m["counts"]) for m in records) >= 7
    assert min(m["density"] for m in records) < .12 and max(m["density"] for m in records) > .75
    assert any(m["blank_output"] for m in records if m["split"] == "test")
    assert any(m["border_ink"] > 20 for m in records)

    # All D4 symmetries of a whole scene must rotate/reflect its answer equally.
    for pair in puzzles[0]["train"][:1] + puzzles[1]["test"]:
        for transform in TRANSFORMS:
            assert np.array_equal(solve(_transform(pair["input"], transform)), _transform(pair["output"], transform))
        permutation = np.random.default_rng(912).permutation(10)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])

    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2027 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    for mutation in ("cell", "extent", "ruler", "guide"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if mutation == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif mutation == "extent":
            pair["output"] = np.pad(pair["output"], ((0, 1), (0, 0)), constant_values=info["background"])
        elif mutation == "guide":
            side = next(iter(info["guides"]))
            points = _guide_positions(info["bounds"], side)
            pair["input"][points[len(points) // 2]] = info["background"]
        else:
            side = next(iter(info["outer"]))
            other = next(s for s in info["outer"] if s != side)
            identity = info["outer"][side]
            q, spacing = info["clocks"][identity]
            pair["input"][_clock_coordinates(pair["input"].shape, side, other)[q + 2 * spacing]] = identity
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("corrupt clock, guide, output cell or output extent accepted")
    return dict(seeds=42, pairs=len(records),
                input_shapes=len({m["input_shape"] for m in records}),
                payload_shapes=len({m["payload_shape"] for m in records}),
                output_shapes=len({m["output_shape"] for m in records}),
                transforms={str(t): sum(m["transform"] == t for m in records) for t in TRANSFORMS},
                clock_counts={str(k): v for k, v in sorted(Counter(clock for m in records for clock in m["clocks"]).items())},
                sample_counts={str(k): v for k, v in sorted(Counter(m["counts"] for m in records).items())},
                density_range=[min(m["density"] for m in records), max(m["density"] for m in records)],
                blank_test_outputs=sum(m["blank_output"] for m in records if m["split"] == "test"))
