"""Two keyed periodic emitters; coincident motif cells acquire the elbow color.

The outer key defines an orientation, not a palette learned from examples. Each
inner elbow is a fixed phase anchor. Its two bars define the COMPLETE tile,
including empty edge strips. Orient the infinite periodic field about that
elbow, then superpose both fields on the canvas interior.

Only Python/NumPy are used. Generation, validation and self_test write no files.
Generation is bounded to 64 candidates per requested distinct example; there is
no accumulating retry history. Public solve reads only its input array.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
MATRICES = tuple(np.array(m, dtype=int) for m in (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)),
    ((0, 1), (-1, 0)), ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0)),
))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid color")
    return a.astype(np.int8, copy=False)


def _core(grid, side):
    if side == "T":
        return grid[0, 1:-1]
    if side == "R":
        return grid[1:-1, -1]
    if side == "B":
        return grid[-1, 1:-1]
    return grid[1:-1, 0]


def _corner(shape, sides):
    return (0 if "T" in sides else shape[0] - 1,
            0 if "L" in sides else shape[1] - 1)


def _matrix(guides, controls):
    result = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        target = next(s for s, c in controls.items() if c == color)
        result += np.outer(NORMAL[target], NORMAL[side])
    return result


def _box(record):
    r, c, h, w = record["bounds"]
    py, px = record["pivot"]
    return min(r, py), min(c, px), h + 1, w + 1


def _separated(a, b):
    """Inclusive objects have at least one complete blank row or column gap."""
    r, c, h, w = a
    y, x, v, u = b
    return r + h + 1 <= y or y + v + 1 <= r or c + w + 1 <= x or x + u + 1 <= c


def _paint_key(grid, controls, elbow, background):
    for side in SIDES:
        _core(grid, side)[:] = controls.get(side, background)
    grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = background
    grid[_corner(grid.shape, controls)] = elbow


def _paint_record(grid, record, elbow):
    r, c, h, w = record["bounds"]
    py, px = record["pivot"]
    grid[r:r + h, c:c + w][record["mask"]] = record["pigment"]
    horizontal = "T" if py < r else "B"
    vertical = "L" if px < c else "R"
    grid[py, c:c + w] = record["guides"][horizontal]
    grid[r:r + h, px] = record["guides"][vertical]
    grid[py, px] = elbow


def _decode(input_grid):
    grid = _grid(input_grid)
    H, W = grid.shape
    _require(17 <= H <= 30 and 17 <= W <= 30, "canvas sides must be 17..30")
    side_colors = {}
    for side in SIDES:
        values = np.unique(_core(grid, side))
        _require(len(values) == 1, "outer side cores must be solid")
        side_colors[side] = int(values[0])
    multiplicities = Counter(side_colors.values())
    _require(sorted(multiplicities.values()) == [1, 1, 2], "outer key must have two controls")
    background = next(c for c, n in multiplicities.items() if n == 2)
    controls = {s: c for s, c in side_colors.items() if c != background}
    _require(any(set(controls) == {SIDES[k], SIDES[(k + 1) % 4]} for k in range(4)),
             "controls must occupy adjacent sides")
    markers = set(controls.values())
    elbow = int(grid[_corner(grid.shape, controls)])
    _require(elbow not in markers | {background}, "elbow role must be distinct")
    pivots = np.argwhere(grid[1:-1, 1:-1] == elbow) + 1
    _require(len(pivots) == 2, "exactly two inner elbows are required")
    records = []
    for point in pivots:
        py, px = (int(v) for v in point)
        horizontal = [(d, int(grid[py, px + d])) for d in (-1, 1)
                      if 0 <= px + d < W and int(grid[py, px + d]) in markers]
        vertical = [(d, int(grid[py + d, px])) for d in (-1, 1)
                    if 0 <= py + d < H and int(grid[py + d, px]) in markers]
        _require(len(horizontal) == len(vertical) == 1, "ambiguous elbow directions")
        dx, hc = horizontal[0]
        dy, vc = vertical[0]
        _require(hc != vc, "both marker identities must occur at each elbow")
        w = h = 0
        while 0 <= px + (w + 1) * dx < W and grid[py, px + (w + 1) * dx] == hc:
            w += 1
        while 0 <= py + (h + 1) * dy < H and grid[py + (h + 1) * dy, px] == vc:
            h += 1
        _require(2 <= h <= 5 and 2 <= w <= 5, "tile sides must be 2..5")
        r, c = (py + 1 if dy == 1 else py - h), (px + 1 if dx == 1 else px - w)
        guides = {"T" if dy == 1 else "B": hc, "L" if dx == 1 else "R": vc}
        tile = grid[r:r + h, c:c + w]
        colors = set(int(v) for v in np.unique(tile))
        pigment = colors - markers - {background, elbow}
        _require(len(pigment) == 1 and colors == pigment | {background},
                 "a tile must contain background and one distinct pigment")
        pigment = next(iter(pigment))
        record = dict(bounds=(r, c, h, w), pivot=(py, px), guides=guides,
                      mask=tile == pigment, pigment=pigment, matrix=_matrix(guides, controls))
        y, x, bh, bw = _box(record)
        _require(y >= 2 and x >= 2 and y + bh <= H - 2 and x + bw <= W - 2,
                 "an inner bracket must have a blank moat from the outer key")
        records.append(record)
    _require(records[0]["pigment"] != records[1]["pigment"], "pigments must differ")
    _require(_separated(*(_box(s) for s in records)), "brackets need a blank gap")
    # Reconstruct all allowed ink, catching extra bars, dots, or ambiguous crops.
    expected = np.full(grid.shape, background, dtype=np.int8)
    _paint_key(expected, controls, elbow, background)
    for record in records:
        _paint_record(expected, record, elbow)
    _require(np.array_equal(expected, grid), "unexpected ink or malformed guide")
    return dict(shape=grid.shape, background=background, elbow=elbow,
                controls=controls, records=records)


def _field(record, shape, ignore_orientation=False):
    """Inverse-coordinate modular lookup on the canvas interior."""
    H, W = shape
    yy, xx = np.indices((H - 2, W - 2)) + 1
    py, px = record["pivot"]
    r, c, h, w = record["bounds"]
    m = np.eye(2, dtype=int) if ignore_orientation else record["matrix"]
    # m maps source displacements to target displacements; inverse is transpose.
    source_y = py + m[0, 0] * (yy - py) + m[1, 0] * (xx - px)
    source_x = px + m[0, 1] * (yy - py) + m[1, 1] * (xx - px)
    return record["mask"][(source_y - r) % h, (source_x - c) % w]


def _combine(info, fields):
    result = np.full((info["shape"][0] - 2, info["shape"][1] - 2),
                     info["background"], dtype=np.int8)
    a, b = fields
    result[a & ~b] = info["records"][0]["pigment"]
    result[b & ~a] = info["records"][1]["pigment"]
    result[a & b] = info["elbow"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and superpose two oriented periodic fields using input alone."""
    info = _decode(input_grid)
    return _combine(info, [_field(s, info["shape"]) for s in info["records"]])


def _stamp_field(record, shape, finite=False):
    """Forward stamping check, without inverse lookup or an oriented tile crop."""
    result = np.zeros((shape[0] - 2, shape[1] - 2), dtype=bool)
    r, c, h, w = record["bounds"]
    pivot = np.array(record["pivot"])
    matrix = record["matrix"]
    periods = np.abs(matrix) @ np.array((h, w))
    for dr, dc in np.argwhere(record["mask"]):
        pos = pivot + matrix @ (np.array((r + dr, c + dc)) - pivot) - 1
        y, x = (int(v) for v in pos)
        if finite:
            if 0 <= y < result.shape[0] and 0 <= x < result.shape[1]:
                result[y, x] = True
        else:
            result[y % periods[0]::periods[0], x % periods[1]::periods[1]] = True
    return result


def _render(info):
    # Generation uses forward stamping; solve uses inverse-coordinate lookup.
    return _combine(info, [_stamp_field(s, info["shape"]) for s in info["records"]])


def _tight(info):
    """Concrete alternative: crop each motif's ink instead of reading guide lengths."""
    records = []
    for s in info["records"]:
        occupied = np.argwhere(s["mask"])
        lo, hi = occupied.min(axis=0), occupied.max(axis=0) + 1
        r, c, _, _ = s["bounds"]
        records.append(dict(s, bounds=(r + int(lo[0]), c + int(lo[1]),
                                       int(hi[0] - lo[0]), int(hi[1] - lo[1])),
                            mask=s["mask"][lo[0]:hi[0], lo[1]:hi[1]]))
    alternative = dict(info, records=records)
    return _render(alternative)


def _origin_tiling(info):
    """Concrete alternative: orient each tile, then repeat from output (0,0)."""
    fields = []
    for s in info["records"]:
        h, w = s["mask"].shape
        coords = s["matrix"] @ np.indices((h, w)).reshape(2, -1)
        coords -= coords.min(axis=1, keepdims=True)
        tile = np.zeros(tuple(coords.max(axis=1) + 1), dtype=bool)
        tile[coords[0], coords[1]] = s["mask"].ravel()
        H, W = info["shape"][0] - 2, info["shape"][1] - 2
        fields.append(tile[np.arange(H)[:, None] % tile.shape[0],
                           np.arange(W)[None, :] % tile.shape[1]])
    return _combine(info, fields)


def _inverse_orientation(info):
    alternative = dict(info, records=[dict(s, matrix=s["matrix"].T) for s in info["records"]])
    return _render(alternative)


def _swap_controls(grid):
    info = _decode(grid)
    controls = info["controls"]
    colors = list(controls.values())
    changed = grid.copy()
    for side, color in controls.items():
        _core(changed, side)[:] = colors[1 - colors.index(color)]
    return changed


def _blank_sides(mask):
    return tuple(s for s, line in zip(SIDES, (mask[0], mask[:, -1], mask[-1], mask[:, 0]))
                 if not line.any())


def _all_states(info, output):
    return set(int(c) for c in np.unique(output)) == {
        info["background"], info["elbow"], *(s["pigment"] for s in info["records"])}


def _useful(info, output):
    return (_all_states(info, output)
            and not np.array_equal(output, _combine(info, [_field(s, info["shape"], True)
                                                         for s in info["records"]]))
            and not np.array_equal(output, _origin_tiling(info)))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _primitive(mask):
    return all(not np.array_equal(mask, np.roll(mask, k, axis=axis))
               for axis, n in enumerate(mask.shape) for k in range(1, n))


def _sample_mask(rng, shape, blank=None):
    allowed = np.ones(shape, dtype=bool)
    if blank == "T":
        allowed[0] = False
    elif blank == "B":
        allowed[-1] = False
    elif blank == "L":
        allowed[:, 0] = False
    elif blank == "R":
        allowed[:, -1] = False
    positions = np.flatnonzero(allowed)
    for _ in range(64):
        count = int(rng.integers(1, min(len(positions), shape[0] * shape[1] - 1) + 1))
        mask = np.zeros(shape, dtype=bool)
        mask.flat[rng.choice(positions, size=count, replace=False)] = True
        if _primitive(mask):
            return mask
    # A single occupied cell always has both full periods, including size two.
    mask = np.zeros(shape, dtype=bool)
    mask.flat[int(rng.choice(positions))] = True
    return mask


def _positions(shape, tile_shape):
    h, w = tile_shape
    return [(r, c, h + 1, w + 1) for r in range(2, shape[0] - h - 2)
            for c in range(2, shape[1] - w - 2)]


def _random_input(rng, *, test, palette, teaching=False):
    H, W = (int(v) for v in rng.integers(17, 31, size=2))
    bg, a, b, elbow, p, q = palette
    dimensions = (2, 3, 4, 5) if test else (2, 3, 4)
    shapes = [(h, w) for h in dimensions for w in dimensions]
    shape0 = _pick(rng, [s for s in shapes if (not teaching or s[0] != s[1])
                         and (not test or 5 in s)])
    shape1 = _pick(rng, [s for s in shapes if s != shape0])
    inner0 = int(rng.integers(4))
    guides0 = dict(zip((SIDES[inner0], SIDES[(inner0 + 1) % 4]), (a, b)))
    if teaching:
        m = MATRICES[int(rng.choice((1, 3)))]
        target = {s: next(t for t in SIDES if tuple(m @ NORMAL[s]) == NORMAL[t])
                  for s in guides0}
        controls = {target[s]: color for s, color in guides0.items()}
    else:
        outer = int(rng.integers(4))
        colors = (a, b) if rng.integers(2) else (b, a)
        controls = dict(zip((SIDES[outer], SIDES[(outer + 1) % 4]), colors))
    inner1 = int(rng.integers(4))
    colors = (a, b) if rng.integers(2) else (b, a)
    guides1 = dict(zip((SIDES[inner1], SIDES[(inner1 + 1) % 4]), colors))
    candidates0, candidates1 = _positions((H, W), shape0), _positions((H, W), shape1)
    # Finite randomized enumeration, with no unbounded placement rejection.
    for index in rng.permutation(len(candidates0)):
        box0 = candidates0[int(index)]
        others = [box for box in candidates1 if _separated(box0, box)]
        if others:
            boxes = (box0, _pick(rng, others))
            break
    else:
        raise AssertionError("two brackets did not fit")
    records = []
    for index, (shape, box, guides, pigment) in enumerate(
            zip((shape0, shape1), boxes, (guides0, guides1), (p, q))):
        y, x, _, _ = box
        h, w = shape
        r = y + ("T" in guides)
        c = x + ("L" in guides)
        pivot = (y if "T" in guides else y + h, x if "L" in guides else x + w)
        blank = _pick(rng, SIDES) if (teaching and index == 0) or rng.random() < .35 else None
        records.append(dict(bounds=(r, c, h, w), pivot=pivot, guides=guides,
                            pigment=pigment, mask=_sample_mask(rng, shape, blank),
                            matrix=_matrix(guides, controls)))
    info = dict(shape=(H, W), background=bg, elbow=elbow, controls=controls, records=records)
    grid = np.full((H, W), bg, dtype=np.int8)
    _paint_key(grid, controls, elbow, bg)
    for s in records:
        _paint_record(grid, s, elbow)
    return grid, info


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)[:6]]
    stable = bool(rng.random() < .25)
    for _ in range(64):
        grid, info = _random_input(rng, test=False, palette=palette, teaching=True)
        output = _render(info)
        swapped = _swap_controls(grid)
        swapped_info = _decode(swapped)
        swapped_output = _render(swapped_info)
        if (_useful(info, output) and _useful(swapped_info, swapped_output)
                and not np.array_equal(output, swapped_output)
                and not np.array_equal(output, _inverse_orientation(info))
                and len({tuple(np.abs(s["matrix"]) @ s["mask"].shape)
                         for s in info["records"]}) == 2
                and not np.array_equal(output, _tight(info))
                and not np.array_equal(swapped_output, _tight(swapped_info))):
            break
    else:
        raise AssertionError("teaching contrast candidate budget exhausted")
    puzzle = {"train": [{"input": grid, "output": output},
                         {"input": swapped, "output": swapped_output}], "test": []}
    seen = {_key(grid), _key(swapped)}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _ in range(count):
            for _ in range(64):
                local = palette if stable else [int(c) for c in rng.permutation(10)[:6]]
                grid, info = _random_input(rng, test=split == "test", palette=local)
                output = _render(info)
                if _key(grid) not in seen and _useful(info, output):
                    break
            else:
                raise AssertionError("distinct example candidate budget exhausted")
            seen.add(_key(grid))
            puzzle[split].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(inp)
            assert out.shape == (inp.shape[0] - 2, inp.shape[1] - 2)
            fields = [_stamp_field(s, info["shape"]) for s in info["records"]]
            assert np.array_equal(out, _combine(info, fields)), "incorrect orientation, phase or overlap"
            assert np.array_equal(out == info["elbow"], fields[0] & fields[1])
            assert np.array_equal(out == info["background"], ~fields[0] & ~fields[1])
            assert _useful(info, out), "both pigments, overlap, key and phase must be observable"
            finite = _combine(info, [_stamp_field(s, info["shape"], finite=True)
                                     for s in info["records"]])
            assert not np.array_equal(out, finite), "periodic extension must change the answer"
            if split == "train":
                assert all(max(s["mask"].shape) <= 4 for s in info["records"])
            else:
                assert any(5 in s["mask"].shape for s in info["records"]), "test needs a longer period"
            decoded[split].append(info)
    first, second = puzzle["train"][:2]
    assert np.array_equal(_swap_controls(np.asarray(first["input"])), second["input"]), \
        "the first two training inputs must form a controlled marker-identity contrast"
    assert not np.array_equal(first["output"], second["output"])
    assert any(tuple(s["matrix"].ravel()) in {tuple(MATRICES[1].ravel()), tuple(MATRICES[3].ravel())}
               for s in decoded["train"][0]["records"]), "demonstrate the direction of a quarter turn"
    assert not np.array_equal(first["output"], _inverse_orientation(decoded["train"][0]))
    assert len({tuple(np.abs(s["matrix"]) @ s["mask"].shape)
                for s in decoded["train"][0]["records"]}) == 2, "demonstrate unequal periods"
    assert any(np.linalg.det(s["matrix"]) < 0 for s in decoded["train"][1]["records"])
    for pair, info in zip(puzzle["train"][:2], decoded["train"][:2]):
        assert any(_blank_sides(s["mask"]) for s in info["records"])
        assert not np.array_equal(pair["output"], _tight(info)), "guide lengths must defeat tight cropping"


def self_test():
    """Bounded, streaming checks of 42 seeds plus geometry/count regressions."""
    metrics = dict(seeds=42, pairs=0, teaching_contrasts=0, test_period_shifts=0,
                   independent_solver=False)
    input_shapes, output_shapes, masks, fingerprints, symmetries = set(), set(), set(), set(), set()
    pigment_colors, marker_colors, blank_sides, counts = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        metrics["teaching_contrasts"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _decode(inp)
                assert np.array_equal(solve(inp), out)
                before = inp.copy()
                solve(inp)
                assert np.array_equal(inp, before), "solve mutated its input"
                metrics["pairs"] += 1
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                marker_colors.update(info["controls"].values())
                for s in info["records"]:
                    masks.add((s["mask"].shape, s["mask"].tobytes()))
                    symmetries.add(tuple(s["matrix"].ravel()))
                    pigment_colors.add(s["pigment"])
                    blank_sides.update(_blank_sides(s["mask"]))
                    counts.add(int(s["mask"].sum()))
                if split == "test":
                    metrics["test_period_shifts"] += 1
    assert len(fingerprints) == 42
    assert len(input_shapes) >= 40 and len(output_shapes) >= 40
    assert len(masks) >= 120 and len(symmetries) == 8
    assert pigment_colors == marker_colors == set(range(10))
    assert blank_sides == set(SIDES) and len(counts) >= 10
    # Exercise all eight signed coordinate maps with a known asymmetric tile,
    # at each of four possible inner guide corners, independent of sampling.
    mask = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=bool)
    for m in MATRICES:
        for py, px in ((5, 6), (5, 11), (9, 6), (9, 11)):
            record = dict(bounds=(6, 7, 3, 4), pivot=(py, px), mask=mask, matrix=m)
            assert np.array_equal(_field(record, (20, 23)), _stamp_field(record, (20, 23)))
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
    for a, b in legal:
        puzzle = generate_puzzle(10000 + 10 * a + b, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for mutation in ("cell", "crop", "guide", "identity", "origin", "tight", "inverse"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad["train"][0] if mutation in ("tight", "inverse") else bad["test"][0]
        info = _decode(pair["input"])
        if mutation == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif mutation == "crop":
            pair["output"] = pair["output"][1:]
        elif mutation == "guide":
            py, px = info["records"][0]["pivot"]
            pair["input"][py, px] = info["background"]
        elif mutation == "identity":
            pair["output"] = _combine(info, [_field(s, info["shape"], True) for s in info["records"]])
        elif mutation == "origin":
            pair["output"] = _origin_tiling(info)
        elif mutation == "inverse":
            pair["output"] = _inverse_orientation(info)
        else:
            pair["output"] = _tight(info)
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"accepted {mutation} corruption")
    metrics.update(distinct_puzzles=len(fingerprints), input_shapes=len(input_shapes),
                   output_shapes=len(output_shapes), distinct_tile_masks=len(masks),
                   symmetries=len(symmetries), motif_occupancies=sorted(counts),
                   legal_count_combinations=len(legal), negative_checks=7)
    return metrics
