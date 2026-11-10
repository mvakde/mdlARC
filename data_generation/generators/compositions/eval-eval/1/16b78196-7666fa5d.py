"""Restore a shattered mosaic inside a continuously bracketed diagonal mold.

Only standard Python and NumPy are used. Labels are made from a sampled connected
partition; the input-only solver searches arbitrary translations of every tile.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(p):
    r, c = p
    return ((r-1, c), (r+1, c), (r, c-1), (r, c+1))


def _components(cells):
    unseen = set(cells)
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, stack = {start}, [start]
        while stack:
            for p in _neighbors(stack.pop()):
                if p in unseen:
                    unseen.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _dilate(mask):
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    h, w = mask.shape
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr:dr+h, dc:dc+w]
    return result


def _read_strokes(cells, slope):
    lines = {}
    for r, c in cells:
        lines.setdefault(c-slope*r, []).append(r)
    strokes = []
    for q, rows in sorted(lines.items()):
        runs = []
        for r in sorted(rows):
            if not runs or r != runs[-1][-1]+1:
                runs.append([])
            runs[-1].append(r)
        for run in runs:
            if len(run) < 2:
                return None
            strokes.append((q, 2*run[0]+slope*q,
                            2*run[-1]+slope*q))
    return strokes if len(strokes) >= 2 else None


def _mold(shape, strokes, slope, frame_mask, alternative=""):
    """Continuous diagonal coordinates: q=c-s*r, t=r+s*c."""
    r, c = np.indices(shape)
    q, t = c-slope*r, r+slope*c
    result = np.zeros(shape, dtype=bool)
    ordered = sorted(strokes)
    if alternative == "consecutive":
        pairs = list(zip(ordered, ordered[1:]))
    elif alternative == "global_extremes":
        pairs = [(ordered[0], ordered[-1])]
    else:
        pairs = [(a, b) for i, a in enumerate(ordered) for b in ordered[i+1:]]
    # The union of all pairwise brackets equals the span of the outermost
    # active strokes on each longitudinal scanline.
    for (qa, la, ha), (qb, lb, hb) in pairs:
        between = ((t >= max(la, lb)) & (t <= min(ha, hb))
                   & (q > min(qa, qb)) & (q < max(qa, qb)))
        if alternative == "pixel_centers":
            between &= ((t-qa) % 2 == 0) & ((t-qb) % 2 == 0)
        result |= between
    return result & ~frame_mask


def _sweep_mask(shape, strokes, slope, frame_mask):
    """A separate scanline calculation checks the pair-union geometry."""
    r, c = np.indices(shape)
    q, t = c-slope*r, r+slope*c
    result = np.zeros(shape, dtype=bool)
    for level in range(int(t.min()), int(t.max())+1):
        active = [sq for sq, lo, hi in strokes if lo <= level <= hi]
        if len(active) >= 2:
            result |= (t == level) & (q > min(active)) & (q < max(active))
    return result & ~frame_mask


def _scene(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size/2, "background must occupy a strict majority"
    candidates = []
    for color in np.flatnonzero(counts):
        color = int(color)
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(grid == color)))
        # Frame pixels have diagonal adjacency only; each tile has orthogonal
        # adjacency. This geometric distinction survives any palette change.
        if any(p in cells for cell in cells for p in _neighbors(cell)):
            continue
        for slope in (-1, 1):
            strokes = _read_strokes(cells, slope)
            if strokes is not None:
                candidates.append((color, slope, strokes))
    assert len(candidates) == 1, "expected one recognizable diagonal frame"
    frame, slope, strokes = candidates[0]
    fm = grid == frame
    target = _mold(grid.shape, strokes, slope, fm)
    assert target.any(), "frame must bracket a nonempty mold"
    pieces = []
    forbidden = _dilate(fm)
    for color in np.flatnonzero(counts):
        color = int(color)
        if color in (bg, frame):
            continue
        for group in _components(map(tuple, np.argwhere(grid == color))):
            cells = np.array(sorted(group), dtype=int)
            assert len(cells) >= 3, "tiles contain at least three cells"
            assert not np.any(forbidden[cells[:, 0], cells[:, 1]]), "tiles must be detached"
            forbidden |= _dilate(_mask(grid.shape, cells))
            assert not np.any(target[cells[:, 0], cells[:, 1]]), "mold is initially empty"
            pieces.append((cells, color))
    assert len(pieces) >= 2, "expected multiple movable tiles"
    assert sum(len(cells) for cells, _ in pieces) == int(target.sum()), "inventory area differs from mold"
    return dict(background=bg, frame=frame, slope=slope, strokes=strokes,
                frame_mask=fm, target=target, pieces=pieces)


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    result[cells[:, 0], cells[:, 1]] = True
    return result


def _pack(target, pieces, base):
    """Exhaustive exact cover; stop after two DISTINCT colored outputs.

    All integer translations are enumerated. No layer, width, position, nearest
    socket or color-order heuristic constrains a destination. Identical tiles
    of identical color are interchangeable, so search one identity at a time.
    """
    positions = np.argwhere(target)
    index = {tuple(p): i for i, p in enumerate(positions)}
    full = (1 << len(positions))-1
    options, identities = [], []
    for i, (source, color) in enumerate(pieces):
        local = source-source.min(axis=0)
        identity = (color, tuple(map(tuple, local)))
        identities.append(identity)
        tile_options = []
        # Any placement sends local[0] to one of the target's cells.
        for anchor in positions:
            cells = local+(anchor-local[0])
            ids = [index.get(tuple(p)) for p in cells]
            if all(k is not None for k in ids):
                bits = sum(1 << k for k in ids)
                tile_options.append((i, bits, cells))
        options.append(tile_options)
    solutions = {}

    def search(used, covered, chosen):
        if len(solutions) >= 2:
            return
        if covered == full:
            if used != (1 << len(pieces))-1:
                return
            answer = base.copy()
            for i, _, cells in chosen:
                answer[cells[:, 0], cells[:, 1]] = pieces[i][1]
            solutions.setdefault(answer.tobytes(), (answer, chosen.copy()))
            return
        representatives = []
        seen = set()
        for i, identity in enumerate(identities):
            if not used & (1 << i) and identity not in seen:
                representatives.append(i)
                seen.add(identity)
        # Piece multiplicity is a constraint even for equivalent identities.
        for i in range(len(pieces)):
            if not used & (1 << i) and not any(not bits & covered for _, bits, _ in options[i]):
                return
        available = [o for i in representatives for o in options[i] if not o[1] & covered]
        best = None
        for k in range(len(positions)):
            if covered & (1 << k):
                continue
            hits = [o for o in available if o[1] & (1 << k)]
            if not hits:
                return
            if best is None or len(hits) < len(best):
                best = hits
                if len(best) == 1:
                    break
        for i, bits, cells in best:
            search(used | (1 << i), covered | bits, chosen+[(i, bits, cells)])

    search(0, 0, [])
    return list(solutions.values())


def _resolve(grid):
    info = _scene(grid)
    base = np.full_like(grid, info["background"])
    base[info["frame_mask"]] = info["frame"]
    solutions = _pack(info["target"], info["pieces"], base)
    assert len(solutions) == 1, "mold must have one distinct colored tiling"
    output, placements = solutions[0]
    info["placements"] = placements
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from this input, without puzzle-level state."""
    return _resolve(_grid(input_grid))[0]


def _stroke_specs(rng, kind):
    if kind == "bracket":
        qs = [0, int(rng.integers(6, 11))]
        starts = [int(rng.integers(-3, 1)), int(rng.integers(1, 5))]
        lengths = [int(rng.integers(4, 8)), int(rng.integers(4, 8))]
    elif kind == "bypass":
        qs = [0, int(rng.integers(4, 7))]
        qs.append(qs[-1]+int(rng.integers(4, 7)))
        starts = [int(rng.integers(-6, -3)), int(rng.integers(-1, 2)),
                  int(rng.integers(-6, -3))]
        lengths = [int(rng.integers(7, 10)), int(rng.integers(2, 4)),
                   int(rng.integers(7, 10))]
        # A delayed fourth stroke changes the outer bracket. Together with
        # the short interior stroke this teaches BOTH distinctions in one
        # example, which is essential when only two training pairs are asked.
        qs.append(qs[-1]+int(rng.integers(4, 7)))
        starts.append(int(rng.integers(3, 7)))
        lengths.append(int(rng.integers(3, 6)))
    else:
        count = int(rng.integers(3, 6))
        qs = [0]
        for _ in range(count-1):
            qs.append(qs[-1]+int(rng.integers(4, 7)))
        order = list(range(count))
        if rng.random() < .5:
            order.reverse()
        starts = [0]*count
        offset = int(rng.integers(-5, 0))
        for i in order:
            starts[i] = offset
            offset += int(rng.integers(2, 5))
        lengths = [int(rng.integers(4, 8)) for _ in qs]
    result = []
    for q, start, length in zip(qs, starts, lengths):
        lo = start-(start-q) % 2
        result.append((q, lo, lo+2*(length-1)))
    return result


def _partition(rng, target, count):
    """Competitive connected growth makes jagged seams and unequal tile areas."""
    all_cells = set(map(tuple, np.argwhere(target)))
    regions = _components(all_cells)
    if len(regions) > count or any(len(g) < 3 for g in regions):
        raise ValueError("mold components cannot support the chosen inventory")
    seeds = [sorted(g)[int(rng.integers(len(g)))] for g in regions]
    while len(seeds) < count:
        pool = sorted(all_cells-set(seeds))
        scores = [min(abs(p[0]-s[0])+abs(p[1]-s[1]) for s in seeds)
                  + rng.random()*2 for p in pool]
        seeds.append(pool[int(np.argmax(scores))])
    groups = [{p} for p in seeds]
    unassigned = all_cells-set(seeds)
    weights = rng.uniform(.7, 1.4, size=count)
    while unassigned:
        frontiers = [sorted({p for cell in group for p in _neighbors(cell)} & unassigned)
                     for group in groups]
        viable = [i for i, f in enumerate(frontiers) if f]
        assert viable
        i = min(viable, key=lambda j: len(groups[j])/weights[j]+rng.random()*2)
        p = frontiers[i][int(rng.integers(len(frontiers[i])))]
        groups[i].add(p)
        unassigned.remove(p)
    if any(len(group) < 3 for group in groups):
        raise ValueError("undersized partition tile")
    return [np.array(sorted(group), dtype=int) for group in groups]


def _crossing_piece(pieces, strokes, slope):
    """Does a tile cross an interior seam on a scanline where it is inactive?"""
    for cells, _ in pieces:
        qs = cells[:, 1]-slope*cells[:, 0]
        ts = cells[:, 0]+slope*cells[:, 1]
        for q, lo, hi in sorted(strokes)[1:-1]:
            for t in np.unique(ts):
                if not lo <= t <= hi:
                    values = qs[ts == t]
                    if values.min() < q < values.max():
                        return True
    return False


def _scatter(rng, base, target, pieces):
    h, w = base.shape
    for _ in range(16):
        result = base.copy()
        # A frame endpoint may occupy a corner: recognize the majority field.
        bg = int(np.argmax(np.bincount(base.ravel(), minlength=10)))
        forbidden = _dilate((base != bg) | target)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*4)
        success = True
        for i in order:
            cells, color = pieces[i]
            local = cells-cells.min(axis=0)
            th, tw = local.max(axis=0)+1
            valid = np.ones((h-th+1, w-tw+1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r+valid.shape[0], c:c+valid.shape[1]]
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            destination = local+choices[int(rng.integers(len(choices)))]
            result[destination[:, 0], destination[:, 1]] = color
            forbidden |= _dilate(_mask(base.shape, destination))
        if success:
            return result
    raise ValueError("insufficient scatter clearance")


def _make_pair(rng, kind, slope, palette, require_crossing=False):
    bg, frame = palette[:2]
    for _ in range(240):
        try:
            specs = _stroke_specs(rng, kind)
            cells = np.array([((t-slope*q)//2, (q+slope*t)//2)
                              for q, lo, hi in specs for t in range(lo, hi+1, 2)])
            span = cells.max(axis=0)-cells.min(axis=0)+1
            if max(span) > 20:
                continue
            minimum = 16 if kind == "bracket" else 18
            dims = tuple(int(rng.integers(max(minimum, int(x)+6), 31)) for x in span)
            origin = np.array([int(rng.integers(n-int(x)+1)) for n, x in zip(dims, span)])
            cells = cells-cells.min(axis=0)+origin
            base = np.full(dims, bg, dtype=np.int8)
            base[cells[:, 0], cells[:, 1]] = frame
            strokes = _read_strokes(map(tuple, cells), slope)
            target = _mold(dims, strokes, slope, base == frame)
            area = int(target.sum())
            if not 18 <= area <= 110:
                continue
            low, high = (2, 3) if kind == "bracket" else ((3, 5) if kind == "bypass" else (4, 6))
            count = int(rng.integers(low, min(high, area//4)+1))
            groups = _partition(rng, target, count)
            # Two distinct colors are always used; later components may reuse
            # a color. Object identity is never equated to an entire color class.
            colors = list(palette[2:])
            rng.shuffle(colors)
            if count > 2 and rng.random() < .5:
                pool_size = int(rng.integers(2, count))
                assignments = colors[:2]+[int(rng.choice(colors[:pool_size])) for _ in range(count-2)]
            else:
                assignments = colors[:count]
            rng.shuffle(assignments)
            pieces = list(zip(groups, assignments))
            if require_crossing and not _crossing_piece(pieces, strokes, slope):
                continue
            if kind == "bypass" and np.array_equal(target, _mold(dims, strokes, slope, base == frame, "consecutive")):
                continue
            if kind in ("switch", "bypass") and np.array_equal(target, _mold(dims, strokes, slope, base == frame, "global_extremes")):
                continue
            answers = _pack(target, pieces, base)
            if len(answers) != 1:
                continue
            output = base.copy()
            for piece, color in pieces:
                output[piece[:, 0], piece[:, 1]] = color
            if not np.array_equal(answers[0][0], output):
                continue
            inp = _scatter(rng, base, target, pieces)
            if np.count_nonzero(inp == bg) <= inp.size/2:
                continue
            # Input recognition and exhaustive tiling must recover the partition
            # after all construction origins have been discarded.
            predicted, _ = _resolve(inp)
            if not np.array_equal(predicted, output):
                continue
            return {"input": inp, "output": output}
        except (ValueError, AssertionError):
            continue
    raise RuntimeError("bounded mosaic construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                  ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    stable = bool(rng.integers(2))
    slope = int(rng.choice([-1, 1]))
    mixed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    # Even two demonstrations teach continuous intervals and bypassing a short
    # interior seam. Tests recombine these with changing outermost boundaries.
    train_kinds = ["bracket", "bypass"]+[str(rng.choice(["bracket", "bypass", "switch"]))
                                        for _ in range(num_train-2)]
    for split, kinds in (("train", train_kinds), ("test", ["switch"]*num_test)):
        for i, kind in enumerate(kinds):
            pair_slope = slope*(-1 if mixed and i % 2 else 1)
            roles = palette if stable else [int(x) for x in rng.permutation(10)]
            for _ in range(8):
                pair = _make_pair(rng, kind, pair_slope, roles, require_crossing=(kind == "bypass"))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, shape):
    target, strokes, slope = info["target"], info["strokes"], info["slope"]
    points = np.argwhere(info["frame_mask"])
    bounding = np.zeros(shape, dtype=bool)
    lower, upper = points.min(axis=0), points.max(axis=0)
    bounding[lower[0]:upper[0]+1, lower[1]:upper[1]+1] = True
    bounding &= ~info["frame_mask"]
    return dict(strokes=len(strokes), tiles=len(info["pieces"]), area=int(target.sum()),
                skip=not np.array_equal(target, _mold(shape, strokes, slope, info["frame_mask"], "consecutive")),
                continuous=not np.array_equal(target, _mold(shape, strokes, slope, info["frame_mask"], "pixel_centers")),
                switch=not np.array_equal(target, _mold(shape, strokes, slope, info["frame_mask"], "global_extremes")),
                bounding=not np.array_equal(target, bounding),
                crossing=_crossing_piece([(cells, info["pieces"][i][1]) for i, _, cells in info["placements"]], strokes, slope),
                repeat=len({c for _, c in info["pieces"]}) < len(info["pieces"]),
                slope=slope, roles=(info["background"], info["frame"]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _resolve(inp)
            assert np.array_equal(out, expected), "incorrect mosaic label"
            assert np.array_equal(info["target"], _sweep_mask(inp.shape, info["strokes"], info["slope"], info["frame_mask"])), "diagonal geometry disagreement"
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)), "color inventory must be conserved"
            assert np.array_equal(inp == info["frame"], out == info["frame"]), "fixed strokes changed"
            assert len({c for _, c in info["pieces"]}) >= 2, "uniform filling must fail"
            assert np.all(out[info["target"]] != info["background"])
            used, occupied = set(), set()
            for i, _, cells in info["placements"]:
                assert i not in used
                used.add(i)
                source, color = info["pieces"][i]
                translated = source+cells.min(axis=0)-source.min(axis=0)
                assert set(map(tuple, translated)) == set(map(tuple, cells))
                assert not occupied.intersection(map(tuple, cells))
                occupied.update(map(tuple, cells))
                assert np.all(out[source[:, 0], source[:, 1]] == info["background"])
                assert np.all(out[cells[:, 0], cells[:, 1]] == color)
            assert used == set(range(len(info["pieces"])))
            assert occupied == set(map(tuple, np.argwhere(info["target"])))
            assert np.all(out[~(info["target"] | info["frame_mask"])] == info["background"])
            features[split].append(_features(info, inp.shape))
    assert any(f["skip"] and f["crossing"] for f in features["train"]), "missing seam-bypass demonstration"
    assert any(f["continuous"] for f in features["train"]), "missing continuous-interval demonstration"
    assert any(f["bounding"] for f in features["train"]), "missing endpoint-boundary demonstration"
    if any(f["switch"] for f in features["test"]):
        assert any(f["switch"] for f in features["train"]), "missing changing-boundary demonstration"


def self_test():
    """Bounded semantic fixtures, corruptions, API boundaries and 42-seed audit."""
    # Hand-counted continuous bracket: target has seven cells, four missed by
    # requiring actual frame pixels on the same perpendicular scanline.
    frame = np.zeros((5, 5), dtype=bool)
    frame[(0, 1, 2, 2, 3, 4), (2, 1, 0, 4, 3, 2)] = True
    strokes = [(2, -2, 2), (6, -2, 2)]
    target = _mold(frame.shape, strokes, -1, frame)
    expected = np.zeros_like(frame)
    expected[(1, 1, 2, 2, 2, 3, 3), (2, 3, 1, 2, 3, 1, 2)] = True
    assert np.array_equal(target, expected)
    assert np.count_nonzero(target & ~_mold(frame.shape, strokes, -1, frame, "pixel_centers")) == 4
    # Equal shapes of different colors yield ambiguity; equal shapes and colors
    # yield one colored answer despite interchangeable component identities.
    small = np.zeros((3, 4), dtype=np.int8)
    tile = np.array([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
    assert len(_pack(np.ones_like(small, dtype=bool), [(tile, 2), (tile, 3)], small)) == 2
    assert len(_pack(np.ones_like(small, dtype=bool), [(tile, 2), (tile, 2)], small)) == 1
    signatures, shapes, structures, records, role_sets = set(), set(), set(), [], set()
    molds, tile_shapes, interval_shapes = set(), set(), set()
    for seed in range(42):
        train_count = 2+seed % 9
        test_count = 1+seed % min(4, train_count-1)
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                output, info = _resolve(inp)
                assert np.array_equal(inp, before), "solver mutated the input"
                assert np.array_equal(output, pair["output"])
                shapes.add(inp.shape)
                roles = {}
                structures.add((inp.shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                record = _features(info, inp.shape)
                records.append(record)
                local_roles.add(record["roles"])
                points = np.argwhere(info["target"])
                lower, upper = points.min(axis=0), points.max(axis=0)+1
                crop = info["target"][lower[0]:upper[0], lower[1]:upper[1]]
                molds.add((crop.shape, crop.tobytes()))
                q0 = min(q for q, _, _ in info["strokes"])
                t0 = min(lo for _, lo, _ in info["strokes"])
                interval_shapes.add(tuple((q-q0, lo-t0, hi-t0) for q, lo, hi in info["strokes"]))
                for cells, _ in info["pieces"]:
                    tile_shapes.add(tuple(map(tuple, cells-cells.min(axis=0))))
        role_sets.add(len(local_roles))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(structures) >= 200
    assert len(molds) >= 200 and len(interval_shapes) >= 200
    assert len(tile_shapes) >= 400
    assert {r["slope"] for r in records} == {-1, 1}
    assert {2, 3, 4, 5} <= {r["strokes"] for r in records}
    assert {2, 3, 4, 5, 6} <= {r["tiles"] for r in records}
    assert max(r["area"] for r in records)-min(r["area"] for r in records) >= 50
    assert sum(r["repeat"] for r in records) >= 30
    assert sum(r["crossing"] for r in records) >= 42
    assert sum(r["switch"] for r in records) >= 42
    assert 1 in role_sets and max(role_sets) >= 3
    original = generate_puzzle(0, 2, 1)
    repeat = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in original for a, b in zip(original[s], repeat[s]) for k in a)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            _counts(a, b)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in original[s]] for s in original}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("identity label was accepted")
    # Correct labels alone do not prove a sufficient teaching set. Keep two
    # correctly solved simple brackets while withholding the required bypass.
    other = generate_puzzle(1, 2, 1)
    bracket_a = next(p for p in original["train"] if len(_scene(p["input"])["strokes"]) == 2)
    bracket_b = next(p for p in other["train"] if len(_scene(p["input"])["strokes"]) == 2)
    inadequate = {"train": [bracket_a, bracket_b], "test": original["test"]}
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as error:
        assert "seam-bypass" in str(error)
    else:
        raise AssertionError("insufficient demonstrations were accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len(shapes),
            "color_normalized_inputs": len(structures),
            "distinct_cropped_molds": len(molds),
            "distinct_interval_layouts": len(interval_shapes),
            "distinct_tile_masks": len(tile_shapes),
            "stroke_counts": sorted({r["strokes"] for r in records}),
            "tile_counts": sorted({r["tiles"] for r in records}),
            "repeat_color_pairs": sum(r["repeat"] for r in records),
            "crossing_tile_pairs": sum(r["crossing"] for r in records)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
