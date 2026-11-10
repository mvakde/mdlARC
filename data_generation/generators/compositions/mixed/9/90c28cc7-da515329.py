"""Contract a nonuniformly enlarged hollow seed, then complete its pinwheel.

Standalone: standard library and NumPy only. No parent data, files, random state,
or demonstration labels are consulted by solve(). See task_spec.md for the
margin-preserving contraction convention and the necessary logical L=1 limit.
"""
from itertools import product

import numpy as np


HOST_SEEDS = (0, 119954157, 735575487, 317707838, 45865756, 330815530,
              868272058, 212155645, 466674563, 337176180, 264504235,
              100225593, 686636541, 141620048, 20006731, 633026841,
              108792655, 138019751, 990845547, 145062474, 492502733,
              219578423, 483512278, 879576784, 409941974, 374211522,
              204607331, 914477861, 930480914, 67655745, 333828560,
              787159520, 948563429, 796086519, 912364254, 105498866,
              880625396, 899725461, 86957163, 37622579, 640776963, 189918566)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _seed(background, foreground):
    return np.array([[background, foreground, background],
                     [foreground, background, foreground],
                     [background, foreground, background]], dtype=np.int8)


def _decode(input_grid):
    """Segment full vectors in the foreground bbox; keep exterior margins."""
    a = _grid(input_grid)
    colors = np.unique(a)
    if len(colors) != 2:
        raise ValueError("one background and one drawing color are required")
    bg = int(a[0, 0])
    if not (a[0, -1] == a[-1, 0] == a[-1, -1] == bg):
        raise ValueError("all four corners must identify the background")
    fg = int(colors[colors != bg][0])
    ys, xs = np.nonzero(a == fg)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    crop = a[top:bottom, left:right]
    ri = np.r_[0, 1 + np.flatnonzero(np.any(crop[1:] != crop[:-1], axis=1))]
    ci = np.r_[0, 1 + np.flatnonzero(np.any(crop[:, 1:] != crop[:, :-1], axis=0))]
    compact = crop[np.ix_(ri, ci)]
    if not np.array_equal(compact, _seed(bg, fg)):
        raise ValueError("maximal run collapse must recover the 3x3 hollow cross")
    rows = tuple(int(v) for v in np.diff(np.r_[ri, crop.shape[0]]))
    columns = tuple(int(v) for v in np.diff(np.r_[ci, crop.shape[1]]))
    margins = (top, a.shape[0] - bottom, left, a.shape[1] - right)
    shape = (margins[0] + 3 + margins[1], margins[2] + 3 + margins[3])
    return {"background": bg, "foreground": fg, "rows": rows,
            "columns": columns, "margins": margins, "shape": shape,
            "center": (top + 1, left + 1), "compact": compact}


def _walk_mask(shape, center):
    """B's short-seed phase: extend to radius 2, then runs 4,8,12,... .

    Endpoints are kept on the infinite plane, and only painting is clipped.
    An arm is never stopped at its first exit from the canvas.
    """
    h, w = shape
    cy, cx = center
    result = np.zeros(shape, dtype=bool)

    def paint(y0, x0, y1, x1):
        y0, y1, x0, x1 = y0 + cy, y1 + cy, x0 + cx, x1 + cx
        if y0 == y1 and 0 <= y0 < h:
            lo, hi = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
            if lo <= hi:
                result[y0, lo:hi + 1] = True
        elif x0 == x1 and 0 <= x0 < w:
            lo, hi = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
            if lo <= hi:
                result[lo:hi + 1, x0] = True

    distance = max(cy, cx, h - 1 - cy, w - 1 - cx)
    # Four turns move every later stage eight cells farther out. Two extra
    # cycles enclose all possible clipped segments, including later re-entry.
    stages = 4 * (distance // 8 + 2)
    for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        y, x = 2 * dy, 2 * dx
        paint(dy, dx, y, x)
        dy, dx = dx, -dy
        for k in range(stages):
            ny, nx = y + (4 * k + 4) * dy, x + (4 * k + 4) * dx
            paint(y, x, ny, nx)
            y, x = ny, nx
            dy, dx = dx, -dy
    return result


def _interval_mask(shape, center, handedness=1):
    """Validator cross-check: closed-form intervals, with no path walking."""
    y, x = np.indices(shape)
    y, x = y - center[0], handedness * (x - center[1])
    result = np.zeros(shape, dtype=bool)
    radius = max(int(np.abs(y).max()), int(np.abs(x).max()))
    for _ in range(4):
        result |= (x == 0) & (y >= -2) & (y <= -1)
        for q in range(radius // 8 + 3):
            starts = ((-2 - 8*q, -8*q), (-2 - 8*q, 4 + 8*q),
                      (6 + 8*q, 4 + 8*q), (6 + 8*q, -8 - 8*q))
            for s, ((sy, sx), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (y - sy)*dy + (x - sx)*dx
                across = (y - sy)*dx - (x - sx)*dy
                result |= ((across == 0) & (along >= 0)
                           & (along <= 16*q + 4*s + 4))
        y, x = x, -y
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _decode(input_grid)
    return np.where(_walk_mask(data["shape"], data["center"]),
                    data["foreground"], data["background"]).astype(np.int8)


def _bands(rng, budget, kind, cap=7):
    """Finite choice space: never an unbounded rejection loop."""
    triples = [t for t in product(range(1, min(cap, budget - 2) + 1), repeat=3)
               if 4 <= sum(t) <= budget
               and (kind != "uneven" or len(set(t)) > 1)
               and (kind != "uniform" or len(set(t)) == 1)
               and (kind != "extreme" or max(t) >= 8)]
    if not triples:
        raise ValueError("infeasible replication profile")
    return triples[int(rng.integers(len(triples)))]


def _shape(rng, low, high):
    mode = int(rng.integers(5))
    if mode == 0:
        side = int(rng.integers(low, high + 1))
        return side, side
    if mode == 1:
        short = int(rng.integers(low, min(high, low + 3) + 1))
        long = int(rng.integers(max(short, high - 3), high + 1))
        return (short, long) if rng.random() < .5 else (long, short)
    return tuple(int(v) for v in rng.integers(low, high + 1, size=2))


def _center(rng, shape, mode):
    h, w = shape
    if mode == "teaching":
        return int(rng.integers(4, h - 4)), int(rng.integers(4, w - 4))
    if mode == "corner":
        return int(rng.choice((1, h - 2))), int(rng.choice((1, w - 2)))
    if mode == "edge":
        axis = int(rng.integers(2))
        values = [int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))]
        values[axis] = int(rng.choice((1, shape[axis] - 2)))
        return tuple(values)
    if rng.random() < .25:
        return (h - 1) // 2, (w - 1) // 2
    values = [int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))]
    # Corner transfer is taught one boundary at a time, then combines two.
    if values[0] in (1, h - 2) and values[1] in (1, w - 2):
        values[1] = w // 2
    return tuple(values)


def _make_input(shape, center, rows, columns, bg, fg):
    cy, cx = center
    h, w = shape
    top, left = cy - 1, cx - 1
    enlarged = np.repeat(np.repeat(_seed(bg, fg), rows, axis=0), columns, axis=1)
    a = np.full((h - 3 + sum(rows), w - 3 + sum(columns)), bg, dtype=np.int8)
    a[top:top + enlarged.shape[0], left:left + enlarged.shape[1]] = enlarged
    return a


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Sample one coherent rule; regimes affect difficulty, never semantics."""
    ntrain, ntest = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("scale", "extent", "corner")))
    common_palette = bool(rng.random() < .25)
    first_palette = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    used, train_shapes = set(), []
    puzzle = {"train": [], "test": []}
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if common_palette or (split == "train" and index == 0):
                bg, fg = first_palette
            elif split == "train" and index == 1:
                fg, bg = first_palette
            else:
                bg, fg = (int(v) for v in rng.choice(10, size=2, replace=False))
            # At most 48 bounded attempts; only tiny grids are retained.
            for _ in range(48):
                if split == "train" and index == 0:
                    shape = _shape(rng, 12, 17)
                    mode = "teaching"
                elif split == "train":
                    shape = _shape(rng, 7, 18 if regime == "extent" else 24)
                    mode = "edge" if index == 1 else "mixed"
                elif regime == "extent":
                    shape = tuple(int(rng.integers(max(s[d] for s in train_shapes) + 1, 28))
                                  for d in (0, 1))
                    mode = "mixed"
                else:
                    shape = _shape(rng, 7, 22 if regime == "scale" else 25)
                    mode = "corner" if regime == "corner" else "mixed"
                if split == "train" and index == 1 and shape == train_shapes[0]:
                    continue
                center = _center(rng, shape, mode)
                kinds = ["uneven", "uneven"]
                if not (split == "train" and index == 0) and rng.random() < .3:
                    kinds[int(rng.integers(2))] = "uniform"
                cap = 7
                if split == "test" and regime == "scale":
                    kinds[int(rng.integers(2))] = "extreme"
                    cap = 9
                rows = _bands(rng, 33 - shape[0], kinds[0], cap)
                columns = _bands(rng, 33 - shape[1], kinds[1], cap)
                if split == "train" and index == 0 and rows == columns:
                    columns = columns[1:] + columns[:1]
                a = _make_input(shape, center, rows, columns, bg, fg)
                if _key(a) in used:
                    continue
                b = np.where(_walk_mask(shape, center), fg, bg).astype(np.int8)
                puzzle[split].append({"input": a, "output": b})
                used.add(_key(a))
                if split == "train":
                    train_shapes.append(shape)
                break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate visible geometry, lossless contraction, labels and teaching."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            d = _decode(a)
            bg, fg = d["background"], d["foreground"]
            h, w = d["shape"]
            cy, cx = d["center"]
            assert min(h, w) >= 7
            # Reconstruct the entire original grid, including empty cells
            # inside the bbox and all exterior padding.
            reconstructed = _make_input(d["shape"], d["center"], d["rows"],
                                        d["columns"], bg, fg)
            assert np.array_equal(a, reconstructed), "not a replicated hollow seed"
            assert sum(d["rows"]) > 3 and sum(d["columns"]) > 3
            assert a.shape[0] > h and a.shape[1] > w, "collapse must matter in both axes"
            assert b.shape == (h, w)
            assert set(map(int, np.unique(b))) == {bg, fg}
            expected = _interval_mask((h, w), (cy, cx))
            assert np.array_equal(b == fg, expected), "wrong clockwise continuation"
            assert b[cy, cx] == bg
            seed_mask = np.zeros((h, w), dtype=bool)
            seed_mask[cy-1:cy+2, cx-1:cx+2] = _seed(0, 1).astype(bool)
            assert np.all(expected[seed_mask]) and np.any(expected & ~seed_mask)
            assert not np.any(expected[:-1, :-1] & expected[1:, :-1]
                              & expected[:-1, 1:] & expected[1:, 1:])
            information[split].append(d)
    train, test = information["train"], information["test"]
    assert len({d["shape"] for d in train}) >= 2
    assert any(min(d["margins"]) >= 3 for d in train), "need visible turns"
    assert any(0 in d["margins"] for d in train), "teach boundary clipping"
    assert any(len(set(d["rows"])) > 1 and len(set(d["columns"])) > 1
               and d["rows"] != d["columns"] for d in train)
    assert any(not np.array_equal(_interval_mask(d["shape"], d["center"]),
                                 _interval_mask(d["shape"], d["center"], -1)) for d in train)
    train_peak = max(max(d["rows"] + d["columns"]) for d in train)
    for d in test:
        m = d["margins"]
        larger_bands = max(d["rows"] + d["columns"]) > train_peak
        larger_canvas = all(d["shape"][axis] > max(t["shape"][axis] for t in train)
                            for axis in (0, 1))
        corner_transfer = (min(m[:2]) == min(m[2:]) == 0
                           and all(not (min(t["margins"][:2]) ==
                                        min(t["margins"][2:]) == 0) for t in train))
        assert larger_bands or larger_canvas or corner_transfer, "missing train/test shift"


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded tests: stream 42 supplied seeds; keep only compact statistics."""
    # Parent B's original L=1 fixture protects the specialized recurrence.
    source_rows = ("080888888888", "080800000008", "080808888808",
                   "080808000808", "080880880808", "080008080808",
                   "088888080808", "000000080808", "888888880808",
                   "000000000808", "888888888808", "000000000008")
    fixture = np.array([[int(c) for c in row] for row in source_rows], dtype=np.int8)
    expanded = _make_input((12, 12), (4, 5), (3, 1, 2), (1, 4, 2), 0, 8)
    assert np.array_equal(solve(expanded), fixture)
    assert np.array_equal(_interval_mask((12, 12), (4, 5)), fixture == 8)
    assert _decode(expanded)["rows"] == (3, 1, 2)
    # Separated identical vectors must remain separate: A,B,A, not A,B.
    compact = _decode(expanded)["compact"]
    assert compact.shape == (3, 3) and np.array_equal(compact[0], compact[2])
    # The solver's rule is wider than generation's modest factor vocabulary.
    # Include a dense foreground-majority crop, an unexpanded axis, and an
    # entirely unexpanded seed; no sampling regime enters prediction.
    for shape, center, rows, columns in (
            ((3, 3), (1, 1), (24, 1, 5), (1, 28, 1)),
            ((3, 3), (1, 1), (1, 20, 1), (10, 1, 10)),
            ((5, 7), (1, 5), (9, 1, 10), (1, 1, 1)),
            ((30, 30), (1, 28), (1, 1, 1), (1, 1, 1))):
        a = _make_input(shape, center, rows, columns, 9, 0)
        expected = np.where(_interval_mask(shape, center), 0, 9)
        assert np.array_equal(solve(a), expected)
    for shape, center in (((30, 30), (1, 1)), ((7, 29), (5, 27)),
                          ((27, 9), (13, 4)), ((17, 17), (8, 8))):
        mask = _walk_mask(shape, center)
        assert np.array_equal(mask, _interval_mask(shape, center))
        rotated_center = (shape[1] - 1 - center[1], center[0])
        assert np.array_equal(np.rot90(mask), _walk_mask(shape[::-1], rotated_center))
        big = _walk_mask((90, 90), (45, 45))
        top, left = 45 - center[0], 45 - center[1]
        assert np.array_equal(mask, big[top:top+shape[0], left:left+shape[1]])
    shapes, layouts, profiles, palettes, signatures = set(), set(), set(), set(), set()
    colors, touches, shifts = set(), [0, 0, 0, 0], [0, 0, 0]
    pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        assert _equal(p, generate_puzzle(seed)), "nondeterminism"
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        data = {s: [_decode(pair["input"]) for pair in p[s]] for s in ("train", "test")}
        peak = max(max(d["rows"] + d["columns"]) for d in data["train"])
        for s in ("train", "test"):
            for pair, d in zip(p[s], data[s]):
                pairs += 1
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add((np.asarray(pair["input"]).shape, d["shape"]))
                layouts.add((d["shape"], d["center"]))
                profiles.add((d["rows"], d["columns"]))
                palettes.add((d["background"], d["foreground"]))
                colors.update((d["background"], d["foreground"]))
                for side in range(4):
                    touches[side] += d["margins"][side] == 0
                if s == "test":
                    shifts[0] += max(d["rows"] + d["columns"]) > peak
                    shifts[1] += all(d["shape"][i] > max(t["shape"][i] for t in data["train"])
                                     for i in (0, 1))
                    shifts[2] += min(d["margins"][:2]) == min(d["margins"][2:]) == 0
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 120 and len(layouts) >= 100 and len(profiles) >= 100
    assert len(palettes) >= 40 and len(colors) == 10
    assert min(touches) > 0 and min(shifts) > 0
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(1729, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validation must reject pixel errors, a reflected answer and collapse-only.
    for mutation in ("pixel", "reflection", "no_winding", "duplicate"):
        p = generate_puzzle(0)
        pair = p["test"][0]
        d = _decode(pair["input"])
        if mutation == "pixel":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif mutation == "reflection":
            pair["output"] = np.where(_interval_mask(d["shape"], d["center"], -1),
                                      d["foreground"], d["background"]).astype(np.int8)
        elif mutation == "no_winding":
            pair["output"] = np.full(d["shape"], d["background"], dtype=np.int8)
            cy, cx = d["center"]
            pair["output"][cy-1:cy+2, cx-1:cx+2] = d["compact"]
        else:
            p["test"][0] = {k: v.copy() for k, v in p["train"][0].items()}
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("invalid puzzle accepted: " + mutation)
    return {"seeds": 42, "pairs": pairs, "input_output_shapes": len(shapes),
            "normalized_layouts": len(layouts), "replication_profiles": len(profiles),
            "ordered_palettes": len(palettes), "edge_contacts": list(map(int, touches)),
            "test_shift_counts": list(map(int, shifts))}
