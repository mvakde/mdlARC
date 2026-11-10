"""Consensus scars: a square aperture emerges from two damaged wallpapers.

Only standard-library modules and NumPy are used.  The solver has no access to
generation parameters, examples, random state, or files.
"""

from __future__ import annotations

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its allowed range")
    return value


def _grid(value):
    result = np.asarray(value)
    _require(result.ndim == 2 and all(1 <= n <= 30 for n in result.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(result.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= result) & (result <= 9))), "invalid grid color")
    return result.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _split(grid):
    """Find the sole central, full, panel-exclusive separator."""
    h, w = grid.shape
    candidates = []
    for axis, length in ((1, w), (0, h)):
        if length < 3 or length % 2 == 0:
            continue
        mid = length // 2
        if axis == 1:
            a, line, b = grid[:, :mid], grid[:, mid], grid[:, mid + 1:]
        else:
            a, line, b = grid[:mid, :], grid[mid, :], grid[mid + 1:, :]
        color = int(line.flat[0])
        if np.all(line == color) and not np.any(a == color) and not np.any(b == color):
            candidates.append((axis, a, b, color))
    _require(len(candidates) == 1, "separator must be unambiguous")
    return candidates[0]


def _recover(a, b):
    """Unique least-disagreement wallpaper with row/column periods <= 4.

    A phase needs at least six observations and a strict majority. Equivalent
    period descriptions are harmless; equally good *different fields* are not.
    """
    h, w = a.shape
    rr, cc = np.indices(a.shape)
    values = np.concatenate((a.ravel(), b.ravel())).astype(int)
    best_error, best, best_period = a.size * 2 + 1, None, None
    ambiguous = False
    for p in range(1, min(4, h) + 1):
        for q in range(1, min(4, w) + 1):
            phase = (rr % p) * q + cc % q
            phases = np.tile(phase.ravel(), 2)
            counts = np.bincount(phases * 10 + values, minlength=p * q * 10).reshape(p * q, 10)
            totals = counts.sum(axis=1)
            modes = counts.argmax(axis=1)
            support = counts[np.arange(p * q), modes]
            if np.any(totals < 6) or np.any(support * 2 <= totals):
                continue
            error = int(values.size - support.sum())
            field = modes[phase].astype(np.int8)
            if error < best_error:
                best_error, best, best_period = error, field, (p, q)
                ambiguous = False
            elif error == best_error:
                if not np.array_equal(field, best):
                    ambiguous = True
                elif (p * q, p, q) < (best_period[0] * best_period[1], *best_period):
                    best_period = (p, q)
    _require(best is not None and not ambiguous, "wallpaper is not uniquely recoverable")
    return best, best_period


def _perimeter(shape, frame):
    r, c, s = frame
    result = np.zeros(shape, dtype=bool)
    result[r, c:c + s] = True
    result[r + s - 1, c:c + s] = True
    result[r:r + s, c] = True
    result[r:r + s, c + s - 1] = True
    return result


def _frames(mask):
    """All all-true square perimeters, side >= 4; no color assumptions."""
    h, w = mask.shape
    hp = np.pad(np.cumsum(mask, axis=1), ((0, 0), (1, 0)))
    vp = np.pad(np.cumsum(mask, axis=0), ((1, 0), (0, 0)))
    result = []
    for s in range(4, min(h, w) + 1):
        rows = hp[:, s:] - hp[:, :-s]
        cols = vp[s:, :] - vp[:-s, :]
        good = ((rows[:h - s + 1] == s) & (rows[s - 1:] == s)
                & (cols[:, :w - s + 1] == s) & (cols[:, s - 1:] == s))
        result.extend((int(r), int(c), s) for r, c in zip(*np.nonzero(good)))
    return result


def _analyze(input_grid):
    grid = _grid(input_grid)
    axis, a, b, marker = _split(grid)
    field, period = _recover(a, b)
    da, db = a != field, b != field
    common = da & db
    frames = _frames(common)
    _require(len(frames) == 1, "shared damage must define exactly one square")
    return axis, a, b, marker, field, period, da, db, common, frames[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover, intersect deviations, and return the inclusive consensus crop."""
    _, _, _, marker, field, _, _, _, common, (r, c, s) = _analyze(input_grid)
    consensus = np.where(common, marker, field).astype(np.int8)
    return consensus[r:r + s, c:c + s].copy()


def _field(rng, shape, palette, style):
    if style == "outlines":
        p = q = 4
        tile = np.full((4, 4), palette[0], dtype=np.int8)
        tile[1:3, 1:3] = rng.choice(palette[1:], size=(2, 2))
    elif style == "stripes":
        p, q = (1, int(rng.integers(2, 5))) if rng.integers(2) else (int(rng.integers(2, 5)), 1)
        tile = rng.choice(palette, size=(p, q))
        tile.flat[0], tile.flat[-1] = palette[0], palette[1]
    else:
        p, q = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        if style == "woven":
            row = rng.integers(0, len(palette), size=p)
            col = rng.integers(0, len(palette), size=q)
            tile = palette[(row[:, None] + col[None, :]) % len(palette)]
        else:
            tile = rng.choice(palette, size=(p, q))
    phase_r, phase_c = int(rng.integers(p)), int(rng.integers(q))
    field = tile[(np.arange(shape[0])[:, None] + phase_r) % p,
                 (np.arange(shape[1])[None, :] + phase_c) % q]
    return field.astype(np.int8)


def _pick(rng, mask, count):
    choices = np.flatnonzero(mask)
    return rng.choice(choices, size=min(count, len(choices)), replace=False)


def _grow(rng, allowed, count, start):
    """A connected patch in an arbitrary allowed region, with bounded work."""
    result = np.zeros(allowed.shape, dtype=bool)
    result.flat[start] = True
    for _ in range(count - 1):
        near = np.zeros(allowed.shape, dtype=bool)
        near[1:] |= result[:-1]
        near[:-1] |= result[1:]
        near[:, 1:] |= result[:, :-1]
        near[:, :-1] |= result[:, 1:]
        available = near & allowed & ~result
        if not available.any():
            break
        result.flat[int(rng.choice(np.flatnonzero(available)))] = True
    return result


def _private_frame(rng, shape, common, interior, forbidden, target):
    h, w = shape
    sides = list(map(int, rng.permutation(np.arange(4, min(7, h, w) + 1))))
    for s in sides:
        positions = rng.permutation((h - s + 1) * (w - s + 1))
        for index in positions:
            r, c = divmod(int(index), w - s + 1)
            frame = (r, c, s)
            if frame == target:
                continue
            mask = _perimeter(shape, frame)
            if np.any(mask & (interior | forbidden)) or (mask & ~common).sum() < 4:
                continue
            return mask
    raise ValueError("no room for a private decoy frame")


def _damages(rng, shape, frame, dense, spur):
    r, c, s = frame
    common = _perimeter(shape, frame)
    box = np.zeros(shape, dtype=bool)
    box[r:r + s, c:c + s] = True
    interior = box & ~common
    # Four explicitly visible interior cases: both / first / second / neither.
    reserved = _pick(rng, interior, 4)
    allowed = interior.copy()
    allowed.flat[reserved[1:]] = False
    count = max(1, int(round(interior.sum() * (rng.uniform(.24, .38) if dense else rng.uniform(.07, .16)))))
    if spur or dense:
        common |= _grow(rng, allowed, count, int(reserved[0]))
    else:
        common.flat[reserved[0]] = True
        common.flat[_pick(rng, allowed & ~common, count - 1)] = True
    # Shared exterior faults make a bounding box of all overlaps incorrect.
    common.flat[_pick(rng, ~box, int(rng.integers(1, 5)))] = True
    if spur:
        extensions = []
        if r > 0:
            extensions.append([(r - k, c + s // 2) for k in range(1, min(3, r) + 1)])
        if r + s < shape[0]:
            extensions.append([(r + s - 1 + k, c + s // 2) for k in range(1, min(3, shape[0] - r - s) + 1)])
        if c > 0:
            extensions.append([(r + s // 2, c - k) for k in range(1, min(3, c) + 1)])
        if c + s < shape[1]:
            extensions.append([(r + s // 2, c + s - 1 + k) for k in range(1, min(3, shape[1] - c - s) + 1)])
        for point in extensions[int(rng.integers(len(extensions)))]:
            common[point] = True
    if _frames(common) != [frame]:
        raise ValueError("accidental common frame")
    private_a = _private_frame(rng, shape, common, interior, np.zeros(shape, bool), frame)
    private_b = _private_frame(rng, shape, common, interior, private_a & ~common, frame)
    da, db = common | private_a, common | private_b
    available = ~(common | private_a | private_b)
    available.flat[reserved] = False
    codes = rng.choice(3, size=shape, p=[.82, .09, .09])
    da |= available & (codes == 1)
    db |= available & (codes == 2)
    da.flat[reserved[1]] = True
    db.flat[reserved[2]] = True
    _require(np.array_equal(da & db, common), "construction changed common mask")
    return da, db, common


def _paint(rng, field, damage, palette, ink, reuse):
    result = field.copy()
    if reuse:
        # Cyclic nonzero offsets guarantee every painted cell really changes.
        indices = np.zeros(field.shape, dtype=int)
        for i, color in enumerate(palette):
            indices[field == color] = i
        offsets = rng.integers(1, len(palette), size=field.shape)
        replacement = palette[(indices + offsets) % len(palette)]
        result[damage] = replacement[damage]
    else:
        result[damage] = ink
    return result


def _make_pair(rng, axis, side, mode, dense, spur, edge, style):
    for _ in range(100):
        short, long = int(rng.integers(11, 15)), int(rng.integers(15, 31))
        shape = (long, short) if axis == 1 else (short, long)
        colors = rng.permutation(10).astype(np.int8)
        marker, ink_a, ink_b = map(int, colors[:3])
        palette = colors[3:3 + int(rng.integers(3, 6))]
        field = _field(rng, shape, palette, style)
        if np.unique(field).size < 2:
            continue
        palette = np.unique(field)
        r, c = int(rng.integers(shape[0] - side + 1)), int(rng.integers(shape[1] - side + 1))
        if edge:
            if rng.integers(2):
                r = 0 if rng.integers(2) else shape[0] - side
            else:
                c = 0 if rng.integers(2) else shape[1] - side
        frame = (r, c, side)
        try:
            da, db, common = _damages(rng, shape, frame, dense, spur)
            reuse_a = mode == "reused" or (mode == "mixed" and bool(rng.integers(2)))
            reuse_b = mode == "reused" or (mode == "mixed" and not reuse_a)
            a = _paint(rng, field, da, palette, ink_a, reuse_a)
            b = _paint(rng, field, db, palette, ink_b, reuse_b)
            recovered, _ = _recover(a, b)
            if not np.array_equal(recovered, field):
                continue
        except (ValueError, AssertionError):
            continue
        separator = np.full((shape[0], 1) if axis == 1 else (1, shape[1]), marker, np.int8)
        inp = np.concatenate((a, separator, b), axis=axis)
        out = np.where(common, marker, field)[r:r + side, c:c + side].astype(np.int8)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction could not find an identifiable pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    initial_axis = int(rng.integers(2))
    shift = int(rng.integers(2))  # larger aperture, or more clustered shared damage
    styles = ["random", "woven", "stripes", "outlines"]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            training = split == "train"
            axis = initial_axis if training and i == 0 else 1 - initial_axis if training and i == 1 else int(rng.integers(2))
            side = int(rng.integers(4, 6)) if training and i == 0 else int(rng.integers(6, 8)) if training and i == 1 else int(rng.integers(4, 8))
            if not training and shift == 0:
                side = int(rng.integers(8, 11))
            mode = "novel" if training and i == 0 else "reused" if training and i == 1 else str(rng.choice(["novel", "reused", "mixed"]))
            dense = not training and shift == 1
            spur = (training and i == 1) or (not training) or rng.random() < .35
            edge = bool(rng.random() < (.45 if training else .6))
            style = str(rng.choice(styles))
            for _ in range(20):
                pair = _make_pair(rng, axis, side, mode, dense, spur, edge, style)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics and the evidence that makes the composition learnable."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, train_modes, train_sides, train_axes = set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            axis, a, b, marker, field, period, da, db, common, frame = _analyze(inp)
            r, c, s = frame
            expected = np.where(common, marker, field)[r:r + s, c:c + s]
            _require(np.array_equal(out, expected), "output breaks consensus-crop rule")
            _require(out.shape == (s, s), "incorrect inclusive crop shape")
            _require(all(np.all(edge == marker) for edge in (out[0], out[-1], out[:, 0], out[:, -1])), "incomplete output frame")
            inside = np.s_[r + 1:r + s - 1, c + 1:c + s - 1]
            cases = set(zip(da[inside].flat, db[inside].flat))
            _require(cases == {(False, False), (False, True), (True, False), (True, True)}, "interior must expose all four truth cases")
            box = np.zeros(a.shape, bool)
            box[r:r + s, c:c + s] = True
            _require(np.any(common & ~box), "need exterior shared damage to distinguish square from bounding box")
            _require(any(f != frame for f in _frames(da)) and any(f != frame for f in _frames(db)), "both views need a private decoy square")
            # Ablations: neither single-view anomaly mask gives the correct crop.
            crop_field = field[r:r + s, c:c + s]
            for mask in (da, db, da | db, da ^ db):
                alternative = np.where(mask[r:r + s, c:c + s], marker, crop_field)
                _require(not np.array_equal(out, alternative), "a simpler Boolean rule gives the same answer")
            _require(not np.array_equal(out, a[r:r + s, c:c + s]) and not np.array_equal(out, b[r:r + s, c:c + s]), "raw crop shortcut survives")
            if split == "train":
                train_sides.add(s)
                train_axes.add(axis)
                base_colors = set(map(int, np.unique(field)))
                if all(set(map(int, np.unique(view[mask]))) <= base_colors for view, mask in ((a, da), (b, db))):
                    train_modes.add("reused")
                if all(not (set(map(int, np.unique(view[mask]))) & base_colors) for view, mask in ((a, da), (b, db))):
                    train_modes.add("novel")
    _require(train_modes >= {"novel", "reused"}, "training must demonstrate both kinds of damage")
    _require(len(train_sides) >= 2 and train_axes == {0, 1}, "training lacks size or orientation evidence")


def _fingerprint(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded 42-seed diversity audit, counterfactuals, and count checks.

    Returns descriptive measurements for convenient recording. They are not an
    independent assessment of whether a human would infer the intended rule.
    """
    shapes, periods, sides, palettes, fingerprints, layouts = set(), set(), set(), set(), set(), set()
    normalized = set()
    edge_count = natural_outline_count = differing_shared_colors = 0
    scale_shifts = cluster_shifts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        scale_shifts += puzzle["test"][0]["output"].shape[0] > max(p["output"].shape[0] for p in puzzle["train"])
        cluster_shifts += puzzle["test"][0]["output"].shape[0] <= 7
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            axis, a, b, marker, field, period, da, db, common, frame = _analyze(inp)
            r, c, s = frame
            shapes.add(inp.shape)
            periods.add(period)
            sides.add(s)
            palettes.add(tuple(map(int, np.unique(inp))))
            layouts.add((a.shape, frame, axis))
            roles = {}
            normalized.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.flat)))
            edge_count += r == 0 or c == 0 or r + s == a.shape[0] or c + s == a.shape[1]
            natural_outline_count += any(_frames(field == color) for color in np.unique(field))
            differing_shared_colors += bool(np.any(common & (a != b)))
        # Swapping the two views and transposing the entire scene preserve rule.
        pair = puzzle["train"][seed % 2]
        inp, out = pair["input"], pair["output"]
        axis, a, b, marker = _split(inp)
        line = np.full((a.shape[0], 1) if axis == 1 else (1, a.shape[1]), marker, np.int8)
        swapped = np.concatenate((b, line, a), axis=axis)
        _require(np.array_equal(solve(swapped), out), "view-swap equivariance failed")
        _require(np.array_equal(solve(inp.T), out.T), "transpose equivariance failed")
        if seed < 3:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic puzzle")
    _require(len(fingerprints) == 42 and len(normalized) >= 190, "insufficient structural variation")
    _require(len(shapes) >= 30 and len(layouts) >= 190 and len(periods) >= 8, "insufficient geometry or wallpaper variation")
    _require(len(sides) >= 7 and len(palettes) >= 40, "insufficient size or palette variation")
    _require(edge_count >= 40 and natural_outline_count >= 10, "boundary or scaffold variation missing")
    _require(differing_shared_colors >= 100 and scale_shifts >= 8 and cluster_shifts >= 8, "damage/shift diversity missing")
    for nt, nq in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(900 + nt * 10 + nq, nt, nq)
        _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts ignored")
    # Valid queries beyond the sampling schedule: no private decoys are needed
    # by the rule, and square sides 11/14 are not restricted by solve.
    for side in (4, 11, 14):
        rr, cc = np.indices((14, 15))
        field = ((rr % 2 + cc % 3) % 3).astype(np.int8)
        frame = (0, 0, side)
        common = _perimeter(field.shape, frame)
        common[1, 1] = True
        da, db = common.copy(), common.copy()
        da[1, 2] = True
        db[2, 1] = True
        a, b = field.copy(), field.copy()
        a[da], b[db] = 8, 9
        inp = np.concatenate((a, np.full((1, 15), 7, np.int8), b), axis=0)
        expected = np.where(common, 7, field)[:side, :side]
        _require(np.array_equal(solve(inp), expected), "solver imposed a generation-only size/decoy restriction")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "layouts": len(layouts),
            "wallpaper_periods": sorted(periods), "frame_sides": sorted(sides),
            "palettes": len(palettes), "normalized_inputs": len(normalized),
            "edge_frames": int(edge_count), "natural_outline_fields": int(natural_outline_count),
            "pairs_with_unequal_shared_scar_colors": int(differing_shared_colors),
            "larger_frame_shifts": int(scale_shifts), "cluster_shifts": int(cluster_shifts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
