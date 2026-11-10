"""Bottom-anchored object shear controls rail-facing cavity projection.

The signal is always color 2. All other roles are visible in the input.
No files, parent modules, sampled parameters, or training labels are read by solve.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


SIGNAL = 2


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    """Eight-neighbor objects; four-neighbor empty regions."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in offsets:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                        and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(np.asarray(cells, dtype=int))
    return result


def _crop(mask):
    r, c = np.nonzero(mask)
    return mask[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _mask(cells):
    low = cells.min(axis=0)
    local = cells - low
    mask = np.zeros(tuple(local.max(axis=0) + 1), dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    return mask, low


def _profile(mask):
    h, w = mask.shape
    active, decoy = [], []
    enclosed = set()
    for empty in _components(~mask):
        if np.any((empty[:, 0] == 0) | (empty[:, 0] == h - 1)
                  | (empty[:, 1] == 0) | (empty[:, 1] == w - 1)):
            continue
        enclosed.update(map(tuple, empty))
        contact = (empty[:, 1] == w - 2) & mask[empty[:, 0], w - 1]
        (active if np.any(contact) else decoy).append(empty)
    active_rows = {int(r) for hole in active for r in hole[:, 0]}
    contact_rows = {int(r) for hole in active for r, c in hole
                    if c == w - 2 and mask[r, w - 1]}
    naive_rows = ({r for r in range(h) if not mask[r, w - 2] and mask[r, w - 1]}
                  if w >= 2 else set())
    open_rows = {r for r in naive_rows if (r, w - 2) not in enclosed}
    return dict(active=active, decoy=decoy, rows=active_rows,
                contact_rows=contact_rows, naive_rows=naive_rows,
                open_rows=open_rows)


def _scene(grid):
    bg = int(grid[0, -1])
    candidates = []
    h, w = grid.shape
    for k in range(1, w - 1):
        color = int(grid[0, k])
        if (color != bg and color != SIGNAL and np.all(grid[:, k] == color)
                and np.count_nonzero(grid == color) == h
                and np.all(grid[:, k + 1:] == bg)
                and np.any(grid[:, :k] != bg)):
            candidates.append((k, color))
    if len(candidates) != 1 or bg == SIGNAL or np.any(grid == SIGNAL):
        raise ValueError("input needs one exclusive vertical rail and a blank right band; 2 is reserved")
    k, rail = candidates[0]
    return bg, rail, k


def _details(grid, shear=True):
    bg, rail, k = _scene(grid)
    result = []
    for cells in _components(grid[:, :k] != bg, diagonal=True):
        r, c = cells.T
        bottom = int(r.max())
        displaced = c + (bottom - r if shear else 0)
        extreme = int(displaced.max())
        shift = k - 1 - extreme
        if shift < 0:
            raise ValueError("sheared object would cross the rail")
        sheared = np.column_stack((r, displaced))
        mask, low = _mask(sheared)
        profile = _profile(mask)
        result.append(dict(cells=cells, mask=mask, low=low, shift=shift,
                           bottom=bottom, extreme=extreme, displaced=displaced,
                           profile=profile))
    return bg, rail, k, result


def _transform(input_grid, *, shear=True, projection="cavity", snap=True):
    grid = _grid(input_grid)
    bg, rail, k, objects = _details(grid, shear=shear)
    out = np.full_like(grid, bg)
    out[:, k] = rail
    for obj in objects:
        r, c = obj["cells"].T
        target = obj["displaced"] + (obj["shift"] if snap else 0)
        out[r, target] = grid[r, c]
        p = obj["profile"]
        if projection == "cavity":
            rows = p["rows"]
        elif projection == "contact":
            rows = p["contact_rows"]
        elif projection == "column":
            rows = p["naive_rows"]
        elif projection == "all_holes":
            rows = {int(y) for hole in p["active"] + p["decoy"] for y in hole[:, 0]}
        elif projection == "none":
            rows = set()
        else:
            raise ValueError("unknown diagnostic projection")
        for y in rows:
            out[int(obj["low"][0]) + y, k + 1:] = SIGNAL
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Shear each input component about its bottom, snap, then project holes.

    If any part of an enclosed pocket is just behind the object's global
    rightmost wall, all rows of that pocket signal. Extreme ties are a union.
    """
    return _transform(input_grid)


def _inverse_shear(mask):
    """A sampler building block, never used by solve: create a skewed source."""
    r, c = np.nonzero(mask)
    c = c - (mask.shape[0] - 1 - r)
    c -= c.min()
    source = np.zeros((mask.shape[0], int(c.max()) + 1), dtype=bool)
    source[r, c] = True
    return _crop(source)


def _intervals(rng, h, count):
    spare = h - 2 * count - 1
    _require(spare >= 0, "too many holes for the height")
    extra = rng.multinomial(spare, np.full(2 * count + 1, 1 / (2 * count + 1)))
    row = 1 + int(extra[0])
    result = []
    for i in range(count):
        stop = row + 1 + int(extra[2 * i + 1])
        result.append((row, stop))
        row = stop + 1 + int(extra[2 * i + 2])
    return result


def _object(rng, kind, *, tall=False):
    """Sample geometry, not a color-coded category. All returns are sources."""
    if kind == "solid":
        h, w = int(rng.integers(2, 8)), int(rng.integers(2, 10))
        target = np.zeros((h, w), dtype=bool)
        spine = int(rng.integers(w))
        for r in range(h):
            left = int(rng.integers(spine + 1))
            right = int(rng.integers(spine, w))
            target[r, left:right + 1] = True
        return _inverse_shear(_crop(target))
    if kind == "partial":
        h = int(rng.integers(8, 12) if tall else rng.integers(5, 8))
        w = int(rng.integers(6, 11))
        target = np.ones((h, w), dtype=bool)
        start = int(rng.integers(1, h - 2))
        stop = int(rng.integers(start + 2, h))
        target[start:stop, 2:w - 2] = False
        # Only one of several cavity rows reaches the extreme wall.
        contact = int(rng.integers(start, stop))
        target[contact, 2:w - 1] = False
        return _inverse_shear(target)
    h = int(rng.integers(8, 12) if tall else rng.integers(4, 8))
    w = int(rng.integers(4, 10))
    target = np.ones((h, w), dtype=bool)
    if kind == "mixed":
        h = max(h, 5)
        w = max(w, 6)
        target = np.ones((h, w), dtype=bool)
        count = int(rng.integers(2, min(4, (h - 1) // 2) + 1))
    elif kind == "active":
        count = int(rng.integers(1, min(3, (h - 1) // 2) + 1))
    else:
        count = 1
    intervals = _intervals(rng, h, count)
    recessed_index = int(rng.integers(count)) if kind == "mixed" else -1
    for i, (a, b) in enumerate(intervals):
        right = w - 3 if i == recessed_index else w - 2
        left = int(rng.integers(1, max(2, right)))
        for r in range(a, b):
            # A varying back edge yields nonrectangular, connected cavities.
            back = int(rng.integers(1, right)) if rng.random() < .35 else left
            target[r, back:right + 1] = False
    if kind == "open":
        a, b = intervals[0]
        side = int(rng.integers(3))
        if side == 0:
            target[a:b, :w - 1] = False
        elif side == 1:
            target[:a + 1, w - 2] = False
        else:
            target[b - 1:, w - 2] = False
    elif kind == "recessed":
        extension = int(rng.integers(1, 4))
        bigger = np.zeros((h, w + extension), dtype=bool)
        bigger[:, :w] = target
        bigger[int(rng.choice([0, h - 1])), w:] = True
        target = bigger
    elif kind == "deactivate":
        # An ordinary unsheared frame loses its leading status after shear.
        return target
    return _inverse_shear(target)


def _extent(mask):
    r, c = np.nonzero(mask)
    return int(np.max(c + mask.shape[0] - 1 - r)) + 1


def _scene_pair(rng, required, roles, multicolor, is_test, shift_mode):
    masks = [m.copy() for m in required]
    if is_test:
        masks.append(_object(rng, "mixed" if shift_mode == "rooms" else "partial", tall=True))
    target_count = int(rng.integers(max(1, len(masks)), max(2, len(masks) + 2) + 1))
    target_count = min(target_count, 5)
    pool = ("active", "open", "recessed", "deactivate", "solid", "mixed", "partial")
    # Bounded additions; no retained candidates or retry logs.
    for _ in range(8):
        if len(masks) >= target_count:
            break
        candidate = _object(rng, str(rng.choice(pool)))
        occupied = sum(m.shape[0] for m in masks) + len(masks)
        if occupied + candidate.shape[0] + 1 <= 29:
            masks.append(candidate)
    if not masks:
        masks.append(_object(rng, "active"))
    rng.shuffle(masks)
    base = sum(m.shape[0] for m in masks) + len(masks) - 1
    height = int(rng.integers(base + 1, min(30, base + 7) + 1))
    extent = max(_extent(m) for m in masks)
    _require(extent <= 28, "object leaves no rail or projection band")
    width = int(rng.integers(extent + 2, 31))
    k = int(rng.integers(extent, width - 1))
    bg, rail, fg = roles
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[:, k] = rail
    extras = rng.multinomial(height - base, np.full(len(masks) + 1, 1 / (len(masks) + 1)))
    row = int(extras[0])
    colors = [c for c in range(10) if c not in (bg, rail, SIGNAL)]
    for i, mask in enumerate(masks):
        _require(len(_components(mask, diagonal=True)) == 1, "source object split")
        margin = k - _extent(mask)
        # Zero snap, equal snaps, and large snaps are all possible.
        left = margin if rng.random() < .25 else int(rng.integers(margin + 1))
        rr, cc = np.nonzero(mask)
        payload = np.full(len(rr), fg, dtype=np.int8)
        if multicolor:
            palette = rng.choice(colors, int(rng.integers(1, 4)), replace=False)
            style = int(rng.integers(3))
            index = (rr if style == 0 else cc if style == 1 else rr + cc) % len(palette)
            payload = palette[index]
        grid[row + rr, left + cc] = payload
        row += mask.shape[0] + 1 + int(extras[i + 1])
    return {"input": grid, "output": solve(grid)}


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                    (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice([c for c in range(10) if c != SIGNAL], 3, replace=False)))
    vary_roles = bool(rng.random() < .5)
    multicolor = bool(rng.random() < .45)
    shift_mode = str(rng.choice(("rooms", "partial")))
    requirements = [[] for _ in range(a)]
    order = list(map(int, rng.permutation(a)))
    # These contrasts are distributed, not assigned semantic pair positions.
    kinds = ["active", "open", "recessed", "partial", "deactivate", "mixed"]
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        requirements[order[i % a]].append(_object(rng, kind))
    seen = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            required = requirements[i] if split == "train" else []
            for attempt in range(32):
                pair_roles = (tuple(map(int, rng.choice([c for c in range(10) if c != SIGNAL],
                                                       3, replace=False))) if vary_roles else roles)
                pair = _scene_pair(rng, required, pair_roles, multicolor, split == "test", shift_mode)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    evidence = Counter()
    train_heights, test_heights = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            bg, rail, k, objects = _details(inp)
            _require(np.array_equal(out, solve(inp)), "incorrect output")
            _require(np.all(out[:, k] == rail), "rail changed")
            previous = -2
            for obj in objects:
                r, c = obj["cells"].T
                _require(int(r.min()) > previous + 1, "object row bands overlap")
                previous = int(r.max())
                p = obj["profile"]
                _require(len(_components(obj["mask"], diagonal=True)) == 1, "sheared object split")
                target = c + obj["bottom"] - r + obj["shift"]
                _require(int(target.min()) >= 0 and int(target.max()) == k - 1, "bad rail contact")
                _require(np.array_equal(out[r, target], inp[r, c]), "payload changed")
                # Each occupied row keeps its ordered colors and horizontal vectors.
                for row in np.unique(r):
                    old_columns = np.sort(c[r == row])
                    new_columns = old_columns + obj["bottom"] - row + obj["shift"]
                    _require(np.array_equal(np.diff(old_columns), np.diff(new_columns)), "row distorted")
                    _require(np.array_equal(inp[row, old_columns], out[row, new_columns]), "row recolored")
                (train_heights if split == "train" else test_heights).append(obj["mask"].shape[0])
                if split == "train":
                    evidence["active"] += bool(p["active"])
                    evidence["open"] += bool(p["open_rows"])
                    evidence["recessed"] += bool(p["decoy"])
                    evidence["partial"] += p["rows"] != p["contact_rows"]
            for color in range(10):
                if color not in (bg, rail, SIGNAL):
                    _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color), "payload lost")
            if split == "train":
                band = slice(k + 1, None)
                evidence["no_shear"] += not np.array_equal(_transform(inp, shear=False)[:, band], out[:, band])
                evidence["one_column"] += not np.array_equal(_transform(inp, projection="column"), out)
                evidence["all_holes"] += not np.array_equal(_transform(inp, projection="all_holes"), out)
                evidence["no_rail"] += not np.array_equal(_transform(inp, projection="none", snap=False), out)
    _require(all(evidence[c] for c in ("active", "open", "recessed", "partial", "no_shear",
                                      "one_column", "all_holes", "no_rail")), "teaching contrasts missing")
    _require(max(test_heights) > max(train_heights), "missing taller-object test shift")
    return None


def self_test():
    """Bounded, in-memory 42-seed geometry audit; no sampled grids are saved."""
    # Small literal regressions check the composition without using the sampler.
    source = np.zeros((5, 12), dtype=np.int8)
    source[:, 8] = 9
    for row, digits in enumerate(("11100", "01010", "00111")):
        source[row, 1:6] = [int(c) for c in digits]
    expected = np.zeros_like(source)
    expected[:, 8] = 9
    expected[0, 5:8] = expected[2, 5:8] = 1
    expected[1, [5, 7]] = 1
    expected[1, 9:] = SIGNAL
    _require(np.array_equal(solve(source), expected), "literal activating-frame regression")
    direct = np.zeros_like(source)
    direct[:, 8] = 9
    direct[:3, 1:4] = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    expected = np.zeros_like(source)
    expected[:, 8] = 9
    expected[0, 5:8] = expected[2, 3:6] = 1
    expected[1, [4, 6]] = 1
    _require(np.array_equal(solve(direct), expected), "literal deactivating-frame regression")
    source = np.zeros((7, 16), dtype=np.int8)
    source[:, 12] = 9
    for row, digits in enumerate(("1111110000", "0110001000", "0011001100",
                                   "0001100110", "0000111111")):
        source[row, :10] = [int(c) for c in digits]
    expected = np.zeros_like(source)
    expected[:, 12] = 9
    expected[:5, 6:12] = [[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 1],
                           [1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]]
    expected[1:4, 13:] = SIGNAL
    _require(np.array_equal(solve(source), expected), "literal whole-pocket regression")
    shapes, masks, counts, palettes, signatures = set(), set(), set(), set(), set()
    widths, heights, bands, gaps, snaps = set(), set(), set(), set(), set()
    stats = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, rail, k, objects = _details(inp)
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                shapes.add(inp.shape)
                widths.add(inp.shape[1])
                heights.add(inp.shape[0])
                bands.add(inp.shape[1] - k - 1)
                counts.add(len(objects))
                palettes.add((bg, rail, tuple(map(int, np.unique(inp)))))
                stats["multicolor"] += len(np.unique(inp)) > 3
                for first, second in zip(objects, objects[1:]):
                    gaps.add(int(second["cells"][:, 0].min() - first["bottom"] - 1))
                for obj in objects:
                    p = obj["profile"]
                    masks.add(_signature(obj["mask"]))
                    snaps.add(obj["shift"])
                    stats["active"] += bool(p["active"])
                    stats["open"] += bool(p["open_rows"])
                    stats["decoy"] += bool(p["decoy"])
                    stats["partial"] += p["rows"] != p["contact_rows"]
                    stats["multi"] += len(p["active"]) + len(p["decoy"]) >= 2
                    stats["silent"] += not p["rows"]
                _require(np.array_equal(solve(inp.copy().tolist()), out), "list/copy solve disagreement")
        signatures.add(digest.digest())
    _require(len(signatures) == 42, "puzzle diversity collapsed")
    _require(len(shapes) >= 60 and len(masks) >= 150 and len(palettes) >= 40, "structural diversity collapsed")
    _require({1, 2, 3, 4} <= counts and 1 in bands and 30 in widths and 30 in heights, "boundary/count coverage")
    _require(0 in snaps and max(snaps) >= 6 and len(gaps) >= 3, "placement diversity collapsed")
    _require(all(stats[key] >= 20 for key in ("active", "open", "decoy", "partial", "multi", "silent")),
             "geometry categories collapsed")
    _require(stats["multicolor"] >= 20, "payload palette variation collapsed")
    first, second = generate_puzzle(614290281), generate_puzzle(614290281)
    _require(all(np.array_equal(x[k], y[k]) for s in ("train", "test")
                 for x, y in zip(first[s], second[s]) for k in ("input", "output")), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, -1] = SIGNAL
    # Choose a definite wrong label even when that row already signals.
    original = int(first["test"][0]["output"][0, -1])
    broken["test"][0]["output"][0, -1] = (original + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_shapes": len(shapes),
            "distinct_object_masks": len(masks), "object_counts": sorted(counts),
            "projection_widths": sorted(bands), "snap_distances": sorted(snaps),
            "palette_profiles": len(palettes), "geometry_counts": dict(stats)}
