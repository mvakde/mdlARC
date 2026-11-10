"""Diagonal aperture scanner. Standard library and NumPy only.

No parent imports, stored puzzles, seed-dependent solution rules, or I/O.
All experiments are bounded and in memory; self_test writes no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        neighbors += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for dr, dc in neighbors:
                x, y = a + dr, b + dc
                if (0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]
                        and mask[x, y] and not seen[x, y]):
                    seen[x, y] = True
                    stack.append((x, y))
        result.append(np.asarray(cells, dtype=int))
    return result


def _crop(cells):
    low, high = cells.min(axis=0), cells.max(axis=0) + 1
    mask = np.zeros(tuple(high - low), dtype=bool)
    mask[tuple((cells - low).T)] = True
    return mask, low


def _holes(mask, diagonal=False):
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [cells - 1 for cells in _components(~padded, diagonal)
            if np.all(cells > 0) and np.all(cells < np.array(padded.shape) - 1)]


def _valid_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid


def _strokes(mask):
    cells = np.argwhere(mask)
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(int(c - slope * r), []).append(int(r))
        strokes = []
        for q, rows in sorted(lines.items()):
            rows.sort()
            runs = []
            for r in rows:
                if not runs or r != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                if len(run) < 2:
                    break
                strokes.append((q, 2 * run[0] + slope * q,
                                2 * run[-1] + slope * q))
            else:
                continue
            break
        else:
            if len(strokes) >= 2:
                candidates.append((slope, strokes))
    if len(candidates) != 1:
        raise ValueError("stroke direction must be unique")
    return candidates[0]


def _envelope(shape, slope, strokes, mode="normal"):
    rr, cc = np.indices(shape)
    q, t = cc - slope * rr, rr + slope * cc
    envelope = np.zeros(shape, dtype=bool)
    for level in range(int(t.min()), int(t.max()) + 1):
        active = [x for x, lo, hi in strokes if lo <= level <= hi
                  and (mode != "pixel_centers" or (level - slope * x) % 2 == 0)]
        if len(active) < 2:
            continue
        if mode == "consecutive":
            ordered = sorted(strokes)
            for a, b in zip(ordered, ordered[1:]):
                if max(a[1], b[1]) <= level <= min(a[2], b[2]):
                    envelope |= (t == level) & (q > a[0]) & (q < b[0])
        else:
            envelope |= (t == level) & (q > min(active)) & (q < max(active))
    return envelope, t


def _parse(input_grid):
    grid = _valid_grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must be a strict majority")
    groups = {int(c): _components(grid == c) for c in colors if c != bg}
    rail_colors = [c for c, comps in groups.items() if len(comps) >= 4
                   and all(len(x) == 1 for x in comps)]
    if len(rail_colors) != 1:
        raise ValueError("expected one isolated diagonal-stroke color")
    rail = rail_colors[0]
    slope, strokes = _strokes(grid == rail)
    neutral_colors = [c for c, comps in groups.items() if c != rail and len(comps) >= 2
                      and all(len(x) >= 3 for x in comps)]
    if len(neutral_colors) != 1:
        raise ValueError("expected one repeated neutral-island color")
    neutral = neutral_colors[0]
    mapping, entries, objects, holes = {}, [], [], []
    protected = grid != bg
    for color, comps in groups.items():
        if color == rail:
            continue
        if color != neutral and len(comps) != 1:
            raise ValueError("each ink color must have one prototype")
        for cells in comps:
            mask, low = _crop(cells)
            cavities = [x + low for x in _holes(mask)]
            for cavity in cavities:
                if np.any(grid[tuple(cavity.T)] != bg):
                    raise ValueError("windows must contain only background")
                protected[tuple(cavity.T)] = True
            if color == neutral:
                objects.append((cells, mask, low, cavities))
                holes.extend(cavities)
            else:
                k = len(cavities)
                if k in mapping:
                    raise ValueError("ink prototypes must have different cavity counts")
                mapping[k] = color
                entries.append((tuple(low), k, color))
    if not mapping:
        raise ValueError("missing topological ink prototypes")
    envelope, t = _envelope(grid.shape, slope, strokes)
    for _, _, color in entries:
        if np.any(envelope & (grid == color)):
            raise ValueError("ink prototypes must lie outside the scan band")
    return dict(grid=grid, bg=bg, rail=rail, neutral=neutral, slope=slope,
                strokes=strokes, mapping=mapping, entries=entries,
                objects=objects, holes=holes, protected=protected,
                envelope=envelope, t=t)


def _paint(info, alternative=""):
    grid, bg = info["grid"], info["bg"]
    env, t = info["envelope"], info["t"]
    if alternative in ("pixel_centers", "consecutive"):
        env, t = _envelope(grid.shape, info["slope"], info["strokes"], alternative)
    if alternative == "omit_A":
        env = np.ones(grid.shape, dtype=bool)
    candidates = info["holes"]
    if alternative == "objects":
        candidates = [x[0] for x in info["objects"]]
    elif alternative == "open_bays":
        candidates = []
        for _, mask, low, _ in info["objects"]:
            candidates.extend(x + low for x in _components(~mask))
    elif alternative == "eight_connected":
        candidates = [x + low for _, mask, low, _ in info["objects"]
                      for x in _holes(mask, True)]
    census = np.zeros(grid.shape, dtype=int)
    for cells in candidates:
        reached = env[tuple(cells.T)]
        if alternative == "ungated_windows":
            reached = np.ones(len(cells), dtype=bool)
        levels = t[tuple(cells[reached].T)] if reached.any() else []
        if alternative != "hole_pixels":
            levels = np.unique(levels)
        for level in levels:
            census[t == level] += 1
    if alternative == "omit_B":
        census[:] = 0
    elif alternative == "total_holes":
        census[:] = len(info["holes"])
    elif alternative == "cap_two":
        census = np.minimum(census, 2)
    mapping = info["mapping"]
    if alternative == "entry_rank":
        mapping = {i: e[2] for i, e in enumerate(sorted(info["entries"]))}
    eligible = env & ~info["protected"]
    if alternative == "fill_windows":
        eligible = env & (grid == bg)
    output = grid.copy()
    for k, color in mapping.items():
        output[eligible & (census == k)] = color
    return output, census, eligible


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, windows, slice census, and colors from this input only."""
    return _paint(_parse(input_grid))[0]


def _shape(k, rng, rich=False, prototype=False):
    if prototype:
        mask = np.ones((3, max(3, 2 * k + 1)), dtype=bool)
        mask[1, 1:2 * k:2] = False
        if rng.random() < .35:
            mask = mask.T.copy()
        return mask
    if k == 0:
        h, w = (int(rng.integers(3, 6)) for _ in range(2))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = mask[-1, :] = mask[:, -1] = True
        if rng.random() < .25:
            mask[:, -1] = False
        return np.rot90(mask, int(rng.integers(4))).copy()
    nr, nc = (1, 1) if k == 1 else (2, 2)
    maxsize = 3 if rich else 2
    hs = [int(rng.integers(1, maxsize + 1)) for _ in range(nr)]
    ws = [int(rng.integers(1, maxsize + 1)) for _ in range(nc)]
    h, w = sum(hs) + nr + 1, sum(ws) + nc + 1
    mask = np.ones((h, w), dtype=bool)
    slots = [(0, 0)] if k == 1 else ([(0, 0), (1, 1)] if k == 2
                                    else [(0, 0), (0, 1), (1, 1)])
    for a, b in slots[:k]:
        r, c = 1 + sum(hs[:a]) + a, 1 + sum(ws[:b]) + b
        cavity = np.ones((hs[a], ws[b]), dtype=bool)
        if rich and min(cavity.shape) >= 2 and rng.random() < .7:
            cavity[0, -1] = False
        mask[r:r + hs[a], c:c + ws[b]][cavity] = False
    for _ in range(4 if rich else 2):
        r = int(rng.choice([0, h - 1]))
        c = int(rng.choice([0, w - 1]))
        candidate = mask.copy()
        candidate[r, c] = False
        if len(_components(candidate)) == 1 and len(_holes(candidate)) == k:
            mask = candidate
    return np.rot90(mask, int(rng.integers(4))).copy()


def _rail_grid(rng, witness):
    gap = int(rng.integers(14, 25))
    n = int(rng.integers(3, 5) if witness else rng.integers(2, 6))
    n = min(n, gap // 4 + 1)
    interior = sorted(int(x) for x in rng.choice(np.arange(4, gap - 3),
                                                size=n - 2, replace=False))
    qs = [0] + interior + [gap]
    if any(b - a < 3 for a, b in zip(qs, qs[1:])):
        return None
    span = int(rng.integers(18, 29))
    specs = []
    for i, q in enumerate(qs):
        lo = -span // 2 + int(rng.integers(-2, 3))
        hi = span // 2 + int(rng.integers(-2, 3))
        if i not in (0, n - 1) and (witness or rng.random() < .7):
            lo = int(rng.integers(-5, 4))
            hi = lo + 2 * int(rng.integers(1, 4))
        lo += (q - lo) % 2
        hi -= (hi - q) % 2
        specs.append((q, lo, hi))
    cells = [(int((q + t) // 2), int((q - t) // 2))
             for q, lo, hi in specs for t in range(lo, hi + 1, 2)]
    cells = np.asarray(cells, dtype=int)
    cells -= cells.min(axis=0)
    span = cells.max(axis=0) + 1
    if max(span) > 28:
        return None
    shape = tuple(int(rng.integers(max(23, int(s)), 31)) for s in span)
    offset = np.array([int(rng.integers(size - s + 1)) for size, s in zip(shape, span)])
    grid = np.zeros(shape, dtype=np.int8)
    grid[tuple((cells + offset).T)] = 1
    try:
        slope, strokes = _strokes(grid == 1)
    except ValueError:
        return None
    env, _ = _envelope(shape, slope, strokes)
    return grid, env


def _place(grid, free, mask, color, rng, region, kind):
    h, w = mask.shape
    holes = _holes(mask)
    tr, tc = np.indices(grid.shape)
    longitudinal = tr - tc
    band_levels = set(int(x) for x in longitudinal[region])
    positions = []
    for r in range(grid.shape[0] - h + 1):
        for c in range(grid.shape[1] - w + 1):
            if not free[r:r + h, c:c + w].all():
                continue
            patch = region[r:r + h, c:c + w]
            if kind in ("prototype", "off_band"):
                if patch.any():
                    continue
                if kind == "off_band" and not any(
                        int(r + a - c - b) in band_levels for hole in holes for a, b in hole):
                    continue
            else:
                if np.sum(patch & mask) < min(3, int(mask.sum())):
                    continue
                if holes and not any(patch[tuple(x.T)].any() for x in holes):
                    continue
            positions.append((r, c))
    if not positions:
        return False
    r, c = positions[int(rng.integers(len(positions)))]
    grid[r:r + h, c:c + w][mask] = color
    free[max(0, r - 1):min(grid.shape[0], r + h + 1),
         max(0, c - 1):min(grid.shape[1], c + w + 1)] = False
    return True


_ALTERNATIVES = ("omit_A", "omit_B", "total_holes", "objects", "open_bays",
                 "pixel_centers", "consecutive", "entry_rank", "fill_windows",
                 "ungated_windows", "eight_connected", "cap_two", "hole_pixels")


def _witnesses(info, output):
    return {name for name in _ALTERNATIVES
            if not np.array_equal(_paint(info, name)[0], output)}


def _fallback_pair(split, index, roles, variant=0):
    """Guaranteed-fitting teaching scene; the same rule handles every pixel."""
    grid = np.zeros((30, 30), dtype=np.int8)
    for q, lo, hi in ((12, -12, 12), (21, -5, -3), (42, -12, 12)):
        for t in range(lo, hi + 1, 2):
            grid[(q + t) // 2, (q - t) // 2] = 1
    grid[10:17, 10:17] = 2
    for r, c in ((10, 10), (11, 11), (12, 11), (13, 11), (12, 12), (13, 12),
                 (15, 14), (15, 15), (16, 16)):
        grid[r, c] = 0
    if split == "test":
        grid[11, 15] = 0  # third neutral window; no new rule
    grid[2:5, 1:4] = 2
    grid[3, 2] = 0  # off-band neutral window, projected across an active count-2 slice
    grid[18:21, 18:21] = 2
    grid[19, 19] = grid[18, 18] = 0  # third simultaneous window, diagonal escape
    grid[5:9, 15] = grid[8, 15:19] = grid[5:9, 18] = 2
    grid[0:2, 23:26] = 3
    k = 1 if split == "train" and index == 0 else 2
    grid[26:29, 0:2 * k + 1] = 3 + k
    grid[27, 1:2 * k:2] = 0
    shift = variant // 8
    if shift:
        shifted = np.zeros_like(grid)
        shifted[:, shift:] = grid[:, :-shift]
        grid = shifted
    if variant % 8 >= 4:
        grid = grid[:, ::-1]
    grid = np.rot90(grid, variant % 4).copy()
    grid = np.asarray(roles, dtype=np.int8)[grid]
    info = _parse(grid)
    output = _paint(info)[0]
    return dict(input=grid, output=output), _witnesses(info, output)


def _sample_pair(rng, split, index, roles, required):
    for _ in range(64):
        result = _rail_grid(rng, witness=split == "train" and index < 2)
        if result is None:
            continue
        grid, env = result
        keys = ([0, 1] if index == 0 else [2, 0]) if split == "train" and index < 2 else None
        if keys is None:
            other = rng.choice(np.arange(1, 5), size=int(rng.integers(1, 4)), replace=False)
            keys = [0] + [int(k) for k in other]
        rng.shuffle(keys)
        free = grid == 0
        prototypes = [(k, _shape(k, rng, prototype=True)) for k in keys]
        prototypes.sort(key=lambda x: -x[1].size)
        if not all(_place(grid, free, mask, 3 + k, rng, env, "prototype")
                   for k, mask in prototypes):
            continue
        counts = [2, 1, 0] if split == "train" else [3] + [
            int(rng.integers(0, 4)) for _ in range(int(rng.integers(1, 5)))]
        if split == "train" and index == 1 and "cap_two" in required:
            counts.append(1)
        if index >= 2 and split == "train":
            counts = [int(rng.integers(1, 3))] + [
                int(rng.integers(0, 3)) for _ in range(int(rng.integers(1, 5)))]
        shapes = [(k, _shape(k, rng, rich=split == "test" or rng.random() < .3)) for k in counts]
        shapes.sort(key=lambda x: -x[1].size)
        off_band_used = False
        packed = True
        for k, mask in shapes:
            outside = k == 1 and "ungated_windows" in required and not off_band_used
            kind = "off_band" if outside else "query"
            if not _place(grid, free, mask, 2, rng, env, kind):
                packed = False
                break
            off_band_used |= outside
        if not packed:
            continue
        # Rotate/reflect the complete scene. Color literals are independent of k.
        grid = np.rot90(grid, int(rng.integers(4))).copy()
        if rng.random() < .5:
            grid = grid[:, ::-1].copy()
        palette = np.asarray(roles, dtype=np.int8)
        grid = palette[grid]
        try:
            info = _parse(grid)
        except ValueError:
            continue
        output, census, eligible = _paint(info)
        if np.array_equal(grid, output):
            continue
        if split == "train" and index < 2:
            # Teach two visible ink classes, simultaneous windows, and an
            # unlisted census with exposed exterior pixels in just two pairs.
            visible_counts = set(int(x) for x in census[eligible])
            if not {0, 1, 2} <= visible_counts:
                continue
        found = _witnesses(info, output)
        if not required <= found:
            continue
        return dict(input=grid, output=output), found
    pair, found = _fallback_pair(split, index, roles, int(rng.integers(24)))
    assert required <= found, ("fallback lacks teaching witness", sorted(required - found))
    return pair, found


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    base_roles = rng.permutation(10)
    puzzle, seen, needed, taught = {"train": [], "test": []}, set(), set(), set()
    for split, count in (("test", num_test), ("train", num_train)):
        for i in range(count):
            roles = base_roles if stable else rng.permutation(10)
            required = set()
            if split == "train" and i == 0:
                required = {"omit_A", "omit_B", "total_holes", "objects", "open_bays",
                            "pixel_centers", "consecutive"}
            if split == "train" and i == 1:
                required = (needed - taught) | {"entry_rank", "fill_windows"}
            for _ in range(8):
                pair, found = _sample_pair(rng, split, i, roles, required)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "test":
                        needed |= found
                    else:
                        taught |= found
                    break
            else:
                # Twenty-four lossless placements of the fitting scene exceed
                # the maximum thirteen preceding inputs. No retry explosion.
                start = int(rng.integers(24))
                for v in range(24):
                    pair, found = _fallback_pair(split, i, roles, (start + v) % 24)
                    signature = _key(pair["input"])
                    if signature not in seen and required <= found:
                        seen.add(signature)
                        puzzle[split].append(pair)
                        if split == "test":
                            needed |= found
                        else:
                            taught |= found
                        break
                else:
                    raise AssertionError("fitting unique-input fallback exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnesses = set(), {"train": set(), "test": set()}
    taught_counts, unlisted, simultaneous, repeated_windows = set(), False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_valid_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape
            signature = _key(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            info = _parse(inp)
            predicted, census, eligible = _paint(info)
            assert np.array_equal(predicted, out), "incorrect slice census or paint"
            assert np.array_equal(inp[info["protected"]], out[info["protected"]])
            assert np.array_equal(inp[~info["envelope"]], out[~info["envelope"]])
            assert np.all(out[eligible & ~np.isin(census, list(info["mapping"]))] == info["bg"])
            assert not np.array_equal(inp, out)
            if split == "train" and witnesses["train"] != set(_ALTERNATIVES):
                witnesses["train"] |= _witnesses(info, out)
            elif split == "test" and witnesses["train"] != set(_ALTERNATIVES):
                witnesses["test"] |= _witnesses(info, out)
            if split == "train":
                visible = set(int(x) for x in census[eligible])
                taught_counts |= visible & set(info["mapping"])
                unlisted |= bool(visible - set(info["mapping"]))
                simultaneous |= any(x >= 2 for x in visible)
                repeated_windows |= any(len(x[3]) >= 2 for x in info["objects"])
    assert witnesses["test"] <= witnesses["train"], "untaught alternative-rule distinction"
    assert set(_ALTERNATIVES[:7]) <= witnesses["train"], "missing geometry/topology evidence"
    assert len(taught_counts) >= 3 and unlisted and simultaneous and repeated_windows


def _analytical_fixture(bent=False):
    """Hand-defined input and answer, separate from random construction/painting."""
    grid = np.zeros((17, 17), dtype=np.int8)
    for q, lo, hi in ((8, -6, 6), (13, 1, 3), (22, -6, 6)):
        for t in range(lo, hi + 1, 2):
            grid[(q + t) // 2, (q - t) // 2] = 1
    grid[0:2, 12:15] = 3  # zero-window ink
    grid[13:16, 0:3] = 4
    grid[14, 1] = 0  # one-window ink
    grid[0:3, 0:5] = 5
    grid[1, 1] = grid[1, 3] = 0  # two-window ink
    grid[10:12, 4:7] = 2  # zero-window neutral island
    if bent:
        grid[4:9, 7:12] = 2
        local = ((1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3))
        window = {(r + 4, c + 7) for r, c in local}
        one_window_levels = {-5, -4, -3, -2, -1}
    else:
        grid[5:8, 8:11] = 2
        window = {(6, 9)}
        one_window_levels = {-3}
    for r, c in window:
        grid[r, c] = 0
    expected = grid.copy()
    for r in range(17):
        for c in range(17):
            if (grid[r, c] == 0 and (r, c) not in window
                    and 8 < r + c < 22 and -6 <= r - c <= 6):
                expected[r, c] = 4 if r - c in one_window_levels else 3
    return grid, expected


def self_test():
    """Bounded 42-seed suite, analytical fixtures, and corruption checks; no files."""
    diagonal_escape = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert len(_holes(diagonal_escape)) == 1
    assert len(_holes(diagonal_escape, True)) == 0
    diagonal_escape[0, 1] = False
    assert len(_holes(diagonal_escape)) == 0
    for bent in (False, True):
        fixture, expected = _analytical_fixture(bent)
        assert np.array_equal(solve(fixture), expected)
        info = _parse(fixture)
        assert len(info["holes"]) == 1
        _, census, eligible = _paint(info)
        assert set(int(x) for x in census[eligible]) == {0, 1}
        assert np.all(census[info["t"] == -3] == 1)
        assert {"omit_A", "omit_B", "objects", "pixel_centers", "consecutive"} <= _witnesses(info, expected)
    opened, _ = _analytical_fixture()
    opened[5, 9] = 0  # orthogonal passage turns the ring into an open bay
    assert not _parse(opened)["holes"]
    assert not _paint(_parse(opened))[1].any()
    shapes, layouts, palettes, stroke_counts, window_counts = set(), set(), set(), set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                palettes.add((info["bg"], info["rail"], info["neutral"]))
                stroke_counts.add(len(info["strokes"]))
                window_counts.add(tuple(sorted(len(x[3]) for x in info["objects"])))
                norm = np.full(pair["input"].shape, 3, dtype=np.int8)
                for value, role in ((info["bg"], 0), (info["rail"], 1), (info["neutral"], 2)):
                    norm[pair["input"] == value] = role
                layouts.add(hashlib.sha256(norm.tobytes()).hexdigest())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 20 and len(palettes) >= 30
    assert len(stroke_counts) >= 3 and len(window_counts) >= 5
    a, b = generate_puzzle(731, 2, 1), generate_puzzle(731, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    fixture, expected = _analytical_fixture(True)
    assert np.array_equal(solve(fixture.tolist()), expected)
    permutation = np.array([8, 4, 0, 7, 6, 1, 9, 3, 2, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[fixture]), permutation[expected])
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(fixture, rotation)), np.rot90(expected, rotation))
        assert np.array_equal(solve(np.rot90(fixture, rotation)[:, ::-1]),
                              np.rot90(expected, rotation)[:, ::-1])
    # Every deterministic fallback placement preserves all required witnesses.
    fallback_keys = set()
    for variant in range(24):
        pair, found = _fallback_pair("train", 1, list(range(10)), variant)
        fallback_keys.add(_key(pair["input"]))
        assert found == set(_ALTERNATIVES)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    assert len(fallback_keys) == 24
    fallback = {"train": [_fallback_pair("train", i, list(range(10)), i)[0] for i in range(10)],
                "test": [_fallback_pair("test", i, list(range(10)), i + 10)[0] for i in range(4)]}
    validate_generated_puzzle(fallback)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
