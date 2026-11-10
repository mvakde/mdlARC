"""Seam-sensitive wallpaper: periodic neighborhoods select five-cell crosses.

Only standard-library modules and NumPy are used. Generation and self_test do
not write files, spawn workers, retain runs, or save execution traces.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_SHAPES = tuple((h, w) for h in range(3, 8) for w in range(3, 8))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _parse(input_grid):
    """Find the unique solid rectangular outline and its exterior field."""
    a = _grid(input_grid)
    colors = [int(c) for c in np.unique(a)]
    if len(colors) != 3:
        raise ValueError("input must contain field, material and frame colors")
    candidates = []
    for ink in colors:
        points = np.argwhere(a == ink)
        top, left = map(int, points.min(axis=0))
        bottom, right = map(int, points.max(axis=0))
        h, w = bottom - top - 1, right - left - 1
        if not (3 <= h <= 7 and 3 <= w <= 7):
            continue
        outline = np.zeros(a.shape, dtype=bool)
        outline[top, left:right + 1] = True
        outline[bottom, left:right + 1] = True
        outline[top:bottom + 1, left] = True
        outline[top:bottom + 1, right] = True
        if not np.array_equal(a == ink, outline):
            continue
        exterior = np.ones(a.shape, dtype=bool)
        exterior[top:bottom + 1, left:right + 1] = False
        outside = np.unique(a[exterior])
        if len(outside) != 1:
            continue
        field = int(outside[0])
        material = next((c for c in colors if c not in (ink, field)), None)
        tile = a[top + 1:bottom, left + 1:right]
        if material is None or set(map(int, np.unique(tile))) != {field, material}:
            continue
        candidates.append({"field": field, "material": material, "ink": ink,
                           "origin": (top + 1, left + 1), "mask": tile == material})
    if len(candidates) != 1:
        raise ValueError("expected one framed swatch and a nonempty uniform exterior")
    return a, candidates[0]


def _centers(mask):
    centers = mask.copy()
    for dr, dc in _CROSS[1:]:
        centers &= np.roll(mask, (dr, dc), axis=(0, 1))
    return centers


def _seam_centers(mask):
    centers = _centers(mask)
    centers[1:-1, 1:-1] = False
    return centers


def _union(centers):
    result = centers.copy()
    for dr, dc in _CROSS[1:]:
        result |= np.roll(centers, (dr, dc), axis=(0, 1))
    return result


def _sequential_union(mask):
    """Plausible wrong rule: destroy matches during a phase-row-major scan."""
    h, w = mask.shape
    remaining, changed = mask.copy(), np.zeros_like(mask)
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            points = [((r + dr) % h, (c + dc) % w) for dr, dc in _CROSS]
            if all(remaining[p] for p in points):
                for p in points:
                    remaining[p] = False
                    changed[p] = True
    return changed


def _render(mask, union, shape, origin, roles):
    field, material, ink = roles
    rows = (np.arange(shape[0]) - origin[0]) % mask.shape[0]
    cols = (np.arange(shape[1]) - origin[1]) % mask.shape[1]
    occupied = mask[rows[:, None], cols[None, :]]
    highlighted = union[rows[:, None], cols[None, :]]
    out = np.full(shape, field, dtype=np.int8)
    out[occupied] = material
    out[highlighted] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, tile, phase, and all highlights from this input alone."""
    a, info = _parse(input_grid)
    mask = info["mask"]
    return _render(mask, _union(_seam_centers(mask)), a.shape, info["origin"],
                   (info["field"], info["material"], info["ink"]))


def _reference_output(input_grid):
    """Scalar check using tile-instance identities to define seam crossing.

    Separate from the production roll/edge-mask/dilation implementation. This
    is an authored check, not an independent external semantic reference.
    """
    a, info = _parse(input_grid)
    mask = info["mask"]
    h, w = mask.shape
    marked = set()
    for r in range(h):
        for c in range(w):
            points = [(r + dr, c + dc) for dr, dc in _CROSS]
            instances = {(y // h, x // w) for y, x in points}
            if len(instances) > 1 and all(mask[y % h, x % w] for y, x in points):
                marked.update((y % h, x % w) for y, x in points)
    out = np.empty(a.shape, dtype=np.int8)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = ((r - info["origin"][0]) % h, (c - info["origin"][1]) % w)
            out[r, c] = (info["ink"] if p in marked else
                         info["material"] if mask[p] else info["field"])
    return out


def _asymmetric(mask):
    alternatives = [mask[::-1], mask[:, ::-1], mask[::-1, ::-1]]
    if mask.shape[0] == mask.shape[1]:
        alternatives += [mask.T, mask.T[::-1], mask.T[:, ::-1], mask.T[::-1, ::-1]]
    return all(not np.array_equal(mask, other) for other in alternatives)


def _features(mask):
    centers = _centers(mask)
    seam = _seam_centers(mask)
    marked = _union(seam)
    row_only, col_only = seam.copy(), seam.copy()
    row_only[1:-1, :] = False
    col_only[:, 1:-1] = False
    boundary = mask.copy()
    boundary[1:-1, 1:-1] = False
    adjacent = np.zeros_like(mask)
    for dr, dc in _CROSS[1:]:
        adjacent |= np.roll(marked, (dr, dc), axis=(0, 1))
    return {"seam_centers": int(seam.sum()),
            "ordinary_centers": int(centers[1:-1, 1:-1].sum()),
            "marked": int(marked.sum()),
            "unmarked": int((mask & ~marked).sum()),
            "overlap": 5 * int(seam.sum()) > int(marked.sum()),
            "sequential_failure": not np.array_equal(marked, _sequential_union(mask)),
            "row_only_failure": not np.array_equal(marked, _union(row_only)),
            "col_only_failure": not np.array_equal(marked, _union(col_only)),
            "boundary_unmarked": int((boundary & ~marked).sum()),
            "attached_unmarked": bool((mask & ~marked & adjacent).any()),
            "asymmetric": _asymmetric(mask)}


def _acceptable(mask, kind):
    if not mask.any() or mask.all():
        return False
    seam = _seam_centers(mask)
    if kind == "positive":
        return bool(seam.any())
    if kind == "free":
        return True
    if kind == "negative":
        boundary = mask.copy()
        boundary[1:-1, 1:-1] = False
        return not seam.any() and bool(_centers(mask)[1:-1, 1:-1].any()) and bool(boundary.any())
    features = _features(mask)
    return (features["seam_centers"] >= 2 and features["ordinary_centers"] >= 1
            and features["unmarked"] > 0 and features["sequential_failure"]
            and features["row_only_failure"] and features["col_only_failure"]
            and features["attached_unmarked"]
            and features["asymmetric"] and features["marked"] > features["seam_centers"])


def _paint_cross(mask, r, c):
    h, w = mask.shape
    for dr, dc in _CROSS:
        mask[(r + dr) % h, (c + dc) % w] = True


def _sample_mask(rng, shape, kind):
    """At most 160 random proposals, followed by a finite constructive fallback."""
    h, w = shape
    for _ in range(160):
        style = int(rng.integers(4))
        density = float(rng.uniform(.10, .68 if kind != "negative" else .40))
        mask = rng.random(shape) < density
        if style == 1:
            # A short branching walk, independent of the final plus detector.
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(2, h * w + 1))):
                mask[r, c] = True
                dr, dc = _CROSS[int(rng.integers(1, 5))]
                r, c = (r + dr) % h, (c + dc) % w
        elif style == 2:
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[r:r + int(rng.integers(1, h + 1)),
                 c:c + int(rng.integers(1, w + 1))] = True
        elif style == 3:
            # Occupancy with correlated rows/columns and freely varying holes.
            mask |= ((rng.random((h, 1)) < .25) & (rng.random((1, w)) < .65))
        if kind in ("witness", "negative"):
            _paint_cross(mask, int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        if kind == "negative":
            # A boundary-colored near miss distinguishes seams from crosses.
            if rng.random() < .5:
                mask[int(rng.choice((0, h - 1))), int(rng.integers(w))] = True
            else:
                mask[int(rng.integers(h)), int(rng.choice((0, w - 1)))] = True
        if kind in ("witness", "positive"):
            edge = int(rng.integers(4))
            if edge < 2:
                r, c = (0 if edge == 0 else h - 1), int(rng.integers(w))
                _paint_cross(mask, r, c)
                if kind == "witness":
                    _paint_cross(mask, r, (c + 1) % w)
                    _paint_cross(mask, int(rng.integers(1, h - 1)), int(rng.choice((0, w - 1))))
            else:
                r, c = int(rng.integers(h)), (0 if edge == 2 else w - 1)
                _paint_cross(mask, r, c)
                if kind == "witness":
                    _paint_cross(mask, (r + 1) % h, c)
                    _paint_cross(mask, int(rng.choice((0, h - 1))), int(rng.integers(1, w - 1)))
        if _acceptable(mask, kind):
            return mask
    # These constructions require no rejection and never depend on seed state.
    # Witness/negative shapes are >=5 on both axes.
    mask = np.zeros(shape, dtype=bool)
    if kind == "negative":
        _paint_cross(mask, h // 2, w // 2)
        mask[0, 0] = True
    elif kind == "witness":
        _paint_cross(mask, 0, 1)
        _paint_cross(mask, 0, 2)
        _paint_cross(mask, h // 2, 0)
        _paint_cross(mask, h // 2, w // 2)
    else:
        _paint_cross(mask, 0, 1)
    if not _acceptable(mask, kind):
        raise AssertionError("constructive mask fallback violated its contract")
    return mask


def _make_pair(rng, mask, roles, register=False):
    h, w = mask.shape
    height = int(rng.integers(max(9, 2 * h + 1), 31))
    width = int(rng.integers(max(9, 2 * w + 1), 31))
    marked = _union(_seam_centers(mask))
    phases = {(r, c) for r in range(h) for c in range(w)}
    if register:
        # A highlighted canvas corner proves that continuation precedes crop:
        # no complete nonwrapping plus can contain a finite canvas corner.
        corner = (int(rng.choice((0, height - 1))), int(rng.choice((0, width - 1))))
        phases = {(r, c) for r, c in phases
                  if marked[(corner[0] - r) % h, (corner[1] - c) % w]
                  and not np.array_equal(np.roll(mask, (r, c), axis=(0, 1)), mask)}
    positions = [(r, c) for r in range(height - h - 1)
                 for c in range(width - w - 1) if ((r + 1) % h, (c + 1) % w) in phases]
    edge = [p for p in positions if p[0] in (0, height - h - 2)
            or p[1] in (0, width - w - 2)]
    choices = edge if edge and rng.random() < .45 else positions
    top, left = choices[int(rng.integers(len(choices)))]
    field, material, ink = roles
    inp = np.full((height, width), field, dtype=np.int8)
    inp[top:top + h + 2, left:left + w + 2] = ink
    inp[top + 1:top + h + 1, left + 1:left + w + 1] = np.where(mask, material, field)
    out = _render(mask, marked, inp.shape, (top + 1, left + 1), roles)
    return {"input": inp, "output": out}


def _check_counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError("num_train must be 2..10 and num_test 1..4 integers")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    palette_mode = int(rng.integers(3))  # stable, role permutations, fresh roles
    available = list(_SHAPES)
    result = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                choices = [s for s in available if min(s) >= 5]
                kind = "witness" if index == 0 else "negative"
            else:
                choices = available
                kind = "positive" if split == "test" or rng.random() < .75 else "free"
            shape = choices[int(rng.integers(len(choices)))]
            available.remove(shape)
            mask = _sample_mask(rng, shape, kind)
            if palette_mode == 0:
                roles = base_roles
            elif palette_mode == 1:
                order = rng.permutation(3)
                roles = tuple(base_roles[int(k)] for k in order)
            else:
                roles = tuple(map(int, rng.choice(10, 3, replace=False)))
            result[split].append(_make_pair(rng, mask, roles,
                                           register=(split == "train" and index == 0)))
    result["train"] = [result["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(result)
    return result


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    crop_witness = phase_witness = False
    periods = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            _, info = _parse(inp)
            mask = info["mask"]
            periods[split].add(mask.shape)
            features[split].append(_features(mask))
            assert np.array_equal(out, _reference_output(inp)), "incorrect periodic seam footprint"
            assert np.array_equal(out, solve(inp)), "solve disagrees"
            rows = (np.arange(inp.shape[0]) - info["origin"][0]) % mask.shape[0]
            cols = (np.arange(inp.shape[1]) - info["origin"][1]) % mask.shape[1]
            tiled = mask[rows[:, None], cols[None, :]]
            assert np.array_equal(out == info["field"], ~tiled), "carrier holes must survive"
            assert np.all(np.isin(out[tiled], (info["material"], info["ink"])))
            assert not np.array_equal(inp, out)
            if split == "train":
                crop_witness |= bool(np.any(out[np.ix_((0, out.shape[0] - 1),
                                                       (0, out.shape[1] - 1))] == info["ink"]))
                origin = info["origin"]
                phase_witness |= not np.array_equal(np.roll(mask, origin, axis=(0, 1)), mask)
    train = features["train"]
    assert any(f["seam_centers"] >= 2 and f["ordinary_centers"] and f["unmarked"]
               and f["sequential_failure"] and f["asymmetric"]
               and f["row_only_failure"] and f["col_only_failure"]
               and f["attached_unmarked"] for f in train), "missing positive witness"
    assert any(not f["seam_centers"] and f["ordinary_centers"] and f["boundary_unmarked"]
               for f in train), "missing interior-only witness"
    assert crop_witness and phase_witness, "missing crop/registration evidence"
    assert all(f["seam_centers"] > 0 for f in features["test"])
    assert periods["train"].isdisjoint(periods["test"]), "tests must change period shape"
    assert len(periods["train"]) >= 2


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, in-memory checks over 42 seeds plus exhaustive 3x3 topology."""
    shapes, periods, masks, outputs, normalized, fingerprints = (set() for _ in range(6))
    role_colors = [set(), set(), set()]
    seam_counts, occupancies = set(), set()
    edge_frames = interior_frames = overlapping = empty_rows = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, info = _parse(pair["input"])
                mask = info["mask"]
                shapes.add(a.shape)
                periods.add(mask.shape)
                masks.add((mask.shape, mask.tobytes()))
                outputs.add(_key(pair["output"]))
                normalized.add((mask.shape, mask.tobytes(), _union(_seam_centers(mask)).tobytes()))
                for observed, role in zip(role_colors, ("field", "material", "ink")):
                    observed.add(info[role])
                f = _features(mask)
                seam_counts.add(f["seam_centers"])
                occupancies.add(int(mask.sum()))
                overlapping += f["overlap"]
                empty_rows += bool((~mask.any(axis=0)).any() or (~mask.any(axis=1)).any())
                top, left = info["origin"][0] - 1, info["origin"][1] - 1
                edge = top == 0 or left == 0 or top + mask.shape[0] + 2 == a.shape[0] or left + mask.shape[1] + 2 == a.shape[1]
                edge_frames += edge
                interior_frames += not edge
                before = pair["input"].copy()
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated input"
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and periods == set(_SHAPES)
    assert len(masks) >= 150 and len(outputs) >= 190 and len(normalized) >= 150
    assert all(len(c) == 10 for c in role_colors)
    assert min(seam_counts) == 0 and max(seam_counts) >= 8
    assert len(occupancies) >= 20 and overlapping >= 42
    assert min(edge_frames, interior_frames) >= 30 and empty_rows >= 5
    # Every binary 3x3: direct neighborhood/instance sets versus vectorization.
    for bits in range(512):
        mask = np.array([(bits >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3)
        marked = set()
        for r in range(3):
            for c in range(3):
                points = [(r + dr, c + dc) for dr, dc in _CROSS]
                if (len({(y // 3, x // 3) for y, x in points}) > 1
                        and all(mask[y % 3, x % 3] for y, x in points)):
                    marked.update((y % 3, x % 3) for y, x in points)
        expected = np.zeros((3, 3), dtype=bool)
        for point in marked:
            expected[point] = True
        assert np.array_equal(expected, _union(_seam_centers(mask)))
    # A split cross needs both seams; an ordinary cross needs neither.
    for h, w in _SHAPES:
        mask = np.zeros((h, w), dtype=bool)
        _paint_cross(mask, 0, 0)
        assert _union(_seam_centers(mask)).sum() >= 5
        if min(h, w) >= 5:
            isolated = np.zeros((h, w), dtype=bool)
            _paint_cross(isolated, h // 2, w // 2)
            assert _centers(isolated).any() and not _seam_centers(isolated).any()
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * train + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt one occupied or one field cell; neither must pass the validator.
    for corrupt_field in (False, True):
        puzzle = generate_puzzle(2026)
        pair = puzzle["test"][0]
        _, info = _parse(pair["input"])
        out = pair["output"]
        cells = np.argwhere((out == info["field"]) if corrupt_field else (out == info["ink"]))
        p = tuple(cells[0])
        out[p] = info["ink"] if corrupt_field else info["material"]
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "canvas_shapes": len(shapes), "period_shapes": len(periods),
            "distinct_masks": len(masks), "distinct_outputs": len(outputs),
            "color_free_rules": len(normalized), "seam_center_counts": sorted(seam_counts),
            "occupancy_counts": sorted(occupancies), "overlapping_pairs": int(overlapping),
            "edge_frames": int(edge_frames), "interior_frames": int(interior_frames),
            "empty_row_or_column_tiles": int(empty_rows),
            "verification": "generator-owned checks only; no independent solver or blind prediction"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
