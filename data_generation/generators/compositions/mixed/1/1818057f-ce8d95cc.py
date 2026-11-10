"""Highlight original monochromatic plus unions, then contract equal vectors.

Only standard library and NumPy are used. No parent code, files, saved examples,
seed, or puzzle state is consulted by solve(). See task_spec.md for conventions.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


HIGHLIGHT = 8
COLORS = tuple(c for c in range(10) if c != HIGHLIGHT)
HOST_SEEDS = (0, 618426936, 215310873, 517719799, 603788237, 327772010,
              385742386, 655192421, 762306366, 232867225, 1930127, 381145496,
              699952593, 608225164, 956162587, 647363429, 984318167, 534295970,
              883364327, 391360841, 208188354, 710116869, 915790173, 213202758,
              219901848, 722068561, 655545732, 8034397, 910966069, 588778506,
              242963586, 952682509, 512704900, 471942750, 249279187, 89181916,
              990896694, 141697161, 82781223, 315287508, 351302191, 931553024)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, sides 1..30, colors 0..9")
    return a


def _highlight(a):
    """All centers are decided before any cell is changed."""
    centers = np.zeros(a.shape, dtype=bool)
    if min(a.shape) >= 3:
        mid = a[1:-1, 1:-1]
        centers[1:-1, 1:-1] = (
            (mid != a[0, 0]) & (mid != HIGHLIGHT)
            & (mid == a[:-2, 1:-1]) & (mid == a[2:, 1:-1])
            & (mid == a[1:-1, :-2]) & (mid == a[1:-1, 2:]))
    union = centers.copy()
    union[:-1] |= centers[1:]
    union[1:] |= centers[:-1]
    union[:, :-1] |= centers[:, 1:]
    union[:, 1:] |= centers[:, :-1]
    stage = a.copy()
    stage[union] = HIGHLIGHT
    return stage, centers, union


def _keeps(a):
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    columns = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return np.flatnonzero(rows), np.flatnonzero(columns)


def _contract(a):
    rows, columns = _keeps(a)
    return a[np.ix_(rows, columns)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only family rule; the upper-left cell identifies the background.

    Color 8 is the fixed newly introduced highlight and is absent from inputs.
    Every other color except the background can form a monochromatic plus.
    """
    a = _grid(input_grid)
    if np.any(a == HIGHLIGHT):
        raise ValueError("family inputs reserve color 8 for the output")
    return _contract(_highlight(a)[0])


def _reference(a):
    """Direct coordinate check and explicit run partition, for validation.

    This is an author-owned cross-check, not an independent semantic oracle.
    """
    stage = a.copy()
    centers, cells = [], set()
    for r in range(1, a.shape[0] - 1):
        for c in range(1, a.shape[1] - 1):
            color = int(a[r, c])
            points = ((r, c), (r - 1, c), (r + 1, c),
                      (r, c - 1), (r, c + 1))
            if color not in (int(a[0, 0]), HIGHLIGHT) and all(
                    int(a[p]) == color for p in points):
                centers.append((r, c))
                cells.update(points)
    for p in cells:
        stage[p] = HIGHLIGHT
    rows = [i for i in range(len(stage))
            if i == 0 or stage[i].tolist() != stage[i - 1].tolist()]
    columns = [j for j in range(stage.shape[1]) if j == 0
               or stage[:, j].tolist() != stage[:, j - 1].tolist()]
    return stage[np.ix_(rows, columns)].copy(), stage, centers, cells, rows, columns


def _greedy_stage(a):
    stage = a.copy()
    for r in range(1, len(a) - 1):
        for c in range(1, a.shape[1] - 1):
            color = int(stage[r, c])
            points = ((r, c), (r - 1, c), (r + 1, c),
                      (r, c - 1), (r, c + 1))
            if color not in (int(a[0, 0]), HIGHLIGHT) and all(
                    int(stage[p]) == color for p in points):
                for p in points:
                    stage[p] = HIGHLIGHT
    return stage


def _occupancy_stage(a):
    """Plausible wrong rule: ignore the colors of the five occupied cells."""
    mask = a != a[0, 0]
    union = np.zeros_like(mask)
    for r in range(1, len(a) - 1):
        for c in range(1, a.shape[1] - 1):
            points = ((r, c), (r - 1, c), (r + 1, c),
                      (r, c - 1), (r, c + 1))
            if all(mask[p] for p in points):
                for p in points:
                    union[p] = True
    stage = a.copy()
    stage[union] = HIGHLIGHT
    return stage


def _global_dedup(a):
    rows, seen = [], set()
    for i, row in enumerate(a):
        key = tuple(map(int, row))
        if key not in seen:
            seen.add(key)
            rows.append(i)
    columns, seen = [], set()
    for j, col in enumerate(a.T):
        key = tuple(map(int, col))
        if key not in seen:
            seen.add(key)
            columns.append(j)
    return a[np.ix_(rows, columns)]


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _features(a, answer=None):
    stage, centers, union = _highlight(a)
    answer = _contract(stage) if answer is None else answer
    horizontal = (a[:, 0] != a[0, 0])[:, None]
    vertical = (a[0] != a[0, 0])[None, :]
    raw_rows, raw_cols = _keeps(a)
    rows, cols = _keeps(stage)
    old_r = np.all(a[1:] == a[:-1], axis=1)
    old_c = np.all(a[:, 1:] == a[:, :-1], axis=0)
    new_r = np.all(stage[1:] == stage[:-1], axis=1)
    new_c = np.all(stage[:, 1:] == stage[:, :-1], axis=0)
    selective = False
    for r, c in zip(*np.nonzero(union)):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            x, y = r + dr, c + dc
            if (0 <= x < len(a) and 0 <= y < a.shape[1]
                    and a[x, y] == a[r, c] and not union[x, y]):
                selective = True
    return {
        "shape": a.shape, "output_shape": answer.shape,
        "centers": int(centers.sum()), "union": int(union.sum()),
        "overlap": int(centers.sum()) * 5 > int(union.sum()),
        "horizontal_stroke_centers": int(np.count_nonzero(centers & horizontal & ~vertical)),
        "vertical_stroke_centers": int(np.count_nonzero(centers & vertical & ~horizontal)),
        "broken_rows": int(np.count_nonzero(old_r & ~new_r)),
        "broken_columns": int(np.count_nonzero(old_c & ~new_c)),
        "created_equalities": int(np.count_nonzero(~old_r & new_r))
                              + int(np.count_nonzero(~old_c & new_c)),
        "row_effect": len(rows) > len(raw_rows),
        "column_effect": len(cols) > len(raw_cols),
        "shrink_rows": len(rows) < len(a),
        "shrink_columns": len(cols) < a.shape[1],
        "positive_colors": set(map(int, a[union])),
        "selective": selective,
        "omit_a_differs": not np.array_equal(_contract(a), answer),
        "omit_b_differs": not np.array_equal(stage, answer),
        "reverse_differs": not np.array_equal(_highlight(_contract(a))[0], answer),
        "greedy_differs": not np.array_equal(_contract(_greedy_stage(a)), answer),
        "occupancy_differs": not np.array_equal(_contract(_occupancy_stage(a)), answer),
        "global_differs": not np.array_equal(_global_dedup(stage), answer),
    }


def _line_inventory(a):
    """Recover band geometry solely from the visible top and left margins."""
    background = int(a[0, 0])
    def bands(vector):
        result, i = [], 0
        while i < len(vector):
            if vector[i] == background:
                i += 1
                continue
            end = i + 1
            while end < len(vector) and vector[end] == vector[i]:
                end += 1
            result.append((i, end, int(vector[i])))
            i = end
        return result
    horizontal, vertical = bands(a[:, 0]), bands(a[0])
    row_positions = {r for start, end, _ in horizontal for r in range(start, end)}
    col_positions = {c for start, end, _ in vertical for c in range(start, end)}
    assert horizontal and vertical
    assert all(a[r, c] == background for r in range(len(a))
               for c in range(a.shape[1])
               if r not in row_positions and c not in col_positions)
    for start, end, color in horizontal:
        assert np.all(a[start:end, [j for j in range(a.shape[1])
                                    if j not in col_positions]] == color)
    for start, end, color in vertical:
        assert np.all(a[[i for i in range(len(a)) if i not in row_positions],
                        start:end] == color)
    owners = set()
    for r0, r1, hc in horizontal:
        for c0, c1, vc in vertical:
            block = a[r0:r1, c0:c1]
            assert np.all(block == block[0, 0])
            assert int(block[0, 0]) in (hc, vc)
            if hc != vc:
                owners.add("H" if block[0, 0] == hc else "V")
    def gaps(bands_, size):
        return (bands_[0][0],) + tuple(bands_[i][0] - bands_[i - 1][1]
                                      for i in range(1, len(bands_))) + (size - bands_[-1][1],)
    row_gaps, col_gaps = gaps(horizontal, len(a)), gaps(vertical, a.shape[1])
    assert min(row_gaps + col_gaps) >= 1
    return {"h": len(horizontal), "v": len(vertical), "owners": owners,
            "row_gaps": row_gaps, "column_gaps": col_gaps,
            "thicknesses": tuple(end - start for start, end, _ in horizontal + vertical)}


def _axis(rng, count, thickness_cap, anchor, chosen, long_gap=False):
    thickness = rng.integers(1, thickness_cap + 1, size=count)
    if anchor == "thin":
        thickness[:] = 1
    elif anchor == "overlap":
        # Teach that a complete local plus can occur inside a wide stroke,
        # away from crossings, before tests combine three-cell-wide bands.
        thickness[chosen] = 3
    gaps = np.full(count + 1, 2 if anchor else 1, dtype=int)
    if anchor:
        gaps[chosen] = 4
    if long_gap:
        gaps[int(rng.integers(count + 1))] = int(rng.integers(6, 10))
    # Long-gap tests have at most four bands, so all minima fit within 30.
    assert int(thickness.sum() + gaps.sum()) <= 30
    capacity = (4 if anchor or not long_gap else 10)
    spare = 30 - int(thickness.sum() + gaps.sum())
    extra = int(rng.integers(min(spare, (count + 1) * 3) + 1))
    for index in rng.permutation(count + 1):
        amount = min(extra, capacity - int(gaps[index]))
        if amount > 0:
            add = int(rng.integers(amount + 1))
            gaps[index] += add
            extra -= add
    return thickness.tolist(), gaps.tolist()


def _assemble(background, horizontal, vertical, ht, vt, rg, cg, bits):
    height, width = sum(ht) + sum(rg), sum(vt) + sum(cg)
    a = np.full((height, width), background, dtype=np.int8)
    row_bands, col_bands = [], []
    cursor = rg[0]
    for i, (thick, color) in enumerate(zip(ht, horizontal)):
        row_bands.append(slice(cursor, cursor + thick))
        a[row_bands[-1], :] = color
        cursor += thick + rg[i + 1]
    cursor = cg[0]
    for j, (thick, color) in enumerate(zip(vt, vertical)):
        col_bands.append(slice(cursor, cursor + thick))
        a[:, col_bands[-1]] = color
        cursor += thick + cg[j + 1]
    for i, rows in enumerate(row_bands):
        for j, cols in enumerate(col_bands):
            a[rows, cols] = vertical[j] if bits[i, j] else horizontal[i]
    return a


def _scene(rng, h, v, background, anchor=None, long_axis=None, test=False):
    palette = rng.choice([c for c in COLORS if c != background],
                         size=int(rng.integers(2, 6)), replace=False)
    horizontal = rng.choice(palette, size=h).tolist()
    vertical = rng.choice(palette, size=v).tolist()
    i, j = int(rng.integers(h)), int(rng.integers(v))
    horizontal[i] = vertical[j] = int(palette[0])
    # Two matching colors and two differently owned mismatched crossings teach
    # color-independent eligibility and literal preservation at crossings.
    k, ell = (i + 1) % h, (j + 1) % v
    if anchor:
        horizontal[k] = vertical[ell] = int(palette[1])
    mode = int(rng.integers(5))
    if mode < 2:
        bits = np.full((h, v), mode, dtype=int)
    elif mode == 2:
        bits = np.repeat(rng.integers(2, size=(h, 1)), v, axis=1)
    elif mode == 3:
        bits = np.repeat(rng.integers(2, size=(1, v)), h, axis=0)
    else:
        bits = rng.integers(2, size=(h, v))
    if anchor:
        bits[i, ell], bits[k, j] = 0, 1
    cap = 1 if anchor == "thin" else (3 if test or anchor == "overlap" else 2)
    ht, rg = _axis(rng, h, cap, anchor, i, long_axis == 0)
    vt, cg = _axis(rng, v, cap, anchor, j, long_axis == 1)
    return _assemble(background, horizontal, vertical, ht, vt, rg, cg, bits)


def _anchor_ok(a, kind):
    if not kind:
        return True
    f = _features(a)
    return (f["row_effect"] and f["column_effect"] and f["shrink_rows"]
            and f["shrink_columns"] and f["selective"]
            and len(f["positive_colors"]) >= 2 and f["occupancy_differs"]
            and f["global_differs"] and f["reverse_differs"]
            and (kind != "overlap" or (f["greedy_differs"]
                 and f["horizontal_stroke_centers"] and f["vertical_stroke_centers"])))


def _fallback(h, v, kind, long_axis, seen):
    """At least 72 palette-distinct inputs for <=14 slots, without retries.

    Fallbacks retain the requested inventory and gap shift. Background stability
    is a sampling preference, so fallback palettes may change it.
    """
    ht, vt = [1] * h, [1] * v
    if kind == "overlap":
        ht[0] = vt[0] = 3
    rg, cg = [2] * (h + 1), [2] * (v + 1)
    rg[0] = cg[0] = 4
    if long_axis is not None:
        (rg if long_axis == 0 else cg)[0] = 6
    bits = np.zeros((h, v), dtype=int)
    if h > 1 and v > 1:
        bits[1, 0] = 1
    for background, first, second in itertools.permutations(COLORS, 3):
        horizontal, vertical = [first] * h, [first] * v
        if h > 1 and v > 1:
            horizontal[1] = vertical[1] = second
        a = _assemble(background, horizontal, vertical, ht, vt, rg, cg, bits)
        if _key(a) not in seen and _anchor_ok(a, kind):
            return a
    raise AssertionError("finite fallback uniqueness bound exceeded")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_background = int(rng.choice(COLORS))
    vary_background = bool(rng.integers(2))
    shift = str(rng.choice(("inventory", "spacing")))
    shift_axis = int(rng.integers(2))
    inventories = [tuple(map(int, rng.integers(2, 4, size=2))) for _ in range(2)]
    inventories += [tuple(map(int, rng.integers(1, 5, size=2)))
                    for _ in range(num_train - 2)]
    test_inventories = []
    maximum = max(inv[shift_axis] for inv in inventories)
    for index in range(num_test):
        if shift == "inventory":
            inv = list(map(int, rng.integers(1, 6, size=2)))
            inv[shift_axis] = maximum + 1
            # Cycle the other inventory if a multi-test draw repeats it.
            while tuple(inv) in test_inventories:
                inv[1 - shift_axis] = inv[1 - shift_axis] % 5 + 1
            test_inventories.append(tuple(inv))
        else:
            # Up to four bands leave room for a long gap and thick strokes.
            inv = tuple(map(int, rng.integers(1, 5, size=2)))
            while inv in test_inventories:
                inv = (inv[0] % 4 + 1, inv[1])
            test_inventories.append(inv)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, invs in (("train", inventories), ("test", test_inventories)):
        for index, (h, v) in enumerate(invs):
            kind = ("thin", "overlap")[index] if split == "train" and index < 2 else None
            background = base_background
            if vary_background and (split == "test" or index > 0):
                choices = [c for c in COLORS if index != 1 or split != "train"
                           or c != base_background]
                background = int(rng.choice(choices))
            long_axis = shift_axis if split == "test" and shift == "spacing" else None
            for _ in range(24):
                a = _scene(rng, h, v, background, kind, long_axis, split == "test")
                if _key(a) not in seen and _anchor_ok(a, kind):
                    break
            else:
                a = _fallback(h, v, kind, long_axis, seen)
            seen.add(_key(a))
            puzzle[split].append({"input": a, "output": solve(a)})
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses, geometry = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert HIGHLIGHT not in a
            assert a[0, 0] == a[0, -1] == a[-1, 0] == a[-1, -1]
            assert _key(a) not in seen
            seen.add(_key(a))
            expected, stage, centers, cells, rows, columns = _reference(a)
            assert np.array_equal(out, expected), "wrong simultaneous-plus/run-contraction label"
            assert np.array_equal(out, solve(a)), "production solver disagrees with coordinate check"
            union = np.zeros_like(a, dtype=bool)
            for p in cells:
                union[p] = True
            assert np.array_equal(a != stage, union)
            assert np.array_equal(a[~union], stage[~union])
            assert np.array_equal(out, stage[np.ix_(rows, columns)])
            # Every vector run is recoverable by expanding its retained vector.
            rr = np.diff(rows + [len(a)])
            cc = np.diff(columns + [a.shape[1]])
            assert np.array_equal(np.repeat(np.repeat(out, rr, axis=0), cc, axis=1), stage)
            assert np.array_equal(_contract(out), out)
            f = _features(a, out)
            assert f["created_equalities"] == 0
            analyses[split].append(f)
            geometry[split].append(_line_inventory(a))
    train = analyses["train"]
    assert any(f["row_effect"] and f["column_effect"] and f["shrink_rows"]
               and f["shrink_columns"] and f["reverse_differs"] for f in train), "missing causal witness"
    assert any(f["selective"] and f["occupancy_differs"] and f["global_differs"] for f in train)
    assert len(set().union(*(f["positive_colors"] for f in train))) >= 2
    assert {"H", "V"} <= set().union(*(g["owners"] for g in geometry["train"]))
    assert any(f["greedy_differs"] for f in train), "teach overlapping union before testing it"
    assert any(f["horizontal_stroke_centers"] and f["vertical_stroke_centers"]
               for f in train), "teach pluses away from crossings in both orientations"
    assert any(f["omit_a_differs"] and f["omit_b_differs"] for f in train)
    assert all(f["centers"] and f["omit_a_differs"] for f in analyses["test"])
    for test in geometry["test"]:
        inventory_shift = any(test[k] > max(g[k] for g in geometry["train"])
                              for k in ("h", "v"))
        spacing_shift = any(max(test[k]) > max(max(g[k]) for g in geometry["train"])
                            for k in ("row_gaps", "column_gaps"))
        assert inventory_shift or spacing_shift, "test must transfer to a new inventory or gap scale"


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
    """Stream 42 puzzles and statistics; never write or retain a seed corpus."""
    fingerprints, shapes, output_shapes, inventories, center_counts = set(), set(), set(), set(), set()
    normalized, backgrounds, positive_colors, thicknesses = set(), set(), set(), set()
    pairs = broken_r = broken_c = overlap = one_axis = no_compression = 0
    ownership = {"H": 0, "V": 0, "mixed": 0}
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a, out = pair["input"], pair["output"]
                f, g = _features(a, out), _line_inventory(a)
                shapes.add(a.shape)
                output_shapes.add(out.shape)
                inventories.add((g["h"], g["v"]))
                center_counts.add(f["centers"])
                backgrounds.add(int(a[0, 0]))
                positive_colors.update(f["positive_colors"])
                thicknesses.update(g["thicknesses"])
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                normalized.add((a.shape, hashlib.sha256(norm).digest()))
                pairs += 1
                broken_r += f["broken_rows"] > 0
                broken_c += f["broken_columns"] > 0
                overlap += f["overlap"]
                one_axis += f["shrink_rows"] != f["shrink_columns"]
                no_compression += not f["shrink_rows"] and not f["shrink_columns"]
                if g["owners"] == {"H"}:
                    ownership["H"] += 1
                elif g["owners"] == {"V"}:
                    ownership["V"] += 1
                elif g["owners"] == {"H", "V"}:
                    ownership["mixed"] += 1
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(output_shapes) >= 25
    assert len(normalized) >= 150 and len(inventories) >= 15
    assert len(center_counts) >= 20 and backgrounds == set(COLORS)
    assert positive_colors == set(COLORS) and {1, 2, 3} <= thicknesses
    assert broken_r >= 84 and broken_c >= 84 and overlap >= 42
    assert all(ownership[k] >= 3 for k in ownership)
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1701, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Local geometry, boundary and compression regressions use exact answers.
    tiny = np.array([[0, 4, 0], [4, 4, 4], [0, 4, 0]], dtype=np.int8)
    assert solve(tiny).tolist() == [[0, 8, 0], [8, 8, 8], [0, 8, 0]]
    cross = np.zeros((7, 7), dtype=np.int8)
    cross[3, :] = cross[:, 3] = 4
    assert solve(cross).tolist() == [[0, 0, 4, 0, 0], [0, 0, 8, 0, 0],
                                    [4, 8, 8, 8, 4], [0, 0, 8, 0, 0],
                                    [0, 0, 4, 0, 0]]
    assert _contract(cross).shape == (3, 3), "highlighting must change run boundaries"
    plain = np.array([[0, 2, 0, 2, 0], [0, 2, 0, 2, 0],
                      [3, 2, 3, 2, 3], [0, 2, 0, 2, 0]], dtype=np.int8)
    assert np.array_equal(solve(plain), plain[[0, 2, 3]])
    assert _highlight(plain)[1].sum() == 0, "mixed-color crossing is not a plus"
    clipped = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    # Set a true background corner: the clipped T is at the top boundary.
    clipped = np.pad(clipped, ((0, 0), (1, 1)))
    assert _highlight(clipped)[1].sum() == 0
    block = np.pad(np.full((3, 4), 4, dtype=np.int8), 2)
    assert _highlight(block)[1].sum() == 2
    assert (_highlight(block)[0] == 8).sum() == 8
    for shape in ((1, 1), (1, 30), (30, 1)):
        assert solve(np.zeros(shape, dtype=int)).tolist() == [[0]]
    # Axis order does not change contraction; rotations commute with this rule.
    a = generate_puzzle(33)["test"][0]["input"]
    out = solve(a)
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(a, turns)), np.rot90(out, turns))
    # A finite fallback with 15 excluded palettes still succeeds.
    seen = set()
    for _ in range(16):
        a = _fallback(2, 2, "overlap", None, seen)
        assert _anchor_ok(a, "overlap")
        seen.add(_key(a))
    # Deliberately corrupt a label: validator must reject it.
    bad = generate_puzzle(5, 2, 1)
    bad["test"][0]["output"][0, 0] = HIGHLIGHT
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "normalized_inputs": len(normalized), "inventories": len(inventories),
            "center_counts": len(center_counts), "broken_row_pairs": broken_r,
            "broken_column_pairs": broken_c, "overlap_pairs": overlap,
            "one_axis_compression_pairs": one_axis, "no_compression_pairs": no_compression,
            "crossing_ownership": ownership}
