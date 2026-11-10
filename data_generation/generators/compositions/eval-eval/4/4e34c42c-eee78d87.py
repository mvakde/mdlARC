"""Registered print sheets: motif phase and overlap depth share one coordinate map.

Black separates physical sheets. Each sheet contains one identical 3x3 accent
stamp on a carrier. Register the stamps, repeat their binary pattern across
the union, and print overlapping accent cells in color 9. No generation state
or training examples are used by solve.
"""

from itertools import product
import json

import numpy as np


_STAMP = frozenset(product(range(3), repeat=2))


def _full_box(mask):
    return (2 <= int(mask.sum()) <= 8 and mask[0].any() and mask[2].any()
            and mask[:, 0].any() and mask[:, 2].any())


_MASKS = tuple(np.array([(code >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3)
               for code in range(1, 511)
               if _full_box(np.array([(code >> k) & 1 for k in range(9)],
                                     dtype=bool).reshape(3, 3)))


def _asymmetric(mask):
    images = [np.rot90(mask, k) for k in range(4)]
    images += [np.fliplr(a) for a in images[:]]
    return len({a.tobytes() for a in images}) == 8


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("a grid must be an integer rectangle, 1..30, with colors 0..9")
    return a.astype(np.int8, copy=True)


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _bounds(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _rectangle(bounds):
    r0, c0, r1, c1 = bounds
    return set(product(range(r0, r1+1), range(c0, c1+1)))


def _read_input(value):
    """Recover each material silhouette and its stamp-relative coordinates."""
    grid = _grid(value)
    if 9 in grid:
        raise ValueError("color 9 is reserved for the output overlap ink")
    parts = _groups(map(tuple, np.argwhere(grid != 0)))
    if not 2 <= len(parts) <= 6:
        raise ValueError("need two to six separated sheets")
    sheets, patterns, roles, anchors = [], [], [], []
    for part in parts:
        values, counts = np.unique([grid[p] for p in part], return_counts=True)
        if len(values) != 2 or counts.max()*2 <= len(part):
            raise ValueError("each sheet needs one dominant carrier and one accent")
        carrier = int(values[counts.argmax()])
        accent = int(values[counts.argmin()])
        marks = {p for p in part if grid[p] == accent}
        r0, c0, r1, c1 = _bounds(marks)
        if (r1-r0, c1-c0) != (2, 2):
            raise ValueError("the accent stamp must have a tight 3x3 box")
        sheet = {(int(r-r0), int(c-c0)) for r, c in part}
        if not _STAMP <= sheet:
            raise ValueError("carrier gaps in the complete stamp must be material")
        mask = np.array([[grid[r0+r, c0+c] == accent for c in range(3)]
                         for r in range(3)], dtype=bool)
        if not _full_box(mask):
            raise ValueError("invalid binary stamp")
        sheets.append(sheet)
        patterns.append(mask)
        roles.append((carrier, accent))
        anchors.append((int(r0), int(c0)))
    if len(set(roles)) != 1 or any(not np.array_equal(patterns[0], m) for m in patterns):
        raise ValueError("all sheets must carry the same colors and exact stamp")
    return sheets, patterns[0], roles[0], anchors


def _coverage(sheets):
    bounds = _bounds(set().union(*sheets))
    r0, c0, r1, c1 = bounds
    if r1-r0 >= 30 or c1-c0 >= 30:
        raise ValueError("registered union exceeds ARC dimensions")
    depth = np.zeros((r1-r0+1, c1-c0+1), dtype=np.int8)
    for sheet in sheets:
        for r, c in sheet:
            depth[r-r0, c-c0] += 1
    return depth, (r0, c0)


def _print_field(sheets, mask, carrier, accent):
    depth, (r0, c0) = _coverage(sheets)
    rows = (np.arange(depth.shape[0]) + r0) % 3
    cols = (np.arange(depth.shape[1]) + c0) % 3
    motif = mask[rows[:, None], cols[None, :]]
    output = np.zeros(depth.shape, dtype=np.int8)
    output[depth > 0] = carrier
    output[motif & (depth == 1)] = accent
    output[motif & (depth >= 2)] = 9
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate by exact stamp agreement; print with observable material depth."""
    sheets, mask, (carrier, accent), _ = _read_input(input_grid)
    return _print_field(sheets, mask, carrier, accent)


def _stamp_reference(sheets, mask, carrier, accent):
    """Construction check: stamp tiles, counting physical sheets cell by cell.

    This is an authored check, not an independent semantic reference solver.
    It uses construction silhouettes, rather than extracting them from inputs.
    """
    r0, c0, r1, c1 = _bounds(set().union(*sheets))
    out = np.zeros((r1-r0+1, c1-c0+1), dtype=np.int8)
    for row in range((r0//3)*3, r1+1, 3):
        for col in range((c0//3)*3, c1+1, 3):
            for dr, dc in product(range(3), repeat=2):
                r, c = row+dr, col+dc
                if not (r0 <= r <= r1 and c0 <= c <= c1):
                    continue
                count = sum((r, c) in sheet for sheet in sheets)
                if count:
                    out[r-r0, c-c0] = (carrier if not mask[dr, dc]
                                      else accent if count == 1 else 9)
    return out


def _random_sheet(rng, maximum):
    h, w = (int(rng.integers(5, maximum+1)) for _ in range(2))
    top, left = int(rng.integers(h-2)), int(rng.integers(w-2))
    bounds = (-top, -left, h-top-1, w-left-1)
    sheet = _rectangle(bounds)
    style = int(rng.integers(4))
    if style == 1:
        # Clip one or more corners, always preserving the complete stamp.
        for _ in range(int(rng.integers(1, 4))):
            rr = bounds[0] if rng.integers(2) else bounds[2]
            cc = bounds[1] if rng.integers(2) else bounds[3]
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            cut = {p for p in sheet if abs(p[0]-rr) < dh and abs(p[1]-cc) < dw}
            if not cut & _STAMP and len(_groups(sheet-cut)) == 1:
                sheet -= cut
    elif style == 2:
        # Crossing strips can form long arms and four reentrant corners.
        sheet = {p for p in sheet if 0 <= p[0] <= 2 or 0 <= p[1] <= 2}
    elif style == 3:
        # Punch carrier holes outside the stamp; keep the sheet connected.
        candidates = sorted(sheet-_STAMP)
        rng.shuffle(candidates)
        for p in candidates[:int(rng.integers(2, 9))]:
            if len(_groups(sheet-{p})) == 1:
                sheet.remove(p)
    return sheet


def _depth_witness(rng):
    """Three sheets, including one contained sheet with a visible ink effect."""
    a = _rectangle((-int(rng.integers(2, 5)), -int(rng.integers(2, 5)),
                    int(rng.integers(4, 7)), int(rng.integers(4, 7))))
    b = _rectangle((-int(rng.integers(2)), -int(rng.integers(2)),
                    int(rng.integers(4, 7)), int(rng.integers(4, 7))))
    ar0, ac0, _, _ = _bounds(a)
    c = _rectangle((ar0+int(rng.integers(2)), ac0+int(rng.integers(2)),
                    int(rng.integers(2, 4)), int(rng.integers(2, 4))))
    # The contained sheet's notch differentiates material from its rectangle.
    corner = min(c)
    if corner not in _STAMP:
        c.remove(corner)
    # A hole near a lower corner may survive or be covered by another sheet.
    br0, bc0, br1, bc1 = _bounds(b)
    p = (br1-1, bc1-1)
    if p not in _STAMP:
        b.remove(p)
    return [a, b, c]


def _witness_checks(sheets, mask, *, contained=False, phase=False):
    depth, (r0, c0) = _coverage(sheets)
    yy, xx = np.indices(depth.shape)
    motif = mask[(yy+r0) % 3, (xx+c0) % 3]
    # Every demonstration shows both inks and carrier gaps in both depth roles.
    if not all(np.any((depth == d) & (motif == v))
               for d in (1, 2) for v in (False, True)):
        return False
    outside = ~((yy+r0 >= 0) & (yy+r0 <= 2) & (xx+c0 >= 0) & (xx+c0 <= 2))
    if np.count_nonzero(outside & motif & (depth > 0)) < 4:
        return False
    if phase:
        # The translation witness also rules out aligning sheet corners.
        if len({_bounds(sheet)[:2] for sheet in sheets}) < 2:
            return False
        if np.array_equal(motif & (depth > 0),
                          mask[yy % 3, xx % 3] & (depth > 0)):
            return False
    if contained:
        union = set().union(*sheets)
        useful = False
        for i, sheet in enumerate(sheets):
            other = [q for j, q in enumerate(sheets) if i != j]
            if sheet <= set().union(*other):
                useful |= any(mask[r % 3, c % 3]
                              and sum((r, c) in q for q in sheets) == 2
                              for r, c in sheet)
        if not useful:
            return False
        rectangles = [_rectangle(_bounds(q)) for q in sheets]
        # Bounding boxes are an incorrect substitute for actual silhouettes.
        if np.array_equal(_print_field(sheets, mask, 1, 2),
                          _print_field(rectangles, mask, 1, 2)):
            return False
        if len(union) == 9:
            return False
    return True


def _pack(sheets, mask, carrier, accent, rng):
    """Random shelf layouts reserve separated boxes, without encoding adjacency."""
    normalized, shapes = [], []
    for sheet in sheets:
        r0, c0, r1, c1 = _bounds(sheet)
        normalized.append({(r-r0, c-c0):
                           accent if (r, c) in _STAMP and mask[r, c] else carrier
                           for r, c in sheet})
        shapes.append((r1-r0+1, c1-c0+1))
    n = len(sheets)
    options = []
    for _ in range(16):
        order = list(map(int, rng.permutation(n)))
        for transpose in (False, True):
            dims = [shape[::-1] if transpose else shape for shape in shapes]
            for cuts in range(1 << (n-1)):
                rows = [[order[0]]]
                for k, i in enumerate(order[1:]):
                    if (cuts >> k) & 1:
                        rows.append([])
                    rows[-1].append(i)
                height = sum(max(dims[i][0] for i in row) for row in rows)+len(rows)-1
                width = max(sum(dims[i][1] for i in row)+len(row)-1 for row in rows)
                if height <= 30 and width <= 30:
                    options.append((transpose, dims, rows, height, width))
    if not options:
        raise ValueError("sheet boxes cannot fit a 30x30 separated canvas")
    transpose, dims, rows, h, w = options[int(rng.integers(len(options)))]
    h += int(rng.integers(min(30-h, 6)+1))
    w += int(rng.integers(min(30-w, 6)+1))
    canvas = np.zeros((w, h) if transpose else (h, w), dtype=np.int8)
    base_height = sum(max(dims[i][0] for i in row) for row in rows)+len(rows)-1
    y = int(rng.integers(h-base_height+1))
    for row in rows:
        rh = max(dims[i][0] for i in row)
        rw = sum(dims[i][1] for i in row)+len(row)-1
        x = int(rng.integers(w-rw+1))
        for i in row:
            dy = int(rng.integers(rh-dims[i][0]+1))
            top, left = (x, y+dy) if transpose else (y+dy, x)
            for (r, c), value in normalized[i].items():
                canvas[top+r, left+c] = value
            x += dims[i][1]+1
        y += rh+1
    return canvas


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen, used_masks = set(), set()
    first_density = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            candidates = [m for m in _MASKS if m.tobytes() not in used_masks]
            if split == "train" and index == 0:
                candidates = [m for m in candidates if _asymmetric(m)]
            if split == "train" and index == 1:
                candidates = [m for m in candidates if int(m.sum()) != first_density]
            # Balance densities, including sparse/disconnected and near-solid stamps.
            density = int(rng.choice(sorted({int(m.sum()) for m in candidates})))
            eligible = [m for m in candidates if int(m.sum()) == density]
            mask = eligible[int(rng.integers(len(eligible)))].copy()
            carrier, accent = map(int, rng.choice(np.arange(1, 9), 2, replace=False))
            contained = split == "train" and index == 1
            n = (2 if index == 0 else 3) if split == "train" and index < 2 else int(
                rng.integers(2, 4) if split == "train" else rng.integers(4, 6))
            for _ in range(160):
                sheets = (_depth_witness(rng) if contained else
                          [_random_sheet(rng, 11 if n <= 3 else 9) for _ in range(n)])
                if any(len(q) <= 2*int(mask.sum()) for q in sheets):
                    continue
                if not _witness_checks(sheets, mask, contained=contained,
                                       phase=split == "train" and index == 0):
                    continue
                inp = _pack(sheets, mask, carrier, accent, rng)
                if _key(inp) in seen:
                    continue
                out = _stamp_reference(sheets, mask, carrier, accent)
                if not np.array_equal(solve(inp), out):
                    raise AssertionError("packed silhouettes or stamp phase changed")
                puzzle[split].append({"input": inp, "output": out})
                seen.add(_key(inp))
                used_masks.add(mask.tobytes())
                first_density = int(mask.sum()) if first_density is None else first_density
                break
            else:
                raise RuntimeError("bounded sheet sampling exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            sheets, mask, (carrier, accent), _ = _read_input(inp)
            depth, origin = _coverage(sheets)
            rr, cc = np.indices(depth.shape)
            motif = mask[(rr+origin[0]) % 3, (cc+origin[1]) % 3]
            assert out.shape == depth.shape
            assert np.array_equal(out == 0, depth == 0), "changed material footprint"
            assert np.array_equal(out == carrier, (depth > 0) & ~motif), "painted carrier gaps"
            assert np.array_equal(out == accent, (depth == 1) & motif), "wrong single-sheet ink"
            assert np.array_equal(out == 9, (depth >= 2) & motif), "wrong overlap ink"
            assert _witness_checks(sheets, mask), "missing observable role witnesses"
            assert not np.array_equal(inp, out)
            information[split].append((sheets, mask, carrier, accent))
    first, second = information["train"][:2]
    assert len(first[0]) == 2 and _asymmetric(first[1])
    assert len(second[0]) == 3 and int(first[1].sum()) != int(second[1].sum())
    assert _witness_checks(first[0], first[1], phase=True)
    assert _witness_checks(second[0], second[1], contained=True)
    assert all(len(v[0]) in (2, 3) for v in information["train"])
    assert all(len(v[0]) in (4, 5) for v in information["test"])
    masks = [v[1].tobytes() for split in ("train", "test") for v in information[split]]
    assert len(set(masks)) == len(masks), "mask novelty shift was lost"


def self_test():
    """Bounded distribution, negative, and metamorphic checks on 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for p in puzzles:
        validate_generated_puzzle(p)
    fingerprints, masks, inputs, outputs, footprints, roles, counts = (set() for _ in range(7))
    holes = missing_corners = both_axis = 0
    for p in puzzles:
        fingerprints.add(tuple(_key(q["input"]) for s in ("train", "test") for q in p[s]))
        for s in ("train", "test"):
            for q in p[s]:
                sheets, mask, role, _ = _read_input(q["input"])
                masks.add(mask.tobytes())
                inputs.add(q["input"].shape)
                outputs.add(q["output"].shape)
                roles.add(role)
                counts.add(len(sheets))
                for sheet in sheets:
                    footprints.add(tuple(sorted(sheet)))
                    rectangle = _rectangle(_bounds(sheet))
                    for gap in _groups(rectangle-sheet):
                        r0, c0, r1, c1 = _bounds(sheet)
                        holes += not any(r in (r0, r1) or c in (c0, c1) for r, c in gap)
                    missing_corners += len(sheet) < len(rectangle)
                origins = [_bounds(sheet)[:2] for sheet in sheets]
                both_axis += (len({r for r, c in origins}) > 1
                              and len({c for r, c in origins}) > 1)
    assert len(fingerprints) == 42
    assert len(masks) >= 110 and len(footprints) >= 200
    assert len(inputs) >= 30 and len(outputs) >= 25 and len(roles) >= 35
    assert counts == {2, 3, 4, 5}
    assert holes >= 10 and missing_corners >= 150 and both_axis >= 150
    # Repacking, recoloring, and whole-grid transpose preserve the rule.
    p = puzzles[0]
    for q in p["train"][:2]:
        sheets, mask, (carrier, accent), _ = _read_input(q["input"])
        repacked = _pack(sheets[::-1], mask, carrier, accent, np.random.default_rng(91))
        assert np.array_equal(solve(repacked), q["output"])
        assert np.array_equal(solve(q["input"].T), q["output"].T)
        perm = np.array([0, 3, 8, 1, 6, 2, 7, 5, 4, 9], dtype=np.int8)
        assert np.array_equal(solve(perm[q["input"]]), perm[q["output"]])
    # Six physical sheets are a legal family member outside the sampling policy.
    # Triplicating both original sheets leaves support/phase unchanged, but every
    # printed cell now has multiple coverage and thus uses the saturated ink.
    q = p["train"][0]
    sheets, mask, (carrier, accent), _ = _read_input(q["input"])
    six = _pack(sheets*3, mask, carrier, accent, np.random.default_rng(92))
    expected = q["output"].copy()
    expected[expected == accent] = 9
    assert np.array_equal(solve(six), expected)
    for seed in (0, 17, 41):
        again = generate_puzzle(seed)
        assert all(np.array_equal(puzzles[seed][s][i][k], again[s][i][k])
                   for s in ("train", "test") for i in range(len(again[s]))
                   for k in ("input", "output"))
    # A contained fragment affects overlap ink despite adding no union material.
    sheets, mask, (carrier, accent), _ = _read_input(p["train"][1]["input"])
    effects = []
    for i, sheet in enumerate(sheets):
        other = [q for j, q in enumerate(sheets) if i != j]
        if sheet <= set().union(*other):
            effects.append(not np.array_equal(_print_field(other, mask, carrier, accent),
                                              p["train"][1]["output"]))
    assert any(effects)
    # Reject two characteristic wrong rules: solid overlap paint and union-only printing.
    for mode in ("solid_overlap", "no_repeat", "crop_phase"):
        damaged = {s: [{k: a.copy() for k, a in q.items()} for q in p[s]] for s in p}
        q = damaged["train"][0]
        sheets, mask, (carrier, accent), _ = _read_input(q["input"])
        depth, (r0, c0) = _coverage(sheets)
        if mode == "solid_overlap":
            q["output"][depth >= 2] = 9
        elif mode == "no_repeat":
            q["output"] = np.where(depth > 0, carrier, 0).astype(np.int8)
            for r, c in np.argwhere(mask):
                q["output"][r-r0, c-c0] = 9
        else:
            rr, cc = np.indices(depth.shape)
            motif = mask[rr % 3, cc % 3]
            q["output"] = np.where(depth == 0, 0, np.where(~motif, carrier,
                                          np.where(depth >= 2, 9, accent))).astype(np.int8)
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {mode}")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(900+10*nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "masks": len(masks), "sheet_footprints": len(footprints),
            "input_shapes": len(inputs), "output_shapes": len(outputs),
            "ordered_palette_roles": len(roles), "holes": holes}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
