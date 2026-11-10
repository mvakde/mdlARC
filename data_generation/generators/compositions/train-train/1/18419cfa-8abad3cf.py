"""Mirror defect ledger: reflection matching supplies unary square areas.

Only NumPy and the standard library are used. The public solver has no access
to the seed, sampling plans, or training examples.
"""
from __future__ import annotations

from math import isqrt
import json
import numpy as np


HOST_SEEDS = (0, 96473853, 905905772, 562430053, 916867018, 483438836,
              694792798, 788311259, 731517942, 477988279, 190078516,
              579764337, 962597673, 98086836, 120473189, 989878261,
              821754903, 827983385, 517490328, 529168337, 739114572,
              5858951, 219531413, 658302504, 35823651, 242176011,
              209535638, 243089433, 305199911, 18667344, 56103698,
              298369638, 846108561, 225338963, 862807515, 556802565,
              395462818, 263519193, 404272903, 174080995, 445564676,
              17896690)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return grid.astype(np.int8, copy=False)


def _contour(cross, axis, depth):
    """Canonical left/right completion frame, including centered side tabs."""
    frame = np.zeros((cross + 2, axis + 2 + 2 * depth), dtype=bool)
    left, right = depth, depth + axis + 1
    frame[0, left:right + 1] = True
    frame[-1, left:right + 1] = True
    frame[:, left] = True
    frame[:, right] = True
    # Odd cross dimensions use one tab row; even dimensions use two.
    lo, hi = (cross + 1) // 2, (cross + 2) // 2
    frame[lo:hi + 1, :left] = True
    frame[lo:hi + 1, right + 1:] = True
    return frame


def _read_input(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    leaders = np.flatnonzero(counts == counts.max())
    if len(leaders) != 1:
        raise ValueError("background must be uniquely most frequent")
    background = int(leaders[0])
    candidates = []
    for color in np.flatnonzero(counts):
        color = int(color)
        if color == background:
            continue
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0) + 1
        box = grid[r0:r1, c0:c1]
        for vertical in (True, False):
            canonical = box if vertical else box.T
            cross = canonical.shape[0] - 2
            for depth in (1, 2):
                axis = canonical.shape[1] - 2 - 2 * depth
                if not (5 <= cross <= 24 and 6 <= axis <= 24):
                    continue
                mask = _contour(cross, axis, depth)
                if not np.array_equal(canonical == color, mask):
                    continue
                local_inside = np.zeros(canonical.shape, dtype=bool)
                local_inside[1:-1, depth + 1:depth + 1 + axis] = True
                owned = mask | local_inside
                if np.any(canonical[~owned] != background):
                    continue
                exterior = np.ones(grid.shape, dtype=bool)
                exterior[r0:r1, c0:c1] = False
                if np.any(grid[exterior] != background):
                    continue
                interior = canonical[1:-1, depth + 1:depth + 1 + axis]
                candidates.append((color, vertical, cross, axis, depth,
                                   tuple(map(int, (r0, r1, c0, c1))), interior))
    if len(candidates) != 1:
        raise ValueError("expected one uniquely marked enclosing frame")
    frame, vertical, cross, axis, depth, box, interior = candidates[0]
    payload = sorted(int(c) for c in np.unique(interior) if c != background)
    if not 2 <= len(payload) <= 4 or frame in payload:
        raise ValueError("expected two to four payload colors")
    reflected = interior[:, ::-1]
    strict = axis // 2
    records = []
    for color in payload:
        mask = interior == color
        defects = mask & (reflected != color)
        left = int(defects[:, :strict].sum())
        right = int(defects[:, axis - strict:].sum())
        count = left + right
        side = isqrt(count)
        if count == 0 or side * side != count or not 1 <= side <= 5:
            raise ValueError("each color's mirror defects must encode a square")
        if left == right:
            raise ValueError("defect side majorities must be strict")
        records.append({"color": color, "side": side, "left": left,
                        "right": right, "defects": defects,
                        "matches": int((mask & (reflected == color))[:, :strict].sum()),
                        "fixed": int(mask[:, strict].sum()) if axis % 2 else 0})
    if len({r["side"] for r in records}) != len(records):
        raise ValueError("decoded square sides must be distinct")
    return {"background": background, "frame": frame, "vertical": vertical,
            "cross": cross, "axis": axis, "depth": depth, "box": box,
            "interior": interior, "records": records,
            "histogram": tuple(map(int, counts))}


def _ledger(background, frame, vertical, records):
    records = sorted(records, key=lambda r: r["side"])
    # A left-heavy source produces a square on the right of the spine.
    left_width = max((r["side"] for r in records if r["right"] > r["left"]), default=0)
    right_width = max((r["side"] for r in records if r["left"] > r["right"]), default=0)
    length = sum(r["side"] for r in records) + len(records) - 1
    result = np.full((length, left_width + 1 + right_width), background, dtype=np.int8)
    result[:, left_width] = frame
    top = 0
    for record in records:
        side = record["side"]
        col = left_width + 1 if record["left"] > record["right"] else left_width - side
        result[top:top + side, col:col + side] = record["color"]
        top += side + 1
    return result if vertical else result.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count exact color-sensitive reflection defects and draw their ledger."""
    a = _read_input(input_grid)
    return _ledger(a["background"], a["frame"], a["vertical"], a["records"])


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _signature(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                              for p in puzzle[split])) for split in ("train", "test"))


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _geometry(rng, needed, pixels, odd=None):
    candidates = []
    for cross in range(7, 25):
        for axis in range(8, 25):
            if odd is not None and axis % 2 != int(odd):
                continue
            for depth in (1, 2):
                if axis + 2 + 2 * depth > 30 or cross * (axis // 2) < needed + 4:
                    continue
                shape = (cross + 2, axis + 2 + 2 * depth)
                structural = int(_contour(cross, axis, depth).sum())
                # Background is identifiable even when the frame touches edges.
                if shape[0] * shape[1] - structural - pixels <= max(structural, pixels // 2):
                    continue
                candidates.append((cross, axis, depth))
    if not candidates:
        raise RuntimeError("no feasible frame geometry")
    return candidates[int(rng.integers(len(candidates)))]


def _rank_slots(rng, cross, half, mode):
    rr, cc = np.indices((cross, half))
    noise = rng.random((cross, half))
    r, c = int(rng.integers(cross)), int(rng.integers(half))
    if mode == "patch":
        score = (rr - r) ** 2 + (cc - c) ** 2 + noise
    elif mode == "bars":
        score = abs(rr - r) * (half + 1) + abs(cc - c) + noise
    elif mode == "fragments":
        r2, c2 = int(rng.integers(cross)), int(rng.integers(half))
        score = np.minimum((rr-r)**2 + (cc-c)**2, (rr-r2)**2 + (cc-c2)**2) + noise * 3
    elif mode == "checker":
        score = ((rr + cc) % 2) * 2 + noise * 3
    else:
        score = noise
    return score.ravel()


def _margins(rng, shape):
    result = []
    for dim in shape:
        total = int(rng.integers(min(5, 30 - dim) + 1))
        before = int(rng.integers(total + 1))
        result.append((before, total-before))
    return result


def _paint(rng, geometry, background, frame, plan, vertical, mode, margins=None,
           force_facing=False):
    cross, axis, depth = geometry
    half = axis // 2
    nslots = cross * half
    left = np.full(nslots, background, dtype=np.int8)
    right = left.copy()  # Same orbit indexing as left; reversed when painted.
    order = list(rng.permutation(len(plan)))
    for index in order:
        item = plan[index]
        score = _rank_slots(rng, cross, half, mode)
        available = np.flatnonzero((left == background) & (right == background))
        chosen = available[np.argsort(score[available], kind="stable")[:item["matched"]]]
        left[chosen] = right[chosen] = item["color"]
    reserved = {}
    if force_facing:
        # Occupancy matching must visibly differ from COLOR matching.
        li = next(i for i, p in enumerate(plan) if p["left"] > p["right"])
        ri = next(i for i, p in enumerate(plan) if p["right"] > p["left"])
        slot = int(rng.choice(np.flatnonzero((left == background) & (right == background))))
        left[slot], right[slot] = plan[li]["color"], plan[ri]["color"]
        reserved[(0, li)] = reserved[(1, ri)] = 1
    # Unmatched cells may face background or a DIFFERENT color. They may not
    # accidentally create an additional same-color mirrored pair.
    for which in rng.permutation(2):
        target, other = (left, right) if which == 0 else (right, left)
        for index in rng.permutation(len(plan)):
            item = plan[int(index)]
            count = (item["left"] if which == 0 else item["right"]) - reserved.get((int(which), int(index)), 0)
            score = _rank_slots(rng, cross, half, mode)
            available = np.flatnonzero((target == background) & (other != item["color"]))
            # Mix opposite-color encounters with empty reflection partners.
            if rng.random() < 0.65:
                score = score - (other != background) * (float(score.max()) + 1) * 0.6
            chosen = available[np.argsort(score[available], kind="stable")[:count]]
            if len(chosen) != count:
                raise RuntimeError("insufficient unmatched orbit capacity")
            target[chosen] = item["color"]
    interior = np.full((cross, axis), background, dtype=np.int8)
    interior[:, :half] = left.reshape(cross, half)
    interior[:, axis-half:] = right.reshape(cross, half)[:, ::-1]
    fixed_total = sum(item["fixed"] for item in plan)
    if fixed_total:
        if not axis % 2 or fixed_total > cross:
            raise RuntimeError("invalid fixed-point plan")
        rows = rng.permutation(cross)
        offset = 0
        for item in plan:
            count = item["fixed"]
            interior[rows[offset:offset + count], half] = item["color"]
            offset += count
    contour = _contour(cross, axis, depth)
    canonical = np.full(contour.shape, background, dtype=np.int8)
    canonical[contour] = frame
    canonical[1:-1, depth + 1:depth + 1 + axis] = interior
    # Independent margins, including edge contact; never exceed 30.
    if margins is None:
        margins = _margins(rng, canonical.shape)
    inp = np.pad(canonical, margins, constant_values=background)
    if not vertical:
        inp = inp.T.copy()
    # Generation labels come from the planned defect areas, not solve().
    out = _ledger(background, frame, vertical, plan)
    return {"input": inp, "output": out}


def _make_plan(rng, sides, colors, matched=None, fixed=None, deltas=None, signs=None):
    n = len(sides)
    if signs is None:
        signs = list(rng.choice([-1, 1], n))
        signs[0], signs[1] = -1, 1
        rng.shuffle(signs)
    plan = []
    for i, (side, color) in enumerate(zip(sides, colors)):
        area = side * side
        choices = list(range(1 if area % 2 else 2, area + 1, 2))
        delta = int(rng.choice(choices)) if deltas is None else deltas[i]
        left = (area + int(signs[i]) * delta) // 2
        plan.append({"color": int(color), "side": int(side), "left": left,
                     "right": area-left, "matched": int(rng.integers(1, 8)) if matched is None else matched[i],
                     "fixed": 0 if fixed is None else fixed[i]})
    return plan


def _requirements(plan):
    # Reserving one orbit per defect is conservative; different colors can
    # subsequently share an orbit on opposite sides without changing counts.
    defects = sum(p["side"] ** 2 for p in plan)
    matched = sum(p["matched"] for p in plan)
    return defects + matched, defects + 2 * matched + sum(p["fixed"] for p in plan)


def _configuration(records):
    return tuple(sorted((r["side"], r["left"] > r["right"]) for r in records))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    background, frame = palette[:2]
    n = int(rng.integers(2, 4))
    low = [2, int(rng.choice([1, 3]))] if n == 2 else [1, 2, 3]
    rng.shuffle(low)
    high = [s + 2 for s in low]
    colors = palette[2:2+n]
    # Both witnesses have identical per-color counts in corresponding halves,
    # and identical fixed-axis cells. Only the mirror coincidences differ.
    k_high = [int(rng.integers(1, 5)) for _ in low]
    k_low = [k + ((s+2)**2-s*s)//2 for k, s in zip(k_high, low)]
    deltas = [1 if s % 2 else 2 for s in low]
    fixed = [int(rng.integers(0, 3)) for _ in low]
    odd = bool(rng.integers(2))
    if not odd:
        fixed = [0] * n
    elif not any(fixed):
        fixed[int(rng.integers(n))] = 1
    signs = [-1, 1] + [int(rng.choice([-1, 1])) for _ in range(n-2)]
    rng.shuffle(signs)
    first = _make_plan(rng, low, colors, k_low, fixed, deltas, signs)
    second = _make_plan(rng, high, colors, k_high, fixed, deltas, signs)
    needs = [_requirements(p) for p in (first, second)]
    geometry = _geometry(rng, max(x[0] for x in needs), max(x[1] for x in needs), odd)
    first_vertical = bool(rng.integers(2))
    witness_margins = _margins(rng, _contour(*geometry).shape)
    puzzle = {"train": [], "test": []}
    modes = ("scatter", "patch", "bars", "fragments", "checker")
    for plan, vertical in ((first, first_vertical), (second, not first_vertical)):
        puzzle["train"].append(_paint(rng, geometry, background, frame, plan,
                                      vertical, str(rng.choice(modes)), witness_margins,
                                      force_facing=(plan is second)))
    used = {_key(p["input"]) for p in puzzle["train"]}
    train_sides = sorted(set(low + high))
    flexible_palette = bool(num_train >= 3 and rng.random() < 0.6)
    test_outputs = set()
    train_configurations = {_configuration(first), _configuration(second)}
    test_configurations = set()
    train_parities = {int(odd)}
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for i in range(count):
            for _ in range(40):
                local_palette = [int(c) for c in rng.permutation(10)] if flexible_palette else palette
                bg, fc = local_palette[:2]
                permitted = list(range(1, 6)) if split == "train" else train_sides
                color_count = int(rng.integers(2, min(4, len(permitted)) + 1))
                sides = [int(s) for s in rng.choice(permitted, color_count, replace=False)]
                chosen_colors = [int(c) for c in rng.choice(local_palette[2:], color_count, replace=False)]
                plan = _make_plan(rng, sides, chosen_colors)
                configuration = _configuration(plan)
                if split == "test" and (configuration in train_configurations
                                        or configuration in test_configurations):
                    continue
                odd_axis = (bool(rng.choice(sorted(train_parities))) if split == "test"
                            else bool(rng.integers(2)))
                if odd_axis:
                    for item in plan:
                        item["fixed"] = int(rng.integers(0, 3))
                    if not any(item["fixed"] for item in plan):
                        plan[int(rng.integers(len(plan)))]["fixed"] = 1
                required, pixels = _requirements(plan)
                geo = _geometry(rng, required, pixels, odd_axis)
                pair = _paint(rng, geo, bg, fc, plan, bool(rng.integers(2)), str(rng.choice(modes)))
                key, okey = _key(pair["input"]), _key(pair["output"])
                if key not in used and (split == "train" or okey not in test_outputs):
                    break
            else:
                raise RuntimeError("unique pair sampling exhausted")
            used.add(key)
            if split == "test":
                test_outputs.add(okey)
                test_configurations.add(configuration)
            else:
                train_sides = sorted(set(train_sides + sides))
                train_configurations.add(configuration)
                train_parities.add(int(odd_axis))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _witness_key(a):
    interior = a["interior"]
    half = a["axis"] // 2
    return tuple((r["color"], int((interior[:, :half] == r["color"]).sum()),
                  int((interior[:, a["axis"]-half:] == r["color"]).sum()), r["fixed"])
                 for r in a["records"])


def validate_generated_puzzle(puzzle):
    """Check geometry, reflection accounting, square reconstruction and teaching."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    used = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in used, "duplicate input"
            used.add(key)
            a = _read_input(inp)
            analyses[split].append(a)
            assert np.array_equal(solve(inp), out), "input-only solver mismatch"
            # A second accounting route enumerates color-sensitive 2-cell
            # orbits, rather than using the solver's full reflected mask.
            for r in a["records"]:
                l = q = matches = 0
                inside = a["interior"]
                for row in range(a["cross"]):
                    for col in range(a["axis"] // 2):
                        x, y = int(inside[row, col]), int(inside[row, -1-col])
                        if x == y == r["color"]:
                            matches += 1
                        elif x != y:
                            l += x == r["color"]
                            q += y == r["color"]
                assert (l, q, matches) == (r["left"], r["right"], r["matches"])
                assert int((inside == r["color"]).sum()) == l+q+2*matches+r["fixed"]
                assert int((out == r["color"]).sum()) == (l+q) == r["side"]**2
                canonical = out if a["vertical"] else out.T
                coords = np.argwhere(canonical == r["color"])
                assert tuple(coords.max(axis=0)-coords.min(axis=0)+1) == (r["side"], r["side"])
                assert matches >= 1, "every color must include cancellable material"
            assert {r["left"] > r["right"] for r in a["records"]} == {True, False}
            assert set(np.unique(out)) == {a["background"], a["frame"]} | {r["color"] for r in a["records"]}
    train = analyses["train"]
    assert {a["vertical"] for a in train} == {True, False}
    # The two guaranteed witnesses refute every histogram-only or signed
    # half-count-only size decoder, including on the minimum 2/1 request.
    witnesses = [(a, b) for i, a in enumerate(train) for b in train[i+1:]
                 if _witness_key(a) == _witness_key(b)
                 and a["histogram"] == b["histogram"]
                 and [(r["color"], r["side"]) for r in a["records"]]
                 != [(r["color"], r["side"]) for r in b["records"]]]
    assert witnesses, "missing same-count/different-matching demonstration"
    assert any(r["left"] and r["right"] and r["side"] > 1
               for a in train for r in a["records"]), "net difference shortcut not refuted"
    def has_opposing_colors(a):
        half = a["axis"] // 2
        x, y = a["interior"][:, :half], a["interior"][:, ::-1][:, :half]
        return bool(((x != y) & (x != a["background"]) & (y != a["background"])).any())
    assert any(has_opposing_colors(a) for a in train), "occupancy-only matching not refuted"
    known_sides = {r["side"] for a in train for r in a["records"]}
    known_parities = {a["axis"] % 2 for a in train}
    known_configurations = {_configuration(a["records"]) for a in train}
    seen_test_configurations = set()
    for a in analyses["test"]:
        assert {r["side"] for r in a["records"]} <= known_sides
        assert a["axis"] % 2 in known_parities
        config = _configuration(a["records"])
        assert config not in known_configurations | seen_test_configurations
        seen_test_configurations.add(config)


def self_test():
    """Bounded audit over the 42 host seeds, plus count and symmetry probes."""
    shapes, outputs, geometries, sidesets, palettes = set(), set(), set(), set(), set()
    structures, signatures = set(), set()
    counts, orientations, parities, depths = set(), set(), set(), set()
    fixed_colors = facing_colors = split_colors = edge_frames = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _read_input(inp)
                shapes.add(inp.shape)
                outputs.add(out.shape)
                geometries.add((a["cross"], a["axis"], a["depth"], a["vertical"]))
                sidesets.add(tuple(sorted(r["side"] for r in a["records"])))
                palettes.add((a["background"], a["frame"], tuple(r["color"] for r in a["records"])))
                counts.add(len(a["records"]))
                orientations.add(a["vertical"])
                parities.add(a["axis"] % 2)
                depths.add(a["depth"])
                normalized = np.full(inp.shape, -1, dtype=np.int8)
                normalized[inp == a["background"]] = 0
                normalized[inp == a["frame"]] = 1
                for r in a["records"]:
                    normalized[inp == r["color"]] = r["side"] + 1
                    fixed_colors += r["fixed"] > 0
                    split_colors += r["left"] > 0 and r["right"] > 0
                structures.add(_key(normalized))
                inside = a["interior"]
                h = a["axis"] // 2
                x, y = inside[:, :h], inside[:, ::-1][:, :h]
                facing_colors += int(((x != y) & (x != a["background"]) & (y != a["background"])).sum())
                r0, r1, c0, c1 = a["box"]
                edge_frames += r0 == 0 or c0 == 0 or r1 == inp.shape[0] or c1 == inp.shape[1]
                # Transpose preserves reflection orbits and transposes ledger.
                assert np.array_equal(solve(inp.T), out.T)
                # Arbitrary palette permutations preserve the rule.
                mapping = np.roll(np.arange(10, dtype=np.int8), (int(seed) % 9) + 1)
                assert np.array_equal(solve(mapping[inp]), mapping[out])
                # Solver must leave the input untouched.
                before = inp.copy()
                solve(inp)
                assert np.array_equal(inp, before)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(outputs) >= 15 and len(geometries) >= 60
    assert len(sidesets) >= 12 and len(palettes) >= 60 and len(structures) >= 180
    assert counts == {2, 3, 4} and orientations == {True, False}
    assert parities == {0, 1} and depths == {1, 2}
    assert fixed_colors > 40 and facing_colors > 100 and split_colors > 100 and edge_frames > 30
    # Exercise supported smaller frames beyond the current sampler's range.
    probe_rng = np.random.default_rng(771)
    for axis in (6, 7):
        plan = _make_plan(probe_rng, [1, 2], [2, 3], matched=[1, 1],
                          fixed=[1, 1] if axis % 2 else [0, 0],
                          deltas=[1, 2], signs=[1, -1])
        for vertical in (True, False):
            p = _paint(probe_rng, (5, axis, 1), 0, 8, plan, vertical,
                       "scatter", margins=[(0, 0), (0, 0)], force_facing=True)
            assert np.array_equal(solve(p["input"].tolist()), p["output"])
    # All legal explicit counts, with the strongest evidence still at 2/1.
    for train in range(2, 11):
        for test in range(1, min(4, train-1) + 1):
            p = generate_puzzle(8073 + 10*train + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(outputs), "frame_geometries": len(geometries),
            "side_sets": len(sidesets), "palette_assignments": len(palettes),
            "normalized_inputs": len(structures), "fixed_point_colors": fixed_colors,
            "opposite_color_encounters": facing_colors, "two_sided_defect_colors": split_colors,
            "edge_frames": edge_frames}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
