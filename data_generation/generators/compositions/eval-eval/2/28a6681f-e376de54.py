"""Wet sluices: conserved liquid activates middle-referenced spill gates.

No parent code or examples are imported. The constructor draws its chosen
complete levels directly; solve parses the returned cells and runs the cascade.
"""
from __future__ import annotations

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _parse(input_grid):
    """Find the U frame, gravity, rooted strokes and occupied edge bay."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    colors = set(map(int, np.unique(grid))) - {bg}
    if len(colors) != 3:
        raise ValueError("a tank uses four distinct semantic colors")
    candidates = []
    for solid in sorted(colors):
        points = np.argwhere(grid == solid)
        low, high = points.min(axis=0), points.max(axis=0) + 1
        r0, c0 = map(int, low)
        r1, c1 = map(int, high)
        if r0 < 1 or c0 < 1 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            continue
        outside = grid.copy()
        outside[r0:r1, c0:c1] = bg
        if np.any(outside != bg):
            continue
        crop = grid[r0:r1, c0:c1]
        for turns in range(4):
            tank = np.rot90(crop, turns).copy()
            height, width = tank.shape
            bottom = height - 1
            if height < 5 or width < 7:
                continue
            if not (np.all(tank[-1] == solid)
                    and np.all(tank[:, 0] == solid)
                    and np.all(tank[:, -1] == solid)
                    and np.all(tank[0, 1:-1] == bg)):
                continue
            for gate in sorted(colors - {solid}):
                water = next(iter(colors - {solid, gate}))
                cols = list(map(int, np.flatnonzero(np.any(tank == gate, axis=0))))
                if len(cols) < 3 or len(cols) % 2 != 1:
                    continue
                if cols[0] <= 1 or cols[-1] >= width - 2 or min(np.diff(cols)) < 2:
                    continue
                lengths = []
                good = True
                for c in cols:
                    rows = np.flatnonzero(tank[:, c] == gate)
                    length = len(rows)
                    if (not 1 <= length <= height - 3
                            or not np.array_equal(rows, np.arange(bottom - length, bottom))
                            or np.any(tank[1:bottom - length, c] != bg)):
                        good = False
                        break
                    lengths.append(length)
                if not good:
                    continue
                reference = lengths[len(cols) // 2]
                if reference < 2:
                    continue
                edges = [0] + cols + [width - 1]
                bays = [(a + 1, b) for a, b in zip(edges, edges[1:])]
                profiles = []
                for left, right in bays:
                    profile = []
                    for c in range(left, right):
                        rows = np.flatnonzero(tank[:bottom, c] == solid)
                        depth = len(rows)
                        if (depth > reference - 2
                                or not np.array_equal(rows, np.arange(bottom - depth, bottom))):
                            good = False
                            break
                        profile.append(depth)
                    if not good or profile[0] or profile[-1]:
                        good = False
                        break
                    profiles.append(tuple(profile))
                if not good:
                    continue
                wet = np.argwhere(tank == water)
                if not len(wet):
                    continue
                occupied = [i for i, (left, right) in enumerate(bays)
                            if np.any((wet[:, 1] >= left) & (wet[:, 1] < right))]
                if occupied not in ([0], [len(bays) - 1]):
                    continue
                candidates.append({
                    "grid": grid, "tank": tank, "rect": (r0, r1, c0, c1),
                    "turns": turns, "background": bg, "solid": solid,
                    "gate": gate, "water": water, "columns": tuple(cols),
                    "lengths": tuple(lengths), "reference": reference,
                    "bays": tuple(bays), "profiles": tuple(profiles),
                    "source_side": 1 if occupied == [0] else -1,
                    "volume": len(wet),
                })
    if len(candidates) != 1:
        raise ValueError("tank must have one unambiguous interpretation")
    return candidates[0]


def _restore(tank, analysis):
    result = analysis["grid"].copy()
    r0, r1, c0, c1 = analysis["rect"]
    result[r0:r1, c0:c1] = np.rot90(tank, -analysis["turns"])
    return result


def _capacity(analysis, bay, depth):
    return sum(max(0, depth - k) for k in analysis["profiles"][bay])


def _cascade(analysis, reference=None, normalize=True, partial=False):
    """Activation and volume allocation are interleaved, not separate passes.

    The partial option is only for testing deliberately simpler alternatives.
    Legal family inputs exhaust volume on a complete horizontal level.
    """
    h = analysis["reference"] if reference is None else int(reference)
    result = analysis["tank"].copy()
    bg, liquid = analysis["background"], analysis["water"]
    result[result == liquid] = bg
    bottom = result.shape[0] - 1
    n = len(analysis["columns"])
    side = analysis["source_side"]
    order = range(n + 1) if side == 1 else range(n, -1, -1)
    remaining = analysis["volume"]
    active, fills = [], []
    for bay in order:
        if remaining == 0:
            break
        gate_index = bay if side == 1 else bay - 1
        has_gate = 0 <= gate_index < n
        depth = h if normalize or not has_gate else analysis["lengths"][gate_index]
        if not 1 <= depth <= bottom - 1:
            raise ValueError("spill height lies outside the tank")
        if has_gate:
            active.append(gate_index)
            if normalize:
                c = analysis["columns"][gate_index]
                result[:bottom, c] = bg
                result[bottom - h:bottom, c] = analysis["gate"]
        left, right = analysis["bays"][bay]
        used = 0
        top_depth = 0
        for d in range(1, depth + 1):
            row = bottom - d
            cols = [c for c in range(left, right) if result[row, c] != analysis["solid"]]
            if not cols:
                continue
            take = min(len(cols), remaining)
            if take < len(cols) and not partial:
                raise ValueError("liquid would leave an ambiguous partial level")
            if side == -1:
                cols.reverse()
            result[row, cols[:take]] = liquid
            remaining -= take
            used += take
            top_depth = d
            if remaining == 0:
                break
        fills.append((bay, used, top_depth))
        if not partial and top_depth <= max(analysis["profiles"][bay]):
            raise ValueError("final pool must submerge its floor features")
    if remaining:
        raise ValueError("volume exceeds the supported cascade capacity")
    return _restore(result, analysis), {"active": tuple(active), "fills": tuple(fills)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible U frame, middle gate and liquid volume."""
    return _cascade(_parse(input_grid))[0]


def _profile(rng, width, h, style):
    heights = np.zeros(width, dtype=int)
    if width <= 2 or style == "flat":
        return heights
    limit = min(h - 2, 3)
    if style == "ridge":
        heights[1:-1] = int(rng.integers(1, limit + 1)) if limit else 0
    elif style == "steps":
        for c in range(1, width - 1):
            heights[c] = min(limit, 1 + c // 2)
    else:
        heights[1:-1] = rng.integers(0, limit + 1, size=width - 2)
    return heights


def _source(rng, tank, bay, volume, bg, solid, water, h, style):
    bottom = tank.shape[0] - 1
    left, right = bay
    eligible = np.zeros(tank.shape, dtype=bool)
    eligible[1:bottom, left:right] = True
    eligible &= tank != solid
    if style != "partly_settled":
        eligible[bottom - h:] = False
    points = np.argwhere(eligible)
    if len(points) < volume:
        raise ValueError("insufficient inlet supply space")
    anchor = points[int(rng.integers(len(points)))]
    dr, dc = abs(points[:, 0] - anchor[0]), abs(points[:, 1] - anchor[1])
    noise = rng.random(len(points))
    if style == "cloud":
        score = dr + dc + noise
    elif style == "band":
        score = 4 * dr + .2 * dc + noise
    elif style == "filaments":
        score = 4 * dc + .2 * dr + noise
    elif style == "partly_settled":
        score = (points[:, 0] < bottom - h) * 10 + noise
    else:
        score = noise
    # A two-cell drop above the spill level both guarantees motion and makes
    # the liquid impossible to confuse with separated, rooted gate strokes.
    row = int(rng.integers(1, bottom - h))
    col = int(rng.integers(left, right - 1))
    mandatory = np.array([(row, col), (row, col + 1)])
    allowed = np.ones(len(points), dtype=bool)
    for point in mandatory:
        allowed &= ~np.all(points == point, axis=1)
    chosen = np.flatnonzero(allowed)[np.argsort(score[allowed])[:volume - 2]]
    selected = np.vstack((mandatory, points[chosen]))
    tank[selected[:, 0], selected[:, 1]] = water


def _make_pair(rng, descriptor):
    n, h = descriptor["count"], descriptor["reference"]
    witness = descriptor["witness"]
    for _ in range(120):
        if witness == "selection":
            widths = [int(rng.integers(4, 7)), 1, 1, 1,
                      int(rng.integers(7, 9)), int(rng.integers(1, 4))]
            lengths = [h - 1, h + 2, h, h + 1, h + 3]
            stop = 2
            final_depth = int(rng.integers(1, h))
        else:
            widths = [int(rng.integers(4, 8))] + [1] * n
            budget = 28 - (n + 2) - sum(widths)
            if budget < 0:
                continue
            additions = int(rng.integers(budget + 1))
            for _ in range(additions):
                i = int(rng.integers(1, n + 1))
                widths[i] += 1
            lengths = list(map(int, rng.integers(1, h + 6, size=n)))
            lengths[n // 2] = h
            if witness == "whole_chain":
                lengths = [h + 2, h, h - 1]
                stop = n
            elif descriptor["split"] == "test":
                stop = int(rng.integers(2, n + 1))
            else:
                stop = int(rng.integers(0, n + 1))
            final_depth = int(rng.integers(1, h + 1))
        width = sum(widths) + n + 2
        if width > 28:
            continue
        styles = ("flat", "ridge", "steps", "jagged")
        profiles = [_profile(rng, w, h, "flat" if witness == "selection"
                             else str(rng.choice(styles))) for w in widths]
        if witness == "whole_chain":
            profiles[0] = _profile(rng, widths[0], h, "steps")
        # Final pools submerge all floor features; no unreachable dry subpocket.
        final_depth = max(final_depth, int(profiles[stop].max()) + 1)
        volume = sum(w * h - int(p.sum())
                     for w, p in zip(widths[:stop], profiles[:stop]))
        volume += sum(max(0, final_depth - int(k)) for k in profiles[stop])
        minimum_height = max(max(lengths) + 3, h + 2 + (volume + widths[0] - 1) // widths[0])
        if minimum_height > 28:
            continue
        height = int(rng.integers(minimum_height, min(28, minimum_height + 5) + 1))
        bg, solid, gate, water = map(int, rng.choice(10, size=4, replace=False))
        tank = np.full((height, width), bg, dtype=np.int8)
        tank[:, 0] = tank[:, -1] = solid
        tank[-1] = solid
        bottom = height - 1
        bays, columns = [], []
        left = 1
        for i, (w, profile) in enumerate(zip(widths, profiles)):
            bays.append((left, left + w))
            for c, k in zip(range(left, left + w), profile):
                tank[bottom - int(k):bottom, c] = solid
            left += w
            if i < n:
                columns.append(left)
                tank[bottom - lengths[i]:bottom, left] = gate
                left += 1
        # Constructor ground truth: draw the chosen wet prefix and full levels.
        output = tank.copy()
        for i in range(min(stop + 1, n)):
            c = columns[i]
            output[:bottom, c] = bg
            output[bottom - h:bottom, c] = gate
        for i in range(stop + 1):
            d = h if i < stop else final_depth
            l, r = bays[i]
            for row in range(bottom - d, bottom):
                for c in range(l, r):
                    if output[row, c] == bg:
                        output[row, c] = water
        style = str(rng.choice(("cloud", "scattered", "band", "filaments", "partly_settled")))
        _source(rng, tank, bays[0], volume, bg, solid, water, h, style)
        if descriptor["mirror"]:
            tank, output = np.fliplr(tank), np.fliplr(output)
        tank, output = np.rot90(tank, descriptor["rotation"]), np.rot90(output, descriptor["rotation"])
        # Asymmetric blank margins vary placement as well as canvas shape.
        margins = []
        for dimension in tank.shape:
            spare = 30 - dimension
            before = int(rng.integers(1, min(4, spare - 1) + 1))
            after = int(rng.integers(1, min(4, spare - before) + 1))
            margins.append((before, after))
        pair = {"input": np.pad(tank, margins, constant_values=bg),
                "output": np.pad(output, margins, constant_values=bg)}
        # This is a consistency check against the input parser, not label creation.
        if not np.array_equal(solve(pair["input"]), pair["output"]):
            raise AssertionError("constructor and input-only cascade disagree")
        return pair
    raise RuntimeError("could not fit a cascade after bounded attempts")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    base_rotation = int(rng.integers(4))
    base_mirror = bool(rng.integers(2))
    references = list(map(int, rng.choice(np.arange(3, 8), size=2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = ("selection" if index == 0 else "whole_chain" if index == 1
                       else "random") if split == "train" else "random"
            n = 5 if witness == "selection" else 3 if witness == "whole_chain" else int(
                rng.choice((5, 7, 9) if split == "test" else (3, 5, 7)))
            reference = references[index] if split == "train" and index < 2 else int(rng.integers(2, 8))
            rotation = ((base_rotation + index) % 4 if split == "train" and index < 2
                        else (base_rotation + 2 + index) % 4 if split == "test"
                        else int(rng.integers(4)))
            mirror = (base_mirror != bool(index)) if split == "train" and index < 2 else bool(rng.integers(2))
            descriptor = {"count": n, "reference": reference, "witness": witness,
                          "split": split, "rotation": rotation, "mirror": mirror}
            for _ in range(30):
                pair = _make_pair(rng, descriptor)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not draw a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out):
    a = _parse(inp)
    expected, stats = _cascade(a)
    assert np.array_equal(expected, out), "wrong cascade output"
    assert np.any(inp != out), "every generated pair must change"
    assert np.array_equal(inp == a["solid"], out == a["solid"]), "rigid terrain moved"
    assert np.count_nonzero(inp == a["water"]) == np.count_nonzero(out == a["water"])
    result = np.rot90(np.asarray(out)[a["rect"][0]:a["rect"][1],
                                       a["rect"][2]:a["rect"][3]], a["turns"])
    bottom = result.shape[0] - 1
    active = set(stats["active"])
    changed_gates = []
    for i, c in enumerate(a["columns"]):
        span = a["reference"] if i in active else a["lengths"][i]
        assert np.array_equal(np.flatnonzero(result[:, c] == a["gate"]),
                              np.arange(bottom - span, bottom))
        if span != a["lengths"][i]:
            changed_gates.append(i)
    fills = stats["fills"]
    assert sum(v for _, v, _ in fills) == a["volume"]
    for position, (bay, volume, depth) in enumerate(fills):
        assert volume == _capacity(a, bay, depth)
        assert depth > max(a["profiles"][bay]), "floor features must be submerged"
        if position < len(fills) - 1:
            assert depth == a["reference"]
        left, right = a["bays"][bay]
        mask = result[:, left:right] == a["water"]
        expected_mask = (np.arange(result.shape[0])[:, None] >= bottom - depth)
        expected_mask = np.broadcast_to(expected_mask, mask.shape).copy()
        expected_mask &= result[:, left:right] != a["solid"]
        assert np.array_equal(mask, expected_mask), "incomplete or unsupported pool"
    widths = tuple(right - left for left, right in a["bays"])
    keys = np.array(a["columns"])
    distances = abs(2 * keys - keys[0] - keys[-1])
    nearest = np.flatnonzero(distances == distances.min())
    geometric_witness = (len(nearest) == 1 and nearest[0] != len(keys) // 2
                         and a["lengths"][int(nearest[0])] != a["reference"])
    try:
        static, _ = _cascade(a, normalize=False, partial=True)
        static_water_differs = bool(np.any((static == a["water"]) != (out == a["water"])))
    except ValueError:
        # In some examples the unadjusted gates cannot retain the conserved
        # supply within the family's spill ceiling at all.
        static_water_differs = True
    return {"count": len(keys), "reference": a["reference"], "lengths": a["lengths"],
            "widths": widths, "profiles": a["profiles"], "volume": a["volume"],
            "active": len(active), "dry": len(keys) - len(active),
            "wet_changes": len(changed_gates), "depths": tuple(d for _, _, d in fills),
            "extends": sum(a["lengths"][i] < a["reference"] for i in active),
            "trims": sum(a["lengths"][i] > a["reference"] for i in active),
            "nonmedian": sorted(a["lengths"])[len(keys) // 2] != a["reference"],
            "geometric_witness": bool(geometric_witness),
            "static_water_differs": static_water_differs,
            "gravity": a["turns"], "source_side": a["source_side"],
            "shape": np.asarray(inp).shape,
            "colors": (a["background"], a["solid"], a["gate"], a["water"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    records, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            records[split].append(_features(inp, out))
    train = records["train"]
    assert len({r["count"] for r in train}) >= 2
    assert len({r["reference"] for r in train}) >= 2
    assert len({r["gravity"] for r in train}) >= 2
    assert {r["source_side"] for r in train} == {-1, 1}
    assert any(r["extends"] and r["trims"] for r in train)
    assert any(r["dry"] and r["wet_changes"] for r in train)
    assert any(r["dry"] == 0 for r in train)
    assert any(r["nonmedian"] and r["geometric_witness"] for r in train)
    assert any(r["static_water_differs"] for r in train), "B must alter liquid, not just gate pixels"
    assert any(any(k for p in r["profiles"] for k in p) for r in train)
    assert any(all(k == 0 for p in r["profiles"] for k in p) for r in train)


def self_test():
    """Bounded 42-seed campaign, semantic counterfactuals and equivariances."""
    shapes, structures, volumes, references, counts = set(), set(), set(), set(), set()
    gravities, sides, frontiers, profiles, signatures = set(), set(), set(), set(), set()
    palettes = [set() for _ in range(4)]
    records = []
    rng = np.random.default_rng(72031)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp, out)
                records.append(f)
                shapes.add(f["shape"])
                volumes.add(f["volume"])
                references.add(f["reference"])
                counts.add(f["count"])
                gravities.add(f["gravity"])
                sides.add(f["source_side"])
                frontiers.add((f["count"], f["active"]))
                profiles.add(f["profiles"])
                for role, color in enumerate(f["colors"]):
                    palettes[role].add(color)
                a = _parse(inp)
                normalized = np.where(inp == a["background"], 0,
                             np.where(inp == a["solid"], 1,
                             np.where(inp == a["gate"], 2, 3)))
                structures.add(_key(normalized))
        # The first witness tests role independence, geometry and source shape.
        pair = puzzle["train"][0]
        inp, out = pair["input"], pair["output"]
        permutation = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        for turns in range(4):
            for mirror in (False, True):
                x, y = np.rot90(inp, turns), np.rot90(out, turns)
                if mirror:
                    x, y = np.fliplr(x), np.fliplr(y)
                assert np.array_equal(solve(x), y)
        a = _parse(inp)
        tank = a["tank"].copy()
        tank[tank == a["water"]] = a["background"]
        bay = 0 if a["source_side"] == 1 else len(a["bays"]) - 1
        left, right = a["bays"][bay]
        eligible = np.zeros(tank.shape, dtype=bool)
        eligible[1:tank.shape[0] - a["reference"] - 1, left:right] = True
        points = np.argwhere(eligible & (tank == a["background"]))
        chosen = points[rng.choice(len(points), size=a["volume"], replace=False)]
        tank[chosen[:, 0], chosen[:, 1]] = a["water"]
        assert np.array_equal(solve(_restore(tank, a)), out), "source shape must not be copied"
        # Less volume stops before the second gate: A controls B's activation.
        small_volume = _capacity(a, bay, a["reference"] - 1)
        tank[tank == a["water"]] = a["background"]
        tank[chosen[:small_volume, 0], chosen[:small_volume, 1]] = a["water"]
        smaller = _parse(_restore(tank, a))
        small_output, stats = _cascade(smaller)
        assert len(stats["active"]) == 1
        assert np.count_nonzero(small_output == a["water"]) == small_volume
        # The second wet gate was corrected in the original, and now stays dry.
        second = 1 if a["source_side"] == 1 else len(a["columns"]) - 2
        assert a["lengths"][second] != a["reference"]
        c = a["columns"][second]
        dry_tank = np.rot90(small_output[a["rect"][0]:a["rect"][1],
                                        a["rect"][2]:a["rect"][3]], a["turns"])
        assert np.count_nonzero(dry_tank[:, c] == a["gate"]) == a["lengths"][second]
        # Numerical-median and geometric-midpoint references change this answer.
        for alternate in (sorted(a["lengths"])[len(a["columns"]) // 2],
                          a["lengths"][int(np.argmin(abs(2 * np.array(a["columns"])
                                                       - a["columns"][0] - a["columns"][-1]))) ]):
            altered, _ = _cascade(a, reference=alternate, partial=True)
            assert not np.array_equal(altered, out)
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 40 and len(volumes) >= 35
    assert counts == {3, 5, 7, 9} and references == set(range(2, 8))
    assert gravities == set(range(4)) and sides == {-1, 1}
    assert len(frontiers) >= 12 and len(profiles) >= 80
    assert all(len(p) >= 8 for p in palettes)
    assert sum(r["dry"] > 0 for r in records) >= 80
    assert sum(r["dry"] == 0 for r in records) >= 42
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(830 + train * 10 + test, train, test)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0)
    pair = broken["train"][0]
    a = _parse(pair["input"])
    point = np.argwhere(pair["output"] == a["water"])[0]
    pair["output"][tuple(point)] = a["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted lost liquid")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, cascade invariants and legal counts")
