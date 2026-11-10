"""Perforated middle master: anchored stencil transfer with damage diagnosis.

Only NumPy and the standard library are used. Generation paints from explicit
construction masks; solve reconstructs those masks from the input's capped bars.
No parent modules, files, seeds, examples, caches, or learned state enter solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _runs(values):
    result = []
    for value in map(int, values):
        if not result or value != result[-1][-1] + 1:
            result.append([value])
        else:
            result[-1].append(value)
    return result


def _decode(input_grid):
    """Find the unique cap color, inward direction, slots, extents and master."""
    original = _grid(input_grid)
    field = int(original[0, 0])
    if any(int(original[y, x]) != field
           for y, x in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("the four corners must show the field color")
    candidates = []
    for turns in range(4):
        a = np.rot90(original, turns)
        for accent in map(int, np.unique(a)):
            if accent == field:
                continue
            points = np.argwhere(a == accent)
            if len(set(map(int, points[:, 1]))) != 1:
                continue
            cap_x = int(points[0, 1])
            if not 1 <= cap_x < a.shape[1] - 3:
                continue
            groups = _runs(points[:, 0])
            if len(groups) not in (3, 5, 7):
                continue
            thick = len(groups[0])
            if thick not in (3, 4, 5) or any(len(g) != thick for g in groups):
                continue
            slots, colors, lengths = [], [], []
            valid = True
            for group in groups:
                y = group[0]
                front = a[y:y + thick, cap_x + 1]
                color = int(front[0])
                if color in (field, accent) or not np.all(front == color):
                    valid = False
                    break
                body = a[y:y + thick, cap_x + 1:]
                if np.any((body != field) & (body != color)):
                    valid = False
                    break
                full = np.flatnonzero(np.all(body == color, axis=0))
                if not len(full) or int(full[-1]) < 2:
                    valid = False
                    break
                extent = int(full[-1]) + 1
                slots.append(y)
                colors.append(color)
                lengths.append(extent)
            if not valid:
                continue
            middle = len(slots) // 2
            length = lengths[middle]
            if not 5 <= length <= 16:
                continue
            master = a[slots[middle]:slots[middle] + thick,
                       cap_x + 1:cap_x + 1 + length] == colors[middle]
            # Clean side rails and end cross-sections make the stencil legible.
            if (master.all() or not master[[0, -1], :].all()
                    or not master[:, [0, -1]].all()):
                continue
            if np.any(a[slots[middle]:slots[middle] + thick,
                        cap_x + 1 + length:] != field):
                continue
            candidates.append({"grid": a, "turns": turns, "field": field,
                               "accent": accent, "cap_x": cap_x,
                               "thick": thick, "slots": tuple(slots),
                               "colors": tuple(colors), "lengths": tuple(lengths),
                               "length": length, "master": master.copy()})
    if len(candidates) != 1:
        raise ValueError("input must have one readable capped-bar interpretation")
    return candidates[0]


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            result |= padded[dy:dy + h, dx:dx + w]
    return result


def _masks(a, spans=None, rectangular=False):
    """Input-derived nominal material, observed domains, and preserved caps."""
    grid = a["grid"]
    target = np.zeros(grid.shape, dtype=bool)
    observed = np.zeros_like(target)
    caps = np.zeros_like(target)
    canvas = np.full(grid.shape, a["field"], dtype=np.int8)
    x, thick, length = a["cap_x"], a["thick"], a["length"]
    spans = [length] * len(a["slots"]) if spans is None else spans
    for y, color, extent, span in zip(a["slots"], a["colors"], a["lengths"], spans):
        template = np.ones((thick, span), dtype=bool)
        overlap = min(span, length)
        if not rectangular:
            template[:, :overlap] = a["master"][:, :overlap]
        target[y:y + thick, x + 1:x + 1 + span] = template
        region = canvas[y:y + thick, x + 1:x + 1 + span]
        region[template] = color
        observed[y:y + thick, x + 1:x + 1 + extent] = True
        caps[y:y + thick, x] = True
    canvas[caps] = a["accent"]
    return target, observed, caps, canvas


def _apply(a, spans=None, rectangular=False, suppress_damage=False):
    # Optional arguments are explicit, local ablations used by the validator.
    target, observed, caps, out = _masks(a, spans, rectangular)
    holes = target & observed & (a["grid"] == a["field"])
    if not suppress_damage:
        halo = _dilate(holes) & ~holes
        out[halo] = a["accent"]
        out[holes] = a["field"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels alone, including all supported palettes and symmetries."""
    a = _decode(input_grid)
    return np.rot90(_apply(a), -a["turns"]).copy()


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


def _pattern(rng, thick, length, richness):
    master = np.ones((thick, length), dtype=bool)
    # An early aperture guarantees a common retained aperture even in short bars.
    master[thick // 2, 1] = False
    for _ in range(richness):
        row = int(rng.integers(1, thick - 1))
        col = int(rng.integers(1, length - 1))
        mode = int(rng.integers(4))
        master[row, col] = False
        if mode == 1 and col + 1 < length - 1:
            master[row, col + 1] = False
        elif mode == 2 and row + 1 < thick - 1:
            master[row + 1, col] = False
        elif mode == 3 and row + 1 < thick - 1 and col + 1 < length - 1:
            master[row + 1, col + 1] = False
    return master


def _lengths(rng, count, length, witness):
    middle = count // 2
    others = [i for i in map(int, rng.permutation(count)) if i != middle]
    lengths = [length] * count
    mode = "mixed" if witness else str(rng.choice(("mixed", "extend", "trim", "single")))
    if witness:
        for i in others:
            lengths[i] = int(rng.integers(3, length))
        lengths[others[0]] = length + int(rng.integers(4, 6))
    elif mode == "single":
        lengths[others[0]] = int(rng.choice((3, length + 3)))
    else:
        for i in others:
            if rng.random() < .22:
                continue
            short = mode == "extend" or (mode == "mixed" and rng.random() < .5)
            lengths[i] = int(rng.integers(3, length)) if short else length + int(rng.integers(2, 6))
        if mode == "mixed":
            lengths[others[0]] = int(rng.integers(3, length))
            lengths[others[1]] = length + int(rng.integers(4, 6))
    if all(e == length for e in lengths):
        lengths[others[0]] = length + 2
    return lengths


def _choose_points(rng, available, motif):
    """Enumerate small, bounded candidate placements; no rejection loop."""
    h, w = available.shape
    choices = []
    for y in range(h):
        for x in range(w):
            points = [(y + dy, x + dx) for dy, dx in motif]
            if all(0 <= r < h and 0 <= c < w and available[r, c] for r, c in points):
                choices.append(points)
    if not choices:
        return []
    return choices[int(rng.integers(len(choices)))]


def _construct_pair(rng, count, length, index, split, direction):
    witness = split == "train" and index < 2
    if split == "train" and index == 0:
        thick = 3
        gaps = [1] * (count - 1)
        gaps[0 if rng.integers(2) else -1] += int(rng.choice((6, 8)))
        richness = 0
    else:
        legal_thicknesses = [t for t in (3, 4, 5) if count * t + count - 1 <= 29]
        thick = int(rng.choice(legal_thicknesses))
        budget = 30 - count * thick
        equal = (split == "train" and index == 1) or rng.random() < .4
        if equal:
            gap = int(rng.integers(1, min(7, budget // (count - 1)) + 1))
            gaps = [gap] * (count - 1)
        else:
            gaps = [1] * (count - 1)
            extra = int(rng.integers(budget - sum(gaps) + 1))
            for _ in range(extra):
                gaps[int(rng.integers(len(gaps)))] += 1
        richness = int(rng.integers(1, 4)) if split == "train" else int(rng.integers(2, 6))
    master = _pattern(rng, thick, length, richness)
    lengths = _lengths(rng, count, length, witness)
    span = count * thick + sum(gaps)
    h = int(rng.integers(max(9, span), 31))
    w = int(rng.integers(max(lengths) + 5, 31))
    top = int(rng.integers(h - span + 1))
    if split == "train" and index == 0:
        top = 0  # One teaching example exposes canvas clipping explicitly.
    anchor = int(rng.integers(1, w - max(lengths) - 3))
    slots = [top]
    for gap in gaps:
        slots.append(slots[-1] + thick + gap)
    palette = list(map(int, rng.permutation(10)))
    field, accent = palette[:2]
    mono = split == "train" and index == 0
    palette_size = 1 if mono else int(rng.integers(2, min(count, 8) + 1))
    payload = palette[2:2 + palette_size]
    colors = payload + [int(rng.choice(payload)) for _ in range(count - palette_size)]
    rng.shuffle(colors)
    inp = np.full((h, w), field, dtype=np.int8)
    target = np.zeros((h, w), dtype=bool)
    defect_mask = np.zeros_like(target)
    out = inp.copy()
    middle = count // 2
    reserved = next((i for i, extent in enumerate(lengths)
                     if i not in (0, middle) and extent < length), -1)
    for i, (y, extent, color) in enumerate(zip(slots, lengths, colors)):
        inp[y:y + thick, anchor] = accent
        out[y:y + thick, anchor] = accent
        local = np.ones((thick, extent), dtype=bool)
        overlap = min(extent, length)
        local[:, :overlap] = master[:, :overlap]
        # Every measured bar ends with an intact full cross-section.
        local[:, [0, -1]] = True
        block = inp[y:y + thick, anchor + 1:anchor + 1 + extent]
        block[local] = color
        expected = out[y:y + thick, anchor + 1:anchor + 1 + length]
        expected[master] = color
        target[y:y + thick, anchor + 1:anchor + 1 + length] = master
        if i == middle:
            continue
        if witness and index == 0 and i == reserved:
            # One undamaged short bar visibly preserves the early aperture.
            # This teaches intentional negative space independently of halos.
            continue
        available = local.copy()
        available[:, [0, -1]] = False
        available[:, length:] = False
        motif = ((0, 0),)
        if witness and index == 1:
            motif = ((0, 0), (0, 1), (1, 0))
        elif rng.random() < .5:
            motif = ((0, 0), (1, 1)) if rng.integers(2) else ((0, 0), (0, 1))
        points = _choose_points(rng, available, motif)
        if not points:
            points = _choose_points(rng, available, ((0, 0),))
        for r, c in points:
            block[r, c] = field
            defect_mask[y + r, anchor + 1 + c] = True
        # Additional missing cells can merge, touch a side, or vanish on trimming.
        for _ in range(int(rng.integers(3))):
            extras = _choose_points(rng, available, ((0, 0),))
            for r, c in extras:
                block[r, c] = field
                defect_mask[y + r, anchor + 1 + c] = True
        if extent >= length + 4:
            # Boundary damage survives, deeper tail damage is discarded.
            r = int(rng.choice((0, thick - 1)))
            block[r, length - 1] = field
            block[(r + 1) % thick, length + 2] = field
            defect_mask[y + r, anchor + length] = True
            defect_mask[y + (r + 1) % thick, anchor + length + 3] = True
        if witness and index == 0:
            teaching_points = [(0, 1)] if i == 0 else []
            if extent > length:
                teaching_points += [(0, 2), (0, 3), (1, 3)]
            for r, c in teaching_points:
                block[r, c] = field
                defect_mask[y + r, anchor + 1 + c] = True
        # Filling a deliberate master aperture is extra material, not a defect.
        apertures = np.argwhere(~master[:, :min(extent - 1, length)])
        if len(apertures) and (witness or rng.random() < .65):
            r, c = map(int, apertures[int(rng.integers(len(apertures)))])
            block[r, c] = color
    # One detached extra beyond every bar's end, with fewer than thick cells in
    # its column, cannot be mistaken for an additional measured cross-section.
    nonmiddle = [i for i in range(count) if i != middle]
    chosen = int(rng.choice(nonmiddle))
    row = slots[chosen] + int(rng.integers(thick))
    inp[row, anchor + max(lengths) + 2] = colors[chosen]
    # A second extra may sit in a gutter; it carries an existing payload color.
    if gaps and rng.random() < .5:
        j = int(rng.integers(count - 1))
        inp[slots[j] + thick, anchor + int(rng.integers(2, length + 1))] = colors[j]
    # Construction output is painted directly from the master and damage masks.
    # It does not invoke the parser, the input solver, or its dilation helper.
    retained = defect_mask & target
    for y, x in np.argwhere(retained):
        out[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = accent
    out[retained] = field
    if rng.integers(2):
        inp, out = np.flipud(inp), np.flipud(out)
    return {"input": np.rot90(inp, direction).copy(),
            "output": np.rot90(out, direction).copy()}


def _signature(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    directions = list(map(int, rng.permutation(4)))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_lengths = set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            choices = [v for v in range(5, 17) if v not in train_lengths]
            # Distinct training reference spans, followed by genuinely unseen
            # spans in tests. At most ten of the twelve choices are consumed.
            length = int(rng.choice(choices))
            if split == "train":
                train_lengths.add(length)
            if split == "train" and index == 0:
                count = 5
            elif split == "train" and index == 1:
                count = int(rng.choice((3, 7)))
            else:
                count = int(rng.choice((3, 5, 7)))
            if regime == 0:
                direction = directions[0]
            elif regime == 1:
                direction = directions[index % 2] if split == "train" else directions[2 + index % 2]
            else:
                direction = directions[index % 4]
            for _ in range(12):
                pair = _construct_pair(rng, count, length, index, split, direction)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            puzzle[split].append(pair)
            seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    target, observed, caps, clean = _masks(a)
    holes = target & observed & (a["grid"] == a["field"])
    halo = _dilate(holes) & ~holes
    out = _apply(a)
    extras = ~target & ~caps & (a["grid"] != a["field"])
    positions = np.array(a["slots"]) * 2 + a["thick"] - 1
    distances = abs(2 * positions - positions[0] - positions[-1])
    nearest = np.flatnonzero(distances == distances.min())
    nearest_index = int(nearest[0]) if len(nearest) == 1 else -1
    aperture_match = np.zeros_like(target)
    tail_holes = np.zeros_like(target)
    for i, (y, extent, color) in enumerate(zip(a["slots"], a["lengths"], a["colors"])):
        x = a["cap_x"] + 1
        if i != len(a["slots"]) // 2:
            overlap = min(extent, a["length"])
            region = aperture_match[y:y + a["thick"], x:x + overlap]
            region[:] = ~a["master"][:, :overlap]
        if extent > a["length"]:
            tail_holes[y:y + a["thick"], x + a["length"]:x + extent] = True
    aperture_match &= (a["grid"] == a["field"]) & (out == a["field"])
    tail_holes &= a["grid"] == a["field"]
    edge = np.zeros_like(target)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    return {"count": len(a["slots"]), "thickness": a["thick"],
            "reference": a["length"], "lengths": list(a["lengths"]),
            "gaps": [b - y - a["thick"] for y, b in zip(a["slots"], a["slots"][1:])],
            "shape": list(a["grid"].shape), "direction": a["turns"],
            "field": a["field"], "accent": a["accent"],
            "payload_colors": len(set(a["colors"])),
            "aperture_cells": int((~a["master"]).sum()),
            "retained_holes": int(holes.sum()), "tail_holes": int(tail_holes.sum()),
            "far_erasure": int((extras & ~halo & (out == a["field"])).sum()),
            "matched_apertures": int(aperture_match.sum()),
            "extension": max(a["length"] - min(a["lengths"]), 0),
            "truncation": max(max(a["lengths"]) - a["length"], 0),
            "nonmedian": sorted(a["lengths"])[len(a["slots"]) // 2] != a["length"],
            "geometric_witness": nearest_index >= 0 and nearest_index != len(a["slots"]) // 2 and a["lengths"][nearest_index] != a["length"],
            "clipped": bool((holes & edge).any()),
            "halo_in_field": int((halo & ~target & ~caps).sum()),
            "orthogonal": bool((holes[1:] & holes[:-1]).any() or (holes[:, 1:] & holes[:, :-1]).any()),
            "diagonal": bool((holes[1:, 1:] & holes[:-1, :-1]).any() or (holes[1:, :-1] & holes[:-1, 1:]).any())}


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
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            a = _decode(inp)
            actual = np.rot90(out, a["turns"])
            assert np.array_equal(actual, _apply(a)), "wrong stencil or diagnostic output"
            assert not np.array_equal(inp, out)
            target, observed, caps, nominal = _masks(a)
            holes = target & observed & (a["grid"] == a["field"])
            halo = _dilate(holes) & ~holes
            assert np.all(actual[holes] == a["field"]), "holes must remain missing"
            assert np.all(actual[halo] == a["accent"]), "eight-neighbor halo union"
            assert np.array_equal(actual[~(holes | halo)], nominal[~(holes | halo)])
            assert np.all(actual[caps] == a["accent"]), "anchor caps are preserved"
            m = len(a["slots"]) // 2
            y, x = a["slots"][m], a["cap_x"] + 1
            assert np.array_equal(actual[y:y + a["thick"], x:x + a["length"]], a["grid"][y:y + a["thick"], x:x + a["length"]]), "master must remain unchanged"
            assert not np.array_equal(actual, _apply(a, suppress_damage=True)), "A must affect answer"
            assert not np.array_equal(actual, _apply(a, spans=a["lengths"])), "B must affect answer"
            assert not np.array_equal(actual, _apply(a, rectangular=True)), "intentional apertures must affect answer"
            f = _features(a)
            assert f["retained_holes"] and f["far_erasure"]
            features[split].append(f)
    train = features["train"]
    assert len({f["count"] for f in train}) > 1
    assert len({f["reference"] for f in train}) == len(train)
    assert any(f["nonmedian"] and f["geometric_witness"] for f in train)
    assert any(f["extension"] and f["truncation"] for f in train)
    assert any(f["payload_colors"] == 1 for f in train)
    assert any(f["payload_colors"] > 1 for f in train)
    assert any(f["matched_apertures"] for f in train)
    assert any(f["tail_holes"] for f in train)
    for effect in ("clipped", "orthogonal", "diagonal"):
        assert any(f[effect] for f in train), "demonstrate " + effect
    for f in features["test"]:
        assert f["reference"] not in {g["reference"] for g in train}, "held-out span shift"


def self_test():
    """Bounded 42-seed diversity, symmetry, ablation, mutation and count checks.

    No files, logs, traces, or grid corpora are stored. Replay generation peaks
    at three small puzzles. The returned report is suitable for compact JSON.
    """
    rows, hashes, shapes, stencils = [], set(), set(), set()
    permutation_rng = np.random.default_rng(92711)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        replay = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for i, pair in enumerate(puzzle[split]):
                other = replay[split][i]
                for role in ("input", "output"):
                    assert np.array_equal(pair[role], other[role])
                    digest.update(bytes(pair[role].shape))
                    digest.update(pair[role].tobytes())
                a = _decode(pair["input"])
                rows.append(_features(a))
                shapes.add(tuple(pair["input"].shape))
                stencils.add(_signature(a["master"]))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                if i == 0:
                    palette = permutation_rng.permutation(10)
                    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
                    for k in range(4):
                        for mirror in (False, True):
                            left, right = pair["input"], pair["output"]
                            if mirror:
                                left, right = np.fliplr(left), np.fliplr(right)
                            assert np.array_equal(solve(np.rot90(left, k)), np.rot90(right, k))
        hashes.add(digest.hexdigest())
    assert len(hashes) == 42
    assert len(shapes) >= 50 and len(stencils) >= 75
    assert {r["count"] for r in rows} == {3, 5, 7}
    assert {r["thickness"] for r in rows} == {3, 4, 5}
    assert {r["direction"] for r in rows} == set(range(4))
    assert len({r["reference"] for r in rows}) >= 10
    assert len({r["field"] for r in rows}) == 10
    assert len({r["accent"] for r in rows}) == 10
    assert any(r["clipped"] for r in rows)
    assert any(r["orthogonal"] for r in rows) and any(r["diagonal"] for r in rows)
    for train_count in range(2, 11):
        for test_count in range(1, min(4, train_count - 1) + 1):
            generate_puzzle(314159, train_count, test_count)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A wrong label is rejected, even when the input and all metadata are valid.
    broken = generate_puzzle(4242, 2, 1)
    pair = broken["test"][0]
    a = _decode(pair["input"])
    target, observed, caps, nominal = _masks(a)
    holes = target & observed & (a["grid"] == a["field"])
    wrong = np.rot90(pair["output"], a["turns"]).copy()
    y, x = map(int, np.argwhere(holes)[0])
    wrong[y, x] = a["accent"]
    pair["output"] = np.rot90(wrong, -a["turns"]).copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a filled defect")
    return {"seeds": 42, "pairs": len(rows), "distinct_puzzles": len(hashes),
            "grid_shapes": len(shapes), "master_stencils": len(stencils),
            "counts": sorted({r["count"] for r in rows}),
            "thicknesses": sorted({r["thickness"] for r in rows}),
            "references": sorted({r["reference"] for r in rows}),
            "clipped_pairs": sum(r["clipped"] for r in rows),
            "tail_damage_pairs": sum(r["tail_holes"] > 0 for r in rows),
            "nonmedian_pairs": sum(r["nonmedian"] for r in rows),
            "geometric_witness_pairs": sum(r["geometric_witness"] for r in rows),
            "semantic_verification": "generator-owned checks; no independent solver"}
