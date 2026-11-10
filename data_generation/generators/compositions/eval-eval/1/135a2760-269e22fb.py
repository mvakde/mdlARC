"""Repair a periodic cloth and unfold its self-referential inset swatch.

The solver reads only the input.  Generation never imports either parent.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _d4(a, k):
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _canonical(a):
    return min(_key(_d4(a, k) ^ s) for k in range(8) for s in range(2))


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _parse(value):
    """Normalize the long direction to columns; find lanes and the one tag."""
    original = _check_grid(value)
    structure = int(original[0, 0])
    if not (np.all(original[0] == structure)
            and np.all(original[-1] == structure)
            and np.all(original[:, 0] == structure)
            and np.all(original[:, -1] == structure)):
        raise ValueError("missing outer frame")
    nr = int(np.sum(np.all(original == structure, axis=1)))
    nc = int(np.sum(np.all(original == structure, axis=0)))
    transpose = nc > 2 or (nr == nc == 2 and original.shape[0] > original.shape[1])
    a = original.T.copy() if transpose else original.copy()
    H, W = a.shape
    dividers = np.flatnonzero(np.all(a == structure, axis=1)).tolist()
    if (np.flatnonzero(np.all(a == structure, axis=0)).tolist() != [0, W - 1]
            or dividers[0] != 0 or dividers[-1] != H - 1
            or any(b - x < 8 for x, b in zip(dividers, dividers[1:]))):
        raise ValueError("invalid parallel lanes")
    rects = [(slice(x + 2, b - 1), slice(2, W - 2))
             for x, b in zip(dividers, dividers[1:])]
    motif_mask = np.zeros(a.shape, dtype=bool)
    frame_mask = np.zeros(a.shape, dtype=bool)
    frame_mask[dividers] = True
    frame_mask[:, [0, W - 1]] = True
    for rect in rects:
        motif_mask[rect] = True
    padding_values = np.unique(a[~(motif_mask | frame_mask)])
    if len(padding_values) != 1 or padding_values[0] == structure:
        raise ValueError("invalid one-cell padding")
    padding = int(padding_values[0])
    colors = set(map(int, np.unique(a)))
    if len(colors) != 4:
        raise ValueError("expected four geometric color roles")
    tag_candidates = []
    for color in sorted(colors - {structure, padding}):
        rr, cc = np.nonzero(a == color)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        perimeter = np.zeros(a.shape, dtype=bool)
        perimeter[[r0, r1], c0:c1 + 1] = True
        perimeter[r0:r1 + 1, [c0, c1]] = True
        if np.array_equal(a == color, perimeter):
            lanes = [i for i, rect in enumerate(rects)
                     if rect[0].start <= r0 and r1 < rect[0].stop
                     and rect[1].start <= c0 and c1 < rect[1].stop]
            if len(lanes) == 1:
                tag_candidates.append((color, lanes[0], (r0, r1, c0, c1)))
    if len(tag_candidates) != 1:
        raise ValueError("expected one rectangular swatch frame")
    tag, target, box = tag_candidates[0]
    ink, = colors - {structure, padding, tag}
    r0, r1, c0, c1 = box
    ignored = np.zeros(a.shape, dtype=bool)
    ignored[r0:r1 + 1, c0:c1 + 1] = True
    for rect in rects:
        known = ~ignored[rect]
        if not set(map(int, np.unique(a[rect][known]))) <= {structure, ink}:
            raise ValueError("nonbinary cloth")
    crop = a[r0 + 1:r1, c0 + 1:c1]
    if set(map(int, np.unique(crop))) != {structure, ink}:
        raise ValueError("swatch must show both binary roles")
    return dict(a=a, transpose=transpose, structure=structure, padding=padding,
                ink=ink, tag=tag, rects=rects, target=target, box=box,
                ignored=ignored, crop=(crop == ink).astype(np.int8))


def _recover(bits, known):
    """All well-supported periods must agree on the complete clean lane."""
    h, length = bits.shape
    candidates = []
    error_limit = max(1, int(known.sum()) // 12)
    for p in range(1, length // 4 + 1):
        tile = np.zeros((h, p), dtype=np.int8)
        errors, valid = 0, True
        for x in range(h):
            for phase in range(p):
                observed = bits[x, phase::p][known[x, phase::p]]
                ones, n = int(observed.sum()), len(observed)
                if n < 3 or 2 * ones == n:
                    valid = False
                    break
                tile[x, phase] = int(2 * ones > n)
                errors += min(ones, n - ones)
            if not valid:
                break
        if valid and errors <= error_limit and len(np.unique(tile)) == 2:
            clean = tile[:, np.arange(length) % p]
            candidates.append((p, tile, clean, errors))
    if not candidates:
        raise ValueError("no supported periodic repair")
    if any(not np.array_equal(c[2], candidates[0][2]) for c in candidates):
        raise ValueError("competing periodic repairs")
    return candidates[0]


def _matches(tiles, crop):
    """Every exact donor / D4 / color bijection / rectangular offset match."""
    h, w = crop.shape
    matches = []
    for donor, tile in enumerate(tiles):
        for k in range(8):
            for swap in range(2):
                image = _d4(tile, k) ^ swap
                H, W = image.shape
                if h > H or w > W:
                    continue
                windows = np.lib.stride_tricks.sliding_window_view(image, (h, w))
                rr, cc = np.nonzero(np.all(windows == crop, axis=(-2, -1)))
                matches.extend((donor, k, swap, int(r), int(c), image)
                               for r, c in zip(rr, cc))
    return matches


def _analyze(value):
    d = _parse(value)
    a = d["a"]
    base = a.copy()
    tiles, periods, errors, known_masks = [], [], [], []
    for rect in d["rects"]:
        bits = (a[rect] == d["ink"]).astype(np.int8)
        known = ~d["ignored"][rect]
        p, tile, clean, count = _recover(bits, known)
        tiles.append(tile)
        periods.append(p)
        errors.append(count)
        known_masks.append(known)
        base[rect] = np.where(clean, d["ink"], d["structure"])
    matches = _matches(tiles, d["crop"])
    if len(matches) != 1:
        raise ValueError("swatch does not identify one donor, orientation and offset")
    donor, k, swap, r, c, image = matches[0]
    if d["crop"].size == image.size:
        raise ValueError("swatch must be a proper crop")
    r0, r1, c0, c1 = d["box"]
    top, left = r0 + 1 - r, c0 + 1 - c
    H, W = image.shape
    target = d["rects"][d["target"]]
    if not (target[0].start <= top and top + H <= target[0].stop
            and target[1].start <= left and left + W <= target[1].stop):
        raise ValueError("completed inlay leaves its lane")
    out = base.copy()
    out[top:top + H, left:left + W] = np.where(image, d["ink"], d["structure"])
    d.update(base=base, output=out, tiles=tiles, periods=periods, errors=errors,
             known_masks=known_masks, donor=donor, transform=k, swap=swap,
             offset=(r, c), footprint=(top, left, H, W))
    return d


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _analyze(input_grid)
    return (d["output"].T if d["transpose"] else d["output"]).copy()


def _tile(rng, h, p):
    """Short binary row runs, changing phase and width, form a woven glyph."""
    a = np.zeros((h, p), dtype=np.int8)
    phase = int(rng.integers(p))
    style = int(rng.integers(4))
    for r in range(h):
        if style == 0:
            phase = int(rng.integers(p))
        else:
            phase = (phase + int(rng.choice([-1, 0, 1]))) % p
        width = int(rng.integers(1, p))
        a[r, (phase + np.arange(width)) % p] = 1
        if style == 2 and r % 3 == 0:
            a[r] ^= 1
        if style == 3 and rng.random() < .25:
            a[r] = int(rng.integers(2))
    return a ^ int(rng.integers(2))


def _choose_swatch(rng, tiles, target, length, cross_donor):
    """Bounded exact matching; returning None requests new glyphs."""
    height = tiles[target].shape[0]
    donors = [i for i in range(len(tiles)) if not cross_donor or i != target]
    for _ in range(160):
        donor = int(rng.choice(donors))
        k, swap = int(rng.integers(8)), int(rng.integers(2))
        image = _d4(tiles[donor], k) ^ swap
        H, W = image.shape
        if H > height or W > length:
            continue
        h = int(rng.integers(2, min(H, height - 2) + 1))
        w = int(rng.integers(2, W + 1))
        if h * w < 8 or h * w == H * W:
            continue
        r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
        # At least two unseen rows/columns on some edge makes the completion
        # extend beyond the one-cell tag, rather than merely replacing its rim.
        if max(r, H - r - h, c, W - c - w) < 2:
            continue
        crop = image[r:r + h, c:c + w]
        if len(np.unique(crop)) != 2 or len(_matches(tiles, crop)) != 1:
            continue
        tops = [x for x in range(height - H + 1) if x + r >= 1 and x + r + h < height]
        lefts = [x for x in range(length - W + 1) if x + c >= 1 and x + c + w < length]
        if not tops or not lefts:
            continue
        top, left = int(rng.choice(tops)), int(rng.choice(lefts))
        return donor, k, swap, r, c, image, crop.copy(), top, left
    return None


def _make_pair(rng, count, length, colors, transpose, cross_donor):
    structure, padding, ink, tag = colors
    for _ in range(180):
        target = int(rng.integers(count))
        crosses = [5] * count
        crosses[target] = 7
        budget = min(29 - 3 * count, 9 + 8 * (count - 1))
        extra = int(rng.integers(budget - sum(crosses) + 1))
        for _ in range(extra):
            choices = [i for i in range(count) if crosses[i] < (9 if i == target else 8)]
            crosses[int(rng.choice(choices))] += 1
        possible = list(range(3, min(5, length // 5) + 1))
        periods = [int(rng.choice(possible)) for _ in crosses]
        if count > 1:
            periods[1] = int(rng.choice([p for p in possible if p != periods[0]]))
        tiles = [_tile(rng, h, p) for h, p in zip(crosses, periods)]
        if any(len({_key(_d4(t, k) ^ s) for k in range(8) for s in range(2)}) != 16
               for t in tiles) or len({_canonical(t) for t in tiles}) != count:
            continue
        swatch = _choose_swatch(rng, tiles, target, length, cross_donor)
        if swatch is None:
            continue
        donor, k, swap, cr, cc, image, crop, top, left = swatch
        dividers = [0]
        for cross in crosses:
            dividers.append(dividers[-1] + cross + 3)
        shape = (dividers[-1] + 1, length + 4)
        clean = np.full(shape, padding, dtype=np.int8)
        clean[dividers, :] = structure
        clean[:, [0, shape[1] - 1]] = structure
        rects = []
        for i, tile in enumerate(tiles):
            rect = (slice(dividers[i] + 2, dividers[i + 1] - 1), slice(2, length + 2))
            rects.append(rect)
            clean[rect] = np.where(tile[:, np.arange(length) % periods[i]], ink, structure)
        top += rects[target][0].start
        left += 2
        H, W = image.shape
        h, w = crop.shape
        r0, c0 = top + cr - 1, left + cc - 1
        r1, c1 = r0 + h + 1, c0 + w + 1
        footprint = np.zeros(shape, dtype=bool)
        footprint[top:top + H, left:left + W] = True
        ignored = np.zeros(shape, dtype=bool)
        ignored[r0:r1 + 1, c0:c1 + 1] = True
        image_colors = np.where(image, ink, structure)
        inlay_difference = clean[top:top + H, left:left + W] != image_colors
        beyond_tag = ~ignored[top:top + H, left:left + W]
        if not np.any(inlay_difference & beyond_tag):
            continue
        inp = clean.copy()
        inp[r0:r1 + 1, c0:c1 + 1] = tag
        inp[r0 + 1:r1, c0 + 1:c1] = np.where(crop, ink, structure)
        # Every lane has a visible error outside the completed inlay. Distinct
        # phase-coordinate groups keep even three observations reliable.
        for i, rect in enumerate(rects):
            allowed = ~ignored[rect] & ~footprint[rect]
            rr, cols = np.nonzero(allowed)
            groups = {}
            for r, c in zip(rr.tolist(), cols.tolist()):
                groups.setdefault((r, c % periods[i]), []).append((r, c))
            group_keys = list(groups)
            n = int(rng.integers(1, min(4, max(1, int(allowed.sum()) // 28), len(group_keys)) + 1))
            for gi in rng.choice(len(group_keys), n, replace=False):
                options = groups[group_keys[int(gi)]]
                r, c = options[int(rng.integers(len(options)))]
                absolute = (rect[0].start + r, rect[1].start + c)
                inp[absolute] = ink if inp[absolute] == structure else structure
        expected = clean.copy()
        expected[top:top + H, left:left + W] = image_colors
        if transpose:
            inp, expected = inp.T.copy(), expected.T.copy()
        try:
            d = _analyze(inp)
        except ValueError:
            continue
        predicted = d["output"].T if d["transpose"] else d["output"]
        if (d["periods"] != periods
                or not np.array_equal(d["base"], clean)
                or not np.array_equal(predicted, expected)):
            continue
        return {"input": inp, "output": expected}
    raise RuntimeError("bounded cloth construction exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError("counts must be legal integers")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    stable_palette = bool(rng.integers(4) == 0)
    base_axis = bool(rng.integers(2))
    stable_axis = bool(rng.integers(5) == 0)
    counts = [int(rng.choice([2, 3]))]
    counts.append(int(rng.choice([c for c in (1, 2, 3) if c != counts[0]])))
    counts += list(map(int, rng.integers(1, 4, size=nt + nq - 2)))
    train_lengths = list(map(int, rng.integers(20, 27, size=nt)))
    if train_lengths[1] == train_lengths[0]:
        train_lengths[1] = 20 + (train_lengths[0] - 19) % 7
    lengths = train_lengths + list(map(int, rng.integers(25, 27, size=nq)))
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(nt + nq):
        axis = base_axis if stable_axis else (base_axis if i == 0 else
               not base_axis if i == 1 else bool(rng.integers(2)))
        colors = palette[:4] if stable_palette else list(map(int, rng.choice(10, 4, replace=False)))
        for _ in range(20):
            pair = _make_pair(rng, counts[i], lengths[i], colors, axis,
                              cross_donor=(i == 0))
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["train" if i < nt else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check actual grids, including both concept-ablation counterexamples."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_records = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _analyze(inp)
            actual = out.T if d["transpose"] else out
            assert inp.shape == out.shape and np.array_equal(actual, d["output"])
            a, base, ignored = d["a"], d["base"], d["ignored"]
            top, left, H, W = d["footprint"]
            footprint = np.zeros(a.shape, dtype=bool)
            footprint[top:top + H, left:left + W] = True
            motifs = np.zeros(a.shape, dtype=bool)
            assert 20 <= a.shape[1] - 4 <= 26
            assert min(d["crop"].shape) >= 2 and d["crop"].size >= 8
            assert len({_canonical(t) for t in d["tiles"]}) == len(d["tiles"])
            for tile in d["tiles"]:
                assert len({_key(_d4(tile, k) ^ s)
                            for k in range(8) for s in range(2)}) == 16
            for i, rect in enumerate(d["rects"]):
                motifs[rect] = True
                assert 3 <= d["periods"][i] <= 5
                assert 1 <= d["errors"][i] <= 4
                assert a.shape[1] - 4 >= 5 * d["periods"][i]
                assert (7 <= d["tiles"][i].shape[0] <= 9 if i == d["target"]
                        else 5 <= d["tiles"][i].shape[0] <= 8)
                assert set(map(int, np.unique(actual[rect]))) == {d["structure"], d["ink"]}
                lane_outside = (~footprint & ~ignored)[rect]
                assert np.any(a[rect][lane_outside] != actual[rect][lane_outside])
                mismatches = (a[rect] != base[rect]) & ~ignored[rect]
                for phase in range(d["periods"][i]):
                    assert np.all(mismatches[:, phase::d["periods"][i]].sum(axis=1) <= 1)
            assert 1 <= len(d["rects"]) <= 3
            assert np.array_equal(a[~motifs], actual[~motifs])
            assert set(map(int, np.unique(out))) == {d["structure"], d["padding"], d["ink"]}
            assert d["crop"].size < d["tiles"][d["donor"]].size
            r, c = d["offset"]
            h, w = d["crop"].shape
            assert max(r, H - r - h, c, W - c - w) >= 2
            # A is necessary: there are corrections outside the whole inlay.
            outside = ~footprint & ~ignored & motifs
            assert np.any(a[outside] != actual[outside])
            # B is necessary: pure periodic repair erases an intentional inlay.
            assert np.any(base[footprint & ~ignored] != actual[footprint & ~ignored])
            # The exact raw crop survives despite not being part of the base.
            r0, r1, c0, c1 = d["box"]
            assert np.array_equal(a[r0 + 1:r1, c0 + 1:c1], actual[r0 + 1:r1, c0 + 1:c1])
            if split == "test":
                assert a.shape[1] - 4 >= 25
            else:
                train_records.append(d)
    assert len({len(d["rects"]) for d in train_records}) >= 2
    assert len({d["a"].shape[1] for d in train_records}) >= 2
    assert any(d["donor"] != d["target"] for d in train_records)
    assert any(len(set(d["periods"])) >= 2 for d in train_records)


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    signatures, shapes, normalized, references = set(), set(), set(), set()
    counts, periods, thicknesses, axes, transforms, swaps, offsets = (set() for _ in range(7))
    cross_donors = partial = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                roles = {d[n]: i for i, n in enumerate(("structure", "padding", "ink", "tag"))}
                normalized.add(_key(np.vectorize(roles.__getitem__)(d["a"])))
                references.update(_canonical(t) for t in d["tiles"])
                counts.add(len(d["rects"]))
                periods.update(d["periods"])
                thicknesses.update(t.shape[0] for t in d["tiles"])
                axes.add(d["transpose"])
                transforms.add(d["transform"])
                swaps.add(d["swap"])
                offsets.add(d["offset"])
                cross_donors += d["donor"] != d["target"]
                partial += any((d["a"].shape[1] - 4) % p for p in d["periods"])
    assert len(signatures) == 42 and len(normalized) == 210
    assert len(shapes) >= 45 and len(references) >= 150
    assert counts == {1, 2, 3} and periods == {3, 4, 5}
    assert len(thicknesses) >= 5 and axes == {False, True}
    assert transforms == set(range(8)) and swaps == {0, 1}
    assert len(offsets) >= 12 and cross_donors >= 42 and partial >= 100
    p = generate_puzzle(0)
    assert _signature(p) == _signature(generate_puzzle(0))
    # Color relabeling and scene transposition commute with the rule. Reversing
    # the long direction reanchors the fundamental unit; all D4 operations are
    # swatch options, rather than arbitrary whole-scene symmetries.
    for pair in p["train"]:
        assert np.array_equal(solve(pair["input"].T), pair["output"].T)
        permutation = np.array([7, 3, 9, 0, 6, 2, 8, 5, 1, 4], dtype=np.int8)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + 10 * nt + nq, nt, nq)
    for nt, nq in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupted = copy.deepcopy(p)
    corrupted["test"][0]["output"][0, 0] ^= 1
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "normalized_inputs": len(normalized),
            "reference_classes": len(references), "cross_lane_inlays": cross_donors,
            "partial_cycle_examples": partial, "swatch_offsets": len(offsets)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
