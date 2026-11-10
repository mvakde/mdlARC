"""Compositional instruction-strip ARC puzzles.

The first and third upper objects are congruent boxes.  The middle T points
from its crossbar toward the side on which to attach the third box to the
matching first box in the work panel.  The divider removes the instruction
strip from the answer.
"""

import hashlib

import numpy as np


SOURCE_PAIR_DIGESTS = {
    "e272326cd2941da68f6174f00df4febffdae2f344a604ca6c9e26b01ac526588",
    "717762e46e9cca3eb866947de6e9ecf24ccda3bb58f7b2fd8c17aec03ff563e7",
    "8fbef298480a3a8b7ced72f5ad9ca752847209ec1272d365e9eabffe80a35a83",
    "33b362fa7c5947ec5df287152f2b311a18313116dcb8da8b950e9e99ea692237",
    "bb49c7318ca476e7657ba7f5d249ac476022bd70774bb0d7b77e603d0e6a8d9d",
    "6df56cab3bf517668a37e6a5624d9664d3f82de02870348313e9236009c8ec19",
}
SIZES = ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (3, 5),
         (5, 3), (4, 4), (4, 5), (5, 4), (5, 5), (2, 4), (4, 2))
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _t_mask(size, direction):
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    if direction == (1, 0):
        mask[0, :] = True
        mask[:, mid] = True
    elif direction == (-1, 0):
        mask[-1, :] = True
        mask[:, mid] = True
    elif direction == (0, 1):
        mask[:, 0] = True
        mask[mid, :] = True
    else:
        mask[:, -1] = True
        mask[mid, :] = True
    return mask


def _motif(rng, height, width):
    kind = int(rng.integers(0, 9))
    rr, cc = np.indices((height, width))
    if kind == 0:  # sparse diagonal staircase
        mask = cc == np.rint(rr * (width - 1) / max(1, height - 1))
    elif kind == 1:  # opposing diagonals
        x = rr * (width - 1) / max(1, height - 1)
        mask = (cc == np.rint(x)) | (cc == np.rint(width - 1 - x))
    elif kind == 2:  # ring
        mask = (rr == 0) | (rr == height - 1) | (cc == 0) | (cc == width - 1)
    elif kind == 3:  # cross and bars
        mask = (rr == int(rng.integers(height))) | (cc == int(rng.integers(width)))
    elif kind == 4:  # horizontally mirrored random motif
        half = rng.random((height, (width + 1) // 2)) < float(rng.uniform(.35, .8))
        mask = np.concatenate((half, np.fliplr(half[:, :width // 2])), axis=1)
    elif kind == 5:  # dense irregular motif
        mask = rng.random((height, width)) > float(rng.uniform(.06, .28))
    elif kind == 6:  # solid rectangle
        mask = np.ones((height, width), dtype=bool)
    elif kind == 7:  # two horizontal bars with a bridge
        mask = (rr == 0) | (rr == height - 1) | (cc == int(rng.integers(width)))
    else:
        mask = rng.random((height, width)) < float(rng.uniform(.25, .75))
    # Occupied outer rows and columns make the complete box observable.
    for row in range(height):
        if not mask[row].any():
            mask[row, int(rng.integers(width))] = True
    for col in range(width):
        if not mask[:, col].any():
            mask[int(rng.integers(height)), col] = True
    return mask


def _colored_motif(rng, mask, background, primary, secondary=None):
    result = np.full(mask.shape, background, dtype=np.int8)
    result[mask] = primary
    if secondary is not None and mask.sum() >= 4:
        occupied = np.argwhere(mask)
        rng.shuffle(occupied)
        for row, col in occupied[:max(1, len(occupied) // 3)]:
            result[row, col] = secondary
    return result


def _put(grid, top, left, motif, background):
    h, w = motif.shape
    block = grid[top:top + h, left:left + w]
    block[motif != background] = motif[motif != background]


def _palette(rng, plan, primary, shared=None):
    if shared is not None:
        return shared
    base = plan["base"]
    style = plan["palette_style"]
    if style == "stable":
        roles = list(base[:4])
    elif style == "partial":
        others = [c for c in range(10) if c not in (base[0], base[1], primary)]
        rng.shuffle(others)
        roles = [base[0], base[1], *others[:2]]
    else:
        others = [c for c in range(10) if c != primary]
        rng.shuffle(others)
        roles = others[:4]
    available = [c for c in range(10) if c not in roles and c != primary]
    rng.shuffle(available)
    return tuple(roles + [primary] + available)


def _layout(rng, size, direction, witness, zone, shared=None):
    if shared is not None:
        return shared
    h, w = size
    marker = int(rng.choice((3, 5)))
    top_pad = int(rng.integers(1, 4))
    bottom_pad = int(rng.integers(1, 4))
    upper_h = max(h, marker) + top_pad + bottom_pad
    min_width = 2 * w + marker + 2
    min_work_h = (3 if witness and direction[0] else 2) * h
    if upper_h + 1 + min_work_h > 30:
        upper_h = 30 - 1 - min_work_h
    if upper_h < max(h, marker) + 1:
        marker = 3
        upper_h = max(h, marker) + 1
    min_width = max(min_width, (3 if witness and direction[1] else 2) * w)
    width = int(rng.integers(min_width, 31))
    work_h = int(rng.integers(min_work_h, 30 - upper_h))
    if h <= 3 and rng.random() < .13:
        # Restore the source dimensions as a genuine member of the family;
        # content hashes, rather than dimensions, exclude original pairs.
        width, upper_h, work_h, marker = 15, 5, 9, 3
    # The two witness destinations fit simultaneously. Other pairs mix
    # constrained and open placements, plus exact boundary contact.
    if direction[0]:
        lo = h if witness else (h if direction[0] < 0 else 0)
        hi = work_h - 2 * h if witness or direction[0] > 0 else work_h - h
        if witness:
            hi = work_h - 2 * h
        if hi < lo:
            raise ValueError("vertical layout too short")
        top = int(rng.integers(lo, hi + 1))
        if not witness and rng.random() < .25:
            top = lo if rng.random() < .5 else hi
        left = _zone_coordinate(rng, width - w, zone)
    else:
        lo = w if witness else (w if direction[1] < 0 else 0)
        hi = width - 2 * w if witness or direction[1] > 0 else width - w
        if witness:
            hi = width - 2 * w
        left = int(rng.integers(lo, hi + 1))
        if not witness and rng.random() < .25:
            left = lo if rng.random() < .5 else hi
        top = int(rng.integers(0, work_h - h + 1))
    if direction[0] and not witness and rng.random() < .22:
        top = h if direction[0] < 0 else work_h - 2 * h
    return (width, upper_h, work_h, marker, top, left)


def _zone_coordinate(rng, maximum, zone):
    if zone == "left":
        return int(rng.integers(0, max(1, maximum // 3 + 1)))
    if zone == "right":
        return int(rng.integers(maximum - maximum // 3, maximum + 1))
    if zone == "center":
        return int(rng.integers(maximum // 3, maximum - maximum // 3 + 1))
    return int(rng.integers(0, maximum + 1))


def _box_clear(grid, top, left, height, width, background, gap=0):
    if top < 0 or left < 0 or top + height > grid.shape[0] or left + width > grid.shape[1]:
        return False
    t = max(0, top - gap)
    l = max(0, left - gap)
    b = min(grid.shape[0], top + height + gap)
    r = min(grid.shape[1], left + width + gap)
    return bool(np.all(grid[t:b, l:r] == background))


def _match_count(work, reference):
    h, w = reference.shape
    return sum(np.array_equal(work[t:t+h, l:l+w], reference)
               for t in range(work.shape[0] - h + 1)
               for l in range(work.shape[1] - w + 1))


def _make_pair(rng, plan, size, direction, zone, primary, witness=False,
               shared=None, want_distractor=False, rotation=0):
    h, w = size
    reference_mask = shared["reference_mask"].copy() if shared else _motif(rng, h, w)
    payload_mask = _motif(rng, h, w)
    palette = _palette(rng, plan, primary, shared["palette"] if shared else None)
    bg, divider, ref_color, indicator, payload_color = palette[:5]
    second_ref = palette[5] if rng.random() < plan["multi_rate"] else None
    second_payload = palette[6] if rng.random() < plan["multi_rate"] else None
    if shared:
        reference = shared["reference"].copy()
    else:
        reference = _colored_motif(rng, reference_mask, bg, ref_color, second_ref)
    payload = _colored_motif(rng, payload_mask, bg, payload_color, second_payload)
    layout = _layout(rng, size, direction, witness, zone, shared["layout"] if shared else None)
    width, upper_h, work_h, marker, anchor_top, anchor_left = layout
    inp = np.full((upper_h + 1 + work_h, width), bg, dtype=np.int8)
    inp[upper_h, :] = divider
    free = width - (2 * w + marker + 2)
    margins = rng.multinomial(free, (.25, .25, .25, .25))
    ref_left = int(margins[0])
    indicator_left = ref_left + w + 1 + int(margins[1])
    payload_left = indicator_left + marker + 1 + int(margins[2])
    object_top = int(rng.integers(0, upper_h - h + 1))
    marker_top = max(0, min(upper_h - marker, object_top + (h - marker) // 2))
    _put(inp, object_top, ref_left, reference, bg)
    _put(inp, marker_top, indicator_left,
         _colored_motif(rng, _t_mask(marker, direction), bg, indicator), bg)
    _put(inp, object_top, payload_left, payload, bg)
    work = inp[upper_h + 1:]
    _put(work, anchor_top, anchor_left, reference, bg)
    dest_top = anchor_top + direction[0] * h
    dest_left = anchor_left + direction[1] * w
    assert _box_clear(work, dest_top, dest_left, h, w, bg)
    if want_distractor:
        count = 1 + int(rng.random() < .25)
        for _ in range(count):
            for _attempt in range(120):
                dh, dw = size if rng.random() < .7 else SIZES[int(rng.integers(len(SIZES)))]
                dt = int(rng.integers(0, work_h - dh + 1)) if dh <= work_h else -1
                dl = int(rng.integers(0, width - dw + 1)) if dw <= width else -1
                if not _box_clear(work, dt, dl, dh, dw, bg, 1):
                    continue
                if not (dt + dh + 1 <= dest_top or dest_top + h + 1 <= dt or
                        dl + dw + 1 <= dest_left or dest_left + w + 1 <= dl):
                    continue
                if witness:
                    other_top = anchor_top - direction[0] * h
                    other_left = anchor_left - direction[1] * w
                    if not (dt + dh + 1 <= other_top or other_top + h + 1 <= dt or
                            dl + dw + 1 <= other_left or other_left + w + 1 <= dl):
                        continue
                dm = _motif(rng, dh, dw)
                dc = ref_color if rng.random() < .65 else palette[7]
                distractor = _colored_motif(rng, dm, bg, dc)
                if dh == h and dw == w and np.array_equal(distractor, reference):
                    continue
                prior = work.copy()
                _put(work, dt, dl, distractor, bg)
                if _match_count(work, reference) != 1:
                    work[:] = prior
                    continue
                break
    out = work.copy()
    _put(out, dest_top, dest_left, payload, bg)
    if rotation:
        inp = np.rot90(inp, rotation).copy()
        out = np.rot90(out, rotation).copy()
    pair = {"input": inp, "output": out}
    state = {"reference_mask": reference_mask, "reference": reference,
             "palette": palette, "layout": layout}
    return pair, state


def _axis_directions(axis):
    return ((1, 0), (-1, 0)) if axis == "vertical" else ((0, 1), (0, -1))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _seed(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.permutation(10))
    palette_style = ("stable", "partial", "free")[int(rng.integers(3))]
    axis_mode = ("vertical", "horizontal", "mixed", "shift")[int(rng.integers(4))]
    scene_mode = ("fixed", "mixed", "shift")[int(rng.integers(3))]
    fixed_rotation = int(rng.integers(4))
    plan = {"base": base, "palette_style": palette_style,
            "multi_rate": float(rng.choice((0.0, .35, .75))),
            "axis_mode": axis_mode, "scene_mode": scene_mode}
    pool = list(base[4:])
    rng.shuffle(pool)
    train_primary = pool[:int(rng.integers(1, min(3, train_count - 1) + 1))]
    novelty = ("familiar", "partial", "free")[int(rng.integers(3))]
    train_colors = [int(train_primary[0]), int(train_primary[0])]
    train_colors += [int(c) for c in train_primary[1:]]
    train_colors += [int(train_primary[int(rng.integers(len(train_primary)))])
                     for _ in range(train_count - len(train_colors))]
    test_colors = []
    for i in range(test_count):
        if novelty == "familiar" or (novelty == "partial" and i):
            test_colors.append(int(train_primary[int(rng.integers(len(train_primary)))]))
        elif novelty == "partial":
            test_colors.append(int(pool[len(train_primary)]))
        else:
            test_colors.append(int(pool[int(rng.integers(len(pool)))]))
    if axis_mode == "horizontal":
        witness_axis = "horizontal"
    else:
        witness_axis = "vertical"
    d0, d1 = _axis_directions(witness_axis)
    if rng.random() < .5:
        d0, d1 = d1, d0
    directions = [d0, d1]
    for i in range(2, train_count):
        axis = ("horizontal" if axis_mode == "horizontal" else
                "vertical" if axis_mode in ("vertical", "shift") else
                ("horizontal" if i == 2 else ("vertical" if rng.random() < .5 else "horizontal")))
        directions.append(_axis_directions(axis)[int(rng.integers(2))])
    test_dirs = []
    for _ in range(test_count):
        axis = ("vertical" if axis_mode == "vertical" else
                "horizontal" if axis_mode in ("horizontal", "shift") else
                ("vertical" if rng.random() < .5 else "horizontal"))
        test_dirs.append(_axis_directions(axis)[int(rng.integers(2))])
    size_mode = ("uniform", "mixed", "scale_shift")[int(rng.integers(3))]
    witness_size = SIZES[int(rng.integers(len(SIZES)))]
    train_sizes = [witness_size, witness_size]
    for _ in range(2, train_count):
        train_sizes.append(witness_size if size_mode == "uniform" else
                           SIZES[int(rng.integers(len(SIZES)))])
    if size_mode == "scale_shift":
        choices = [s for s in SIZES if s not in train_sizes]
        test_sizes = [choices[int(rng.integers(len(choices)))] if choices else witness_size
                      for _ in range(test_count)]
    elif size_mode == "uniform":
        test_sizes = [witness_size] * test_count
    else:
        test_sizes = [SIZES[int(rng.integers(len(SIZES)))] for _ in range(test_count)]
    position_mode = ("shared", "left_shift", "right_shift", "mixed")[int(rng.integers(4))]
    def zone(split):
        if position_mode == "left_shift" and split == "test":
            return "left"
        if position_mode == "right_shift" and split == "test":
            return "right"
        if position_mode in ("left_shift", "right_shift") and split == "train":
            return "center"
        return ("left", "center", "right", "any")[int(rng.integers(4))]
    rotation0 = fixed_rotation if scene_mode == "fixed" else 0
    def rotation(split):
        if scene_mode == "fixed":
            return fixed_rotation
        if scene_mode == "shift":
            return 0 if split == "train" else int(rng.integers(1, 4))
        return int(rng.integers(4))
    distractor_regime = bool(rng.random() < .62)
    train = []
    shared = None
    for i in range(train_count):
        r = rotation0 if i < 2 else rotation("train")
        # Witness examples share reference, anchor, palette, and dimensions;
        # only the T and payload change. Their order is shuffled below.
        pair, state = _make_pair(rng, plan, train_sizes[i], directions[i], zone("train"),
                                 train_colors[0] if i < 2 else train_colors[i],
                                 witness=i < 2, shared=shared if i == 1 else None,
                                 want_distractor=distractor_regime and (i == 0 or rng.random() < .55),
                                 rotation=r)
        if i == 0:
            shared = state
        train.append(pair)
    rng.shuffle(train)
    test = []
    for i in range(test_count):
        pair, _ = _make_pair(rng, plan, test_sizes[i], test_dirs[i], zone("test"),
                             test_colors[i], want_distractor=distractor_regime and rng.random() < .7,
                             rotation=rotation("test"))
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(pair):
    a, b = pair["input"], pair["output"]
    raw = bytes((a.shape[0], a.shape[1], b.shape[0], b.shape[1]))
    raw += a.astype(np.uint8).tobytes() + b.astype(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _runs(values):
    indices = np.flatnonzero(values)
    if len(indices) == 0:
        return []
    starts = [int(indices[0])]
    ends = []
    for previous, current in zip(indices[:-1], indices[1:]):
        if current > previous + 1:
            ends.append(int(previous) + 1)
            starts.append(int(current))
    ends.append(int(indices[-1]) + 1)
    return list(zip(starts, ends))


def _parse_canonical(inp):
    height, width = inp.shape
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    dividers = [r for r in range(height) if np.all(inp[r] == inp[r, 0]) and inp[r, 0] != bg]
    assert len(dividers) == 1
    divider = dividers[0]
    assert 0 < divider < height - 1
    divider_color = int(inp[divider, 0])
    assert np.count_nonzero(inp == divider_color) == width
    upper = inp[:divider]
    work = inp[divider + 1:]
    bands = _runs(np.any(upper != bg, axis=0))
    assert len(bands) == 3
    objects = []
    for left, right in bands:
        rows = np.flatnonzero(np.any(upper[:, left:right] != bg, axis=1))
        assert len(rows) > 0
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        obj = upper[top:bottom, left:right]
        assert np.any(obj != bg, axis=0).all()
        assert np.any(obj != bg, axis=1).all()
        objects.append((top, left, obj))
    rt, rl, reference = objects[0]
    mt, ml, marker = objects[1]
    pt, pl, payload = objects[2]
    assert reference.shape == payload.shape
    assert rt == pt
    assert marker.shape[0] == marker.shape[1] and marker.shape[0] in (3, 5)
    assert len(set(int(x) for x in marker.ravel() if x != bg)) == 1
    marker_color = int(marker[marker != bg][0])
    directions = [d for d in DIRECTIONS if np.array_equal(marker != bg, _t_mask(marker.shape[0], d))]
    assert len(directions) == 1
    direction = directions[0]
    assert marker_color not in (bg, divider_color)
    assert not any(np.any(obj == divider_color) for _, _, obj in objects)
    h, w = reference.shape
    matches = []
    for t in range(work.shape[0] - h + 1):
        for l in range(work.shape[1] - w + 1):
            if np.array_equal(work[t:t+h, l:l+w], reference):
                matches.append((t, l))
    assert len(matches) == 1
    at, al = matches[0]
    dt, dl = at + direction[0] * h, al + direction[1] * w
    assert _box_clear(work, dt, dl, h, w, bg)
    expected = work.copy()
    _put(expected, dt, dl, payload, bg)
    assert not np.array_equal(expected, work)
    return {"expected": expected, "work": work, "background": bg,
            "divider": divider_color, "reference": reference,
            "payload": payload, "marker_color": marker_color,
            "direction": direction, "anchor": (at, al),
            "destination": (dt, dl), "size": (h, w),
            "upper_height": divider, "upper_positions": (rl, ml, pl),
            "both_sides": all(_box_clear(work, at + d[0]*h, al + d[1]*w, h, w, bg)
                              for d in _axis_directions("vertical" if direction[0] else "horizontal")),
            "distractors": int(np.count_nonzero(work != bg) > np.count_nonzero(reference != bg))}


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    candidates = []
    for rotation in range(4):
        try:
            canonical_inp = np.rot90(inp, rotation)
            analysis = _parse_canonical(canonical_inp)
            analysis["rotation"] = rotation
            candidates.append(analysis)
        except (AssertionError, ValueError, IndexError):
            continue
    assert len(candidates) == 1, "instruction, anchor, direction, or answer ambiguous"
    analysis = candidates[0]
    assert np.array_equal(out, np.rot90(analysis["expected"], -analysis["rotation"]))
    return analysis


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    keys = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = (pair["input"].shape, pair["input"].tobytes())
            assert input_key not in keys
            keys.add(input_key)
            assert _digest(pair) not in SOURCE_PAIR_DIGESTS
            analyses[split].append(_analyze(pair))
    # The rotated T always means motion away from its crossbar. The two
    # opposite witness examples make the convention observable in training.
    witness = False
    for a in analyses["train"]:
        for b in analyses["train"]:
            if (a is not b and a["size"] == b["size"] and
                a["direction"] == tuple(-x for x in b["direction"]) and
                np.array_equal(a["reference"], b["reference"]) and
                a["anchor"] == b["anchor"] and a["both_sides"] and b["both_sides"]):
                witness = True
    assert witness
    if any(a["distractors"] for a in analyses["test"]):
        assert any(a["distractors"] for a in analyses["train"])


def self_test() -> None:
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        assert all(np.array_equal(a[name], b[name])
                   for split in ("train", "test")
                   for a, b in zip(puzzle[split], again[split])
                   for name in ("input", "output"))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(91 + train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(200 + (train or 0) + (test or 0), train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({(len(p["train"]), len(p["test"])) for p in puzzles}) >= 8
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    records = [_analyze(pair) for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert len({r["size"] for r in records}) >= 8
    assert len({r["direction"] for r in records}) == 4
    assert len({r["rotation"] for r in records}) == 4
    assert len({r["upper_height"] for r in records}) >= 5
    assert any(r["both_sides"] for r in records)
    assert any(r["distractors"] for r in records)
    assert len({r["background"] for r in records}) >= 6
    assert len({r["divider"] for r in records}) >= 6
