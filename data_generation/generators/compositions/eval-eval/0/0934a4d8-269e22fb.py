"""Complete an erased mirror atlas using a registered, recolored loose crop.

Only NumPy and the standard library are used. Inference receives one input and
knows neither the seed nor any construction parameters or training examples.
"""
from itertools import permutations

import numpy as np


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _canonical_pattern(a):
    candidates = []
    for code in range(8):
        transformed = _d4(a, code)
        roles = {}
        normalized = np.array([roles.setdefault(int(v), len(roles))
                               for v in transformed.flat], dtype=np.int8)
        candidates.append((transformed.shape, normalized.tobytes()))
    return min(candidates)


def _d4(a, code):
    """Four rotations, then four rotations after a left/right reflection."""
    return np.rot90(a if code < 4 else a[:, ::-1], code % 4).copy()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(a):
    remaining = set(zip(*np.nonzero(a)))
    boxes = []
    while remaining:
        start = remaining.pop()
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        rr, cc = zip(*cells)
        r, s, c, d = min(rr), max(rr) + 1, min(cc), max(cc) + 1
        if len(cells) != (s - r) * (d - c):
            raise ValueError("view components must be filled rectangles")
        boxes.append((int(r), int(s), int(c), int(d)))
    return boxes


def _unframe(view):
    color = int(view[0, 0])
    thickness = 0
    for t in (1, 2):
        if min(view.shape) <= 2 * t:
            break
        border = np.ones(view.shape, dtype=bool)
        border[t:-t, t:-t] = False
        if np.all(view[border] == color):
            thickness = t
    if thickness == 0:
        raise ValueError("a view needs a closed frame")
    content = view[thickness:-thickness, thickness:-thickness].copy()
    if np.any(content == color):
        raise ValueError("frame color cannot occur inside a view")
    return color, thickness, content


def _parse(input_grid):
    a = _grid(input_grid)
    boxes = _components(a)
    if len(boxes) != 2:
        raise ValueError("expected exactly two framed views")
    views = []
    for r, s, c, d in boxes:
        views.append(_unframe(a[r:s, c:d]))
    if sorted(t for _, t, _ in views) != [1, 2] or views[0][0] != views[1][0]:
        raise ValueError("expected common-color double and single frames")
    atlas = next(v for _, t, v in views if t == 2)
    fragment = next(v for _, t, v in views if t == 1)
    H, W = atlas.shape
    if min(H, W) < 7:
        raise ValueError("atlas is too small")
    masks = []
    for color in np.unique(atlas):
        rr, cc = np.nonzero(atlas == color)
        r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        central = (r <= (H - 1) // 2 and s > H // 2
                   and c <= (W - 1) // 2 and d > W // 2)
        if central and len(rr) == (s - r) * (d - c) and len(rr) < atlas.size:
            masks.append((int(color), (r, s, c, d)))
    if len(masks) != 1:
        raise ValueError("expected one solid central rectangular mask")
    mask, box = masks[0]
    palette = tuple(int(v) for v in np.unique(atlas[atlas != mask]))
    fragment_palette = tuple(int(v) for v in np.unique(fragment))
    if (not 2 <= len(palette) <= 4 or len(fragment_palette) != len(palette)
            or mask in fragment_palette or 0 in palette or 0 in fragment_palette):
        raise ValueError("invalid content palettes")
    return atlas, fragment, mask, box, palette, fragment_palette, views[0][0]


def _orbit_evidence(atlas, mask):
    H, W = atlas.shape
    bh, bw = (H + 1) // 2, (W + 1) // 2
    rows = np.minimum(np.arange(H), H - 1 - np.arange(H))
    cols = np.minimum(np.arange(W), W - 1 - np.arange(W))
    ids = rows[:, None] * bw + cols[None, :]
    visible = atlas != mask
    low = np.full(bh * bw, 10, dtype=np.int8)
    high = np.full(bh * bw, -1, dtype=np.int8)
    np.minimum.at(low, ids[visible], atlas[visible])
    np.maximum.at(high, ids[visible], atlas[visible])
    if np.any((high >= 0) & (low != high)):
        raise ValueError("visible atlas copies contradict reflection")
    return ids, high, (bh, bw)


def _infer(input_grid):
    atlas, fragment, mask, box, palette, fragment_palette, frame = _parse(input_grid)
    ids, known, base_shape = _orbit_evidence(atlas, mask)
    H, W = atlas.shape
    r0, r1, c0, c1 = box
    target_ids = ids[r0:r1, c0:c1]
    known_full = known[ids]
    unknown = known < 0
    if not np.any(unknown):
        raise ValueError("the loose fragment must supply erased orbit values")
    answers, registrations = {}, []
    for code in range(8):
        query = _d4(fragment, code)
        h, w = query.shape
        if h > H or w > W:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(known_full, (h, w))
        for assigned in permutations(palette):
            lookup = np.zeros(10, dtype=np.int8)
            lookup[list(fragment_palette)] = assigned
            recolored = lookup[query]
            fits = np.all((windows < 0) | (windows == recolored), axis=(-1, -2))
            for r, c in zip(*np.nonzero(fits)):
                patch_ids = ids[r:r + h, c:c + w]
                low = np.full(len(known), 10, dtype=np.int8)
                high = np.full(len(known), -1, dtype=np.int8)
                np.minimum.at(low, patch_ids, recolored)
                np.maximum.at(high, patch_ids, recolored)
                if np.any((high >= 0) & (low != high)):
                    continue  # even wholly erased orbits must agree internally
                if np.any(unknown & (high < 0)):
                    continue
                completed = np.where(unknown, high, known)
                prediction = completed[target_ids].copy()
                answers[_key(prediction)] = prediction
                footprint = np.zeros(atlas.shape, dtype=bool)
                footprint[r:r + h, c:c + w] = True
                omitted = ~footprint[r0:r1, c0:c1]
                registrations.append({
                    "orientation": code, "offset": (int(r), int(c)),
                    "shape": (h, w), "mapping": tuple(assigned),
                    "reflection_needed": int(np.sum(omitted & (known[target_ids] < 0))),
                    "direct_missing": int(np.sum(omitted)),
                    "seams": (bool(r <= (H - 1) // 2 < r + h - 1),
                              bool(c <= (W - 1) // 2 < c + w - 1)),
                })
    if len(answers) != 1:
        raise ValueError(f"fragment permits {len(answers)} distinct hole answers")
    answer = next(iter(answers.values()))
    record = {"atlas": atlas, "fragment": fragment, "mask": mask, "frame": frame,
              "box": box, "palette": palette, "fragment_palette": fragment_palette,
              "base_shape": base_shape, "ids": ids, "known": known,
              "erased_orbits": int(np.sum(unknown)),
              "native_cells": int(np.sum(known[target_ids] >= 0)),
              "erased_depth": tuple(int(len(np.unique(np.nonzero(unknown.reshape(base_shape))[k])))
                                    for k in (0, 1)),
              "registrations": registrations}
    return answer, record


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique target-oriented hole, using only the supplied grid."""
    return _infer(input_grid)[0]


def _pattern(rng, shape, count, style):
    h, w = shape
    if style == 0:
        result = rng.integers(count, size=shape, dtype=np.int8)
    elif style == 1:
        small = rng.integers(count, size=((h + 1) // 2, (w + 1) // 2), dtype=np.int8)
        result = np.repeat(np.repeat(small, 2, 0), 2, 1)[:h, :w].copy()
    elif style == 2:
        weights = rng.dirichlet(np.full(count, 1.3)) + .12
        result = rng.choice(count, size=shape, p=weights / weights.sum()).astype(np.int8)
    else:
        result = np.full(shape, int(rng.integers(count)), dtype=np.int8)
        for j in range(int(rng.integers(4, 9))):
            color = int(rng.integers(count))
            r, s = sorted(int(v) for v in rng.choice(h + 1, 2, replace=False))
            c, d = sorted(int(v) for v in rng.choice(w + 1, 2, replace=False))
            kind = int(rng.integers(4))
            if kind == 0:
                result[r:s, c:d] = color
            elif kind == 1:
                result[r:s, c] = color
                result[s - 1, c:d] = color
            elif kind == 2:
                result[r:s:2, c:d] = color
            else:
                for t in range(max(s - r, d - c)):
                    result[min(s - 1, r + t), min(d - 1, c + t)] = color
        # Sparse landmarks retain the structured field while improving registration.
        for p in rng.choice(h * w, min(count + 1, h * w), replace=False):
            result.flat[p] = int(rng.integers(count))
    return result


def _frame(a, color, thickness):
    return np.pad(a, thickness, constant_values=color).astype(np.int8)


def _pack(rng, atlas, fragment):
    """Free placement of the two views, with a wholly empty separating band."""
    choices = []
    for axis in (0, 1):
        for gap in (1, 2, 3):
            lengths = [max(atlas.shape[k], fragment.shape[k]) for k in (0, 1)]
            lengths[axis] = atlas.shape[axis] + fragment.shape[axis] + gap
            if max(lengths) <= 30:
                choices.append((axis, gap, lengths))
    if not choices:
        return None
    axis, gap, lengths = choices[int(rng.integers(len(choices)))]
    shape = tuple(n + int(rng.integers(min(4, 30 - n) + 1)) for n in lengths)
    a = np.zeros(shape, dtype=np.int8)
    objects = [atlas, fragment]
    if rng.random() < .5:
        objects.reverse()
    start = int(rng.integers(shape[axis] - lengths[axis] + 1))
    for obj in objects:
        offset = [0, 0]
        offset[axis] = start
        offset[1 - axis] = int(rng.integers(shape[1 - axis] - obj.shape[1 - axis] + 1))
        r, c = offset
        a[r:r + obj.shape[0], c:c + obj.shape[1]] = obj
        start += obj.shape[axis] + gap
    return a


def _role_colors(rng, count):
    frame, mask = (int(v) for v in rng.choice(np.arange(1, 10), 2, replace=False))
    available = [v for v in range(1, 10) if v not in (frame, mask)]
    palette = tuple(int(v) for v in rng.choice(available, count, replace=False))
    if rng.random() < .4:
        fragment_palette = tuple(int(v) for v in rng.permutation(palette))
    else:
        fragment_palette = tuple(int(v) for v in rng.choice(available, count, replace=False))
    return frame, mask, palette, fragment_palette


def _make_pair(rng, test, swapped, preferences, seen, need_recolor=False):
    for attempt in range(180):
        H, W = (int(v) for v in rng.integers(12 if test else 8, 17, 2))
        bh, bw = (H + 1) // 2, (W + 1) // 2
        if test:
            dr, dc = (int(v) for v in rng.integers(2, 4, 2))
            if rng.random() < .5:
                dr = 3
            else:
                dc = 3
        else:
            dr, dc = (int(v) for v in rng.integers(1, 3, 2))
        lo_r, lo_c = bh - dr, bw - dc
        er, ec = int(rng.integers(1, min(3, lo_r) + 1)), int(rng.integers(1, min(3, lo_c) + 1))
        # Erasure extends farther toward the lower/right atlas edges initially.
        box = (lo_r, H - lo_r + er, lo_c, W - lo_c + ec)
        r0, r1, c0, c1 = box
        if r1 > H or c1 > W:
            continue
        ar, ac = int(rng.integers(1, min(4, lo_r) + 1)), int(rng.integers(1, min(4, lo_c) + 1))
        # A fragment can end on or beyond either central seam.
        qr, qc = lo_r - ar, lo_c - ac
        qs = min(H, bh + int(rng.integers(0, 3)))
        qd = min(W, bw + int(rng.integers(0, 3)))
        if (qs - qr) * (qd - qc) >= (r1 - r0) * (c1 - c0):
            continue
        count = preferences["count"] if rng.random() < .65 else int(rng.integers(2, 5))
        colors = (preferences["colors"] if preferences["stable"] and count == preferences["count"]
                  else _role_colors(rng, count))
        frame, mask, palette, fragment_palette = colors
        if need_recolor and set(palette) == set(fragment_palette):
            unused = [v for v in range(1, 10) if v not in (frame, mask, *palette)]
            fragment_palette = (int(rng.choice(unused)), *fragment_palette[1:])
        # The bounded final attempts change a sampling preference, never the rule.
        style = 0 if attempt >= 120 else (preferences["style"] if rng.random() < .7 else int(rng.integers(4)))
        base = _pattern(rng, (bh, bw), count, style)
        rows = np.minimum(np.arange(H), H - 1 - np.arange(H))
        cols = np.minimum(np.arange(W), W - 1 - np.arange(W))
        full = np.asarray(palette, dtype=np.int8)[base[rows[:, None], cols[None, :]]]
        damaged = full.copy()
        damaged[r0:r1, c0:c1] = mask
        # Reflect the damaged view to vary the hole's longer side and crop quadrant.
        crop_support = np.zeros((H, W), dtype=bool)
        crop_support[qr:qs, qc:qd] = True
        for axis in (0, 1):
            if rng.random() < .5:
                damaged = np.flip(damaged, axis).copy()
                crop_support = np.flip(crop_support, axis).copy()
        rr, cc = np.nonzero(crop_support)
        qr, qs, qc, qd = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        crop = full[qr:qs, qc:qd]
        lookup = np.zeros(10, dtype=np.int8)
        lookup[list(palette)] = fragment_palette
        crop = lookup[crop]
        code = int(rng.choice([1, 3, 5, 7] if swapped else [0, 2, 4, 6]))
        crop = _d4(crop, code)
        packed = _pack(rng, _frame(damaged, frame, 2), _frame(crop, frame, 1))
        if packed is None or _key(packed) in seen:
            continue
        try:
            predicted, record = _infer(packed)
        except ValueError:
            continue
        r0, r1, c0, c1 = record["box"]
        output = full[r0:r1, c0:c1].copy()
        if (not np.array_equal(predicted, output) or len(np.unique(output)) < 2
                or np.array_equal(output, output[::-1, ::-1])
                or record["native_cells"] == 0
                or {x["orientation"] % 2 for x in record["registrations"]} != {int(swapped)}
                or any(x["reflection_needed"] == 0 for x in record["registrations"])):
            continue
        seen.add(_key(packed))
        return {"input": packed, "output": output}
    raise RuntimeError("bounded atlas/fragment construction exhausted")


def _counts(num_train, num_test):
    for name, n, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not low <= n <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    count = int(rng.integers(2, 5))
    preferences = {"count": count, "style": int(rng.integers(4)),
                   "stable": bool(rng.integers(2)), "colors": _role_colors(rng, count)}
    puzzle, seen = {"train": [], "test": []}, set()
    for j in range(num_train):
        swapped = bool(j % 2) if j < 2 else bool(rng.integers(2))
        puzzle["train"].append(_make_pair(rng, False, swapped, preferences, seen, need_recolor=j == 0))
    for _ in range(num_test):
        puzzle["test"].append(_make_pair(rng, True, bool(rng.integers(2)), preferences, seen))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check actual views, orbit evidence, ambiguity, both ablations and split."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    demonstrated_axes, demonstrated_recolor = set(), False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            try:
                predicted, record = _infer(inp)
            except ValueError as exc:
                raise AssertionError(str(exc)) from exc
            assert np.array_equal(predicted, out), "incorrect restored rectangle"
            assert len(np.unique(out)) >= 2 and not np.array_equal(out, out[::-1, ::-1])
            assert record["fragment"].size < out.size, "fragment must be a proper source of evidence"
            assert record["erased_orbits"] > 0, "B ablation must lose orbit values"
            assert record["native_cells"] > 0, "retain atlas-based recovery too"
            assert all(x["reflection_needed"] > 0 for x in record["registrations"]), "A ablation must leave holes"
            dr, dc = record["erased_depth"]
            if split == "train":
                assert 1 <= dr <= 2 and 1 <= dc <= 2
                classes = {x["orientation"] % 2 for x in record["registrations"]}
                if len(classes) == 1:
                    demonstrated_axes.update(classes)
                demonstrated_recolor |= all(x["mapping"] != record["fragment_palette"]
                                            for x in record["registrations"])
            else:
                assert 2 <= dr <= 3 and 2 <= dc <= 3 and max(dr, dc) == 3
            atlas = record["atlas"].copy()
            r, s, c, d = record["box"]
            atlas[r:s, c:d] = out
            assert np.array_equal(atlas, atlas[::-1]) and np.array_equal(atlas, atlas[:, ::-1])
            assert record["mask"] not in out and record["frame"] not in out
    assert demonstrated_axes == {0, 1}, "training must expose both axis behaviors"
    assert demonstrated_recolor, "training must require a color-role correspondence"
    return None


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded 42-seed structural diversity plus explicit invariant failures."""
    signatures, structures, input_shapes, output_shapes = set(), set(), set(), set()
    atlas_shapes, fragment_shapes, masks, frames, palettes, offsets, depths = (set() for _ in range(7))
    odd = rectangles = seam_crossing = swapped = nonidentity_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed in (0, 17, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                answer, rec = _infer(pair["input"])
                assert np.array_equal(answer, pair["output"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(answer.shape)
                atlas_shapes.add(rec["atlas"].shape)
                fragment_shapes.add(rec["fragment"].shape)
                masks.add(rec["mask"])
                frames.add(rec["frame"])
                palettes.add(rec["palette"])
                depths.add(rec["erased_depth"])
                odd += any(n % 2 for n in rec["atlas"].shape)
                rectangles += rec["atlas"].shape[0] != rec["atlas"].shape[1]
                for reg in rec["registrations"]:
                    offsets.add(reg["offset"])
                reg = rec["registrations"][0]
                seam_crossing += any(reg["seams"])
                swapped += reg["orientation"] % 2 == 1
                nonidentity_palette += reg["mapping"] != rec["fragment_palette"]
                full = rec["atlas"].copy()
                r, s, c, d = rec["box"]
                full[r:s, c:d] = answer
                structures.add(_canonical_pattern(full))
    assert len(signatures) == 42 and len(structures) >= 190
    assert len(input_shapes) >= 55 and len(output_shapes) >= 35
    assert len(atlas_shapes) >= 40 and len(fragment_shapes) >= 25
    assert len(masks) == 9 and len(frames) == 9 and len(palettes) >= 35
    assert len(offsets) >= 40 and {(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)} <= depths
    assert odd >= 100 and rectangles >= 120 and seam_crossing >= 30 and swapped >= 50
    assert nonidentity_palette >= 100
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(2026, 2, 1)
    assert _signature(generate_puzzle(-13, 2, 1)) == _signature(generate_puzzle(-13, 2, 1))
    # Concrete counter-completions demonstrate information lost by each ablation.
    answer, rec = _infer(p["train"][0]["input"])
    r, s, c, d = rec["box"]
    full = rec["atlas"].copy()
    full[r:s, c:d] = answer
    absent_id = int(np.flatnonzero(rec["known"] < 0)[0])
    alternate = full.copy()
    old_value = int(full[rec["ids"] == absent_id][0])
    replacement = next(v for v in rec["palette"] if v != old_value)
    alternate[rec["ids"] == absent_id] = replacement
    assert np.array_equal(alternate, alternate[::-1])
    assert np.array_equal(alternate, alternate[:, ::-1])
    assert np.array_equal(alternate[rec["atlas"] != rec["mask"]],
                          full[rec["atlas"] != rec["mask"]])
    assert not np.array_equal(alternate[r:s, c:d], answer)
    # Both completions fit the atlas after the fragment (B) has been discarded.
    reg = rec["registrations"][0]
    qr, qc = reg["offset"]
    qh, qw = reg["shape"]
    direct = np.zeros(full.shape, dtype=bool)
    direct[qr:qr + qh, qc:qc + qw] = True
    missing = (rec["atlas"] == rec["mask"]) & ~direct
    pr, pc = (int(v) for v in np.argwhere(missing)[0])
    alternate = full.copy()
    alternate[pr, pc] = next(v for v in rec["palette"] if v != full[pr, pc])
    assert np.array_equal(alternate[direct], full[direct])
    assert np.array_equal(alternate[rec["atlas"] != rec["mask"]],
                          full[rec["atlas"] != rec["mask"]])
    assert not np.array_equal(alternate[r:s, c:d], answer)
    # Both fit all directly observed pixels when reflection (A) is discarded.
    # Independent per-pair color bijections preserve grammar and correspondence.
    recolored = {s: [] for s in p}
    rng = np.random.default_rng(843)
    for split in p:
        for pair in p[split]:
            lut = np.r_[0, rng.permutation(np.arange(1, 10))].astype(np.int8)
            recolored[split].append({k: lut[v] for k, v in pair.items()})
    validate_generated_puzzle(recolored)
    # Rotating the entire input rotates the atlas target as well.
    rotated = {s: [{k: np.rot90(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(rotated)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong target accepted")
    # Destroy a visible reflection equality without modifying the crop.
    inp = p["train"][0]["input"].copy()
    _, rec = _infer(inp)
    atlas = rec["atlas"].copy()
    atlas[0, 0] = next(v for v in rec["palette"] if v != atlas[0, 0])
    try:
        _orbit_evidence(atlas, rec["mask"])
    except ValueError:
        pass
    else:
        raise AssertionError("contradictory atlas evidence accepted")
    return {"seeds": 42, "pairs": 210, "atlas_structures": len(structures),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "atlas_shapes": len(atlas_shapes), "fragment_shapes": len(fragment_shapes)}


if __name__ == "__main__":
    print(self_test())
