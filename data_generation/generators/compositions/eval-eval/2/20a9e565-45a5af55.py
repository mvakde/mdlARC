"""Growing ribbon on a rectangular coil: arithmetic dashes, continuous phase.

Generation paints explicit perimeter walks. The input-only solver instead uses
a closed-form edge-depth/perimeter-rank map and fits only visible tape runs.
No parent generator, file, seed, or demonstration enters solve().
"""
from __future__ import annotations

import math
import numpy as np


def _arc_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(int, copy=False)


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _legal_bounds(bounds):
    r0, r1, c0, c1 = bounds
    # Six distinct cells, in two disconnected three-cell components.
    return r1 > r0 and c1 > c0 and r1 - r0 + c1 - c0 >= 4


def _annotation(a):
    candidates = []
    for color in np.unique(a):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        bounds = (r0, r1, c0, c1)
        if _legal_bounds(bounds) and set(map(tuple, cells)) == _corners(bounds):
            candidates.append((int(color), bounds))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous pair of opposite L annotations")
    marker, bounds = candidates[0]
    r0, r1, c0, c1 = bounds
    region = a[r0:r1 + 1, c0:c1 + 1]
    blank = np.unique(region[region != marker])
    if len(blank) != 1:
        raise ValueError("the marked rectangle must be erased to one background")
    return marker, int(blank[0]), bounds


def _walk(h, w):
    """Ground-truth geometry: visit each nested border, clockwise, once."""
    cells = []
    top, bottom, left, right = 0, h - 1, 0, w - 1
    while top <= bottom and left <= right:
        cells.extend((top, c) for c in range(left, right + 1))
        cells.extend((r, right) for r in range(top + 1, bottom + 1))
        if bottom > top:
            cells.extend((bottom, c) for c in range(right - 1, left - 1, -1))
        if right > left:
            cells.extend((r, left) for r in range(bottom - 1, top, -1))
        top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
    return cells


def _index_map(h, w):
    """Solver geometry: boundary depth plus clockwise rank on that border."""
    r, c = np.indices((h, w))
    d = np.minimum.reduce((r, c, h - 1 - r, w - 1 - c))
    hh, ww, rr, cc = h - 2 * d, w - 2 * d, r - d, c - d
    start = h * w - hh * ww
    offset = np.where(rr == 0, cc,
                      np.where(cc == ww - 1, ww + rr - 1,
                               np.where(rr == hh - 1, 2 * ww + hh - 3 - cc,
                                        2 * ww + 2 * hh - 4 - rr)))
    offset = np.where(hh == 1, cc, np.where(ww == 1, rr, offset))
    return start + offset, d, offset


def _paint_tape(n, base, step, gap, palette, background):
    """Ground truth: paint whole dashes and their separating gaps in order."""
    result = np.full(n, background, dtype=int)
    pos, i = 0, 0
    while pos < n:
        width = base + step * i
        if width < 1:
            raise ValueError("a shrinking ribbon ends before the board is covered")
        result[pos:min(n, pos + width)] = palette[i % len(palette)]
        pos += width + gap
        i += 1
    return result


def _fit_input(a):
    marker, background, bounds = _annotation(a)
    indices, depth, offsets = _index_map(*a.shape)
    r0, r1, c0, c1 = bounds
    unknown = np.zeros(a.shape, dtype=bool)
    unknown[r0:r1 + 1, c0:c1 + 1] = True
    observed = np.empty(a.size, dtype=int)
    observed[indices] = a
    missing = np.empty(a.size, dtype=bool)
    missing[indices] = unknown
    cut = int(np.flatnonzero(missing)[0])
    prefix = observed[:cut]
    if not len(prefix) or prefix[0] in (background, marker):
        raise ValueError("the beginning of the ribbon must be visible")
    colors, widths, gaps = [], [], []
    pos = 0
    # Keep only complete colored-run + separator-run observations. A run cut
    # by the erased rectangle is never mistaken for a shorter object.
    while pos < cut:
        color = int(prefix[pos])
        if color in (background, marker):
            raise ValueError("malformed visible ribbon prefix")
        end = pos + 1
        while end < cut and prefix[end] == color:
            end += 1
        if end == cut:
            break
        if prefix[end] != background:
            raise ValueError("colored dashes require background separators")
        after = end + 1
        while after < cut and prefix[after] == background:
            after += 1
        if after == cut:
            break
        colors.append(color)
        widths.append(end - pos)
        gaps.append(after - end)
        pos = after
    if len(widths) < 3 or len(set(gaps)) != 1:
        raise ValueError("insufficient complete dashes or inconsistent spacing")
    base, step, gap = widths[0], widths[1] - widths[0], gaps[0]
    if any(width != base + i * step for i, width in enumerate(widths)):
        raise ValueError("dash lengths do not follow an arithmetic progression")
    periods = [p for p in range(1, 9) if len(colors) >= 2 * p + 1
               and all(color == colors[i % p] for i, color in enumerate(colors))]
    if not periods:
        raise ValueError("the visible prefix must show two color cycles plus one dash")
    palette = tuple(colors[:periods[0]])
    if marker in palette or background in palette:
        raise ValueError("annotation, background and content roles must be distinct")
    # Solver's tape path: enumerate interval endpoints, then look up their
    # colors. This is separate from the slice-painting label construction.
    ends, labels, cursor, i = [], [], 0, 0
    while cursor < a.size:
        width = base + i * step
        if width < 1:
            raise ValueError("nonpositive dash in the requested continuation")
        cursor += width
        ends.append(cursor)
        labels.append(palette[i % len(palette)])
        cursor += gap
        ends.append(cursor)
        labels.append(background)
        i += 1
    tape = np.asarray(labels)[np.searchsorted(ends, np.arange(a.size), side="right")]
    field = tape[indices]
    if not np.array_equal(field[~unknown], a[~unknown]):
        raise ValueError("the inferred continuation contradicts visible cells")
    info = dict(marker=marker, background=background, bounds=bounds,
                base=base, step=step, gap=gap, palette=palette,
                complete_prefix=len(widths), cut=cut, indices=indices,
                depth=depth, offsets=offsets, tape=tape, field=field)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the continuous winding ribbon and return its inclusive L crop."""
    info = _fit_input(_arc_grid(input_grid))
    r0, r1, c0, c1 = info["bounds"]
    return info["field"][r0:r1 + 1, c0:c1 + 1].copy()


def _crop(field, bounds):
    r0, r1, c0, c1 = bounds
    return field[r0:r1 + 1, c0:c1 + 1].copy()


def _alternatives(info):
    """Concrete competing rules used to check demonstration evidence."""
    field, tape, d, offset = (info[k] for k in ("field", "tape", "depth", "offsets"))
    h, w = field.shape
    start = h * w - (h - 2 * d) * (w - 2 * d)
    constant = _paint_tape(h * w, info["base"], 0, info["gap"],
                           info["palette"], info["background"])
    word = info["palette"]
    compressed = tuple(c for i, c in enumerate(word) if i == 0 or c != word[i - 1])
    compressed_tape = _paint_tape(h * w, info["base"], info["step"], info["gap"],
                                  compressed, info["background"])
    return {
        "constant_length": constant[info["indices"]],
        "no_color_cycle": np.where(field == info["background"],
                                   info["background"], info["palette"][0]),
        "row_major": tape.reshape(h, w),
        "counterclockwise": tape[_index_map(w, h)[0].T],
        "restart_each_frame": tape[offset],
        "uniform_frames": tape[start],
        "depth_only": tape[d],
        "compress_color_word": compressed_tape[info["indices"]],
    }


def _features(a, output=None):
    info = _fit_input(a)
    expected = _crop(info["field"], info["bounds"])
    if output is not None and not np.array_equal(output, expected):
        raise AssertionError("incorrect ribbon reconstruction")
    differences = {name: not np.array_equal(_crop(field, info["bounds"]), expected)
                   for name, field in _alternatives(info).items()}
    r0, r1, c0, c1 = info["bounds"]
    # A turn is a cell at one of the four corners of a nondegenerate frame.
    r, c = np.indices(a.shape)
    d = info["depth"]
    turns = ((r == d) | (r == a.shape[0] - 1 - d)) & (
        (c == d) | (c == a.shape[1] - 1 - d))
    colors = set(map(int, np.unique(expected))) - {info["background"]}
    signature_roles = {}
    normalized = tuple(signature_roles.setdefault(int(v), len(signature_roles)) for v in a.flat)
    return dict(shape=a.shape, output_shape=expected.shape, step=info["step"],
                base=info["base"], gap=info["gap"], period=len(info["palette"]),
                marker=info["marker"], background=info["background"],
                prefix=info["complete_prefix"], bounds=info["bounds"],
                output_colors=len(colors), has_blank=bool(np.any(expected == info["background"])),
                turn=bool(np.any(turns[r0:r1 + 1, c0:c1 + 1])),
                depth_count=len(np.unique(d[r0:r1 + 1, c0:c1 + 1])),
                center=bool(r0 <= (a.shape[0] - 1) // 2 <= r1
                            and c0 <= (a.shape[1] - 1) // 2 <= c1),
                word_repeat=any(info["palette"][i] == info["palette"][i - 1]
                                for i in range(1, len(info["palette"]))),
                alternatives=differences, structure=(a.shape, normalized))


def _requested_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _roles(rng):
    background = 0 if rng.random() < 0.55 else int(rng.integers(10))
    rest = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    return background, rest[0], tuple(rest[1:])


def _sample_word(rng, pool, period, repeat=False):
    if repeat:
        word = list(map(int, rng.choice(pool, size=period - 1, replace=False)))
        i = int(rng.integers(len(word)))
        word.insert(i, word[i])
        return tuple(word)
    richness = (period if period <= 2 or rng.random() < 0.6 else
                int(rng.integers(2, period)))
    word = list(map(int, rng.choice(pool, size=richness, replace=False)))
    while len(word) < period:
        word.insert(int(rng.integers(len(word) + 1)), int(rng.choice(word)))
    return tuple(word)


def _bounds_candidate(rng, h, w, kind):
    margin = int(rng.integers(1, min(6, (min(h, w) - 3) // 2 + 1)))
    ih, iw = h - 2 * margin, w - 2 * margin
    if kind == "large":
        # Vary centering and aspect ratio rather than always choosing a square.
        r0, r1 = margin, h - 1 - margin
        c0, c1 = margin, w - 1 - margin
        if rng.random() < 0.4 and ih > 5:
            r0 += int(rng.integers(0, ih // 3 + 1))
        if rng.random() < 0.4 and iw > 5:
            c1 -= int(rng.integers(0, iw // 3 + 1))
    else:
        hh = int(rng.integers(3, min(12, ih) + 1))
        ww = int(rng.integers(3, min(12, iw) + 1))
        if kind == "strip":
            if rng.integers(2):
                hh, ww = int(rng.integers(2, min(5, ih) + 1)), int(rng.integers(min(4, iw), iw + 1))
            else:
                hh, ww = int(rng.integers(min(4, ih), ih + 1)), int(rng.integers(2, min(5, iw) + 1))
        r0 = int(rng.integers(margin, h - margin - hh + 1))
        c0 = int(rng.integers(margin, w - margin - ww + 1))
        if kind == "corner":
            if rng.integers(2):
                r0 = margin
            else:
                r0 = h - margin - hh
            if rng.integers(2):
                c0 = margin
            else:
                c0 = w - margin - ww
        r1, c1 = r0 + hh - 1, c0 + ww - 1
    return r0, r1, c0, c1


def _make_pair(rng, shared_roles, varying_roles, shift, split, index, seen,
               first_shape=None):
    # Rejections reflect observable evidence/fit, not arbitrary output rules.
    for attempt in range(400):
        bg, marker, pool = _roles(rng) if varying_roles else shared_roles
        is_test = split == "test"
        if shift == 0:
            low, high = (24, 30) if is_test else (14, 22)
            h, w = map(int, rng.integers(low, high + 1, size=2))
        elif shift == 3:
            if is_test:
                h, w = int(rng.integers(12, 17)), int(rng.integers(26, 31))
                if rng.integers(2):
                    h, w = w, h
            else:
                h = int(rng.integers(16, 27))
                w = int(rng.integers(max(14, h - 3), min(28, h + 3) + 1))
        else:
            h, w = map(int, rng.integers(14, 31, size=2))
        if not is_test and index == 1 and ((h, w) == first_shape or h == w):
            continue
        if not is_test and index == 0:
            step, base, period = 1, int(rng.integers(1, 4)), int(rng.integers(2, 4))
        elif not is_test and index == 1:
            step, base, period = 0, int(rng.integers(1, 6)), int(rng.integers(3, 6))
        else:
            step = (2 if is_test and shift == 2 else
                    int(rng.integers(0, 2)) if not is_test and shift == 2 else
                    int(rng.choice([-1, 0, 1, 2], p=[0.12, 0.36, 0.40, 0.12])))
            period = int(rng.integers(1 if index > 0 else 2, 6))
            base = int(rng.integers(1, 7))
            if step == -1:
                # Shrinking dashes have enough total positive length to cover
                # this finite board; shorter boards leave useful hidden space.
                if shift not in (0, 3):
                    h, w = map(int, rng.integers(11, 19, size=2))
                base = math.ceil(math.sqrt(2 * h * w)) + int(rng.integers(0, 4))
                period = int(rng.integers(1, 4))
                # Preserve the size/aspect shift if shrinking would defeat it.
                if is_test and shift in (0, 3):
                    continue
        gap = int(rng.integers(1, 4))
        palette = _sample_word(rng, pool, period, repeat=not is_test and index == 1)
        try:
            tape = _paint_tape(h * w, base, step, gap, palette, bg)
        except ValueError:
            continue
        latent = np.full((h, w), bg, dtype=int)
        for cell, color in zip(_walk(h, w), tape):
            latent[cell] = color
        kind = ("large" if is_test and shift == 1 else
                "corner" if not is_test and index == 0 else
                str(rng.choice(["corner", "random", "strip", "large"])))
        # Smaller training windows make the masked-footprint shift genuine.
        if not is_test and shift == 1:
            kind = "random"
        for _ in range(35):
            bounds = _bounds_candidate(rng, h, w, kind)
            if not _legal_bounds(bounds):
                continue
            out = _crop(latent, bounds)
            inp = latent.copy()
            r0, r1, c0, c1 = bounds
            inp[r0:r1 + 1, c0:c1 + 1] = bg
            for cell in _corners(bounds):
                inp[cell] = marker
            key = _grid_key(inp)
            if key in seen:
                continue
            try:
                f = _features(inp, out)
            except (ValueError, AssertionError):
                continue
            if not f["output_colors"] or not f["has_blank"]:
                continue
            if not is_test and index < 2 and f["output_colors"] < 2:
                continue
            if not is_test and index == 0:
                if (not f["turn"] or f["depth_count"] < 2
                        or not all(value for name, value in f["alternatives"].items()
                                   if name != "compress_color_word")):
                    continue
            if not is_test and index == 1 and not f["alternatives"]["compress_color_word"]:
                continue
            if not is_test and shift == 1 and out.size > 70:
                continue
            if is_test and shift == 1 and out.size <= 70:
                continue
            seen.add(key)
            return {"input": inp, "output": out}
    raise RuntimeError("bounded search could not construct an informative ribbon crop")


def _grid_key(a):
    return a.shape, tuple(map(int, a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _requested_count(num_train, "num_train", 2, 10)
    num_test = _requested_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_roles, varying_roles = _roles(rng), bool(rng.integers(2))
    shift = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            first_shape = puzzle["train"][0]["input"].shape if puzzle["train"] else None
            puzzle[split].append(_make_pair(rng, shared_roles, varying_roles, shift,
                                          split, index, seen, first_shape))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
            key = _grid_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            f = _features(inp, out)
            assert f["marker"] not in out
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert f["prefix"] >= 2 * f["period"] + 1
            assert f["output_colors"] and f["has_blank"]
            assert f["depth_count"] >= 1
            assert np.array_equal(solve(inp.copy()), out)
            if split == "train":
                training.append(f)
    assert len({f["shape"] for f in training}) >= 2
    assert any(f["shape"][0] != f["shape"][1] for f in training)
    assert any(f["step"] > 0 and f["turn"] and f["depth_count"] >= 2
               and f["output_colors"] >= 2 for f in training)
    assert any(f["step"] == 0 and f["output_colors"] >= 2 for f in training)
    assert any(f["word_repeat"] for f in training)
    for alternative in training[0]["alternatives"]:
        assert any(f["alternatives"][alternative] for f in training), (
            "training permits shortcut " + alternative)


def self_test():
    """Bounded semantic fixtures, 42-seed diversity, counts and corruption."""
    # Every board cell, odd/even center, thin final strip and rectangular turn
    # is cross-checked by explicit traversal versus closed-form coordinates.
    for h in range(1, 31):
        for w in range(1, 31):
            walk = _walk(h, w)
            indices = _index_map(h, w)[0]
            assert len(walk) == len(set(walk)) == h * w
            assert [int(indices[cell]) for cell in walk] == list(range(h * w))
    # Fixtures are drawn without generate_puzzle: odd/square centers,
    # rectangular final strips, nonblack background, repeated code entries,
    # negative growth, and a crop reaching the board boundary.
    fixtures = [
        (9, 11, 1, 0, 1, (4, 4, 7), 0, 9, (2, 6, 2, 8)),
        (12, 13, 20, -1, 2, (0, 7), 3, 9, (4, 8, 4, 9)),
        (16, 19, 2, 1, 2, (3, 7, 1), 0, 9, (4, 11, 5, 14)),
        (9, 17, 1, 0, 1, (4, 7), 0, 9, (5, 8, 2, 11)),
        (30, 30, 1, 3, 1, tuple(range(1, 9)), 0, 9, (5, 24, 5, 24)),
        (20, 20, 45, -2, 1, (2, 4, 6), 0, 9, (7, 12, 7, 12)),
    ]
    for h, w, base, step, gap, word, bg, marker, bounds in fixtures:
        tape = _paint_tape(h * w, base, step, gap, word, bg)
        field = np.full((h, w), bg, dtype=int)
        for cell, color in zip(_walk(h, w), tape):
            field[cell] = color
        inp = field.copy()
        r0, r1, c0, c1 = bounds
        inp[r0:r1 + 1, c0:c1 + 1] = bg
        for cell in _corners(bounds):
            inp[cell] = marker
        assert np.array_equal(solve(inp), _crop(field, bounds))
    features, signatures, colors = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_grid_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"], pair["output"])
                f["split"] = split
                features.append(f)
                colors.update(map(int, np.unique(pair["input"])))
    assert len(signatures) == 42
    assert colors == set(range(10))
    assert len({f["structure"] for f in features}) >= 180
    assert len({f["shape"] for f in features}) >= 70
    assert len({f["output_shape"] for f in features}) >= 50
    assert {f["step"] for f in features} >= {-1, 0, 1, 2}
    assert {f["period"] for f in features} >= {1, 2, 3, 4, 5}
    assert {f["gap"] for f in features} == {1, 2, 3}
    assert len({f["base"] for f in features}) >= 8
    assert len({f["background"] for f in features}) >= 7
    assert len({f["marker"] for f in features}) >= 7
    assert sum(f["center"] for f in features) >= 25
    assert sum(f["turn"] and f["step"] != 0 for f in features) >= 40
    assert sum(f["word_repeat"] for f in features) >= 42
    assert any(f["output_shape"][0] == 2 or f["output_shape"][1] == 2 for f in features)
    assert any(max(f["shape"]) == 30 for f in features)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(9000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(7654321, 10, 4), generate_puzzle(7654321, 10, 4)
    assert all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], second[s])
               for k in ("input", "output"))
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(features),
            "structures": len({f["structure"] for f in features}),
            "input_shapes": len({f["shape"] for f in features}),
            "output_shapes": len({f["output_shape"] for f in features}),
            "steps": sorted({f["step"] for f in features}),
            "periods": sorted({f["period"] for f in features})}
