"""Anchor-routed prefix cascades. Only NumPy and the Python standard library.

The source carries a seed, not a completed triangle. Reflect its placement and
order first; evolve in destination screen coordinates second. Anchor side is the
logical cell size. No generation metadata, examples, or files are used by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 436809650, 207822469, 872764033, 929065275, 434028243,
              992883541, 40928876, 259451036, 390998368, 345556813,
              743733006, 567131857, 595990147, 212532802, 456456134,
              338785052, 563815716, 561989414, 418911715, 898459780,
              554080044, 629201122, 658946203, 264669628, 894499302,
              86071446, 825158799, 501756170, 957741579, 143071628,
              285870544, 975951127, 902110581, 293072224, 982977006,
              465099007, 398049815, 317880010, 696468994, 823827343,
              625289102)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _bounds(h, w, r, c):
    return ((0, r, 0, c), (0, r, c + 1, w),
            (r + 1, h, 0, c), (r + 1, h, c + 1, w))


def _signs(reference, target):
    return (1 if reference // 2 == target // 2 else -1,
            1 if reference % 2 == target % 2 else -1)


def _square(panel, color):
    pts = np.argwhere(panel == color)
    assert len(pts), "missing anchor"
    r, c = pts.min(0).tolist()
    end_r, end_c = (pts.max(0) + 1).tolist()
    side = end_r - r
    assert side == end_c - c and len(pts) == side * side, "anchor is not one solid square"
    return r, c, side


def _decode(value):
    a = _grid(value)
    rows = [r for r in range(1, a.shape[0] - 1) if np.all(a[r] == a[r, 0])]
    cols = [c for c in range(1, a.shape[1] - 1) if np.all(a[:, c] == a[0, c])]
    crosses = [(r, c) for r in rows for c in cols if a[r, 0] == a[0, c]]
    assert len(crosses) == 1, "one full cross required"
    r, c = crosses[0]
    divider = int(a[r, c])
    bounds = _bounds(*a.shape, r, c)
    panels = [a[y0:y1, x0:x1] for y0, y1, x0, x1 in bounds]
    values = np.concatenate([p.ravel() for p in panels])
    counts = np.bincount(values.astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > len(values) and bg != divider, "background must be the strict majority"
    colors = [set(map(int, np.unique(p))) for p in panels]
    shared = set.intersection(*colors) - {bg}
    assert len(shared) == 1, "one common anchor color required"
    anchor = next(iter(shared))
    assert anchor != divider
    refs = [i for i, cs in enumerate(colors) if cs - {bg, anchor}]
    assert len(refs) == 1, "one reference required"
    ref = refs[0]
    signals = colors[ref] - {bg, anchor}
    assert len(signals) == 2 and divider not in signals, "exactly two distinct signal colors"
    assert all(colors[p] == {bg, anchor} for p in range(4) if p != ref)
    anchors = [_square(p, anchor) for p in panels]
    panel = panels[ref]
    ar, ac, side = anchors[ref]
    # The lower-left two-pixel swatch is fixed geometry, not copied content.
    swatch = tuple(map(int, panel[-1, :2]))
    assert len(set(swatch)) == 2 and set(swatch) == signals, "missing signal key"
    signal_mask = np.isin(panel, list(signals))
    signal_mask[-1, :2] = False
    pts = np.argwhere(signal_mask)
    assert len(pts), "missing seed"
    sr, sc = pts.min(0).tolist()
    er, ec = (pts.max(0) + 1).tolist()
    assert er - sr == side and (ec - sc) % side == 0, "seed cells must match reference anchor"
    length = (ec - sc) // side
    assert length >= 3 and length % 2 == 1, "odd seed length >=3 required"
    assert (ar - sr) % side == 0 and (sc - ac) % side == 0
    d, e = (ar - sr) // side, (sc - ac) // side
    assert d >= 1 and e >= 1, "seed must be above and right of its anchor"
    assert ar + side <= panel.shape[0] - 2, "blank row must separate object and key"
    seq = panel[sr, sc:ec:side].copy()
    assert int(seq[0]) in signals and int(seq[-1]) in signals
    assert set(map(int, seq)) <= {bg, *signals}
    reconstruction = np.full(panel.shape, bg, dtype=np.int8)
    reconstruction[ar:ar + side, ac:ac + side] = anchor
    reconstruction[sr:er, sc:ec] = np.repeat(seq, side)[None, :]
    reconstruction[-1, :2] = swatch
    assert np.array_equal(reconstruction, panel), "unexpected source detail or nonuniform seed blocks"
    return {"grid": a, "bounds": bounds, "panels": panels, "background": bg,
            "divider": divider, "anchor": anchor, "reference": ref,
            "anchors": anchors, "signals": swatch, "sequence": seq, "d": d, "e": e}


def _swap(values, signals):
    a, b = signals
    return np.where(values == a, b, np.where(values == b, a, values)).astype(np.int8)


def _pattern(seq, unit, d, e, vertical, horizontal, bg, signals,
             mode="correct"):
    """Recursive construction; placement offsets are relative to anchor top-left."""
    n = len(seq)
    values = seq.copy() if horizontal == 1 else seq[::-1].copy()
    dy = -d if vertical == 1 else d
    dx = e if horizontal == 1 else -(e + n - 1)
    if mode == "omit_anchor":
        unit, dy, dx = 1, -d, e
        values = seq.copy()
    levels = (n + 1) // 2
    canvas = np.full((levels * unit, n * unit), bg, dtype=np.int8)
    for k in range(levels):
        if k and mode == "seed_only":
            break
        if k:
            values = values[1:-1] if mode == "symmetric" else values[:-2]
            if mode != "no_swap":
                values = _swap(values, signals)
        left = 0 if mode == "no_inset" else k * unit
        canvas[k * unit:(k + 1) * unit, left:left + len(values) * unit] = np.repeat(values, unit)
    return canvas, dy * unit, dx * unit


def _paste(panel, raster, top, left):
    y0, x0 = max(0, top), max(0, left)
    y1 = min(panel.shape[0], top + raster.shape[0])
    x1 = min(panel.shape[1], left + raster.shape[1])
    if y0 < y1 and x0 < x1:
        panel[y0:y1, x0:x1] = raster[y0 - top:y1 - top, x0 - left:x1 - left]


def _predict(data, mode="correct"):
    out = data["grid"].copy()
    for p, (y0, y1, x0, x1) in enumerate(data["bounds"]):
        if p == data["reference"]:
            continue
        ar, ac, unit = data["anchors"][p]
        sy, sx = _signs(data["reference"], p)
        if mode == "no_reflection":
            sy, sx = 1, 1
        if mode == "no_scale":
            unit = data["anchors"][data["reference"]][2]
        if mode == "reflect_cascade":
            # Plausible competing rule: reflect the entire source-side evolution.
            raster, dy, dx = _pattern(data["sequence"], unit, data["d"], data["e"],
                                      1, 1, data["background"], data["signals"])
            if sy == -1:
                raster = raster[::-1]
                dy = unit - dy - raster.shape[0]
            if sx == -1:
                raster = raster[:, ::-1]
                dx = unit - dx - raster.shape[1]
        else:
            raster, dy, dx = _pattern(data["sequence"], unit, data["d"], data["e"],
                                      sy, sx, data["background"], data["signals"], mode)
        _paste(out[y0:y1, x0:x1], raster, ar + dy, ac + dx)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible roles, transform only the seed, then evolve and clip."""
    return _predict(_decode(input_grid))


def _pixel_check(data, output):
    """Closed-form destination-pixel check, separate from recursive raster paint."""
    expected = data["grid"].copy()
    records = []
    bg, signals, n = data["background"], data["signals"], len(data["sequence"])
    for p, (y0, y1, x0, x1) in enumerate(data["bounds"]):
        if p == data["reference"]:
            continue
        ar, ac, u = data["anchors"][p]
        sy, sx = _signs(data["reference"], p)
        seq = data["sequence"] if sx == 1 else data["sequence"][::-1]
        top = ar + (-data["d"] if sy == 1 else data["d"]) * u
        left = ac + (data["e"] if sx == 1 else -data["e"] - n + 1) * u
        panel = expected[y0:y1, x0:x1]
        total, visible, levels, clipped_sides = 0, 0, set(), set()
        for k in range((n + 1) // 2):
            for j in range(n - 2 * k):
                value = int(seq[j])
                if value == bg:
                    continue
                if k % 2:
                    value = signals[1] if value == signals[0] else signals[0]
                r, c = top + k * u, left + (k + j) * u
                total += u * u
                for dy in range(u):
                    for dx in range(u):
                        yy, xx = r + dy, c + dx
                        if 0 <= yy < panel.shape[0] and 0 <= xx < panel.shape[1]:
                            panel[yy, xx] = value
                            visible += 1
                            levels.add(k)
                        else:
                            if yy < 0:
                                clipped_sides.add("top")
                            if yy >= panel.shape[0]:
                                clipped_sides.add("bottom")
                            if xx < 0:
                                clipped_sides.add("left")
                            if xx >= panel.shape[1]:
                                clipped_sides.add("right")
        assert 0 in levels and 1 in levels, "each destination must show seed and first recurrence"
        assert visible >= max(2, total // 8), "too little visible cascade"
        records.append({"panel": p, "unit": u, "total": total, "visible": visible,
                        "lost": total - visible, "sides": sorted(clipped_sides),
                        "visible_levels": sorted(levels)})
    assert np.array_equal(expected, output), "wrong anchor placement, scale, reflection, recurrence, or clipping"
    return records


def _placements(shape, unit, raster, dy, dx, bg, clipped):
    """Evaluate all <=900 anchor positions with bounded summed-area arrays."""
    ph, pw = shape
    if min(ph, pw) < unit:
        return np.empty((0, 2), dtype=int)
    rr, cc = np.indices((ph - unit + 1, pw - unit + 1))
    top, left = rr + dy, cc + dx
    mask = (raster != bg).astype(np.int32)
    prefix = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = raster.shape

    def area(lower, upper):
        y0 = np.clip(np.maximum(-top, lower), 0, h)
        y1 = np.clip(np.minimum(ph - top, upper), 0, h)
        y1 = np.maximum(y0, y1)
        x0, x1 = np.clip(-left, 0, w), np.clip(pw - left, 0, w)
        return prefix[y1, x1] - prefix[y0, x1] - prefix[y1, x0] + prefix[y0, x0]

    total = int(mask.sum())
    visible = area(0, h)
    ok = ((visible < total) if clipped else (visible == total))
    ok &= visible >= max(2, total // 8)
    ok &= (area(0, unit) > 0) & (area(unit, 2 * unit) > 0)
    return np.column_stack((rr[ok], cc[ok]))


def _motif(seq, bg, signals):
    roles = {bg: 0, signals[0]: 1, signals[1]: 2}
    values = tuple(roles[int(c)] for c in seq)
    inverse = tuple(0 if v == 0 else 3 - v for v in values)
    return min(values, inverse)


def _sequence(rng, length, style, bg, signals):
    a, b = signals
    if style == "solid":
        return np.full(length, int(rng.choice(signals)), dtype=np.int8)
    if style in ("alternating", "gapped"):
        phase = int(rng.integers(2))
        seq = np.array([signals[(i + phase) % 2] for i in range(length)], dtype=np.int8)
    elif style == "runs":
        seq = np.empty(length, dtype=np.int8)
        at, phase = 0, int(rng.integers(2))
        while at < length:
            size = int(rng.integers(1, min(4, length - at) + 1))
            seq[at:at + size] = signals[phase]
            at, phase = at + size, 1 - phase
    else:
        seq = rng.choice([bg, a, b], size=length, p=[.30, .35, .35]).astype(np.int8)
    seq[0], seq[-1] = int(rng.choice(signals)), int(rng.choice(signals))
    if style == "diagnostic":
        # A retained gap, both phases, and two different ends: not a solid triangle.
        seq[0], seq[1], seq[2], seq[-1] = a, bg, b, b
        if length >= 7 and rng.random() < .5:
            seq[3] = bg
    elif style == "gapped" and length >= 5:
        gaps = rng.choice(np.arange(1, length - 1),
                          size=int(rng.integers(1, length - 2)), replace=False)
        seq[gaps] = bg
    return seq


def _axis(rng, minima):
    low = sum(minima) + 1
    if low > 30:
        return None
    total = int(rng.integers(low, 31))
    extra = total - low
    a = int(rng.integers(extra + 1))
    return minima[0] + a, minima[1] + extra - a


def _make_pair(rng, seq, roles, ref, clip_count, witness=False, shift=None):
    bg, divider, anchor, a, b = roles
    n = len(seq)
    d, e = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    source_unit = int(rng.choice([1, 2])) if witness else int(rng.choice([1, 1, 1, 2, 2, 3]))
    units = {p: int(rng.choice([1, 1, 2])) for p in range(4)}
    units[ref] = source_unit
    if witness:
        units = {p: 1 for p in range(4)}
        units[ref] = source_unit
        # Both one- and two-pixel cascades must actually be demonstrated,
        # including when the source itself uses two-pixel cells.
        units[ref ^ 2] = 2
    elif shift == "scale":
        units[int(rng.choice([p for p in range(4) if p != ref]))] = int(rng.choice([3, 4]))
    targets = [p for p in range(4) if p != ref]
    clipped = set(map(int, rng.choice(targets, size=clip_count, replace=False)))
    patterns, min_h, min_w = {}, [3, 3], [3, 3]
    for p in range(4):
        u = units[p]
        if p == ref:
            mh, mw = (d + 1) * u + 2, (e + n) * u
        else:
            raster, dy, dx = _pattern(seq, u, d, e, *_signs(ref, p), bg, (a, b))
            patterns[p] = raster, dy, dx
            if p in clipped:
                mh, mw = (d + 2) * u, (e + 2) * u
            else:
                mh = max(u, dy + raster.shape[0]) - min(0, dy)
                mw = max(u, dx + raster.shape[1]) - min(0, dx)
        min_h[p // 2] = max(min_h[p // 2], mh)
        min_w[p % 2] = max(min_w[p % 2], mw)
    hs, ws = _axis(rng, min_h), _axis(rng, min_w)
    if hs is None or ws is None:
        return None
    inp = np.full((sum(hs) + 1, sum(ws) + 1), bg, dtype=np.int8)
    inp[hs[0], :], inp[:, ws[0]] = divider, divider
    bounds = _bounds(*inp.shape, hs[0], ws[0])
    placements = {}
    for p, (y0, y1, x0, x1) in enumerate(bounds):
        panel, u = inp[y0:y1, x0:x1], units[p]
        if p == ref:
            ar = int(rng.integers(d * u, panel.shape[0] - u - 1))
            ac = int(rng.integers(0, panel.shape[1] - (e + n) * u + 1))
            panel[ar - d * u:ar - (d - 1) * u, ac + e * u:ac + (e + n) * u] = np.repeat(seq, u)
            panel[-1, :2] = (a, b)
        else:
            raster, dy, dx = patterns[p]
            candidates = _placements(panel.shape, u, raster, dy, dx, bg, p in clipped)
            if not len(candidates):
                return None
            ar, ac = map(int, candidates[int(rng.integers(len(candidates)))])
        panel[ar:ar + u, ac:ac + u] = anchor
        placements[p] = ar, ac
    out = inp.copy()
    for p in targets:
        y0, y1, x0, x1 = bounds[p]
        ar, ac = placements[p]
        raster, dy, dx = patterns[p]
        _paste(out[y0:y1, x0:x1], raster, ar + dy, ac + dx)
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.permutation(10)[:5]))
    varying_palette, varying_ref = bool(rng.integers(2)), bool(rng.integers(4))
    base_ref = int(rng.integers(4))
    shift = str(rng.choice(["span", "clipping", "scale"]))
    puzzle, used_inputs, used_motifs = {"train": [], "test": []}, set(), set()
    train_styles = ["diagnostic", "solid"] + [str(rng.choice(["gapped", "runs", "random", "alternating"]))
                                                     for _ in range(num_train - 2)]
    test_styles = ["gapped", "runs", "solid", "alternating"]
    rng.shuffle(test_styles)
    for split, styles in (("train", train_styles), ("test", test_styles[:num_test])):
        for index, style in enumerate(styles):
            witness = split == "train" and index == 0
            ref = (base_ref if not varying_ref else (base_ref + index + (1 if split == "test" else 0)) % 4)
            if split == "train":
                clip_count = 0 if witness else (1 if index == 1 else int(rng.integers(2)))
            else:
                clip_count = int(rng.integers(2, 4)) if shift == "clipping" and index == 0 else int(rng.integers(3))
            for attempt in range(256):
                pair_roles = tuple(map(int, rng.permutation(10)[:5])) if varying_palette else roles
                bg, _, _, a, b = pair_roles
                if witness:
                    lengths = [5, 7]
                elif split == "test" and index == 0 and shift == "span":
                    lengths = [9, 11, 13]
                elif shift == "span":
                    lengths = [3, 5, 7]
                else:
                    lengths = [3, 5, 7, 9]
                length = int(rng.choice(lengths))
                seq = _sequence(rng, length, style, bg, (a, b))
                motif = _motif(seq, bg, (a, b))
                if motif in used_motifs:
                    continue
                candidate = _make_pair(rng, seq, pair_roles, ref, clip_count, witness,
                                       shift if split == "test" and index == 0 else None)
                if candidate is None or _key(candidate["input"]) in used_inputs:
                    continue
                try:
                    data = _decode(candidate["input"])
                    _pixel_check(data, candidate["output"])
                except AssertionError:
                    continue
                used_inputs.add(_key(candidate["input"]))
                used_motifs.add(motif)
                puzzle[split].append(candidate)
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split}/{index}/{shift}")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, motifs, training = set(), set(), []
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            data = _decode(inp)
            motif = _motif(data["sequence"], data["background"], data["signals"])
            assert motif not in motifs, "same seed motif repeated under recoloring/margins"
            motifs.add(motif)
            records = _pixel_check(data, out)
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out)
            features[split].append((len(data["sequence"]), max(r["unit"] for r in records),
                                    sum(r["lost"] > 0 for r in records)))
            if split == "train":
                training.append((data, out, records))
    # The same full mixed/gapped example must defeat all these alternatives.
    diagnostics = []
    solid_demo = False
    clipping_demo = False
    for data, out, records in training:
        seq, bg = data["sequence"], data["background"]
        solid_demo |= bg not in seq and len(set(map(int, seq))) == 1
        clipping_demo |= any(r["lost"] for r in records)
        full = all(r["lost"] == 0 for r in records)
        scaled = any(r["unit"] != data["anchors"][data["reference"]][2] for r in records)
        retained = bg in seq[1:-2] and set(data["signals"]) <= set(map(int, seq))
        alternatives = ("omit_anchor", "no_scale", "no_reflection", "seed_only",
                        "symmetric", "no_swap", "no_inset", "reflect_cascade")
        if full and scaled and retained and all(not np.array_equal(_predict(data, mode), out) for mode in alternatives):
            diagnostics.append(data)
    assert diagnostics, "missing complete scaled, asymmetric retained-gap diagnostic"
    assert solid_demo and clipping_demo, "require single-phase and boundary demonstrations"
    assert any(max(f[k] for f in features["test"]) > max(f[k] for f in features["train"])
               for k in range(3)), "test must extend span, destination unit, or simultaneous clipping"


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed semantic/diversity suite; no files or saved traces."""
    hashes, shapes, motifs, refs, units, source_units, palettes, sides = (set() for _ in range(8))
    clipped_counts, lengths = set(), set()
    single_phase_tests = mixed_tests = gapped_tests = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_hash(puzzle))
        if seed == 0:
            assert _puzzle_hash(puzzle) == _puzzle_hash(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _decode(pair["input"])
                records = _pixel_check(data, pair["output"])
                seq, bg = data["sequence"], data["background"]
                shapes.add(pair["input"].shape)
                refs.add(data["reference"])
                source_units.add(data["anchors"][data["reference"]][2])
                units.update(r["unit"] for r in records)
                palettes.add((bg, data["divider"], data["anchor"], *data["signals"]))
                lengths.add(len(seq))
                motifs.add(_motif(seq, bg, data["signals"]))
                clipped_counts.add(sum(r["lost"] > 0 for r in records))
                sides.update(side for r in records for side in r["sides"])
                # Array copying and input-only operation are checked explicitly.
                before = pair["input"].copy()
                result = solve(before)
                assert np.array_equal(before, pair["input"])
                assert not np.shares_memory(before, result)
                if split == "test":
                    present = set(map(int, seq)) - {bg}
                    single_phase_tests += len(present) == 1
                    mixed_tests += len(present) == 2
                    gapped_tests += bg in seq
    assert len(hashes) == 42 and len(shapes) >= 35 and len(motifs) >= 50
    assert refs == {0, 1, 2, 3} and {1, 2, 3} <= units
    assert {1, 2} <= source_units and len(palettes) >= 25
    assert {0, 1, 2, 3} <= clipped_counts and len(lengths) >= 5
    assert sides == {"top", "bottom", "left", "right"}
    assert single_phase_tests and mixed_tests and gapped_tests
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(700 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0, 2, 1)
    broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]]
              for s in puzzle}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "shapes": len(shapes),
            "seed_motifs": len(motifs), "reference_panels": sorted(refs),
            "source_units": sorted(source_units), "destination_units": sorted(units),
            "lengths": sorted(lengths), "palettes": len(palettes),
            "clipped_destination_counts": sorted(clipped_counts),
            "clipping_sides": sorted(sides), "single_phase_tests": single_phase_tests,
            "mixed_phase_tests": mixed_tests, "gapped_tests": gapped_tests}
