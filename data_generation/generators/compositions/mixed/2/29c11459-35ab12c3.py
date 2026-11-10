"""Partitioned horizontal links and marker-directed, separator-preserving copies.

All semantics are input-visible or fixed here: separator 5, odd horizontal
spans, unique adjacent companions, and original/source/copy precedence.
No file access, saved examples, global random state, or seed-aware solving.
"""

from itertools import combinations
import numpy as np


SEPARATOR = 5
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid of size 1..30")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _source_pixels(lines):
    return {(r, c): SEPARATOR if 2*c == a+b else left if 2*c < a+b else right
            for r, a, b, left, right in lines for c in range(a, b+1)}


def _copy_pixels(line, marker, preserve_separator=True):
    r, a, b, _, _ = line
    mr, mc, color, dr, dc = marker
    return {(r+dr, c+dc): SEPARATOR if preserve_separator and 2*c == a+b else color
            for c in range(a, b+1)}


def _parse(inp):
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(max(counts))*2 <= inp.size or bg == SEPARATOR or np.any(inp == SEPARATOR):
        raise ValueError("Input needs a strict-majority background and no color 5")
    lines, endpoints = [], set()
    for r, row in enumerate(inp):
        cells = [int(c) for c in np.flatnonzero(row != bg)]
        candidates = [(a, b) for a, b in combinations(cells, 2)
                      if b-a >= 2 and (b-a) % 2 == 0 and row[a] != row[b]]
        if len(candidates) > 1:
            raise ValueError("A row has competing endpoint pairs")
        if candidates:
            a, b = candidates[0]
            lines.append((r, a, b, int(row[a]), int(row[b])))
            endpoints.update(((r, a), (r, b)))
    if not lines:
        raise ValueError("No odd horizontal link")
    source = _source_pixels(lines)
    markers, inert = {}, []
    for r, c in np.argwhere(inp != bg):
        p = int(r), int(c)
        if p in endpoints:
            continue
        if p in source:
            raise ValueError("An extra landmark lies inside a source band")
        adjacent = [(i, (rr, cc)) for i, (rr, a, b, _, _) in enumerate(lines)
                    for cc in (a, b) if max(abs(p[0]-rr), abs(p[1]-cc)) == 1]
        if not adjacent:
            inert.append(p)
            continue
        if len(adjacent) != 1:
            raise ValueError("Companion touches multiple endpoints")
        i, anchor = adjacent[0]
        color = int(inp[p])
        if i in markers or color in lines[i][3:5]:
            raise ValueError("Companion is repeated or shares an endpoint color")
        marker = (p[0], p[1], color, p[0]-anchor[0], p[1]-anchor[1])
        copy = _copy_pixels(lines[i], marker)
        if not all(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1] for rr, cc in copy):
            raise ValueError("A translated band would be clipped")
        markers[i] = marker
    # Copy/copy disagreements on otherwise writable cells are outside this family.
    claims = {}
    for i, marker in markers.items():
        for p, color in _copy_pixels(lines[i], marker).items():
            if inp[p] == bg and p not in source:
                if p in claims and claims[p] != color:
                    raise ValueError("Unresolved companion overlap")
                claims[p] = color
    return {"background": bg, "lines": lines, "markers": markers, "inert": inert}


def _render(inp, info, *, copies=True, preserve_separator=True, copy_first=True):
    """Paint from geometric descriptions; generation supplies its own descriptions."""
    bg = info["background"]
    source = _source_pixels(info["lines"])
    out = inp.copy()
    def paint_source():
        for p, color in source.items():
            if inp[p] == bg:
                out[p] = color
    def paint_copies():
        if copies:
            for i, marker in info["markers"].items():
                for p, color in _copy_pixels(info["lines"][i], marker, preserve_separator).items():
                    if inp[p] == bg:
                        out[p] = color
    if copy_first:
        paint_copies()
        paint_source()
    else:
        paint_source()
        paint_copies()
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover row links and local offsets entirely from the given input."""
    inp = _grid(input_grid)
    return _render(inp, _parse(inp))


def _measure(inp, out, info):
    bg, lines, markers = info["background"], info["lines"], info["markers"]
    source = _source_pixels(lines)
    visible, copied_centers, overlap, hidden = 0, 0, 0, 0
    for i, marker in markers.items():
        copy = _copy_pixels(lines[i], marker)
        new = [p for p in copy if inp[p] == bg and p not in source]
        visible += bool(new)
        hidden += not bool(new)
        copied_centers += sum(color == SEPARATOR and p in new for p, color in copy.items())
        overlap += sum(p in source and inp[p] == bg and source[p] != color
                       for p, color in copy.items())
    # Same-colored aligned landmarks in different rows tempt an incorrect link.
    # Count only omitted cells whose output actually stays background.
    chord_cells = set()
    vertices = [(r, c, color) for r, a, b, left, right in lines
                for c, color in ((a, left), (b, right))]
    for (r, c, color), (rr, cc, other) in combinations(vertices, 2):
        dr, dc = rr-r, cc-c
        if dr == 0 or color != other or not (dc == 0 or abs(dr) == abs(dc)):
            continue
        length = max(abs(dr), abs(dc))
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        chord_cells.update((r+k*sr, c+k*sc) for k in range(1, length)
                           if out[r+k*sr, c+k*sc] == bg)
    return {"bands": len(lines), "marked": len(markers), "visible": visible,
            "hidden": hidden, "copied_centers": int(copied_centers), "overlap": int(overlap),
            "rejected_chord_cells": len(chord_cells), "inert": len(info["inert"]),
            "offsets": tuple(sorted({m[3:5] for m in markers.values()})),
            "lengths": tuple(b-a+1 for _, a, b, _, _ in lines)}


def _finish(inp, bg, lines, markers):
    design = {"background": bg, "lines": sorted(lines), "markers": {}, "inert": []}
    # Markers are stored beside the row they belong to; sorting changes indices.
    for row, marker in markers.items():
        i = next(i for i, line in enumerate(design["lines"]) if line[0] == row)
        design["markers"][i] = marker
    out = _render(inp, design)
    parsed = _parse(inp)
    if parsed["lines"] != design["lines"] or parsed["markers"] != design["markers"]:
        raise ValueError("Construction does not recover its intended geometry")
    if not np.array_equal(solve(inp), out):
        raise ValueError("Visible-input parsing differs from construction")
    return {"input": inp, "output": out}


def _palette(rng, bg, count=3):
    return [int(c) for c in rng.choice([c for c in range(10) if c not in (bg, SEPARATOR)],
                                      count, replace=False)]


def _demo(rng, bg, interaction=False):
    """Two informative, geometrically randomized witnesses, even for 2/1."""
    half = int(rng.integers(4, 10)) if interaction else int(rng.integers(2, 11))
    length = 2*half+1
    width = int(rng.integers(length+2, 31))
    a = int(rng.integers(1, width-length))
    b = a+length-1
    height = int(rng.integers(8, 24)) if interaction else int(rng.integers(4, 17))
    r = int(rng.integers(1, height-5)) if interaction else int(rng.integers(1, height-1))
    left, right, companion = _palette(rng, bg)
    lines = [(r, a, b, left, right)]
    side = int(rng.integers(2))
    if interaction:
        # Marker parity differs from the second row's true endpoints, so its
        # presence cannot introduce another odd-span pair in that row.
        aa, bb = (a+3, b-1) if side == 0 else (a+1, b-3)
        c, d = _palette(rng, bg, 2)
        lines += [(r+1, aa, bb, c, d), (r+4, a, b, left, right)]
        dr, dc = 1, 0
    else:
        dr, dc = OFFSETS[int(rng.integers(8))]
        # The first witness always has a visible translated midpoint.
        if dr == 0:
            dr = int(rng.choice([-1, 1]))
    anchor = a if side == 0 else b
    marker = (r+dr, anchor+dc, companion, dr, dc)
    inp = np.full((height, width), bg, dtype=np.int8)
    for rr, aa, bb, c, d in lines:
        inp[rr, aa], inp[rr, bb] = c, d
    inp[marker[0], marker[1]] = companion
    # Reflect vertically sometimes; this also changes the marker's vector.
    if rng.random() < .5:
        inp = np.flipud(inp).copy()
        lines = [(height-1-rr, aa, bb, c, d) for rr, aa, bb, c, d in lines]
        marker = (height-1-marker[0], marker[1], companion, -dr, dc)
        r = height-1-r
    return _finish(inp, bg, lines, {r: marker})


def _random_pair(rng, bg, count, require_visible=False):
    # Finite rejection; each attempt retains only two <=30x30 arrays.
    for _ in range(120):
        h = int(rng.integers(max(5, 3*count+1), 31))
        w = int(rng.integers(7, 31))
        inp = np.full((h, w), bg, dtype=np.int8)
        rows = sorted(int(x) for x in rng.choice(h, count, replace=False))
        lines = []
        for r in rows:
            half = int(rng.integers(1, (w-1)//2+1))
            a = int(rng.integers(w-2*half))
            left, right = _palette(rng, bg, 2)
            if lines and rng.random() < .30:
                left, right = lines[int(rng.integers(len(lines)))][3:5]
            b = a+2*half
            lines.append((r, a, b, left, right))
            inp[r, a], inp[r, b] = left, right
        source = _source_pixels(lines)
        vertices = [(r, c) for r, a, b, _, _ in lines for c in (a, b)]
        markers = {}
        for line in lines:
            if rng.random() < .40 and not (require_visible and not markers):
                continue
            r, a, b, left, right = line
            candidates = [(c, dr, dc) for c in (a, b) for dr, dc in OFFSETS]
            rng.shuffle(candidates)
            for anchor, dr, dc in candidates:
                p = r+dr, anchor+dc
                if not (0 <= p[0] < h and 0 <= p[1] < w and
                        0 <= r+dr < h and 0 <= a+dc <= b+dc < w):
                    continue
                if p in source or inp[p] != bg:
                    continue
                if sum(max(abs(p[0]-rr), abs(p[1]-cc)) == 1 for rr, cc in vertices) != 1:
                    continue
                if require_visible and not markers and dr == 0:
                    continue
                color = int(rng.choice([c for c in range(10)
                                        if c not in (bg, SEPARATOR, left, right)]))
                inp[p] = color
                marker = (p[0], p[1], color, dr, dc)
                try:
                    _parse(inp)
                except ValueError:
                    inp[p] = bg
                    continue
                markers[r] = marker
                break
        # Up to three inert points. They may occlude a copy but never a source.
        for _ in range(int(rng.integers(4))):
            for _ in range(20):
                p = int(rng.integers(h)), int(rng.integers(w))
                if inp[p] != bg or p in source or any(
                        max(abs(p[0]-rr), abs(p[1]-cc)) <= 1 for rr, cc in vertices):
                    continue
                inp[p] = _palette(rng, bg, 1)[0]
                try:
                    if _parse(inp)["lines"] != lines:
                        raise ValueError("Inert landmark introduced a link")
                except ValueError:
                    inp[p] = bg
                    continue
                break
        try:
            pair = _finish(inp, bg, lines, markers)
            metrics = _measure(inp, pair["output"], _parse(inp))
            if require_visible and not metrics["copied_centers"]:
                continue
            return pair
        except ValueError:
            continue
    # Spacious finite fallback preserves count, a visible copy, and an inert
    # free row between groups. Its colors, lengths and margins still vary.
    h, w = 4*count+1, int(rng.integers(9, 31))
    inp = np.full((h, w), bg, dtype=np.int8)
    lines, markers = [], {}
    for i in range(count):
        r = 4*i+1
        half = int(rng.integers(2, (w-2)//2+1))
        a = int(rng.integers(1, w-2*half))
        b = a+2*half
        left, right, color = _palette(rng, bg)
        lines.append((r, a, b, left, right))
        inp[r, a], inp[r, b] = left, right
        marker = (r+1, a, color, 1, 0)
        markers[r] = marker
        inp[r+1, a] = color
    return _finish(inp, bg, lines, markers)


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError("Illegal example count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    backgrounds = [int(c) for c in range(10) if c != SEPARATOR]
    puzzle, seen = {"train": [], "test": []}, set()
    # Randomize demo positions, keeping the evidence even at the minimum count.
    demo_positions = [int(x) for x in rng.choice(nt, 2, replace=False)]
    for split, total in (("train", nt), ("test", nq)):
        for i in range(total):
            bg = int(rng.choice(backgrounds))
            for _ in range(24):
                if split == "train" and i in demo_positions:
                    pair = _demo(rng, bg, interaction=(i == demo_positions[1]))
                else:
                    count = int(rng.integers(4, 7)) if split == "test" else int(rng.integers(1, 4))
                    pair = _random_pair(rng, bg, count, require_visible=(split == "test"))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct example budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Illegal counts")
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Shape changed")
            key = _key(inp)
            _require(key not in seen, "Repeated input")
            seen.add(key)
            info = _parse(inp)
            bg = info["background"]
            _require(np.array_equal(inp[inp != bg], out[inp != bg]), "Original landmark changed")
            _require(np.array_equal(solve(inp), out), "Wrong completion")
            source = _source_pixels(info["lines"])
            copies = [_copy_pixels(info["lines"][i], marker)
                      for i, marker in info["markers"].items()]
            # Explicit per-cell ownership check also verifies outer preservation.
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    p = r, c
                    color = int(inp[p])
                    if color == bg:
                        if p in source:
                            color = source[p]
                        else:
                            claims = {copy[p] for copy in copies if p in copy}
                            _require(len(claims) <= 1, "Conflicting copies")
                            color = next(iter(claims), bg)
                    _require(out[p] == color, "Incorrect cell ownership")
            for r, a, b, left, right in info["lines"]:
                m = (a+b)//2
                _require(out[r, m] == SEPARATOR, "Missing source midpoint")
                _require(np.all(out[r, a:m] == left) and np.all(out[r, m+1:b+1] == right),
                         "Unequal or incorrect source territories")
            _require(np.any(inp != out), "Identity example")
            metrics[split].append(_measure(inp, out, info))
    train = metrics["train"]
    _require(any(m["copied_centers"] for m in train), "No visible copied separator demonstration")
    _require(any(m["overlap"] for m in train), "No source priority demonstration")
    _require(any(m["bands"] > m["marked"] for m in train), "No unmarked demonstration")
    _require(any(m["rejected_chord_cells"] for m in train), "No rejected alignment demonstration")
    _require(len({length for m in train for length in m["lengths"]}) >= 2, "Only one span scale")
    _require(max(m["bands"] for m in train) < min(m["bands"] for m in metrics["test"]),
             "Missing count-composition shift")
    _require(all(m["copied_centers"] for m in metrics["test"]), "Test lacks visible partition/copy dependency")


def _regressions():
    def rows(*values):
        return np.array([[int(c) for c in row] for row in values], dtype=np.int8)
    inp = rows("0000000", "0100020", "0300000", "0000000")
    expected = rows("0000000", "0115220", "0335330", "0000000")
    _require(np.array_equal(solve(inp), expected), "Full band / copied separator regression")
    info = _parse(inp)
    _require(not np.array_equal(_render(inp, info, copies=False), expected), "B ablation ineffective")
    _require(not np.array_equal(_render(inp, info, preserve_separator=False), expected),
             "A does not affect the copied pattern")
    # Inline outward marker: copy is hidden except at the existing marker.
    inp = rows("00000000", "03100020", "00000000")
    _require(np.array_equal(solve(inp), rows("00000000", "03115220", "00000000")),
             "Inline companion/source priority regression")
    # The input-only solver has no hidden three/six-row generation limit.
    inp = np.zeros((30, 15), dtype=np.int8)
    for r in range(12):
        inp[r, 2], inp[r, 10] = 1, 2
    out = solve(inp)
    _require(np.all(out[:12, 6] == SEPARATOR), "Extended family member unsupported")


def self_test():
    """42 streamed seeds, all legal counts, ablations and bounded regressions."""
    _regressions()
    shapes, geometry, backgrounds, offsets, band_counts, lengths = set(), set(), set(), set(), set(), set()
    features = {k: 0 for k in ("visible", "hidden", "copied_centers", "overlap", "rejected_chord_cells", "inert")}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _parse(inp)
                m = _measure(inp, out, info)
                shapes.add(inp.shape)
                geometry.add((inp.shape, tuple(line[:3] for line in info["lines"]), m["offsets"]))
                backgrounds.add(info["background"])
                offsets.update(m["offsets"])
                band_counts.add(m["bands"])
                lengths.update(m["lengths"])
                for k in features:
                    features[k] += m[k]
        # Ablations are checked on every puzzle's returned training grids.
        evidence = []
        for pair in puzzle["train"]:
            inp, out = pair["input"], pair["output"]
            info = _parse(inp)
            evidence.append((not np.array_equal(_render(inp, info, copies=False), out),
                             not np.array_equal(_render(inp, info, preserve_separator=False), out),
                             not np.array_equal(_render(inp, info, copy_first=False), out)))
        _require(all(any(e[i] for e in evidence) for i in range(3)), "Training shortcut survived")
    _require(len(signatures) == 42 and len(shapes) >= 60 and len(geometry) >= 150,
             "Insufficient structural variation")
    _require(backgrounds == set(range(10)) - {SEPARATOR}, "Missing background colors")
    _require(offsets == set(OFFSETS) and band_counts == set(range(1, 7)), "Missing vectors/counts")
    _require(len(lengths) >= 10 and all(v > 0 for v in features.values()), "Missing family features")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(531+10*nt+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")
    a, b = generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "Not deterministic")
    # Returned nested lists must also validate; output corruption must not.
    nested = {s: [{k: v.tolist() for k, v in p.items()} for p in a[s]] for s in a}
    validate_generated_puzzle(nested)
    pair = nested["test"][0]
    loc = tuple(np.argwhere(np.array(pair["input"]) != np.array(pair["output"]))[0])
    pair["output"][loc[0]][loc[1]] = pair["input"][loc[0]][loc[1]]
    try:
        validate_generated_puzzle(nested)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing output cell accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "geometry": len(geometry),
            "backgrounds": sorted(backgrounds), "offsets": sorted(offsets),
            "band_counts": sorted(band_counts), "lengths": sorted(lengths), "features": features}
