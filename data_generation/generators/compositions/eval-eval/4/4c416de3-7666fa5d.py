"""Corner ornaments restored through a virtual diagonal aperture.

All roles, the per-input stencil, and the aperture are recovered from input.
No parent generator or data file is imported. Only NumPy and stdlib are used.
"""

from itertools import combinations
import numpy as np


HOST_SEEDS = (0, 920877455, 721952729, 831723842, 508543100, 937883104,
              832539695, 923049670, 212039563, 544186187, 220981891,
              875989885, 609132949, 11584145, 869632889, 795250134,
              595607704, 933888530, 141722048, 427853482, 569687392,
              994657817, 932473434, 506692304, 88000036, 95820459,
              444485341, 925668558, 321843744, 7474011, 716164458,
              877424177, 494317676, 666886843, 991731743, 189991698,
              525593020, 948663105, 120561956, 538230177, 368504305, 91908910)
_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ALTERNATIVES = ("no_aperture", "no_stencil", "corner_gate", "consecutive",
                 "pixel_centers", "rotation", "bounding_box")


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "bad color"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _outline(rect):
    t, l, b, r = rect
    return ({(t, c) for c in range(l, r+1)} | {(b, c) for c in range(l, r+1)}
            | {(y, l) for y in range(t, b+1)} | {(y, r) for y in range(t, b+1)})


def _corners(rect):
    t, l, b, r = rect
    return [(t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3)]


def _positions(corner, pattern):
    r, c, q = corner
    dr, dc = _DIRS[q]
    return {(r+dr*u, c+dc*v) for u, v in pattern}


def _connected(pattern):
    seen = {min(pattern)}
    stack = list(seen)
    while stack:
        u, v = stack.pop()
        for x, y in pattern - seen:
            if max(abs(x-u), abs(y-v)) <= 1:
                seen.add((x, y))
                stack.append((x, y))
    return len(seen) == len(pattern)


def _strokes(points, slope):
    lines = {}
    for r, c in points:
        lines.setdefault(c-slope*r, []).append(r)
    result = []
    for q, rows in sorted(lines.items()):
        runs = []
        for r in sorted(rows):
            if not runs or r != runs[-1][-1]+1:
                runs.append([])
            runs[-1].append(r)
        for run in runs:
            assert len(run) >= 2, "a rail cannot be a singleton"
            result.append((q, 2*run[0]+slope*q, 2*run[-1]+slope*q))
    assert len(result) >= 2
    return result


def _read_input(value):
    a = _grid(value)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size/2, "background must be the strict majority"
    frame_options = []
    for color in np.flatnonzero(counts):
        color = int(color)
        if color == bg:
            continue
        rects = []
        for comp in _components(map(tuple, np.argwhere(a == color))):
            t = min(p[0] for p in comp)
            l = min(p[1] for p in comp)
            b = max(p[0] for p in comp)
            r = max(p[1] for p in comp)
            if min(b-t+1, r-l+1) < 8 or comp != _outline((t, l, b, r)):
                break
            rects.append((int(t), int(l), int(b), int(r)))
        else:
            frame_options.append((color, sorted(rects)))
    assert len(frame_options) == 1, "frames must have a unique outline color"
    frame, rects = frame_options[0]
    corners = [corner for rect in rects for corner in _corners(rect)]
    windows = {}
    for corner in corners:
        for p in _positions(corner, {(u, v) for u in range(1, 4) for v in range(1, 4)}):
            assert p not in windows, "corner neighborhoods overlap"
            windows[p] = corner
    accents = {corner: [] for corner in corners}
    outside = []
    for r, c in np.argwhere((a != bg) & (a != frame)):
        p = int(r), int(c)
        if p in windows:
            accents[windows[p]].append(p)
        else:
            outside.append(p)
    rail_colors = {int(a[p]) for p in outside}
    assert len(rail_colors) == 1, "one rail color is required"
    rail = rail_colors.pop()
    assert not any(a[p] == rail for ps in accents.values() for p in ps), "rail crosses a corner window"
    possibilities = []
    for slope in (-1, 1):
        try:
            possibilities.append((slope, _strokes(outside, slope)))
        except AssertionError:
            pass
    assert len(possibilities) == 1, "rail slope is ambiguous"
    slope, strokes = possibilities[0]
    occurrences = []
    for corner, points in accents.items():
        if not points:
            continue
        colors = {int(a[p]) for p in points}
        assert len(colors) == 1
        color = colors.pop()
        r, c, orientation = corner
        dr, dc = _DIRS[orientation]
        pattern = frozenset(((y-r)*dr, (x-c)*dc) for y, x in points)
        occurrences.append((corner, color, pattern))
    assert len(occurrences) >= 2
    largest = {p for _, _, p in occurrences if all(clue <= p for _, _, clue in occurrences)}
    assert len(largest) == 1, "there must be an intact compatible reference"
    stencil = largest.pop()
    assert 3 <= len(stencil) <= 9 and _connected(stencil)
    assert all(1 <= u <= 3 and 1 <= v <= 3 for u, v in stencil)
    return dict(grid=a, background=bg, frame=frame, rail=rail, rectangles=rects,
                corners=corners, occurrences=occurrences, stencil=stencil,
                slope=slope, strokes=strokes)


def _aperture(info, alternative=""):
    """A sweep on inclusive continuous longitudinal intervals."""
    a, slope, strokes = info["grid"], info["slope"], info["strokes"]
    rr, cc = np.indices(a.shape)
    q, t = cc-slope*rr, rr+slope*cc
    result = np.zeros(a.shape, dtype=bool)
    if alternative == "consecutive":
        # Deliberately weaker: only consecutive transverse rails may bracket.
        remaining = set(range(len(strokes)))
        pairs = []
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            indices, stack = [start], [start]
            while stack:
                i = stack.pop()
                neighbors = [j for j in sorted(remaining)
                             if max(strokes[i][1], strokes[j][1]) <= min(strokes[i][2], strokes[j][2])]
                for j in neighbors:
                    remaining.remove(j)
                    indices.append(j)
                    stack.append(j)
            ordered = sorted(strokes[i] for i in indices)
            pairs.extend(zip(ordered, ordered[1:]))
        for x, y in pairs:
            result |= ((t >= max(x[1], y[1])) & (t <= min(x[2], y[2]))
                       & (q > min(x[0], y[0])) & (q < max(x[0], y[0])))
        return result
    for level in range(int(t.min()), int(t.max())+1):
        active = [x[0] for x in strokes if x[1] <= level <= x[2]
                  and (alternative != "pixel_centers" or (level-x[0]) % 2 == 0)]
        if len(active) >= 2:
            result |= (t == level) & (q > min(active)) & (q < max(active))
    return result


def _prediction(info, alternative=""):
    a = info["grid"]
    out = a.copy()
    aperture = _aperture(info, alternative)
    pattern = info["stencil"]
    if alternative == "no_stencil":
        # This ablation uses an accent already visible in input; any such
        # choice still incorrectly changes cells outside the corner stencils.
        out[aperture & (a == info["background"])] = info["occurrences"][0][1]
        return out
    if alternative == "bounding_box":
        pattern = frozenset((u, v) for u in range(min(u for u, _ in pattern), max(u for u, _ in pattern)+1)
                            for v in range(min(v for _, v in pattern), max(v for _, v in pattern)+1))
    if alternative == "rotation":
        normalized = [(corner, color, frozenset((v, u) for u, v in clue)
                       if corner[2] in (1, 2) else clue)
                      for corner, color, clue in info["occurrences"]]
        maximal = {p for _, _, p in normalized if all(clue <= p for _, _, clue in normalized)}
        if len(maximal) != 1:
            return None  # The alternative already contradicts visible clues.
        pattern = maximal.pop()
    for corner, color, _ in info["occurrences"]:
        local = frozenset((v, u) for u, v in pattern) if alternative == "rotation" and corner[2] in (1, 2) else pattern
        for p in _positions(corner, local):
            allowed = (alternative == "no_aperture" or
                       (bool(aperture[corner[:2]]) if alternative == "corner_gate" else bool(aperture[p])))
            if allowed:
                assert int(a[p]) in (info["background"], color)
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore only missing motif cells admitted by the virtual aperture."""
    return _prediction(_read_input(input_grid))


def _differences(pair, info=None):
    info = _read_input(pair["input"]) if info is None else info
    result = set()
    for alt in _ALTERNATIVES:
        prediction = _prediction(info, alt)
        if prediction is None or not np.array_equal(prediction, pair["output"]):
            result.add(alt)
    return result


def _mixed_corner(pair, info):
    """At one partial corner, some holes appear and others remain holes."""
    a, out = info["grid"], np.asarray(pair["output"])
    for corner, _, clue in info["occurrences"]:
        missing = _positions(corner, info["stencil"]-clue)
        if missing and any(a[p] != out[p] for p in missing) and any(a[p] == out[p] for p in missing):
            return True
    return False


def _sample_stencil(rng, extent):
    domain = {(u, v) for u in range(1, extent+1) for v in range(1, extent+1)}
    pattern = {sorted(domain)[int(rng.integers(len(domain)))]}
    size = int(rng.integers(3, len(domain)+1))
    while len(pattern) < size:
        choices = sorted(p for p in domain-pattern if any(max(abs(p[0]-u), abs(p[1]-v)) <= 1 for u, v in pattern))
        pattern.add(choices[int(rng.integers(len(choices)))])
    return frozenset(pattern)


def _stroke_cells(stroke, slope):
    q, lo, hi = stroke
    return {((t-slope*q)//2, (q+slope*t)//2) for t in range(lo, hi+1, 2)}


def _point_bracket(p, strokes, slope):
    """Construction uses pairwise geometric overlap, rather than a sweep."""
    r, c = p
    q, t = c-slope*r, r+slope*c
    return any(max(x[1], y[1]) <= t <= min(x[2], y[2])
               and min(x[0], y[0]) < q < max(x[0], y[0])
               for x, y in combinations(strokes, 2))


def _draw_pair(rng, frame_count, rail_count, extent, severe=False):
    """Free corner marks, a random motif, and geometry-aware rail placement."""
    pattern = _sample_stencil(rng, extent)
    sh = int(rng.integers(8, 14 if frame_count == 1 else 11))
    sw = int(rng.integers(8, 14 if frame_count == 1 else 11))
    rects = [(0, 0, sh-1, sw-1)]
    for _ in range(frame_count-1):
        for _attempt in range(100):
            h2, w2 = int(rng.integers(8, 11)), int(rng.integers(8, 11))
            t, l = int(rng.integers(-12, 20)), int(rng.integers(-12, 20))
            b, r = t+h2-1, l+w2-1
            if not all(b+2 < x[0] or x[2]+2 < t or r+2 < x[1] or x[3]+2 < l for x in rects):
                continue
            candidate = rects+[(t, l, b, r)]
            if max(x[2] for x in candidate)-min(x[0] for x in candidate) >= 26:
                continue
            if max(x[3] for x in candidate)-min(x[1] for x in candidate) >= 26:
                continue
            rects.append((t, l, b, r))
            break
        else:
            raise AssertionError("could not pack separated frames")
    all_corners = [c for rect in rects for c in _corners(rect)]
    windows = set().union(*(_positions(c, {(u, v) for u in range(1, 4) for v in range(1, 4)}) for c in all_corners))
    outlines = set().union(*(_outline(rect) for rect in rects))
    slope = int(rng.choice((-1, 1)))
    primary = _corners(rects[0])
    target = primary[int(rng.integers(4))]
    pts = sorted(_positions(target, pattern))
    ts = [r+slope*c for r, c in pts]
    cutoff = int(rng.integers(min(ts), max(ts)+1))
    low_side = bool(rng.integers(2))
    qvalues = [c-slope*r for r, c in _outline(rects[0])]
    qlow = min(qvalues)-int(rng.integers(2, 6))
    qhigh = max(qvalues)+int(rng.integers(2, 6))
    span = int(rng.integers(3, 9))
    strokes = []
    for q in (qlow, qhigh):
        if low_side:
            lo = cutoff+((q-cutoff) % 2)
            hi = lo+2*span
        else:
            hi = cutoff-((cutoff-q) % 2)
            lo = hi-2*span
        strokes.append((q, lo, hi))
    # Additional short rails can sit inside the frame or outside it. They
    # need not share a common overlap, nor have different transverse lines.
    used = set().union(*(_stroke_cells(s, slope) for s in strokes))
    assert not used & (outlines | windows)
    for _ in range(rail_count-2):
        for _attempt in range(100):
            q = int(rng.integers(qlow-5, qhigh+6))
            anchor = cutoff+int(rng.integers(-5, 8))
            lo = anchor-((anchor-q) % 2)
            hi = lo+2*int(rng.integers(1, 5))
            s = (q, lo, hi)
            pixels = _stroke_cells(s, slope)
            # Separate collinear runs so visible endpoints remain exact.
            if any(q == x[0] and not (hi+2 < x[1] or x[2]+2 < lo) for x in strokes):
                continue
            if pixels & (outlines | windows | used):
                continue
            strokes.append(s)
            used |= pixels
            break
        else:
            raise AssertionError("could not place a rail")
    marked = [target]
    # The intact reference has an adjacent inward orientation to target.
    adjacent = [c for c in primary if c[2] ^ target[2] in (1, 2)]
    reference = adjacent[int(rng.integers(2))]
    marked.append(reference)
    for corner in all_corners:
        if corner not in marked and rng.random() < .55:
            marked.append(corner)
    palette = [int(c) for c in rng.permutation(10)]
    bg, frame, rail = palette[:3]
    accents = palette[3:3+int(rng.integers(1, min(5, len(marked))+1))]
    bounds = outlines | used
    rmin, cmin = min(r for r, _ in bounds), min(c for _, c in bounds)
    rmax, cmax = max(r for r, _ in bounds), max(c for _, c in bounds)
    h, w = rmax-rmin+1, cmax-cmin+1
    assert h <= 30 and w <= 30
    height = h+int(rng.integers(0, min(4, 30-h)+1))
    width = w+int(rng.integers(0, min(4, 30-w)+1))
    roff = -rmin+int(rng.integers(0, height-h+1))
    coff = -cmin+int(rng.integers(0, width-w+1))
    def shift(p):
        return p[0]+roff, p[1]+coff
    a = np.full((height, width), bg, dtype=np.int8)
    for p in outlines:
        a[shift(p)] = frame
    for p in used:
        a[shift(p)] = rail
    records = []
    for corner in marked:
        color = int(rng.choice(accents))
        entire = _positions(corner, pattern)
        if corner == reference or (corner != target and rng.random() < .2):
            retained = entire
        else:
            points = sorted(entire)
            rng.shuffle(points)
            count = 1 if severe and corner == target else int(rng.integers(1, len(pattern)))
            retained = set(points[:count])
        for p in retained:
            a[shift(p)] = color
        records.append((entire, retained, color))
    out = a.copy()
    for entire, retained, color in records:
        for p in entire-retained:
            if _point_bracket(p, strokes, slope):
                out[shift(p)] = color
    assert np.any(a != out)
    pair = {"input": a, "output": out}
    info = _read_input(a)
    assert np.array_equal(_prediction(info), out)
    assert len(info["strokes"]) == rail_count
    return pair, info


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    # Shift in boundary count or object count, with identical transformation.
    shift = int(rng.integers(2))
    required = set()
    train_stencils, train_slopes = set(), set()
    for split, count in (("test", nq), ("train", nt)):
        for index in range(count):
            if shift == 0:
                frames = int(rng.integers(1, 4))
                rails = int(rng.integers(4, 6)) if split == "test" else int(rng.integers(2, 4))
            else:
                frames = 3 if split == "test" else int(rng.integers(1, 3))
                rails = int(rng.integers(2, 5))
            # Use three or more rails for the teaching witness regardless of
            # whether the shift is in frames or in the number of boundaries.
            if split == "train" and index == 0:
                rails = 3
            extent = int(rng.choice((2, 3, 3)))
            wanted = required.copy() if split == "train" and index == 0 else set()
            if split == "train" and index == 0:
                wanted |= {"no_aperture", "no_stencil", "consecutive", "pixel_centers"}
            for _ in range(1600):
                try:
                    pair, info = _draw_pair(rng, frames, rails, extent, split == "test")
                    assert _key(pair["input"]) not in seen
                    differences = _differences(pair, info)
                    assert wanted <= differences
                    if split == "train" and index == 0:
                        assert _mixed_corner(pair, info)
                    if split == "train" and index == 1:
                        assert info["stencil"] not in train_stencils
                        assert info["slope"] not in train_slopes
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded sampler could not construct an informative pair")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "test":
                required |= differences
            else:
                train_stencils.add(info["stencil"])
                train_slopes.add(info["slope"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, demonstrated, needed = set(), set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and np.any(a != b)
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _read_input(a)
            assert np.array_equal(_prediction(info), b), "input-only prediction disagrees"
            assert np.all(a[a != b] == info["background"])
            assert np.array_equal(a[a != info["background"]], b[a != info["background"]]), "original content changed"
            assert set(np.unique(a)) == set(np.unique(b)), "invented output color"
            # Check each cell against pairwise interval geometry, a separate
            # calculation from the vectorized solver sweep.
            expected = a.copy()
            for corner, color, clue in info["occurrences"]:
                for p in _positions(corner, info["stencil"]):
                    if _point_bracket(p, info["strokes"], info["slope"]):
                        expected[p] = color
            assert np.array_equal(expected, b), "pairwise geometric check disagrees"
            diffs = _differences(pair, info)
            (demonstrated if split == "train" else needed).update(diffs)
            features[split].append(info)
    assert needed <= demonstrated, "test depends on an undemonstrated distinction"
    assert {"no_aperture", "no_stencil", "consecutive", "pixel_centers"} <= demonstrated
    assert any(_mixed_corner(p, info) for p, info in zip(puzzle["train"], features["train"])), "no cellwise aperture witness"
    assert len({x["stencil"] for x in features["train"]}) >= 2, "stencil variability is untaught"
    assert {x["slope"] for x in features["train"]} == {-1, 1}, "both rail slopes must be taught"
    train_frames = max(len(x["rectangles"]) for x in features["train"])
    train_rails = max(len(x["strokes"]) for x in features["train"])
    assert all(len(x["rectangles"]) > train_frames or len(x["strokes"]) > train_rails for x in features["test"]), "missing structural test shift"


def self_test():
    """Bounded semantic, reproducibility, count, and diversity regressions."""
    # Hand-specified case: the missing corner cell at (9,9) is outside the
    # aperture; precisely (9,10) and (11,10) are revealed. The short central
    # rail is inactive there. The former lies on a boundary-pixel scanline,
    # while the latter does too; an additional fixture below isolates t=20.
    stencil = frozenset({(1, 1), (1, 2), (2, 2), (3, 2)})
    fixture = np.zeros((23, 23), dtype=np.int8)
    rect = (8, 8, 17, 17)
    for p in _outline(rect):
        fixture[p] = 5
    rails = [(-11, 19, 27), (0, 24, 26), (11, 19, 27)]
    for rail in rails:
        for p in _stroke_cells(rail, 1):
            fixture[p] = 6
    for p in _positions((8, 17, 1), stencil):
        fixture[p] = 3
    fixture[10, 10] = 2
    expected = fixture.copy()
    expected[9, 10] = expected[11, 10] = 2
    assert np.array_equal(solve(fixture), expected)
    assert expected[9, 9] == 0
    assert {"no_aperture", "no_stencil", "corner_gate", "consecutive", "rotation", "bounding_box"} <= _differences({"input": fixture, "output": expected})
    continuous = fixture.copy()
    continuous[10, 10] = 0
    continuous[9, 10] = 2
    continuous_expected = continuous.copy()
    continuous_expected[10, 10] = continuous_expected[11, 10] = 2
    assert np.array_equal(solve(continuous), continuous_expected)
    assert "pixel_centers" in _differences({"input": continuous, "output": continuous_expected})
    simple = fixture.copy()
    for p in _stroke_cells(rails[1], 1):
        simple[p] = 0
    # Same output without the short middle rail, and both diagonal slopes in
    # training, still fail to teach skipping an inactive intermediate rail.
    insufficient = {"train": [{"input": simple, "output": expected.copy()},
                              {"input": np.fliplr(simple).copy(), "output": np.fliplr(expected).copy()}],
                    "test": [{"input": fixture.copy(), "output": expected.copy()}]}
    for p in _stroke_cells(rails[1], 1):
        insufficient["train"][0]["output"][p] = 0
    insufficient["train"][1]["output"] = np.fliplr(insufficient["train"][0]["output"]).copy()
    try:
        validate_generated_puzzle(insufficient)
    except AssertionError as error:
        assert "undemonstrated" in str(error)
    else:
        raise AssertionError("inactive-middle teaching gap accepted")
    shapes, patterns, geometries, palettes, slopes, fingerprints = set(), set(), set(), set(), set(), set()
    frame_counts, rail_counts, reference_counts, clue_sizes = set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in puzzle:
            for pair in puzzle[split]:
                a = pair["input"]
                info = _read_input(a)
                assert np.array_equal(solve(a.copy()), pair["output"])
                shapes.add(a.shape)
                patterns.add(info["stencil"])
                palettes.add((info["background"], info["frame"], info["rail"]))
                slopes.add(info["slope"])
                frame_counts.add(len(info["rectangles"]))
                rail_counts.add(len(info["strokes"]))
                reference_counts.add(sum(p == info["stencil"] for _, _, p in info["occurrences"]))
                clue_sizes.update(len(p) for _, _, p in info["occurrences"] if p != info["stencil"])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.ravel())
                geometries.add((a.shape, normalized))
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(patterns) >= 20 and len(geometries) >= 180
    assert len(palettes) >= 60 and slopes == {-1, 1}
    assert frame_counts == {1, 2, 3} and {2, 3, 4, 5} <= rail_counts
    assert max(reference_counts) >= 2 and {1, 2, 3} <= clue_sizes
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123+nt*5+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    one, two = generate_puzzle(99, 2, 1), generate_puzzle(99, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in one for x, y in zip(one[s], two[s]) for k in x)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in one[s]] for s in one}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
