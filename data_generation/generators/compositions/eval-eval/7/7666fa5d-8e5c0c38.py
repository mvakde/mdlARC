"""A diagonal loom: interval-defined reflection frames weave supported marks.

Only NumPy and the standard library are used. The public solver receives one
input, and recovers all color roles and geometry from that input.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _runs(grid, color, slope):
    lines = defaultdict(list)
    for r, c in np.argwhere(grid == color):
        lines[int(c - slope*r)].append(int(r))
    strokes = []
    for q, rows in sorted(lines.items()):
        runs = []
        for r in sorted(rows):
            if not runs or r != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(r)
        for run in runs:
            strokes.append((q, 2*run[0] + slope*q,
                            2*run[-1] + slope*q, len(run)))
    return strokes


def _inspect(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size/2:
        raise ValueError("the field must occupy a strict majority")
    colors = sorted(set(map(int, np.unique(grid))) - {bg})
    if len(colors) < 2:
        raise ValueError("rails and at least one mark color are required")
    candidates = []
    for color in colors:
        for slope in (-1, 1):
            strokes = _runs(grid, color, slope)
            qs = sorted({s[0] for s in strokes})
            if (len(strokes) < 2 or len(qs) < 2
                    or any(s[3] < 3 for s in strokes)
                    or len({q % 2 for q in qs}) != 1
                    or any(b-a < 4 for a, b in zip(qs, qs[1:]))):
                continue
            if not any(a[0] != b[0] and max(a[1], b[1]) <= min(a[2], b[2])
                       for i, a in enumerate(strokes) for b in strokes[i+1:]):
                continue
            candidates.append((color, slope, strokes))
    if len(candidates) != 1:
        raise ValueError("input must identify exactly one rail color and slope")
    rail, slope, strokes = candidates[0]
    marks = defaultdict(dict)
    for r, c in np.argwhere((grid != bg) & (grid != rail)):
        q, t = int(c-slope*r), int(r+slope*c)
        marks[t][q] = int(grid[r, c])
    return dict(bg=bg, rail=rail, slope=slope, strokes=strokes, marks=dict(marks))


def _frames(strokes, t, alternative=""):
    if alternative == "static_frame":
        active = {s[0] for s in strokes}
    elif alternative == "pixel_centers":
        active = {s[0] for s in strokes if s[1] <= t <= s[2] and (t-s[0]) % 2 == 0}
    else:
        active = {s[0] for s in strokes if s[1] <= t <= s[2]}
    if len(active) < 2:
        return []
    if alternative == "adjacent_frame":
        order = sorted({s[0] for s in strokes})
        return [(a, b) for a, b in zip(order, order[1:]) if a in active and b in active]
    return [(min(active), max(active))]


def _paint(grid, info, alternative=""):
    bg, rail, slope = info["bg"], info["rail"], info["slope"]
    out = np.full_like(grid, bg)
    out[grid == rail] = rail
    if alternative == "preserve_noise":
        out[:] = grid
    for t, line in sorted(info["marks"].items()):
        for low, high in _frames(info["strokes"], t, alternative):
            inside = dict(line) if alternative == "unbounded_reflection" else {
                q: color for q, color in line.items() if low < q < high}
            supported = defaultdict(list)
            for q, color in inside.items():
                reflected = low + high - q
                if alternative == "no_symmetry":
                    supported[color].append(q)
                elif alternative == "ignore_color":
                    if reflected in inside:
                        supported[min(color, inside[reflected])].append(q)
                elif inside.get(reflected) == color:
                    supported[color].append(q)
            if len(supported) > 1 and not alternative:
                raise ValueError("a scanline may have only one supported thread color")
            for color, qs in sorted(supported.items()):
                if alternative == "no_weaving":
                    paint_qs = sorted(qs)
                elif alternative == "fill_corridor":
                    paint_qs = range(low+1, high)
                else:
                    paint_qs = range(min(qs), max(qs)+1)
                for q in paint_qs:
                    if (q-t) % 2:
                        continue
                    r, c = (t-slope*q)//2, (q+slope*t)//2
                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != rail:
                        out[r, c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase unsupported marks and weave each supported local mirror span."""
    grid = _grid(input_grid)
    return _paint(grid, _inspect(grid))


_ALTERNATIVES = ("static_frame", "pixel_centers", "adjacent_frame",
                 "no_symmetry", "preserve_noise", "no_weaving", "fill_corridor",
                 "unbounded_reflection", "ignore_color")


def _witnesses(pair, info=None):
    grid, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    info = _inspect(grid) if info is None else info
    return {name for name in _ALTERNATIVES
            if not np.array_equal(out, _paint(grid, info, name))}


def _sample_strokes(rng, count, kind, scale):
    # Even transverse positions preserve the perpendicular pixel lattice under
    # reflection. Gaps are geometric choices, not object identities or slots.
    budget = 30
    gaps = [4]*(count-1)
    for i in rng.permutation(count-1):
        extra = int(rng.choice([0, 2, 4, 6]))
        gaps[int(i)] += min(extra, budget-sum(gaps))
    if count == 2:
        gaps = [int(rng.choice([6, 8, 10, 12, 14]))]
    while count > 2 and sum(gaps) < 16:
        i = int(rng.integers(count-1))
        gaps[i] += 2
    qs = [0]
    for gap in gaps:
        qs.append(qs[-1]+gap)
    lo_base, hi_base = (-8, 12) if scale == "long" else ((-2, 4) if scale == "compact" else (-6, 10))
    intervals = []
    for _ in qs:
        lo = 2*int(rng.integers(lo_base//2, 2))
        length = int(rng.integers(3, 10 if scale == "long" else (5 if scale == "compact" else 8)))
        intervals.append((lo, lo+2*(length-1)))
    if kind == "interior":
        intervals[0] = (lo_base, hi_base)
        intervals[-1] = (lo_base+2, hi_base-2)
        short = int(rng.integers(1, count-1))
        intervals[short] = (-2, 2)
    elif kind == "moving":
        # Only one outer boundary ends early; thereafter an inner boundary
        # supplies a different center. Reflecting the whole grid exchanges ends.
        side = int(rng.choice([0, count-1]))
        intervals = [(lo_base, hi_base) for _ in qs]
        intervals[side] = (-2, 2)
        if count > 3:
            other = int(rng.choice([i for i in range(1, count-1)]))
            intervals[other] = (lo_base+2, hi_base-2)
    elif kind == "equal":
        lo = 2*int(rng.integers(-1 if scale == "compact" else -4, 2))
        length = int(rng.integers(3, 5)) if scale == "compact" else int(rng.integers(4, 9))
        intervals = [(lo, lo+2*(length-1)) for _ in qs]
    elif kind == "chain":
        order = rng.permutation(count)
        start = -8
        for i in order:
            length = int(rng.integers(3, 6))
            intervals[int(i)] = (start, start+2*(length-1))
            start += 2*int(rng.integers(1, 3))
    elif kind == "groups":
        cut = int(rng.integers(2, count-1))
        intervals = [(-10, -4) if i < cut else (0, 8) for i in range(count)]
    return [(q, lo, hi, (hi-lo)//2+1) for q, (lo, hi) in zip(qs, intervals)]


def _rail_cells(strokes):
    return {(q, t) for q, lo, hi, _ in strokes for t in range(lo, hi+1, 2)}


def _pair_options(strokes, rails, t):
    frames = _frames(strokes, t)
    if not frames:
        return []
    a, b = frames[0]
    return [(q, a+b-q) for q in range(a+1, (a+b)//2+1)
            if (q-t) % 2 == 0 and (a+b-q-t) % 2 == 0
            and q <= a+b-q and (q, t) not in rails and (a+b-q, t) not in rails]


def _sample_marks(rng, strokes, colors, noise_count, pattern, teach_axis=False):
    rails = _rail_cells(strokes)
    levels = list(range(min(s[1] for s in strokes), max(s[2] for s in strokes)+1))
    options = {t: _pair_options(strokes, rails, t) for t in levels}
    options = {t: pairs for t, pairs in options.items() if pairs}
    if len(options) < 5:
        raise ValueError("too few bracketed scanlines")
    signals = {}
    current_color = int(rng.choice(colors))
    radius_preference = int(rng.integers(3, 9))
    for t, pairs in sorted(options.items()):
        if pattern == "fragmented" and t % 4 == 0:
            continue
        if pattern != "continuous" and rng.random() < .20:
            continue
        if rng.random() < .20:
            current_color = int(rng.choice(colors))
        # A correlated varying width produces actual ribbons rather than a
        # mandatory rectangular stencil. Some lines carry just an axis mark.
        radius_preference = max(0, radius_preference+int(rng.integers(-2, 3)))
        choice = min(pairs, key=lambda p: (abs(p[1]-p[0]-radius_preference), p))
        if rng.random() < .13:
            choice = pairs[int(rng.integers(len(pairs)))]
        for q in choice:
            signals[q, t] = current_color
        if rng.random() < .28:
            for q in pairs[int(rng.integers(len(pairs)))]:
                signals[q, t] = current_color
    # Teach that local symmetry is conditional on being inside the corridor.
    # This deliberately symmetric outside pair must vanish.
    t_out = int(rng.choice(list(options)))
    a, b = _frames(strokes, t_out)[0]
    extension = 2 if (a-t_out) % 2 == 0 else 1
    outside_color = int(rng.choice(colors))
    signals[a-extension, t_out] = outside_color
    signals[b+extension, t_out] = outside_color
    # Reserve a scanline for unmatched evidence; with multiple pigment colors
    # its two reflected positions have different colors and both must vanish.
    reserve = int(rng.choice([t for t in options if t != t_out]))
    signals = {p: c for p, c in signals.items() if p[1] != reserve}
    choices = [p for p in options[reserve] if p[0] < p[1]]
    if not choices:
        raise ValueError("no off-axis evidence position")
    left, right = choices[int(rng.integers(len(choices)))]
    signals[left, reserve] = colors[0]
    if len(colors) > 1:
        signals[right, reserve] = colors[1]
    if teach_axis:
        axis_choices = [(q, t) for t, pairs in options.items() if t not in (reserve, t_out)
                        for q, mirror in pairs if q == mirror]
        if not axis_choices:
            raise ValueError("no visible axis cell")
        axis_q, axis_t = axis_choices[int(rng.integers(len(axis_choices)))]
        signals = {p: c for p, c in signals.items() if p[1] != axis_t}
        signals[axis_q, axis_t] = int(rng.choice(colors))
    # Isolated asymmetric marks, hole additions and attached protrusions are
    # sampled from one coordinate neighborhood. Each is checked against its
    # actual reflected partner, so additions never change an intended thread.
    candidates = [(q, t) for t in range(min(levels)-2, max(levels)+3)
                  for q in range(-2, max(s[0] for s in strokes)+3)
                  if (q-t) % 2 == 0 and (q, t) not in rails and (q, t) not in signals]
    rng.shuffle(candidates)
    added = 0
    for q, t in candidates:
        color = colors[added % len(colors)]
        frames = _frames(strokes, t)
        if frames:
            low, high = frames[0]
            if low < q < high:
                mirror = (low+high-q, t)
                if mirror == (q, t) or signals.get(mirror) == color:
                    continue
        signals[q, t] = color
        added += 1
        if added >= noise_count:
            break
    if added < noise_count or not set(colors) <= set(signals.values()):
        raise ValueError("could not place requested noise/colors")
    return signals


def _render(rng, strokes, signals, palette, slope):
    bg, rail = palette[:2]
    rails = _rail_cells(strokes)
    all_cells = rails | set(signals)
    converted = {p: ((p[1]-slope*p[0])//2, (p[0]+slope*p[1])//2) for p in all_cells}
    r0 = min(r for r, c in converted.values())
    c0 = min(c for r, c in converted.values())
    spans = (max(r for r, c in converted.values())-r0+1,
             max(c for r, c in converted.values())-c0+1)
    if max(spans) > 30:
        raise ValueError("layout exceeds ARC dimensions")
    shape = tuple(n+int(rng.integers(0, min(30-n, 7)+1)) for n in spans)
    offsets = [int(rng.integers(n-span+1)) for n, span in zip(shape, spans)]
    if rng.random() < .40:
        k = int(rng.integers(2))
        offsets[k] = int(rng.choice([0, shape[k]-spans[k]]))
    grid = np.full(shape, bg, dtype=np.int8)
    for p, (r, c) in converted.items():
        grid[r-r0+offsets[0], c-c0+offsets[1]] = rail if p in rails else signals[p]
    # Label from construction coordinates, not from solve/input recognition.
    painted = {}
    by_level = defaultdict(dict)
    for (q, t), color in signals.items():
        by_level[t][q] = color
    for t, line in by_level.items():
        active = sorted({q for q, lo, hi, _ in strokes if lo <= t <= hi})
        if len(active) < 2:
            continue
        a, b = active[0], active[-1]
        supported = defaultdict(list)
        for q, color in line.items():
            if a < q < b and line.get(a+b-q) == color:
                supported[color].append(q)
        if len(supported) > 1:
            raise ValueError("competing threads")
        for color, qs in supported.items():
            for q in range(min(qs), max(qs)+1, 2):
                if (q, t) not in rails:
                    painted[q, t] = color
    out = np.full_like(grid, bg)
    out[grid == rail] = rail
    for (q, t), color in painted.items():
        r, c = (t-slope*q)//2, (q+slope*t)//2
        r, c = r-r0+offsets[0], c-c0+offsets[1]
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            out[r, c] = color
    return dict(input=grid, output=out)


def _make_pair(rng, index, target, settings, seen):
    required = set()
    if not target and index < 2:
        required = {"pixel_centers", "no_symmetry", "preserve_noise", "no_weaving", "fill_corridor"}
        required.add("unbounded_reflection")
        required.add("adjacent_frame" if index == 0 else "static_frame")
    if settings["colors"] > 1 and not target and index < 2:
        required.add("ignore_color")
    for _ in range(240):
        if target and settings["shift"] == "count":
            count = int(rng.integers(5, 8))
        elif not target and index < 2:
            count = int(rng.integers(3, 5 if settings["shift"] == "count" else 6))
        else:
            count = int(rng.integers(2, 5 if settings["shift"] == "count" else 7))
        kind = ("interior" if index == 0 else "moving") if not target and index < 2 else str(
            rng.choice(["interior", "moving", "random", "equal", "chain", "groups"],
                       p=[.18, .25, .22, .10, .15, .10]))
        if kind == "groups" and count < 4:
            kind = "random"
        if count == 2 and kind in ("interior", "moving"):
            kind = "equal" if rng.random() < .5 else "random"
        scale = "long" if rng.random() < .30 else "normal"
        if count == 2 and rng.random() < .7:
            scale = "compact"
        strokes = _sample_strokes(rng, count, kind, scale)
        palette = list(map(int, rng.permutation(10)[:2+settings["colors"]]))
        noise_count = int(rng.integers(9, 16)) if target and settings["shift"] == "noise" else int(rng.integers(2, 6))
        pattern = ("fragmented" if target else "continuous") if settings["shift"] == "gaps" else "random"
        slope = settings["slope"]
        if settings["mixed"]:
            slope *= (-1 if index % 2 else 1) if not target else int(rng.choice([-1, 1]))
        try:
            signals = _sample_marks(rng, strokes, palette[2:], noise_count, pattern,
                                    teach_axis=not target and index == 0)
            pair = _render(rng, strokes, signals, palette, slope)
            info = _inspect(pair["input"])
            if not np.array_equal(_paint(pair["input"], info), pair["output"]):
                raise AssertionError("construction/solver disagreement")
            if not required <= _witnesses(pair, info):
                continue
            detail = _support_details(info)
            if not target and index == 0 and detail["max_support"] < 4:
                continue
            if settings["shift"] == "gaps" and not (
                    detail["support_runs"] >= 3 if target else detail["support_runs"] <= 2):
                continue
            if _key(pair["input"]) in seen:
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded loom construction exhausted")


def _count(name, value, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    settings = dict(colors=int(rng.integers(1, 4)), slope=int(rng.choice([-1, 1])),
                    mixed=bool(rng.integers(2)), shift=str(rng.choice(["count", "noise", "gaps"])))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair = _make_pair(rng, index, split == "test", settings, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _semantic_check(grid, out, info):
    """Grid-derived partner/span check, separate from _paint's scanline builder."""
    bg, rail, slope = info["bg"], info["rail"], info["slope"]
    assert np.array_equal(out[grid == rail], grid[grid == rail]), "rail was modified"
    expected = np.full_like(grid, bg)
    expected[grid == rail] = rail
    rows, cols = np.indices(grid.shape)
    qgrid, tgrid = cols-slope*rows, rows+slope*cols
    # Test each seed point's mate and union its symmetric horizontal-in-loom
    # span. Interior pairs naturally disappear inside the outermost span.
    for t, line in info["marks"].items():
        active = [s[0] for s in info["strokes"] if s[1] <= t <= s[2]]
        if len(set(active)) < 2:
            continue
        a, b = min(active), max(active)
        supported_colors = set()
        for q, color in line.items():
            mirror = a+b-q
            if a < q < b and line.get(mirror) == color:
                supported_colors.add(color)
                mask = ((tgrid == t) & (qgrid >= min(q, mirror))
                        & (qgrid <= max(q, mirror)) & (grid != rail))
                expected[mask] = color
        assert len(supported_colors) <= 1, "ambiguous thread colors"
    assert np.array_equal(out, expected), "incorrect supported span"
    assert set(map(int, np.unique(out))) <= set(map(int, np.unique(grid)))


def _support_details(info):
    supported_levels, axis_only, max_support, matched = [], 0, 0, 0
    for t, line in sorted(info["marks"].items()):
        frames = _frames(info["strokes"], t)
        if not frames:
            continue
        a, b = frames[0]
        qs = [q for q, color in line.items() if a < q < b and line.get(a+b-q) == color]
        if qs:
            supported_levels.append(t)
            matched += len(qs)
            max_support = max(max_support, len(qs))
            axis_only += len(qs) == 1
    runs = sum(i == 0 or t != supported_levels[i-1]+1 for i, t in enumerate(supported_levels))
    return dict(unmatched=sum(map(len, info["marks"].values()))-matched,
                support_runs=runs, max_support=max_support, axis_only=axis_only)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, witnesses = set(), {"train": set(), "test": set()}
    pigment_counts, slopes = {s: set() for s in puzzle}, {s: set() for s in puzzle}
    details = {s: [] for s in puzzle}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            assert grid.shape == out.shape
            assert _key(grid) not in seen, "duplicate input"
            seen.add(_key(grid))
            info = _inspect(grid)
            _semantic_check(grid, out, info)
            assert np.array_equal(solve(grid), out)
            witnesses[split] |= _witnesses(pair, info)
            pigment_counts[split].add(len(np.unique(grid))-2)
            slopes[split].add(info["slope"])
            details[split].append(_support_details(info))
    assert witnesses["test"] <= witnesses["train"], "untaught alternative-rule distinction"
    essential = {"static_frame", "pixel_centers", "adjacent_frame", "no_symmetry",
                 "preserve_noise", "no_weaving", "fill_corridor", "unbounded_reflection"}
    assert essential <= witnesses["train"], "training lacks a mechanism witness"
    assert max(pigment_counts["test"]) <= max(pigment_counts["train"])
    assert slopes["test"] <= slopes["train"], "untaught slope"
    if any(d["axis_only"] for d in details["test"]):
        assert any(d["axis_only"] for d in details["train"]), "untaught self-supported axis mark"
    if any(d["max_support"] >= 4 for d in details["test"]):
        assert any(d["max_support"] >= 4 for d in details["train"]), "untaught multiple supported pairs"


def _features(pair):
    grid, out = pair["input"], pair["output"]
    info = _inspect(grid)
    frames = [_frames(info["strokes"], t) for t in info["marks"]]
    centers = {a+b for frame in frames for a, b in frame}
    changed = grid != out
    return dict(**_support_details(info), shape=grid.shape, slope=info["slope"], background=info["bg"], rail=info["rail"],
                colors=len(np.unique(grid))-2, strokes=len(info["strokes"]),
                lengths=tuple(s[3] for s in info["strokes"]), centers=len(centers),
                deleted=int(np.sum(changed & (out == info["bg"]))),
                added=int(np.sum(changed & (grid == info["bg"]))),
                recolored=int(np.sum(changed & (out != info["bg"]) & (grid != info["bg"]))),
                edge=bool(np.any(grid[0] != info["bg"]) or np.any(grid[-1] != info["bg"])
                          or np.any(grid[:, 0] != info["bg"]) or np.any(grid[:, -1] != info["bg"])))


def _observed_shifts(puzzle):
    features = {s: [_features(p) for p in puzzle[s]] for s in puzzle}
    shifts = {name for name, key in (("count", "strokes"), ("noise", "unmatched"))
              if min(f[key] for f in features["test"]) > max(f[key] for f in features["train"])}
    if (max(f["support_runs"] for f in features["train"]) <= 2
            and min(f["support_runs"] for f in features["test"]) >= 3):
        shifts.add("gaps")
    return shifts


def self_test():
    # Hand fixture: q=0/12 rails, endpoints t=0..8. The t=3 marks q=3,9
    # support one another even though neither rail has a pixel at t=3.
    # t=4 has a supported axis singleton, while t=5 has an unmatched mark.
    grid = np.zeros((11, 12), dtype=np.int8)
    for q in (0, 12):
        for t in range(0, 9, 2):
            grid[(t-q)//2+6, (q+t)//2] = 5
    for q, t in ((3, 3), (9, 3), (6, 4), (3, 5)):
        grid[(t-q)//2+6, (q+t)//2] = 2
    expected = np.zeros_like(grid)
    expected[grid == 5] = 5
    for q, t in ((3, 3), (5, 3), (7, 3), (9, 3), (6, 4)):
        expected[(t-q)//2+6, (q+t)//2] = 2
    assert np.array_equal(solve(grid), expected)
    assert "pixel_centers" in _witnesses(dict(input=grid, output=expected))
    fingerprints, structures, shapes, records, shifts = set(), set(), set(), [], set()
    for seed in range(42):
        # Vary requested counts as well as geometric family choices.
        b = 1+seed % 4
        a = max(b+1, 2)+(seed//4) % (11-max(b+1, 2))
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        observed = _observed_shifts(puzzle)
        assert observed, "missing actual train/test shift"
        shifts |= observed
        fingerprint = tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s])
        fingerprints.add(fingerprint)
        if seed < 3:
            repeat = generate_puzzle(seed, a, b)
            assert fingerprint == tuple(_key(p["input"]) for s in ("train", "test") for p in repeat[s])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair["input"])
                q0 = min(s[0] for s in info["strokes"])
                t0 = min(s[1] for s in info["strokes"])
                structures.add(tuple((q-q0, lo-t0, hi-t0) for q, lo, hi, n in info["strokes"]))
                shapes.add(pair["input"].shape)
                records.append(_features(pair))
    assert len(fingerprints) == 42 and len(structures) >= 150 and len(shapes) >= 50
    assert shifts == {"count", "noise", "gaps"}
    assert {r["slope"] for r in records} == {-1, 1}
    assert {r["colors"] for r in records} == {1, 2, 3}
    assert {2, 3, 4, 5, 6, 7} <= {r["strokes"] for r in records}
    assert len({r["background"] for r in records}) == 10
    assert len({r["rail"] for r in records}) == 10
    assert sum(r["centers"] > 1 for r in records) >= 80
    assert sum(r["edge"] for r in records) >= 20
    assert all(r["deleted"] > 0 for r in records)
    assert sum(r["recolored"] > 0 for r in records) >= 10
    assert any(r["added"] >= 20 for r in records)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(900+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = generate_puzzle(117, 2, 1)
    info = _inspect(bad["test"][0]["input"])
    r, c = np.argwhere(bad["test"][0]["input"] == info["rail"])[0]
    bad["test"][0]["output"][r, c] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("modified rail accepted")
    bad = generate_puzzle(118, 2, 1)
    pair = bad["test"][0]
    info = _inspect(pair["input"])
    r, c = np.argwhere((pair["input"] == info["bg"]) & (pair["output"] != info["bg"]))[0]
    pair["output"][r, c] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing thread interior accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations, analytic fixture)")
