"""Diagonal pulse weaving: interval boundaries that also encode pulse periods.

No puzzle-specific semantic settings are needed by solve. Colors are copied
from visible strokes. Labels use a ray launcher; solve checks sample distances.
Both implementations share the input parser, so this is self-authored checking.
"""
from __future__ import annotations

import math
import numpy as np


_RIVALS = (
    "solid", "unbracketed", "static", "centers", "adjacent", "all_sources",
    "shortest", "longest", "farthest", "nearest_only", "tie_left",
    "erase_same_ties", "period_plus", "period_minus", "phase_shift",
    "color_low", "color_high", "fixed_two",
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _parse(grid):
    """Recognize homogeneous maximal diagonal runs using only input cells."""
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    background = int(colors[np.argmax(counts)])
    if counts.max() <= a.size // 2:
        raise ValueError("the background must occupy more than half the input")
    foreground = np.argwhere(a != background)
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in foreground:
            lines.setdefault(int(c - slope*r), []).append(int(r))
        strokes, valid = [], True
        for q, rows in lines.items():
            rows.sort()
            runs = []
            for row in rows:
                if not runs or row != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(row)
            for run in runs:
                shades = {int(a[r, q + slope*r]) for r in run}
                if not 2 <= len(run) <= 12 or len(shades) != 1:
                    valid = False
                    break
                strokes.append({"q": q, "lo": 2*run[0] + slope*q,
                                "hi": 2*run[-1] + slope*q,
                                "length": len(run), "color": shades.pop()})
            if not valid:
                break
        if valid and 2 <= len(strokes) <= 12:
            candidates.append((slope, sorted(strokes, key=lambda s: (s["q"], s["lo"]))))
    if len(candidates) != 1:
        raise ValueError("input must have one uniquely recognizable diagonal slope")
    slope, strokes = candidates[0]
    return {"background": background, "slope": slope, "strokes": strokes}


def _sample_colors(qs, sources, background, rival=""):
    """Compute colors for perpendicular lattice samples, not drawing rays."""
    positions = np.array([s["q"] for s in sources])
    lengths = np.array([s["length"] for s in sources])
    shades = np.array([s["color"] for s in sources])
    distances = np.abs(qs[:, None] - positions[None, :])
    steps = (distances + 1) // 2
    if rival == "period_plus":
        lengths = lengths + 1
    elif rival == "period_minus":
        lengths = np.maximum(1, lengths - 1)
    elif rival == "fixed_two":
        lengths = np.full_like(lengths, 2)
    phase = 1 if rival == "phase_shift" else 0
    firing = (distances > 0) & ((steps - 1) % lengths == phase)
    if rival == "solid":
        firing[:] = True
    elif rival == "nearest_only":
        firing &= distances == distances.min(axis=1, keepdims=True)
    score = distances.copy()
    if rival == "shortest":
        score = 100*lengths + distances
    elif rival == "longest":
        score = -100*lengths + distances
    elif rival == "farthest":
        score = -distances
    elif rival == "color_low":
        score = 100*shades + distances
    elif rival == "color_high":
        score = -100*shades + distances
    score = np.where(firing, score, 10000)
    minimum = score.min(axis=1)
    tied = score == minimum[:, None]
    winners = score.argmin(axis=1)
    answer = shades[winners].copy()
    answer[minimum == 10000] = background
    if rival != "tie_left":
        for i in np.flatnonzero(tied.sum(axis=1) > 1):
            if rival == "erase_same_ties" or len(set(shades[tied[i]].tolist())) > 1:
                answer[i] = background
    return answer


def _predict(grid, rival=""):
    a = _grid(grid)
    info = _parse(a)
    rr, cc = np.indices(a.shape)
    transverse = cc - info["slope"]*rr
    longitudinal = rr + info["slope"]*cc
    result = a.copy()
    strokes = info["strokes"]
    for level in range(int(longitudinal.min()), int(longitudinal.max()) + 1):
        active = [s for s in strokes if (rival == "static" or s["lo"] <= level <= s["hi"])
                  and (rival != "centers" or (level - s["q"]) % 2 == 0)]
        if rival == "unbracketed":
            brackets = [(None, None, active)] if active else []
        elif rival == "adjacent":
            brackets = [(x["q"], y["q"], [x, y]) for x, y in zip(strokes, strokes[1:])
                        if x in active and y in active and x["q"] < y["q"]]
        elif len(active) >= 2 and active[0]["q"] < active[-1]["q"]:
            sources = active if rival == "all_sources" else [active[0], active[-1]]
            brackets = [(active[0]["q"], active[-1]["q"], sources)]
        else:
            brackets = []
        for left, right, sources in brackets:
            mask = (a == info["background"]) & (longitudinal == level)
            if left is not None:
                mask &= (transverse > left) & (transverse < right)
            if np.any(mask):
                result[mask] = _sample_colors(transverse[mask], sources, info["background"], rival)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family member without examples, RNG, or hidden settings."""
    return _predict(input_grid)


def _ray_output(grid, info=None):
    """Separate label algorithm: launch every Lth lattice sample inward."""
    info = _parse(grid) if info is None else info
    slope, bg, strokes = info["slope"], info["background"], info["strokes"]
    result = grid.copy()
    # All possible longitudinal coordinates, including between pixel centers.
    rr, cc = np.indices(grid.shape)
    levels = rr + slope*cc
    for t in range(int(levels.min()), int(levels.max()) + 1):
        active = [s for s in strokes if s["lo"] <= t <= s["hi"]]
        if len(active) < 2 or active[0]["q"] == active[-1]["q"]:
            continue
        left, right = active[0], active[-1]
        proposals = {}
        for source, sign in ((left, 1), (right, -1)):
            first = 1 if (t - source["q"]) % 2 else 2
            q = source["q"] + sign*first
            while left["q"] < q < right["q"]:
                r, c = (t - slope*q)//2, (q + slope*t)//2
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == bg:
                    proposals.setdefault((r, c), []).append((abs(q-source["q"]), source["color"]))
                q += sign*2*source["length"]
        for cell, offers in proposals.items():
            closest = min(distance for distance, _ in offers)
            shades = {color for distance, color in offers if distance == closest}
            if len(shades) == 1:
                result[cell] = shades.pop()
    return result


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _render(rng, specs, bg, colors, slope, padding=True):
    cells = []
    for i, (q, lo, length) in enumerate(specs):
        if (lo - q) % 2:
            raise ValueError("endpoint parity")
        for t in range(lo, lo + 2*length, 2):
            cells.append(((t-slope*q)//2, (q+slope*t)//2, colors[i % len(colors)]))
    if len({(r, c) for r, c, _ in cells}) != len(cells):
        raise ValueError("overlapping strokes")
    r0, c0 = min(x[0] for x in cells), min(x[1] for x in cells)
    height = max(x[0] for x in cells) - r0 + 1
    width = max(x[1] for x in cells) - c0 + 1
    if max(height, width) > 30:
        raise ValueError("does not fit")
    extras = [int(rng.integers(0, min(6, 30-span)+1)) if padding else 0
              for span in (height, width)]
    offsets = [int(rng.integers(extra+1)) for extra in extras]
    grid = np.full((height+extras[0], width+extras[1]), bg, dtype=np.int8)
    for r, c, shade in cells:
        grid[r-r0+offsets[0], c-c0+offsets[1]] = shade
    # Reflecting the complete geometry does not change any semantic convention.
    if rng.random() < .5:
        grid = grid[::-1, ::-1].copy()
    info = _parse(grid)
    if len(info["strokes"]) != len(specs):
        raise ValueError("strokes merged")
    return {"input": grid, "output": _ray_output(grid, info)}


def _fallback(specs, bg, colors, slope, seen):
    """Fourteen distinct heights suffice for at most thirteen preceding inputs."""
    cells = []
    for i, (q, lo, length) in enumerate(specs):
        for t in range(lo, lo+2*length, 2):
            cells.append(((t-q)//2, (t+q)//2, colors[i % len(colors)]))
    r0, c0 = min(x[0] for x in cells), min(x[1] for x in cells)
    h, w = max(x[0] for x in cells)-r0+1, max(x[1] for x in cells)-c0+1
    for extra in range(14):
        grid = np.full((h+extra, w), bg, dtype=np.int8)
        for r, c, shade in cells:
            grid[r-r0, c-c0] = shade
        if slope == -1:
            grid = grid[:, ::-1].copy()
        if _key(grid) not in seen:
            return {"input": grid, "output": _ray_output(grid)}
    raise RuntimeError("unreachable: fourteen unique fallback heights exhausted")


def _recurrence(pair):
    """Observed, unoccluded first and repeated pulses from a two-bar lesson."""
    a, out = pair["input"], pair["output"]
    info = _parse(a)
    strokes = info["strokes"]
    if len(strokes) != 2 or strokes[0]["length"] != strokes[1]["length"]:
        return set()
    rr, cc = np.indices(a.shape)
    q, t = cc-info["slope"]*rr, rr+info["slope"]*cc
    periods = set()
    for source, other in ((strokes[0], strokes[1]), (strokes[1], strokes[0])):
        d = (np.abs(q-source["q"])+1)//2
        interior = ((a == info["background"]) & (q > strokes[0]["q"])
                    & (q < strokes[1]["q"]) & (t >= max(source["lo"], other["lo"]))
                    & (t <= min(source["hi"], other["hi"]))
                    & (np.abs(q-source["q"]) < np.abs(q-other["q"])))
        length = source["length"]
        visible = interior & (out == source["color"])
        if np.any(visible & (d == 1)) and np.any(visible & (d == length+1)):
            periods.add(length)
    return periods


def _collision_stats(pair):
    a, out = pair["input"], pair["output"]
    info = _parse(a)
    stats = {"short_near": 0, "long_near": 0, "different_tie": 0,
             "same_tie": 0, "far_only": 0}
    for r, c in np.argwhere(a == info["background"]):
        q, t = int(c-info["slope"]*r), int(r+info["slope"]*c)
        active = [s for s in info["strokes"] if s["lo"] <= t <= s["hi"]]
        if len(active) < 2 or not active[0]["q"] < q < active[-1]["q"]:
            continue
        x, y = active[0], active[-1]
        dx, dy = q-x["q"], y["q"]-q
        fx = (((dx+1)//2)-1) % x["length"] == 0
        fy = (((dy+1)//2)-1) % y["length"] == 0
        if fx and fy:
            if dx == dy:
                name = "same_tie" if x["color"] == y["color"] else "different_tie"
                expected = x["color"] if name == "same_tie" else info["background"]
                if out[r, c] == expected:
                    stats[name] += 1
            elif x["color"] != y["color"]:
                near, far = (x, y) if dx < dy else (y, x)
                if out[r, c] == near["color"]:
                    stats["short_near"] += near["length"] < far["length"]
                    stats["long_near"] += near["length"] > far["length"]
        elif fx != fy:
            stats["far_only"] += (fy and dx < dy) or (fx and dy < dx)
    return stats


def _different(pair, rival):
    return not np.array_equal(pair["output"], _predict(pair["input"], rival))


def _simple_lesson(rng, bg, markers, slope):
    length = int(rng.integers(2, 6))
    repeats = int(rng.integers(1, 3))
    # Midpoint distance = repeats*length+1, so a same-color tie is observable.
    gap = 4*(repeats*length+1)
    if gap//2 + length > 28:
        gap = 4*(length+1)
    before = int(rng.integers(length))
    lo = -2*before
    specs = [(0, lo, length), (gap, lo, length)]
    color = int(rng.choice(markers))
    return _render(rng, specs, bg, [color], slope)


def _complex_lesson(rng, bg, markers, slope, seen):
    # Arithmetic witness construction, with variable periods, interior count,
    # locations, colors, containment and endpoints. Not stored grid templates.
    options = [(2, 3), (2, 4), (2, 6), (3, 6), (2, 8), (4, 8)]
    for _ in range(128):
        x, y = options[int(rng.integers(len(options)))]
        if rng.random() < .5:
            x, y = y, x
        gap = 4*(math.lcm(x, y)+1)
        n = int(rng.integers(3, 6))
        inner = sorted(int(q) for q in rng.choice(range(3, gap-2), n-2, replace=False))
        if any(b-a < 3 for a, b in zip(inner, inner[1:])):
            continue
        specs = [(0, -2*int(rng.integers(x-1)), x)]
        for q in inner:
            lo = 1 if q % 2 else 2
            specs.append((q, lo, int(rng.integers(2, 6))))
        specs.append((gap, -2*int(rng.integers(y-1)), y))
        first, last = [int(v) for v in rng.choice(markers, 2, replace=False)]
        colors = [first] + [int(rng.choice(markers)) for _ in inner] + [last]
        try:
            pair = _render(rng, specs, bg, colors, slope)
        except ValueError:
            continue
        stats = _collision_stats(pair)
        if (_key(pair["input"]) not in seen
                and all(stats[k] for k in ("short_near", "long_near", "different_tie", "far_only"))
                and all(_different(pair, rival) for rival in _RIVALS if rival != "erase_same_ties")):
            return pair
    return _fallback([(0, 0, 2), (8, 2, 2), (20, 0, 4)], bg,
                     [markers[0], markers[1], markers[2]], slope, seen)


def _random_pair(rng, bg, markers, slope, count, seen):
    for _ in range(192):
        qs = [0]
        for _ in range(count-1):
            qs.append(qs[-1] + int(rng.integers(3, 10)))
        lengths = [int(rng.integers(2, 10)) for _ in qs]
        mode = int(rng.integers(5))
        specs = []
        for i, (q, length) in enumerate(zip(qs, lengths)):
            if mode == 0:  # aligned starts, respecting endpoint parity
                lo = q % 2
            elif mode == 1:  # overlapping, freely contained intervals
                lo = -2*int(rng.integers(length-1)) + q % 2
            elif mode == 2:  # staggered interval relay
                lo = 2*i - int(rng.integers(0, 5))
                lo += (q-lo) % 2
            elif mode == 3:  # two longitudinally separate assemblies
                group = i >= count//2
                length = min(length, 4)
                lo = 12*group + q % 2
            else:
                lo = int(rng.integers(-10, 11))
                lo += (q-lo) % 2
            specs.append((q, lo, length))
        colors = [int(rng.choice(markers)) for _ in qs]
        try:
            pair = _render(rng, specs, bg, colors, slope)
        except ValueError:
            continue
        if _key(pair["input"]) not in seen and np.any(pair["input"] != pair["output"]):
            return pair
    return _fallback([(3*i, (3*i) % 2, 2) for i in range(count)],
                     bg, markers, slope, seen)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    markers = palette[:int(rng.integers(3, 6))]
    backgrounds = [c for c in palette if c not in markers]
    vary_bg = bool(rng.random() < .55)
    base_bg, base_slope = backgrounds[0], int(rng.choice([-1, 1]))
    mixed_slopes = bool(rng.random() < .6)
    slots = [int(i) for i in rng.permutation(num_train)]
    simple_slot, complex_slot = slots[:2]
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(num_train):
        bg = int(rng.choice(backgrounds)) if vary_bg else base_bg
        slope = base_slope * (-1 if mixed_slopes and i % 2 else 1)
        if i == simple_slot:
            pair = _simple_lesson(rng, bg, markers, slope)
        elif i == complex_slot:
            pair = _complex_lesson(rng, bg, markers, slope, seen)
        else:
            pair = _random_pair(rng, bg, markers, slope, int(rng.integers(2, 6)), seen)
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
    largest = max(len(_parse(p["input"])["strokes"]) for p in puzzle["train"])
    for i in range(num_test):
        bg = int(rng.choice(backgrounds)) if vary_bg else base_bg
        slope = int(rng.choice([-1, 1])) if mixed_slopes else base_slope
        count = int(rng.integers(largest+1, 9))
        pair = _random_pair(rng, bg, markers, slope, count, seen)
        puzzle["test"].append(pair)
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned grids, visible witnesses and rule invariants, not RNG plans."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = (a.shape, tuple(int(c) for c in a.flat))
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(a)
            assert np.array_equal(a[a != info["background"]], out[a != info["background"]]), "stroke changed"
            assert set(np.unique(out)) <= set(np.unique(a)), "invented output color"
            assert np.array_equal(out, _ray_output(a, info)), "pulse, boundary, phase, tie or preservation error"
            assert np.array_equal(out, solve(a)), "distance and ray algorithms disagree"
            assert np.any(a != out), "generated examples must be informative"
            infos[split].append(info)
    train = [{k: np.asarray(v) for k, v in p.items()} for p in puzzle["train"]]
    assert any(_recurrence(p) for p in train), "missing visible run-length recurrence"
    stats = [_collision_stats(p) for p in train]
    for concept in ("short_near", "long_near", "different_tie", "same_tie", "far_only"):
        assert any(s[concept] for s in stats), f"missing training witness: {concept}"
    for rival in _RIVALS:
        assert any(_different(p, rival) for p in train), f"training admits rival: {rival}"
    assert max(len(i["strokes"]) for i in infos["test"]) > max(len(i["strokes"]) for i in infos["train"]), "missing compositional count shift"


def _features(pair):
    a = pair["input"]
    info = _parse(a)
    strokes = info["strokes"]
    q0, t0 = min(s["q"] for s in strokes), min(s["lo"] for s in strokes)
    layout = tuple((s["q"]-q0, s["lo"]-t0, s["length"]) for s in strokes)
    foreground = a != info["background"]
    edge = bool(np.any(foreground[0]) or np.any(foreground[-1])
                or np.any(foreground[:, 0]) or np.any(foreground[:, -1]))
    # Reconstruct geometric interaction components from interval overlap.
    pending, groups = set(range(len(strokes))), []
    while pending:
        reached, stack = [], [pending.pop()]
        while stack:
            i = stack.pop()
            reached.append(i)
            neighbors = {j for j in pending if max(strokes[i]["lo"], strokes[j]["lo"])
                         <= min(strokes[i]["hi"], strokes[j]["hi"])}
            pending -= neighbors
            stack.extend(neighbors)
        groups.append(reached)
    staggered = any(len(g) > 2 and max(strokes[i]["lo"] for i in g)
                    > min(strokes[i]["hi"] for i in g) for g in groups)
    return {"shape": a.shape, "layout": layout, "count": len(strokes),
            "slope": info["slope"], "background": info["background"],
            "periods": tuple(s["length"] for s in strokes), "edge": edge,
            "groups": len(groups), "staggered": staggered,
            "colors": tuple(sorted({s["color"] for s in strokes})),
            "changed": int(np.sum(a != pair["output"]))}


def self_test():
    """Bounded analytical, adversarial, metamorphic and 42-seed checks."""
    # Exact hand-counted fixture. Two length-2 strokes, q gap 12, t interval
    # 0..2 before translation. The t=1 line is between boundary pixel centers.
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[(6, 7), (0, 1)] = 3
    grid[(0, 1), (6, 7)] = 7
    expected = grid.copy()
    left_cells = [(5, 1), (6, 1), (4, 3), (2, 5), (6, 2)]
    right_cells = [(1, 5), (5, 2), (3, 4), (1, 6), (2, 6)]
    for r, c in left_cells:
        expected[r, c] = 3
    for r, c in right_cells:
        expected[r, c] = 7
    assert expected[3, 3] == expected[4, 4] == 0  # destructive ties
    assert np.array_equal(solve(grid), expected)
    assert np.array_equal(_ray_output(grid), expected)
    same = grid.copy()
    same[same == 7] = 3
    same_expected = expected.copy()
    same_expected[same_expected == 7] = 3
    same_expected[3, 3] = same_expected[4, 4] = 3
    assert np.array_equal(solve(same), same_expected)
    for inp, out in ((grid, expected), (same, same_expected)):
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(out, turn))
        assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
        mapping = np.array([8, 1, 2, 0, 4, 5, 6, 9, 3, 7], dtype=np.int8)
        assert np.array_equal(solve(mapping[inp]), mapping[out])
    # Three-stroke analytical relay used in task_spec.md. Original coordinates
    # q=0:[0,2], q=8:[2,4], q=20:[0,6], translated ten rows downward.
    relay = np.zeros((12, 14), dtype=np.int8)
    for q, lo, hi, shade in ((0, 0, 2, 3), (8, 2, 4, 5), (20, 0, 6, 7)):
        for t in range(lo, hi+1, 2):
            relay[(t-q)//2+10, (q+t)//2] = shade
    relay_out = solve(relay)
    checkpoints = ((0, 2, 3), (0, 18, 7), (0, 10, 0), (1, 3, 7),
                   (2, 6, 3), (0, 14, 3), (4, 10, 5), (5, 3, 0),
                   (2, 22, 0), (0, 4, 0))
    for t, q, shade in checkpoints:
        assert relay_out[(t-q)//2+10, (q+t)//2] == shade
    assert np.array_equal(relay_out, _ray_output(relay))
    # Supported input-only scope is broader than the default sampler: lengths
    # 10..12, up to twelve strokes, and reuse of one q in disjoint intervals.
    rng = np.random.default_rng(910)
    for length in range(2, 13):
        pair = _render(rng, [(0, 0, length), (12, 0, length)], 0, [3, 7], 1)
        assert np.array_equal(solve(pair["input"]), pair["output"])
    for count in range(2, 13):
        specs = [(3*i, (3*i) % 2, 2) for i in range(count)]
        pair = _render(rng, specs, 0, [3, 5, 7], 1)
        assert len(_parse(pair["input"])["strokes"]) == count
        assert np.array_equal(solve(pair["input"]), pair["output"])
    pair = _render(rng, [(0, 0, 2), (12, 0, 2), (0, 20, 2), (12, 20, 2)],
                   0, [3, 7], -1)
    assert np.array_equal(solve(pair["input"]), pair["output"])
    seen = set()
    for i in range(14):
        p = _fallback([(0, 0, 2), (8, 2, 2), (20, 0, 4)], 0, [3, 5, 7], 1, seen)
        seen.add(_key(p["input"]))
        assert all(_different(p, r) for r in _RIVALS if r != "erase_same_ties")
    assert len(seen) == 14
    records, signatures, puzzles = [], set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        signatures.add(_fingerprint(puzzle))
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                records.append(_features(pair))
    assert len(signatures) == 42
    assert len({r["shape"] for r in records}) >= 40
    assert len({r["layout"] for r in records}) >= 160
    assert {r["slope"] for r in records} == {-1, 1}
    assert len({r["background"] for r in records}) == 10
    assert set(range(2, 9)) <= {r["count"] for r in records}
    assert set(range(2, 10)) <= {p for r in records for p in r["periods"]}
    assert {r["edge"] for r in records} == {False, True}
    assert any(r["groups"] > 1 for r in records)
    assert any(r["staggered"] for r in records)
    assert any(len(r["colors"]) == 1 for r in records)
    assert any(len(r["colors"]) >= 3 for r in records)
    # Every legal explicit count, including the minimal two-demo curriculum.
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(8000 + train*10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corrupted = {s: [{k: a.copy() for k, a in pair.items()} for pair in puzzles[0][s]]
                 for s in ("train", "test")}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Correct labels alone do not establish the collision/boundary lessons.
    weak = {"train": [{"input": same, "output": same_expected},
                      {"input": np.rot90(same), "output": np.rot90(same_expected)}],
            "test": puzzles[0]["test"]}
    try:
        validate_generated_puzzle(weak)
    except AssertionError:
        pass
    else:
        raise AssertionError("inadequate teaching accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
