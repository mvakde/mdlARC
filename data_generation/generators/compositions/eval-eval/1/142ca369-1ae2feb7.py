"""A run-programmed traveling clock whose matching pulses turn point mirrors.

Only input geometry and maximal run lengths enter solve(). No parent code,
stored scenes, seed, or generation metadata is consulted by the solver.
"""
from __future__ import annotations

import math
import itertools
from functools import lru_cache
from numbers import Integral

import numpy as np


# All non-dividing period pairs whose collision and following short-period
# beat leave room for a row-face point on the 30-cell monotone construction.
_WITNESS_PERIODS = tuple((a, b) for a in range(2, 8) for b in range(a + 1, 8)
                         if b % a and math.lcm(a, b) + a <= 26)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        todo, cells = [root], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining and a[p] == a[root]:
                    remaining.remove(p)
                    todo.append(p)
        result.append(set(cells))
    return result


def _decode(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size:
        raise ValueError("the background must occupy a strict input majority")
    components = _components(a, bg)
    sources = []
    for cells in components:
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        if len(cells) == 3 and len(rows) == len(cols) == 2:
            missing = next(iter({(r, c) for r in rows for c in cols} - cells))
            elbow = (min(rows) + max(rows) - missing[0],
                     min(cols) + max(cols) - missing[1])
            sources.append((cells, elbow, (elbow[0] - missing[0],
                                          elbow[1] - missing[1])))
    if len(sources) != 1:
        raise ValueError("need exactly one monochromatic L launcher")
    source_cells, elbow, direction = sources[0]
    source_color = int(a[elbow])
    tail_options = []
    for axis in (0, 1):
        positions = []
        for distance in range(2, 31):
            p = list(elbow)
            p[axis] -= distance * direction[axis]
            p = tuple(p)
            if not _inside(p, a.shape):
                break
            positions.append(p)
        if any(a[p] != bg for p in positions):
            tail_options.append((axis, positions))
    if len(tail_options) != 1:
        raise ValueError("a tail must extend exactly one backward L arm")
    tail_axis, positions = tail_options[0]
    if a[positions[0]] != bg:
        raise ValueError("leave a background cell between launcher arm and tail")
    program, tail_cells = [], set()
    cursor = 0
    while cursor < len(positions):
        p = positions[cursor]
        color = int(a[p])
        if color == bg:
            cursor += 1
            continue
        if color == source_color:
            raise ValueError("the launcher color is reserved for the L")
        start = cursor
        while cursor < len(positions) and a[positions[cursor]] == color:
            tail_cells.add(positions[cursor])
            cursor += 1
        program.append({"color": color, "length": cursor - start,
                        "distance": start + 2})
    if not 1 <= len(program) <= 4 or any(b["length"] > 7 for b in program):
        raise ValueError("tail requires 1..4 maximal runs of length 1..7")
    occupied = set(map(tuple, np.argwhere(a != bg)))
    mirrors = {p: int(a[p]) for p in occupied - source_cells - tail_cells}
    colors = {b["color"] for b in program}
    for p, color in mirrors.items():
        if color not in colors:
            raise ValueError("mirror colors must occur in the clock program")
        r, c = p
        if any(q in occupied for q in ((r - 1, c), (r + 1, c),
                                       (r, c - 1), (r, c + 1))):
            raise ValueError("point mirrors must be cardinally isolated")
    return {"background": bg, "source_cells": source_cells,
            "source_color": source_color, "elbow": elbow,
            "direction": direction, "tail_axis": tail_axis,
            "tail_cells": tail_cells, "program": program,
            "mirrors": mirrors, "occupied": occupied}


def _pulse(program, step, priority="near", timing="length"):
    beat = step - 1
    firing = []
    for i, block in enumerate(program):
        period = block["length"]
        if timing == "plus_one":
            period += 1
        elif timing == "minus_one":
            period = max(1, period - 1)
        elif timing == "unit":
            period = 1
        local_beat = beat - (block["distance"] - 2) if timing == "gap_phase" else beat
        if local_beat % period == 0:
            firing.append(i)
    if not firing:
        return None, ()
    if priority == "far":
        winner = firing[-1]
    elif priority == "short":
        winner = min(firing, key=lambda i: (program[i]["length"], i))
    elif priority == "long":
        winner = min(firing, key=lambda i: (-program[i]["length"], i))
    else:
        winner = firing[0]
    return int(program[winner]["color"]), tuple(firing)


def _trace(a, *, priority="near", timing="length", reflection="match",
           reset=False, coordinate_clock=False, strict=True):
    info = _decode(a)
    out = a.copy()
    direction = info["direction"]
    p = tuple(info["elbow"][i] + direction[i] for i in (0, 1))
    step, local_step = 1, 1
    seen, records = set(), []
    contacts = []
    while _inside(p, a.shape):
        if p in seen or p in info["occupied"]:
            if strict:
                raise ValueError("carrier must not revisit a cell or hit an input object")
            break
        seen.add(p)
        clock_step = (max(abs(p[i] - info["elbow"][i]) for i in (0, 1))
                      if coordinate_clock else local_step if reset else step)
        shade, firing = _pulse(info["program"], clock_step,
                               priority, timing)
        candidates = []
        for axis in (0, 1):
            mid = list(p)
            mid[axis] += direction[axis]
            mid = tuple(mid)
            if mid in info["mirrors"]:
                candidates.append((mid, axis))
        if len(candidates) > 1:
            if strict:
                raise ValueError("two point mirrors compete at one directed contact")
            break
        hit = candidates[0] if candidates else None
        active = bool(hit and (reflection == "all" or
                               (reflection == "match" and shade == info["mirrors"][hit[0]])))
        if reflection == "none":
            active = False
        if shade is not None:
            out[p] = shade
        records.append({"position": p, "step": step, "color": shade,
                        "firing": firing, "direction": direction,
                        "mirror": hit[0] if hit else None,
                        "axis": hit[1] if hit else None, "active": active})
        if hit:
            contacts.append(records[-1])
        if active:
            d = list(direction)
            d[hit[1]] *= -1
            direction = tuple(d)
            if reset:
                # Rival: restart with the next flight's first cell as beat 1.
                local_step = 0
        p = tuple(p[i] + direction[i] for i in (0, 1))
        step += 1
        local_step += 1
    info["records"], info["contacts"] = records, contacts
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode a tail clock and trace color-conditional reflections from its input."""
    return _trace(_grid(input_grid))[0]


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _place_point(a, point, color):
    if not _inside(point, a.shape):
        return None
    info = _decode(a)
    if point in info["occupied"]:
        return None
    r, c = point
    if any(q in info["occupied"] for q in ((r - 1, c), (r + 1, c),
                                           (r, c - 1), (r, c + 1))):
        return None
    # Neither backward arm may acquire a false tail extension.
    er, ec = info["elbow"]
    dr, dc = info["direction"]
    if (r == er and (ec - c) * dc >= 2) or (c == ec and (er - r) * dr >= 2):
        return None
    b = a.copy()
    b[point] = color
    return b


def _base(rng, lengths, colors, bg, source, teaching=False):
    lengths, colors = list(lengths), list(colors)
    # Gaps do not encode period, including repeated colors in separate runs.
    gaps = [int(rng.integers(1, 4))]
    gaps += [int(rng.integers(int(colors[i] == colors[i - 1]), 3))
             for i in range(1, len(lengths))]
    margin = int(rng.integers(0, 4))
    c = sum(lengths) + sum(gaps) + 1 + margin
    if c > 21:
        gaps = [1] + [int(colors[i] == colors[i - 1]) for i in range(1, len(colors))]
        margin = 0
        c = sum(lengths) + sum(gaps) + 1
    if c > 24:
        raise ValueError("program is too wide for this sampled canvas")
    h = int(rng.integers(22, 31) if teaching else rng.integers(10, 31))
    w = int(rng.integers(max(c + 7, 18 if teaching else 10), 31))
    r = int(rng.integers(1, min(6, h - 6)))
    a = np.full((h, w), bg, dtype=np.int64)
    a[r, c] = a[r - 1, c] = a[r, c - 1] = source
    cursor = c - 2
    for length, color, gap in zip(lengths, colors, gaps):
        cursor -= gap
        a[r, cursor - length + 1:cursor + 1] = color
        cursor -= length
    return a


def _extend(rng, a, want="any", axis_preference=None):
    _, before = _trace(a)
    records = before["records"]
    active = [r for r in records if r["active"]]
    last_step = active[-1]["step"] if active else 0
    options = []
    for record in records:
        if record["step"] < last_step + 2 or record["color"] is None:
            continue
        if want == "collision" and len(record["firing"]) < 2:
            continue
        if want == "solo" and len(record["firing"]) != 1:
            continue
        if want == "asymmetric" and (len(record["firing"]) != 1
                or all((2 * (record["step"] - 1)) % b["length"] == 0 for b in before["program"])):
            continue
        for axis in ((axis_preference,) if axis_preference is not None else (0, 1)):
            options.append((record, axis))
    rng.shuffle(options)
    if axis_preference is not None:
        options.sort(key=lambda x: x[1] != axis_preference)
    prefix = [(r["mirror"], r["position"]) for r in active]
    for record, axis in options:
        point = list(record["position"])
        point[axis] += record["direction"][axis]
        b = _place_point(a, tuple(point), record["color"])
        if b is None:
            continue
        try:
            _, after = _trace(b)
        except ValueError:
            continue
        turns = [r for r in after["records"] if r["active"]]
        sequence = [(r["mirror"], r["position"]) for r in turns]
        if sequence != prefix + [(tuple(point), record["position"])]:
            continue
        if len(after["records"]) - record["step"] < max(before["program"][0]["length"], 3):
            continue
        if {r["mirror"] for r in after["contacts"]} != set(after["mirrors"]):
            continue
        return b
    return None


def _add_probe(rng, a, kind="any"):
    _, before = _trace(a)
    program_colors = sorted({b["color"] for b in before["program"]})
    options = []
    for record in before["records"]:
        if record["mirror"] is not None:
            continue
        if kind == "blank" and record["color"] is not None:
            continue
        if kind == "wrong" and record["color"] is None:
            continue
        for color in program_colors:
            if color != record["color"]:
                for axis in (0, 1):
                    options.append((record, axis, color))
    rng.shuffle(options)
    route_key = [(r["position"], r["color"], r["direction"]) for r in before["records"]]
    for record, axis, color in options:
        point = list(record["position"])
        point[axis] += record["direction"][axis]
        b = _place_point(a, tuple(point), color)
        if b is None:
            continue
        try:
            _, after = _trace(b)
        except ValueError:
            continue
        if [(r["position"], r["color"], r["direction"]) for r in after["records"]] != route_key:
            continue
        if {r["mirror"] for r in after["contacts"]} != set(after["mirrors"]):
            continue
        return b
    return None


def _transform(a, turn, flip):
    return np.rot90(a[:, ::-1] if flip else a, turn).copy()


def _fallback(rng, lengths, colors, bg, source):
    """Analytic monotone-row construction for the two teaching witnesses."""
    lo, hi = sorted(lengths)
    gap = 1
    margin = int(rng.integers(0, 3))
    first = lo + 1
    second = math.lcm(lo, hi) + 1
    c, r = max(sum(lengths) + 3, second - 2 * first + 1) + margin, 1
    a = np.full((30, int(rng.integers(max(24, c + lo + 3), 31))), bg, dtype=np.int64)
    a[r, c] = a[r - 1, c] = a[r, c - 1] = source
    cursor = c - 3
    for length, color in zip(lengths, colors):
        a[r, cursor - length + 1:cursor + 1] = color
        cursor -= length + gap
    # Two horizontal-component reversals; row always increases.
    a[r + first, c + first + 1] = colors[lengths.index(lo)]
    a[r + second, c + 2 * first - second - 1] = colors[0]
    _trace(a)
    # A noncollision turn on the other component distinguishes travel time
    # from distance to the launcher even in the minimal two-demo curriculum.
    b = _extend(rng, a, "asymmetric", axis_preference=0)
    if b is None:
        raise RuntimeError("analytic teaching construction could not turn the other axis")
    a = b
    # Put both bypass witnesses early, so clipping cannot hide the different
    # path an unconditional reflector would take. They are two cells apart.
    probes = [((r + 1, c + 2), colors[1]),
              ((r + 3, c + 2), int(rng.choice(colors)))]
    for point, shade in probes:
        b = _place_point(a, point, shade)
        if b is None:
            raise RuntimeError("analytic teaching construction could not add a probe")
        a = b
    return a


def _scene(rng, lengths, colors, bg, source, teaching=False):
    for _ in range(40 if teaching else 12):
        a = _base(rng, lengths, colors, bg, source, teaching)
        if teaching:
            # A solo-colored turn exposes clock phase; a later collision turn
            # makes the priority choice affect geometry, not only ink color.
            b = _extend(rng, a, "solo", axis_preference=1)
            if b is None:
                continue
            a = b
            b = _extend(rng, a, "collision")
            if b is None:
                continue
            a = b
            if np.array_equal(_trace(a)[0], _trace(a, coordinate_clock=True, strict=False)[0]):
                b = _extend(rng, a, "asymmetric", axis_preference=0)
                if b is None:
                    continue
                a = b
        else:
            requested = int(rng.integers(0, 5))
            for _ in range(requested):
                b = _extend(rng, a)
                if b is None:
                    break
                a = b
        probes = ["blank", "wrong"] if teaching else ["any"] * int(rng.integers(0, 4))
        success = True
        for kind in probes:
            b = _add_probe(rng, a, kind)
            if b is None:
                success = not teaching
                break
            a = b
        if not success:
            continue
        out, info = _trace(a)
        if teaching:
            if len(info["records"]) < math.lcm(*lengths) + 4:
                continue
            if any(sum(bool(x["firing"]) and x["firing"][0] == j
                       for x in info["records"]) < 2 for j in range(2)):
                continue
            if np.array_equal(out, _trace(a, reset=True, strict=False)[0]):
                continue
            if np.array_equal(out, _trace(a, coordinate_clock=True, strict=False)[0]):
                continue
            if any(np.array_equal(out, _trace(a, timing=timing, reflection=reflection,
                                             strict=False)[0])
                   for timing, reflection in (("gap_phase", "match"), ("length", "all"))):
                continue
        if np.array_equal(a, out):
            continue
        return a
    if teaching:
        return _fallback(rng, lengths, colors, bg, source)
    # A straight clock is also a valid member, with no unnecessary routing.
    return _base(rng, lengths, colors, bg, source)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _palette(rng, protected=()):
    allowed = [c for c in range(10) if c not in protected]
    bg, source = map(int, rng.choice(allowed, 2, replace=False))
    return bg, source, [c for c in range(10) if c not in (bg, source)]


@lru_cache(maxsize=32)
def _visible_programs(count, maximum):
    """Arithmetic compositions whose every instruction wins within 12 beats."""
    options = []
    for lengths in itertools.product(range(1, maximum + 1), repeat=count):
        if sum(lengths) + count > 21:
            continue
        winners = set()
        for t in range(12):
            firing = [i for i, length in enumerate(lengths) if t % length == 0]
            if firing:
                winners.add(firing[0])
        if len(winners) == count:
            options.append(lengths)
    return tuple(options)


def _distinct_fallback(rng, split, shift, turn, flip, seen):
    """A clear first-beat turn, with more candidate shapes than legal pairs."""
    bg, source, palette = _palette(rng)
    # In the count shift, all three instructions win within the first three
    # ticks. Otherwise a single period-2 clock retains both required concepts.
    lengths = [3, 2, 1] if split == "test" and shift == 0 else [2]
    colors = list(map(int, rng.choice(palette, len(lengths), replace=False)))
    c, r = sum(lengths) + 2, 1
    heights = list(range(12, 31))
    widths = list(range(c + 3, 31))
    rng.shuffle(heights)
    rng.shuffle(widths)
    for height in heights:
        for width in widths:
            a = np.full((height, width), bg, dtype=np.int64)
            a[r, c] = a[r - 1, c] = a[r, c - 1] = source
            cursor = c - 3
            for length, color in zip(lengths, colors):
                a[r, cursor - length + 1:cursor + 1] = color
                cursor -= length
            a[r + 1, c + 2] = colors[0]
            a = _transform(a, turn, flip)
            if _key(a) not in seen:
                return {"input": a, "output": solve(a)}
    raise AssertionError("fewer than 15 inputs cannot occupy hundreds of distinct fallback shapes")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    periods = _WITNESS_PERIODS[int(rng.integers(len(_WITNESS_PERIODS)))]
    witness_colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    base_turn, flip = int(rng.integers(4)), bool(rng.integers(2))
    # Either extrapolate the number of clock layers or reserve two whole-scene
    # orientations for tests. These sampling regimes never alter semantics.
    shift = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(ntrain + ntest):
        split = "train" if index < ntrain else "test"
        teaching = index < 2
        for attempt in range(100):
            if teaching:
                lengths = list(periods if index == 0 else periods[::-1])
                colors = list(witness_colors if index == 0 else witness_colors[::-1])
                bg, source, _ = _palette(rng, witness_colors)
            else:
                bg, source, palette = _palette(rng)
                max_period = max(periods)
                if split == "test" and shift == 0:
                    counts = [n for n in (3, 4) if _visible_programs(n, max_period)]
                    count = int(rng.choice(counts))
                    options = _visible_programs(count, max_period)
                    lengths = list(options[int(rng.integers(len(options)))])
                    colors = list(map(int, rng.choice(palette, count, replace=False)))
                elif split == "train" and shift == 0:
                    count = int(rng.integers(1, 3))
                else:
                    count = int(rng.integers(1, 5))
                if not (split == "test" and shift == 0):
                    lengths = list(map(int, rng.integers(1, max_period + 1, count)))
                    while sum(lengths) + count > 21:
                        lengths[int(rng.integers(count))] = 1
                    colors = list(map(int, rng.choice(palette, count, replace=True)))
            a = _scene(rng, lengths, colors, bg, source, teaching)
            if teaching:
                turn = (base_turn + index) % 4
                local_flip = flip
            elif shift == 1:
                if split == "train":
                    turn = (base_turn + int(rng.integers(2))) % 4
                else:
                    turn = (base_turn + 2 + (index - ntrain) % 2) % 4
                local_flip = flip
            else:
                turn, local_flip = int(rng.integers(4)), bool(rng.integers(2))
            a = _transform(a, turn, local_flip)
            out = solve(a)
            if split == "test" and any(np.array_equal(out, _rival(a, rival))
                                        for rival in ("none", "unit")):
                continue
            if split == "test" and shift == 0:
                info = _trace(a)[1]
                if len({r["firing"][0] for r in info["records"] if r["firing"]}) != len(info["program"]):
                    continue
            if _key(a) not in seen:
                seen.add(_key(a))
                puzzle[split].append({"input": a, "output": out})
                break
        else:
            pair = _distinct_fallback(rng, split, shift, turn, local_flip, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _rival(a, name):
    kwargs = {"strict": False}
    if name in ("far", "short", "long"):
        kwargs["priority"] = name
    elif name in ("plus_one", "minus_one", "unit", "gap_phase"):
        kwargs["timing"] = name
    elif name == "reset":
        kwargs["reset"] = True
    elif name == "radius":
        kwargs["coordinate_clock"] = True
    else:
        kwargs["reflection"] = name
    return _trace(a, **kwargs)[0]


def _teaching_checks(pairs):
    infos = [_trace(_grid(p["input"]))[1] for p in pairs]
    contacts = [r for i in infos for r in i["contacts"]]
    assert {r["axis"] for r in contacts if r["active"]} == {0, 1}, "teach both reflection components"
    assert any(r["active"] and len(r["firing"]) > 1 and r["step"] > 1 for r in contacts), "priority must steer a collision"
    assert any(r["active"] and len(r["firing"]) == 1 for r in contacts), "teach a turn away from the common phase"
    assert any(not r["active"] and r["color"] is None for r in contacts), "teach empty-beat bypass"
    assert any(not r["active"] and r["color"] is not None for r in contacts), "teach wrong-color bypass"
    two_run = [i for i in infos if len(i["program"]) == 2
               and i["program"][0]["color"] != i["program"][1]["color"]
               and min(b["length"] for b in i["program"]) > 1
               and max(b["length"] for b in i["program"]) % min(b["length"] for b in i["program"])
               and len(i["records"]) >= math.lcm(*(b["length"] for b in i["program"])) + 4
               and all(sum(bool(r["firing"]) and r["firing"][0] == j
                           for r in i["records"]) >= 2 for j in range(2))]
    assert any(i["program"][0]["length"] < i["program"][1]["length"] for i in two_run)
    assert any(i["program"][0]["length"] > i["program"][1]["length"] for i in two_run)
    order_edges = {(i["program"][0]["color"], i["program"][1]["color"]) for i in two_run}
    assert any((b, a) in order_edges for a, b in order_edges), "refute a fixed color ranking"
    assert any(len({r["axis"] for r in i["contacts"] if r["active"]}) == 2 for i in infos), "teach turns on both axes within one route"
    for rival in ("far", "short", "long", "plus_one", "minus_one", "unit", "gap_phase", "none", "all", "reset", "radius"):
        assert any(not np.array_equal(_grid(p["output"]), _rival(_grid(p["input"]), rival)) for p in pairs), f"rival {rival} fits all demonstrations"
    # Each witness period must have a visible, uncontested recurrence beyond
    # its shared first beat, rather than existing only as a hidden instruction.
    for info in two_run:
        for j, block in enumerate(info["program"]):
            visible = [r for r in info["records"] if r["firing"] and r["firing"][0] == j]
            assert len(visible) >= 2, "need repeated visible evidence for both periods"


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, split_infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            expected, info = _trace(a)
            split_infos[split].append(info)
            assert np.array_equal(out, expected), "wrong timing, resonance, reflection, or preservation"
            assert np.array_equal(out[a != info["background"]], a[a != info["background"]])
            assert set(np.unique(out)) <= set(np.unique(a))
            assert not np.array_equal(a, out)
            if split == "test":
                for ablation in ("none", "unit"):
                    assert not np.array_equal(out, _rival(a, ablation)), "each test must need both inherited mechanisms"
            assert {r["mirror"] for r in info["contacts"]} == set(info["mirrors"]), "every mirror must be encountered geometrically"
            for record in info["records"]:
                p = record["position"]
                if record["color"] is None:
                    assert out[p] == info["background"]
                else:
                    assert out[p] == record["color"]
            # No carrier cell can have two incompatible clock times, and the
            # endpoint really leaves the canvas after the last directed step.
            assert len({r["position"] for r in info["records"]}) == len(info["records"])
            last = info["records"][-1]
            direction = list(last["direction"])
            if last["active"]:
                direction[last["axis"]] *= -1
            assert not _inside(tuple(last["position"][i] + direction[i] for i in (0, 1)), a.shape)
    _teaching_checks(puzzle["train"])
    # The shift is observable in the returned grids, not a sampled intention.
    train_infos, test_infos = split_infos["train"], split_infos["test"]
    count_shift = (max(len(i["program"]) for i in train_infos) <= 2
                   and min(len(i["program"]) for i in test_infos) >= 3)
    direction_shift = {i["direction"] for i in train_infos}.isdisjoint(
        {i["direction"] for i in test_infos})
    assert count_shift or direction_shift, "missing visible train/test shift"
    if count_shift and not direction_shift:
        for info in test_infos:
            assert len({b["color"] for b in info["program"]}) == len(info["program"])
            assert {r["color"] for r in info["records"] if r["color"] is not None} == {
                b["color"] for b in info["program"]}, "the larger test program must visibly contribute"
    assert max(b["length"] for i in test_infos for b in i["program"]) <= max(
        b["length"] for i in train_infos for b in i["program"]), "test periods exceed the taught numerical range"


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s]) for s in ("train", "test"))


def _diagnostics():
    # Hand-specified, fixed coordinates. Program nearest-first 2 red, 3 blue.
    a = np.zeros((14, 15), dtype=np.int64)
    a[1, 7] = a[0, 7] = a[1, 6] = 8
    a[1, 3:5] = 2
    a[1, 0:3] = 3
    a[4, 11] = 2             # step 3, red: turn left
    a[6, 7] = 3              # step 5, red: blue mirror is bypassed
    a[8, 5] = 2              # step 7, both fire: near red turns right
    expected = a.copy()
    pulses = {(2, 8): 2, (4, 10): 2, (5, 9): 3, (6, 8): 2,
              (8, 6): 2, (10, 8): 2, (11, 9): 3, (12, 10): 2}
    for p, color in pulses.items():
        expected[p] = color
    for turn in range(4):
        for flip in (False, True):
            assert np.array_equal(solve(_transform(a, turn, flip)), _transform(expected, turn, flip))
    # A third turn is on the other axis. The next blue beat is step 10 at
    # (9,9), although that cell's Chebyshev distance to the elbow is only 8.
    mixed = a.copy()
    mixed[11, 8] = 2
    mixed_expected = mixed.copy()
    mixed_pulses = {(2, 8): 2, (4, 10): 2, (5, 9): 3, (6, 8): 2,
                    (8, 6): 2, (10, 8): 2, (9, 9): 3, (8, 10): 2,
                    (6, 12): 2, (4, 14): 2}
    for p, color in mixed_pulses.items():
        mixed_expected[p] = color
    for turn in range(4):
        for flip in (False, True):
            assert np.array_equal(solve(_transform(mixed, turn, flip)),
                                  _transform(mixed_expected, turn, flip))
    assert not np.array_equal(_rival(mixed, "radius"), mixed_expected)
    assert not np.array_equal(_rival(mixed, "reset"), mixed_expected)
    # A sparse one-run period-7 clock, with a late reflected recurrence.
    long_period = np.zeros((12, 28), dtype=np.int64)
    long_period[1, 10] = long_period[0, 10] = long_period[1, 9] = 8
    long_period[1, 1:8] = 2
    long_period[10, 18] = 2
    long_expected = long_period.copy()
    for p in ((2, 11), (9, 18), (2, 25)):
        long_expected[p] = 2
    assert np.array_equal(solve(long_period), long_expected)
    # Move runs farther from the L without changing their lengths or order.
    wide = np.zeros((14, 19), dtype=np.int64)
    wide[:, 4:] = a
    wide[1, 4:9] = 0
    wide[1, 5:7] = 2
    wide[1, 1:4] = 3
    out = solve(wide)
    reference = np.zeros_like(wide)
    reference[:, 4:] = expected
    reference[1] = wide[1]
    assert np.array_equal(out, reference), "tail gaps must not shift the clock"


def self_test() -> dict:
    """Bounded regressions and measured structural variation over 42 seeds."""
    _diagnostics()
    for shift in (0, 1):
        seen = set()
        rng = np.random.default_rng(801 + shift)
        for index in range(14):
            pair = _distinct_fallback(rng, "test", shift, index % 4, bool(index % 2), seen)
            seen.add(_key(pair["input"]))
            out, info = _trace(pair["input"])
            for ablation in ("none", "unit"):
                assert not np.array_equal(out, _rival(pair["input"], ablation))
            if shift == 0:
                assert len({r["color"] for r in info["records"] if r["color"] is not None}) == 3
    # Exercise the guaranteed fallback even when random generation never
    # needs it. A solo pulse can still be a temporal symmetry center, so
    # explicitly check the full distance-clock contradiction after the turn.
    fallback_scenes = 0
    for periods in _WITNESS_PERIODS:
        for trial in range(4):
            rng = np.random.default_rng(trial)
            demos = []
            for reverse in (False, True):
                lengths = list(periods[::-1] if reverse else periods)
                colors = [3, 2] if reverse else [2, 3]
                a = _fallback(rng, lengths, colors, 0, 8)
                out, info = _trace(a)
                assert {r["mirror"] for r in info["contacts"]} == set(info["mirrors"])
                assert not np.array_equal(out, _rival(a, "radius"))
                demos.append({"input": a, "output": out})
                fallback_scenes += 1
            _teaching_checks(demos)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, shapes, normalized, backgrounds, directions = [], set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                _, info = _trace(a)
                records.append((split, info))
                shapes.add(a.shape)
                backgrounds.add(info["background"])
                directions.add(info["direction"])
                roles = {info["background"]: 0}
                normalized.add((a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 35 and len(normalized) >= 150
    assert backgrounds == set(range(10))
    assert directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert {len(i["program"]) for _, i in records} == {1, 2, 3, 4}
    assert len({tuple(b["length"] for b in i["program"]) for _, i in records}) >= 35
    assert any(not i["mirrors"] for _, i in records)
    assert any(sum(r["active"] for r in i["contacts"]) >= 3 for _, i in records)
    assert any(len({b["color"] for b in i["program"]}) < len(i["program"]) for _, i in records)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for seed in range(42):
        validate_generated_puzzle(generate_puzzle(seed, 2, 1))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            validate_generated_puzzle(generate_puzzle(73 + ntrain * 4 + ntest, ntrain, ntest))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
    pair = bad["test"][0]
    changed = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][changed] = pair["input"][changed]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased pulse was accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len(shapes),
            "normalized_structures": len(normalized),
            "analytic_fallback_scenes": fallback_scenes,
            "run_count_range": [1, 4],
            "programs": len({tuple(b["length"] for b in i["program"]) for _, i in records}),
            "active_turn_counts": sorted({sum(r["active"] for r in i["contacts"]) for _, i in records}),
            "semantic_checks": "author-owned; no independent solver or blind model"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
