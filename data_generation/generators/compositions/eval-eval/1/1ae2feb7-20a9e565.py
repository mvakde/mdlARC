"""A loom of growing clocks: length controls timing, beat count controls shape.

Labels use explicit stroke painting. The input-only solver recovers endpoint
velocities from two exposed beats and checks the entire observed prefix using
cell predicates. No files, hidden parameters, or parent modules are used.
"""
from __future__ import annotations

import hashlib
import numpy as np


_STYLES = ("solid", "ends", "comb2", "comb3")
_WITNESS_PERIODS = ((2, 3), (2, 5), (3, 4))


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(name + " must be an integer in the supported range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= s <= 30 for s in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _turn_bounds(bounds, shape, turns):
    h, w = shape
    r0, r1, c0, c1 = bounds
    points = []
    for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        for _ in range(turns % 4):
            r, c, h, w = w - 1 - c, r, w, h
        points.append((r, c))
        h, w = shape
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _marker(grid):
    found = []
    for raw in np.unique(grid):
        pos = np.argwhere(grid == raw)
        if len(pos) != 6:
            continue
        r0, c0 = map(int, pos.min(axis=0))
        r1, c1 = map(int, pos.max(axis=0))
        if r1 <= r0 or c1 <= c0 or (r1 - r0 + 1) + (c1 - c0 + 1) < 6:
            continue
        bounds = r0, r1, c0, c1
        if set(map(tuple, pos)) == _corners(bounds):
            found.append((int(raw), bounds))
    if len(found) != 1:
        raise ValueError("expected two unique disconnected canonical L markers")
    return found[0]


def _occupied(row, top, bottom, style):
    if not top <= row <= bottom:
        return False
    if style == "solid":
        return True
    if style == "ends":
        return row == top or row == bottom
    return row == bottom or (row - top) % int(style[-1]) == 0


def _cell(info, row, t, variant="rule"):
    blocks = info["blocks"]
    if variant == "far":
        blocks = blocks[::-1]
    elif variant in ("short", "long"):
        blocks = sorted(blocks, key=lambda b: b[0], reverse=variant == "long")
    if variant == "crop_phase":
        t -= info["bounds"][2] - info["sep"] - 1
    for length, color, gap in blocks:
        period = 1 if variant == "no_length" else length
        age_t = t - gap if variant == "gap_phase" else t
        if age_t < 0 or age_t % period:
            continue
        n = age_t // period
        if variant == "global_age":
            n = t
        u, v = (0, 0) if variant == "frozen" else (info["u"], info["v"])
        top, bottom = info["top"] + u * n, info["bottom"] + v * n
        if _occupied(row, top, bottom, info["style"]):
            return color
    return info["background"]


def _crop(info, variant="rule"):
    r0, r1, c0, c1 = info["bounds"]
    a = np.full((r1 - r0 + 1, c1 - c0 + 1), info["background"], dtype=np.int8)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            a[r - r0, c - c0] = _cell(info, r, c - info["sep"] - 1, variant)
    return np.rot90(a, -info["rotation"]).copy()


def _analyze(input_grid):
    grid = _grid(input_grid)
    annotation, bounds = _marker(grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if bg == annotation or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    clean = grid.copy()
    clean[clean == annotation] = bg
    if not np.all(clean[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1] == bg):
        raise ValueError("the entire requested rectangle must be erased")
    candidates = []
    for rotation in range(4):
        a = np.rot90(clean, rotation)
        bb = _turn_bounds(bounds, clean.shape, rotation)
        h, w = a.shape
        for sep in range(1, w - 2):
            sc = int(a[0, sep])
            if (sc == bg or not np.all(a[:, sep] == sc) or
                    np.count_nonzero(a == sc) != h or bb[2] <= sep):
                continue
            rows = np.flatnonzero(np.any(a[:, :sep] != bg, axis=1))
            if len(rows) != 1:
                continue
            source_row = int(rows[0])
            blocks, cursor = [], sep - 1
            while cursor >= 0:
                gap = 0
                while cursor >= 0 and a[source_row, cursor] == bg:
                    cursor -= 1
                    gap += 1
                if cursor < 0:
                    break
                color, length = int(a[source_row, cursor]), 0
                while cursor >= 0 and a[source_row, cursor] == color:
                    cursor -= 1
                    length += 1
                blocks.append((length, color, gap))
            if not 1 <= len(blocks) <= 3 or len({b[1] for b in blocks}) != len(blocks):
                continue
            field = a[:, sep + 1:]
            occupied_cols = np.flatnonzero(np.any(field != bg, axis=0))
            if not len(occupied_cols) or occupied_cols[0] != 0:
                continue
            tobs, fastest = int(occupied_cols[-1]), min(b[0] for b in blocks)
            if tobs < 2 * fastest or sep + 1 + tobs >= bb[2]:
                continue
            first = np.flatnonzero(field[:, 0] != bg)
            next_beat = np.flatnonzero(field[:, fastest] != bg)
            if min(len(first), len(next_beat)) < 2:
                continue
            if (np.any(field[first, 0] != blocks[0][1]) or
                    np.any(field[next_beat, fastest] != next(b[1] for b in blocks if b[0] == fastest))):
                continue
            top, bottom = int(first[0]), int(first[-1])
            u, v = int(next_beat[0]) - top, int(next_beat[-1]) - bottom
            for style in _STYLES:
                info = dict(rotation=rotation, canonical=a, bounds=bb, background=bg,
                            marker=annotation, sep=sep, separator=sc, blocks=blocks,
                            source_row=source_row, top=top, bottom=bottom,
                            u=u, v=v, style=style, observed=tobs)
                valid = True
                # Full observed teeth, including occluded contributors, fit inside
                # the canvas. Requested teeth may extend past its edges.
                for n in range(tobs // fastest + 1):
                    lo, hi = top + u * n, bottom + v * n
                    if not 0 <= lo < hi < h:
                        valid = False
                        break
                if not valid:
                    continue
                for t in range(tobs + 1):
                    if any(int(field[r, t]) != _cell(info, r, t) for r in range(h)):
                        valid = False
                        break
                last_age = (bb[3] - sep - 1) // fastest
                if any(bottom + v * n <= top + u * n for n in range(last_age + 1)):
                    valid = False
                if valid:
                    candidates.append(info)
    if not candidates:
        raise ValueError("no consistent growing-clock interpretation")
    answer = _crop(candidates[0])
    if any(not np.array_equal(answer, _crop(i)) for i in candidates[1:]):
        raise ValueError("visible teeth do not uniquely determine the requested crop")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive timing, geometry, priority, and inclusive crop from this input."""
    return _crop(_analyze(input_grid))


def _paint_tooth(canvas, col, top, bottom, color, style):
    """Construction path: explicit endpoint/stroke painting, no cell predictor."""
    if style == "solid":
        canvas[top:bottom + 1, col] = color
    elif style == "ends":
        canvas[top, col] = color
        canvas[bottom, col] = color
    else:
        canvas[top:bottom + 1:int(style[-1]), col] = color
        canvas[bottom, col] = color


def _make_pair(rng, periods, colors, roles, rotation, witness, partial, skip):
    bg, separator, marker = roles
    shortest = min(periods)
    obs = 3 * shortest + int(rng.integers(shortest))
    if witness:
        collision = (obs // int(np.lcm(*periods)) + 1) * int(np.lcm(*periods))
        start = max(obs + 1, collision - int(rng.integers(1, 3)))
        end = collision + max(periods) + int(rng.integers(2))
    else:
        start = obs + 1 + skip
        end = start + int(rng.integers(4, 9)) - 1
    budget = 28 - end
    if budget < sum(periods) + int(witness):
        return None
    gap_budget = min(5, budget - sum(periods)) - int(witness)
    gaps = list(map(int, rng.multinomial(int(rng.integers(gap_budget + 1)),
                                        [1 / len(periods)] * len(periods))))
    if witness:
        gaps[0] += 1
    instruction = sum(periods) + sum(gaps)
    instruction += int(rng.integers(min(3, budget - instruction) + 1))
    h0 = int(rng.integers(5, 12))
    u, v = map(int, rng.integers(-2, 3, size=2))
    if u == v == 0 or abs(v - u) > 2:
        return None
    if witness and v <= u:
        return None
    style = str(rng.choice(_STYLES))
    max_age = end // shortest
    tops = [u * n for n in range(max_age + 1)]
    bottoms = [h0 - 1 + v * n for n in range(max_age + 1)]
    if any(b - t < 1 for t, b in zip(tops, bottoms)):
        return None
    pulse_times = [t for t in range(start, end + 1) if any(t % p == 0 for p in periods)]
    if not pulse_times:
        return None
    target_tops = [u * (t // p) for p in periods for t in pulse_times if t % p == 0]
    target_bottoms = [h0 - 1 + v * (t // p) for p in periods for t in pulse_times if t % p == 0]
    low, high = min(target_tops), max(target_bottoms)
    if partial:
        if high - low < 2:
            return None
        crop_height = int(rng.integers(2, min(high - low, 9) + 1))
        r0 = int(rng.integers(low, high - crop_height + 2))
        r1 = r0 + crop_height - 1
    else:
        r0, r1 = low - int(rng.integers(2)), high + int(rng.integers(2))
    ymin, ymax = min(min(tops), r0), max(max(bottoms), r1)
    extent = ymax - ymin + 1
    if extent > 28:
        return None
    upper = int(rng.integers(1, min(3, 29 - extent) + 1))
    lower = int(rng.integers(1, min(3, 30 - extent - upper) + 1))
    height = extent + upper + lower
    dy = upper - ymin
    r0, r1 = r0 + dy, r1 + dy
    width = instruction + end + 2
    width += int(rng.integers(min(2, 30 - width) + 1))
    full = np.full((height, width), bg, dtype=np.int8)
    # Each emitter's n-th tooth is a separate growing object. Painting the
    # farthest emitter first gives occupied-cell proximity precedence.
    for p, color in reversed(list(zip(periods, colors))):
        for n, t in enumerate(range(0, end + 1, p)):
            _paint_tooth(full, instruction + 1 + t, dy + u * n,
                         dy + h0 - 1 + v * n, color, style)
    bounds = r0, r1, instruction + 1 + start, instruction + 1 + end
    out = full[r0:r1 + 1, bounds[2]:bounds[3] + 1].copy()
    inp = np.full_like(full, bg)
    inp[:, instruction + 1:instruction + 2 + obs] = full[:, instruction + 1:instruction + 2 + obs]
    inp[:, instruction] = separator
    source_row = int(rng.integers(1, height - 1))
    cursor = instruction
    for p, color, gap in zip(periods, colors, gaps):
        cursor -= gap
        inp[source_row, cursor - p:cursor] = color
        cursor -= p
    rotated_bounds = _turn_bounds(bounds, inp.shape, rotation)
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    for r, c in _corners(rotated_bounds):
        inp[r, c] = marker
    try:
        info = _analyze(inp)
    except ValueError:
        return None
    if not np.array_equal(_crop(info), out) or not np.any(out != bg):
        return None
    if np.array_equal(_crop(info, "frozen"), out) or np.array_equal(_crop(info, "no_length"), out):
        return None
    if witness:
        # An actually visible overlap in the requested answer, plus both colors,
        # makes priority observable rather than merely present in latent strokes.
        if np.array_equal(_crop(info, "far"), out) or not set(colors) <= set(map(int, np.unique(out))):
            return None
        if any(np.array_equal(_crop(info, rival), out) for rival in
               ("gap_phase", "crop_phase", "global_age")):
            return None
    return {"input": inp, "output": out}


def _fallback_pair(roles, payload, period_pair, rotation, witness_index, partial, skip, three, variant):
    """32 distinct constructive members; same rule, shifts, and witness roles.

    Four initial heights times eight source rows provide distinct inputs even
    with a stable palette. This exceeds the maximum 13 preceding inputs.
    """
    bg, separator, marker = roles
    if witness_index is not None:
        periods = list(period_pair if witness_index == 0 else period_pair[::-1])
        colors = payload[:2] if witness_index == 0 else payload[1::-1]
    else:
        periods = [4, 3, 2] if three else [2]
        cycle = payload[variant % len(payload):] + payload[:variant % len(payload)]
        colors = cycle[:len(periods)]
    obs = 3 * min(periods)
    if witness_index is not None:
        collision = (obs // int(np.lcm(*periods)) + 1) * int(np.lcm(*periods))
        start, end = collision - 1, collision + max(periods)
    else:
        start = obs + 1 + skip
        end = start + 5
    h0 = 5 + variant // 8
    last_age = end // min(periods)
    height, instruction = h0 + last_age + 2, sum(periods) + 1
    full = np.full((height, instruction + end + 2), bg, dtype=np.int8)
    for p, color in reversed(list(zip(periods, colors))):
        for n, t in enumerate(range(0, end + 1, p)):
            full[1:h0 + n + 1, instruction + 1 + t] = color
    r0, r1 = (h0 + 1, h0 + 3) if partial else (1, h0 + last_age)
    bounds = r0, r1, instruction + 1 + start, instruction + 1 + end
    out = full[r0:r1 + 1, bounds[2]:bounds[3] + 1].copy()
    inp = np.full_like(full, bg)
    inp[:, instruction + 1:instruction + obs + 2] = full[:, instruction + 1:instruction + obs + 2]
    inp[:, instruction] = separator
    cursor = instruction - 1
    for p, color in zip(periods, colors):
        inp[1 + variant % 8, cursor - p:cursor] = color
        cursor -= p
    bb = _turn_bounds(bounds, inp.shape, rotation)
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    for r, c in _corners(bb):
        inp[r, c] = marker
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    roles, payload = tuple(palette[:3]), palette[3:]
    period_pair = _WITNESS_PERIODS[int(rng.integers(len(_WITNESS_PERIODS)))]
    base_turn = int(rng.integers(4))
    shift = str(rng.choice(("turn", "distance", "slice", "clocks")))
    direction_mode = int(rng.integers(3))
    varying_roles = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    max_train_gap = 0
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index < 2
            rotation = base_turn
            if shift == "turn" and split == "test":
                rotation = (base_turn + 1 + 2 * int(rng.integers(2))) % 4
            elif shift != "turn" and direction_mode == 1:
                rotation = (base_turn + 2 * int(rng.integers(2))) % 4
            elif shift != "turn" and direction_mode == 2:
                rotation = int(rng.integers(4))
            for attempt in range(240):
                if attempt >= 208:
                    pair = _fallback_pair(roles, payload, period_pair, rotation,
                                          index if witness else None,
                                          split == "test" and shift == "slice",
                                          max_train_gap + 2 if split == "test" and shift == "distance" else 0,
                                          split == "test" and shift == "clocks", attempt - 208)
                else:
                    if witness:
                        periods = list(period_pair if index == 0 else period_pair[::-1])
                        colors = payload[:2] if index == 0 else payload[1::-1]
                        pair_roles = roles
                    else:
                        n = 3 if split == "test" and shift == "clocks" else int(rng.integers(1, 3 if shift == "clocks" else 4))
                        periods = list(map(int, rng.integers(1, 7, size=n)))
                        if sum(periods) > 12:
                            continue
                        chosen = list(map(int, rng.permutation(10))) if varying_roles else list(palette)
                        pair_roles, colors = tuple(chosen[:3]), chosen[3:3 + n]
                    partial = split == "test" and shift == "slice"
                    if shift != "slice" and not witness:
                        partial = bool(rng.integers(2))
                    skip = (max_train_gap + int(rng.integers(2, 5)) if split == "test" and shift == "distance"
                            else int(rng.integers(3)))
                    pair = _make_pair(rng, periods, colors, pair_roles, rotation, witness, partial, skip)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                info = _analyze(pair["input"])
                if split == "test" and shift == "clocks" and not {b[1] for b in info["blocks"]} <= set(map(int, np.unique(pair["output"]))):
                    continue
                actual_gap = info["bounds"][2] - info["sep"] - 2 - info["observed"]
                if split == "test" and shift == "distance" and actual_gap <= max_train_gap:
                    continue
                if split == "train":
                    max_train_gap = max(max_train_gap, actual_gap)
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded growing-clock construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(_crop(info), out), "wrong timing, geometry, priority, or crop"
            assert info["marker"] not in out and info["separator"] not in out
            assert set(map(int, np.unique(out))) <= {info["background"]} | {b[1] for b in info["blocks"]}
            assert np.any(out != info["background"])
            assert not np.array_equal(_crop(info, "frozen"), out), "growth/drift must affect the answer"
            assert not np.array_equal(_crop(info, "no_length"), out), "run lengths must affect the answer"
            if split == "train":
                train_infos.append(info)
    # These are observations of returned crops, not flags sampled at generation.
    assert any(len(i["blocks"]) > 1 and i["blocks"][0][0] < i["blocks"][1][0] and
               not np.array_equal(_crop(i), _crop(i, "far")) for i in train_infos)
    assert any(len(i["blocks"]) > 1 and i["blocks"][0][0] > i["blocks"][1][0] and
               not np.array_equal(_crop(i), _crop(i, "far")) for i in train_infos)
    edges = set()
    for i in train_infos:
        r0, r1, c0, c1 = i["bounds"]
        for c in range(c0, c1 + 1):
            t = c - i["sep"] - 1
            for r in range(r0, r1 + 1):
                hits = [(p, color) for p, color, gap in i["blocks"] if t % p == 0 and
                        _occupied(r, i["top"] + i["u"] * (t // p),
                                  i["bottom"] + i["v"] * (t // p), i["style"])]
                if len(hits) > 1:
                    edges.update((hits[0][1], color) for p, color in hits[1:])
    assert any((b, a) in edges for a, b in edges), "a fixed color ranking fits the training collisions"
    for rival in ("short", "long", "far", "global_age", "crop_phase", "gap_phase"):
        assert any(not np.array_equal(_crop(i), _crop(i, rival)) for i in train_infos), rival
    return True


def self_test():
    """Bounded, streaming coverage over 42 seeds; no persisted puzzle corpus."""
    # Hand-calculated teeth: periods 2/3, top 1, bottom 5+n. The
    # longer-period-near case leaves a shorter-clock tip visible at t=12.
    # These fixed coordinates/answers do not use the construction painter.
    for reverse in (False, True):
        fixture = np.zeros((15, 23), dtype=np.int8)
        fixture[:, 6] = 8
        fixture[13, :6] = list(map(int, "011222" if reverse else "022211"))
        for col, bottom, color in ((7, 5, 2 if reverse else 1), (9, 6, 1),
                                    (10, 6, 2), (11, 7, 1), (13, 8, 1), (15, 9, 1)):
            fixture[1:bottom + 1, col] = color
        if reverse:
            fixture[1:8, 13] = 2
        bounds = 0, 13, 16, 22
        rows = (["0000000"] + ["2102012"] * 8 + ["0102012", "0101012", "0001010", "0000010", "0000000"]
                if reverse else ["0000000"] + ["2101012"] * 8 + ["0101012"] * 2 + ["0001010", "0000010", "0000000"])
        expected = np.asarray([list(map(int, row)) for row in rows], dtype=np.int8)
        for turn in range(4):
            a = np.rot90(fixture, turn).copy()
            for r, c in _corners(_turn_bounds(bounds, fixture.shape, turn)):
                a[r, c] = 9
            assert np.array_equal(solve(a), np.rot90(expected, turn))
            # An arbitrary bijective palette change preserves geometric roles.
            mapping = np.asarray([4, 7, 0, 2, 9, 1, 3, 8, 6, 5], dtype=np.int8)
            assert np.array_equal(solve(mapping[a]), mapping[np.rot90(expected, turn)])
    # A growing tooth's upper endpoint leaves the canvas after the observation.
    # Its requested, in-frame slice must still be predicted rather than rejected.
    clipped = np.zeros((15, 23), dtype=np.int8)
    clipped[:, 6] = 8
    clipped[13, :6] = list(map(int, "022211"))
    for col, top, color in ((7, 4, 1), (9, 3, 1), (10, 3, 2),
                             (11, 2, 1), (13, 1, 1), (15, 0, 1)):
        clipped[top:9, col] = color
    for r, c in _corners((0, 3, 16, 22)):
        clipped[r, c] = 9
    assert np.array_equal(solve(clipped), np.asarray([list(map(int, row)) for row in
                                                     ("0101012", "2101012", "2101012", "2101012")]))
    # Exercise every constructive fallback member, including the largest
    # possible distance shift (training blank gap <=7, then skip <=9).
    fallback_modes = [(ps, ix, False, 0, False) for ps in _WITNESS_PERIODS for ix in (0, 1)]
    fallback_modes += [((2, 3), None, False, 9, False),
                       ((2, 3), None, True, 0, False),
                       ((2, 3), None, False, 0, True)]
    for ps, ix, partial, skip, three in fallback_modes:
        fallback_keys = set()
        for variant in range(32):
            p = _fallback_pair((0, 8, 9), [1, 2, 3, 4, 5, 6, 7], ps,
                               variant % 4, ix, partial, skip, three, variant)
            fallback_keys.add(_key(p["input"]))
            info = _analyze(p["input"])
            assert np.array_equal(_crop(info), p["output"])
            for rival in ("frozen", "no_length"):
                assert not np.array_equal(_crop(info, rival), p["output"])
            if three:
                assert {b[1] for b in info["blocks"]} <= set(map(int, np.unique(p["output"])))
        assert len(fallback_keys) == 32
    signatures, shapes, periods, styles, rotations, velocities, source_counts = set(), set(), set(), set(), set(), set(), set()
    protrusions = shrinking = drifting = gaps = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _analyze(pair["input"])
                for role in ("input", "output"):
                    a = pair[role]
                    digest.update(bytes(a.shape) + a.tobytes())
                    shapes.add((role, a.shape))
                periods.update(b[0] for b in i["blocks"])
                styles.add(i["style"])
                rotations.add(i["rotation"])
                velocities.add((i["u"], i["v"]))
                source_counts.add(len(i["blocks"]))
                shrinking += i["v"] < i["u"]
                drifting += i["u"] == i["v"] != 0
                gaps += any(b[2] for b in i["blocks"])
                # Nearer color and farther protruding color at the same beat.
                r0, r1, c0, c1 = i["bounds"]
                for c in range(c0, c1 + 1):
                    t = c - i["sep"] - 1
                    firing = [b for b in i["blocks"] if t % b[0] == 0]
                    actual = {_cell(i, r, t) for r in range(r0, r1 + 1)}
                    protrusions += len(firing) > 1 and len(actual - {i["background"]}) > 1
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 60
    assert styles == set(_STYLES) and rotations == set(range(4))
    assert periods == set(range(1, 7)) and source_counts == {1, 2, 3}
    assert len(velocities) >= 10 and shrinking > 0 and drifting > 0 and gaps > 30 and protrusions > 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(733, nt, ne))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(81, 2, 1), generate_puzzle(81, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    changed = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
    changed["test"][0]["output"][0, 0] = (int(changed["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "styles": sorted(styles), "periods": sorted(periods), "velocities": len(velocities),
            "shrinking_pairs": shrinking, "drifting_pairs": drifting,
            "gapped_pairs": gaps, "collision_columns_with_protrusions": protrusions}
