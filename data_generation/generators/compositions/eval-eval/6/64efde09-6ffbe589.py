"""Phase-coded mosaic repair; no parent imports, files, or latent solver state.

The unequal L key defines local handedness. Unary bars set per-window,
per-color phases. Undo those phases, union fragments, and repair in place.
Generation uses orbit territories only to avoid overlapping rotated layers;
the input-only solver does not assume any territory or motif construction.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _orient(a, t):
    """Reflect left/right first, then turn counterclockwise t modulo four."""
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _unorient(a, t):
    a = np.rot90(a, -(t % 4))
    return (np.fliplr(a) if t >= 4 else a).copy()


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _frame(s, bg, rim, key_color):
    a = np.full((s, s), bg, dtype=np.int8)
    a[0] = a[-1] = rim
    a[:, 0] = a[:, -1] = rim
    # Three cells along local top, two down local left, sharing the corner.
    for p in ((0, 0), (0, 1), (0, 2), (1, 0)):
        a[p] = key_color
    return a


def _separate(a, b):
    """Exclusive rectangles with at least one completely empty separating line."""
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _parse(value):
    """Read frames, correspondence, ownership, colors and counters from pixels."""
    a = _grid(value)
    h, w = a.shape
    values, frequencies = np.unique(a, return_counts=True)
    _require(np.sum(frequencies == frequencies.max()) == 1, "ambiguous background")
    bg = int(values[frequencies.argmax()])
    windows = []
    # The main rim is a connected monochromatic component even after its key
    # replaces four cells. Its full bounding box is the frame's bounding box.
    for rim in map(int, values):
        if rim == bg:
            continue
        for component in _components(a == rim):
            rs, cs = zip(*component)
            r, c, bottom, right = min(rs), min(cs), max(rs) + 1, max(cs) + 1
            s = bottom - r
            if s != right - c or not 6 <= s <= 10 or len(component) != 4 * s - 8:
                continue
            border = np.zeros((s, s), dtype=bool)
            border[0] = border[-1] = True
            border[:, 0] = border[:, -1] = True
            block = a[r:bottom, c:right]
            if not all(y in (r, bottom - 1) or x in (c, right - 1) for y, x in component):
                continue
            minor = block[border & (block != rim)]
            if len(minor) != 4 or len(set(map(int, minor))) != 1:
                continue
            key_color = int(minor[0])
            if key_color in (bg, rim) or np.any(np.isin(block[1:-1, 1:-1], [rim, key_color])):
                continue
            template = _frame(s, bg, rim, key_color)
            ts = [t for t in range(8) if np.array_equal(_orient(template, t)[border], block[border])]
            if len(ts) == 1:
                windows.append({"r": r, "c": c, "s": s, "t": ts[0],
                                "rim": rim, "key_color": key_color,
                                "raw": block[1:-1, 1:-1].copy(), "counts": {}, "bars": []})
    windows.sort(key=lambda d: (d["r"], d["c"]))
    _require(2 <= len(windows) <= 4, "need two through four uniquely keyed windows")
    _require(len({(d["s"], d["rim"], d["key_color"]) for d in windows}) == 1,
             "frames must share size and colored key")
    s, rim, key_color = windows[0]["s"], windows[0]["rim"], windows[0]["key_color"]
    boxes = []
    exterior = a != bg
    for d in windows:
        r, c = d["r"], d["c"]
        box = (r - 2, c - 2, r + s + 2, c + s + 2)
        _require(box[0] >= 0 and box[1] >= 0 and box[2] <= h and box[3] <= w,
                 "window requires a two-cell counter margin")
        _require(all(_separate(box, other) for other in boxes), "overlapping window margins")
        boxes.append(box)
        exterior[r:r + s, c:c + s] = False
    colors = sorted({int(x) for d in windows for x in np.unique(d["raw"]) if x != bg})
    _require(2 <= len(colors) <= 4 and not set(colors) & {rim, key_color}, "invalid mosaic palette")
    for component in _components(exterior):
        cells = np.asarray(component, dtype=int)
        rs, cs = cells.T
        bar_colors = {int(a[p]) for p in component}
        _require(len(bar_colors) == 1, "counters must be monochromatic and separated")
        color = bar_colors.pop()
        _require(color in colors, "counter has no mosaic color")
        length = len(component)
        horizontal = np.ptp(rs) == 0 and np.ptp(cs) + 1 == length
        vertical = np.ptp(cs) == 0 and np.ptp(rs) + 1 == length
        _require(horizontal or vertical, "counter must be a solid straight bar")
        owners = []
        for i, d in enumerate(windows):
            r, c = d["r"], d["c"]
            if horizontal and min(cs) >= c and max(cs) < c + s:
                if rs[0] == r - 2:
                    owners.append((i, 0))
                elif rs[0] == r + s + 1:
                    owners.append((i, 2))
            if vertical and min(rs) >= r and max(rs) < r + s:
                if cs[0] == c - 2:
                    owners.append((i, 3))
                elif cs[0] == c + s + 1:
                    owners.append((i, 1))
        _require(len(owners) == 1, "counter ownership is ambiguous")
        i, side = owners[0]
        d = windows[i]
        _require(color not in d["counts"] and all(b["side"] != side for b in d["bars"]),
                 "one counter per color and per occupied side")
        d["counts"][color] = length
        d["bars"].append({"color": color, "length": length, "side": side,
                          "horizontal": bool(horizontal), "cells": tuple(component)})
    for d in windows:
        _require(all(np.any(d["raw"] == color) for color in colors),
                 "every window must visibly contain every mosaic color")
    return {"input": a, "background": bg, "rim": rim, "key_color": key_color,
            "n": s - 2, "colors": colors, "windows": windows}


def _complete(info, *, use_counts=True, respect_keys=True, direction=1, world_turns=False,
              count_rule="actual"):
    """Canonical union followed by independent forward rendering.

    Optional switches are diagnostic alternative rules, never solver state.
    A conflicting alternative raises ValueError instead of assigning priority.
    """
    n, colors = info["n"], info["colors"]
    pools = {color: np.zeros((n, n), dtype=bool) for color in colors}

    def turns(d, color):
        if not use_counts:
            return 0
        if count_rule == "one_color_for_all":
            return d["counts"].get(colors[0], 0)
        if count_rule == "first_window_for_all":
            return info["windows"][0]["counts"].get(color, 0)
        if count_rule == "rank":
            k = d["counts"].get(color, 0)
            return sorted(set(d["counts"].values())).index(k) + 1 if k else 0
        _require(count_rule == "actual", "unknown diagnostic count rule")
        return d["counts"].get(color, 0)

    observations = []
    for d in info["windows"]:
        masks = {}
        for color in colors:
            k = turns(d, color)
            mask = d["raw"] == color
            if world_turns:
                mask = np.rot90(mask, direction * k)
            if respect_keys:
                mask = _unorient(mask, d["t"])
            if not world_turns:
                mask = np.rot90(mask, direction * k)
            masks[color] = mask
            pools[color] |= mask
        observations.append(masks)
    occupied = np.zeros((n, n), dtype=bool)
    for mask in pools.values():
        _require(not np.any(mask & occupied), "canonical color layers conflict")
        occupied |= mask
    result = info["input"].copy()
    completed_masks = []
    for d in info["windows"]:
        local = np.full((n, n), info["background"], dtype=np.int8)
        occupied = np.zeros((n, n), dtype=bool)
        masks = {}
        for color in colors:
            k = turns(d, color)
            mask = pools[color]
            if not world_turns:
                mask = np.rot90(mask, -direction * k)
            if respect_keys:
                mask = _orient(mask, d["t"])
            if world_turns:
                mask = np.rot90(mask, -direction * k)
            _require(not np.any(mask & occupied), "displayed color layers conflict")
            local[mask] = color
            occupied |= mask
            masks[color] = mask
        r, c = d["r"] + 1, d["c"] + 1
        result[r:r + n, c:c + n] = local
        completed_masks.append(masks)
    return result, pools, observations, completed_masks


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair from the input alone. Clockwise is measured relative to each key."""
    return _complete(_parse(input_grid))[0]


def _different(info, correct, **alternative):
    try:
        return not np.array_equal(_complete(info, **alternative)[0], correct)
    except ValueError:
        return True


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    info = _parse(pair["input"])
    out = _grid(pair["output"])
    expected, pools, observations, final_masks = _complete(info)
    _require(np.array_equal(out, expected), "incorrect repaired mosaic")
    source, bg = info["input"], info["background"]
    _require(out.shape == source.shape, "output must retain canvas")
    _require(np.array_equal(out[source != bg], source[source != bg]), "existing cells changed")
    allowed = np.zeros(source.shape, dtype=bool)
    areas = {color: int(mask.sum()) for color, mask in pools.items()}
    for i, d in enumerate(info["windows"]):
        r, c, n = d["r"] + 1, d["c"] + 1, info["n"]
        allowed[r:r + n, c:c + n] = True
        for color in info["colors"]:
            area = int(observations[i][color].sum())
            _require(0 < area < areas[color], "fragments must be nonempty and incomplete per color")
            _require(int(final_masks[i][color].sum()) == areas[color], "layer area not conserved")
    _require(np.array_equal(out[~allowed], source[~allowed]), "changed exterior/frame cells")
    _require(np.any(out != source), "must make a visible repair")
    info["areas"] = areas
    info["pools"] = pools
    info["needs_phases"] = _different(info, out, use_counts=False)
    info["needs_keys"] = _different(info, out, respect_keys=False)
    info["clockwise_witness"] = _different(info, out, direction=-1)
    info["handedness_witness"] = _different(info, out, world_turns=True)
    info["independent_color_witness"] = _different(info, out, count_rule="one_color_for_all")
    info["independent_window_witness"] = _different(info, out, count_rule="first_window_for_all")
    info["rank_witness"] = _different(info, out, count_rule="rank")
    # Test union alternatives directly on the phase-corrected observations.
    supports = {color: np.stack([m[color] for m in observations]) for color in info["colors"]}
    info["singleton_evidence"] = any(np.any(m.sum(axis=0) == 1) for m in supports.values())
    info["majority_witness"] = any(not np.array_equal(m.sum(axis=0) > len(m) // 2, pools[color])
                                   for color, m in supports.items())
    info["intersection_witness"] = any(not np.array_equal(m.all(axis=0), pools[color])
                                       for color, m in supports.items())
    info["largest_fragment_witness"] = any(not np.array_equal(m[np.argmax(m.sum(axis=(1, 2)))], pools[color])
                                           for color, m in supports.items())
    return info


def validate_generated_puzzle(puzzle):
    """Check returned pixels, canonical consistency, evidence, and shift."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
        _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
        a, b = len(puzzle["train"]), len(puzzle["test"])
        _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
        seen, infos = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                k = _key(info["input"])
                _require(k not in seen, "duplicate input")
                seen.add(k)
                infos[split].append(info)
        train = infos["train"]
        for feature in ("needs_phases", "needs_keys", "clockwise_witness", "handedness_witness",
                        "independent_color_witness", "independent_window_witness", "rank_witness",
                        "singleton_evidence", "majority_witness", "intersection_witness",
                        "largest_fragment_witness"):
            _require(any(f[feature] for f in train), "training does not distinguish " + feature)
        lengths = {k for f in train for d in f["windows"] for k in d["counts"].values()}
        _require({1, 2, 3, 4, 5} <= lengths, "demonstrate counts one through five")
        _require(any(len(d["counts"]) < len(f["colors"]) for f in train for d in f["windows"]),
                 "demonstrate the absent-counter default")
        _require(all(f["needs_phases"] for fs in infos.values() for f in fs),
                 "phase-blind pooling must not suffice")
        _require(all(max(d["counts"].values(), default=0) <= 5 for f in train for d in f["windows"]),
                 "training counter curriculum exceeded")
        _require(all(any(k >= 6 for d in f["windows"] for k in d["counts"].values())
                     for f in infos["test"]), "tests must extrapolate counter length")
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise AssertionError(str(error)) from error


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        r, c = min(remaining)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n - 1 - r
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _latent_masks(rng, n, colors, informative):
    """Build varied partial shapes in disjoint rotation-invariant territories."""
    orbits = [o for o in _orbits(n) if len(o) == 4]
    mode = int(rng.integers(3))
    if mode == 0:
        rng.shuffle(orbits)
    elif mode == 1:
        orbits.sort(key=lambda o: min(min(r, c, n - 1 - r, n - 1 - c) for r, c in o))
    else:
        orbits.sort(key=lambda o: min(abs(r - c) for r, c in o))
    shares = np.ones(len(colors), dtype=int)
    shares += rng.multinomial(len(orbits) - len(colors), rng.dirichlet(np.ones(len(colors)) * 2))
    territories, cursor = [], 0
    for size in shares:
        territories.append(orbits[cursor:cursor + int(size)])
        cursor += int(size)
    rng.shuffle(territories)
    masks = {}
    for color, territory in zip(colors, territories):
        kind = 0 if informative else int(rng.choice(3, p=[0.65, 0.2, 0.15]))
        mask = np.zeros((n, n), dtype=bool)
        if kind == 2:
            for orbit in territory:
                if rng.random() < 0.7 or not mask.any():
                    for p in orbit:
                        mask[p] = True
        elif kind == 1:
            for orbit in territory:
                if rng.random() < 0.65 or not mask.any():
                    phase = int(rng.integers(2))
                    mask[orbit[phase]] = mask[orbit[phase + 2]] = True
        else:
            coords = [p for o in territory for p in o]
            style = int(rng.integers(4))
            anchor = rng.uniform(0, n, 2)
            phase = int(rng.integers(n))
            scores = []
            for r, c in coords:
                if style == 0:
                    score = (r - anchor[0]) ** 2 + (c - anchor[1]) ** 2 + rng.random()
                elif style == 1:
                    score = (r + phase) % 3 + rng.random() * 0.4
                elif style == 2:
                    score = (r + c + phase) % 2 + rng.random() * 0.4
                else:
                    score = rng.random()
                scores.append(score)
            wanted = max(2, int(round(len(coords) * rng.uniform(0.3, 0.85))))
            for j in np.argsort(scores)[:wanted]:
                mask[coords[int(j)]] = True
            # This adjacent pair on one orbit gives an exact asymmetric witness.
            orbit = territory[0]
            for p in orbit:
                mask[p] = False
            phase = int(rng.integers(4))
            mask[orbit[phase]] = mask[orbit[(phase + 1) % 4]] = True
        masks[color] = mask
    if n % 2 and rng.random() < 0.5:
        masks[int(rng.choice(colors))][n // 2, n // 2] = True
    return masks


def _fragments(rng, masks, count):
    fragments = [{color: np.zeros_like(mask) for color, mask in masks.items()} for _ in range(count)]
    for color, mask in masks.items():
        coords = np.argwhere(mask)
        area = len(coords)
        offset = int(rng.integers(area))
        forbidden = (np.arange(count) + offset) % area
        visible = rng.random((count, area)) < rng.uniform(0.35, 0.8)
        visible[np.arange(count), forbidden] = False
        for j in range(area):
            if not visible[:, j].any():
                eligible = np.flatnonzero(forbidden != j)
                visible[int(rng.choice(eligible)), j] = True
        for i in range(count):
            if not visible[i].any():
                eligible = np.flatnonzero(np.arange(area) != forbidden[i])
                visible[i, int(rng.choice(eligible))] = True
            for r, c in coords[visible[i]]:
                fragments[i][color][r, c] = True
    return fragments


def _partition_axis(rng, total, side):
    gap = int(rng.integers(1, total - 2 * side + 1))
    cut = int(rng.integers(side, total - gap - side + 1))
    return ((0, cut), (cut + gap, total))


def _layout(rng, side, count):
    mode = int(rng.integers(3)) if count == 2 else 2
    if mode == 0:
        h, w = int(rng.integers(side, 31)), int(rng.integers(2 * side + 1, 31))
        xs = _partition_axis(rng, w, side)
        positions = [(int(rng.integers(h - side + 1)), int(rng.integers(lo, hi - side + 1)))
                     for lo, hi in xs]
    elif mode == 1:
        h, w = int(rng.integers(2 * side + 1, 31)), int(rng.integers(side, 31))
        ys = _partition_axis(rng, h, side)
        positions = [(int(rng.integers(lo, hi - side + 1)), int(rng.integers(w - side + 1)))
                     for lo, hi in ys]
    else:
        h, w = map(int, rng.integers(2 * side + 1, 31, size=2))
        ys, xs = _partition_axis(rng, h, side), _partition_axis(rng, w, side)
        cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        if count == 2:
            cells = [(0, 0), (1, 1)] if rng.integers(2) else [(0, 1), (1, 0)]
        else:
            rng.shuffle(cells)
            cells = cells[:count]
        positions = [(int(rng.integers(ys[i][0], ys[i][1] - side + 1)),
                      int(rng.integers(xs[j][0], xs[j][1] - side + 1))) for i, j in cells]
    rng.shuffle(positions)
    return h, w, positions


def _counts(rng, color_count, window_count, cap, witness, is_test):
    if witness == 0:
        table = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]], dtype=int)
        return table[rng.permutation(3)][:, rng.permutation(3)]
    if witness == 1:
        table = np.array([[0, 4, 5], [4, 5, 0]], dtype=int)
        return table[rng.permutation(2)][:, rng.permutation(3)]
    table = rng.integers(0, cap + 1, size=(window_count, color_count))
    if is_test:
        table[0, 0] = int(rng.integers(6, cap + 1))
    # Avoid an invisible shared phase that could be ignored by raw union.
    for j in range(color_count):
        if len(set(map(int, table[:, j] % 4))) == 1:
            choices = [k for k in range(cap + 1) if k % 4 != table[0, j] % 4]
            table[-1, j] = int(rng.choice(choices))
    return table


def _put_counter(module, s, color, length, side, offset):
    if side == 0:
        module[0, 2 + offset:2 + offset + length] = color
    elif side == 1:
        module[2 + offset:2 + offset + length, s + 3] = color
    elif side == 2:
        module[s + 3, 2 + offset:2 + offset + length] = color
    else:
        module[2 + offset:2 + offset + length, 0] = color


def _make_pair(rng, n, witness, is_test, previous_background):
    color_count = 3 if witness in (0, 1) else int(rng.integers(2, 5))
    window_count = (3, 2)[witness] if witness in (0, 1) else int(rng.integers(2, 5))
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_background:
        palette[0], palette[-1] = palette[-1], palette[0]
    bg, rim, key_color = palette[:3]
    colors = palette[3:3 + color_count]
    s, side = n + 2, n + 6
    for attempt in range(64):
        masks = _latent_masks(rng, n, colors, witness in (0, 1))
        fragments = _fragments(rng, masks, window_count)
        phases = _counts(rng, color_count, window_count, s if is_test else 5, witness, is_test)
        base = int(rng.integers(4))
        transforms = [base, 4 + base, (base + 1) % 4] if witness == 0 else list(map(int, rng.integers(8, size=window_count)))
        if witness == 1:
            transforms = [int(rng.integers(4)), int(rng.integers(4)) + 4]
        h, w, positions = _layout(rng, side, window_count)
        inp = np.full((h, w), bg, dtype=np.int8)
        out = inp.copy()
        for i, ((r, c), t) in enumerate(zip(positions, transforms)):
            module = np.full((side, side), bg, dtype=np.int8)
            module[2:2 + s, 2:2 + s] = _frame(s, bg, rim, key_color)
            finished = module.copy()
            sides = list(map(int, rng.permutation(4)))
            for j, color in enumerate(colors):
                k = int(phases[i, j])
                partial = np.rot90(fragments[i][color], -k)
                full = np.rot90(masks[color], -k)
                module[3:3 + n, 3:3 + n][partial] = color
                finished[3:3 + n, 3:3 + n][full] = color
                if k:
                    offset = int(rng.integers(s - k + 1))
                    _put_counter(module, s, color, k, sides[j], offset)
                    _put_counter(finished, s, color, k, sides[j], offset)
            inp[r:r + side, c:c + side] = _orient(module, t)
            out[r:r + side, c:c + side] = _orient(finished, t)
        pair = {"input": inp, "output": out}
        try:
            info = _analyze_pair(pair)
        except ValueError:
            continue
        if not info["needs_phases"]:
            continue
        if witness == 0 and not all(info[f] for f in ("needs_keys", "clockwise_witness", "handedness_witness",
                                                     "independent_color_witness", "independent_window_witness")):
            continue
        # The construction's full answer is checked against input-only union.
        # Parse order differs from construction order; latent supports do not.
        if not all(np.array_equal(info["pools"][color], masks[color]) for color in colors):
            raise AssertionError("a latent cell was not visibly evidenced")
        return pair
    raise RuntimeError("bounded construction exhausted; no weak fallback puzzle is substituted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, used = {"train": [], "test": []}, set()
    previous_background, previous_n = None, None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            witness = index if split == "train" and index < 2 else -1
            sizes = [n for n in range(4, 9) if n != previous_n]
            n = int(rng.choice(sizes))
            for _ in range(8):
                pair = _make_pair(rng, n, witness, split == "test", previous_background)
                k = _key(pair["input"])
                if k not in used:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            used.add(k)
            puzzle[split].append(pair)
            info = _parse(pair["input"])
            previous_background, previous_n = info["background"], n
    validate_generated_puzzle(puzzle)
    return puzzle


def _symmetry(mask):
    if np.array_equal(mask, np.rot90(mask)):
        return "quarter"
    if np.array_equal(mask, np.rot90(mask, 2)):
        return "half"
    return "asymmetric"


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds streamed in memory; no files, logs, traces, or puzzle collection."""
    shapes, sizes, counts, layers, backgrounds, orientations = set(), set(), set(), set(), set(), set()
    lengths, symmetries, fingerprints, normalized = set(), set(), set(), set()
    total_pairs = additions = absent = full_cycles = mixed_sides = phase_needed = key_needed = 0
    singleton_pairs = majority_pairs = rank_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                a, bg = info["input"], info["background"]
                shapes.add(a.shape)
                sizes.add(info["n"])
                counts.add(len(info["windows"]))
                layers.add(len(info["colors"]))
                backgrounds.add(bg)
                orientations.update(d["t"] for d in info["windows"])
                symmetries.update(_symmetry(m) for m in info["pools"].values())
                mapping = {}
                norm = np.array([mapping.setdefault(int(x), len(mapping)) for x in a.flat], dtype=np.int8)
                normalized.add((a.shape, norm.tobytes()))
                for d in info["windows"]:
                    lengths.update(d["counts"].values())
                    absent += len(d["counts"]) < len(info["colors"])
                    full_cycles += any(k in (4, 8) for k in d["counts"].values())
                    mixed_sides += len({b["horizontal"] for b in d["bars"] if b["length"] > 1}) == 2
                additions += int(np.sum(pair["output"] != a))
                phase_needed += info["needs_phases"]
                key_needed += info["needs_keys"]
                singleton_pairs += info["singleton_evidence"]
                majority_pairs += info["majority_witness"]
                rank_pairs += info["rank_witness"]
                total_pairs += 1
        # A color permutation and a proper whole-grid rotation preserve the rule.
        # Whole-grid reflection reverses physical chirality together with the key,
        # and is also valid because the rule measures turns in keyed coordinates.
        if seed < 4:
            pair = puzzle["test"][0]
            permutation = np.random.default_rng(9000 + seed).permutation(10)
            for t in range(8):
                a = _orient(permutation[pair["input"]], t)
                expected = _orient(permutation[pair["output"]], t)
                assert np.array_equal(solve(a), expected)
    assert len(fingerprints) == 42 and len(normalized) >= 200
    assert sizes == set(range(4, 9)) and counts == {2, 3, 4} and layers == {2, 3, 4}
    assert backgrounds == set(range(10)) and orientations == set(range(8))
    assert lengths == set(range(1, 11)) and len(shapes) >= 30
    assert symmetries == {"asymmetric", "half", "quarter"}
    assert phase_needed == total_pairs and key_needed >= 42
    assert absent > 42 and full_cycles > 42 and mixed_sides > 42
    assert singleton_pairs >= 42 and majority_pairs >= 42 and rank_pairs >= 42
    assert additions > 500
    p = generate_puzzle(7001, 2, 1)
    assert _puzzle_digest(p) == _puzzle_digest(generate_puzzle(7001, 2, 1))
    assert _puzzle_digest(p) != _puzzle_digest(generate_puzzle(7002, 2, 1))
    assert len(generate_puzzle(7003, np.int64(10), np.int64(4))["test"]) == 4
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(17, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A changed output pixel must be rejected even if it preserves the palette.
    bad = {split: [{role: value.copy() for role, value in pair.items()} for pair in p[split]]
           for split in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": total_pairs, "distinct_puzzles": len(fingerprints),
            "normalized_inputs": len(normalized), "input_shapes": len(shapes),
            "interior_sizes": sorted(sizes), "window_counts": sorted(counts),
            "layer_counts": sorted(layers), "counter_lengths": sorted(lengths),
            "symmetries": sorted(symmetries), "phase_essential_pairs": phase_needed,
            "key_essential_pairs": key_needed, "cells_added": additions,
            "singleton_evidence_pairs": int(singleton_pairs), "majority_counterexample_pairs": int(majority_pairs),
            "rank_counterexample_pairs": int(rank_pairs),
            "unmarked_windows": int(absent), "full_cycle_windows": int(full_cycles),
            "mixed_marker_orientation_windows": int(mixed_sides)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
