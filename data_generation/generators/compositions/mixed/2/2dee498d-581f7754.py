"""Boundary registration exposes redundant full-height column signatures.

Only NumPy and the standard library are used. No evidence files are loaded.
Generation constructs the registered scene; solve recovers it from input cells.
No puzzles, execution traces, or diagnostics are written by this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 535334150, 204881295, 59770432, 451296342, 70852974,
              125071536, 355466607, 476938062, 604792182, 54348001,
              792788940, 834185646, 204893271, 934627478, 588280697,
              722548763, 654652447, 901957508, 959034014, 932986059,
              795169521, 836927829, 895198249, 760548523, 110441685,
              718436886, 554249575, 530575099, 685129934, 5659442,
              637300050, 316463627, 610541565, 620714390, 111261976,
              998247476, 16455224, 200569452, 231617948, 402912550,
              242082832)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(a, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    points.add(p)
                    stack.append(p)
        result.append(points)
    return result


def _decode(a):
    """Infer background, singleton anchors, guides, and local registration cells."""
    h, w = a.shape
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("background must occupy more than half the input")
    anchors, carriers = {}, []
    for points in _components(a, bg):
        r, c = min(points)
        if len(points) == 1 and (r in (0, h - 1) or c in (0, w - 1)):
            anchors.setdefault(int(a[r, c]), []).append((r, c))
        else:
            carriers.append(points)
    if not anchors or not carriers:
        raise ValueError("need isolated boundary anchors and matching components")
    systems = {}
    for color, points in anchors.items():
        rows, cols = {r for r, c in points}, {c for r, c in points}
        horizontal = len(rows) == 1 and all(c in (0, w - 1) for r, c in points)
        vertical = len(cols) == 1 and all(r in (0, h - 1) for r, c in points)
        if horizontal == vertical:
            raise ValueError("ambiguous or contradictory boundary guide")
        axis = 0 if horizontal else 1
        guide = next(iter(rows if horizontal else cols))
        if not 0 < guide < a.shape[axis] - 1:
            raise ValueError("corner anchors are ambiguous")
        systems[color] = (axis, guide, tuple(sorted(points)))
    movers = []
    for points in carriers:
        local = Counter(int(a[p]) for p in points)
        candidates = [color for color in systems if local[color] == 1]
        if len(candidates) != 1:
            raise ValueError("component needs one uniquely occurring anchored identity")
        color = candidates[0]
        reg = next(p for p in points if int(a[p]) == color)
        axis, guide, _ = systems[color]
        delta = (guide - reg[0], 0) if axis == 0 else (0, guide - reg[1])
        movers.append((points, color, reg, delta))
    if set(m[1] for m in movers) != set(systems):
        raise ValueError("an anchor has no matching mover")
    return bg, systems, movers


def _register(a):
    bg, systems, movers = _decode(a)
    h, w = a.shape
    out = np.full(a.shape, bg, dtype=np.int8)
    occupied = set()
    for color, (_, _, anchors) in systems.items():
        for p in anchors:
            out[p] = color
            occupied.add(p)
    for points, _, _, (dr, dc) in movers:
        for r, c in points:
            p = r + dr, c + dc
            if not (0 <= p[0] < h and 0 <= p[1] < w):
                continue
            if p in occupied:
                raise ValueError("registered footprints overlap")
            out[p] = a[r, c]
            occupied.add(p)
    return out


def _column_keys(a):
    return [bytes(a[:, c]) for c in range(a.shape[1])]


def _first_indices(keys):
    seen, indices = set(), []
    for c, key in enumerate(keys):
        if key not in seen:
            seen.add(key)
            indices.append(c)
    return indices


def _dedup(a):
    return a[:, _first_indices(_column_keys(a))].copy()


def _foil_scene(a, bg, systems, movers, convention):
    """Explicit simpler alternatives for demonstration checks, never for solve."""
    out = a.copy() if convention == "marker_only" else np.full_like(a, bg)
    for color, (_, _, anchors) in systems.items():
        for p in anchors:
            out[p] = color
    if convention == "marker_only":
        for _, _, reg, _ in movers:
            out[reg] = bg
        for _, marker, reg, (dr, dc) in movers:
            out[reg[0] + dr, reg[1] + dc] = marker
    else:
        for points, marker, _, _ in movers:
            axis, guide, _ = systems[marker]
            low, high = min(p[axis] for p in points), max(p[axis] for p in points)
            center = (low + high + (convention == "center_ceil")) // 2
            dr, dc = (guide - center, 0) if axis == 0 else (0, guide - center)
            for r, c in points:
                if 0 <= r + dr < a.shape[0] and 0 <= c + dc < a.shape[1]:
                    out[r + dr, c + dc] = a[r, c]
    return _dedup(out)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Register simultaneously, then deduplicate *columns* globally left to right."""
    return _dedup(_register(_grid(input_grid)))


def _signature(a):
    return a.shape, a.tobytes()


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def pair_features(pair):
    """Visible-grid evidence, independent of construction plans or random seed."""
    a, b = _grid(pair["input"]), _grid(pair["output"])
    bg, systems, movers = _decode(a)
    aligned = _register(a)
    keys, raw_keys = _column_keys(aligned), _column_keys(a)
    retained = _first_indices(keys)
    positions = {}
    for c, key in enumerate(keys):
        positions.setdefault(key, []).append(c)
    fg_groups = [cols for cols in positions.values()
                 if np.any(aligned[:, cols[0]] != bg) and len(cols) > 1]
    newly_equal = [cols for cols in fg_groups
                   if len({raw_keys[c] for c in cols}) > 1]
    nonadjacent = any(any(cols[i + 1] - cols[i] > 1 for i in range(len(cols) - 1))
                      for cols in fg_groups)
    late_new = any(c > i for i, c in enumerate(retained))
    fg_seen, repeated_foreground, fg_repeat_before_new = set(), False, False
    for c, key in enumerate(keys):
        if np.any(aligned[:, c] != bg):
            if key in fg_seen:
                repeated_foreground = True
            elif repeated_foreground:
                fg_repeat_before_new = True
            fg_seen.add(key)
    last_order = [c for c, key in enumerate(keys) if key not in keys[c + 1:]]
    sorted_order = [keys.index(key) for key in sorted(set(keys))]
    near_matches = sum(np.count_nonzero(b[:, i] != b[:, j]) == 1
                       for i in range(b.shape[1]) for j in range(i))
    full_moving = sum(len(points) > 1 and delta != (0, 0)
                      for points, _, _, delta in movers)
    row_shortcut = all(not np.array_equal(
        aligned[:, _first_indices([bytes([int(x)]) for x in aligned[r]])], b)
        for r in range(a.shape[0]))
    return {"input_shape": list(a.shape), "output_shape": list(b.shape),
            "background": bg, "axes": sorted({v[0] for v in systems.values()}),
            "systems": len(systems), "movers": len(movers),
            "sizes": sorted(len(m[0]) for m in movers),
            "guides": [[axis, guide] for axis, guide, _ in systems.values()],
            "bilateral": sum(len(v[2]) == 2 for v in systems.values()),
            "new_foreground_duplicate_types": len(newly_equal),
            "nonadjacent_foreground_repeat": bool(nonadjacent),
            "late_first_occurrence": bool(late_new),
            "foreground_repeat_before_new": fg_repeat_before_new,
            "near_match_pairs": int(near_matches),
            "full_moving_carriers": int(full_moving),
            "single_row_shortcut_fails": row_shortcut,
            "omit_registration_fails": not np.array_equal(_dedup(a), b),
            "omit_dedup_fails": not np.array_equal(aligned, b),
            "crop_fails": not np.array_equal(aligned[:, :b.shape[1]], b),
            "last_occurrence_fails": not np.array_equal(aligned[:, last_order], b),
            "sort_fails": not np.array_equal(aligned[:, sorted_order], b),
            "marker_only_fails": not np.array_equal(_foil_scene(a, bg, systems, movers, "marker_only"), b),
            "center_alignment_fails": all(not np.array_equal(
                _foil_scene(a, bg, systems, movers, convention), b)
                for convention in ("center_floor", "center_ceil")),
            "palette": list(map(int, np.unique(a))),
            "occupancy": float(np.mean(a != bg))}


def _profile(rng, height, spine, body):
    low = int(rng.integers(0, spine + 1))
    high = int(rng.integers(spine, height))
    return {r: int(rng.choice(body)) for r in range(low, high + 1)}


def _draw_scene(rng, mode, repeats, bands, palette):
    """Construct repeated *composite column profiles*, then scatter their carriers.

    Horizontal bands use several compact carriers with independent row offsets.
    Vertical bands use a long spine with independently shifted column phases.
    All bands share body-column types; a late one-cell exception makes whole-height
    equality necessary. Row lanes are only a collision-free construction method.
    """
    if mode == "H":
        axes = [0] * bands
    elif mode == "V":
        axes = [1] * bands
    else:
        axes = [0, 1] + list(map(int, rng.integers(0, 2, size=bands - 2)))
        rng.shuffle(axes)
    has_v = 1 in axes
    block_width = int(rng.integers(4, 6 if repeats >= 4 else 7))
    gaps = [int(rng.integers(1, 3)) for _ in range(repeats - 1)]
    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + block_width + gap)
    span = starts[-1] + block_width
    phase_room = 2 if has_v else 0
    min_width = span + 4 + 2 * phase_room
    if min_width > 30:
        return None
    width = int(rng.integers(min_width, min(30, min_width + 4) + 1))
    left = int(rng.integers(2 + phase_room, width - span - 1 - phase_room))
    k = int(rng.integers(2, min(4, block_width - 1) + 1))
    base = list(range(k))
    while len(base) < block_width - 1:
        base.append(int(rng.integers(k)))
    rng.shuffle(base)
    mark_index = int(rng.choice([c for c in range(block_width)
                                 if 2 * c != block_width - 1]))
    base.insert(mark_index, -1)
    tokens = [None] * span
    for start in starts:
        tokens[start:start + block_width] = base
    # A vertical system has only one marker per long carrier; it can mark any copy.
    marked_copy = int(rng.integers(repeats))
    v_mark_x = starts[marked_copy] + mark_index

    # Separate source lanes; guides and destinations need not be centered in them.
    heights, slacks, spines = [], [], []
    for axis in axes:
        mh = int(rng.integers(2, 5))
        heights.append(mh)
        slacks.append(int(rng.integers(repeats - 1, 5)) if axis == 0
                      else int(rng.integers(0, 3)))
        spines.append(int(rng.integers(mh)))
    row_gaps = [int(rng.integers(1, 3)) for _ in range(bands - 1)]
    core_height = sum(h + s for h, s in zip(heights, slacks)) + sum(row_gaps)
    if core_height + 4 > 30:
        return None
    height = int(rng.integers(core_height + 4, min(30, core_height + 9) + 1))
    top = int(rng.integers(2, height - core_height - 1))
    bg = int(palette[0])
    h_count = axes.count(0)
    # At most two vertical colors can share a column: one top, one bottom anchor.
    v_count = int(rng.integers(1, min(2, axes.count(1)) + 1)) if has_v else 0
    markers = list(map(int, palette[1:1 + h_count + v_count]))
    payload = list(map(int, palette[1 + len(markers):]))
    a = np.full((height, width), bg, dtype=np.int8)
    registered = a.copy()
    occupied_source, occupied_target = set(), set()

    def paint(shape, origin, target):
        for r, c, color in shape:
            p, q = (origin[0] + r, origin[1] + c), (target[0] + r, target[1] + c)
            if not (0 <= p[0] < height and 0 <= p[1] < width
                    and 0 <= q[0] < height and 0 <= q[1] < width):
                return False
            if p in occupied_source or q in occupied_target:
                return False
        for r, c, color in shape:
            p, q = (origin[0] + r, origin[1] + c), (target[0] + r, target[1] + c)
            a[p] = registered[q] = color
            occupied_source.add(p)
            occupied_target.add(q)
        return True

    # Vertical anchors share the visible guide, but can have different identities.
    v_colors = markers[h_count:]
    for j, color in enumerate(v_colors):
        ends = [j] if v_count == 2 else ([0, 1] if rng.random() < .35
                                      else [int(rng.integers(2))])
        for end in ends:
            p = (0 if end == 0 else height - 1, left + v_mark_x)
            paint([(0, 0, color)], p, p)
    phases = list(map(int, rng.choice(np.arange(-phase_room, phase_room + 1),
                                    size=axes.count(1), replace=False))) if has_v else []
    # Choose a later column and change exactly one body cell in exactly one band.
    exception_band = int(rng.integers(bands))
    exception_copy = int(rng.integers(1, repeats))
    exception_col = int(rng.choice([c for c in range(block_width) if c != mark_index]))
    exception_x = starts[exception_copy] + exception_col
    horizontal_index = vertical_index = 0
    for band, axis in enumerate(axes):
        mh, slack, spine = heights[band], slacks[band], spines[band]
        local_body = list(map(int, rng.choice(payload, size=int(rng.integers(2, min(4, len(payload)) + 1)),
                                             replace=False)))
        profiles, seen_profiles = {}, set()
        for token in range(k):
            for _ in range(80):
                profile = _profile(rng, mh, spine, local_body)
                key = tuple(profile.get(r, bg) for r in range(mh))
                if key not in seen_profiles:
                    seen_profiles.add(key)
                    profiles[token] = profile
                    break
            else:
                return None
        profiles[-1] = dict(profiles[int(rng.integers(k))])
        target_top = top + int(rng.integers(slack + 1))
        if axis == 0:
            color = markers[horizontal_index]
            horizontal_index += 1
            guide = target_top + spine
            ends = [0, 1] if rng.random() < .30 else [int(rng.integers(2))]
            for end in ends:
                p = guide, (0 if end == 0 else width - 1)
                if not paint([(0, 0, color)], p, p):
                    return None
            origins = list(map(int, rng.choice(slack + 1, repeats, replace=False)))
            for copy, start in enumerate(starts):
                shape = []
                for c, token in enumerate(base):
                    for r, value in profiles[token].items():
                        if c == mark_index and r == spine:
                            value = color
                        if band == exception_band and start + c == exception_x and r == spine:
                            value = local_body[(local_body.index(value) + 1) % len(local_body)]
                        shape.append((r, c, value))
                if not paint(shape, (top + origins[copy], left + start),
                             (target_top, left + start)):
                    return None
        else:
            color = v_colors[vertical_index % v_count]
            phase = phases[vertical_index]
            vertical_index += 1
            shape = []
            for c, token in enumerate(tokens):
                profile = {spine: local_body[0]} if token is None else profiles[token]
                for r, value in profile.items():
                    if c == v_mark_x and r == spine:
                        value = color
                    if band == exception_band and c == exception_x and r == spine:
                        value = local_body[(local_body.index(value) + 1) % len(local_body)]
                    shape.append((r, c, value))
            if not paint(shape, (target_top, left + phase), (target_top, left)):
                return None
        top += mh + slack + (row_gaps[band] if band < bands - 1 else 0)
    if np.mean(a == bg) <= .5:
        return None
    try:
        # This comparison is between a target drawn from placement coordinates
        # and an input-only reconstruction. They do not share placement records.
        if not np.array_equal(_register(a), registered):
            return None
        pair = {"input": a, "output": _dedup(registered)}
        f = pair_features(pair)
    except ValueError:
        return None
    if not (f["new_foreground_duplicate_types"] >= 1
            and f["nonadjacent_foreground_repeat"] and f["late_first_occurrence"]
            and f["foreground_repeat_before_new"]
            and f["near_match_pairs"] >= 1 and f["full_moving_carriers"] >= 1
            and f["omit_registration_fails"] and f["crop_fails"]
            and f["single_row_shortcut_fails"] and f["marker_only_fails"]
            and f["center_alignment_fails"] and f["last_occurrence_fails"]
            and f["sort_fails"]):
        return None
    return pair


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("H", "V", "mixed", "varied")))
    if regime == "varied":
        train_modes = ["H", "V"] + [str(rng.choice(("H", "V", "mixed")))
                                      for _ in range(num_train - 2)]
        if num_train >= 3:
            train_modes[2] = "mixed"
        rng.shuffle(train_modes)
        test_modes = [str(rng.choice(sorted(set(train_modes)))) for _ in range(num_test)]
    else:
        train_modes, test_modes = [regime] * num_train, [regime] * num_test
    # Shift either repetition depth or the number of independently registered bands.
    shift = str(rng.choice(("repeats", "bands")))
    stable_palette = bool(rng.random() < .25)
    common_palette = rng.permutation(10)
    seen, puzzle = set(), {"train": [], "test": []}
    for split, modes in (("train", train_modes), ("test", test_modes)):
        for mode in modes:
            repeats = int(rng.integers(3, 5) if split == "test" else 2) if shift == "repeats" else int(rng.integers(2, 4))
            if shift == "bands":
                bands = (3 if mode != "V" else int(rng.integers(3, 5))) if split == "test" else (1 if mode == "H" else 2)
            else:
                bands = int(rng.integers(1, 4)) if mode == "H" else int(rng.integers(2, 4))
            palette = common_palette if stable_palette else rng.permutation(10)
            for _ in range(160):
                pair = _draw_scene(rng, mode, repeats, bands, palette)
                if pair is not None and _signature(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction failed to find a valid distinct scene")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, train_modes = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            signature = _signature(source)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            aligned = _register(source)
            keys = [tuple(int(v) for v in aligned[:, c]) for c in range(aligned.shape[1])]
            expected = np.array(list(dict.fromkeys(keys)), dtype=np.int8).T
            _require(np.array_equal(expected, target), "wrong registered, stable column compression")
            _require(target.shape[0] == source.shape[0] and target.shape[1] < source.shape[1], "wrong compression dimensions")
            _require(len(set(_column_keys(target))) == target.shape[1], "duplicate output column")
            _require(set(map(int, np.unique(target))) == set(map(int, np.unique(source))), "palette changed")
            f = pair_features(pair)
            for name in ("nonadjacent_foreground_repeat", "late_first_occurrence",
                         "omit_registration_fails", "omit_dedup_fails", "crop_fails",
                         "single_row_shortcut_fails", "foreground_repeat_before_new",
                         "marker_only_fails", "center_alignment_fails",
                         "last_occurrence_fails", "sort_fails"):
                _require(f[name], f"missing evidence: {name}")
            _require(f["new_foreground_duplicate_types"] >= 1, "registration exposes no foreground duplicates")
            _require(f["near_match_pairs"] >= 1, "missing exact-equality near match")
            _require(f["full_moving_carriers"] >= 1, "no rigid carrier movement")
            mode = tuple(f["axes"])
            if split == "train":
                train_modes.add(mode)
            else:
                _require(mode in train_modes, "test guide arrangement is not demonstrated")


def self_test():
    """Bounded, streaming 42-seed semantic and variation checks; no disk writes."""
    shapes, structures, palettes, axes, backgrounds = set(), set(), set(), set(), set()
    systems, movers, widths, fingerprints = set(), set(), set(), set()
    pair_count = new_duplicates = near_matches = both_ends = 0
    local_geometry = local_palette = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprint = _digest(puzzle)
        _require(fingerprint == _digest(generate_puzzle(seed)), "nondeterministic generation")
        fingerprints.add(fingerprint)
        local_shapes, local_colors = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                _require(np.array_equal(solve(source), target), "solver mismatch")
                f = pair_features(pair)
                pair_count += 1
                shapes.add(tuple(f["input_shape"]))
                systems.add(f["systems"])
                movers.add(f["movers"])
                widths.add(f["output_shape"][1])
                axes.add(tuple(f["axes"]))
                backgrounds.add(f["background"])
                palettes.add(tuple(f["palette"]))
                local_colors.add(tuple(f["palette"]))
                local_shapes.add((source.shape, target.shape))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in source.flat)
                structures.add(hashlib.sha256(bytes(source.shape) + normalized).hexdigest())
                new_duplicates += f["new_foreground_duplicate_types"]
                near_matches += f["near_match_pairs"]
                both_ends += f["bilateral"] > 0
        local_geometry += len(local_shapes) > 1
        local_palette += len(local_colors) > 1
        del puzzle
    _require(len(fingerprints) == 42 and len(structures) >= 150, "insufficient structural variety")
    _require(len(shapes) >= 40 and len(widths) >= 5, "insufficient dimension variety")
    _require(axes == {(0,), (1,), (0, 1)}, "missing guide orientations")
    _require(backgrounds == set(range(10)) and len(palettes) >= 30, "insufficient palette variety")
    _require({1, 2, 3} <= systems and len(movers) >= 6, "insufficient count variety")
    _require(new_duplicates >= pair_count and near_matches >= pair_count, "causal/exactness witnesses missing")
    _require(both_ends >= 10 and local_geometry >= 40 and local_palette >= 20, "missing within-puzzle variety")

    # All 30 legal explicit count combinations, including maximum retained size.
    count_cases = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(101 + 10 * train + test, train, test)
            _require(len(p["train"]) == train and len(p["test"]) == test, "counts ignored")
            count_cases += 1
            del p
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    # Per-pair recoloring and vertical reflection preserve the rule. Horizontal
    # reflection requires recomputing first-occurrence order and is checked too.
    p = generate_puzzle(314)
    rng = np.random.default_rng(2718)
    for pair in p["train"] + p["test"]:
        permutation = rng.permutation(10).astype(np.int8)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]), "color dependence")
        _require(np.array_equal(solve(pair["input"][::-1]), pair["output"][::-1]), "vertical reflection")
        reflected = pair["input"][:, ::-1]
        expected = _dedup(_register(pair["input"])[:, ::-1])
        _require(np.array_equal(solve(reflected), expected), "first occurrence after reflection")

    # Solver boundary cases beyond construction: touching carriers, clipping,
    # singleton movers, and reciprocal marker/body colors have input-only roles.
    edge = np.zeros((9, 14), np.int8)
    edge[1, 0] = 2
    edge[4:8, 3] = [3, 3, 3, 2]
    edge[3:7, 8] = [3, 3, 3, 2]
    registered = np.zeros_like(edge)
    registered[1, 0] = 2
    registered[0:2, 3] = [3, 2]
    registered[0:2, 8] = [3, 2]
    _require(np.array_equal(solve(edge), _dedup(registered)), "clipping")
    touching = np.zeros((8, 13), np.int8)
    touching[0, 5] = 2
    touching[3:5, 0:2] = [[3, 2], [3, 3]]
    expected = np.zeros_like(touching)
    expected[0, 5] = 2
    expected[3:5, 4:6] = [[3, 2], [3, 3]]
    _require(np.array_equal(solve(touching), _dedup(expected)), "boundary carrier mistaken for anchor")
    reciprocal = np.zeros((12, 14), np.int8)
    reciprocal[2, 0], reciprocal[9, 13] = 1, 2
    reciprocal[5, 3:6] = [2, 1, 2]
    reciprocal[6, 8:11] = [1, 2, 1]
    reciprocal[7, 1] = 1
    expected = np.zeros_like(reciprocal)
    expected[2, 0], expected[9, 13] = 1, 2
    expected[2, 3:6], expected[9, 8:11], expected[2, 1] = [2, 1, 2], [1, 2, 1], 1
    _require(np.array_equal(solve(reciprocal), _dedup(expected)), "local registration roles")
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    ambiguous = np.zeros((9, 13), np.int8)
    ambiguous[2, 0], ambiguous[6, 12] = 1, 2
    ambiguous[4, 4:7] = [1, 3, 2]
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous carrier accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "palettes": len(palettes), "guide_arrangements": sorted(map(list, axes)),
            "mover_counts": sorted(movers), "output_widths": sorted(widths),
            "new_foreground_duplicate_types": new_duplicates,
            "near_match_pairs": near_matches, "legal_count_cases": count_cases,
            "within_puzzle_geometry_variation": local_geometry,
            "within_puzzle_palette_variation": local_palette}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
