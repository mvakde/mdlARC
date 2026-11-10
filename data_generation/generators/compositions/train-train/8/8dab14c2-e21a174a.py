"""An L-shaped reversing loom: reverse packets of endpoint obligations.

No parent generator is imported.  The solver parses only visible geometry,
the colored packet spine, and the signed boundary deviations.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer)
             and np.all((a >= 0) & (a <= 9)), "grid must contain color integers")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _blocks(values):
    cuts = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]


def _reverse_packets(pattern, blocks):
    return np.concatenate([pattern[a:b] for a, b in reversed(blocks)])


def _baseline(shape, bg, body, y0, h, x0, narrow, span, foot, foot_top,
              spine_x, labels):
    g = np.full(shape, bg, dtype=np.int8)
    g[y0:y0 + h, x0:x0 + narrow] = body
    start = y0 if foot_top else y0 + h - foot
    g[start:start + foot, x0:x0 + span] = body
    g[y0:y0 + h, spine_x] = labels
    return g


def _canonical_candidate(g, bg, body):
    """Recognize the form with the long arm left and the source face left."""
    markers = (g != bg) & (g != body)
    ys, xs = np.nonzero(markers)
    if not len(ys) or len(set(map(int, xs))) != 1:
        return None
    y0, y1, spine_x = int(ys.min()), int(ys.max()), int(xs[0])
    h = y1 - y0 + 1
    if h < 6 or len(ys) != h or y0 < 1 or y1 >= g.shape[0] - 1:
        return None
    labels = g[y0:y1 + 1, spine_x].copy()
    if np.any((labels == bg) | (labels == body)):
        return None
    blocks = _blocks(labels)
    if not 2 <= len(blocks) <= 8:
        return None
    colors = tuple(int(labels[a]) for a, _ in blocks)
    if len(set(colors)) != len(colors):
        return None
    occupied = g != bg
    if occupied[:y0].any() or occupied[y1 + 1:].any():
        return None
    left, right = [], []
    for row in occupied[y0:y1 + 1]:
        indices = np.flatnonzero(row)
        left.append(int(indices[0]))
        right.append(int(indices[-1]))
    left, right = np.array(left), np.array(right)
    vals, counts = np.unique(left, return_counts=True)
    x0 = int(vals[int(np.argmax(counts))])
    # A strict majority of untouched source rows makes the nominal face visible.
    if int(counts.max()) <= h // 2 or np.any(np.abs(left - x0) > 1):
        return None
    ends = np.unique(right)
    if len(ends) != 2:
        return None
    narrow, span = int(ends[0] - x0 + 1), int(ends[1] - x0 + 1)
    if narrow < 4 or span - narrow < 3:
        return None
    if not x0 + 1 <= spine_x <= x0 + narrow - 2:
        return None
    if x0 < 2 or int(ends[1]) + 2 >= g.shape[1]:
        return None
    wide_rows = right == ends[1]
    foot = int(wide_rows.sum())
    if foot < 2 or h - foot < 3:
        return None
    foot_top = bool(wide_rows[0])
    expected_wide = np.zeros(h, dtype=bool)
    expected_wide[:foot] = foot_top
    if not foot_top:
        expected_wide[-foot:] = True
    if not np.array_equal(wide_rows, expected_wide):
        return None
    base = _baseline(g.shape, bg, body, y0, h, x0, narrow, span, foot,
                     foot_top, spine_x, labels)
    pattern = (x0 - left).astype(np.int8)  # -1 dent; +1 protrusion
    if not np.any(pattern):
        return None
    expected = base.copy()
    for i, sign in enumerate(pattern):
        if sign == -1:
            expected[y0 + i, x0] = bg
        elif sign == 1:
            expected[y0 + i, x0 - 1] = body
    if not np.array_equal(expected, g):
        return None
    return dict(grid=g, bg=bg, body=body, y0=y0, h=h, x0=x0,
                narrow=narrow, span=span, foot=foot, foot_top=foot_top,
                spine_x=spine_x, labels=labels, blocks=blocks, colors=colors,
                right=right, pattern=pattern, base=base, wide=wide_rows)


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg),
             "the outer frame must be background")
    values, counts = np.unique(a[a != bg], return_counts=True)
    _require(len(values) >= 3, "need a body and at least two packet labels")
    body = int(values[int(np.argmax(counts))])
    _require(np.sum(counts == counts.max()) == 1, "body role is ambiguous")
    candidates = []
    for k in range(4):
        info = _canonical_candidate(np.rot90(a, k), bg, body)
        if info is not None:
            info["rotation"] = k
            candidates.append(info)
    _require(len(candidates) == 1, "input must define one L and source face")
    return candidates[0]


def _complete(info, routed_pattern):
    result = info["grid"].copy()
    for i, sign in enumerate(routed_pattern):
        if sign == -1:
            result[info["y0"] + i, info["right"][i] + 1] = info["body"]
        elif sign == 1:
            result[info["y0"] + i, info["right"][i]] = info["bg"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the opposite face using reversed whole signed packets."""
    info = _decode(input_grid)
    routed = _reverse_packets(info["pattern"], info["blocks"])
    return np.rot90(_complete(info, routed), -info["rotation"]).copy()


def _layout(info):
    return (info["h"], info["narrow"], info["span"], info["foot"],
            info["foot_top"], tuple(b - a for a, b in info["blocks"]))


def _packet_layout(info):
    return tuple(b - a for a, b in info["blocks"])


def _sorting_vectors(info):
    p, blocks = info["pattern"], info["blocks"]
    order = sorted(range(len(blocks)), key=lambda j: info["colors"][j])
    return [np.concatenate([p[blocks[j][0]:blocks[j][1]] for j in seq])
            for seq in (order, order[::-1])]


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"),
                                (num_test, 1, 4, "num_test")):
        _require(not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, np.integer))
                 and lo <= int(value) <= hi, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_pair(rng, palette, rotation, mode, order, witness=False,
               shifted=False, forbidden_layouts=(), forbidden_packets=()):
    """Sample independent geometry, then reject uninformative cue patterns."""
    for _ in range(400):
        count = int(rng.integers(3 if order == "nonmonotone" else 2,
                                 8 if shifted else 7))
        h = int(rng.integers(max(9, count + 4), 25))
        lengths = np.ones(count, dtype=int)
        lengths += rng.multinomial(h - count, rng.dirichlet(np.ones(count) * 2))
        bounds = np.r_[0, np.cumsum(lengths)]
        blocks = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        narrow = int(rng.integers(4, 9))
        span = int(rng.integers(max(9, narrow + 4), 23))
        foot = int(rng.integers(2, min(8, h - 3)))
        foot_top = bool(rng.integers(2))
        y0, bottom = map(int, rng.integers(1, 4, size=2))
        x0, margin = map(int, rng.integers(2, 5, size=2))
        shape = (y0 + h + bottom, x0 + span + margin)
        spine_x = int(rng.integers(x0 + 1, x0 + narrow - 1))
        if palette is None:
            colors = rng.permutation(10)
        else:
            colors = np.array(palette)
        bg, body = map(int, colors[:2])
        packet_colors = rng.choice(colors[2:], size=count, replace=False)
        if order == "monotone":
            packet_colors.sort()
            if rng.random() < 0.5:
                packet_colors = packet_colors[::-1]
        elif order == "nonmonotone":
            if np.all(np.diff(packet_colors) > 0) or np.all(np.diff(packet_colors) < 0):
                packet_colors[[0, 1]] = packet_colors[[1, 0]]
        labels = np.repeat(packet_colors, lengths)
        base = _baseline(shape, bg, body, y0, h, x0, narrow, span, foot,
                         foot_top, spine_x, labels)
        wide = np.zeros(h, dtype=bool)
        if foot_top:
            wide[:foot] = True
        else:
            wide[-foot:] = True
        right = np.where(wide, x0 + span - 1, x0 + narrow - 1)
        layout = (h, narrow, span, foot, foot_top, tuple(map(int, lengths)))
        if layout in forbidden_layouts or tuple(map(int, lengths)) in forbidden_packets:
            continue
        for _pattern_attempt in range(100):
            p = np.zeros(h, dtype=np.int8)
            # Strict majority of zero rows makes the unshifted face recoverable.
            amount = int(rng.integers(2, (h - 1) // 2 + 1))
            positions = rng.choice(h, size=amount, replace=False)
            if mode == "mixed":
                signs = rng.choice([-1, 1], size=amount)
                signs[:2] = [-1, 1]
                rng.shuffle(signs)
            else:
                signs = np.full(amount, -1 if mode == "dent" else 1)
            p[positions] = signs
            routed = _reverse_packets(p, blocks)
            if np.array_equal(routed, p) or np.array_equal(routed, p[::-1]):
                continue
            active_packets = sum(np.any(p[a:b]) for a, b in blocks)
            if active_packets < 2:
                continue
            if witness or shifted:
                # At least one instruction crosses the elbow's width regime.
                transport = np.concatenate([np.arange(a, b) for a, b in reversed(blocks)])
                if not np.any((routed != 0) & (wide != wide[transport])):
                    continue
                if not (np.any(routed[wide]) and np.any(routed[~wide])):
                    continue
            if order == "nonmonotone":
                idx = sorted(range(count), key=lambda j: int(packet_colors[j]))
                sorted_vectors = [np.concatenate([p[blocks[j][0]:blocks[j][1]]
                                                  for j in seq])
                                  for seq in (idx, idx[::-1])]
                if any(np.array_equal(routed, v) for v in sorted_vectors):
                    continue
            g = base.copy()
            for i in range(h):
                if p[i] == -1:
                    g[y0 + i, x0] = bg
                elif p[i] == 1:
                    g[y0 + i, x0 - 1] = body
            # Build the answer directly from sampled source-to-destination mapping.
            out = g.copy()
            cursor = 0
            for a, b in reversed(blocks):
                for local, source in enumerate(range(a, b)):
                    dest = cursor + local
                    if p[source] == -1:
                        out[y0 + dest, right[dest] + 1] = body
                    elif p[source] == 1:
                        out[y0 + dest, right[dest]] = bg
                cursor += b - a
            pair = {"input": np.rot90(g, rotation).copy(),
                    "output": np.rot90(out, rotation).copy()}
            return pair
    raise RuntimeError("could not sample an informative loom")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10))) if rng.random() < 0.45 else None
    fixed_rotation = int(rng.integers(4)) if rng.random() < 0.3 else None
    simple = "dent" if rng.random() < 0.5 else "protrusion"
    specs = [(simple, "monotone", False), ("mixed", "nonmonotone", True)]
    modes = ("dent", "protrusion", "mixed")
    specs += [(str(rng.choice(modes)), "random", bool(rng.random() < 0.3))
              for _ in range(nt - 2)]
    rng.shuffle(specs)
    puzzle = {"train": [], "test": []}
    seen, layouts, packets = set(), set(), set()
    for split in ("train", "test"):
        current_specs = specs if split == "train" else [
            (str(rng.choice(modes)), "random", True) for _ in range(ns)]
        for mode, order, witness in current_specs:
            for _ in range(40):
                rotation = fixed_rotation if fixed_rotation is not None else int(rng.integers(4))
                pair = _make_pair(rng, palette, rotation, mode, order, witness,
                                  split == "test", layouts if split == "test" else (),
                                  packets if split == "test" else ())
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        info = _decode(pair["input"])
                        layouts.add(_layout(info))
                        packets.add(_packet_layout(info))
                    break
            else:
                raise RuntimeError("could not sample a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, transported endpoints, conservation, and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos, test_infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must have exactly input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            g, p = info["grid"], info["pattern"]
            actual = np.rot90(out, info["rotation"])
            edits = {}
            for a, b in info["blocks"]:
                # The start in reversed order is the total length after this block.
                target_start = info["h"] - b
                for local, source in enumerate(range(a, b)):
                    if p[source]:
                        dest = target_start + local
                        col = int(info["right"][dest] + (p[source] == -1))
                        edits[(info["y0"] + dest, col)] = (
                            info["body"] if p[source] == -1 else info["bg"])
            expected = g.copy()
            for point, color in edits.items():
                expected[point] = color
            _require(np.array_equal(actual, expected), "wrong routed endpoints")
            _require(np.array_equal(solve(inp), out), "solve disagrees")
            _require(np.count_nonzero(actual != g) == np.count_nonzero(p),
                     "must make exactly one complementary edit per clue")
            _require(np.sum(actual == info["body"]) == np.sum(info["base"] == info["body"]),
                     "carrier area must return to nominal L area")
            marker_mask = (g != info["body"]) & (g != info["bg"])
            _require(np.array_equal(actual[marker_mask], g[marker_mask]),
                     "packet spine moved")
            for color in info["colors"]:
                _require(np.sum(actual == color) == np.sum(g == color),
                         "label count changed")
            routed = _reverse_packets(p, info["blocks"])
            _require(not np.array_equal(routed, p), "same-row shortcut succeeds")
            _require(not np.array_equal(routed, p[::-1]), "row-flip shortcut succeeds")
            _require(not np.array_equal(actual, _complete(info, -routed)),
                     "copying the same boundary sign must fail")
            (train_infos if split == "train" else test_infos).append(info)
    signs = {int(v) for i in train_infos for v in i["pattern"] if v}
    _require(signs == {-1, 1}, "training must teach both complementary signs")
    _require(any(set(map(int, i["pattern"])) >= {-1, 1} for i in train_infos),
             "training must teach mixed packets")
    _require(any(all(not np.array_equal(_reverse_packets(i["pattern"], i["blocks"]), v)
                         for v in _sorting_vectors(i)) for i in train_infos),
             "training does not distinguish numeric sorting")
    _require(any(_crosses_elbow(i) and _both_exits(i) for i in train_infos),
             "training lacks an elbow-routing witness")
    train_layouts = {_layout(i) for i in train_infos}
    train_packets = {_packet_layout(i) for i in train_infos}
    for info in test_infos:
        _require(_layout(info) not in train_layouts and _packet_layout(info) not in train_packets,
                 "test must have new geometry and a new packet-length composition")
        _require(_crosses_elbow(info) and _both_exits(info),
                 "test must route through both widths with an elbow crossing")


def _crosses_elbow(info):
    indices = np.concatenate([np.arange(a, b) for a, b in reversed(info["blocks"])])
    routed = _reverse_packets(info["pattern"], info["blocks"])
    return bool(np.any((routed != 0) & (info["wide"] != info["wide"][indices])))


def _both_exits(info):
    routed = _reverse_packets(info["pattern"], info["blocks"])
    return bool(np.any(routed[info["wide"]]) and np.any(routed[~info["wide"]]))


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded structural and semantic coverage over 42 reproducible seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, packet_counts, heights, widths = set(), set(), set(), set(), set()
    rotations, foot_sides, patterns, modes, bgs, bodies = set(), set(), set(), set(), set(), set()
    inactive = singletons = long_packets = crossed = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                shapes.add(pair["input"].shape)
                layouts.add(_layout(info))
                packet_counts.add(len(info["blocks"]))
                heights.add(info["h"])
                widths.add(info["narrow"])
                rotations.add(info["rotation"])
                foot_sides.add(info["foot_top"])
                p = info["pattern"]
                patterns.add(tuple(map(int, p)))
                modes.add(tuple(sorted(set(map(int, p)) - {0})))
                bgs.add(info["bg"])
                bodies.add(info["body"])
                inactive += sum(not np.any(p[a:b]) for a, b in info["blocks"])
                singletons += sum(b - a == 1 for a, b in info["blocks"])
                long_packets += sum(b - a >= 3 for a, b in info["blocks"])
                crossed += _crosses_elbow(info)
                # The input-only solver respects rotation and color role changes.
                if split == "test":
                    for k in range(4):
                        _require(np.array_equal(solve(np.rot90(pair["input"], k)),
                                                np.rot90(pair["output"], k)),
                                 "rotation equivariance failed")
                    permutation = np.random.default_rng(800 + info["h"]).permutation(10)
                    _require(np.array_equal(solve(permutation[pair["input"]]),
                                            permutation[pair["output"]]),
                             "palette equivariance failed")
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity failed")
    _require(len(shapes) >= 80 and len(layouts) >= 180 and len(patterns) >= 150,
             "structural variation is too narrow")
    _require(packet_counts >= {2, 3, 4, 5, 6, 7} and len(heights) >= 12,
             "packet counts and spans need broader coverage")
    _require(widths == {4, 5, 6, 7, 8} and rotations == {0, 1, 2, 3}
             and foot_sides == {False, True}, "L geometry coverage failed")
    _require(modes == {(-1,), (1,), (-1, 1)} and len(bgs) >= 8 and len(bodies) >= 8,
             "cue or palette coverage failed")
    _require(inactive >= 20 and singletons >= 20 and long_packets >= 200 and crossed >= 80,
             "packet structure coverage failed")
    _require(_signature(generate_puzzle(912, 10, 4))
             == _signature(generate_puzzle(912, 10, 4)), "determinism failed")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + nt * 10 + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                     "requested counts ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Legal family members beyond the informative generation distribution:
    # eight packets, a single clue, and the minimum-height form.
    for lengths, packet_colors, source, sign in (
        ([1] * 7 + [5], list(range(2, 10)), 0, -1),
        ([2, 4], [2, 3], 1, 1),
    ):
        h = sum(lengths)
        labels = np.repeat(packet_colors, lengths)
        g = _baseline((h + 2, 13), 0, 1, 1, h, 2, 4, 9, 2, False, 3, labels)
        g[1 + source, 2 if sign == -1 else 1] = 0 if sign == -1 else 1
        expected = g.copy()
        start = 0
        for length in lengths:
            if start <= source < start + length:
                dest = h - start - length + source - start
                break
            start += length
        endpoint = 10 if dest >= h - 2 else 5
        expected[1 + dest, endpoint + (sign == -1)] = 1 if sign == -1 else 0
        _require(np.array_equal(solve(g), expected), "broader-family endpoint case failed")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "layouts": len(layouts), "patterns": len(patterns),
            "packet_counts": sorted(packet_counts), "inactive_packets": inactive,
            "single_row_packets": singletons, "long_packets": long_packets,
            "elbow_crossing_pairs": crossed}
