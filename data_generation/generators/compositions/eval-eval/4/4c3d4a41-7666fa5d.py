"""Chromatic loom: keyed pin displacement creates color-agreeing brackets.

The constructor uses destination formulas. The input-only solver reads the
fixture and pushes actual pixel sequences one step at a time. No parent code
or generation metadata is used by the solver.
"""
from __future__ import annotations

import itertools
import numpy as np


HOST_SEEDS = (0, 833806676, 395692426, 280128175, 185233943, 458698532,
              577512086, 637942199, 582736738, 709211848, 284504945, 201396214,
              642682413, 302315205, 836053774, 938295464, 779073657, 412388380,
              234132013, 4122112, 315330451, 136075374, 450930677, 387100598,
              772843356, 773604433, 748319491, 496723205, 32688916, 229407668,
              354633218, 839599276, 157795815, 745102085, 753016508, 786992338,
              125216987, 253486872, 864248784, 271995266, 499195873, 913529168)


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _orient(a, rotation):
    return np.rot90(a, rotation).copy()


def _frame(f):
    a = np.full(f["shape"], f["bg"], dtype=np.int8)
    t, b, l, r, s = (f[k] for k in ("top", "base", "left", "right", "structure"))
    a[t, l:r + 1] = s
    a[t:b + 1, l] = a[t:b + 1, r] = s
    return a


def _comb(a, f, base):
    a[base, f["lanes"][0]:f["lanes"][-1] + 1] = f["structure"]
    for c, h in zip(f["lanes"], f["heights"]):
        a[base - h:base, c] = f["structure"]


def _input(f):
    a = _frame(f)
    _comb(a, f, f["key_base"])
    for c, seq in zip(f["lanes"], f["stacks"]):
        a[f["top"] + 1:f["top"] + 1 + len(seq), c] = seq
    return a


def _weave(a, f, alternative=""):
    """Sweep pin-relative rows. Structural cells are hardware, not payload."""
    out = a.copy()
    for row in range(f["top"] + 1, f["base"]):
        active = [(c, int(a[row, c])) for c in f["lanes"]
                  if int(a[row, c]) not in (f["bg"], f["structure"])]
        spans = []
        if len(active) >= 2:
            left, right = active[0], active[-1]
            if alternative == "any_pair":
                spans = [(c, d, color) for (c, color), (d, other)
                         in itertools.combinations(active, 2) if color == other]
            elif alternative == "consecutive":
                for c, d in zip(f["lanes"], f["lanes"][1:]):
                    color = int(a[row, c])
                    if color not in (f["bg"], f["structure"]) and color == int(a[row, d]):
                        spans.append((c, d, color))
            elif left[1] == right[1]:
                if alternative != "unanimous" or all(color == left[1] for _, color in active):
                    spans = [(left[0], right[0], left[1])]
            elif alternative == "ignore_color":
                spans = [(left[0], right[0], left[1])]
        for left, right, color in spans:
            strip = out[row, left + 1:right]
            strip[strip == f["bg"]] = color
    return out


def _output(f, alternative=""):
    """Direct construction, with explicit sequence offsets and roof clipping."""
    a = _frame(f)
    for c, h, seq in zip(f["lanes"], f["heights"], f["stacks"]):
        start = f["base"] - h - len(seq)
        if alternative == "no_push":
            start = f["top"] + 1
        for offset, value in enumerate(seq):
            row = start + offset
            if f["top"] < row < f["base"]:
                a[row, c] = value
    _comb(a, f, f["base"])
    return _weave(a, f, alternative)


def _runs(row, color):
    positions = np.flatnonzero(row == color)
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def _read_canonical(a, bg, s, top, left, right):
    """Recognize an open frame, a separated comb and matched hanging pins."""
    # Both side rails end on the insertion line. They need not reach the canvas.
    ends = []
    for c in (left, right):
        row = top + 1
        while row < a.shape[0] and int(a[row, c]) == s:
            row += 1
        ends.append(row - 1)
    assert ends[0] == ends[1]
    base = ends[0]
    depth = base - top
    assert depth >= 3
    lanes = tuple(int(c) for c in np.flatnonzero(a[top + 1, left + 1:right] != bg) + left + 1)
    assert len(lanes) >= 2 and min(np.diff(lanes)) >= 2
    assert lanes[0] - left >= 2 and right - lanes[-1] >= 2
    stacks = []
    for c in lanes:
        segment = a[top + 1:base, c]
        occupied = np.flatnonzero(segment != bg)
        assert len(occupied) and np.array_equal(occupied, np.arange(len(occupied)))
        seq = tuple(int(x) for x in segment[:len(occupied)])
        prefix = next((i for i, x in enumerate(seq) if x != s), len(seq))
        assert prefix < len(seq) and all(x not in (bg, s) for x in seq[prefix:])
        stacks.append(seq)
    # The entire lower comb has a solid base and one thin tooth at every lane.
    keys = []
    for k in range(base + 2, a.shape[0]):
        if (lanes[0], lanes[-1]) not in _runs(a[k], s):
            continue
        heights = []
        for c in lanes:
            row = k - 1
            while row >= 0 and int(a[row, c]) == s:
                row -= 1
            heights.append(k - row - 1)
        if not all(1 <= h <= depth - 2 for h in heights):
            continue
        if k - max(heights) <= base + 1:
            continue
        if not all(len(seq) >= depth - h - 1 for h, seq in zip(heights, stacks)):
            continue
        f = dict(shape=a.shape, bg=bg, structure=s, top=top, base=base,
                 left=left, right=right, lanes=lanes, key_base=k,
                 heights=tuple(heights), stacks=tuple(stacks))
        if np.array_equal(a, _input(f)):
            keys.append(f)
    assert len(keys) == 1
    return keys[0]


def _read(grid):
    a = _array(grid)
    values, counts = np.unique(a, return_counts=True)
    assert counts.max() > a.size // 2, "background must occupy a strict majority"
    bg = int(values[np.argmax(counts)])
    candidates = []
    for rotation in range(4):
        canonical = _orient(a, -rotation)
        for s in values:
            s = int(s)
            if s == bg:
                continue
            for top, row in enumerate(canonical):
                for left, right in _runs(row, s):
                    if right - left < 6:
                        continue
                    try:
                        f = _read_canonical(canonical, bg, s, top, left, right)
                    except (AssertionError, IndexError, ValueError):
                        continue
                    f["rotation"] = rotation
                    candidates.append(f)
    assert len(candidates) == 1, "input must have one recognizable loom"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input's visible geometry and colors."""
    f = _read(input_grid)
    a = _orient(_array(input_grid), -f["rotation"])
    out = a.copy()
    # Erase the lower comb by its actual pixels, then copy it to the rail ends.
    comb_cells = np.argwhere(a[f["base"] + 1:] == f["structure"])
    comb_cells[:, 0] += f["base"] + 1
    for row, col in comb_cells:
        out[row, col] = f["bg"]
    for c in f["lanes"]:
        out[f["top"] + 1:f["base"], c] = f["bg"]
    delta = f["key_base"] - f["base"]
    for row, col in comb_cells:
        out[row - delta, col] = a[row, col]
    for c, h in zip(f["lanes"], f["heights"]):
        cells = [(r, int(a[r, c])) for r in range(f["top"] + 1, f["base"])
                 if int(a[r, c]) != f["bg"]]
        tooth_top = f["base"] - h
        while cells[-1][0] >= tooth_top:
            cells = [(r - 1, color) for r, color in cells if r - 1 > f["top"]]
        assert cells and cells[-1][0] == tooth_top - 1
        for r, color in cells:
            out[r, c] = color
    # This implementation checks each cell directly, rather than painting spans.
    woven = out.copy()
    for row in range(f["top"] + 1, f["base"]):
        colored = [c for c in f["lanes"] if int(out[row, c]) not in (f["bg"], f["structure"])]
        if len(colored) < 2:
            continue
        left, right = colored[0], colored[-1]
        if out[row, left] != out[row, right]:
            continue
        for col in range(left + 1, right):
            if int(out[row, col]) == f["bg"]:
                woven[row, col] = out[row, left]
    return _orient(woven, f["rotation"])


def _geometry(rng, n, depth, bg, structure):
    insets = rng.integers(2, 4, size=2).tolist()
    margin_l, margin_r = rng.integers(0, 3, size=2).tolist()
    budget = 30 - (1 + sum(insets) + margin_l + margin_r)
    gaps = [2] * (n - 1)
    slack = budget - sum(gaps)
    for i in rng.permutation(n - 1):
        add = int(rng.integers(0, min(3, slack) + 1))
        gaps[int(i)] += add
        slack -= add
    left = margin_l
    lanes = [left + insets[0]]
    for gap in gaps:
        lanes.append(lanes[-1] + gap)
    right = lanes[-1] + insets[1]
    top = int(rng.integers(0, 3))
    return dict(bg=bg, structure=structure, left=left, right=right,
                top=top, base=top + depth, lanes=tuple(lanes),
                width=right + margin_r + 1)


def _finish_geometry(rng, f):
    k = f["base"] + max(f["heights"]) + int(rng.integers(2, 4))
    margin = int(rng.integers(0, min(2, 29 - k) + 1))
    f["key_base"] = k
    f["shape"] = (k + margin + 1, f.pop("width"))
    return f


def _profile(rng, n, depth, flat=False):
    cap = depth - 2
    mode = int(rng.integers(6))
    if flat or mode == 0:
        return (int(rng.integers(1, cap + 1)),) * n
    h = rng.integers(1, cap + 1, n).tolist()
    if mode in (1, 2):
        h.sort(reverse=mode == 2)
    elif mode == 3:
        low, high = sorted(rng.choice(np.arange(1, cap + 1), 2, replace=False).tolist())
        h = rng.choice([low, high], n).tolist()
    elif mode == 4:
        high = int(rng.integers(2, cap + 1))
        h[0] = h[-1] = high
        h[1:-1] = rng.integers(1, high + 1, n - 2).tolist()
    return tuple(int(x) for x in h)


def _sequence(rng, length, prefix, palette, structure):
    # Payload consists of runs, sometimes revisiting a previous color.
    result = [structure] * prefix
    color = int(rng.choice(palette))
    for _ in range(length - prefix):
        if rng.random() < .35:
            color = int(rng.choice(palette))
        result.append(color)
    return tuple(result)


def _sample(rng, n, depth, bg, structure, palette, witness=False, flat=False):
    f = _geometry(rng, n, depth, bg, structure)
    if witness:
        # This witness varies scale, lane count, spacings, shifts and colors.
        # Its outer survivors are longer than interior survivors, establishing
        # that an inactive interior lane does not block a bracket.
        outer_caps = [int(rng.integers(4, depth - 1)) for _ in range(2)]
        inner_caps = rng.integers(3, min(outer_caps), n - 2).tolist()
        capacities = [outer_caps[0], *inner_caps, outer_caps[1]]
        h = [depth - c - 1 for c in capacities]
        shifts = [int(rng.integers(0, x + 1)) for x in h]
        shifts[0] = int(rng.integers(1, h[0] + 1))
        shifts[-1] = 0
        prefix_lane = int(rng.integers(1, n - 1))
        shifts[prefix_lane] = int(rng.integers(1, h[prefix_lane] + 1))
        stacks = []
        color_a, color_b = (int(x) for x in palette[:2])
        for i, (capacity, shift) in enumerate(zip(capacities, shifts)):
            prefix = shift + 1 if i == prefix_lane else int(rng.integers(0, shift + 1))
            if i == 0:
                prefix = 0  # actual colored payload is lost at the roof
            seq = list(_sequence(rng, capacity + shift, prefix, palette, structure))
            for j in range(capacity):
                if shift + j < prefix:
                    continue
                if i in (0, n - 1):
                    seq[shift + j] = color_b if i == n - 1 and j == 2 else color_a
                elif j == 1:
                    seq[shift + j] = color_b
                elif j == 2:
                    seq[shift + j] = color_a
            stacks.append(tuple(seq))
        f.update(heights=tuple(h), stacks=tuple(stacks))
    else:
        heights = _profile(rng, n, depth, flat)
        stacks = []
        for h in heights:
            capacity = depth - h - 1
            shift = int(rng.integers(0, h + 1))
            length = capacity + shift
            prefix = 0 if rng.random() < .4 else int(rng.integers(0, length))
            stacks.append(_sequence(rng, length, prefix, palette, structure))
        f.update(heights=heights, stacks=tuple(stacks))
    return _finish_geometry(rng, f)


def _blank_lane_mask(f):
    mask = np.zeros(f["shape"], dtype=bool)
    mask[f["top"] + 1:f["base"], f["lanes"][0] + 1:f["lanes"][-1]] = True
    mask[:, list(f["lanes"])] = False
    return mask


def _evidence(f):
    answer = _output(f)
    blank = _blank_lane_mask(f)
    depth = f["base"] - f["top"]
    shifts = [len(seq) - (depth - h - 1) for h, seq in zip(f["heights"], f["stacks"])]
    prefixes = [next(i for i, v in enumerate(seq) if v != f["structure"]) for seq in f["stacks"]]
    fixed_neighbor_answer = _output(f, "consecutive")
    inactive_witness = False
    for row in range(f["top"] + 1, f["base"]):
        # Read true post-push pins before any gaps are filled. A tooth or a
        # structural prefix is inactive for weaving, but still occupies a cell.
        colors = []
        for c, h, seq, shift in zip(f["lanes"], f["heights"], f["stacks"], shifts):
            offset = row - f["top"] - 1 + shift
            value = seq[offset] if offset < len(seq) else f["structure"]
            if value not in (f["bg"], f["structure"]):
                colors.append((c, value))
        if len(colors) < 2 or colors[0][1] != colors[-1][1]:
            continue
        active_columns = {c for c, _ in colors}
        if any(colors[0][0] < c < colors[-1][0] and c not in active_columns for c in f["lanes"]):
            inactive_witness |= bool(np.any(answer[row, blank[row]] != fixed_neighbor_answer[row, blank[row]]))
    survivors = [tuple(v for v in seq[s:] if v != f["structure"])
                 for s, seq in zip(shifts, f["stacks"])]
    return dict(
        fill=bool(np.any(answer[blank] != f["bg"])),
        no_push=bool(np.any(answer[blank] != _output(f, "no_push")[blank])),
        skip=not np.array_equal(answer, fixed_neighbor_answer),
        inactive=inactive_witness,
        any_pair=not np.array_equal(answer, _output(f, "any_pair")),
        unanimous=not np.array_equal(answer, _output(f, "unanimous")),
        color=not np.array_equal(answer, _output(f, "ignore_color")),
        moving_prefix=any(0 < s < p for s, p in zip(shifts, prefixes)),
        payload_clip=any(s > p for s, p in zip(shifts, prefixes)),
        stationary=0 in shifts, differential=len(set(shifts)) > 1,
        multicolor=any(len(set(seq[max(s, p):])) > 1
                      for s, p, seq in zip(shifts, prefixes, f["stacks"])),
        sequence_order=any(seq != seq[::-1] for seq in survivors),
        sequence_sort=any(seq != tuple(sorted(seq)) for seq in survivors),
    )


def _rekey(rng, template):
    f = dict(template)
    depth = f["base"] - f["top"]
    lower = [max(1, depth - len(seq) - 1) for seq in f["stacks"]]
    f["heights"] = tuple(int(rng.integers(lo, depth - 1)) for lo in lower)
    # Keep every pin and the frame fixed; the external key may move down if a
    # taller tooth needs extra room. No hidden off-canvas information is used.
    f["key_base"] = f["base"] + max(f["heights"]) + 2
    f["shape"] = (max(f["shape"][0], f["key_base"] + 1), f["shape"][1])
    return f


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    bg, structure = (int(c) for c in rng.choice(10, 2, replace=False))
    pool = rng.permutation([c for c in range(10) if c not in (bg, structure)]).tolist()
    shift_mode = str(rng.choice(["lanes", "depth", "flat", "rekey"], p=[.35, .30, .20, .15]))
    fixed_rotation = int(rng.integers(4))
    mixed_rotation = bool(rng.random() < .6)
    result = dict(train=[], test=[])
    seen, training = set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index == 0
            for attempt in range(160):
                palette_count = int(rng.integers(2 if witness else 1, 5))
                palette = rng.choice(pool, palette_count, replace=False).tolist()
                if split == "test" and shift_mode == "rekey":
                    f = _rekey(rng, training[int(rng.integers(len(training)))])
                else:
                    if shift_mode == "lanes":
                        n = int(rng.integers(3 if witness else 2, 5)) if split == "train" else int(rng.integers(5, 7))
                    else:
                        n = int(rng.integers(3 if witness else 2, 6 if witness else 7))
                    if shift_mode == "depth":
                        depth = int(rng.integers(6, 10)) if split == "train" else int(rng.integers(10, 13))
                    else:
                        depth = int(rng.integers(6, 12))
                    f = _sample(rng, n, depth, bg, structure, palette, witness,
                                flat=split == "test" and shift_mode == "flat")
                    if shift_mode == "flat" and split == "train" and len(set(f["heights"])) == 1:
                        continue
                evidence = _evidence(f)
                if not evidence["fill"]:
                    continue
                if witness and not all(evidence.values()):
                    continue
                if split == "test" and not evidence["no_push"]:
                    continue
                rotation = fixed_rotation
                if mixed_rotation:
                    rotation = (fixed_rotation + index) % 4 if split == "train" and index < 2 else int(rng.integers(4))
                # Reflection changes tooth/sequence order without a separate rule.
                if rng.random() < .5:
                    inp, out = np.fliplr(_input(f)).copy(), np.fliplr(_output(f)).copy()
                else:
                    inp, out = _input(f), _output(f)
                inp, out = _orient(inp, rotation), _orient(out, rotation)
                if _key(inp) in seen:
                    continue
                if split == "test" and shift_mode == "rekey" and f["heights"] in [t["heights"] for t in training if t["stacks"] == f["stacks"]]:
                    continue
                seen.add(_key(inp))
                result[split].append(dict(input=inp, output=out))
                if split == "train":
                    training.append(f)
                break
            else:
                raise RuntimeError("bounded loom construction exhausted")
        rng.shuffle(result[split])
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, evidence = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _read(inp)
            roles.add((f["bg"], f["structure"]))
            canonical_out = _orient(out, -f["rotation"])
            assert np.array_equal(solve(inp), out), "wrong output"
            assert np.array_equal(canonical_out, _output(f)), "formula/pixel disagreement"
            assert not np.array_equal(inp, out)
            # Frame survives literally; the old key vanishes; no output leaks
            # beyond the open frame. Ordered pin payload is preserved after loss.
            assert np.all(canonical_out[f["base"] + 1:] == f["bg"])
            t, b, l, r = (f[k] for k in ("top", "base", "left", "right"))
            assert np.all(canonical_out[t, l:r + 1] == f["structure"])
            assert np.all(canonical_out[t:b + 1, [l, r]] == f["structure"])
            for c, h, seq in zip(f["lanes"], f["heights"], f["stacks"]):
                capacity = b - t - h - 1
                assert np.array_equal(canonical_out[t + 1:t + capacity + 1, c], seq[-capacity:])
                assert np.all(canonical_out[b - h:b + 1, c] == f["structure"])
            if split == "train":
                evidence.append(_evidence(f))
    assert len(roles) == 1, "background/hardware roles must be shared within a puzzle"
    for feature in ("fill", "no_push", "skip", "inactive", "any_pair", "unanimous", "color",
                    "moving_prefix", "payload_clip", "stationary", "differential", "multicolor",
                    "sequence_order", "sequence_sort"):
        assert any(e[feature] for e in evidence), f"missing training evidence: {feature}"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded authored checks: 42 seeds, all legal counts, analytic fixtures."""
    # Literal three-lane example: capacities 5,3,5; shifts 1,1,0. The middle
    # moving structural prefix survives at row 1. At row 2 a different-colored
    # middle payload is preserved inside a bridge. Row 3 has disagreeing
    # extremes, despite a matching left/middle pair. Rows 4,5 skip the short pin.
    f = dict(shape=(13, 11), bg=0, structure=5, top=0, base=7, left=0, right=10,
             lanes=(2, 5, 8), key_base=12, heights=(1, 3, 1),
             stacks=((9, 2, 2, 2, 2, 2), (5, 5, 3, 2), (2, 2, 3, 2, 2)))
    expected = _frame(f)
    _comb(expected, f, 7)
    expected[1:6, 2] = [2, 2, 2, 2, 2]
    expected[1:4, 5] = [5, 3, 2]
    expected[1:6, 8] = [2, 2, 3, 2, 2]
    expected[1, [3, 4, 6, 7]] = 2
    expected[2, [3, 4, 6, 7]] = 2
    expected[4:6, [3, 4, 6, 7]] = 2
    assert all(_evidence(f).values())
    for rotation in range(4):
        for reflected in (False, True):
            a, answer = _input(f), expected
            if reflected:
                a, answer = np.fliplr(a), np.fliplr(answer)
            a, answer = _orient(a, rotation), _orient(answer, rotation)
            before = a.copy()
            assert np.array_equal(solve(a), answer)
            assert np.array_equal(a, before), "solve must not mutate the input"
    # Every color is allowed in every visible role, including nonzero fields.
    for offset in range(10):
        mapping = np.array([(3 * c + offset) % 10 for c in range(10)], dtype=np.int8)
        assert np.array_equal(solve(mapping[_input(f)]), mapping[expected])
    # Clipping is at the roof, not at the edge of the padded canvas.
    assert np.array_equal(solve(np.pad(_input(f), ((2, 0), (0, 0)))),
                          np.pad(expected, ((2, 0), (0, 0))))
    stationary = dict(f, stacks=((2,) * 5, (5, 3, 2), (2, 2, 3, 2, 2)))
    assert np.array_equal(solve(_input(stationary)), expected)
    # A flat key does not imply equal pin displacement or equal visible colors.
    flat = dict(f, heights=(2, 2, 2),
                stacks=((2, 2, 3, 2), (5, 5, 3, 3, 2, 2), (9, 2, 2, 3, 2)))
    flat_answer = _frame(flat)
    _comb(flat_answer, flat, 7)
    flat_answer[1:5, 2] = [2, 2, 3, 2]
    flat_answer[1:5, 5] = [3, 3, 2, 2]
    flat_answer[1:5, 8] = [2, 2, 3, 2]
    for row, color in ((1, 2), (2, 2), (3, 3), (4, 2)):
        flat_answer[row, [3, 4, 6, 7]] = color
    assert np.array_equal(solve(_input(flat)), flat_answer)
    # Valid family inputs may yield no weaving; disagreement is not an error.
    disagree = dict(f, stacks=((2,) * 6, (5, 5, 2, 2), (3,) * 5))
    disagree_answer = _frame(disagree)
    _comb(disagree_answer, disagree, 7)
    disagree_answer[1:6, 2] = 2
    disagree_answer[1:4, 5] = [5, 2, 2]
    disagree_answer[1:6, 8] = 3
    assert np.array_equal(solve(_input(disagree)), disagree_answer)
    fingerprints, shapes, layouts, lane_counts, depths, profiles, rotations = (set() for _ in range(7))
    payloads, gaps, fill_counts = set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _read(pair["input"])
                shapes.add(f["shape"])
                depths.add(f["base"] - f["top"])
                lane_counts.add(len(f["lanes"]))
                profiles.add(f["heights"])
                rotations.add(f["rotation"])
                gaps.update(np.diff(f["lanes"]).tolist())
                payloads.add(tuple(seq for seq in f["stacks"]))
                layouts.add((tuple(np.diff(f["lanes"])), f["heights"], tuple(map(len, f["stacks"]))))
                mask = _blank_lane_mask(f)
                fill_counts.add(int(np.sum(_output(f)[mask] != f["bg"])))
    assert len(fingerprints) == 42 and len(layouts) >= 160
    assert len(shapes) >= 60 and len(profiles) >= 100 and len(payloads) >= 160
    assert lane_counts == set(range(2, 7)) and rotations == set(range(4))
    assert depths.issuperset(range(6, 13)) and gaps == set(range(2, 6))
    assert len(fill_counts) >= 25
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(0, train, test)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt answer accepted")
    # A correctly solved collection can still omit essential teaching evidence.
    f = dict(
        shape=(13, 11), bg=0, structure=5, top=0, base=7, left=0, right=10,
        lanes=(2, 8), key_base=12, heights=(1, 3),
        stacks=((2,) * 5, (2,) * 3))
    simple = dict(input=_input(f), output=_output(f))
    rotated = dict(input=_orient(_input(f), 1), output=_orient(_output(f), 1))
    query = dict(input=_orient(_input(f), 2), output=_orient(_output(f), 2))
    try:
        validate_generated_puzzle(dict(train=[simple, rotated], test=[query]))
    except AssertionError as error:
        assert "missing training evidence" in str(error)
    else:
        raise AssertionError("insufficient training accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), layouts=len(layouts),
                grid_shapes=len(shapes), profiles=len(profiles),
                lane_counts=sorted(lane_counts), depths=sorted(depths),
                gaps=sorted(gaps), fill_sizes=len(fill_counts))


if __name__ == "__main__":
    print(self_test())
