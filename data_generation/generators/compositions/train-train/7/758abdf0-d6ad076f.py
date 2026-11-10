"""Inset handshakes: stateful prongs on two facing rectangular banks.

Only NumPy and the Python standard library are used. No parent code, examples,
randomness, or puzzle-level state is consulted by solve().
"""
from __future__ import annotations

import itertools
import numpy as np


_STATES = tuple(itertools.product(range(3), repeat=2))
_SHORT = ((1, 0), (0, 1), (1, 1))
_MATURE = ((2, 0), (0, 2), (2, 1), (1, 2), (2, 2))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any(g < 0) or np.any(g > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return g.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("Illegal training/test counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _bbox(g, color):
    ys, xs = np.where(g == color)
    if not len(ys):
        return None
    box = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    y0, y1, x0, x1 = box
    if not np.all(g[y0:y1 + 1, x0:x1 + 1] == color):
        return None
    return box


def _analyze(input_grid):
    """Infer roles and orientation by reconstructing the entire input."""
    original = _grid(input_grid)
    bg = int(original[0, 0])
    if not (np.all(original[0] == bg) and np.all(original[-1] == bg)
            and np.all(original[:, 0] == bg) and np.all(original[:, -1] == bg)):
        raise ValueError("The outer background frame must be intact")
    colors = [int(c) for c in np.unique(original) if c != bg]
    if len(colors) != 3:
        raise ValueError("Need background, two bank colors, and one prong color")
    candidates = []
    for accent in colors:
        banks = [c for c in colors if c != accent]
        for transposed in (False, True):
            g = original.T if transposed else original
            boxes = [_bbox(g, c) for c in banks]
            if any(b is None for b in boxes):
                continue
            ordered = sorted(zip(boxes, banks), key=lambda item: item[0][2])
            (left, lc), (right, rc) = ordered
            lo, hi = left[3] + 1, right[2]  # Half-open gap.
            start, end = max(left[0], right[0]), min(left[1], right[1])
            if hi - lo < 5 or end - start + 1 < 3:
                continue
            expected = np.full(g.shape, bg, dtype=np.int8)
            for box, color in ((left, lc), (right, rc)):
                y0, y1, x0, x1 = box
                expected[y0:y1 + 1, x0:x1 + 1] = color
            states = np.zeros((g.shape[0], 2), dtype=np.int8)
            valid = True
            for side, box in enumerate((left, right)):
                near = (lo, lo + 1) if side == 0 else (hi - 1, hi - 2)
                for y in range(box[0], box[1] + 1):
                    first, second = (int(g[y, x]) for x in near)
                    if first == accent:
                        length = 2 if second == accent else 1
                        states[y, side] = length
                        for x in near[:length]:
                            expected[y, x] = accent
                    elif second == accent:
                        valid = False
            if valid and np.array_equal(g, expected) and np.any(states):
                candidates.append(dict(grid=g, transposed=transposed,
                    background=bg, accent=accent, left_color=lc, right_color=rc,
                    left=left, right=right, lo=lo, hi=hi, start=start, end=end,
                    states=states))
    if len(candidates) != 1:
        raise ValueError("Input must have a unique facing-bank interpretation")
    return candidates[0]


def _apply(info, inset=True):
    g = info["grid"]
    out = g.copy()
    lo, hi = info["lo"], info["hi"]
    bg, accent = info["background"], info["accent"]
    margin = 1 if inset else 0
    for y in range(info["start"] + margin, info["end"] + 1 - margin):
        a, b = (int(v) for v in info["states"][y])
        if a and b:
            if a == b == 1:
                out[y, lo:lo + 2] = accent
                out[y, hi - 2:hi] = accent
            else:
                out[y, lo:hi] = accent
        elif a == 1:
            out[y, lo + 1] = accent
        elif b == 1:
            out[y, hi - 2] = accent
        elif a == 2:
            out[y, lo:lo + 2] = bg
            out[y, hi - 2:hi] = info["left_color"]
        elif b == 2:
            out[y, hi - 2:hi] = bg
            out[y, lo:lo + 2] = info["right_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    out = _apply(info)
    return out.T.copy() if info["transposed"] else out


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _construct(rng, palette, overlap, gap, rotation, anchor=None):
    """Build labels directly from known geometry, without calling solve."""
    bg, lc, rc, accent = palette
    pre_mode, post_mode = int(rng.integers(3)), int(rng.integers(3))
    if anchor == 0:
        pre_mode = int(rng.integers(1, 3))
        post_mode = 3 - pre_mode  # Partial offset teaches true intersection.
    pre = [0, 0]
    post = [0, 0]
    if pre_mode:
        pre[pre_mode - 1] = int(rng.integers(1, 5))
    if post_mode:
        post[post_mode - 1] = int(rng.integers(1, 5))
    top, bottom, before, after = (int(x) for x in rng.integers(1, 4, size=4))
    start = top + max(pre)
    end = start + overlap - 1
    spans = [(start - pre[i], end + post[i]) for i in range(2)]
    da, db = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    height = end + max(post) + bottom + 1
    width = before + da + gap + db + after
    lo, hi = before + da, before + da + gap
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[spans[0][0]:spans[0][1] + 1, before:lo] = lc
    inp[spans[1][0]:spans[1][1] + 1, hi:hi + db] = rc
    states = np.zeros((height, 2), dtype=np.int8)
    style = int(rng.integers(6))
    previous = (0, 0)
    for y in range(start + 1, end):
        if style == 0:
            state = (0, 0) if rng.random() < .65 else _pick(rng, _STATES)
        elif style == 1:
            state = _pick(rng, ((0, 0), (1, 0), (2, 0)))
            if anchor == 1 or rng.random() < .5:
                state = state[::-1]
        elif style == 2:
            state = _pick(rng, ((1, 1), (1, 2), (2, 1), (2, 2)))
        elif style == 3:
            state = (0, 0) if y % 2 == 0 else _pick(rng, _STATES)
        elif style == 4:
            state = _pick(rng, _STATES)
        else:
            state = previous if rng.random() < .65 else _pick(rng, _STATES)
        states[y] = state
        previous = state
    active_rows = rng.permutation(np.arange(start + 1, end))
    if anchor is not None:
        evidence = list(_STATES)
        if anchor == 1:
            evidence = [p[::-1] for p in evidence]
        for y, state in zip(active_rows, evidence):
            states[y] = state
    else:
        # Even a small test exhibits a short-sensitive and a mature outcome.
        states[active_rows[0]] = _pick(rng, _SHORT)
        states[active_rows[1]] = _pick(rng, _MATURE)
    # The rim visibly contains both states but is frozen, in both splits.
    rim = [_pick(rng, _SHORT), _pick(rng, _MATURE)]
    if rng.random() < .5:
        rim.reverse()
    states[start], states[end] = rim
    for side, (s, e) in enumerate(spans):
        overhang = list(range(s, start)) + list(range(end + 1, e + 1))
        for y in overhang:
            states[y, side] = int(rng.integers(3))
        if overhang:
            states[_pick(rng, overhang), side] = int(rng.integers(1, 3))
        for y in range(s, e + 1):
            length = int(states[y, side])
            if side == 0:
                inp[y, lo:lo + length] = accent
            else:
                inp[y, hi - length:hi] = accent
    out = inp.copy()
    # Construct each active lane as a complete output stripe. This differs from
    # the solver's edits to an inferred input, and makes endpoint errors visible.
    for y in range(start + 1, end):
        a, b = (int(v) for v in states[y])
        stripe = np.full(gap, bg, dtype=np.int8)
        if a > 0 and b > 0 and max(a, b) == 2:
            stripe[:] = accent
        elif a > 0 and b > 0:
            stripe[:2] = stripe[-2:] = accent
        elif a == 1:
            stripe[:2] = accent
        elif b == 1:
            stripe[-2:] = accent
        elif a == 2:
            stripe[-2:] = lc
        elif b == 2:
            stripe[:2] = rc
        out[y, lo:hi] = stripe
    return np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(10, 4, replace=False))
    stable_palette = bool(rng.integers(2))
    shift = _pick(rng, ("long_gap", "narrow_window"))
    first_rotation = int(rng.integers(4))
    seen = set()
    puzzle = {"train": [], "test": []}
    first_overlap = None
    for index in range(num_train + num_test):
        train = index < num_train
        anchor = index if index < 2 else None
        if shift == "narrow_window":
            overlap = int(rng.integers(11, 17) if train else rng.integers(4, 11))
            gap = int(rng.integers(5, 16))
        else:
            overlap = int(rng.integers(11, 17) if anchor is not None
                          else rng.integers(4, 17))
            gap = int(rng.integers(5, 9) if train else rng.integers(9, 16))
        if index == 1 and overlap == first_overlap:
            overlap = 11 + ((overlap - 11 + 1) % 6)
        if index == 0:
            first_overlap = overlap
        rotation = ((first_rotation + index) % 4 if anchor is not None
                    else int(rng.integers(4)))
        colors = palette if stable_palette else tuple(int(c) for c in rng.choice(10, 4, False))
        for _ in range(64):
            inp, out = _construct(rng, colors, overlap, gap, rotation, anchor)
            if _key(inp) not in seen:
                break
        else:
            raise RuntimeError("Could not construct a distinct input")
        seen.add(_key(inp))
        puzzle["train" if train else "test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _without_maturity(info, mature):
    """Concrete alternatives: treat every occupied prong as short or mature."""
    out = info["grid"].copy()
    lo, hi = info["lo"], info["hi"]
    for y in range(info["start"] + 1, info["end"]):
        a, b = (bool(v) for v in info["states"][y])
        out[y, lo:hi] = info["background"]
        if mature and a and b:
            out[y, lo:hi] = info["accent"]
        elif mature and a:
            out[y, hi - 2:hi] = info["left_color"]
        elif mature and b:
            out[y, lo:lo + 2] = info["right_color"]
        else:
            if a:
                out[y, lo:lo + 2] = info["accent"]
            if b:
                out[y, hi - 2:hi] = info["accent"]
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Pairs must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_states, train_axes = set(), set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Canvas changed")
            _require(_key(inp) not in seen, "Repeated input")
            seen.add(_key(inp))
            info = _analyze(inp)
            canonical_out = out.T if info["transposed"] else out
            _require(np.array_equal(solve(inp), out), "Output violates handshake rule")
            _require(set(np.unique(out)).issubset(set(np.unique(inp))), "New unexplained color")
            lo, hi, start, end = (info[k] for k in ("lo", "hi", "start", "end"))
            mutable = np.zeros(info["grid"].shape, dtype=bool)
            mutable[start + 1:end, lo:hi] = True
            _require(np.array_equal(canonical_out[~mutable], info["grid"][~mutable]),
                     "A bank, rim, overhang, or unrelated background cell changed")
            changed_count = 0
            for a, b in info["states"][start + 1:end]:
                a, b = int(a), int(b)
                if a and b:
                    changed_count += 2 if a == b == 1 else hi - lo - a - b
                elif a or b:
                    changed_count += 1 if max(a, b) == 1 else 4
            _require(changed_count == int(np.count_nonzero(inp != out)) > 0,
                     "Wrong lane-wise change count")
            # Every generated pair, including tests, defeats these alternatives.
            _require(not np.array_equal(_apply(info, inset=False), canonical_out),
                     "No evidence for B's one-cell inset")
            for mature in (False, True):
                _require(not np.array_equal(_without_maturity(info, mature), canonical_out),
                         "No evidence for A's length-dependent state")
            active = {tuple(int(v) for v in row) for row in info["states"][start + 1:end]}
            if split == "train":
                train_states.update(active)
                train_axes.add(info["transposed"])
            infos[split].append(info)
    _require(set(_STATES).issubset(train_states), "Training does not expose the full lane table")
    _require(train_axes == {False, True}, "Training must demonstrate both axes")
    overlaps = lambda items: [i["end"] - i["start"] + 1 for i in items]
    gaps = lambda items: [i["hi"] - i["lo"] for i in items]
    _require(overlaps(infos["train"])[0] != overlaps(infos["train"])[1],
             "First demonstrations must vary shared-span length")
    _require(min(gaps(infos["test"])) > max(gaps(infos["train"]))
             or max(overlaps(infos["test"])) < min(overlaps(infos["train"])),
             "Missing geometric train/test shift")


def self_test():
    """Bounded, semantic checks and descriptive diversity over 42 seeds."""
    shapes, geometries, palettes, profiles, signatures = set(), set(), set(), set(), set()
    alignments, depths, gaps, overlaps, axes, shifts = set(), set(), set(), set(), set(), set()
    source_colors, bg_colors, accent_colors = set(), set(), set()
    stable = variable = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        replay = generate_puzzle(seed)
        signature = tuple(_key(p[role]) for s in ("train", "test")
                          for p in puzzle[s] for role in ("input", "output"))
        _require(signature == tuple(_key(p[role]) for s in ("train", "test")
                 for p in replay[s] for role in ("input", "output")), "Non-determinism")
        signatures.add(signature)
        local_palettes, split_info = set(), {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                split_info[split].append(info)
                left, right = info["left"], info["right"]
                overlap = info["end"] - info["start"] + 1
                gap = info["hi"] - info["lo"]
                aligned = int(left[0] == right[0]) + int(left[1] == right[1])
                shapes.add(pair["input"].shape)
                geometries.add((info["transposed"], gap, overlap, left, right))
                palette = tuple(info[k] for k in ("background", "left_color", "right_color", "accent"))
                local_palettes.add(frozenset(palette))
                palettes.add(palette)
                alignments.add(aligned)
                depths.update((left[3] - left[2] + 1, right[3] - right[2] + 1))
                gaps.add(gap)
                overlaps.add(overlap)
                axes.add(info["transposed"])
                source_colors.update((info["left_color"], info["right_color"]))
                bg_colors.add(info["background"])
                accent_colors.add(info["accent"])
                profiles.add(tuple(map(tuple, info["states"].tolist())))
        stable += len(local_palettes) == 1
        variable += len(local_palettes) > 1
        if min(i["hi"] - i["lo"] for i in split_info["test"]) > max(
                i["hi"] - i["lo"] for i in split_info["train"]):
            shifts.add("long_gap")
        else:
            shifts.add("narrow_window")
    _require(len(signatures) == 42, "Seeds repeat puzzles")
    _require(len(shapes) >= 50 and len(geometries) >= 150 and len(profiles) >= 150,
             "Insufficient structural variation")
    _require(alignments == {0, 1, 2} and depths == {1, 2, 3, 4}, "Missing bank geometries")
    _require(len(gaps) >= 10 and len(overlaps) >= 10 and axes == {False, True}, "Narrow geometry")
    _require(shifts == {"long_gap", "narrow_window"} and min(stable, variable) >= 5,
             "Missing puzzle-level variation")
    _require(min(len(source_colors), len(bg_colors), len(accent_colors)) >= 9,
             "Color roles are tied to literal colors")
    # All explicit legal counts, plus type and range rejections.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(42000 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Explicit gap-five labels, rather than reusing the rule implementation.
    # A frozen rim prong keeps the (0,0) fixture a four-color family input.
    expected_stripes = {
        (0, 0): "00000", (1, 0): "88000", (0, 1): "00088",
        (2, 0): "00033", (0, 2): "44000", (1, 1): "88088",
        (2, 1): "88888", (1, 2): "88888", (2, 2): "88888",
    }
    for (a, b), stripe in expected_stripes.items():
        fixture = np.zeros((5, 9), dtype=int)
        fixture[1:4, 1] = 3
        fixture[1:4, 7] = 4
        fixture[1, 2] = 8  # Frozen first shared lane.
        fixture[2, 2:2 + a] = 8
        fixture[2, 7 - b:7] = 8
        expected_fixture = fixture.copy()
        expected_fixture[2, 2:7] = [int(c) for c in stripe]
        _require(np.array_equal(solve(fixture), expected_fixture), "Lane-table fixture failed")
    # A minimal family member outside the evidence-rich sampling regime:
    # one interior lane (overlap three), one-cell banks, and no training context.
    tiny = np.zeros((5, 9), dtype=int)
    tiny[1:4, 1] = 3
    tiny[1:4, 7] = 4
    tiny[2, 2:4] = tiny[2, 6] = 8
    expected = tiny.copy()
    expected[2, 2:7] = 8
    _require(np.array_equal(solve(tiny.tolist()), expected), "Minimal collision unsupported")
    # Solve must commute with all rotations and reflections, independently of
    # the constructor's orientation choices.
    for k in range(4):
        for reflected in (False, True):
            inp = np.rot90(tiny, k)
            out = np.rot90(expected, k)
            if reflected:
                inp, out = np.fliplr(inp), np.fliplr(out)
            _require(np.array_equal(solve(inp), out), "Symmetry error")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "geometries": len(geometries), "prong_profiles": len(profiles),
            "palettes": len(palettes), "stable_palettes": stable, "variable_palettes": variable,
            "shift_modes": sorted(shifts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
