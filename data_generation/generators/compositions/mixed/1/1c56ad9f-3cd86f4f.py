"""A bottom-anchored affine shear with an additive four-row wave.

Only NumPy and the standard library are used. Generation and self-tests keep
one puzzle at a time in memory and write no files or execution traces.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer ARC arrays with dimensions 1..30")
    return a


def _analyze(value):
    """Parse the visible scaffold, without consulting generation parameters."""
    a = _grid(value)
    bg = int(a[0, 0])
    if not np.all(a[:, 0] == bg):
        raise ValueError("the leftmost column must be background")
    mask = a != bg
    rr, cc = np.nonzero(mask)
    if not len(rr):
        raise ValueError("missing scaffold")
    top, bottom, left, right = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
    h = bottom - top + 1
    if not 5 <= h <= 27 or a.shape[1] + h - 1 > 30:
        raise ValueError("scaffold height or expanded width out of bounds")
    crop = mask[top:bottom + 1, left:right + 1]
    if not np.all(crop[0]) or not np.all(crop[-1]):
        raise ValueError("complete top and bottom bars required")
    core = crop[1:-1]
    rail_mask = np.all(core, axis=0)
    starts = np.flatnonzero(rail_mask & ~np.r_[False, rail_mask[:-1]])
    ends = np.flatnonzero(rail_mask & ~np.r_[rail_mask[1:], False])
    supports = tuple((int(a), int(b)) for a, b in zip(starts, ends))
    if (not rail_mask[0] or not rail_mask[-1] or not 2 <= len(supports) <= 6
            or sum(np.array_equal(row, rail_mask) for row in core) < 2):
        raise ValueError("need 2..6 continuous supports and two open rail rows")
    bars, partial = [], []
    for row, row_mask in enumerate(core, top + 1):
        filled = []
        for (_, end), (start, _) in zip(supports, supports[1:]):
            gap = row_mask[end + 1:start]
            if np.any(gap) and not np.all(gap):
                raise ValueError("crossbars must span whole gaps")
            filled.append(bool(np.all(gap)))
        if any(filled):
            bars.append(row)
            if not all(filled):
                partial.append(row)
    return dict(bg=bg, top=top, bottom=bottom, left=left, right=right,
                height=h, width=right - left + 1, supports=supports,
                gaps=tuple(b[0] - a[1] - 1 for a, b in zip(supports, supports[1:])),
                bars=tuple(bars), partial=tuple(partial),
                bottom_padding=a.shape[0] - bottom - 1,
                right_padding=a.shape[1] - right - 1)


def _shift(distance):
    return distance + (0, -1, 0, 1)[distance % 4]


def _place(a, info, offsets, width):
    out = np.full((a.shape[0], width), info["bg"], dtype=np.int8)
    for row, delta in zip(range(info["top"], info["bottom"] + 1), offsets):
        cols = np.flatnonzero(a[row] != info["bg"])
        targets = cols + int(delta)
        valid = (targets >= 0) & (targets < width)
        out[row, targets[valid]] = a[row, cols[valid]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move whole scaffold rows by d + wave(d), then clip to W + h - 1.

    Background is visibly encoded by the complete blank left column. Empty
    exterior rows stay empty. All foreground colors are copied as payload.
    """
    a = _grid(input_grid)
    info = _analyze(a)
    offsets = [_shift(info["bottom"] - row)
               for row in range(info["top"], info["bottom"] + 1)]
    return _place(a, info, offsets, a.shape[1] + info["height"] - 1)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return (a.shape, a.tobytes())


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _make_input(rng, height, style_bias, kind="general"):
    # The affine width budget is visible in the input and never silently capped.
    budget = 31 - height
    if kind == "clipped":
        right = 0
    elif kind == "lattice":
        right = int(rng.integers(1, min(4, budget - 6) + 1))
    else:
        right = int(rng.integers(0, min(4, budget - 4) + 1))
    min_core = 5 if kind == "lattice" else 3
    left = int(rng.integers(1, min(4, budget - right - min_core) + 1))
    max_core = budget - left - right
    if kind == "clipped":
        count = 2
    else:
        count = int(rng.integers(3 if kind == "lattice" else 2,
                                 min(6, (max_core + 1) // 2) + 1))
    widths, gaps = [1] * count, [1] * (count - 1)
    max_core = min(max_core, 3 * count + 8 * (count - 1))
    core_width = (max_core if rng.random() < 0.25 else
                  int(rng.integers(2 * count - 1, max_core + 1)))
    for _ in range(core_width - (2 * count - 1)):
        choices = [(0, i) for i, w in enumerate(widths) if w < 3]
        choices += [(1, i) for i, gap in enumerate(gaps) if gap < 8]
        lane, index = choices[int(rng.integers(len(choices)))]
        (gaps if lane else widths)[index] += 1
    supports, cursor = [], 0
    for i, width in enumerate(widths):
        supports.append((cursor, cursor + width - 1))
        cursor += width + (gaps[i] if i < count - 1 else 0)
    mask = np.zeros((height, core_width), dtype=bool)
    for start, end in supports:
        mask[:, start:end + 1] = True
    mask[0] = True
    mask[-1] = True
    if kind == "clipped":
        bar_rows = []
    else:
        if kind == "lattice":
            # Both signs of the residual wave act on crossbars, not just rails.
            forced = [int(rng.choice([r for r in range(1, height - 1)
                                      if (height - 1 - r) % 4 == residue]))
                      for residue in (1, 3)]
        else:
            forced = []
        hollow = kind != "lattice" and rng.random() < style_bias
        density = float(rng.uniform(0.12, 0.48))
        chosen = [r for r in range(1, height - 1)
                  if r not in forced and not hollow and rng.random() < density]
        rng.shuffle(chosen)
        bar_rows = forced + chosen[:height - 4 - len(forced)]
    for index, row in enumerate(bar_rows):
        if kind == "lattice" and index == 0:
            filled = [int(rng.integers(0, count - 1))]
        elif rng.random() < 0.50:
            filled = list(range(count - 1))
        else:
            filled = [i for i in range(count - 1) if rng.random() < 0.5]
            if not filled:
                filled = [int(rng.integers(0, count - 1))]
        for i in filled:
            mask[row, supports[i][1] + 1:supports[i + 1][0]] = True

    bg = 0 if rng.random() < 0.45 else int(rng.integers(1, 10))
    colors = [c for c in range(10) if c != bg]
    size = 1 if kind == "clipped" else int(rng.integers(2 if kind == "lattice" else 1, 4))
    palette = rng.choice(colors, size=size, replace=False).astype(np.int8)
    rr, cc = np.indices(mask.shape)
    pattern = int(rng.integers(0, 5)) if kind != "lattice" else int(rng.integers(0, 3))
    if pattern == 0:
        indices = cc % size
    elif pattern == 1:
        indices = rr % size
    elif pattern == 2:
        indices = (rr + cc) % size
    elif pattern == 3:
        indices = (rr // int(rng.integers(1, 4)) + cc // int(rng.integers(1, 4))) % size
    else:
        indices = rng.integers(0, size, size=mask.shape)
    core = np.full(mask.shape, bg, dtype=np.int8)
    core[mask] = palette[indices[mask]]
    top = int(rng.integers(0, min(4, 30 - height) + 1))
    if kind == "clipped":
        bottom = 0
    elif kind == "lattice":
        bottom = int(rng.integers(1, min(4, 30 - height - top) + 1))
    else:
        bottom = int(rng.integers(0, min(5, 30 - height - top) + 1))
    a = np.full((top + height + bottom, left + core_width + right), bg, dtype=np.int8)
    a[top:top + height, left:left + core_width] = core
    return a


def _unique_input(rng, height, bias, kind, seen):
    for _ in range(32):
        a = _make_input(rng, height, bias, kind)
        if _key(a) not in seen:
            seen.add(_key(a))
            return a
    # At most 13 previous inputs versus at least 90 role-preserving recolorings.
    colors = [int(a[0, 0])] + [int(c) for c in np.unique(a) if c != a[0, 0]]
    for targets in itertools.permutations(range(10), len(colors)):
        candidate = np.empty_like(a)
        for source, target in zip(colors, targets):
            candidate[a == source] = target
        key = _key(candidate)
        if key not in seen:
            seen.add(key)
            return candidate
    raise AssertionError("unreachable: more recolorings than previous pairs")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    ceiling = int(rng.integers(9, 17))
    bias = float(rng.choice((0.20, 0.45, 0.70)))
    clipped_height = int(rng.choice([h for h in (8, 12, 16) if h <= ceiling]))
    lattice_height = int(rng.choice([h for h in range(9, ceiling + 1) if h % 4 != 0]))
    plan = [(clipped_height, "clipped"), (lattice_height, "lattice")]
    plan += [(int(rng.integers(5, ceiling + 1)), "general") for _ in range(num_train - 2)]
    seen, train, test = set(), [], []
    for height, kind in plan:
        a = _unique_input(rng, height, bias, kind, seen)
        train.append({"input": a, "output": solve(a)})
    rng.shuffle(train)
    for _ in range(num_test):
        height = int(rng.integers(ceiling + 1, 28))
        a = _unique_input(rng, height, bias, "general", seen)
        test.append({"input": a, "output": solve(a)})
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(inp, out, info):
    """Check coordinates/cell counts directly, independently of solve/_shift."""
    h = info["height"]
    width = inp.shape[1] + h - 1
    _require(out.shape == (inp.shape[0], width), "wrong expansion")
    expected = np.full(out.shape, info["bg"], dtype=np.int8)
    observed, lost = {}, 0
    for row in range(info["top"], info["bottom"] + 1):
        distance = info["bottom"] - row
        residual = 0 if distance % 2 == 0 else (-1 if distance % 4 == 1 else 1)
        delta = distance + residual
        cols = np.flatnonzero(inp[row] != info["bg"])
        kept = []
        for column in cols:
            target = int(column) + delta
            if 0 <= target < width:
                expected[row, target] = inp[row, column]
                kept.append(int(column))
            else:
                lost += 1
        actual = np.flatnonzero(out[row] != info["bg"])
        _require(len(actual) == len(kept), "row count or clipping wrong")
        _require(np.array_equal(actual - np.asarray(kept), np.full(len(kept), delta)),
                 "within-row vectors changed")
        _require(np.array_equal(out[row, actual], inp[row, kept]), "row colors/order changed")
        observed[distance] = int(actual[0]) - int(cols[0])
    _require(np.array_equal(out, expected), "wrong cell copy or background fill")
    _require(np.array_equal(out[info["bottom"], :inp.shape[1]], inp[info["bottom"]]),
             "bottom row moved")
    _require(np.all(out[info["bottom"], inp.shape[1]:] == info["bg"]), "bottom padding wrong")
    predicted_loss = int(h % 4 == 0 and info["right_padding"] == 0)
    _require(lost == predicted_loss, "unexpected clipping geometry")
    _require(np.count_nonzero(inp != info["bg"]) - np.count_nonzero(out != info["bg"]) == lost,
             "foreground conservation failed")
    # Compare on the SAME expanded canvas, so these cannot fail only by size.
    distances = [info["bottom"] - row for row in range(info["top"], info["bottom"] + 1)]
    periodic_only = [(0, -1, 0, 1)[d % 4] for d in distances]
    _require(not np.array_equal(out, _place(inp, info, distances, width)), "A has no causal effect")
    _require(not np.array_equal(out, _place(inp, info, periodic_only, width)), "B has no causal effect")
    return observed, lost


def _learn(train):
    """Recover the affine slope, residual table and width factor from demos."""
    observations, width_factors = {}, set()
    for pair in train:
        a, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        info = _analyze(a)
        extension = out.shape[1] - a.shape[1]
        _require(extension % (info["height"] - 1) == 0, "nonintegral expansion factor")
        width_factors.add(extension // (info["height"] - 1))
        for row in range(info["top"], info["bottom"] + 1):
            d = info["bottom"] - row
            source = np.flatnonzero(a[row] != info["bg"])
            target = np.flatnonzero(out[row] != info["bg"])
            delta = int(target[0]) - int(source[0])
            _require(d not in observations or observations[d] == delta, "inconsistent demonstrations")
            observations[d] = delta
    _require(set(range(9)).issubset(observations), "need two complete cycles and distance eight")
    slope = (observations[4] - observations[0]) / 4
    _require(slope == int(slope), "nonintegral inferred slope")
    slope = int(slope)
    residuals = {}
    for d, delta in observations.items():
        residue, residual = d % 4, delta - slope * d
        _require(residue not in residuals or residuals[residue] == residual, "residual is not periodic")
        residuals[residue] = residual
    _require(slope == 1 and residuals == {0: 0, 1: -1, 2: 0, 3: 1}, "wrong learned rule")
    _require(width_factors == {1}, "wrong learned canvas convention")
    return slope, residuals, next(iter(width_factors))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal counts")
    seen, infos, losses = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            _, loss = _check_pair(a, out, info)
            infos[split].append(info)
            losses[split].append(loss)
    ti = infos["train"]
    _require(1 in losses["train"] and 0 in losses["train"], "training must expose clipping and retention")
    _require(any(not i["bars"] and len(i["supports"]) == 2 for i in ti), "missing simple frame demo")
    _require(any(i["partial"] and len(i["supports"]) >= 3 for i in ti), "missing partial-crossbar demo")
    _require(len({i["height"] % 4 for i in ti}) >= 2, "top phases need contrast")
    _require(len({i["bottom_padding"] for i in ti}) >= 2, "canvas and scaffold anchors ambiguous")
    _require(min(i["height"] for i in infos["test"]) > max(i["height"] for i in ti),
             "missing held-out taller-scaffold shift")
    # Reject natural alternate anchors, reversed wave signs and bar resets.
    for alternative in ("top", "canvas", "reverse", "reset"):
        matches = True
        for pair, info in zip(train, ti):
            a = np.asarray(pair["input"])
            offsets = []
            for row in range(info["top"], info["bottom"] + 1):
                d = info["bottom"] - row
                phase_d = d
                if alternative == "top":
                    phase_d = row - info["top"]
                elif alternative == "canvas":
                    phase_d = a.shape[0] - 1 - row
                elif alternative == "reset":
                    below = [r for r in info["bars"] if r >= row] + [info["bottom"]]
                    phase_d = min(below) - row
                wave = (0, -1, 0, 1)[phase_d % 4]
                offsets.append((phase_d if alternative == "canvas" else d)
                               + (-wave if alternative == "reverse" else wave))
            predicted = _place(a, info, offsets, a.shape[1] + info["height"] - 1)
            matches &= np.array_equal(predicted, np.asarray(pair["output"]))
        _require(not matches, f"training fails to distinguish {alternative} shortcut")
    slope, residuals, width_factor = _learn(train)
    for pair, info in zip(test, infos["test"]):
        a = np.asarray(pair["input"])
        ds = [info["bottom"] - row for row in range(info["top"], info["bottom"] + 1)]
        predicted = _place(a, info, [slope * d + residuals[d % 4] for d in ds],
                           a.shape[1] + width_factor * (info["height"] - 1))
        _require(np.array_equal(predicted, np.asarray(pair["output"])), "training-based prediction failed")


def self_test():
    """Bounded tests: 42 seeds, all 30 legal counts, hand-worked boundaries."""
    source_rows = ["07770"] + ["07070"] * 6 + ["07770"]
    expected_rows = ["000000000777", "000000070700", "000007070000",
                     "000007070000", "000007070000", "000707000000",
                     "070700000000", "077700000000"]
    a = np.array([[int(c) for c in row] for row in source_rows], dtype=np.int8)
    expected = np.array([[int(c) for c in row] for row in expected_rows], dtype=np.int8)
    _require(np.array_equal(solve(a), expected), "hand-worked wave/shear example failed")
    _require(np.array_equal(solve(a[:, :-1]), expected[:, :-1]), "one-cell clipping example failed")
    inverted = np.where(a == 0, 6, 0)
    _require(np.array_equal(solve(inverted), np.where(expected == 0, 6, 0)), "zero foreground failed")
    tall = np.zeros((30, 4), dtype=np.int8)
    tall[1:28, 1] = tall[1:28, 3] = 5
    tall[1, 1:4] = tall[27, 1:4] = 5
    _require(solve(tall).shape == (30, 30), "maximum canvas case failed")
    heights, widths, structures, shapes, supports, gaps, thicknesses = (set() for _ in range(7))
    backgrounds, foregrounds, bar_counts, phases, patterns, signatures = (set() for _ in range(6))
    clipped_tests = partial_tests = multicolor = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        if seed in (0, 17, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                crop = a[info["top"]:info["bottom"] + 1, info["left"]:info["right"] + 1]
                heights.add(info["height"])
                widths.add(info["width"])
                shapes.add(a.shape)
                structures.add((crop.shape, (crop != info["bg"]).tobytes()))
                supports.add(len(info["supports"]))
                gaps.update(info["gaps"])
                thicknesses.update(b - s + 1 for s, b in info["supports"])
                backgrounds.add(info["bg"])
                foregrounds.update(int(c) for c in np.unique(a) if c != info["bg"])
                bar_counts.add(len(info["bars"]))
                patterns.add(tuple(int(c) for c in np.unique(a)))
                multicolor += len(np.unique(a)) > 2
                if split == "test":
                    phases.add(info["height"] % 4)
                    partial_tests += bool(info["partial"])
                    clipped_tests += info["height"] % 4 == 0 and info["right_padding"] == 0
    _require(len(signatures) == 42, "puzzles repeated across seeds")
    _require(len(heights) >= 18 and len(widths) >= 12 and len(shapes) >= 65, "scale diversity too low")
    _require(len(structures) >= 130, "geometry diversity too low")
    _require(supports == {2, 3, 4, 5, 6} and thicknesses == {1, 2, 3}, "support variation missing")
    _require(len(gaps) >= 6 and len(bar_counts) >= 8, "spacing/crossbar variation missing")
    _require(len(backgrounds) >= 8 and foregrounds == set(range(10)), "color roles missing")
    _require(phases == {0, 1, 2, 3} and clipped_tests >= 1 and partial_tests >= 3,
             "test structural combinations missing")
    _require(multicolor >= 40 and len(patterns) >= 40, "payload color variation missing")
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(1000 + ntrain * 10 + ntest, ntrain, ntest)
            _require(len(p["train"]) == ntrain and len(p["test"]) == ntest, "explicit counts ignored")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
