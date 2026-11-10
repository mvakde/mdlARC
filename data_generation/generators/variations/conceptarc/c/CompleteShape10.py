"""Generate single-color shapes with small, inferable missing strokes."""

import hashlib
import numpy as np

SOURCE_DIGESTS = {
    "5c136b72c50e03dac1682c940baac2bff2fdd26ebd747",
    "4bae8a309eb8443782042f3b9c46940366fba37a5f520d473e3bcd8c9b9e966f",
    "a0696612715b73678d43c86382ecd4e030da9549fc698f33cae3c7976a5e00a7",
    "15f4143ab1b8556eb6144b5905f0faff30d3310d37e22c124f3416f4a718b8c6",
    "e1327782b9c1e6f7327c1b6a1f93d669f9e3940525d5167e37923421401db337",
}
DAMAGE_TYPES = ("interior_h", "interior_v", "left", "right", "top", "bottom")


def _counts(rng, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer from {lo} through {hi}")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None:
        choices = np.arange(max(2, (num_test or 0) + 1), 11)
        weights = np.array([4 if n <= 4 else 1 for n in choices], dtype=float)
        num_train = int(rng.choice(choices, p=weights / weights.sum()))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(num_train), int(num_test)


def _ramp(start, end, count, rng):
    steps = np.bincount(rng.integers(1, count, size=abs(end - start)), minlength=count)
    return start + np.sign(end - start) * np.cumsum(steps)


def _outline(rng, axis, profile):
    scale = str(rng.choice(("compact", "wide", "tall", "medium", "large"),
                           p=(0.2, 0.2, 0.2, 0.25, 0.15)))
    ranges = {"compact": ((8, 13), (6, 13)),
              "wide": ((8, 15), (16, 25)),
              "tall": ((16, 25), (6, 15)),
              "medium": ((11, 20), (9, 21)),
              "large": ((18, 25), (15, 25))}
    (hlo, hhi), (wlo, whi) = ranges[scale]
    h, w = int(rng.integers(hlo, hhi)), int(rng.integers(wlo, whi))
    if profile == "bulge":
        ls = int(rng.integers(0, max(2, w // 3 + 1)))
        rs = w - 1 - int(rng.integers(0, max(2, w // 3 + 1)))
        le, re = 0, w - 1
    else:
        ls, rs = 0, w - 1
        le = int(rng.integers(1, max(2, w // 3 + 1)))
        re = w - 1 - int(rng.integers(1, max(2, w // 3 + 1)))
    if axis == "both":
        rs, re = w - 1 - ls, w - 1 - le
    if min(rs - ls, re - le) < 3:
        return None
    half = (h + 1) // 2
    left = _ramp(ls, le, half, rng)
    right = w - 1 - left if axis == "both" else _ramp(rs, re, half, rng)
    mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        i = min(y, h - 1 - y)
        mask[y, int(left[i]):int(right[i]) + 1] = True
    if rng.random() < 0.35:
        row = int(rng.integers(1, half - 1))
        side = str(rng.choice(("left", "right")))
        for yy in {row, h - 1 - row}:
            occupied = np.flatnonzero(mask[yy])
            if side == "left" and occupied[0] >= 2:
                mask[yy, occupied[0] - 2:occupied[0]] = True
            elif side == "right" and occupied[-1] <= w - 3:
                mask[yy, occupied[-1] + 1:occupied[-1] + 3] = True
            if axis == "both":
                mask[yy] |= mask[yy, ::-1]
    if axis == "v":
        mask = mask.T.copy()
    if axis == "h" and np.array_equal(mask, np.fliplr(mask)):
        return None
    if axis == "v" and np.array_equal(mask, np.flipud(mask)):
        return None
    return mask


def _reflect(mask, axis):
    return np.flipud(mask) if axis == "h" else np.fliplr(mask)


def _add_openings(rng, outline, axis):
    h, w = outline.shape
    mask = outline.copy()
    desired = 1 + int(axis != "both" and rng.random() < 0.35)
    for hole_index in range(desired):
        placed = False
        for _ in range(100):
            hh = int(rng.integers(2, min(9, h - 2)))
            ww = int(rng.integers(2, min(11, w - 2)))
            y, x = int(rng.integers(1, h - hh)), int(rng.integers(1, w - ww))
            hole = np.zeros_like(mask)
            hole[y:y + hh, x:x + ww] = True
            if axis in ("h", "both"):
                hole |= np.flipud(hole)
            if axis in ("v", "both"):
                hole |= np.fliplr(hole)
            surround = hole.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                surround |= np.roll(hole, (dy, dx), axis=(0, 1))
            if np.all(mask[surround]) and not np.any(hole[[0, -1], :]) and not np.any(hole[:, [0, -1]]):
                mask[hole] = False
                placed = True
                break
        if not placed and hole_index == 0:
            return None
    return mask


def _crop(mask):
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _connected(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return False
    seen = {(int(ys[0]), int(xs[0]))}
    stack = list(seen)
    h, w = mask.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and (yy, xx) not in seen:
                seen.add((yy, xx))
                stack.append((yy, xx))
    return len(seen) == len(ys)


def _repair_from_visible(train_pairs, inp):
    """Predict using only displayed demonstrations and the input grid."""
    for pair in train_pairs:
        a, b = pair["input"], pair["output"]
        bg = int(a[0, 0])
        fg = [int(c) for c in np.unique(a) if c != bg]
        if len(fg) != 1 or not np.all((a == b) | ((a == bg) & (b == fg[0]))):
            return None
    bg = int(inp[0, 0])
    colors = [int(c) for c in np.unique(inp) if c != bg]
    if len(colors) != 1:
        return None
    top, bottom, left, right = _crop(inp != bg)
    local = (inp != bg)[top:bottom, left:right]
    choices = []
    for axis in ("h", "v"):
        repaired = local | _reflect(local, axis)
        additions = int(np.count_nonzero(repaired & ~local))
        if 0 < additions <= 7:
            result = inp.copy()
            result[top:bottom, left:right][repaired] = colors[0]
            choices.append((additions, result))
    if not choices:
        return None
    best = min(cost for cost, _ in choices)
    winners = [grid for cost, grid in choices if cost == best]
    if any(not np.array_equal(winners[0], grid) for grid in winners[1:]):
        return None
    return winners[0]


def _damage_candidates(mask, axis):
    h, w = mask.shape
    groups = {kind: [] for kind in DAMAGE_TYPES}

    def add(kind, cells):
        cells = tuple(cells)
        mirror = tuple((h - 1 - y, x) if axis == "h" else (y, w - 1 - x) for y, x in cells)
        if (all(mask[y, x] for y, x in cells) and
                not set(cells).intersection(mirror) and all(mask[y, x] for y, x in mirror)):
            groups[kind].append(cells)

    for y in range(h):
        occupied = np.flatnonzero(mask[y])
        if len(occupied) < 4:
            continue
        lo, hi = int(occupied[0]), int(occupied[-1])
        for length in (1, 2, 3):
            if hi - lo + 1 < length + 2:
                continue
            add("left", ((y, x) for x in range(lo, lo + length)))
            add("right", ((y, x) for x in range(hi - length + 1, hi + 1)))
            for start in range(lo + 1, hi - length + 1):
                if mask[y, start - 1] and mask[y, start + length]:
                    add("interior_h", ((y, x) for x in range(start, start + length)))
    for x in range(w):
        occupied = np.flatnonzero(mask[:, x])
        if len(occupied) < 4:
            continue
        lo, hi = int(occupied[0]), int(occupied[-1])
        for length in (1, 2, 3):
            if hi - lo + 1 < length + 2:
                continue
            add("top", ((y, x) for y in range(lo, lo + length)))
            add("bottom", ((y, x) for y in range(hi - length + 1, hi + 1)))
            for start in range(lo + 1, hi - length + 1):
                if mask[start - 1, x] and mask[start + length, x]:
                    add("interior_v", ((y, x) for y in range(start, start + length)))
    return groups


def _place(rng, mask, bg, fg):
    h, w = mask.shape
    familiar_sizes = [(17, 17), (13, 20), (12, 12), (16, 15), (14, 14)]
    fitting = [(hh, ww) for hh, ww in familiar_sizes if hh >= h and ww >= w]
    if fitting and rng.random() < 0.18:
        canvas_h, canvas_w = fitting[int(rng.integers(0, len(fitting)))]
        extra_h, extra_w = canvas_h - h, canvas_w - w
    else:
        max_margin = 4 if rng.random() < 0.35 else 13
        extra_h = int(rng.integers(0, min(max_margin, 31 - h)))
        extra_w = int(rng.integers(0, min(max_margin, 31 - w)))
    top, left = int(rng.integers(0, extra_h + 1)), int(rng.integers(0, extra_w + 1))
    if rng.random() < 0.25:
        top = 0 if rng.random() < 0.5 else extra_h
    if rng.random() < 0.25:
        left = 0 if rng.random() < 0.5 else extra_w
    canvas = np.full((h + extra_h, w + extra_w), bg, dtype=np.uint8)
    canvas[top:top + h, left:left + w][mask] = fg
    return canvas, top, left


def _make_pair(rng, regime, axis, profile, preferred, bg, fg, double):
    for _ in range(80):
        outline = _outline(rng, axis, profile)
        if outline is None:
            continue
        mask = _add_openings(rng, outline, axis)
        if mask is None or not _connected(mask):
            continue
        groups = _damage_candidates(mask, "h" if axis == "both" else axis)
        output, top, left = _place(rng, mask, bg, fg)
        if output[0, 0] != bg:
            continue
        kinds = [preferred] + [k for k in rng.permutation(DAMAGE_TYPES) if k != preferred]
        for kind in kinds:
            for index in rng.permutation(len(groups[kind]))[:50]:
                cells = groups[kind][int(index)]
                damaged = mask.copy()
                for y, x in cells:
                    damaged[y, x] = False
                if _crop(damaged) != _crop(mask):
                    continue
                if double:
                    found = False
                    for other in rng.permutation(DAMAGE_TYPES):
                        for j in rng.permutation(len(groups[other]))[:35]:
                            second = groups[other][int(j)]
                            if any(abs(y - yy) + abs(x - xx) < 4 for y, x in cells for yy, xx in second):
                                continue
                            proposal = damaged.copy()
                            for y, x in second:
                                proposal[y, x] = False
                            if _crop(proposal) == _crop(mask):
                                damaged = proposal
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        continue
                inp = output.copy()
                inp[top:top + mask.shape[0], left:left + mask.shape[1]][mask & ~damaged] = bg
                prediction = _repair_from_visible([], inp)
                if prediction is not None and np.array_equal(prediction, output):
                    return {"input": inp, "output": output}
    raise RuntimeError(f"could not construct an inferable {regime} pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    color_policy = str(rng.choice(("zero", "mixed", "colored"), p=(0.18, 0.64, 0.18)))
    palette = [int(c) for c in rng.choice(np.arange(1, 10), size=int(rng.integers(2, 5)), replace=False)]
    axis_regime = str(rng.choice(("shift", "mixed", "same"), p=(0.4, 0.35, 0.25)))
    shared_axis = str(rng.choice(("h", "v")))
    puzzle = {"train": [], "test": []}
    used = set()
    preferred = list(rng.permutation(DAMAGE_TYPES))
    if preferred[0] not in ("interior_h", "interior_v"):
        j = preferred.index("interior_h")
        preferred[0], preferred[j] = preferred[j], preferred[0]
    for index in range(num_train + num_test):
        split = "train" if index < num_train else "test"
        regime = "source_train" if split == "train" else "target_test"
        if axis_regime == "shift":
            axis = "v" if split == "train" else "h"
            if index > 1 and rng.random() < 0.2:
                axis = "both"
        elif axis_regime == "same":
            axis = shared_axis if rng.random() < 0.8 else "both"
        else:
            axis = str(rng.choice(("h", "v", "both"), p=(0.4, 0.4, 0.2)))
        profile = str(rng.choice(("bulge", "pinch")))
        if color_policy == "zero" or (color_policy == "mixed" and index == 0):
            bg = 0
        elif color_policy == "colored" or (color_policy == "mixed" and index == 1):
            bg = int(rng.integers(1, 10))
        else:
            bg = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
        fg = int(rng.choice([c for c in palette if c != bg]))
        damage_type = preferred[index % len(preferred)]
        double = bool(rng.random() < (0.08 if split == "train" else 0.23))
        for _ in range(20):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, regime, axis, profile, damage_type, bg, fg, double)
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not construct distinct pairs")
    return puzzle


def _hole_count(mask):
    background = ~mask
    h, w = mask.shape
    visited = np.zeros_like(mask)
    count = 0
    for y in range(h):
        for x in range(w):
            if not background[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            touches_edge = False
            while stack:
                cy, cx = stack.pop()
                touches_edge |= cy in (0, h - 1) or cx in (0, w - 1)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            count += not touches_edge
    return count


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= size <= 30 for size in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert inp.shape == out.shape
        input_key = (inp.shape, inp.tobytes())
        pair_key = (input_key, out.tobytes())
        assert input_key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        digest = hashlib.sha256(inp.astype(np.uint8).tobytes() + b"|" + out.astype(np.uint8).tobytes()).hexdigest()
        assert digest not in SOURCE_DIGESTS
        bg = int(inp[0, 0])
        colors = set(int(c) for c in np.unique(inp))
        assert len(colors) == 2 and bg in colors
        fg = (colors - {bg}).pop()
        assert set(int(c) for c in np.unique(out)) == {bg, fg}
        assert np.all((inp == out) | ((inp == bg) & (out == fg)))
        assert np.any(inp != out)
        clean, damaged = out != bg, inp != bg
        assert _crop(clean) == _crop(damaged)
        t, b, l, r = _crop(clean)
        local = clean[t:b, l:r]
        assert np.array_equal(local, np.flipud(local)) or np.array_equal(local, np.fliplr(local))
        assert _connected(local)
        assert _hole_count(local) >= 1
        assert np.array_equal(_repair_from_visible(train, inp), out)


def _same_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"])
        for p, q in zip(a[s], b[s])) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, shapes, colors, backgrounds = (set() for _ in range(6))
    axes, damage_sizes, holes, fingerprints = set(), set(), set(), set()
    short = long = horizontal_asym_tests = 0
    side_repairs = separated_repairs = compact = wide = tall = source_sizes = 0
    colored = border_touch = large_margin = 0
    varied_axis = varied_background = varied_foreground = repeated_foreground = varied_damage = 0
    original_sizes = {(17, 17), (13, 20), (12, 12), (16, 15), (14, 14)}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(seed))
        puzzle_axes, puzzle_backgrounds, puzzle_foregrounds, puzzle_damage = set(), set(), set(), set()
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        short += len(puzzle["train"]) <= 4
        long += len(puzzle["train"]) >= 7
        fingerprints.add(tuple((p["input"].shape, p["input"].tobytes())
                               for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = int(inp[0, 0])
                fg = next(int(c) for c in np.unique(inp) if c != bg)
                backgrounds.add(bg)
                colors.add(fg)
                sizes.add(inp.shape)
                t, b, l, r = _crop(out != bg)
                local = (out != bg)[t:b, l:r]
                shapes.add((local.shape, local.tobytes()))
                hsym, vsym = np.array_equal(local, np.flipud(local)), np.array_equal(local, np.fliplr(local))
                axes.add((hsym, vsym))
                horizontal_asym_tests += split == "test" and hsym and not vsym
                holes.add(_hole_count(local))
                changed = inp != out
                damage_sizes.add(int(np.count_nonzero(changed)))
                separated_repairs += not _connected(changed)
                colored += bg != 0
                border_touch += t == 0 or b == inp.shape[0] or l == 0 or r == inp.shape[1]
                large_margin += max(t, inp.shape[0] - b, l, inp.shape[1] - r) >= 6
                compact += max(inp.shape) <= 12
                wide += inp.shape[1] >= 2 * inp.shape[0]
                tall += inp.shape[0] >= 2 * inp.shape[1]
                source_sizes += inp.shape in original_sizes
                ys, xs = np.nonzero(changed[t:b, l:r])
                side = set()
                for y, x in zip(ys, xs):
                    row = np.flatnonzero(local[y])
                    col = np.flatnonzero(local[:, x])
                    if x == row[0]: side.add("left")
                    if x == row[-1]: side.add("right")
                    if y == col[0]: side.add("top")
                    if y == col[-1]: side.add("bottom")
                if not side: side.add("interior")
                side_repairs += split == "test" and hsym and not vsym and bool(side & {"left", "right"})
                puzzle_axes.add((hsym, vsym))
                puzzle_backgrounds.add(bg)
                puzzle_foregrounds.add(fg)
                puzzle_damage.add(frozenset(side))
                assert np.array_equal(_repair_from_visible(puzzle["train"], inp), out)
        varied_axis += len(puzzle_axes) > 1
        varied_background += len(puzzle_backgrounds) > 1
        varied_foreground += len(puzzle_foregrounds) > 1
        repeated_foreground += len(puzzle_foregrounds) < len(puzzle["train"]) + len(puzzle["test"])
        varied_damage += len(puzzle_damage) > 1
    assert len(fingerprints) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert short >= 15 and long >= 2
    assert len(sizes) >= 35 and len(shapes) >= 100
    assert len(colors) >= 7 and len(backgrounds) >= 7
    assert (True, False) in axes and (False, True) in axes and (True, True) in axes
    assert horizontal_asym_tests >= 15 and max(holes) >= 2 and max(damage_sizes) >= 4
    assert side_repairs >= 8 and separated_repairs >= 10
    assert compact >= 3 and wide >= 5 and tall >= 5 and source_sizes >= 3
    assert colored >= 50 and border_touch >= 100 and large_margin >= 30
    assert varied_axis >= 20 and varied_background >= 15 and varied_foreground >= 30
    assert repeated_foreground >= 20 and varied_damage >= 30
    for nt, ne in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(101, nt, ne)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
        assert _same_puzzles(puzzle, generate_puzzle(101, nt, ne))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 2, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(3, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
