"""Balancing reservoirs: largest blank rectangle, local arm imbalance, inset band.

Only Python's standard library and NumPy are required. No parent code is imported.
The solver reads roles, rectangle bounds, arms, and direction from the input.
"""

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(a.dtype.kind in "iu" and np.all((a >= 0) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int16, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _rectangles(grid, color):
    """Histogram-stack candidates, including every largest solid >=3x3 rectangle.

    Rectangles use exclusive bottom/right. A maximum rectangle must extend to
    its full available histogram width and have one of the histogram heights.
    """
    height, width = grid.shape
    histogram = [0] * width
    found = set()
    for row in range(height):
        for col in range(width):
            histogram[col] = histogram[col] + 1 if grid[row, col] == color else 0
        stack = []
        for col in range(width + 1):
            current = histogram[col] if col < width else 0
            left = col
            while stack and stack[-1][1] > current:
                start, extent = stack.pop()
                if extent >= 3 and col - start >= 3:
                    found.add((row + 1 - extent, row + 1, start, col))
                left = start
            if current and (not stack or stack[-1][1] < current):
                stack.append((left, current))
    return found


def _area(rect):
    top, bottom, left, right = rect
    return (bottom - top) * (right - left)


def _maximal_rectangles(grid, color):
    candidates = sorted(_rectangles(grid, color), key=_area, reverse=True)
    maximal = []
    for rect in candidates:
        t, b, l, r = rect
        if not any(tt <= t and bb >= b and ll <= l and rr >= r
                   for tt, bb, ll, rr in maximal):
            maximal.append(rect)
    return maximal


def _largest(grid):
    best, winners = 0, []
    for color in np.unique(grid):
        for rect in _rectangles(grid, int(color)):
            area = _area(rect)
            if area > best:
                best, winners = area, [(int(color), rect)]
            elif area == best:
                winners.append((int(color), rect))
    _require(len(winners) == 1, "largest solid rectangle must be unique across colors")
    return winners[0]


def _roles(grid, neutral):
    colors, counts = np.unique(grid, return_counts=True)
    nonneutral = [(int(n), int(c)) for c, n in zip(colors, counts) if c != neutral]
    _require(len(nonneutral) in (3, 4, 5), "requires texture and two to four signals")
    nonneutral.sort(reverse=True)
    _require(nonneutral[0][0] > nonneutral[1][0], "texture must dominate nonneutral colors")
    texture = nonneutral[0][1]
    signals = tuple(sorted(c for _, c in nonneutral[1:]))
    return texture, signals


def _read_arms(grid, rect, signals):
    """Read only contiguous signal runs touching opposite rectangle edges."""
    top, bottom, left, right = rect
    results = []
    for axis in (0, 1):  # 0: horizontal arms, 1: vertical arms
        view = grid if axis == 0 else grid.T
        cross0, cross1, start, stop = ((top, bottom, left, right) if axis == 0
                                     else (left, right, top, bottom))
        lengths, lanes = {}, {}
        for color in signals:
            occupied = []
            for lane in range(cross0 + 1, cross1 - 1):
                a = b = 0
                pos = start - 1
                while pos >= 0 and view[lane, pos] == color:
                    a += 1
                    pos -= 1
                pos = stop
                while pos < view.shape[1] and view[lane, pos] == color:
                    b += 1
                    pos += 1
                if a or b:
                    occupied.append((lane, a, b))
            if len(occupied) != 1:
                break
            lane, a, b = occupied[0]
            lengths[color], lanes[color] = (a, b), lane
        if len(lengths) == len(signals) and len(set(lanes.values())) == len(signals):
            differences = {c: abs(b - a) for c, (a, b) in lengths.items()}
            ranking = sorted(signals, key=lambda c: differences[c], reverse=True)
            if differences[ranking[0]] > differences[ranking[1]]:
                winner = ranking[0]
                delta = differences[winner]
                extent = (right - left - 2) if axis == 0 else (bottom - top - 2)
                if 1 <= delta <= extent:
                    results.append({"axis": axis, "lengths": lengths, "lanes": lanes,
                                    "differences": differences, "winner": winner, "delta": delta,
                                    "sign": int(np.sign(lengths[winner][1] - lengths[winner][0])),
                                    "extent": extent})
    _require(len(results) == 1, "rectangle must have one readable arm axis and unique winner")
    return results[0]


def _band_mask(shape, rect, axis, lengths, winner):
    top, bottom, left, right = rect
    a, b = lengths[winner]
    delta = abs(b - a)
    mask = np.zeros(shape, dtype=bool)
    if axis == 0:
        first = left + 1 if a < b else right - 1 - delta
        mask[top + 1:bottom - 1, first:first + delta] = True
    else:
        first = top + 1 if a < b else bottom - 1 - delta
        mask[first:first + delta, left + 1:right - 1] = True
    return mask


def _analyze(grid, all_reservoirs=False):
    neutral, rect = _largest(grid)
    texture, signals = _roles(grid, neutral)
    info = _read_arms(grid, rect, signals)
    info.update(neutral=neutral, texture=texture, signals=signals, rectangle=rect)
    if all_reservoirs:
        reservoirs = []
        for candidate in sorted(_maximal_rectangles(grid, neutral)):
            try:
                arms = _read_arms(grid, candidate, signals)
            except AssertionError:
                continue
            reservoirs.append(dict(arms, rectangle=candidate))
        _require(any(r["rectangle"] == rect for r in reservoirs), "target absent from reservoirs")
        info["reservoirs"] = reservoirs
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select the largest rectangle, then balance its most unequal arm pair."""
    grid = _grid(input_grid)
    info = _analyze(grid)
    result = grid.copy()
    mask = _band_mask(grid.shape, info["rectangle"], info["axis"],
                      info["lengths"], info["winner"])
    result[mask] = info["winner"]
    return result


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _arm_lengths(rng, count, winner, delta, sign, profile):
    # Every signal occurs on at least one side; no two greatest differences tie.
    base = 0 if profile == "missing_winner" else int(rng.integers(1, 9 - delta))
    winning = (base, base + delta) if sign > 0 else (base + delta, base)
    options = [(a, b) for a in range(9) for b in range(9)
               if a + b > 0 and abs(b - a) < delta]
    lengths = [_choice(rng, options) for _ in range(count)]
    lengths[winner] = winning
    loser = (winner + 1) % count
    if profile == "missing_winner":
        lengths[loser] = (8, 8)  # defeats longest-bar and most-total-color rules
    elif profile == "present_winner":
        lengths[loser] = (0, 1) if sign > 0 else (1, 0)
    return lengths


def _noise(rng, shape, neutral, texture):
    mask = rng.random(shape) < float(rng.uniform(0.20, 0.56))
    style = int(rng.integers(3))
    if style == 1:
        for _ in range(int(rng.integers(2, 8))):
            row = int(rng.integers(shape[0] - 1))
            col = int(rng.integers(shape[1] - 1))
            mask[row:row + 2, col:col + 2] = bool(rng.integers(2))
    elif style == 2:
        for _ in range(int(rng.integers(2, 8))):
            row = int(rng.integers(shape[0]))
            col = int(rng.integers(shape[1]))
            mask[row, col:min(shape[1], col + int(rng.integers(2, 6)))] = True
    return np.where(mask, neutral, texture).astype(np.int16)


def _scene(rng, colors, number, rotation, desired_sign, winner_index, profile,
           excluded_delta=None):
    """Construct horizontally probed reservoirs, then rotate the entire field."""
    neutral, texture, *signals = colors
    count = len(signals)
    for _ in range(400):
        if profile == "missing_winner":
            target_h = int(rng.integers(count + 2, count + 5))
            target_w = int(rng.integers(9, 13))
        elif profile == "present_winner":
            target_h = int(rng.integers(count + 4, count + 7))
            target_w = int(rng.integers(5, 9))
        else:
            target_h = int(rng.integers(count + 2, 11))
            target_w = int(rng.integers(5, 13))
        limit = min(6, target_w - (3 if profile in ("missing_winner", "present_winner") else 2))
        deltas = [d for d in range(2 if profile in ("missing_winner", "present_winner") else 1,
                                  limit + 1) if d != excluded_delta]
        if not deltas:
            continue
        delta = _choice(rng, deltas)
        # Rotation by 90 or 180 degrees reverses the measured first->second sign.
        sign = desired_sign * (-1 if rotation in (1, 2) else 1)
        lengths = _arm_lengths(rng, count, winner_index, delta, sign, profile)
        blocks = [{"h": target_h, "w": target_w, "lengths": lengths, "winner": winner_index}]
        for index in range(number - 1):
            stronger = index == 0
            options = [(h, w) for h in range(count + 2, 12) for w in range(4, 14)
                       if h * w <= 0.84 * target_h * target_w
                       and (not stronger or w - 2 > delta)
                       and (not stronger or profile != "missing_winner" or h > target_h)
                       and (not stronger or profile != "present_winner" or w > target_w)]
            if not options:
                break
            h, w = _choice(rng, options)
            d = int(rng.integers(delta + 1 if stronger else 1, min(8, w - 2) + 1))
            other_winner = _choice(rng, [c for c in range(count) if c != winner_index])
            # delta=8 has only base zero; otherwise decoys can be present or absent.
            if d == 8:
                other_lengths = _arm_lengths(rng, count, other_winner, d,
                                             _choice(rng, [-1, 1]), "missing_winner")
            else:
                other_lengths = _arm_lengths(rng, count, other_winner, d,
                                             _choice(rng, [-1, 1]), "free")
            blocks.append({"h": h, "w": w, "lengths": other_lengths, "winner": other_winner})
        if len(blocks) != number:
            continue
        order = list(range(number))
        rng.shuffle(order)
        gaps = [int(rng.integers(1, 4)) for _ in range(number - 1)]
        margin_top, margin_bottom = int(rng.integers(3)), int(rng.integers(3))
        height = sum(b["h"] for b in blocks) + sum(gaps) + margin_top + margin_bottom
        if height > 30:
            continue
        widths = [b["w"] + max(a for a, _ in b["lengths"]) +
                  max(a for _, a in b["lengths"]) for b in blocks]
        width = max(widths) + int(rng.integers(0, 5))
        if width > 30:
            continue
        grid = _noise(rng, (height, width), neutral, texture)
        row = margin_top
        for place, block_index in enumerate(order):
            block = blocks[block_index]
            left_room = max(a for a, _ in block["lengths"])
            right_room = max(a for _, a in block["lengths"])
            left = int(rng.integers(left_room, width - block["w"] - right_room + 1))
            rect = (row, row + block["h"], left, left + block["w"])
            block["rect"] = rect
            grid[row:row + block["h"], left:left + block["w"]] = neutral
            lanes = rng.choice(np.arange(row + 1, row + block["h"] - 1), count, replace=False)
            for color, lane, (a, b) in zip(signals, lanes, block["lengths"]):
                grid[lane, left - a:left] = color
                grid[lane, left + block["w"]:left + block["w"] + b] = color
            row += block["h"] + (gaps[place] if place < number - 1 else 0)
        target = blocks[0]
        # Compute ground truth directly from sampled geometry, before solver parsing.
        top, bottom, left, right = target["rect"]
        a, b = target["lengths"][winner_index]
        output = grid.copy()
        if a < b:
            output[top + 1:bottom - 1, left + 1:left + 1 + delta] = signals[winner_index]
        else:
            output[top + 1:bottom - 1, right - 1 - delta:right - 1] = signals[winner_index]
        try:
            analysis = _analyze(grid, all_reservoirs=True)
            if (analysis["neutral"] != neutral or analysis["texture"] != texture or
                    analysis["rectangle"] != target["rect"] or
                    {r["rectangle"] for r in analysis["reservoirs"]} != {b["rect"] for b in blocks}):
                continue
            if _ring_neutral(grid, target["rect"], neutral) == 0:
                continue
        except AssertionError:
            continue
        return {"input": np.rot90(grid, rotation).copy(),
                "output": np.rot90(output, rotation).copy()}
    raise RuntimeError("failed to construct readable reservoirs within bounded attempts")


def _ring_neutral(grid, rect, neutral):
    top, bottom, left, right = rect
    ring = np.zeros(grid.shape, dtype=bool)
    ring[max(0, top - 1):min(grid.shape[0], bottom + 1),
         max(0, left - 1):min(grid.shape[1], right + 1)] = True
    ring[top:bottom, left:right] = False
    return int(np.count_nonzero(ring & (grid == neutral)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    count = int(rng.integers(2, 4))
    palette = tuple(int(c) for c in rng.choice(10, count + 2, replace=False))
    mixed_palette = bool(rng.integers(2))
    first_axis = int(rng.integers(2))
    first_sign = _choice(rng, [-1, 1])
    first_winner = int(rng.integers(count))
    second_winner = _choice(rng, [i for i in range(count) if i != first_winner])
    puzzle, seen = {"train": [], "test": []}, set()
    first_delta = None
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            anchor = split == "train" and index < 2
            profile = (("missing_winner", "present_winner")[index] if anchor else "free")
            number = 2 if anchor else (3 if split == "test" else int(rng.integers(1, 3)))
            axis = (first_axis + index) % 2 if anchor else int(rng.integers(2))
            rotation = _choice(rng, [0, 2] if axis == 0 else [1, 3])
            sign = (first_sign if index == 0 else -first_sign) if anchor else _choice(rng, [-1, 1])
            winner = ((first_winner, second_winner)[index] if anchor else int(rng.integers(count)))
            colors = (tuple(int(c) for c in rng.choice(10, count + 2, replace=False))
                      if mixed_palette else palette)
            for _ in range(20):
                pair = _scene(rng, colors, number, rotation, sign, winner, profile,
                              first_delta if anchor and index == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if anchor and index == 0:
                        first_delta = _analyze(pair["input"])["delta"]
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    # No answer order is part of the rule.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    grid, output = _grid(pair["input"]), _grid(pair["output"])
    _require(grid.shape == output.shape, "canvas dimensions must be preserved")
    info = _analyze(grid, all_reservoirs=True)
    rect = info["rectangle"]
    mask = _band_mask(grid.shape, rect, info["axis"], info["lengths"], info["winner"])
    _require(np.array_equal(grid != output, mask), "changes must be exactly the inset band")
    _require(np.all(grid[mask] == info["neutral"]), "band must begin in solid neutral cells")
    _require(np.all(output[mask] == info["winner"]), "wrong band color")
    _require(np.all(output[~mask] == grid[~mask]), "border, noise, and unused reservoirs must be copied")
    _require(np.array_equal(solve(grid), output), "input-only solver disagrees")
    _require(_ring_neutral(grid, rect, info["neutral"]) > 0, "clean artificial halo is excluded")
    reservoirs = info["reservoirs"]
    _require(1 <= len(reservoirs) <= 3, "unexpected reservoir count")
    _require(all(r["axis"] == info["axis"] for r in reservoirs), "arms share an axis within a scene")
    _require(all(_area(r["rectangle"]) <= 0.84 * _area(rect)
                 for r in reservoirs if r["rectangle"] != rect), "competitor area margin is too small")
    # All signal pixels are arms of detected reservoirs, not unrelated colored noise.
    explained = np.zeros(grid.shape, dtype=bool)
    view = explained if info["axis"] == 0 else explained.T
    for r in reservoirs:
        t, b, l, rr = r["rectangle"]
        start, stop = (l, rr) if info["axis"] == 0 else (t, b)
        for c, (a, z) in r["lengths"].items():
            lane = r["lanes"][c]
            view[lane, start - a:start] = True
            view[lane, stop:stop + z] = True
    _require(np.array_equal(explained, np.isin(grid, info["signals"])), "unexplained signal cells")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid example counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(info)
    train = analyses["train"]
    _require({r["axis"] for r in train} == {0, 1}, "demonstrations must teach both orientations")
    _require({r["sign"] for r in train} == {-1, 1}, "demonstrations must teach both imbalance signs")
    partial = [r for r in train if r["delta"] < r["extent"]]
    _require({r["sign"] for r in partial} == {-1, 1}, "direction must be visible in partial bands")
    _require(len({r["delta"] for r in partial}) >= 2, "training must distinguish a fixed band depth")
    _require(any(0 in r["lengths"][r["winner"]] for r in partial), "missing winning arm must be taught")
    _require(any(all(r["lengths"][r["winner"]]) and
                 any(0 in lengths for c, lengths in r["lengths"].items() if c != r["winner"])
                 for r in partial), "present winner and missing loser must be taught")
    _require(all(any(lengths[side] == 0 for r in train for lengths in r["lengths"].values())
                 for side in (0, 1)), "training must show absence on both sides")
    _require(any(max(sum(v) for c, v in r["lengths"].items() if c != r["winner"]) >
                 sum(r["lengths"][r["winner"]]) for r in train), "raw color count shortcut survives")
    larger_cross = larger_axial = False
    for info in train:
        t, b, l, r = info["rectangle"]
        cross, axial = ((b - t, r - l) if info["axis"] == 0 else (r - l, b - t))
        for other in info["reservoirs"]:
            tt, bb, ll, rr = other["rectangle"]
            oc, oa = ((bb - tt, rr - ll) if info["axis"] == 0 else (rr - ll, bb - tt))
            larger_cross |= oc > cross
            larger_axial |= oa > axial
    _require(larger_cross and larger_axial, "dimension-only rectangle shortcuts survive")
    for split in ("train", "test"):
        for info in analyses[split]:
            n = len(info["reservoirs"])
            _require((n <= 2 if split == "train" else n == 3), "count shift is missing")
            if n > 1:
                _require(any(r["delta"] > info["delta"] and r["winner"] != info["winner"]
                             for r in info["reservoirs"] if r["rectangle"] != info["rectangle"]),
                         "needs a stronger, differently colored smaller reservoir")
    _require(len({len(r["signals"]) for s in analyses.values() for r in s}) == 1,
             "signal count convention changes within puzzle")


def self_test():
    """Bounded 42-seed semantic and structural checks; return compact measurements."""
    features = {name: set() for name in ("shapes", "rectangles", "palettes", "lengths", "lanes",
                                         "counts", "delta", "position", "structures", "directions")}
    total = full = zero = edge = adjacent = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s])
        signatures.add(signature)
        if seed == 0:
            repeated = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                         for a, b in zip(puzzle[s], repeated[s]) for k in ("input", "output")),
                     "generation is not deterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                r = _check_pair(pair)
                top, bottom, left, right = r["rectangle"]
                features["shapes"].add(grid.shape)
                features["rectangles"].add((bottom - top, right - left))
                features["palettes"].add((r["neutral"], r["texture"], r["signals"]))
                features["lengths"].add(tuple(sorted(r["lengths"].values())))
                features["lanes"].add(tuple(sorted(r["lanes"].values())))
                features["counts"].add(len(r["reservoirs"]))
                features["delta"].add(r["delta"])
                features["position"].add((2 * top < grid.shape[0] - (bottom - top),
                                           2 * left < grid.shape[1] - (right - left)))
                features["directions"].add((r["axis"], r["sign"]))
                roles = {}
                structure = tuple(roles.setdefault(int(c), len(roles)) for c in grid.flat)
                features["structures"].add((grid.shape, structure))
                total += 1
                full += r["delta"] == r["extent"]
                zero += any(0 in v for v in r["lengths"].values())
                edge += top == 0 or left == 0 or bottom == grid.shape[0] or right == grid.shape[1]
                lanes = sorted(r["lanes"].values())
                adjacent += any(b - a == 1 for a, b in zip(lanes, lanes[1:]))
    _require(len(signatures) == 42, "seeds must yield distinct puzzles")
    for name, minimum in (("shapes", 70), ("rectangles", 40), ("palettes", 35),
                          ("lengths", 100), ("lanes", 40), ("counts", 3), ("delta", 6),
                          ("position", 4), ("directions", 4), ("structures", 200)):
        _require(len(features[name]) >= minimum, "insufficient variation: " + name)
    _require(min(full, zero, edge, adjacent) >= 5, "boundary/missing/adjacent variation is absent")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2042, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": total, "variation": {k: len(v) for k, v in features.items()},
            "full_bands": full, "missing_arm_pairs": zero, "edge_targets": edge,
            "adjacent_lane_pairs": adjacent}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
