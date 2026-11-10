"""Vacancy-sized canvases containing occurrence-rank square nests.

Only standard-library code and NumPy are used. Generation and self-testing do
not write files. The solver knows neither the seed nor the placement recipe.
"""

from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.uint8).tobytes()).hexdigest()


def _runs(row):
    start = None
    for x, value in enumerate(row):
        if value and start is None:
            start = x
        elif not value and start is not None:
            yield start, x
            start = None
    if start is not None:
        yield start, len(row)


def _rectangles(mask):
    """Enumerate maximal horizontal runs for each span of >=2 rows.

    Every maximum-area eligible rectangle occurs here: a horizontally
    extendible rectangle cannot be a maximum. No candidate list is retained.
    """
    h, w = mask.shape
    for top in range(h - 1):
        common = mask[top].copy()
        for bottom in range(top + 1, h):
            common &= mask[bottom]
            if not common.any():
                break
            for left, right in _runs(common):
                if right - left >= 2:
                    yield top, left, bottom - top + 1, right - left


def _largest(mask):
    best, winners = 0, []
    for rect in _rectangles(mask):
        area = rect[2] * rect[3]
        if area > best:
            best, winners = area, [rect]
        elif area == best:
            winners.append(rect)
    if len(winners) != 1:
        raise ValueError("background must have one unique largest eligible rectangle")
    return winners[0]


def _components(a, background):
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for y in range(h):
        for x in range(w):
            color = int(a[y, x])
            if seen[y, x] or color == background:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            top = bottom = y
            left = right = x
            count = 0
            while stack:
                r, c = stack.pop()
                count += 1
                top, bottom = min(top, r), max(bottom, r)
                left, right = min(left, c), max(right, c)
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= rr < h and 0 <= cc < w and not seen[rr, cc]
                            and a[rr, cc] == color):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append((color, top, left, bottom - top + 1, right - left + 1, count))
    return result


def _facts(value):
    a = _grid(value)
    histogram = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(histogram))
    if np.count_nonzero(histogram == histogram[bg]) != 1:
        raise ValueError("background must be the uniquely most frequent color")
    rect = _largest(a == bg)
    classes, squares, other_colors = {}, [], set()
    for color, y, x, h, w, n in _components(a, bg):
        if h == w and h >= 2 and n == h * w:
            if h in classes and classes[h][0] != color:
                raise ValueError("all squares of one side must share a color")
            if h not in classes:
                classes[h] = [color, 0]
            classes[h][1] += 1
            squares.append((h, color, y, x))
        else:
            other_colors.add(color)
    if not classes or len({v[0] for v in classes.values()}) != len(classes):
        raise ValueError("square classes require distinct colors")
    if other_colors & {v[0] for v in classes.values()}:
        raise ValueError("payload colors must occur only in complete square components")
    ranks = [tuple(s for s in sorted(classes, reverse=True) if classes[s][1] >= rank)
             for rank in range(1, max(v[1] for v in classes.values()) + 1)]
    width = sum(active[0] + 1 for active in ranks)
    if rect[2] < max(classes) or rect[3] < width:
        raise ValueError("the selected vacancy cannot hold the complete rank construction")
    return {"background": bg, "rect": rect, "classes": classes,
            "ranks": ranks, "packed_width": width, "squares": squares,
            "noise_colors": other_colors, "shape": a.shape}


def _render(facts, *, ignore_threshold=False, shape=None):
    classes = facts["classes"]
    h, w = facts["rect"][2:] if shape is None else shape
    out = np.full((h, w), facts["background"], dtype=np.int8)
    cursor = 0
    for active in facts["ranks"]:
        if ignore_threshold:
            active = tuple(sorted(classes, reverse=True))
        for side in active:
            out[:side, cursor:cursor + side] = classes[side][0]
        cursor += active[0] + 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select the input-local vacancy, then draw the visible square inventory."""
    return _render(_facts(input_grid))


def _origins(blocked, h, w):
    """All free h-by-w windows, via a tiny integral image."""
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    integral = np.pad(blocked.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    return np.argwhere(sums == 0)


def _mark(mask, y, x, h, w, halo=0):
    mask[max(0, y - halo):min(mask.shape[0], y + h + halo),
         max(0, x - halo):min(mask.shape[1], x + w + halo)] = True


def _expand(mask):
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[y:y + mask.shape[0], x:x + mask.shape[1]]
                                 for y in range(3) for x in range(3)])


def _position(rng, limit):
    if limit == 0:
        return 0
    mode = int(rng.integers(4))
    return 0 if mode == 0 else limit if mode == 1 else int(rng.integers(limit + 1))


def _inventory(rng, stage, first_small):
    if stage == 0:
        return (2, 3), (2, 1) if first_small else (1, 2)
    if stage == 1:
        sides = tuple(sorted(int(s) for s in rng.choice(np.arange(2, 6), 3, replace=False)))
        counts = [int(n) for n in rng.integers(1, 3, 3)]
        leader = int(rng.choice((1, 2) if first_small else (0, 1)))
        counts[leader] = 3
        return sides, tuple(counts)
    nclass = int(rng.choice((2, 3, 4), p=(0.3, 0.45, 0.25)))
    sides = tuple(sorted(int(s) for s in rng.choice(np.arange(2, 6), nclass, replace=False)))
    counts = tuple(int(n) for n in rng.integers(1, 5, nclass))
    return sides, counts


def _texture(rng, shape, bg, noise):
    a = np.full(shape, bg, dtype=np.int8)
    occupied = rng.random(shape) < float(rng.uniform(0.32, 0.49))
    a[occupied] = rng.choice(noise, size=int(occupied.sum()))
    # A filled noise 2x2 could masquerade as inventory. Delete a corner of
    # every such block; deletion cannot create another uniform noise block.
    for color in noise:
        for y in range(shape[0] - 1):
            for x in range(shape[1] - 1):
                if a[y, x] == color and np.all(a[y:y + 2, x:x + 2] == color):
                    a[y + 1, x + 1] = bg
    return a


def _make_pair(rng, parameters, stage, forbidden_shapes, forbidden_profiles):
    for _ in range(160):
        sides, counts = _inventory(rng, stage, parameters["first_small"])
        profile = tuple(zip(sides, counts))
        if profile in forbidden_profiles:
            continue
        ranks = [max(s for s, n in profile if n >= r) for r in range(1, max(counts) + 1)]
        packed = sum(s + 1 for s in ranks)
        if packed > 25:
            continue
        if stage == 0:
            ch, cw = 3, int(rng.integers(packed, 10))
            horizontal = bool(rng.integers(2))
            ih, iw = (int(rng.integers(18, 31)), 30) if horizontal else (30, int(rng.integers(18, 31)))
        else:
            ch = max(sides) + int(rng.integers(1, 5) if stage == 1 else rng.integers(0, 7))
            cw = packed + int(rng.integers(1, 6) if stage == 1 else rng.integers(0, 7))
            if cw > 28 or (ch, cw) in forbidden_shapes:
                continue
            ih = int(rng.integers(max(18, ch + 8), 31))
            iw = int(rng.integers(max(18, cw), 31))
        square_area = sum(s * s * n for s, n in profile)
        if square_area > 0.29 * ih * iw or square_area + ch * cw > 0.59 * ih * iw:
            continue
        y, x = _position(rng, ih - ch), _position(rng, iw - cw)
        rect = (y, x, ch, cw)
        protected = np.zeros((ih, iw), dtype=bool)
        _mark(protected, *rect)
        # An attached spike makes connected-background cropping observably
        # different from rectangle selection. Longer strips are optional.
        attachments = []
        if y:
            attachments.append((y - 1, x + int(rng.integers(cw))))
        if y + ch < ih:
            attachments.append((y + ch, x + int(rng.integers(cw))))
        if x:
            attachments.append((y + int(rng.integers(ch)), x - 1))
        if x + cw < iw:
            attachments.append((y + int(rng.integers(ch)), x + cw))
        if stage in (0, 1) or rng.random() < 0.65:
            r, c = attachments[int(rng.integers(len(attachments)))]
            protected[r, c] = True
        if stage == 0:
            # A close-area loser has the larger extent along the long axis.
            rh, rw = (2, (ch * cw - 1) // 2) if horizontal else ((ch * cw - 1) // 2, 2)
            positions = _origins(_expand(protected), rh, rw)
            if not len(positions):
                continue
            r, c = positions[int(rng.integers(len(positions)))]
            _mark(protected, int(r), int(c), rh, rw)
            length = int(rng.integers(ch * cw + 1, 31))
            lh, lw = (1, length) if horizontal else (length, 1)
            positions = _origins(_expand(protected), lh, lw)
            if not len(positions):
                continue
            r, c = positions[int(rng.integers(len(positions)))]
            _mark(protected, int(r), int(c), lh, lw)
        elif rng.random() < 0.55:
            shapes = [(h, w) for h in range(2, min(ih, 10) + 1)
                      for w in range(2, min(iw, 16) + 1)
                      if ch * cw * 0.5 <= h * w < ch * cw and h * w < ih * iw * 0.18]
            rng.shuffle(shapes)
            for rh, rw in shapes[:12]:
                positions = _origins(_expand(protected), rh, rw)
                if len(positions):
                    r, c = positions[int(rng.integers(len(positions)))]
                    _mark(protected, int(r), int(c), rh, rw)
                    break
        palette = parameters["palette"] if parameters["stable"] else [int(v) for v in rng.permutation(10)]
        bg = palette[0]
        noise = [palette[1]] if rng.random() < 0.55 else [palette[1], palette[6]]
        colors = {s: palette[s] for s in sides}
        blocked = protected.copy()
        placements = []
        items = sorted([s for s, n in profile for _ in range(n)], reverse=True)
        failed = False
        for side in items:
            positions = _origins(blocked, side, side)
            if not len(positions):
                failed = True
                break
            if rng.random() < 0.38:
                edge = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0)
                                 | (positions[:, 0] + side == ih) | (positions[:, 1] + side == iw)]
                if len(edge):
                    positions = edge
            r, c = positions[int(rng.integers(len(positions)))]
            r, c = int(r), int(c)
            placements.append((r, c, side, colors[side]))
            _mark(blocked, r, c, side, side, halo=1)
        if failed:
            continue
        a = _texture(rng, (ih, iw), bg, noise)
        a[protected] = bg
        for r, c, side, color in placements:
            a[r:r + side, c:c + side] = color
        try:
            facts = _facts(a)
        except ValueError:
            continue
        if facts["rect"] != rect or facts["classes"] != {s: [colors[s], n] for s, n in profile}:
            continue
        features = _features(a, facts)
        if stage == 0 and not (features["long_line"] and features["close_rival"]
                               and features["connected_diff"]):
            continue
        return {"input": a, "output": _render(facts)}, facts
    raise RuntimeError("bounded layout attempts exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    parameters = {"palette": [int(v) for v in rng.permutation(10)],
                  "stable": bool(rng.integers(2)), "first_small": bool(rng.integers(2))}
    puzzle = {"train": [], "test": []}
    seen, train_shapes, train_profiles = set(), set(), set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            stage = index if split == "train" and index < 2 else 2
            forbidden_shapes = train_shapes if split == "test" else set()
            forbidden_profiles = train_profiles if split == "test" else set()
            for _ in range(8):
                pair, facts = _make_pair(rng, parameters, stage, forbidden_shapes, forbidden_profiles)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_shapes.add(facts["rect"][2:])
                        train_profiles.add(tuple((s, v[1]) for s, v in sorted(facts["classes"].items())))
                    break
            else:
                raise RuntimeError("bounded distinct-input attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _background_component(mask, start):
    seen, stack = {start}, [start]
    h, w = mask.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and (yy, xx) not in seen:
                seen.add((yy, xx))
                stack.append((yy, xx))
    ys, xs = zip(*seen)
    return len(seen), (max(ys) - min(ys) + 1, max(xs) - min(xs) + 1)


def _features(a, facts):
    bg, (y, x, h, w) = facts["background"], facts["rect"]
    mask = a == bg
    rival = 0
    overlapping = False
    for yy, xx, hh, ww in _rectangles(mask):
        contained = y <= yy and x <= xx and yy + hh <= y + h and xx + ww <= x + w
        if not contained:
            rival = max(rival, hh * ww)
            overlapping |= yy < y + h and y < yy + hh and xx < x + w and x < xx + ww
    remaining = mask.copy()
    remaining[y:y + h, x:x + w] = False
    longest = max([r - l for row in remaining for l, r in _runs(row)]
                  + [r - l for row in remaining.T for l, r in _runs(row)] + [0])
    size, component_shape = _background_component(mask, (y, x))
    return {"long_line": longest > h * w, "close_rival": rival >= 0.70 * h * w,
            "overlap": bool(overlapping), "connected_diff": component_shape != (h, w),
            "component_area": size, "component_shape": component_shape,
            "edge_square": any(yy == 0 or xx == 0 or yy + s == a.shape[0] or xx + s == a.shape[1]
                               for s, _, yy, xx in facts["squares"])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, train_shapes, train_profiles, training = set(), set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts = _facts(a)
            classes, bg = facts["classes"], facts["background"]
            _require(2 <= len(classes) <= 4 and all(2 <= s <= 5 for s in classes), "size classes")
            _require(all(1 <= v[1] <= 4 for v in classes.values()), "multiplicities")
            _require(1 <= len(facts["noise_colors"]) <= 2, "noise convention")
            for color in facts["noise_colors"]:
                m = a == color
                _require(not np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]),
                         "noise must not contain a filled 2-by-2 block")
            _require(out.shape == facts["rect"][2:], "canvas must come from the selected rectangle")
            # Check each tile cell, its separator, and unused canvas directly.
            # The smallest covering eligible side determines a pixel's color.
            cursor = 0
            accounted = np.zeros(out.shape, dtype=bool)
            for active in facts["ranks"]:
                tile_side = active[0]
                for row in range(tile_side):
                    for column in range(tile_side):
                        side = min(s for s in active if row < s and column < s)
                        _require(out[row, cursor + column] == classes[side][0], "incorrect nesting/rank color")
                accounted[:tile_side, cursor:cursor + tile_side] = True
                _require(np.all(out[:, cursor + tile_side] == bg), "missing separator, including last")
                cursor += tile_side + 1
            _require(np.all(out[~accounted] == bg), "unused canvas must remain blank")
            _require(set(np.unique(out)) == {bg} | {v[0] for v in classes.values()}, "payload visibility/noise leak")
            profile = tuple((s, v[1]) for s, v in sorted(classes.items()))
            if split == "train":
                train_shapes.add(out.shape)
                train_profiles.add(profile)
                training.append((facts, _features(a, facts), out))
            else:
                _require(out.shape not in train_shapes and profile not in train_profiles, "missing test shift")
    _require(len(train_profiles) >= 2 and len(train_shapes) >= 2, "demonstrations must vary inventory and canvas")
    _require(any(f["long_line"] and f["close_rival"] and f["connected_diff"]
                 for _, f, _ in training), "missing discriminating rectangle demonstration")
    _require(any(not np.array_equal(out, _render(facts, ignore_threshold=True)) for facts, _, out in training),
             "rank threshold must affect a demonstration")
    _require(any(out.shape != (max(facts["classes"]) + 1, facts["packed_width"])
                 for facts, _, out in training), "inventory alone must not determine canvas dimensions")
    _require(any(len(facts["classes"]) >= 3 for facts, _, _ in training), "training must demonstrate three layers")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded checks: 42 seeds, edge counts, negative calls and metamorphisms.

    Returns small aggregate measurements; no grids or traces are retained.
    These are generator-owned checks, not external semantic adjudication.
    """
    shapes, canvases, profiles, layouts, backgrounds, class_counts = (set() for _ in range(6))
    observed = {"seeds": 42, "pairs": 0, "long_line": 0, "close_rival": 0,
                "connected_diff": 0, "overlap": 0, "edge_square": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                facts = _facts(a)
                _require(np.array_equal(solve(a), out), "solve mismatch")
                shapes.add(a.shape)
                canvases.add(out.shape)
                profiles.add(tuple((s, v[1]) for s, v in sorted(facts["classes"].items())))
                layouts.add(tuple(facts["squares"]))
                backgrounds.add(facts["background"])
                class_counts.add(len(facts["classes"]))
                observed["pairs"] += 1
                for name, value in _features(a, facts).items():
                    if name in observed:
                        observed[name] += int(value)
        # Relabeling every color must commute with solving, including bg.
        permutation = np.random.default_rng(seed + 10000).permutation(10)
        pair = puzzle["test"][0]
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "color-role equivariance failed")
        # Horizontal input reflection preserves inventory and vacancy size.
        _require(np.array_equal(solve(pair["input"][:, ::-1]), pair["output"]), "position dependence")
    for nt, nq in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(901 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a = generate_puzzle(77, 2, 1)
    _require(_same(a, generate_puzzle(77, 2, 1)), "nondeterminism")
    _require(not _same(a, generate_puzzle(78, 2, 1)), "seed has no effect")
    # Explicitly reject tied eligible maxima; never silently break ties.
    mask = np.zeros((4, 8), dtype=bool)
    mask[:2, :2] = True
    mask[2:, 6:] = True
    try:
        _largest(mask)
    except ValueError:
        pass
    else:
        raise AssertionError("area tie accepted")
    _require(len(shapes) >= 30 and len(canvases) >= 40 and len(profiles) >= 50, "insufficient geometry/inventory variation")
    _require(len(layouts) == 210 and backgrounds == set(range(10)) and class_counts == {2, 3, 4}, "insufficient layout/palette variation")
    _require(all(observed[k] >= 20 for k in ("long_line", "close_rival", "connected_diff", "edge_square")), "weak concept witnesses")
    observed.update({"input_shapes": len(shapes), "canvas_shapes": len(canvases),
                     "inventory_profiles": len(profiles), "layouts": len(layouts),
                     "background_colors": len(backgrounds), "class_counts": sorted(class_counts)})
    return observed
