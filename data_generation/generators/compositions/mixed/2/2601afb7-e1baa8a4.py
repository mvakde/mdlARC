"""Opposite bar-attribute cycles, followed by exact whole-line run collapse.

No file access, saved examples, or generation state is used by solve().
The touching-slot regime makes the order observable: cycling can create new
identical neighboring column/row vectors which must subsequently coalesce.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRECTIONS = ("up", "down", "left", "right")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if not _integer(value) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _canonical(a, direction):
    # Canonical bars point up. Cross-axis slot order never reverses.
    if direction == "up":
        return a
    if direction == "down":
        return a[::-1]
    if direction == "left":
        return a.T
    return a.T[::-1]


def _restore(a, direction):
    if direction == "up":
        return a.copy()
    if direction == "down":
        return a[::-1].copy()
    if direction == "left":
        return a.T.copy()
    return a[::-1].T.copy()


def _draw(span, cross, direction, bg, colors, lengths, width, gap, before, end):
    a = np.full((span, cross), bg, dtype=np.int8)
    for i, (color, length) in enumerate(zip(colors, lengths)):
        start = before + i * (width + gap)
        a[end-int(length):end, start:start+width] = color
    return _restore(a, direction)


def _collapse(a):
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return a[rows][:, cols].copy()


def _reference_collapse(a):
    # A separately expressed line-run check for constructed answers.
    rr = [0] + [r for r in range(1, len(a)) if tuple(a[r]) != tuple(a[r-1])]
    cc = [0] + [c for c in range(1, a.shape[1])
                if tuple(a[:, c]) != tuple(a[:, c-1])]
    return a[np.ix_(rr, cc)].copy()


def _candidates(a):
    """Recover slots from visible separated strips or equal-width profile runs."""
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        return []
    result = []
    for direction in DIRECTIONS:
        b = _canonical(a, direction)
        active = np.any(b != bg, axis=0)
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        if not len(starts):
            continue
        if len(starts) == 1:
            first, last = int(starts[0]), int(ends[0])
            cuts = [first] + [c for c in range(first+1, last)
                             if not np.array_equal(b[:, c], b[:, c-1])] + [last]
            starts, ends = np.array(cuts[:-1]), np.array(cuts[1:])
            gap = 0
        elif len(starts) >= 3:
            gaps = starts[1:] - ends[:-1]
            if len(set(gaps.tolist())) != 1:
                continue
            gap = int(gaps[0])
        else:
            continue
        widths = ends - starts
        if len(starts) < 3 or len(set(widths.tolist())) != 1:
            continue
        colors, lengths, common_end = [], [], None
        for start, stop in zip(starts, ends):
            block = b[:, start:stop]
            occupied = np.flatnonzero(np.any(block != bg, axis=1))
            if not len(occupied):
                break
            first, last = int(occupied[0]), int(occupied[-1])+1
            color = int(block[first, 0])
            if (first < 1 or last >= b.shape[0] or color == bg
                    or not np.all(block[first:last] == color)
                    or not np.all(block[:first] == bg)
                    or not np.all(block[last:] == bg)
                    or (common_end is not None and last != common_end)):
                break
            common_end = last
            colors.append(color)
            lengths.append(last-first)
        if len(colors) != len(starts) or (gap == 0 and len(set(colors)) < 2):
            continue
        result.append({"direction": direction, "background": bg,
                       "colors": np.array(colors), "lengths": np.array(lengths),
                       "width": int(widths[0]), "gap": gap,
                       "before": int(starts[0]), "end": common_end,
                       "span": b.shape[0], "cross": b.shape[1]})
    return result


def _cycle(info, color_shift=1, length_shift=-1):
    return _draw(info["span"], info["cross"], info["direction"], info["background"],
                 np.roll(info["colors"], color_shift),
                 np.roll(info["lengths"], length_shift), info["width"], info["gap"],
                 info["before"], info["end"])


def _analyze(a):
    possibilities = _candidates(a)
    if not possibilities:
        raise ValueError("input is not an unambiguous regular bar-slot layout")
    outputs = [_collapse(_cycle(info)) for info in possibilities]
    if any(not np.array_equal(outputs[0], other) for other in outputs[1:]):
        raise ValueError("input permits incompatible bar interpretations")
    # Equal lengths can permit both ends as the baseline. Both give the same
    # answer, so no latent orientation flag is needed.
    return possibilities[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _collapse(_cycle(_analyze(a)))


def _levels(rng, count, maximum):
    return rng.choice(np.arange(1, maximum+1), size=count, replace=False).astype(int)


def _attributes(rng, n, maximum, mode, bg, touching):
    palette = [c for c in range(10) if c != bg]
    if mode == "distinct":
        colors = rng.choice(palette, size=n, replace=False)
        lengths = _levels(rng, n, maximum)
        # Avoid a demonstration consistent with merely sorting heights.
        if np.all(np.diff(lengths) > 0) or np.all(np.diff(lengths) < 0):
            lengths[[0, 1]] = lengths[[1, 0]]
        return colors, lengths
    if mode in ("collision", "double_collision"):
        # Specify an interesting post-cycle profile, then invert the two
        # attribute cycles. These are new bar examples, not parent puzzles.
        if mode == "collision":
            labels = [0, 0, 1, 0, 2] + list(range(3, n-2))
        else:
            labels = [0, 0, 1, 2, 2, 3] + list(range(4, n-2))
        k = max(labels)+1
        pool = rng.choice(palette, size=k, replace=False)
        values = _levels(rng, k, maximum)
        out_c, out_l = pool[labels], values[labels]
        # Move the collision around the physical sequence; never assume that
        # the compressible pair is in a fixed slot.
        offset = int(rng.integers(n))
        out_c, out_l = np.roll(out_c, offset), np.roll(out_l, offset)
        return np.roll(out_c, -1), np.roll(out_l, 1)
    colors = rng.choice(rng.choice(palette, size=int(rng.integers(2, min(n, 9)+1)),
                                    replace=False), size=n)
    kind = int(rng.integers(5))
    if kind == 0:
        lengths = _levels(rng, n, maximum)
    elif kind == 1:
        lengths = np.full(n, int(rng.integers(1, maximum+1)))
    elif kind == 2:
        lengths = rng.choice(_levels(rng, 2, maximum), size=n)
    else:
        lengths = rng.integers(1, maximum+1, size=n)
    if touching and len(set(colors.tolist())) < 2:
        colors[-1] = next(c for c in palette if c != int(colors[0]))
    return colors, lengths


def _make_pair(rng, direction, n, touching, mode, scale, anchor=False):
    bg = int(rng.integers(10))
    feasible = [(w, g) for w in range(1, 4) for g in ([0] if touching else range(1, 5))
                if n*w+(n-1)*g+2 <= 30
                and (scale != "large" or w >= 2)
                and (scale != "small" or w <= 2)]
    if anchor and not touching:
        feasible = [(w, g) for w, g in feasible if w > 1 or g > 1]
    width, gap = feasible[int(rng.integers(len(feasible)))]
    used = n*width+(n-1)*gap
    before = int(rng.integers(1, min(5, 29-used)+1))
    after = int(rng.integers(1, min(5, 30-used-before)+1))
    cross = before+used+after
    far = int(rng.integers(1, 5))
    inset = int(rng.integers(1, 4))
    if scale == "large":
        span = int(rng.integers(max(23, n+far+inset), 31))
    elif scale == "small":
        span = int(rng.integers(max(10, n+far+inset), 19))
    else:
        span = int(rng.integers(max(10, n+far+inset), 31))
    maximum = span-far-inset
    colors, lengths = _attributes(rng, n, maximum, mode, bg, touching)
    profiles = list(zip(colors.tolist(), lengths.tolist()))
    if touching and any(a == b for a, b in zip(profiles, profiles[1:])):
        return None
    # In collision modes, rotation can put the only adjacent match across the
    # sequence boundary, which B must not merge. Retry for a visible collision.
    out_profiles = list(zip(np.roll(colors, 1).tolist(), np.roll(lengths, -1).tolist()))
    collisions = sum(a == b for a, b in zip(out_profiles, out_profiles[1:]))
    if touching and mode == "collision" and collisions < 1:
        return None
    if touching and mode == "double_collision" and collisions < 2:
        return None
    end = span-inset
    inp = _draw(span, cross, direction, bg, colors, lengths, width, gap, before, end)
    middle = _draw(span, cross, direction, bg, np.roll(colors, 1), np.roll(lengths, -1),
                   width, gap, before, end)
    out = _reference_collapse(middle)
    if np.array_equal(_reference_collapse(inp), out):
        return None
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    if rng.random() < .5:
        directions = [str(rng.choice(DIRECTIONS))]
    else:
        directions = [str(rng.choice(("up", "down"))), str(rng.choice(("left", "right")))]
        if rng.random() < .5:
            directions.reverse()
    shift = str(rng.choice(("count", "scale", "recombine")))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            if split == "train" and i == 0:
                n, touching, mode = int(rng.integers(3, 6)), False, "distinct"
            elif split == "train" and i == 1:
                n, touching, mode = int(rng.integers(5, 7)), True, "collision"
            elif split == "test" and shift == "count":
                n = int(rng.integers(7, 10))
                touching = bool(rng.integers(2))
                mode = str(rng.choice(("random", "collision", "double_collision"))) if touching else "random"
            elif split == "test" and shift == "recombine":
                n, touching, mode = int(rng.integers(6, 10)), True, "double_collision"
            else:
                n = int(rng.integers(3, 7))
                touching = bool(rng.integers(2))
                mode = "collision" if touching and n >= 5 and rng.random() < .45 else "random"
            direction = directions[i % len(directions)] if i < 2 and split == "train" else str(rng.choice(directions))
            scale = ("small" if split == "train" else "large") if shift == "scale" else "free"
            for _ in range(128):
                pair = _make_pair(rng, direction, n, touching, mode, scale,
                                  anchor=(split == "train" and i == 0))
                if pair is None:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    # Verify that every convention used by solve is visible.
                    if np.array_equal(solve(pair["input"]), pair["output"]):
                        break
            else:
                raise RuntimeError("bounded sampler exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _partial_collapse(a, mode):
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)] if mode in ("rows", "both") else np.ones(a.shape[0], bool)
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)] if mode in ("cols", "both") else np.ones(a.shape[1], bool)
    return a[rows][:, cols]


def _evidence_check(train, infos):
    # These alternatives omit either attribute, reverse a cycle, move whole
    # objects together, or omit either compression axis.
    fits = []
    for cs in (-1, 0, 1):
        for ls in (-1, 0, 1):
            for compression in ("none", "rows", "cols", "both"):
                if all(np.array_equal(_partial_collapse(_cycle(info, cs, ls), compression),
                                      _grid(pair["output"]))
                       for pair, info in zip(train, infos)):
                    fits.append((cs, ls, compression))
    _require(fits == [(1, -1, "both")], f"insufficient directional/axis evidence: {fits}")
    # Equal-length bars have two equivalent baseline interpretations. Do not
    # mistake the parser's arbitrary representative for a new direction that
    # requires another anchor.
    for direction in {info["direction"] for info in infos if len(set(info["lengths"])) > 1}:
        subset = [(pair, info) for pair, info in zip(train, infos) if info["direction"] == direction]
        shifts = [(cs, ls) for cs in (-1, 0, 1) for ls in (-1, 0, 1)
                  if all(np.array_equal(_collapse(_cycle(info, cs, ls)), _grid(pair["output"]))
                         for pair, info in subset)]
        _require(shifts == [(1, -1)], "an orientation lacks attribute-direction evidence")
    before_is_wrong = False
    for pair in train:
        pre = _collapse(_grid(pair["input"]))
        before_is_wrong |= not np.array_equal(_cycle(_analyze(pre)), _grid(pair["output"]))
    _require(before_is_wrong, "training does not distinguish compression before cycling")


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_infos = set(), []
    collisions_in_train = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(a)
            middle = _cycle(info)
            _require(np.array_equal(_reference_collapse(middle), out), "incorrect post-cycle collapse")
            _require(np.array_equal(solve(a), out), "input-only solve disagreement")
            _require(np.array_equal(np.unique(a), np.unique(out)), "palette changed")
            _require(not np.array_equal(_collapse(a), out), "A has no effect")
            _require(middle.shape != out.shape, "B has no effect")
            rr = np.r_[0, np.flatnonzero(np.any(middle[1:] != middle[:-1], axis=1))+1, middle.shape[0]]
            cc = np.r_[0, np.flatnonzero(np.any(middle[:, 1:] != middle[:, :-1], axis=0))+1, middle.shape[1]]
            restored = np.repeat(np.repeat(out, np.diff(rr), axis=0), np.diff(cc), axis=1)
            _require(np.array_equal(restored, middle), "compression is not lossless in line-run coordinates")
            _require(np.array_equal(_collapse(out), out), "output retains adjacent duplicate lines")
            prof = list(zip(np.roll(info["colors"], 1), np.roll(info["lengths"], -1)))
            expected_cross = (2*len(prof)+1 if info["gap"] > 0
                              else 3+sum(a != b for a, b in zip(prof, prof[1:])))
            _require(_canonical(out, info["direction"]).shape ==
                     (len(set(info["lengths"]))+2, expected_cross),
                     "incorrect height-level count or post-cycle profile-run count")
            if split == "train":
                train_infos.append(info)
                if info["gap"] == 0:
                    collisions_in_train |= any(a == b for a, b in zip(prof, prof[1:]))
    _require(any(i["gap"] > 0 for i in train_infos), "no separated-slot demonstration")
    _require(collisions_in_train, "no newly coalescing touching-slot demonstration")
    _evidence_check(puzzle["train"], train_infos)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    # Hand-derived collision example: first two bars merge, but the same full
    # column at a separated third position and both background margins remain.
    a = _draw(10, 12, "up", 0, [1, 2, 1, 3, 1], [7, 2, 2, 4, 2], 2, 0, 1, 9)
    expected = np.array([[0,0,0,0,0,0], [0,0,0,0,3,0], [0,0,2,0,3,0],
                         [0,1,2,1,3,0], [0,0,0,0,0,0]], dtype=np.int8)
    _require(np.array_equal(solve(a), expected), "hand-derived collision failed")
    _require(not np.array_equal(_collapse(a), expected), "hand case fails A necessity")
    _require(not np.array_equal(_cycle(_analyze(_collapse(a))), expected), "hand case fails order necessity")
    for direction in DIRECTIONS:
        _require(np.array_equal(solve(_restore(a, direction)), _restore(expected, direction)),
                 "axis/order convention failed")
    # Solver domain extends beyond the sampler's 3..9 slots and width 1..3.
    # Alternating colors keep all 28 touching input boundaries observable.
    for n, width, gap in ((28, 1, 0), (3, 8, 0), (3, 4, 5)):
        colors = np.array([1+i % 2 for i in range(n)])
        lengths = np.array([1+(3*i) % 7 for i in range(n)])
        cross = n*width+(n-1)*gap+2
        sample = _draw(12, cross, "up", 9, colors, lengths, width, gap, 1, 10)
        answer = _reference_collapse(_draw(12, cross, "up", 9, np.roll(colors, 1),
                                           np.roll(lengths, -1), width, gap, 1, 10))
        _require(np.array_equal(solve(sample), answer), "solver narrower than family constraints")
    shapes, output_shapes, directions, counts, backgrounds, layouts, puzzles = (set() for _ in range(7))
    widths, gaps, length_patterns, palettes = (set() for _ in range(4))
    touching = merged = repeated_lengths = repeated_colors = equal_lengths = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                info = _analyze(a)
                shapes.add(a.shape)
                output_shapes.add(out.shape)
                directions.add(info["direction"])
                counts.add(len(info["colors"]))
                backgrounds.add(info["background"])
                widths.add(info["width"])
                gaps.add(info["gap"])
                length_patterns.add(tuple(info["lengths"]))
                palettes.add(tuple(int(c) for c in np.unique(a)))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                layouts.add((a.shape, hashlib.sha256(normalized).digest()))
                touching += info["gap"] == 0
                repeated_lengths += len(set(info["lengths"])) < len(info["lengths"])
                repeated_colors += len(set(info["colors"])) < len(info["colors"])
                equal_lengths += len(set(info["lengths"])) == 1
                cross_axis = 1 if info["direction"] in ("up", "down") else 0
                merged += info["gap"] == 0 and out.shape[cross_axis] < len(info["colors"])+2
    _require(len(puzzles) == 42 and len(layouts) >= 150, "poor structural diversity")
    _require(len(shapes) >= 50 and len(output_shapes) >= 20, "poor shape diversity")
    _require(directions == set(DIRECTIONS) and counts == set(range(3, 10)), "missing orientation/count")
    _require(backgrounds == set(range(10)) and len(palettes) >= 30, "poor palette role diversity")
    _require(widths == {1, 2, 3} and gaps == set(range(5)), "poor strip geometry diversity")
    _require(merged >= 42 and repeated_lengths >= 42 and repeated_colors >= 42 and equal_lengths > 0,
             "missing collision or tie coverage")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(10000+nt*10+ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "count mismatch")
    for nt, ns in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1),
                   (4,False), (4,1.5), (None,1), (4,None), (4,"1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_digest(generate_puzzle(987, 7, 3)) == _digest(generate_puzzle(987, 7, 3)),
             "explicit-count nondeterminism")
    _require(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)), "default nondeterminism")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "color_normalized_inputs": len(layouts), "palettes": len(palettes),
            "length_patterns": len(length_patterns), "touching_pairs": touching,
            "merged_pairs": merged, "repeated_length_pairs": repeated_lengths,
            "repeated_color_pairs": repeated_colors, "equal_length_pairs": equal_lengths}
