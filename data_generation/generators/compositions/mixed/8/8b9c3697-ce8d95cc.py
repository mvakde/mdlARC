"""Sweep matching rectangular tokens into forks, then contract full-vector runs.

The only newly painted color is fixed black (0).  No puzzle state is learned by
solve.  Geometry is measured before any compression, including token length.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _rect(r, c, h, w):
    return {(i, j) for i in range(r, r + h) for j in range(c, c + w)}


def _is_rectangle(cells):
    rr, cc = zip(*cells)
    return len(cells) == (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)


def _paint(a, cells, color):
    for p in cells:
        a[p] = color


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _parse_fork(cells):
    """Find a one-cell base with three separated positive rectangular spokes.

    The middle spoke, rather than the shortest spoke, defines the stopping tip.
    A centered middle spoke and equally thick guards are family conventions.
    """
    candidates = []
    for outward in _DIRECTIONS:
        perp = -outward[1], outward[0]
        k0 = min(r * outward[0] + c * outward[1] for r, c in cells)
        q0 = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - k0,
                  r * perp[0] + c * perp[1] - q0) for r, c in cells}
        width = max(q for k, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        heights = [sum(k > 0 and x == q for k, x in local) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, n in enumerate(heights) for k in range(1, n + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if heights[q] == 0:
                q += 1
                continue
            start, n = q, heights[q]
            while q < width and heights[q] == n:
                q += 1
            runs.append((start, q, n))
        if (len(runs) != 3 or runs[0][1] >= runs[1][0]
                or runs[1][1] >= runs[2][0]):
            continue
        left, middle, right = runs
        if (middle[0] + middle[1] != width
                or left[1] - left[0] != right[1] - right[0]
                or left[0] != width - right[1]):
            continue
        candidates.append(dict(cells=cells, outward=outward, perp=perp,
                               k0=k0, q0=q0, width=width, runs=tuple(runs),
                               span=middle[1] - middle[0], stop=middle[2] + 1))
    if len(candidates) != 1:
        raise ValueError("receiver needs a unique base and three positive spokes")
    return candidates[0]


def _point(fork, k, q):
    k, q = k + fork["k0"], q + fork["q0"]
    dr, dc = fork["outward"]
    pr, pc = fork["perp"]
    return k * dr + q * pr, k * dc + q * pc


def _candidate(fork, token, a, background):
    local = {(r * fork["outward"][0] + c * fork["outward"][1] - fork["k0"],
              r * fork["perp"][0] + c * fork["perp"][1] - fork["q0"])
             for r, c in token["cells"]}
    qs = set(range(fork["runs"][1][0], fork["runs"][1][1]))
    if {q for k, q in local} != qs:
        return None
    lo, hi = min(k for k, q in local), max(k for k, q in local)
    if lo <= fork["stop"]:
        return None
    route = {_point(fork, k, q) for k in range(fork["stop"], hi + 1) for q in qs}
    if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1])
           or ((r, c) not in token["cells"] and a[r, c] != background)
           for r, c in route):
        return None
    length = hi - lo + 1
    final = {_point(fork, k, q)
             for k in range(fork["stop"], fork["stop"] + length) for q in qs}
    return dict(route=route, final=final, distance=lo - fork["stop"], length=length)


def _analyze(value):
    a = _grid(value)
    values, counts = np.unique(a, return_counts=True)
    if 0 in values or np.count_nonzero(counts == max(counts)) != 1:
        raise ValueError("black is reserved and the background must be uniquely most frequent")
    background = int(values[np.argmax(counts)])
    forks, tokens = [], []
    for color in values:
        if color == background:
            continue
        for cells in _components(a == color):
            obj = dict(cells=cells, color=int(color))
            if _is_rectangle(cells):
                tokens.append(obj)
            else:
                fork = _parse_fork(cells)
                fork["color"] = int(color)
                forks.append(fork)
    if not forks or not tokens:
        raise ValueError("expected receivers and rectangular tokens")
    matches, used, occupied = [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = [(ti, _candidate(fork, token, a, background))
                      for ti, token in enumerate(tokens)]
        candidates = [(ti, m) for ti, m in candidates if m is not None]
        if len(candidates) > 1:
            raise ValueError("multiple matching tokens")
        if candidates:
            ti, match = candidates[0]
            if ti in used or occupied & match["route"]:
                raise ValueError("assignments and swept routes must be disjoint")
            used.add(ti)
            occupied |= match["route"]
            matches.append(dict(match, fi=fi, ti=ti))
    return dict(background=background, forks=forks, tokens=tokens,
                matches=matches, unmatched=len(tokens) - len(used))


def _sweep(value):
    a = _grid(value)
    info = _analyze(a)
    for token in info["tokens"]:
        _paint(a, token["cells"], info["background"])
    for match in info["matches"]:
        _paint(a, match["route"] - match["final"], 0)
        _paint(a, match["final"], info["tokens"][match["ti"]]["color"])
    return a, info


def _keep(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _compress(a):
    # Both masks are computed on the same complete swept grid. Thus every
    # retained crossing is copied verbatim; there is no repainting priority.
    return a[np.ix_(_keep(a, 0), _keep(a, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    swept, _ = _sweep(input_grid)
    return _compress(swept)


def _boxes(rng, h, w, count):
    if count == 1:
        return [(1, 1, h - 2, w - 2)]
    axis = int(rng.integers(2))
    length = (h if axis == 0 else w) - 2
    cut = int(rng.integers(8, length - 8))
    if axis == 0:
        boxes = [(1, 1, cut, w - 2), (cut + 2, 1, h - cut - 3, w - 2)]
    else:
        boxes = [(1, 1, h - 2, cut), (1, cut + 2, h - 2, w - cut - 3)]
    if count == 3:
        # Split the second box in the other axis, keeping eight-cell minima.
        r, c, bh, bw = boxes.pop()
        if axis == 0:
            cut2 = int(rng.integers(8, bw - 8))
            boxes += [(r, c, bh, cut2), (r, c + cut2 + 1, bh, bw - cut2 - 1)]
        else:
            cut2 = int(rng.integers(8, bh - 8))
            boxes += [(r, c, cut2, bw), (r + cut2 + 1, c, bh - cut2 - 1, bw)]
    return [boxes[int(i)] for i in rng.permutation(count)]


def _module(rng, box, outward, active, anchor):
    r, c, bh, bw = box
    cross, depth = (bw, bh) if outward[0] else (bh, bw)
    # Anchors contrast a token across a guard tip with one beyond both guards.
    span = int(rng.integers(2, 4)) if anchor else int(rng.integers(1, 4))
    choices = [(g, gap, tail) for g in (1, 2) for gap in (1, 2, 3) for tail in (0, 1)
               if span + 2 * (g + gap + tail) <= cross]
    guard, gap, tail = choices[int(rng.integers(len(choices)))]
    width = span + 2 * (guard + gap + tail)
    center_q = tail + guard + gap
    if anchor:
        length = 2
        center = int(rng.integers(1, min(3, depth - 6) + 1))
        if anchor == 1:
            left, right = center + 1, center + 2
        else:
            center = max(2, center)
            left, right = map(int, rng.integers(1, center, size=2))
        if rng.integers(2):
            left, right = right, left
        start = center + 4
    else:
        length = int(rng.integers(1, min(3, depth - 5) + 1))
        cap = min(6, depth - length - 3)
        left, center, right = map(int, rng.integers(1, cap + 1, size=3))
        start = max(left, center, right) + 2
        if active:
            start += int(rng.integers(0, depth - length - start + 1))
    structure = {(0, q) for q in range(width)}
    for qa, qb, n in ((tail, tail + guard, left), (center_q, center_q + span, center),
                       (width - tail - guard, width - tail, right)):
        structure |= {(k, q) for k in range(1, n + 1) for q in range(qa, qb)}
    original = _rect(start, center_q, length, span) if active else set()
    final = _rect(center + 1, center_q, length, span) if active else set()
    route = _rect(center + 1, center_q, start + length - center - 1, span) if active else set()
    reserve = structure | route
    perp = -outward[1], outward[0]

    def raw(cells):
        return {(k * outward[0] + q * perp[0], k * outward[1] + q * perp[1]) for k, q in cells}

    points = raw(reserve)
    mr, mc = min(x for x, y in points), min(y for x, y in points)
    height = max(x for x, y in points) - mr + 1
    wide = max(y for x, y in points) - mc + 1
    rr = r + int(rng.integers(bh - height + 1))
    cc = c + int(rng.integers(bw - wide + 1))

    def placed(cells):
        return {(x - mr + rr, y - mc + cc) for x, y in raw(cells)}

    return dict(structure=placed(structure), original=placed(original),
                final=placed(final), route=placed(route), outward=outward)


def _new_equalities(before, after, axis):
    return int(np.count_nonzero(_keep(before, axis)[1:] & ~_keep(after, axis)[1:]))


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _make_pair(rng, split, index, background, directions):
    anchor = split == "train" and index < 2
    count = 1 if anchor else (int(rng.integers(1, 3)) if split == "train"
                              else int(rng.integers(2, 4)))
    h = int(rng.integers(14, 26)) if count == 1 else int(rng.integers(23, 31))
    w = int(rng.integers(14, 26)) if count == 1 else int(rng.integers(23, 31))
    palette = list(map(int, rng.permutation([x for x in range(1, 10) if x != background])))
    fcolors, tcolors = palette[:3], palette[3:6]
    for attempt in range(40):
        boxes = _boxes(rng, h, w, count)
        if anchor:
            outs = [directions[index]]
        elif split == "train":
            # A scene contains parallel motions; tests combine both axes.
            axis = int(rng.integers(2))
            outs = [_DIRECTIONS[axis + 2 * int(rng.integers(2))] for _ in range(count)]
        else:
            outs = [directions[0], directions[1]]
            outs += [_DIRECTIONS[int(rng.integers(4))] for _ in range(count - 2)]
            rng.shuffle(outs)
        active = [True] * count
        if split == "train" and not anchor and count == 2:
            active[1] = bool(rng.integers(2))
        try:
            modules = [_module(rng, box, out, act, index + 1 if anchor else 0)
                       for box, out, act in zip(boxes, outs, active)]
        except ValueError:
            continue
        a = np.full((h, w), background, dtype=np.int8)
        swept = a.copy()
        blocked = set()
        for i, m in enumerate(modules):
            _paint(a, m["structure"], fcolors[i])
            _paint(swept, m["structure"], fcolors[i])
            _paint(a, m["original"], tcolors[i])
            _paint(swept, m["route"] - m["final"], 0)
            _paint(swept, m["final"], tcolors[i])
            blocked |= _halo(m["structure"] | m["route"])
        try:
            info = _analyze(a)
        except ValueError:
            continue
        if len(info["matches"]) != sum(active):
            continue
        # Geometric distractors have no role-specific color: even the matched
        # token's color is reused. They cannot obstruct a reserved swept route.
        for di in range(int(rng.integers(2, 5))):
            dh, dw = map(int, rng.integers(1, 4, size=2))
            positions = []
            for rr in range(1, h - dh):
                for cc in range(1, w - dw):
                    cells = _rect(rr, cc, dh, dw)
                    if blocked & cells:
                        continue
                    token = dict(cells=cells)
                    if any(_candidate(f, token, a, background) is not None for f in info["forks"]):
                        continue
                    positions.append(cells)
            if not positions:
                break
            cells = positions[int(rng.integers(len(positions)))]
            _paint(a, cells, tcolors[di % 3])
            blocked |= _halo(cells)
        try:
            predicted, info = _sweep(a)
        except ValueError:
            continue
        if not np.array_equal(predicted, swept) or info["unmatched"] < 2:
            continue
        # Demand causal redundancy, not just empty-margin trimming.
        if not any(_new_equalities(a, swept, ax) for ax in (0, 1)):
            continue
        output = _compress(swept)
        if not all(x < y for x, y in zip(output.shape, a.shape)):
            continue
        if anchor:
            motion_axis = 0 if outs[0][0] else 1
            if not _new_equalities(a, swept, motion_axis):
                continue
            # First demo preserves two different final slices at a guard tip;
            # second collapses identical final slices beyond both guards.
            match = info["matches"][0]
            coords = sorted({p[motion_axis] for p in match["final"]})
            retained = sum(bool(_keep(swept, motion_axis)[v]) for v in coords)
            if retained != (2 if index == 0 else 1):
                continue
        return dict(input=a, output=output)
    raise RuntimeError("bounded construction exhausted")


def _count(value, low, high):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and low <= int(value) <= high)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _count(num_train, 2, 10) or not _count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.permutation(np.arange(1, 10))))
    # These directions are orthogonal, but either sign and order is possible.
    vertical = _DIRECTIONS[2 * int(rng.integers(2))]
    horizontal = _DIRECTIONS[1 + 2 * int(rng.integers(2))]
    directions = [vertical, horizontal]
    rng.shuffle(directions)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(int(count)):
            background = backgrounds[(i + (num_train if split == "test" else 0)) % 9]
            for _ in range(8):
                pair = _make_pair(rng, split, i, background, directions)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
    return puzzle


def _run_lengths(a, axis):
    starts = np.flatnonzero(_keep(a, axis))
    return tuple(map(int, np.diff(np.r_[starts, a.shape[axis]])))


def _separated_repeat(a, axis):
    vectors = a if axis == 0 else a.T
    return any(np.array_equal(vectors[i], vectors[j])
               and not np.all(vectors[i:j + 1] == vectors[i])
               for i in range(len(vectors)) for j in range(i + 2, len(vectors)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert _count(len(puzzle["train"]), 2, 10) and _count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_axes = set(), set()
    for split in ("train", "test"):
        for i, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            swept, info = _sweep(a)
            assert info["matches"] and info["unmatched"] >= 2
            assert np.array_equal(output, _compress(swept))
            assert all(x < y for x, y in zip(output.shape, a.shape))
            # Literal-value preservation and lossless expansion of this final
            # quotient establish that full-vector runs were the only removal.
            runs_r, runs_c = _run_lengths(swept, 0), _run_lengths(swept, 1)
            assert np.array_equal(np.repeat(np.repeat(output, runs_r, 0), runs_c, 1), swept)
            assert not np.any(~_keep(output, 0)) and not np.any(~_keep(output, 1))
            assert _separated_repeat(output, 0) and _separated_repeat(output, 1)
            assert any(_new_equalities(a, swept, ax) for ax in (0, 1))
            assert not np.array_equal(_compress(a), output), "omitting A must change the answer"
            assert not np.array_equal(swept, output), "omitting B must change the answer"
            occupied = set()
            for obj in info["forks"] + info["tokens"]:
                assert not occupied & _halo(obj["cells"])
                occupied |= obj["cells"]
            for fork in info["forks"]:
                assert min(run[2] for run in fork["runs"]) >= 1
                assert all(swept[p] == fork["color"] for p in fork["cells"])
            for m in info["matches"]:
                assert all(swept[p] == info["tokens"][m["ti"]]["color"] for p in m["final"])
                assert all(swept[p] == 0 for p in m["route"] - m["final"])
            axes = {0 if info["forks"][m["fi"]]["outward"][0] else 1 for m in info["matches"]}
            if split == "train":
                assert len(axes) == 1
                train_axes |= axes
                if i < 2:
                    assert len(info["forks"]) == 1
                    motion_axis = next(iter(axes))
                    assert _new_equalities(a, swept, motion_axis)
                    m = info["matches"][0]
                    coords = {p[motion_axis] for p in m["final"]}
                    assert len(coords) == 2
                    retained = sum(bool(_keep(swept, motion_axis)[c]) for c in coords)
                    assert retained == (2 if i == 0 else 1)
                    left, center, right = (run[2] for run in info["forks"][0]["runs"])
                    assert (center < min(left, right) if i == 0 else center > max(left, right))
            else:
                assert 2 <= len(info["forks"]) == len(info["matches"]) <= 3
                assert axes == {0, 1}
    assert train_axes == {0, 1}


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed metrics; no disk writes and no retained grid corpus."""
    shapes, outputs, geometries, palettes, fingerprints, layouts = (set() for _ in range(6))
    directions, counts, spans, lengths = (set() for _ in range(4))
    inactive = long_center = order_witnesses = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                shapes.add(a.shape)
                outputs.add(pair["output"].shape)
                palettes.add(tuple(map(int, np.unique(a))))
                layouts.add(hashlib.sha256(a.tobytes()).hexdigest())
                counts.add(len(info["forks"]))
                inactive += len(info["forks"]) - len(info["matches"])
                for fork in info["forks"]:
                    directions.add(fork["outward"])
                    geometries.add((fork["width"], fork["runs"]))
                    spans.add(fork["span"])
                    left, center, right = (run[2] for run in fork["runs"])
                    long_center += center > max(left, right)
                lengths.update(m["length"] for m in info["matches"])
                # Counterfactual deliberately compresses first. Some compressed
                # scenes cease to be admissible forks; count only valid parses.
                try:
                    wrong = _compress(_sweep(_compress(a))[0])
                except ValueError:
                    pass
                else:
                    order_witnesses += not np.array_equal(wrong, pair["output"])
                pairs += 1
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(outputs) >= 15
    assert len(geometries) >= 60 and len(palettes) >= 20 and len(layouts) == pairs
    assert len(directions) == 4 and counts == {1, 2, 3} and spans == {1, 2, 3}
    assert lengths == {1, 2, 3} and inactive > 0 and long_center > 0 and order_witnesses >= 20
    for nt in range(2, 11):
        for ne in range(1, min(nt - 1, 4) + 1):
            puzzle = generate_puzzle(100 + 10 * nt + ne, nt, ne)
            validate_generated_puzzle(puzzle)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _fingerprint(generate_puzzle(923, 5, 2)) == _fingerprint(generate_puzzle(923, 5, 2))
    # Reject valid-schema but incorrect labels, and a zero-length center.
    bad = generate_puzzle(3, 2, 1)
    bad["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    try:
        _parse_fork(_rect(0, 0, 1, 7) | _rect(1, 0, 3, 1) | _rect(1, 6, 3, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("missing central spoke accepted")
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes), output_shapes=len(outputs),
                fork_geometries=len(geometries), palettes=len(palettes),
                inactive_receivers=int(inactive), long_center_receivers=int(long_center),
                valid_compress_first_counterexamples=int(order_witnesses))
