"""Largest physical blank rectangle mapped through two-axis equality-run collapse.

Only NumPy and the standard library are used. No files, traces, or caches are
written. Background=0, highlight=2; the other input color is a blocker.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


BACKGROUND = 0
HIGHLIGHT = 2
BLOCKERS = (1, 3, 4, 5, 6, 7, 8, 9)


def _integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test) -> tuple[int, int]:
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be in 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors in 0..9")
    return a


def _segments(values):
    """Half-open maximal true runs; never remove separated recurrences."""
    edges = np.flatnonzero(np.diff(np.r_[False, values, False]))
    return zip(edges[::2], edges[1::2])


def _bands(a):
    """Independent maps are computed from complete vectors in the raw input."""
    rs = np.r_[0, np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1]
    cs = np.r_[0, np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0)) + 1]
    re = np.r_[rs[1:], a.shape[0]]
    ce = np.r_[cs[1:], a.shape[1]]
    return a[np.ix_(rs, cs)].copy(), rs, cs, re - rs, ce - cs


def _physical_winner(a):
    """Search the original pixels, not the compressed image.

    For every row span of height >=2, only maximal common-zero column spans
    need consideration: any shorter one can be widened and cannot win.
    """
    best_area, winners = 0, []
    for top in range(a.shape[0] - 1):
        common = a[top] == BACKGROUND
        for bottom in range(top + 2, a.shape[0] + 1):
            common = common & (a[bottom - 1] == BACKGROUND)
            for left, right in _segments(common):
                if right - left < 2:
                    continue
                area = (bottom - top) * int(right - left)
                rectangle = (top, bottom, int(left), int(right))
                if area > best_area:
                    best_area, winners = area, [rectangle]
                elif area == best_area:
                    winners.append(rectangle)
    if len(winners) != 1:
        raise ValueError("input must have a unique largest eligible zero rectangle")
    return best_area, winners[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output solely from the input and fixed family conventions."""
    a = _grid(input_grid)
    colors = set(int(x) for x in np.unique(a))
    if len(colors) != 2 or BACKGROUND not in colors or HIGHLIGHT in colors:
        raise ValueError("input must use zero and one blocker color other than two")
    _, (top, bottom, left, right) = _physical_winner(a)
    out, rs, cs, _, _ = _bands(a)
    # Image of the rectangle under the two monotone coordinate quotient maps.
    # Intersection, rather than a side>=2 test, is used in output coordinates.
    r0 = int(np.searchsorted(rs, top, side="right") - 1)
    r1 = int(np.searchsorted(rs, bottom - 1, side="right"))
    c0 = int(np.searchsorted(cs, left, side="right") - 1)
    c1 = int(np.searchsorted(cs, right - 1, side="right"))
    out[r0:r1, c0:c1] = HIGHLIGHT
    return out


def _weighted_rectangles(core, rw, cw):
    """Sampler's coordinate-space search; does not call the pixel solver.

    A maximal rectangle includes whole equality bands. Weighted band sums
    give its original dimensions, even for a one-cell compressed footprint.
    """
    rr, cc = np.r_[0, np.cumsum(rw)], np.r_[0, np.cumsum(cw)]
    candidates = []
    for top in range(core.shape[0]):
        common = np.ones(core.shape[1], dtype=bool)
        for bottom in range(top + 1, core.shape[0] + 1):
            common &= core[bottom - 1] == BACKGROUND
            height = int(rr[bottom] - rr[top])
            if height < 2:
                continue
            for left, right in _segments(common):
                width = int(cc[right] - cc[left])
                if width >= 2:
                    candidates.append((height * width, top, bottom, int(left), int(right)))
    candidates.sort(reverse=True)
    return candidates


def _returning(vectors) -> bool:
    # Call on an already run-compressed axis. Duplicates here are separated.
    return len({tuple(int(x) for x in row) for row in vectors}) < len(vectors)


def _longest_unchanged_line(a, rectangle):
    blank = a == BACKGROUND
    top, bottom, left, right = rectangle
    blank[top:bottom, left:right] = False
    longest = 0
    for values in list(blank) + list(blank.T):
        for start, end in _segments(values):
            longest = max(longest, int(end - start))
    return longest


def _analyze(a):
    core, rs, cs, rw, cw = _bands(a)
    candidates = _weighted_rectangles(core, rw, cw)
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return None
    area, top, bottom, left, right = candidates[0]
    rectangle = (int(rs[top]), int(rs[bottom]) if bottom < len(rs) else a.shape[0],
                 int(cs[left]), int(cs[right]) if right < len(cs) else a.shape[1])
    foot = (bottom - top, right - left)
    out = core.copy()
    out[top:bottom, left:right] = HIGHLIGHT
    return {"core": core, "rw": rw, "cw": cw, "area": area,
            "rectangle": rectangle, "footprint": foot, "target": (top, bottom, left, right),
            "output": out,
            "rank_reversal": max((b - t) * (r - l) for _, t, b, l, r in candidates) > foot[0] * foot[1],
            "rows_return": _returning(core), "columns_return": _returning(core.T),
            "both_compress": bool(np.any(rw > 1) and np.any(cw > 1)),
            "long_line": _longest_unchanged_line(a, rectangle) > area}


def _weights(rng, n):
    mode = int(rng.integers(5))
    if mode == 0:
        values = np.ones(n, dtype=int)
    elif mode == 1:
        values = np.full(n, int(rng.integers(2, min(4, 30 // n) + 1)), dtype=int)
    elif mode == 2:
        values = np.ones(n, dtype=int)
        values[int(rng.integers(n))] = int(rng.integers(2, min(8, 31 - n) + 1))
    else:
        values = rng.integers(1, 5, size=n)
    while int(values.sum()) > 30:
        index = int(rng.choice(np.flatnonzero(values > 1)))
        values[index] -= 1
    return values


def _sequence(rng, n, kinds):
    seq = [int(rng.integers(kinds))]
    for _ in range(n - 1):
        seq.append(int(rng.choice([v for v in range(kinds) if v != seq[-1]])))
    return seq


def _mosaic(rng, larger=False):
    h = int(rng.integers(9, 13) if larger else rng.integers(4, 9))
    w = int(rng.integers(4, 12) if larger else rng.integers(4, 9))
    empty_density = float(rng.uniform(.40, .68))
    if rng.random() < .60:
        bh = int(rng.integers(3, min(h, 7) + 1))
        bw = int(rng.integers(3, min(w, 7) + 1))
        bank = (rng.random((bh, bw)) > empty_density).astype(np.int8)
        core = bank[np.ix_(_sequence(rng, h, bh), _sequence(rng, w, bw))]
    else:
        core = (rng.random((h, w)) > empty_density).astype(np.int8)
        for _ in range(int(rng.integers(4))):
            rh = int(rng.integers(1, h // 2 + 1))
            width = int(rng.integers(1, w // 2 + 1))
            top = int(rng.integers(h - rh + 1))
            left = int(rng.integers(w - width + 1))
            core[top:top + rh, left:left + width] = BACKGROUND
        if rng.random() < .5:
            core[-1] = core[0]
        if rng.random() < .5:
            core[:, -1] = core[:, 0]
    core = _bands(core)[0]
    if min(core.shape) < 3 or not np.any(core):
        return None
    return np.repeat(np.repeat(core, _weights(rng, core.shape[0]), axis=0),
                     _weights(rng, core.shape[1]), axis=1)


def _thin(rng):
    """Small-area rectangle plus a genuinely one-pixel-wide longer blank line."""
    h, w = int(rng.integers(5, 9)), int(rng.integers(6, 9))
    core = (rng.random((h, w)) > .10).astype(np.int8)
    line = int(rng.integers(h))
    start, end = int(rng.integers(2)), int(rng.integers(w - 1, w + 1))
    core[line, start:end] = BACKGROUND
    top = int(rng.choice([t for t in range(h - 1) if line not in (t, t + 1)]))
    left = int(rng.integers(w - 1))
    core[top:top + 2, left:left + 2] = BACKGROUND
    other_columns = [x for x in range(w) if not left <= x < left + 2]
    core[top, int(rng.choice(other_columns))] = BACKGROUND
    rw, cw = np.ones(h, dtype=int), rng.integers(1, 4, size=w)
    other_rows = [y for y in range(h) if y not in (line, top, top + 1)]
    rw[int(rng.choice(other_rows))] = int(rng.integers(2, 4))
    cw[left:left + 2] = 1
    return np.repeat(np.repeat(core, rw, axis=0), cw, axis=1)


def _orient(rng, a):
    if rng.random() < .5:
        a = a.T
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, seen, palette, profile, minimum_bands=0):
    # Finite rejection bound: one candidate at a time, no histories retained.
    # Typical acceptance is 10-30% for witnesses and >60% for ordinary pairs.
    for _ in range(1800):
        raw = _thin(rng) if profile == "thin" else _mosaic(rng, profile == "larger")
        if raw is None:
            continue
        raw = _orient(rng, raw)
        a = np.where(raw == BACKGROUND, BACKGROUND, int(rng.choice(palette))).astype(np.int8)
        info = _analyze(a)
        if info is None or max(info["core"].shape) < minimum_bands:
            continue
        if profile == "weighted" and not (info["rank_reversal"] and info["rows_return"]
                and info["columns_return"] and info["both_compress"]):
            continue
        if profile == "thin" and not (info["long_line"] and min(info["footprint"]) >= 2
                                      and info["both_compress"]):
            continue
        key = _key(a)
        if key in seen:
            continue
        seen.add(key)
        return {"input": a, "output": info["output"]}
    raise RuntimeError("bounded sampling could not find the requested witness")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return exactly the requested pair counts, reproducibly and without I/O."""
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(int(x) for x in rng.choice(BLOCKERS, size=int(rng.integers(1, 9)), replace=False))
    seen = set()
    train = [_make_pair(rng, seen, palette, "weighted"), _make_pair(rng, seen, palette, "thin")]
    for _ in range(num_train - 2):
        profile = "thin" if rng.random() < .15 else "ordinary"
        train.append(_make_pair(rng, seen, palette, profile))
    rng.shuffle(train)
    # Novel band count, rather than a fixed physical shape or answer location.
    bound = max(max(_bands(p["input"])[0].shape) for p in train)
    test = [_make_pair(rng, seen, palette, "larger", minimum_bands=bound + 1)]
    for _ in range(num_test - 1):
        test.append(_make_pair(rng, seen, palette, "larger" if rng.random() < .65 else "ordinary"))
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Check the actual returned arrays and didactic coverage, not latent plans."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            colors = set(int(x) for x in np.unique(a))
            assert len(colors) == 2 and BACKGROUND in colors and HIGHLIGHT not in colors
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(a)
            assert info is not None, "ambiguous physical rectangle"
            # Raw-pixel search checks the weighted coordinate-space construction.
            area, rectangle = _physical_winner(a)
            assert (area, rectangle) == (info["area"], info["rectangle"])
            assert np.array_equal(out, info["output"])
            assert np.array_equal(out, solve(a))
            plain, rs, cs, rw, cw = _bands(a)
            changed = out != plain
            top, bottom, left, right = info["target"]
            expected_mask = np.zeros(plain.shape, dtype=bool)
            expected_mask[top:bottom, left:right] = True
            assert np.array_equal(changed, expected_mask)
            assert np.all(plain[changed] == BACKGROUND) and np.all(out[changed] == HIGHLIGHT)
            assert np.array_equal(out[~changed], plain[~changed]), "compressed context altered"
            assert np.sum(rw[top:bottom]) >= 2 and np.sum(cw[left:right]) >= 2
            assert int(np.sum(rw[top:bottom]) * np.sum(cw[left:right])) == area
            # No equality class is partially selected at a maximum rectangle.
            assert rectangle[0] in rs and rectangle[2] in cs
            assert rectangle[1] in np.r_[rs, a.shape[0]]
            assert rectangle[3] in np.r_[cs, a.shape[1]]
            infos[split].append(info)
    train = infos["train"]
    assert any(i["rank_reversal"] and i["rows_return"] and i["columns_return"]
               and i["both_compress"] for i in train), "missing weighted recurrence witness"
    assert any(i["long_line"] and min(i["footprint"]) >= 2 and i["both_compress"]
               for i in train), "missing thin-line and 2D-footprint witness"
    assert max(max(i["core"].shape) for i in infos["test"]) > max(max(i["core"].shape) for i in train)


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, data = _key(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def _brute_winner(a):
    """Tiny-grid test oracle: enumerate all four rectangle boundaries."""
    best, winners = 0, []
    for top in range(a.shape[0] - 1):
        for bottom in range(top + 2, a.shape[0] + 1):
            for left in range(a.shape[1] - 1):
                for right in range(left + 2, a.shape[1] + 1):
                    if np.all(a[top:bottom, left:right] == BACKGROUND):
                        area = (bottom - top) * (right - left)
                        if area > best:
                            best, winners = area, [(top, bottom, left, right)]
                        elif area == best:
                            winners.append((top, bottom, left, right))
    return best, winners


def self_test():
    """Bounded 42-seed coverage; no puzzle archive or execution trace is saved."""
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(811, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.25):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Enumerated tiny binary fields check unique and tied winners separately.
    rng = np.random.default_rng(17)
    for _ in range(80):
        a = (rng.random((5, 6)) > .65).astype(np.int8)
        area, winners = _brute_winner(a)
        try:
            actual = _physical_winner(a)
        except ValueError:
            assert len(winners) != 1
        else:
            assert len(winners) == 1 and actual == (area, winners[0])
    # A physically eligible rectangle may map to a single cell.
    a = np.asarray([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=int)
    assert np.array_equal(solve(a), [[2, 1], [1, 0]])
    # Separated identical rows/columns survive; no global deduplication.
    a = np.asarray([[0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]])
    assert np.array_equal(solve(a), [[2, 1, 0], [1, 0, 1], [0, 1, 0]])
    shapes, cores, footprints, weights, colors, digests = set(), set(), set(), set(), set(), set()
    coverage = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        digests.add(digest)
        assert digest == _puzzle_digest(generate_puzzle(seed)), "nondeterministic puzzle"
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                shapes.add(a.shape)
                cores.add(_key((info["core"] != 0).astype(np.int8)))
                footprints.add(info["footprint"])
                weights.add((tuple(info["rw"]), tuple(info["cw"])))
                colors.update(int(c) for c in np.unique(a) if c != BACKGROUND)
                coverage["pairs"] += 1
                coverage["rank_reversal"] += int(info["rank_reversal"])
                coverage["long_line"] += int(info["long_line"])
                coverage["single_cell"] += int(info["footprint"] == (1, 1))
                coverage["one_axis"] += int(min(info["footprint"]) == 1 < max(info["footprint"]))
                coverage["two_dimensions"] += int(min(info["footprint"]) >= 2)
                coverage["return_both"] += int(info["rows_return"] and info["columns_return"])
                coverage["wide"] += int(a.shape[1] > a.shape[0])
                coverage["tall"] += int(a.shape[0] > a.shape[1])
                coverage["size_30"] += int(30 in a.shape)
                t, b, l, r = info["rectangle"]
                coverage["interior"] += int(t > 0 and l > 0 and b < a.shape[0] and r < a.shape[1])
                coverage["boundary"] += int(t == 0 or l == 0 or b == a.shape[0] or r == a.shape[1])
                # Equivariance is a semantic check, independent of sampler plans.
                before = a.copy()
                assert np.array_equal(solve(a.T), pair["output"].T)
                assert np.array_equal(solve(a[::-1, ::-1]), pair["output"][::-1, ::-1])
                recolored = np.where(a == 0, 0, BLOCKERS[(seed + 3) % len(BLOCKERS)])
                expected = np.where(pair["output"] == HIGHLIGHT, HIGHLIGHT,
                                    np.where(pair["output"] == 0, 0, BLOCKERS[(seed + 3) % len(BLOCKERS)]))
                assert np.array_equal(solve(recolored), expected)
                assert np.array_equal(a, before), "solver mutated its input"
    assert len(digests) == 42 and len(shapes) >= 65 and len(cores) >= 150
    assert len(footprints) >= 12 and len(weights) >= 100 and colors == set(BLOCKERS)
    for feature in ("rank_reversal", "long_line", "single_cell", "one_axis", "two_dimensions",
                    "return_both", "wide", "tall", "size_30", "interior", "boundary"):
        assert coverage[feature] >= 3, f"insufficient {feature}: {coverage[feature]}"
    return {"seeds": 42, "input_shapes": len(shapes), "binary_cores": len(cores),
            "footprint_shapes": len(footprints), "band_weight_patterns": len(weights),
            "blocker_colors": len(colors), "coverage": dict(coverage)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
