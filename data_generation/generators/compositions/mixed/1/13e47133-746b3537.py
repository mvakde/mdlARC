"""Diagonal contour keys followed by region-local adjacent-vector compression.

The lattice retains its incidence and one-cell dividers, not pixel coordinates.
Only compatible compact rooms belong to this family. No files are read by solve.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json

import numpy as np


_LAYOUTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2),
            (2, 3), (3, 2), (3, 3))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _digest(a):
    return hashlib.sha256(bytes(a.shape) + a.astype(np.int8).tobytes()).hexdigest()


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                h.update(_digest(np.asarray(pair[role])).encode("ascii"))
    return h.hexdigest()


def _intervals(length, walls):
    stops = [-1] + [int(x) for x in walls] + [length]
    return [(a + 1, b) for a, b in zip(stops, stops[1:])]


def _parse_lattice(a):
    candidates = []
    for color in np.unique(a):
        mask = a == color
        rows = np.flatnonzero(np.all(mask, axis=1))
        cols = np.flatnonzero(np.all(mask, axis=0))
        if not (len(rows) or len(cols)):
            continue
        union = np.zeros(a.shape, dtype=bool)
        union[rows, :] = True
        union[:, cols] = True
        if not np.array_equal(mask, union):
            continue
        ri = _intervals(a.shape[0], rows)
        ci = _intervals(a.shape[1], cols)
        # This also excludes border walls and adjacent/thick divider lines.
        if any(b - s < 5 for s, b in ri + ci):
            continue
        candidates.append((int(color), ri, ci))
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous rectangular divider lattice")
    divider, ri, ci = candidates[0]
    counts = np.bincount(a[a != divider], minlength=10)
    neutral = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[neutral]) != 1:
        raise ValueError("neutral canvas must be the unique most frequent open color")
    return neutral, divider, ri, ci


def _depth(h, w):
    r, c = np.indices((h, w))
    return np.minimum(np.minimum(r, h - 1 - r), np.minimum(c, w - 1 - c))


def _decode_key(room, neutral, depth):
    positions = np.argwhere(room != neutral)
    if not len(positions):
        return (neutral,), ()
    length = max(int(depth[r, c]) for r, c in positions) + 1
    h, w = room.shape
    readings = []
    for r0, c0, dr, dc in ((0, 0, 1, 1), (0, w - 1, 1, -1),
                            (h - 1, 0, -1, 1), (h - 1, w - 1, -1, -1)):
        if not all((int(r), int(c)) == (r0 + int(depth[r, c]) * dr,
                                       c0 + int(depth[r, c]) * dc)
                   for r, c in positions):
            continue
        path = [(r0 + d * dr, c0 + d * dc) for d in range(length)]
        if not all(0 <= r < h and 0 <= c < w and depth[r, c] == d
                   for d, (r, c) in enumerate(path)):
            continue
        readings.append((tuple(int(room[r, c]) for r, c in path), (dr, dc)))
    if not readings or len({p for p, _ in readings}) != 1:
        raise ValueError("all cues in a room must determine one inward diagonal key")
    return readings[0][0], tuple(direction for _, direction in readings)


def _run_indices(a):
    rows = np.r_[True, np.any(a[1:, :] != a[:-1, :], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return np.flatnonzero(rows), np.flatnonzero(cols)


def _compress(a):
    rows, cols = _run_indices(a)
    return a[np.ix_(rows, cols)].copy()


def _assemble(blocks, divider):
    nr, nc = len(blocks), len(blocks[0])
    if any(len(row) != nc for row in blocks):
        raise ValueError("ragged room lattice")
    heights = [row[0].shape[0] for row in blocks]
    widths = [blocks[0][j].shape[1] for j in range(nc)]
    if any(blocks[i][j].shape != (heights[i], widths[j])
           for i in range(nr) for j in range(nc)):
        raise ValueError("compact rooms do not align across their divider lattice")
    out = np.full((sum(heights) + nr - 1, sum(widths) + nc - 1),
                  divider, dtype=np.int8)
    r = 0
    for i, height in enumerate(heights):
        c = 0
        for j, width in enumerate(widths):
            out[r:r + height, c:c + width] = blocks[i][j]
            c += width + 1
        r += height + 1
    return out


def _analyze(a):
    neutral, divider, ri, ci = _parse_lattice(a)
    rooms, compact, filled = [], [], []
    for rs, re in ri:
        info_row, compact_row, filled_row = [], [], []
        for cs, ce in ci:
            raw = a[rs:re, cs:ce]
            depth = _depth(*raw.shape)
            period, directions = _decode_key(raw, neutral, depth)
            full = np.asarray(period, dtype=np.int8)[depth % len(period)]
            small = _compress(full)
            info_row.append({"raw": raw, "depth": depth, "period": period,
                             "directions": directions, "filled": full,
                             "compact": small})
            compact_row.append(small)
            filled_row.append(full)
        rooms.append(info_row)
        compact.append(compact_row)
        filled.append(filled_row)
    return {"neutral": neutral, "divider": divider, "rooms": rooms,
            "filled": _assemble(filled, divider),
            "output": _assemble(compact, divider),
            "layout": (len(ri), len(ci))}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode input-local roles and keys, fill, then compress and reassemble."""
    return _analyze(_grid(input_grid))["output"]


@lru_cache(maxsize=16)
def _patterns(depth):
    """Canonical equality patterns, up to three colors and five key positions."""
    result = []

    def extend(prefix, length):
        if len(prefix) == length:
            p = tuple(prefix)
            changes = sum(p[d % length] != p[(d - 1) % length]
                          for d in range(1, depth + 1))
            result.append((p, changes))
            return
        for symbol in range(min(3, max(prefix) + 2)):
            extend(prefix + [symbol], length)

    for length in range(1, min(depth + 1, 5) + 1):
        extend([0], length)
    return tuple(result)


def _sample_key(rng, depth, changes, neutral, divider, requirement=None,
                avoid_first_transition=None):
    choices = []
    for p, q in _patterns(depth):
        if q != changes:
            continue
        first_transition = next((d for d in range(1, len(p)) if p[d] != p[0]), None)
        if first_transition == avoid_first_transition and avoid_first_transition is not None:
            continue
        if requirement == "cycle_thick" and not (
                len(p) <= depth and p[-1] != p[0]
                and any(a == b for a, b in zip(p, p[1:]))):
            continue
        if requirement == "leading" and p[0] == p[-1]:
            continue
        if requirement == "internal" and not (set(p[1:-1]) - {p[0], p[-1]}):
            continue
        choices.append(p)
    if not choices:
        raise RuntimeError("unsupported key/depth combination")
    p = choices[int(rng.integers(len(choices)))]
    colors = [c for c in range(10) if c not in (neutral, divider)]
    labels = [int(c) for c in rng.choice(colors, max(p) + 1, replace=False)]
    if requirement == "leading":
        labels[p[0]] = neutral
    elif requirement == "internal":
        eligible = sorted(set(p[1:-1]) - {p[0], p[-1]})
        labels[int(rng.choice(eligible))] = neutral
    elif requirement != "cycle_thick" and rng.random() < 0.3:
        eligible = sorted(set(p) - {p[-1]})
        if eligible:
            labels[int(rng.choice(eligible))] = neutral
    return tuple(labels[v] for v in p)


def _axis_sizes(rng, count, minimum):
    # Leave at least one removable row/column even with all-unit contour bands.
    cap = 30 if count == 1 else (14 if count == 2 else 10)
    low = count * minimum + count - 1 + 1
    high = min(30, count * cap + count - 1)
    total = int(rng.integers(low, high + 1)) - (count - 1)
    sizes = [minimum] * count
    for _ in range(total - sum(sizes)):
        possible = [i for i, v in enumerate(sizes) if v < cap]
        sizes[int(rng.choice(possible))] += 1
    rng.shuffle(sizes)
    return sizes


def _make_pair(rng, layout, changes, neutral, divider, demonstration=None):
    nr, nc = layout
    minimum = max(7, 2 * changes + 1)
    heights = _axis_sizes(rng, nr, minimum)
    widths = _axis_sizes(rng, nc, minimum)
    raw_blocks, output_blocks = [], []
    first_transition = None
    for i, h in enumerate(heights):
        raw_row, out_row = [], []
        for j, w in enumerate(widths):
            slot = i * nc + j
            depth = _depth(h, w)
            requirement = None
            if demonstration == 0 and slot == 0:
                requirement = "cycle_thick"
            elif demonstration == 0 and slot == 1:
                requirement = "leading"
            elif demonstration == 1 and slot == 0:
                requirement = "internal"
            period = _sample_key(rng, int(depth.max()), changes,
                                 neutral, divider, requirement,
                                 first_transition if demonstration == 0 and slot == 1 else None)
            if demonstration == 0 and slot == 0:
                first_transition = next(d for d in range(1, len(period))
                                        if period[d] != period[0])
            # Both inward directions occur in demonstrations, including 2/1 counts.
            dr = 1 if demonstration == 0 else (-1 if demonstration == 1
                                               else int(rng.choice((-1, 1))))
            dc = int(rng.choice((-1, 1)))
            r0, c0 = (0 if dr == 1 else h - 1), (0 if dc == 1 else w - 1)
            raw = np.full((h, w), neutral, dtype=np.int8)
            for d, color in enumerate(period):
                raw[r0 + dr * d, c0 + dc * d] = color
            full = np.asarray(period, dtype=np.int8)[depth % len(period)]
            raw_row.append(raw)
            out_row.append(_compress(full))
        raw_blocks.append(raw_row)
        output_blocks.append(out_row)
    return {"input": _assemble(raw_blocks, divider),
            "output": _assemble(output_blocks, divider)}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _count("num_train", num_train, 2, 10)
    ne = _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    backgrounds = [int(c) for c in rng.permutation(10)]
    offset = int(rng.integers(1, 10))
    dividers = backgrounds[offset:] + backgrounds[:offset]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_layout = _LAYOUTS[int(rng.choice((0, 1, 4)))]
    for index in range(nt + ne):
        split = "train" if index < nt else "test"
        demo = index if index < 2 else None
        changes = (2 if index == 0 else 3) if index < 2 else (
            int(rng.choice((2, 3))) if split == "train" else 4)
        if index == 0:
            layout = first_layout
        elif index == 1:
            options = ((2, 2), (2, 3), (3, 2)) if 1 in first_layout else (
                (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))
            layout = options[int(rng.integers(len(options)))]
        else:
            layout = _LAYOUTS[int(rng.integers(len(_LAYOUTS)))]
        for _ in range(32):
            pair = _make_pair(rng, layout, changes, backgrounds[index % 10],
                              dividers[index % 10], demo)
            signature = _digest(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("bounded uniqueness search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _sequential_compress(a, columns_first=False):
    """List-based check, separate from the vectorized simultaneous selection."""
    def rows(b):
        result = [b[0].copy()]
        for row in b[1:]:
            if tuple(row) != tuple(result[-1]):
                result.append(row.copy())
        return np.asarray(result, dtype=np.int8)
    return rows(rows(a.T).T) if columns_first else rows(rows(a).T).T


def _unique_vectors(a):
    def indices(vectors):
        seen, keep = set(), []
        for i, row in enumerate(vectors):
            key = tuple(int(v) for v in row)
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return keep
    return a[np.ix_(indices(a), indices(a.T))]


def _features(analysis):
    rooms = [r for row in analysis["rooms"] for r in row]
    neutral = analysis["neutral"]
    return {
        "cyclic_thick": any(len(r["period"]) <= int(r["depth"].max())
                            and r["period"][-1] != r["period"][0]
                            and any(a == b for a, b in zip(r["period"], r["period"][1:]))
                            for r in rooms),
        "leading": any(r["period"][0] == neutral and len(r["period"]) > 1 for r in rooms),
        "internal": any(neutral in r["period"][1:-1] for r in rooms),
        "without_a": any(not np.array_equal(_compress(r["raw"]), r["compact"]) for r in rooms),
        "separated": any(not np.array_equal(_unique_vectors(r["filled"]), r["compact"])
                         for r in rooms),
    }


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _digest(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            a = _analyze(inp)
            assert np.array_equal(out, a["output"]), "incorrect contour/compression result"
            assert out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1]
            assert not np.array_equal(out, a["filled"]), "B must change the answer"
            assert set(int(v) for v in out.flat) <= set(int(v) for v in inp.flat)
            shapes = set()
            for row in a["rooms"]:
                for r in row:
                    period, full, small = r["period"], r["filled"], r["compact"]
                    assert 2 <= len(period) <= 5 and period[-1] != a["neutral"]
                    assert a["divider"] not in period
                    assert np.array_equal(small, _sequential_compress(full))
                    assert np.array_equal(small, _sequential_compress(full, True))
                    assert np.array_equal(small, _compress(small)), "compression must be idempotent"
                    rr, cc = _run_indices(full)
                    # Repeating the retained vectors by their actual run lengths
                    # reconstructs the entire filled room, including separated repeats.
                    rw = np.diff(np.r_[rr, full.shape[0]])
                    cw = np.diff(np.r_[cc, full.shape[1]])
                    rebuilt = np.repeat(np.repeat(small, rw, axis=0), cw, axis=1)
                    assert np.array_equal(rebuilt, full), "lossy vector-run quotient"
                    shapes.add(small.shape)
                    for v in (small, small.T):
                        assert all(not np.array_equal(x, y) for x, y in zip(v, v[1:]))
            assert len(shapes) == 1
            target = next(iter(shapes))
            assert target[0] == target[1]
            assert target[0] in ((5, 7) if split == "train" else (9,))
            nr, nc = a["layout"]
            assert np.count_nonzero(out == a["divider"]) == (
                (nr - 1) * out.shape[1] + (nc - 1) * out.shape[0] - (nr - 1) * (nc - 1))
            f = _features(a)
            assert f["without_a"] and f["separated"]
            analyses[split].append((a, f))
    assert any(f["cyclic_thick"] for _, f in analyses["train"])
    assert any(f["leading"] for _, f in analyses["train"])
    assert any(f["internal"] for _, f in analyses["train"])
    assert any(not np.array_equal(_compress(a["filled"]), a["output"])
               for a, _ in analyses["train"]), "demonstrate local rather than whole-grid compression"
    assert {r["compact"].shape[0] for a, _ in analyses["train"]
            for row in a["rooms"] for r in row} == {5, 7}
    assert any(all(d[0] == 1 for d in r["directions"]) for a, _ in analyses["train"]
               for row in a["rooms"] for r in row)
    assert any(all(d[0] == -1 for d in r["directions"]) for a, _ in analyses["train"]
               for row in a["rooms"] for r in row)
    assert any(min(a["layout"]) >= 2 for a, _ in analyses["train"])


def _erosion_depth(h, w):
    current = np.ones((h, w), dtype=bool)
    result = np.full((h, w), -1, dtype=int)
    d = 0
    while np.any(current):
        padded = np.pad(current, 1)
        inside = current.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~inside] = d
        current, d = inside, d + 1
    return result


def self_test():
    """Streaming 42-seed diagnostics; no files or retained puzzle collection."""
    signatures, shapes, layouts, roles, periods, run_widths, directions = (set() for _ in range(7))
    total_pairs = total_rooms = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                inp, out = pair["input"], pair["output"]
                a = _analyze(inp)
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(solve(inp.T), out.T)
                assert np.array_equal(solve(inp[::-1, ::-1]), out[::-1, ::-1])
                relabel = (np.arange(10) * 3 + seed) % 10
                assert np.array_equal(solve(relabel[inp]), relabel[out])
                shapes.add(tuple(inp.shape))
                layouts.add(a["layout"])
                roles.add((a["neutral"], a["divider"]))
                for row in a["rooms"]:
                    for r in row:
                        total_rooms += 1
                        assert np.array_equal(r["depth"], _erosion_depth(*r["raw"].shape))
                        ids = {}
                        periods.add(tuple(ids.setdefault(v, len(ids)) for v in r["period"]))
                        directions.update(r["directions"])
                        rr, cc = _run_indices(r["filled"])
                        run_widths.add(tuple(int(x) for x in np.diff(np.r_[rr, r["filled"].shape[0]])))
                        run_widths.add(tuple(int(x) for x in np.diff(np.r_[cc, r["filled"].shape[1]])))
    assert len(signatures) == 42 and len(shapes) >= 60
    assert layouts == set(_LAYOUTS) and len(roles) >= 50
    assert len(periods) >= 20 and len(run_widths) >= 60
    assert directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(90210, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _puzzle_digest(generate_puzzle(17, 7, 3)) == _puzzle_digest(generate_puzzle(17, 7, 3))
    broken = generate_puzzle(4, 2, 1)
    a = _analyze(broken["test"][0]["input"])
    out = broken["test"][0]["output"].copy()
    out[out == a["divider"]] = a["neutral"]
    broken["test"][0]["output"] = out
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted erased dividers")
    return {"seeds": 42, "pairs": total_pairs, "rooms": total_rooms,
            "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "layouts": sorted(layouts), "role_pairs": len(roles),
            "key_equality_patterns": len(periods), "axis_run_patterns": len(run_widths),
            "directions": sorted(directions)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
