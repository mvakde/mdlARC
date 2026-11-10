"""Registered relief: odd squares deposit ink; even squares punch windows.

Only Python's standard library and NumPy are used.  Input drawings and output
stamps are different representations: every input object is an outlined square
with a same-color registration dot, regardless of its parity.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _border(side):
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def _parse(input_grid):
    """Recover squares, including occluded border cells, from input alone.

    A record contains side, color, and the dot's local row/column. Positions
    are returned separately because they are irrelevant to the output.
    """
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not all(int(a[r, c]) == bg for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("the four canvas corners must identify background")
    colors = [int(x) for x in np.unique(a) if int(x) != bg]
    if not 2 <= len(colors) <= 6:
        raise ValueError("expected two to six square colors")
    records, positions, strokes = [], {}, {}
    for color in colors:
        coords = np.argwhere(a == color)
        r, c = map(int, coords.min(axis=0))
        bottom, right = map(int, coords.max(axis=0))
        side = bottom - r + 1
        if side != right - c + 1 or not 5 <= side <= 16:
            raise ValueError("a color must recover a square of side 5..16")
        region = a[r:r + side, c:c + side]
        border = _border(side)
        dots = np.argwhere((region == color) & ~border)
        if len(dots) != 1:
            raise ValueError("each frame must have one visible same-color dot")
        pr, pc = map(int, dots[0])
        if not (2 <= pr <= side - 3 and 2 <= pc <= side - 3):
            raise ValueError("a blank cell must separate dot from its own border")
        stroke = border.copy()
        stroke[pr, pc] = True
        if np.any(region[stroke] == bg):
            raise ValueError("a missing border cell needs a visible occluder")
        records.append((side, color, pr, pc))
        positions[color] = (r, c)
        strokes[color] = set(map(tuple, np.argwhere(stroke) + (r, c)))
    records.sort(reverse=True)
    if len({x[0] for x in records}) != len(records):
        raise ValueError("side lengths must be distinct; no order ties")
    # Every apparent occlusion must be explainable by some global input draw
    # order.  This also excludes arbitrary colored fragments masquerading as
    # the missing parts of a frame.
    later = {color: set() for color in colors}
    for color, stroke in strokes.items():
        for r, c in stroke:
            visible = int(a[r, c])
            if visible != color:
                if visible not in strokes or (r, c) not in strokes[visible]:
                    raise ValueError("unexplained stroke occlusion")
                later[color].add(visible)
    pending = set(colors)
    while pending:
        removable = {c for c in pending if not (later[c] & pending)}
        if not removable:
            raise ValueError("cyclic input occlusion")
        pending -= removable
    return bg, records, positions


def _bounds(records):
    top = min(-pr for s, c, pr, pc in records)
    left = min(-pc for s, c, pr, pc in records)
    bottom = max(s - pr for s, c, pr, pc in records)
    right = max(s - pc for s, c, pr, pc in records)
    return top, left, bottom, right


def _compose(bg, records, *, all_solid=False, transparent=False,
             reverse=False, ignore_dots=False):
    records = sorted(records, reverse=True)
    if ignore_dots:
        records = [(s, c, 0, 0) for s, c, pr, pc in records]
    top, left, bottom, right = _bounds(records)
    out = np.full((bottom - top, right - left), bg, dtype=np.int8)
    owner = np.full(out.shape, -1, dtype=np.int8)
    indices = list(range(len(records)))
    if reverse:
        indices.reverse()
    for i in indices:
        side, color, pr, pc = records[i]
        r, c = -pr - top, -pc - left
        region = out[r:r + side, c:c + side]
        if side % 2 or all_solid:
            region[:] = color
        else:
            if not transparent:
                region[:] = bg
            region[_border(side)] = color
        owner[r:r + side, c:c + side] = i
    return out, owner


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Align visible registration dots and apply the fixed parity stamp rule."""
    bg, records, _ = _parse(input_grid)
    return _compose(bg, records)[0]


def _pixel_raster(bg, records):
    """Cell-ownership invariant: the smallest covering square wins.

    Used to label construction records and check the painting implementation;
    this is a second rasterization, not independent semantic verification.
    """
    top, left, bottom, right = _bounds(records)
    result = np.full((bottom - top, right - left), bg, dtype=np.int8)
    smallest_first = sorted(records)
    for rr in range(top, bottom):
        for cc in range(left, right):
            for side, color, pr, pc in smallest_first:
                r, c = rr + pr, cc + pc
                if 0 <= r < side and 0 <= c < side:
                    edge = r in (0, side - 1) or c in (0, side - 1)
                    result[rr - top, cc - left] = color if side % 2 or edge else bg
                    break
    return result


def _interior_evidence(records, owner):
    top, left, _, _ = _bounds(records)
    evidence = {0: set(), 1: set()}
    for i, (side, color, pr, pc) in enumerate(records):
        r, c = -pr - top, -pc - left
        if np.any(owner[r + 1:r + side - 1, c + 1:c + side - 1] == i):
            evidence[side % 2].add(side)
    return evidence


def _occluded(a, records, positions):
    return any(np.count_nonzero(a == color) < 4 * side - 3
               for side, color, pr, pc in records)


def _overlapping(records, positions):
    for i, (s, c, _, _) in enumerate(records):
        r, col = positions[c]
        for t, d, _, _ in records[i + 1:]:
            u, v = positions[d]
            if max(r, u) < min(r + s, u + t) and max(col, v) < min(col + s, v + t):
                return True
    return False


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _sizes(rng, count, cap, repeated_parity=None, novel_low=None):
    pool = np.arange(5, cap + 1)
    for _ in range(1000):
        sides = sorted(map(int, rng.choice(pool, size=count, replace=False)), reverse=True)
        odds = [s for s in sides if s % 2]
        evens = [s for s in sides if not s % 2]
        # At least one smaller even square can cut ink from a larger odd one.
        if not odds or not evens or max(odds) < min(evens):
            continue
        if repeated_parity is not None:
            if sum(s % 2 == repeated_parity for s in sides) < 2:
                continue
            if not any(a % 2 == b % 2 for a, b in zip(sides, sides[1:])):
                continue
            if count == 4 and max(evens) < min(odds):
                # The four-object teaching case crosses the parity classes
                # in size order, defeating a single large/small threshold.
                continue
        if novel_low is not None and max(sides) < novel_low:
            continue
        return sides
    raise RuntimeError("size schedule construction exhausted")


def _records(rng, sides, colors, pin_mode):
    records = []
    for side, color in zip(sides, colors):
        if pin_mode == "central":
            lo, hi = max(2, (side - 1) // 2 - 1), min(side - 3, side // 2 + 1)
            pr, pc = map(int, rng.integers(lo, hi + 1, size=2))
        elif pin_mode == "extreme":
            pr, pc = int(rng.choice([2, side - 3])), int(rng.choice([2, side - 3]))
        else:
            pr, pc = map(int, rng.integers(2, side - 2, size=2))
        records.append((side, int(color), pr, pc))
    return sorted(records, reverse=True)


def _draw(rng, records, bg, mode):
    """Transparent outline drawings; later strokes may occlude earlier ones."""
    area = sum(s * s for s, _, _, _ in records)
    largest = records[0][0]
    for attempt in range(180):
        # Independent row/column slack, square canvases, loose and dense cases.
        if mode == "separate":
            base = min(30, max(largest + 2, int(np.sqrt(area * 1.65)) + 2))
        else:
            base = max(largest + 2, int(np.sqrt(area * float(rng.uniform(.6, 1.4)))))
        if attempt >= 100:
            height = width = 30
        else:
            height, width = [int(np.clip(base + rng.integers(-3, 7), largest + 1, 30))
                             for _ in range(2)]
        positions = {}
        for side, color, pr, pc in records:
            for _ in range(100):
                r = int(rng.integers(height - side + 1))
                c = int(rng.integers(width - side + 1))
                if any((rr in (r, r + side - 1) and c <= cc < c + side)
                       or (cc in (c, c + side - 1) and r <= rr < r + side)
                       for rr, cc in ((0, 0), (0, width - 1),
                                      (height - 1, 0), (height - 1, width - 1))):
                    continue
                if mode == "separate" and any(
                    not (r + side < u or u + t < r or c + side < v or v + t < c)
                    for t, d, _, _ in records if d in positions
                    for u, v in [positions[d]]
                ):
                    continue
                positions[color] = (r, c)
                break
            else:
                break
        if len(positions) != len(records):
            continue
        a = np.full((height, width), bg, dtype=np.int8)
        for i in rng.permutation(len(records)):
            side, color, pr, pc = records[int(i)]
            r, c = positions[color]
            region = a[r:r + side, c:c + side]
            region[_border(side)] = color
            region[pr, pc] = color
        try:
            parsed_bg, parsed, parsed_pos = _parse(a)
        except ValueError:
            continue
        if parsed_bg != bg or parsed != records or parsed_pos != positions:
            continue
        if mode == "occluded" and not _occluded(a, records, positions):
            continue
        return a
    return None


def _make_pair(rng, count, cap, repeated_parity, pin_mode, layout, bg, novel_low=None):
    for attempt in range(220):
        sides = _sizes(rng, count, cap, repeated_parity, novel_low)
        colors = rng.choice([c for c in range(10) if c != bg], size=count, replace=False)
        records = _records(rng, sides, colors, pin_mode)
        output, owner = _compose(bg, records)
        evidence = _interior_evidence(records, owner)
        # Teach both operations at every shown size rather than hiding a
        # whole stamp's interior behind its successors.
        if sum(map(len, evidence.values())) != count:
            continue
        if np.array_equal(output, _compose(bg, records, transparent=True)[0]):
            continue
        if np.array_equal(output, _compose(bg, records, ignore_dots=True)[0]):
            continue
        if np.array_equal(output, _compose(bg, records, reverse=True)[0]):
            continue
        a = _draw(rng, records, bg, layout)
        if a is not None:
            # Labels use per-cell ownership from the construction records;
            # solve independently recovers those records from the drawing.
            return {"input": a, "output": _pixel_raster(bg, records)}
    raise RuntimeError("pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["scale", "count", "registration"]))
    cap = int(rng.integers(10, 14)) if regime == "scale" else int(rng.integers(11, 15))
    first_count = int(rng.integers(3, 5))
    first_parity = int(rng.integers(2))
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            if split == "train":
                count = first_count if index == 0 else 7 - first_count if index == 1 else int(
                    rng.integers(2, 5 if regime == "count" else 6))
                repeat = first_parity if index == 0 else 1 - first_parity if index == 1 else None
                pins = "central" if regime == "registration" else "free"
                layout = "occluded" if index == 0 else "separate" if index == 1 else str(
                    rng.choice(["free", "occluded", "separate"]))
                pair_cap, novel = cap, None
            else:
                count = int(rng.integers(5, 7)) if regime == "count" else int(rng.integers(2, 7))
                repeat = None
                pins = "extreme" if regime == "registration" else "free"
                layout = str(rng.choice(["free", "occluded"]))
                pair_cap = 16 if regime == "scale" else cap
                novel = cap + 1 if regime == "scale" else 10 if regime == "registration" else None
            bg = backgrounds[(index + (num_train if split == "test" else 0)) % 10]
            for _ in range(24):
                pair = _make_pair(rng, count, pair_cap, repeat, pins, layout, bg, novel)
                key = _key(pair["input"])
                if split == "train" and index == 1:
                    previous = _parse(puzzle["train"][0]["input"])[1]
                    current = _parse(pair["input"])[1]
                    if not any(c == d and s % 2 != t % 2
                               for s, c, _, _ in previous for t, d, _, _ in current):
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, training, testing = set(), {0: set(), 1: set()}, [], []
    backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            bg, records, positions = _parse(a)
            expected, owner = _compose(bg, records)
            assert np.array_equal(b, expected)
            assert np.array_equal(b, _pixel_raster(bg, records))
            # Both concepts and the new cut-through interaction have a
            # measurable effect in every example, including every test.
            for kwargs in ({"all_solid": True}, {"transparent": True},
                           {"ignore_dots": True}, {"reverse": True}):
                assert not np.array_equal(b, _compose(bg, records, **kwargs)[0])
            local = _interior_evidence(records, owner)
            assert sum(map(len, local.values())) == len(records)
            assert local[0] and local[1]
            top, left, bottom, right = _bounds(records)
            assert b.shape == (bottom - top, right - left)
            # The common dot is inside every square; it is owned by the
            # smallest stamp, whose parity determines ink versus a hole.
            smallest = records[-1]
            assert int(b[-top, -left]) == (smallest[1] if smallest[0] % 2 else bg)
            info = (records, positions, _occluded(a, records, positions))
            if split == "train":
                training.append(info)
                backgrounds.add(bg)
                for p in (0, 1):
                    evidence[p].update(local[p])
            else:
                testing.append(info)
    assert len(backgrounds) >= 2
    assert all(len(evidence[p]) >= 2 for p in (0, 1))
    assert training[0][2] and not _overlapping(training[1][0], training[1][1])
    assert len(training[0][0]) != len(training[1][0])
    assert any(c == d and s % 2 != t % 2
               for s, c, _, _ in training[0][0] for t, d, _, _ in training[1][0])
    four = next(records for records, _, _ in training[:2] if len(records) == 4)
    assert max(s for s, _, _, _ in four if not s % 2) > min(
        s for s, _, _, _ in four if s % 2)
    assert any(a[0] % 2 == b[0] % 2 for records, _, _ in training
               for a, b in zip(records, records[1:]))
    max_count = max(len(records) for records, _, _ in training)
    max_side = max(records[0][0] for records, _, _ in training)
    max_offset = max(max(abs(pr - (s - 1) / 2), abs(pc - (s - 1) / 2))
                     for records, _, _ in training for s, c, pr, pc in records)
    for records, _, _ in testing:
        # Each test shifts size, count, or registration eccentricity.
        assert (len(records) > max_count or records[0][0] > max_side
                or max(max(abs(pr - (s - 1) / 2), abs(pc - (s - 1) / 2))
                       for s, c, pr, pc in records) > max_offset)


def self_test():
    shapes, layouts, inventories, signatures = set(), set(), set(), set()
    counts, sizes, backgrounds = set(), set(), set()
    occluded = edge = protruding = hidden_dot = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        assert all(_key(x[k]) == _key(y[k])
                   for split in p for x, y in zip(p[split], q[split])
                   for k in ("input", "output"))
        signatures.add(tuple(_key(pair["input"]) for split in p for pair in p[split]))
        for split in p:
            for pair in p[split]:
                a, b = pair["input"], pair["output"]
                bg, records, positions = _parse(a)
                assert np.array_equal(solve(a.copy()), b)
                assert np.array_equal(solve(a.tolist()), b)
                backgrounds.add(bg)
                counts.add(len(records))
                sizes.update(s for s, _, _, _ in records)
                shapes.add((a.shape, b.shape))
                inventories.add(tuple(s for s, _, _, _ in records))
                layouts.add((a.shape, tuple(positions[c] for s, c, pr, pc in records)))
                occluded += _occluded(a, records, positions)
                edge += bool(np.any(a[0] != bg) or np.any(a[-1] != bg)
                             or np.any(a[:, 0] != bg) or np.any(a[:, -1] != bg))
                protruding += max(b.shape) > records[0][0]
                hidden_dot += records[-1][0] % 2 == 0
                # Equivariance: recoloring input recolors output. Input
                # translation and extra background have no output effect.
                permutation = np.random.default_rng(seed + 10000).permutation(10)
                assert np.array_equal(solve(permutation[a]), permutation[b])
                if a.shape[0] <= 28 and a.shape[1] <= 28:
                    padded = np.pad(a, ((1, 1), (1, 1)), constant_values=bg)
                    assert np.array_equal(solve(padded), b)
                assert np.array_equal(solve(a.T), b.T)
    assert len(signatures) == 42
    assert counts == {2, 3, 4, 5, 6}
    assert sizes == set(range(5, 17))
    assert len(shapes) >= 60 and len(inventories) >= 60 and len(layouts) >= 100
    assert len(backgrounds) == 10
    assert occluded >= 42 and edge >= 10 and protruding >= 30 and hidden_dot >= 10
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(701, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2),
                        (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "shape_pairs": len(shapes), "size_inventories": len(inventories),
            "input_layouts": len(layouts), "occluded_pairs": occluded,
            "edge_touching_pairs": edge, "outputs_exceeding_largest_side": protruding,
            "counts": sorted(counts), "sides": sorted(sizes)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
