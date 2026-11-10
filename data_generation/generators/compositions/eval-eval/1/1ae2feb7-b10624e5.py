"""Resonant grafts: a source spine is both a clock program and a shape.

Only NumPy and the standard library are used. Generation paints rectangles;
solve pulls pixels from the observed source in anchor-relative coordinates.
No generated payloads, logs, caches, or execution traces are saved by this module.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer colors required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _scene(a):
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1, "ambiguous background"
    bg = int(counts.argmax())
    components = _components(a, bg)
    sources = [p for p in components if len({int(a[q]) for q in p}) > 1]
    assert len(sources) == 1, "one connected multicolored source required"
    source = sources[0]
    rest = [p for p in components if p is not source]
    assert rest, "missing destination squares"
    colors = {int(a[min(p)]) for p in rest}
    assert len(colors) == 1, "destinations must share an anchor identity"
    anchor = colors.pop()
    points = [q for q in source if int(a[q]) == anchor]
    assert len(points) == 1, "source anchor must be a unit square"
    targets = []
    for p in rest:
        rows, cols = zip(*p)
        r, c = min(rows), min(cols)
        side = max(rows)-r+1
        assert side == max(cols)-c+1 and len(p) == side*side
        assert all(int(a[q]) == anchor for q in p), "non-square destination"
        assert 1 <= side <= 5, "destination side outside this family"
        targets.append((r, c, side))
    return bg, anchor, source, points[0], sorted(targets)


def _decode(value):
    """Recover the local rightward spine, without a seed or training examples."""
    raw = _grid(value)
    bg, anchor, source, (sr, sc), targets = _scene(raw)
    neighbors = [(dr, dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                 if (sr+dr, sc+dc) in source]
    assert len(neighbors) == 1, "source has one attachment direction"
    turns = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}[neighbors[0]]
    a = np.rot90(raw, turns)
    if turns:
        bg, anchor, source, (sr, sc), targets = _scene(a)
    motif = {(r-sr, c-sc): int(a[r, c]) for r, c in source}
    spine, c = [], 1
    while (0, c) in motif:
        spine.append(motif[0, c])
        c += 1
    assert spine and all(col > 0 for row, col in motif if (row, col) != (0, 0))
    runs, cursor = [], 0
    while cursor < len(spine):
        start, color = cursor, spine[cursor]
        while cursor < len(spine) and spine[cursor] == color:
            cursor += 1
        runs.append((color, cursor-start, start+1))
    assert 1 <= len(runs) <= 3 and all(1 <= p <= 3 for _, p, _ in runs)
    assert len({color for color, _, _ in runs}) == len(runs), "one run per pigment"
    for (r, c), color in motif.items():
        if (r, c) == (0, 0):
            continue
        assert 1 <= c <= len(spine) and color == spine[c-1] and abs(r) <= 3
        # Every branch is an uninterrupted extension of its own spine cell.
        assert all(motif.get((v, c)) == color for v in range(min(0, r), max(0, r)+1))
    assert all(any(r != 0 and value == color for (r, c), value in motif.items())
               for color, _, _ in runs), "each pigment needs a visible spur"
    return {"grid": a, "turns": turns, "background": bg, "anchor": anchor,
            "source": (sr, sc), "motif": motif, "runs": runs, "targets": targets}


def _winner(runs, side, mode="normal"):
    phase = side if mode == "wrong_phase" else side-1
    eligible = [run for run in runs if phase % run[1] == 0]
    if mode == "always_near":
        eligible = runs[:1]
    if not eligible:
        return None
    if mode == "short":
        return min(eligible, key=lambda run: (run[1], run[2]))[0]
    if mode == "long":
        return min(eligible, key=lambda run: (-run[1], run[2]))[0]
    if mode == "far":
        return eligible[-1][0]
    if mode == "low_color":
        return min(run[0] for run in eligible)
    if mode == "high_color":
        return max(run[0] for run in eligible)
    return eligible[0][0]


def _prediction(value, mode="normal"):
    data = _decode(value)
    a, bg = data["grid"], data["background"]
    sr, sc = data["source"]
    out = a.copy()
    # A pull renderer, deliberately different from construction's rectangle painter.
    for r, c, side in data["targets"]:
        color = _winner(data["runs"], side, mode)
        if color is None and mode != "all_colors":
            continue
        vertical = -1 if 2*r+side < 2*sr+1 else 1
        horizontal = -1 if 2*c+side < 2*sc+1 else 1
        if mode == "global_frame":
            # A rival interprets left/above in the canvas frame instead of
            # the source's oriented frame. Rotate the center displacement back.
            dy, dx = 2*r+side-(2*sr+1), 2*c+side-(2*sc+1)
            for _ in range(data["turns"]):
                dy, dx = dx, -dy
            gv, gh = (-1 if dy < 0 else 1), (-1 if dx < 0 else 1)
            vertical, horizontal = (gh, gv) if data["turns"] % 2 else (gv, gh)
        if mode in ("no_reflection", "no_vertical"):
            vertical = 1
        if mode in ("no_reflection", "no_horizontal"):
            horizontal = 1
        unit = 1 if mode == "no_scale" else side
        offset = next((start-1 for value, _, start in data["runs"] if value == color), 0)
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                source_r = vertical*((y-r)//unit)
                source_c = horizontal*((x-c)//unit) + (offset if mode == "reattach" else 0)
                source_color = data["motif"].get((source_r, source_c))
                if source_color is None or source_color == data["anchor"]:
                    continue
                if mode == "no_spurs" and source_r != 0:
                    continue
                if mode != "all_colors" and source_color != color:
                    continue
                # Valid scenes have no overwritten foreground or overlapping grafts.
                if mode == "normal":
                    assert a[y, x] == bg, "graft would overwrite input foreground"
                    assert out[y, x] == bg, "overlapping destination grafts"
                if a[y, x] == bg:
                    out[y, x] = source_color
    return np.rot90(out, -data["turns"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _prediction(input_grid)


def _glyph(motif, color, side, vertical, horizontal):
    """Construction uses scaled cell rectangles, including their exact clipping."""
    result = []
    for (r, c), value in motif.items():
        if value == color:
            result.append((vertical*r*side, (vertical*r+1)*side,
                           horizontal*c*side, (horizontal*c+1)*side))
    return result


def _patch_mask(shape, r, c, rectangles):
    mask = np.zeros(shape, dtype=bool)
    total = 0
    for y0, y1, x0, x1 in rectangles:
        total += (y1-y0)*(x1-x0)
        top, bottom = max(0, r+y0), min(shape[0], r+y1)
        left, right = max(0, c+x0), min(shape[1], c+x1)
        if top < bottom and left < right:
            mask[top:bottom, left:right] = True
    return mask, total


def _near_foreground(mask, r, c, side):
    h, w = mask.shape
    # Four-neighbor exclusion, so input components retain unique roles.
    if mask[max(0, r-1):min(h, r+side+1), c:c+side].any():
        return True
    return bool(mask[r:r+side, max(0, c-1):min(w, c+side+1)].any())


def _make_motif(rng, lengths, colors, depth):
    motif, cursor = {}, 1
    runs = []
    for length, color in zip(lengths, colors):
        runs.append((color, length, cursor))
        columns = list(range(cursor, cursor+length))
        for c in columns:
            motif[0, c] = color
        # Nonempty, independently placed spurs make each pigment a 2D graft.
        chosen = rng.choice(columns, int(rng.integers(1, length+1)), replace=False)
        direction = int(rng.choice([-1, 1]))
        for c in chosen:
            branch = int(rng.integers(1, depth+1))
            for d in range(1, branch+1):
                motif[direction*d, int(c)] = color
        cursor += length
    return motif, runs


def _build_pair(rng, lengths, colors, roles, *, witness=False, test=False, rotation=0):
    bg, anchor = roles
    motif, runs = _make_motif(rng, lengths, colors, 1 if witness else 3 if test else 2)
    if witness:
        shape = (int(rng.integers(27, 31)), int(rng.integers(29, 31)))
        sr, sc = shape[0]//2, shape[1]-sum(lengths)-1
        sides = [4, 3, 2, 1]
    else:
        shape = (int(rng.integers(18, 31)), int(rng.integers(21, 31)))
        sr = int(rng.integers(4, shape[0]-4))
        sc = int(rng.integers(max(1, shape[1]//2), shape[1]-sum(lengths)))
        count = int(rng.integers(3, 7))
        sides = [5] if test else [int(rng.integers(1, 5))]
        sides += [int(rng.integers(1, 5)) for _ in range(count-1)]
    inp = np.full(shape, bg, dtype=np.int8)
    inp[sr, sc] = anchor
    for (r, c), color in motif.items():
        inp[sr+r, sc+c] = color
    out = inp.copy()
    occupied, original = inp != bg, inp != bg
    records = []
    down_for_four = bool(rng.integers(2))
    for index, side in enumerate(sides):
        color = _winner(runs, side)
        options = {}
        for vertical in (-1, 1):
            for horizontal in (-1, 1):
                options[vertical, horizontal] = _glyph(motif, color, side, vertical, horizontal)
        placed = False
        h, w = shape[0]-side+1, shape[1]-side+1
        for position in rng.permutation(h*w):
            r, c = divmod(int(position), w)
            vertical = -1 if 2*r+side < 2*sr+1 else 1
            horizontal = -1 if 2*c+side < 2*sc+1 else 1
            if witness:
                if side in (3, 4):
                    desired_down = down_for_four if side == 4 else not down_for_four
                    if horizontal != -1 or (vertical == 1) != desired_down:
                        continue
                elif side == 1 and horizontal != 1:
                    continue
            if _near_foreground(original, r, c, side):
                continue
            square = np.zeros(shape, dtype=bool)
            square[r:r+side, c:c+side] = True
            if (square & occupied).any():
                continue
            graft, total = _patch_mask(shape, r, c, options[vertical, horizontal])
            visible = int(graft.sum())
            if color is not None and visible < max(side, total//8):
                continue
            if witness and visible != total:
                continue
            if test and index == 0 and (color is None or visible == total):
                continue
            if (graft & occupied).any() or (graft & square).any():
                continue
            inp[square] = anchor
            out[square] = anchor
            if color is not None:
                out[graft] = color
            original = original | square
            occupied = occupied | square | graft
            records.append((visible, total))
            placed = True
            break
        if not placed:
            return None
    if not any(v for v, _ in records):
        return None
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    if int(counts.argmax()) != bg or np.count_nonzero(counts == counts.max()) != 1:
        return None
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _fallback_pair(lengths, colors, roles, rotation, test):
    """A finite, explicitly packed member of the same rule, never an exception."""
    bg, anchor = roles
    motif, runs, cursor = {}, [], 1
    for color, length in zip(colors, lengths):
        runs.append((color, length, cursor))
        for c in range(cursor, cursor+length):
            motif[0, c] = color
        motif[-1, cursor] = color
        cursor += length
    sr, sc = 14, 29-sum(lengths)
    inp = np.full((30, 30), bg, dtype=np.int8)
    inp[sr, sc] = anchor
    for (r, c), color in motif.items():
        inp[sr+r, sc+c] = color
    out = inp.copy()
    if test:
        targets = ((20, 19, 5), (1, 17, 3), (3, 26, 1), (8, 16, 2))
    elif len(lengths) == 2:
        targets = ((20, 21, 4), (1, 15, 3), (3, 26, 1), (8, 16, 2))
    else:
        targets = ((20, 20, 4), (1, 17, 3), (3, 26, 1), (8, 16, 2))
    for r, c, side in targets:
        inp[r:r+side, c:c+side] = anchor
        out[r:r+side, c:c+side] = anchor
        color = _winner(runs, side)
        v = -1 if 2*r+side < 2*sr+1 else 1
        h = -1 if 2*c+side < 2*sc+1 else 1
        mask, _ = _patch_mask(inp.shape, r, c, _glyph(motif, color, side, v, h))
        if color is not None:
            out[mask] = color
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    pigment_two, pigment_three, pigment_one = palette[:3]
    base = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index < 2
            if witness:
                lengths = (2, 3) if index == 0 else (3, 2, 1)
                colors = ((pigment_two, pigment_three) if index == 0 else
                          (pigment_three, pigment_two, pigment_one))
                rotation = (base+index) % 4
            else:
                rotation = int(rng.integers(4))
            pair = None
            # Resample structural choices too: an all-period-3 source cannot
            # produce the required fired side-5 test witness, for example.
            # All failed candidates are discarded in RAM.
            for _ in range(48):
                if not witness:
                    number = int(rng.integers(1, 4)) if split == "train" else 3
                    lengths = tuple(int(x) for x in rng.integers(1, 4, size=number))
                    colors = tuple(int(x) for x in rng.choice(10, number, replace=False))
                free = [c for c in range(10) if c not in colors]
                roles = tuple(int(x) for x in rng.choice(free, 2, replace=False))
                candidate = _build_pair(rng, lengths, colors, roles, witness=witness,
                                        test=split == "test", rotation=rotation)
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                if witness and index == 0:
                    lengths, colors = (2, 3), (pigment_two, pigment_three)
                else:
                    lengths, colors = (3, 2, 1), (pigment_three, pigment_two, pigment_one)
                free = [c for c in range(10) if c not in colors]
                # At least 42 distinct role assignments against <=13 old inputs.
                for bg in free:
                    for anchor in free:
                        if bg == anchor:
                            continue
                        candidate = _fallback_pair(lengths, colors, (bg, anchor), rotation, split == "test")
                        if _key(candidate["input"]) not in seen:
                            pair = candidate
                            break
                    if pair is not None:
                        break
                assert pair is not None, "finite role space exhausted"
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    return puzzle


def _features(pair):
    d = _decode(pair["input"])
    sr, sc = d["source"]
    clips, sides, flips, winners = 0, set(), set(), set()
    for r, c, side in d["targets"]:
        color = _winner(d["runs"], side)
        sides.add(side)
        v = -1 if 2*r+side < 2*sr+1 else 1
        h = -1 if 2*c+side < 2*sc+1 else 1
        if color is not None:
            mask, total = _patch_mask(d["grid"].shape, r, c, _glyph(d["motif"], color, side, v, h))
            clips += int(mask.sum()) < total
            flips.add((v, h))
            winners.add(color)
    return {"shape": tuple(np.asarray(pair["input"]).shape), "direction": d["turns"],
            "run_lengths": tuple(run[1] for run in d["runs"]), "targets": len(d["targets"]),
            "sides": sides, "flips": flips, "clips": clips, "winners": winners,
            "depth": max(abs(r) for r, c in d["motif"]),
            "roles": (d["background"], d["anchor"]),
            "spurs": sum(r != 0 for r, c in d["motif"])}


_RIVALS = ("short", "long", "far", "low_color", "high_color", "always_near",
           "wrong_phase", "all_colors", "no_scale", "no_horizontal", "no_vertical",
           "reattach", "no_spurs", "global_frame")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            assert np.array_equal(solve(inp), out), "gating, graft geometry, clipping, or preservation"
            d = _decode(inp)
            assert np.array_equal(inp[inp != d["background"]], out[inp != d["background"]])
            features[split].append(_features(pair))
    # Examine the actual visible labels, not a plan or a latent witness flag.
    for mode in _RIVALS:
        assert any(not np.array_equal(_prediction(p["input"], mode), p["output"])
                   for p in puzzle["train"]), f"training fails to exclude {mode}"
    assert any(f["run_lengths"] == (2, 3) and f["clips"] == 0 for f in features["train"])
    assert any(f["run_lengths"] == (3, 2, 1) and f["clips"] == 0 for f in features["train"])
    assert all(5 not in f["sides"] for f in features["train"])
    assert all(5 in f["sides"] and f["clips"] > 0 for f in features["test"])


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 puzzles; no sample archives, traces, or growing logs."""
    # A hand-computed regression: no output is obtained from generation or a
    # second renderer. In particular the side-4 far pigment keeps its offset,
    # reflects upward, and loses half its spur at the top canvas edge.
    inp = np.zeros((30, 30), dtype=np.int8)
    inp[14, 23] = 9
    inp[14, 24:26] = 2
    inp[13, 24] = 2
    inp[14, 26:29] = 3
    inp[15, 28] = 3
    for r, c, s in ((1, 25, 1), (8, 15, 2), (21, 12, 3), (2, 21, 4)):
        inp[r:r+s, c:c+s] = 9
    expected = inp.copy()
    expected[1, 26:28] = 2
    expected[2, 26] = 2
    expected[21:24, 6:12] = 2
    expected[18:21, 9:12] = 2
    expected[2:6, 1:13] = 3
    expected[0:2, 1:5] = 3
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn)), "hand-computed fixture"
    # Exercise the finite packing fallback directly; it should not be dead code.
    for turn in range(4):
        fallback_train = []
        for ls, cs, test in (((2, 3), (2, 3), False), ((3, 2, 1), (3, 2, 1), False),
                             ((3, 2, 1), (3, 2, 1), True)):
            p = _fallback_pair(ls, cs, (0, 9), turn, test)
            assert np.array_equal(solve(p["input"]), p["output"]), "fallback geometry"
            assert (_features(p)["clips"] > 0) == test
            if test:
                validate_generated_puzzle({"train": fallback_train, "test": [p]})
            else:
                fallback_train.append(p if not fallback_train else
                    _fallback_pair(ls, cs, (0, 9), (turn+1) % 4, False))
    digests, shapes, roles, lengths, directions, counts, depths = set(), set(), set(), set(), set(), set(), set()
    max_spurs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism"
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                roles.add(f["roles"])
                lengths.add(f["run_lengths"])
                directions.add(f["direction"])
                counts.add(f["targets"])
                depths.add(f["depth"])
                max_spurs = max(max_spurs, f["spurs"])
    assert len(digests) == 42 and len(shapes) >= 35 and len(roles) >= 35
    assert len(lengths) >= 15 and directions == {0, 1, 2, 3}
    assert counts >= {3, 4, 5, 6} and depths >= {1, 2, 3} and max_spurs >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(808+a*10+b, a, b)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    for turn in range(4):
        for pair in original["train"]+original["test"]:
            assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
    permutation = np.array([7, 3, 9, 0, 5, 2, 8, 4, 1, 6], dtype=np.int8)
    for pair in original["train"]+original["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    broken = {s: [{r: np.array(g, copy=True) for r, g in p.items()} for p in original[s]] for s in original}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "distinct_puzzles": len(digests), "shapes": len(shapes),
            "role_assignments": len(roles), "run_sequences": len(lengths),
            "directions": sorted(directions), "target_counts": sorted(counts),
            "depths": sorted(depths), "max_spur_cells": max_spurs}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
