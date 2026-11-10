"""Unfold primitive repeat cells, guided by an asymmetric T/cross marker.

Only standard library and NumPy are used. Outputs are constructed from sampled
unit cells; solve() independently recovers those cells from input grids.
"""
from __future__ import annotations

import numpy as np

# Clockwise ordering: subtract one to turn counterclockwise.
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_NAMES = ("up", "right", "down", "left")


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    arr = np.asarray(value)
    if (arr.ndim != 2 or not all(1 <= n <= 30 for n in arr.shape)
            or not np.issubdtype(arr.dtype, np.integer)
            or not np.all((0 <= arr) & (arr <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return arr


def _key(arr):
    arr = np.asarray(arr, dtype=np.int8)
    return arr.shape, arr.tobytes()


def _axis_period(arr, axis):
    """Shortest visible translation supported by at least two full repeats."""
    n = arr.shape[axis]
    for period in range(1, n // 2 + 1):
        a = np.take(arr, np.arange(period, n), axis=axis)
        b = np.take(arr, np.arange(n - period), axis=axis)
        if np.array_equal(a, b):
            return period
    raise ValueError("sample has no sufficiently evidenced axis period")


def _decode_marker(mask):
    cells = {tuple(map(int, pos)) for pos in np.argwhere(mask)}
    candidates = []
    for jr, jc in sorted(cells):
        rays, lengths = {(jr, jc)}, []
        for dr, dc in _DELTAS:
            length = 0
            while (jr + (length + 1) * dr, jc + (length + 1) * dc) in cells:
                length += 1
                rays.add((jr + length * dr, jc + length * dc))
            lengths.append(length)
        longest = max(lengths)
        if (rays == cells and sum(v > 0 for v in lengths) in (3, 4)
                and longest >= 2 and lengths.count(longest) == 1):
            candidates.append((lengths.index(longest), (jr, jc), tuple(lengths)))
    return candidates[0] if len(candidates) == 1 else None


def _analyze_input(input_grid):
    inp = _grid(input_grid)
    background = int(inp[0, 0])
    if not (np.all(inp[0] == background) and np.all(inp[-1] == background)
            and np.all(inp[:, 0] == background) and np.all(inp[:, -1] == background)):
        raise ValueError("the entire perimeter must be background")
    markers = []
    for color in map(int, np.unique(inp)):
        if color != background:
            decoded = _decode_marker(inp == color)
            if decoded is not None:
                markers.append((color, decoded))
    if len(markers) != 1:
        raise ValueError("expected exactly one geometric marker color")
    marker_color, (long_direction, junction, lengths) = markers[0]
    occupied = np.argwhere((inp != background) & (inp != marker_color))
    if not len(occupied):
        raise ValueError("missing fabric")
    r0, c0 = map(int, occupied.min(axis=0))
    r1, c1 = map(int, occupied.max(axis=0))
    sample = inp[r0:r1 + 1, c0:c1 + 1]
    if marker_color in sample or not (np.all(np.any(sample != background, axis=0))
                                     and np.all(np.any(sample != background, axis=1))):
        raise ValueError("fabric must have a tight rectangular window")
    h, w = sample.shape
    if h > 15 or w > 15:
        raise ValueError("sample would exceed output size bounds")
    marker_pos = np.argwhere(inp == marker_color)
    mr0, mc0 = map(int, marker_pos.min(axis=0))
    mr1, mc1 = map(int, marker_pos.max(axis=0))
    if not (mr1 <= r0 - 2 or r1 <= mr0 - 2 or mc1 <= c0 - 2 or c1 <= mc0 - 2):
        raise ValueError("marker and fabric need a background gap between their boxes")
    p, q = _axis_period(sample, 0), _axis_period(sample, 1)
    if min(p, q) < 2:
        raise ValueError("both tile axes must have nontrivial periods")
    tile = sample[:p, :q]
    if (np.array_equal(tile, tile[::-1]) or np.array_equal(tile, tile[:, ::-1])):
        raise ValueError("the tile must show both reflection orientations")
    if not np.array_equal(sample, tile[np.arange(h)[:, None] % p,
                                       np.arange(w)[None, :] % q]):
        raise ValueError("sample is not a rectangular periodic texture")
    foreground = set(map(int, np.unique(sample))) - {background}
    if not 2 <= len(foreground) <= 4:
        raise ValueError("fabric needs two to four foreground colors")
    return {"background": background, "marker": marker_color,
            "long_direction": long_direction, "side": (long_direction - 1) % 4,
            "junction": junction, "lengths": lengths,
            "bbox": (r0, c0, h, w), "marker_bbox": (mr0, mc0, mr1, mc1),
            "sample": sample, "tile": tile, "p": p, "q": q,
            "foreground_count": len(foreground), "gap": background in sample}


def _double_tile(tile, side):
    if side == 1:
        return np.concatenate((tile, tile[:, ::-1]), axis=1)
    if side == 3:
        return np.concatenate((tile[:, ::-1], tile), axis=1)
    if side == 2:
        return np.concatenate((tile, tile[::-1]), axis=0)
    return np.concatenate((tile[::-1], tile), axis=0)


def _out_shape(h, w, side):
    return (h, 2 * w) if side % 2 else (2 * h, w)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every role, period, and phase solely from the supplied input."""
    info = _analyze_input(input_grid)
    h, w = info["sample"].shape
    unit = _double_tile(info["tile"], info["side"])
    rows, cols = _out_shape(h, w, info["side"])
    return unit[np.arange(rows)[:, None] % unit.shape[0],
                np.arange(cols)[None, :] % unit.shape[1]].copy()


def _make_tile(rng, p, q, foreground, background, gap):
    symbols = list(foreground) + ([background] if gap else [])
    for _ in range(512):
        values = symbols + list(map(int, rng.choice(symbols, p * q - len(symbols))))
        rng.shuffle(values)
        tile = np.asarray(values, dtype=np.int8).reshape(p, q)
        # Reflective symmetry depends on the chosen cut through a cyclic tile.
        tile = np.roll(tile, (int(rng.integers(p)), int(rng.integers(q))), axis=(0, 1))
        if not (np.all(np.any(tile != background, axis=0))
                and np.all(np.any(tile != background, axis=1))):
            continue
        if np.array_equal(tile, tile[::-1]) or np.array_equal(tile, tile[:, ::-1]):
            continue
        twice = np.tile(tile, (2, 2))
        if _axis_period(twice, 0) != p or _axis_period(twice, 1) != q:
            continue
        # The observable corner, not an external phase label, defines the tile.
        return tile
    raise RuntimeError("could not sample an asymmetric primitive tile")


def _extent(rng, period, mode):
    options = [n for n in range(2 * period, 16)
               if mode == "free" or (n % period != 0) == (mode == "partial")]
    return int(rng.choice(options))


def _make_marker(rng, long_direction):
    longest = int(rng.integers(2, 6))
    lengths = [int(rng.integers(1, min(3, longest - 1) + 1)) for _ in range(4)]
    lengths[long_direction] = longest
    if rng.random() < 0.5:
        shorts = [d for d in range(4) if d != long_direction]
        lengths[int(rng.choice(shorts))] = 0
    up, right, down, left = lengths
    mask = np.zeros((up + down + 1, left + right + 1), dtype=bool)
    mask[up, left] = True
    for direction, length in enumerate(lengths):
        dr, dc = _DELTAS[direction]
        for k in range(1, length + 1):
            mask[up + dr * k, left + dc * k] = True
    return mask


def _pack(rng, sample, marker_mask, background, marker_color):
    """Random packing direction is unrelated to the semantic marker direction."""
    h, w = sample.shape
    mh, mw = marker_mask.shape
    packing = int(rng.integers(4))  # marker above/right/below/left of sample
    gap = int(rng.integers(1, 4))
    vertical = packing % 2 == 0
    minimum_h = h + mh + gap + 2 if vertical else max(h, mh) + 2
    minimum_w = max(w, mw) + 2 if vertical else w + mw + gap + 2
    rows = minimum_h + int(rng.integers(min(6, 30 - minimum_h) + 1))
    cols = minimum_w + int(rng.integers(min(8, 30 - minimum_w) + 1))
    if vertical:
        extra = rows - 2 - h - mh - gap
        leading = int(rng.integers(extra + 1))
        middle = int(rng.integers(extra - leading + 1))
        first = 1 + leading
        if packing == 0:
            mr, sr = first, first + mh + gap + middle
        else:
            sr, mr = first, first + h + gap + middle
        sc = int(rng.integers(1, cols - w))
        mc = int(rng.integers(1, cols - mw))
    else:
        extra = cols - 2 - w - mw - gap
        leading = int(rng.integers(extra + 1))
        middle = int(rng.integers(extra - leading + 1))
        first = 1 + leading
        if packing == 3:
            mc, sc = first, first + mw + gap + middle
        else:
            sc, mc = first, first + w + gap + middle
        sr = int(rng.integers(1, rows - h))
        mr = int(rng.integers(1, rows - mh))
    inp = np.full((rows, cols), background, dtype=np.int8)
    inp[sr:sr + h, sc:sc + w] = sample
    area = inp[mr:mr + mh, mc:mc + mw]
    area[marker_mask] = marker_color
    return inp


def _pair(rng, side, roles, mode):
    background, marker_color, *color_pool = roles
    p, q = map(int, rng.integers(2, 6, size=2))
    if mode == "horizontal_evidence":
        h, w = _extent(rng, p, "complete"), _extent(rng, q, "partial")
        gap = False
    elif mode == "vertical_evidence":
        h, w = _extent(rng, p, "partial"), _extent(rng, q, "complete")
        gap = True
    elif mode == "combined":
        h, w = _extent(rng, p, "partial"), _extent(rng, q, "partial")
        gap = True
    else:
        h, w = _extent(rng, p, "free"), _extent(rng, q, "free")
        gap = bool(rng.integers(2))
    count = int(rng.integers(2, min(4, p * q - int(gap)) + 1))
    foreground = tuple(map(int, rng.choice(color_pool, count, replace=False)))
    tile = _make_tile(rng, p, q, foreground, background, gap)
    sample = np.tile(tile, ((h + p - 1) // p, (w + q - 1) // q))[:h, :w].copy()
    marker_mask = _make_marker(rng, (side + 1) % 4)
    inp = _pack(rng, sample, marker_mask, background, marker_color)
    unit = _double_tile(tile, side)
    oh, ow = _out_shape(h, w, side)
    out = np.tile(unit, ((oh + unit.shape[0] - 1) // unit.shape[0],
                         (ow + unit.shape[1] - 1) // unit.shape[1]))[:oh, :ow].copy()
    return {"input": inp, "output": out}


def _direction_plan(rng, num_train, num_test):
    horizontal = int(rng.choice((1, 3)))
    vertical = int(rng.choice((0, 2)))
    basis = [horizontal, vertical]
    opposite = [(s + 2) % 4 for s in basis]
    modes = ["heldout", "shared"] + (["full"] if num_train >= 4 else [])
    mode = str(rng.choice(modes))
    train = basis + (opposite if mode == "full" else [])
    while len(train) < num_train:
        train.append(int(rng.choice(range(4) if mode == "full" else basis)))
    pool = opposite if mode == "heldout" else list(range(4)) if mode == "full" else basis
    pool = list(map(int, rng.permutation(pool)))
    test = [pool[i % len(pool)] for i in range(num_test)]
    return train, test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 6, replace=False)))
    varying_palette = bool(rng.integers(2))
    train_sides, test_sides = _direction_plan(rng, num_train, num_test)
    puzzle, used = {"train": [], "test": []}, set()
    for split, sides in (("train", train_sides), ("test", test_sides)):
        for index, side in enumerate(sides):
            global_index = index if split == "train" else num_train + index
            if not varying_palette or global_index == 0:
                roles = base
            elif global_index == 1:
                roles = base[2:] + base[:2]  # visibly reassign color roles in training
            elif rng.random() < 0.5:
                roles = tuple(map(int, rng.choice(10, 6, replace=False)))
            else:
                roles = tuple(map(int, rng.permutation(base)))
            mode = ("horizontal_evidence" if index == 0 else "vertical_evidence"
                    if index == 1 else "free") if split == "train" else (
                        "combined" if index == 0 else "free")
            for _ in range(64):
                pair = _pair(rng, side, roles, mode)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _whole_mirror(sample, side):
    """Ablation: use the complete finite window as the unit, omitting A."""
    return _double_tile(sample, side)


def _continue_old(tile, shape):
    """Ablation: continue without any mirror, omitting B's unfolding."""
    h, w = shape
    return tile[np.arange(h)[:, None] % tile.shape[0],
                np.arange(w)[None, :] % tile.shape[1]]


def _check_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _analyze_input(inp)
    h, w = info["sample"].shape
    p, q, side = info["p"], info["q"], info["side"]
    tile = info["tile"]
    assert 2 <= p <= 5 and 2 <= q <= 5
    assert out.shape == _out_shape(h, w, side)
    # A coordinate-level check, separate from the concatenation/tile renderer.
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if side % 2:
                x = c % (2 * q)
                j = (x if x < q else 2 * q - 1 - x) if side == 1 else (
                    q - 1 - x if x < q else x - q)
                i = r % p
            else:
                y = r % (2 * p)
                i = (y if y < p else 2 * p - 1 - y) if side == 2 else (
                    p - 1 - y if y < p else y - p)
                j = c % q
            assert int(out[r, c]) == int(tile[i, j]), "wrong unit reflection or phase"
    assert set(map(int, np.unique(out))) == set(map(int, np.unique(info["sample"])))
    assert not np.any(out == info["marker"])
    assert not np.array_equal(out, _whole_mirror(info["sample"], side)), "A must matter"
    assert not np.array_equal(out, _continue_old(tile, out.shape)), "B must matter"
    # Opposite selected side changes doublet order, even if the output shape agrees.
    opposite_unit = _double_tile(tile, (side + 2) % 4)
    assert not np.array_equal(out, _continue_old(opposite_unit, out.shape))
    assert out.shape != _out_shape(h, w, info["long_direction"])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, profiles = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            profiles[split].append(_check_pair(pair))
    a, b = profiles["train"][:2]
    assert a["side"] % 2 == 1 and a["bbox"][3] % a["q"] != 0
    assert a["bbox"][2] % a["p"] == 0 and not a["gap"]
    assert b["side"] % 2 == 0 and b["bbox"][2] % b["p"] != 0
    assert b["bbox"][3] % b["q"] == 0 and b["gap"]
    t = profiles["test"][0]
    assert t["bbox"][2] % t["p"] != 0 and t["bbox"][3] % t["q"] != 0
    assert t["gap"]


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _canonical(arr):
    colors = {}
    return tuple(colors.setdefault(int(c), len(colors)) for c in arr.flat)


def _literal_examples():
    # Literal targets make direction/order checks independent of the renderer.
    a, b = "1231231", "2312312"
    sample = np.array([[int(c) for c in row] for row in (a, b, a, b, a)])
    right_a, right_b = "12332112332112", "23113223113223"
    left_a, left_b = "32112332112332", "13223113223113"
    targets = {
        1: [right_a, right_b, right_a, right_b, right_a],
        3: [left_a, left_b, left_a, left_b, left_a],
        2: [a, b, b, a, a, b, b, a, a, b],
        0: [b, a, a, b, b, a, a, b, b, a],
    }
    for side, rows in targets.items():
        inp = np.zeros((13, 18), dtype=int)
        inp[1:6, 1:8] = sample
        lengths = [1, 1, 1, 1]
        lengths[(side + 1) % 4] = 2
        inp[9, 12] = 9
        for d, length in enumerate(lengths):
            dr, dc = _DELTAS[d]
            for k in range(1, length + 1):
                inp[9 + k * dr, 12 + k * dc] = 9
        expected = np.array([[int(c) for c in row] for row in rows])
        original = inp.copy()
        assert np.array_equal(solve(inp), expected)
        assert np.array_equal(inp, original), "solve mutated its input"


def self_test():
    """Bounded 42-seed structural sweep, semantic mutations, and API checks."""
    _literal_examples()
    puzzles, shapes, tiles, periods, sides, arms = set(), set(), set(), set(), set(), set()
    backgrounds, markers, color_counts, layouts = set(), set(), set(), set()
    partial, complete, gaps, no_gaps = 0, 0, 0, 0
    heldout = shared = varying = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        split_sides = {"train": set(), "test": set()}
        roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                before = pair["input"].copy()
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(before, pair["input"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                tiles.add((info["tile"].shape, _canonical(info["tile"])))
                periods.add((info["p"], info["q"]))
                sides.add(info["side"])
                split_sides[split].add(info["side"])
                roles.add((info["background"], info["marker"]))
                backgrounds.add(info["background"])
                markers.add(info["marker"])
                color_counts.add(info["foreground_count"])
                arms.add((info["long_direction"], info["lengths"]))
                r, c, h, w = info["bbox"]
                mr0, mc0, mr1, mc1 = info["marker_bbox"]
                layouts.add((mr1 < r, mc0 > c + w - 1, mr0 > r + h - 1, mc1 < c))
                partial += h % info["p"] != 0 or w % info["q"] != 0
                complete += h % info["p"] == 0 and w % info["q"] == 0
                gaps += bool(info["gap"])
                no_gaps += not info["gap"]
        heldout += bool(split_sides["test"] - split_sides["train"])
        shared += bool(split_sides["test"] & split_sides["train"])
        varying += len(roles) > 1
    assert len(puzzles) == 42
    assert len(shapes) >= 80 and len(tiles) >= 100 and len(periods) >= 12
    assert sides == {0, 1, 2, 3} and len(arms) >= 30
    assert len(backgrounds) >= 7 and len(markers) >= 7 and color_counts == {2, 3, 4}
    assert len(layouts) >= 4 and min(partial, complete, gaps, no_gaps) > 0
    assert min(heldout, shared, varying) >= 5
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(202600 + nt * 10 + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
            validate_generated_puzzle(puzzle)
    assert _puzzle_key(generate_puzzle(81723, 7, 3)) == _puzzle_key(generate_puzzle(81723, 7, 3))
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    illegal = ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
               (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None))
    for nt, ne in illegal:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {nt}/{ne}")
    for bad_seed in (-1, True, 0.0):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an invalid seed")
    # Require semantic rejection, not merely shape/schema checks.
    for ablation in ("whole", "old", "opposite"):
        puzzle = generate_puzzle(0, 2, 1)
        pair = puzzle["train"][0]
        info = _analyze_input(pair["input"])
        pair["output"] = (
            _whole_mirror(info["sample"], info["side"]) if ablation == "whole" else
            _continue_old(info["tile"], pair["output"].shape) if ablation == "old" else
            _continue_old(_double_tile(info["tile"], (info["side"] + 2) % 4),
                          pair["output"].shape))
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {ablation} ablation")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "shape_pairs": len(shapes),
            "normalized_tiles": len(tiles), "period_pairs": len(periods),
            "marker_variants": len(arms), "heldout_direction_puzzles": heldout,
            "shared_direction_puzzles": shared, "varying_palette_puzzles": varying,
            "pairs_with_partial_cycles": partial, "complete_cycle_pairs": complete}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
