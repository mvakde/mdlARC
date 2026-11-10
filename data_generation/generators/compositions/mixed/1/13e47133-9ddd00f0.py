"""Row-key contour lattices with coordinate-indexed holes.

No parent modules, files, random state, or training examples are used by solve.
Generation and all tests are bounded and keep their grids in memory.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= d <= 30 for d in array.shape), "Invalid grid dimensions")
    _require(np.issubdtype(array.dtype, np.integer), "Grid must have integer cells")
    _require(bool(np.all((0 <= array) & (array <= 9))), "Invalid grid color")
    return array.astype(np.int8, copy=False)


def _order(shape):
    matches = [n for n in (3, 4, 5) if shape == (n * n + n - 1,) * 2]
    _require(len(matches) == 1, "Expected a square order-3, 4, or 5 lattice")
    return matches[0]


def _separators(n):
    side = n * n + n - 1
    mask = np.zeros((side, side), dtype=bool)
    mask[n::n + 1, :] = True
    mask[:, n::n + 1] = True
    return mask


def _slice(n, row, col):
    a, b = row * (n + 1), col * (n + 1)
    return np.s_[a:a + n, b:b + n]


def _depths(n):
    rr, cc = np.indices((n, n))
    return np.minimum.reduce((rr, cc, n - 1 - rr, n - 1 - cc))


def _eroded_depths(n):
    """Check construction with literal eight-neighbor erosion, before punching."""
    current = np.ones((n, n), dtype=bool)
    result = np.full((n, n), -1, dtype=np.int8)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inside = current.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                inside &= padded[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]
        result[current & ~inside] = level
        current = inside
        level += 1
    return result


def _path(n, corner, length):
    # 0 = upper left, 1 = upper right, 2 = lower left, 3 = lower right.
    down, right = corner < 2, corner % 2 == 0
    return [(d if down else n - 1 - d, d if right else n - 1 - d)
            for d in range(length)]


def _decode_key(tile, neutral):
    n = tile.shape[0]
    points = {tuple(map(int, p)) for p in np.argwhere(tile != neutral)}
    length = len(points)
    _require(1 <= length <= (n + 1) // 2, "Invalid diagonal-key length")
    candidates = []
    for corner in range(4):
        path = _path(n, corner, length)
        if set(path) == points:
            candidates.append((tuple(int(tile[p]) for p in path), corner))
    _require(bool(candidates), "Sparse tile is not a consecutive inward diagonal key")
    periods = {p for p, _ in candidates}
    _require(len(periods) == 1, "Ambiguous diagonal key")
    period = next(iter(periods))
    _require(len(set(period)) == len(period), "Key entries must be distinct")
    return period, candidates[0][1]


def _tile(n, period, neutral, row, col):
    tile = np.asarray(period, dtype=np.int8)[_depths(n) % len(period)]
    tile[row, col] = neutral
    return tile


def _analyze_input(input_grid):
    inp = _grid(input_grid)
    n = _order(inp.shape)
    separators = _separators(n)
    colors = np.unique(inp[separators])
    _require(len(colors) == 1, "Damaged separator lattice")
    divider = int(colors[0])
    _require(not np.any(inp[~separators] == divider), "Divider color inside a compartment")
    uniform = set()
    for row in range(n):
        for col in range(n):
            tile = inp[_slice(n, row, col)]
            if np.all(tile == tile[0, 0]):
                uniform.add(int(tile[0, 0]))
    _require(len(uniform) == 1, "Blank compartments must identify one neutral color")
    neutral = next(iter(uniform))
    _require(neutral != divider, "Neutral and divider must differ")
    states = np.zeros((n, n), dtype=np.int8)  # 0 blank, 1 key, 2 complete.
    periods, corners = [], []
    for row in range(n):
        row_periods = set()
        for col in range(n):
            tile = inp[_slice(n, row, col)]
            colored = int(np.count_nonzero(tile != neutral))
            if colored == 0:
                continue
            if colored == n * n - 1:
                states[row, col] = 2
                continue
            period, corner = _decode_key(tile, neutral)
            states[row, col] = 1
            row_periods.add(period)
            corners.append((row, col, corner))
        _require(len(row_periods) == 1, "Each row needs agreeing explicit diagonal keys")
        _require(bool(np.any(states[row] == 0)), "Each row needs a blank compartment")
        periods.append(next(iter(row_periods)))
    _require(bool(np.any(states == 2)), "At least one completed witness is required")
    _require(len({p[0] for p in periods}) == n, "Rows need distinct outer band colors")
    _require(sum(len(p) >= 2 for p in periods) >= 2, "At least two rows need nontrivial bands")
    for row, col in np.argwhere(states == 2):
        _require(np.array_equal(inp[_slice(n, row, col)],
                                _tile(n, periods[row], neutral, row, col)),
                 "Completed compartment disagrees with its row key or coordinate hole")
    return {"n": n, "neutral": neutral, "divider": divider,
            "periods": tuple(periods), "states": states, "corners": tuple(corners)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer lattice, roles, row keys and holes entirely from the given input."""
    info = _analyze_input(input_grid)
    n, neutral = info["n"], info["neutral"]
    out = np.full((n * n + n - 1,) * 2, info["divider"], dtype=np.int8)
    for row in range(n):
        for col in range(n):
            out[_slice(n, row, col)] = _tile(n, info["periods"][row], neutral, row, col)
    return out


def _periods(rng, n, neutral, divider, anchor):
    palette = [c for c in range(10) if c not in (neutral, divider)]
    outer = rng.permutation(palette)[:n]
    if anchor:
        lengths = list(map(int, rng.permutation([1, 2, 2, 2, 3])))
        # A key at (1,1), pointing down-right, will have its second cue punched.
        swap = lengths.index(2)
        lengths[1], lengths[swap] = lengths[swap], lengths[1]
    else:
        lengths = list(map(int, rng.integers(1, (n + 1) // 2 + 1, size=n)))
        for index in rng.choice(n, size=2, replace=False):
            lengths[int(index)] = 2
    result = []
    for row, length in enumerate(lengths):
        inner = rng.choice([c for c in palette if c != outer[row]], size=length - 1, replace=False)
        result.append((int(outer[row]),) + tuple(map(int, inner)))
    return tuple(result)


def _states(rng, n, regime):
    states = np.zeros((n, n), dtype=np.int8)
    if regime == "anchor":
        shift = int(rng.integers(1, n))
        for row in range(n):
            complete_col = (row + shift) % n
            key_col = 1 if row == 1 else int(rng.choice([c for c in range(n) if c != complete_col]))
            blank_col = int(rng.choice([c for c in range(n) if c not in (key_col, complete_col)]))
            states[row] = 2
            states[row, key_col], states[row, blank_col] = 1, 0
            for col in range(n):
                if col not in (complete_col, key_col, blank_col) and rng.random() < 0.3:
                    states[row, col] = 1
    else:
        complete_rows = set(map(int, rng.choice(n, size=1 if regime == "test"
                                               else int(rng.integers(1, n)), replace=False)))
        for row in range(n):
            complete_col = int(rng.integers(n)) if row in complete_rows else None
            available = [c for c in range(n) if c != complete_col]
            key_count = 1 if regime == "test" else int(rng.integers(1, len(available)))
            for col in rng.choice(available, size=key_count, replace=False):
                states[row, int(col)] = 1
            if complete_col is not None:
                states[row, complete_col] = 2
    return states


def _make_pair(rng, n, neutral, divider, regime):
    periods = _periods(rng, n, neutral, divider, regime == "anchor")
    states = _states(rng, n, regime)
    side = n * n + n - 1
    inp = np.full((side, side), neutral, dtype=np.int8)
    out = np.full_like(inp, divider)
    inp[_separators(n)] = divider
    for row in range(n):
        key_index = 0
        for col in range(n):
            location = _slice(n, row, col)
            target = _tile(n, periods[row], neutral, row, col)
            out[location] = target
            if states[row, col] == 2:
                inp[location] = target
            elif states[row, col] == 1:
                if regime == "anchor":
                    corner = 0 if (row, col) == (1, 1) else int(rng.integers(2))
                elif regime == "train" and key_index == 0:
                    corner = int(rng.integers(2, 4))
                else:
                    corner = int(rng.integers(4))
                tile = inp[location]
                for point, color in zip(_path(n, corner, len(periods[row])), periods[row]):
                    tile[point] = color
                key_index += 1
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    puzzle = {"train": [], "test": []}
    # Up to 14 distinct ordered role pairs guarantee distinct input grids.
    for index in range(nt + ne):
        neutral = colors[index % 10]
        divider = colors[(index + 1 + index // 10) % 10]
        regime = "anchor" if index == 0 else "train" if index < nt else "test"
        n = 5 if regime == "anchor" else int(rng.choice((3, 4, 5)))
        pair = _make_pair(rng, n, neutral, divider, regime)
        puzzle["train" if index < nt else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "Input/output dimensions differ")
    info = _analyze_input(inp)
    n, neutral, divider = info["n"], info["neutral"], info["divider"]
    sep = _separators(n)
    _require(np.array_equal(inp[sep], out[sep]), "Separator changed")
    _require(not np.any(out[~sep] == divider), "Divider leaked into tiles")
    depths = _eroded_depths(n)
    _require(np.array_equal(depths, _depths(n)), "Depth construction disagrees with erosion")
    without_a, without_b = inp.copy(), inp.copy()
    for row in range(n):
        period = np.asarray(info["periods"][row], dtype=np.int8)
        for col in range(n):
            location = _slice(n, row, col)
            expected = period[depths % len(period)]
            expected[row, col] = neutral
            actual = out[location]
            _require(np.array_equal(actual, expected), "Incorrect bands or indexed hole")
            _require(int(np.count_nonzero(actual == neutral)) == 1, "Not exactly one neutral hole")
            if info["states"][row, col] == 2:
                _require(np.array_equal(inp[location], actual), "Completed tile changed")
                continue
            solid = np.full((n, n), period[0], dtype=np.int8)
            solid[row, col] = neutral
            without_a[location] = solid
            common_hole = period[depths % len(period)]
            common_hole[0, 0] = neutral
            without_b[location] = common_hole
    _require(not np.array_equal(inp, out), "Identity pair")
    _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "Unseen output color")
    info["a_ablation_cells"] = int(np.count_nonzero(without_a != out))
    info["b_ablation_cells"] = int(np.count_nonzero(without_b != out))
    _require(info["a_ablation_cells"] > 0, "Ignoring A does not change answer")
    _require(info["b_ablation_cells"] > 0, "Ignoring B does not change answer")
    _require(np.array_equal(solve(inp), out), "solve disagrees")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    _require(any(a["n"] == 5 and any(len(p) == 2 for p in a["periods"])
                 and any(len(p) == 3 for p in a["periods"]) for a in train),
             "Training must demonstrate cycle repetition and a three-entry key")
    _require(any(any(not np.any(a["states"][r] == 2) for r in range(a["n"])) for a in train),
             "Training must show row-key reconstruction without a completed donor")
    _require(any(any(corner >= 2 and len(a["periods"][r]) >= 2
                         for r, _, corner in a["corners"]) for a in train),
             "Training needs a lower-corner multi-entry key")
    _require(any(np.any((np.asarray(pair["input"]) != a["neutral"])
                        & (np.asarray(pair["output"]) == a["neutral"]))
                 for pair, a in zip(puzzle["train"], train)), "Training must show a punched key cue")
    for a in infos["test"]:
        _require(np.count_nonzero(a["states"] == 2) == 1, "Test needs one completed witness")
        _require(np.all(np.sum(a["states"] == 1, axis=1) == 1), "Test needs one key per row")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def self_test():
    """Forty-two streamed seeds; bounded hashes/metrics, never saved grids/traces."""
    puzzles, layouts, normalized, shapes, role_pairs = (set() for _ in range(5))
    lengths, orientations, counts = set(), set(), set()
    events = {k: 0 for k in ("repeat_rows", "three_color_rows", "donorless_rows",
                            "punched_key_pairs", "a_ablation_cells", "b_ablation_cells")}
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "Nondeterministic seed")
        puzzles.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                n = info["n"]
                shapes.add(pair["input"].shape)
                role_pairs.add((info["neutral"], info["divider"]))
                layouts.add((n, info["states"].tobytes()))
                mapping = {}
                canonical = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                normalized.add((pair["input"].shape, hashlib.sha256(canonical).hexdigest()))
                lengths.update(map(len, info["periods"]))
                orientations.update(corner for r, _, corner in info["corners"] if len(info["periods"][r]) >= 2)
                counts.add(tuple(int(np.count_nonzero(info["states"] == k)) for k in range(3)))
                events["repeat_rows"] += sum(n == 5 and len(p) == 2 for p in info["periods"])
                events["three_color_rows"] += sum(len(p) == 3 for p in info["periods"])
                events["donorless_rows"] += int(np.count_nonzero(~np.any(info["states"] == 2, axis=1)))
                events["punched_key_pairs"] += bool(np.any((pair["input"] != info["neutral"])
                                                            & (pair["output"] == info["neutral"])))
                events["a_ablation_cells"] += info["a_ablation_cells"]
                events["b_ablation_cells"] += info["b_ablation_cells"]
                pairs += 1
    _require(len(puzzles) == 42 and len(layouts) >= 100 and len(normalized) >= 150,
             "Insufficient structural diversity")
    _require(shapes == {(11, 11), (19, 19), (29, 29)}, "Missing supported scale")
    _require(lengths == {1, 2, 3} and orientations == {0, 1, 2, 3}, "Missing key variation")
    _require(len(role_pairs) >= 50 and len(counts) >= 30, "Insufficient roles or object counts")
    _require(all(events.values()), "Missing family-specific witness")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9173, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "Wrong explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, 1.5)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    sample = generate_puzzle(810, 2, 1)
    original = _digest(sample)
    palette = np.random.default_rng(81).permutation(10)
    for split in ("train", "test"):
        for pair in sample[split]:
            _require(np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]]),
                     "Solver depends on literal colors")
    _require(_digest(sample) == original, "Solver mutated input")
    # Returned-data validator must reject separator, band, and hole corruption.
    info = _analyze_input(sample["test"][0]["input"])
    row, col = map(int, np.argwhere(info["states"] == 0)[0])
    a, b = row * (info["n"] + 1), col * (info["n"] + 1)
    for position, color in (((info["n"], 0), info["neutral"]),
                            ((a + row, b + col), info["periods"][row][0]),
                            ((a, b), info["divider"])):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
        broken["test"][0]["output"][position] = color
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "distinct_state_layouts": len(layouts), "normalized_inputs": len(normalized),
            "shapes": sorted(shapes), "ordered_role_pairs": len(role_pairs),
            "state_count_profiles": len(counts), "events": events,
            "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
