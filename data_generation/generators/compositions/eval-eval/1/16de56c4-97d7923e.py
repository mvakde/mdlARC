"""Rhythm capacity: unary ranks compare virtual, phase-clipped bead counts.

Only NumPy and the standard library are used. No file I/O, saved examples,
generation records, seed, or puzzle state are consulted by solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 294797655, 947208623, 905028952, 297727092, 945514812,
              39056075, 214043393, 553424690, 366071213, 513808157,
              394018742, 808501596, 166785229, 864599651, 651973940,
              331329033, 312708436, 955118195, 273033850, 82783663,
              675057478, 478039178, 300404802, 480753350, 102857753,
              655197440, 458868063, 601590539, 249412345, 460523264,
              368471305, 717485802, 555167197, 526290677, 485883351,
              765094580, 666957198, 271802885, 545155082, 574702357,
              749541595)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if counts.max() * 2 <= grid.size:
        raise ValueError("input must have a strict majority background")
    return int(colors[np.argmax(counts)])


def _parse_rows(grid, bg):
    """Parse either horizontal rows or the transposed vertical scene."""
    keys, rods = {}, []
    key_rows, rod_rows = [], []
    for row, values in enumerate(grid):
        occupied = np.flatnonzero(values != bg)
        if not len(occupied):
            continue
        colors = values[occupied]
        if np.all(colors == colors[0]):
            cap = int(colors[0])
            if cap in keys or not np.all(np.diff(occupied) == 1):
                raise ValueError("key needs one contiguous unary run per cap")
            keys[cap] = len(occupied)
            key_rows.append(row)
            continue
        if len(occupied) != 4:
            raise ValueError("each rod has two caps and two reference beads")
        left, a, b, right = map(int, occupied)
        cap, bead = int(colors[0]), int(colors[1])
        if cap != int(colors[3]) or bead != int(colors[2]) or cap == bead:
            raise ValueError("matching outer caps and matching inner beads required")
        period = b - a
        sites = tuple(range(left + 1 + (a - left - 1) % period, right, period))
        if len(sites) < 2:
            raise ValueError("rod has fewer than two phase sites")
        rods.append({"row": row, "left": left, "right": right, "a": a,
                     "b": b, "cap": cap, "bead": bead, "period": period,
                     "sites": sites, "capacity": len(sites)})
        rod_rows.append(row)
    if not keys or not rods:
        raise ValueError("missing key or rods")
    # The key is a small band, separated from all rods by a blank line.
    if not (max(key_rows) + 1 < min(rod_rows)
            or max(rod_rows) + 1 < min(key_rows)):
        raise ValueError("key band must be separate from the rods")
    if max(key_rows) - min(key_rows) > 2 * len(keys) - 2:
        raise ValueError("key band is scattered")
    if {r["cap"] for r in rods} != set(keys):
        raise ValueError("every cap family needs exactly one matching key")
    if {r["bead"] for r in rods} & set(keys):
        raise ValueError("bead and cap palettes must be disjoint within an input")
    families = {cap: [r for r in rods if r["cap"] == cap] for cap in keys}
    selected = []
    for cap, family in families.items():
        if len(family) < 2 or len({r["capacity"] for r in family}) != len(family):
            raise ValueError("families need competitors with distinct capacities")
        if not 1 <= keys[cap] <= len(family):
            raise ValueError("unary rank does not exist")
        selected.append(sorted(family, key=lambda r: -r["capacity"])[keys[cap] - 1])
    return {"background": bg, "keys": keys, "key_rows": key_rows,
            "rods": rods, "families": families, "selected": selected}


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = _background(grid)
    candidates = []
    for transposed in (False, True):
        rows = grid.T if transposed else grid
        try:
            data = _parse_rows(rows, bg)
        except ValueError:
            continue
        candidates.append((transposed, rows, data))
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous common rod axis")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count virtual beads, rank by cap family, complete only the winners."""
    transposed, rows, data = _decode(input_grid)
    result = rows.copy()
    for rod in data["selected"]:
        result[rod["row"], list(rod["sites"])] = rod["cap"]
    return np.ascontiguousarray(result.T if transposed else result)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _geometry_options(capacity, *, period=None, shorter=None, longer=None):
    # A geometry is (interior length, period, first phase site relative to cap).
    # <=26 interior cells leaves room for both end caps in a <=30-cell canvas.
    options = []
    for d in ([period] if period is not None else range(1, 14)):
        for first in range(1, d + 1):
            last = first + (capacity - 1) * d
            for interior in range(last, min(26, last + d - 1) + 1):
                if shorter is not None and interior >= shorter:
                    continue
                if longer is not None and interior <= longer:
                    continue
                options.append((interior, d, first))
    return options


def _rod(rng, family, capacity, geometry=None, middle=False):
    interior, period, first = geometry or _pick(rng, _geometry_options(capacity))
    # The two reference beads are consecutive members of the hidden lattice.
    index = int(rng.integers(1, capacity - 2)) if middle and capacity >= 4 else int(rng.integers(capacity - 1))
    return {"family": family, "capacity": capacity, "interior": interior,
            "period": period, "first": first, "ref_index": index}


def _witness_rods(rng, rank):
    # Four competitors: equal-span/equal-period phase twins, a physically
    # longer but sparser loser, and a physically shorter/dense loser.
    options = []
    for high in range(5, 8):
        for d in range(2, 5):
            for remainder in range(1, d):
                interior = (high - 1) * d + remainder
                if interior > 24:
                    continue
                long_options = [g for g in _geometry_options(high - 2, longer=interior) if g[1] > d]
                if long_options:
                    options.append((high, d, remainder, interior, long_options))
    high, d, remainder, interior, long_options = _pick(rng, options)
    hi = _rod(rng, 0, high, (interior, d, int(rng.integers(1, remainder + 1))), True)
    lo = _rod(rng, 0, high - 1, (interior, d, int(rng.integers(remainder + 1, d + 1))), True)
    long = _rod(rng, 0, high - 2, _pick(rng, long_options))
    short = _rod(rng, 0, high - 3, (high - 3, 1, 1))
    rods = [hi, lo, long, short]
    rng.shuffle(rods)
    # Rank one has an interior spatial position, refuting either spatial order.
    if rank == 1:
        index = next(i for i, r in enumerate(rods) if r is hi)
        middle = int(rng.integers(1, 3))
        rods[index], rods[middle] = rods[middle], rods[index]
    return rods


def _arrange(rng, families, mode):
    # Preserve each already-shuffled family's order while interleaving.
    order = list(map(int, rng.permutation(len(families))))
    if mode == "grouped":
        return [rod for f in order for rod in families[f]]
    cursors = [0] * len(families)
    result = []
    while len(result) < sum(map(len, families)):
        active = [f for f in order if cursors[f] < len(families[f])]
        f = int(_pick(rng, active))
        take = 1 if mode == "mixed" else int(rng.integers(1, len(families[f]) - cursors[f] + 1))
        result.extend(families[f][cursors[f]:cursors[f] + take])
        cursors[f] += take
    if mode == "mixed" and len(families) >= 2:
        sequence = [r["family"] for r in result]
        split = sum(sequence.index(f) + sequence[::-1].index(f) + len(families[f]) < len(sequence)
                    for f in order)
        if split < 2:
            # Guarantee actual family interleaving, retaining each family's
            # internal order and thus the interior-position teaching witness.
            a, b = order[:2]
            result = [families[a][0], families[b][0], families[a][1], families[b][1]]
            result += [r for f in order for r in families[f][2 if f in (a, b) else 0:]]
    return result


def _build_pair(rng, settings, witness, target, trained_ranks):
    population_shift = settings["shift"] == "population"
    limit = 4 if not target and population_shift else 5
    minimum_families = 2 if witness == 1 or (target and settings["shift"] == "mixing") else 1
    f = int(rng.integers(minimum_families, settings["max_families"] + 1))
    families, ranks = [], []
    rank_pool = list(sorted(trained_ranks)) if target else [1, 2, 3]
    for family in range(f):
        if family == 0 and witness is not None:
            rank = witness
            rods = _witness_rods(rng, rank)
        else:
            count = 5 if target and population_shift and family == 0 else int(rng.integers(2, limit + 1))
            available_ranks = [r for r in rank_pool if r <= count]
            rank = int(_pick(rng, available_ranks))
            capacities = list(map(int, rng.choice(np.arange(3, 14), count, replace=False)))
            rods = [_rod(rng, family, c) for c in capacities]
            # A second cap family teaches grouping and the period-one limit.
            if witness == 1 and family == 1:
                rank = int(rng.integers(1, min(2, count) + 1))
                chosen = sorted(rods, key=lambda r: -r["capacity"])[rank - 1]
                chosen.update(interior=chosen["capacity"], period=1, first=1, ref_index=0)
            rng.shuffle(rods)
        ranks.append(rank)
        families.append(rods)

    palette = list(settings["palette"])
    anchor = palette[1]
    if settings["rename"]:
        choices = [c for c in range(10) if c != anchor]
        rng.shuffle(choices)
        palette = [choices[0], anchor] + choices[1:]
    bg, caps = palette[0], palette[1:f + 1]
    bead_pool = palette[f + 1:]
    # The teaching examples share a bead color across families; other scenes
    # allow one, several, or per-rod payload identities, independent of caps.
    classes = 1 if witness is not None else int(rng.integers(1, min(3, len(bead_pool)) + 1))
    bead_pool = bead_pool[:classes]
    mode = "grouped" if settings["shift"] == "mixing" and not target else str(_pick(rng, ("grouped", "mixed", "chunks")))
    if settings["shift"] == "mixing" and target:
        mode = "mixed"
    rods = _arrange(rng, families, mode)
    width = int(rng.integers(max(12, max(r["interior"] + 2 for r in rods)), 31))
    key_order = list(map(int, rng.permutation(f)))
    key_gaps = list(map(int, rng.integers(0, 2, max(0, f - 1))))
    # Canvas budget includes every key, rod, blank separator, gap and margin.
    base_height = f + sum(key_gaps) + 1 + len(rods)
    gap_budget = 30 - base_height
    rod_gaps = [0] * max(0, len(rods) - 1)
    for _ in range(int(rng.integers(0, min(gap_budget, len(rods)) + 1))):
        available = [i for i, g in enumerate(rod_gaps) if g < 2]
        if not available:
            break
        rod_gaps[int(_pick(rng, available))] += 1
    height = base_height + sum(rod_gaps)
    padding = int(rng.integers(0, min(3, 30 - height) + 1))
    top = int(rng.integers(padding + 1))
    grid = np.full((height + padding, width), bg, dtype=np.int8)
    output = grid.copy()
    key_start = int(rng.integers(width - max(ranks) - 2))
    row = top
    for i, family in enumerate(key_order):
        start = key_start + int(rng.integers(3))
        grid[row, start:start + ranks[family]] = caps[family]
        if i < f - 1:
            row += 1 + key_gaps[i]
    row += 2
    selected_ids = {id(sorted(family, key=lambda r: -r["capacity"])[ranks[i] - 1])
                    for i, family in enumerate(families)}
    for i, rod in enumerate(rods):
        interior, d, first = rod["interior"], rod["period"], rod["first"]
        left = int(rng.integers(width - interior - 1))
        right = left + interior + 1
        a = left + first + rod["ref_index"] * d
        cap, bead = caps[rod["family"]], int(_pick(rng, bead_pool))
        grid[row, [left, right]] = cap
        grid[row, [a, a + d]] = bead
        output[row] = grid[row]
        if id(rod) in selected_ids:
            # Label construction uses the sampled lattice, not solve().
            sites = left + first + np.arange(rod["capacity"]) * d
            output[row, sites] = cap
        if i < len(rods) - 1:
            row += 1 + rod_gaps[i]
    # Copy the key after constructing the rods. All other cells stay background.
    output[:top + f + sum(key_gaps)] = grid[:top + f + sum(key_gaps)]
    if bool(rng.integers(2)):
        grid, output = np.fliplr(grid), np.fliplr(output)
    rotation = settings["rotation"]
    if settings["shift"] == "orientation" and target:
        rotation = (rotation + 1) % 4
    elif settings["mixed_axes"] and settings["shift"] != "orientation":
        rotation = int(rng.integers(4))
    return {"input": np.ascontiguousarray(np.rot90(grid, rotation)),
            "output": np.ascontiguousarray(np.rot90(output, rotation))}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _input_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    settings = {"palette": list(map(int, rng.permutation(10))),
                "rotation": int(rng.integers(4)), "rename": bool(rng.integers(2)),
                "max_families": int(rng.integers(2, 4)),
                "mixed_axes": bool(rng.integers(2)),
                "shift": str(_pick(rng, ("population", "orientation", "mixing")))}
    witnesses = [None] * num_train
    i, j = map(int, rng.choice(num_train, 2, replace=False))
    witnesses[i], witnesses[j] = 1, 2
    puzzle, seen, trained_ranks = {"train": [], "test": []}, set(), set()
    # No retained retry history: each attempt is <= two 30x30 int8 arrays.
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = witnesses[index] if split == "train" else None
            for _ in range(32):
                pair = _build_pair(rng, settings, witness, split == "test", trained_ranks)
                key = _input_key(pair["input"])
                if key in seen:
                    continue
                try:
                    _, _, data = _decode(pair["input"])
                except ValueError:
                    continue
                if not np.array_equal(solve(pair["input"]), pair["output"]):
                    raise AssertionError("sampled lattice disagrees with input decode")
                break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                trained_ranks.update(data["keys"].values())
    validate_generated_puzzle(puzzle)
    return puzzle


def _selection_shortcuts(data):
    selected = {r["row"] for r in data["selected"]}
    shortcuts = {}
    for mode in ("span_up", "span_down", "period_up", "period_down",
                 "spatial_up", "spatial_down", "capacity_up", "always_first", "always_last"):
        choices = set()
        for cap, family in data["families"].items():
            rank = data["keys"][cap] - 1
            if mode.startswith("span"):
                value = lambda r: r["right"] - r["left"]
            elif mode.startswith("period"):
                value = lambda r: r["period"]
            elif mode.startswith("spatial"):
                value = lambda r: r["row"]
            else:
                value = lambda r: r["capacity"]
            ordered = sorted(family, key=value, reverse=mode.endswith("down") or mode.startswith("always"))
            if mode == "always_first":
                rank = 0
            elif mode == "always_last":
                rank = len(family) - 1
            choices.add(ordered[rank]["row"])
        shortcuts[mode] = choices == selected
    return shortcuts


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape changed")
    transposed, rows, data = _decode(inp)
    answer = out.T if transposed else out
    expected = solve(inp)
    _require(np.array_equal(out, expected), "incorrect capacity selection or periodic fill")
    _require(np.any(inp != out), "identity example")
    selected_rows = {r["row"] for r in data["selected"]}
    for r in data["rods"]:
        row = r["row"]
        _require(answer[row, r["left"]] == r["cap"] == answer[row, r["right"]], "caps changed")
        if row not in selected_rows:
            _require(np.array_equal(rows[row], answer[row]), "unselected rod changed")
        else:
            mask = np.ones(rows.shape[1], dtype=bool)
            mask[list(r["sites"])] = False
            _require(np.array_equal(rows[row, mask], answer[row, mask]), "off-phase or outside cell changed")
            _require(np.all(answer[row, list(r["sites"])] == r["cap"]), "winner not completed")
    data["transposed"] = transposed
    return data


def _layout(data):
    sequence = [r["cap"] for r in data["rods"]]
    runs = [cap for i, cap in enumerate(sequence) if i == 0 or cap != sequence[i - 1]]
    return len(runs) == len(data["families"]), sum(runs.count(cap) > 1 for cap in data["families"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid = _grid(pair["input"])
            key = _input_key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(_analyze(pair))
    train = analyses["train"]
    ranks = {rank for data in train for rank in data["keys"].values()}
    _require({1, 2} <= ranks, "training needs ranks one and two")
    cap_ranks = {}
    for data in train:
        for cap, rank in data["keys"].items():
            cap_ranks.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= values for values in cap_ranks.values()), "same cap needs a rank contrast")
    _require(any(1 < data["keys"][cap] < len(family) for data in train
                 for cap, family in data["families"].items()), "need a non-extreme rank")
    _require(any(len(data["families"]) > 1 and len({r["bead"] for r in data["rods"]}) == 1
                 for data in train), "training needs grouping despite shared bead hue")
    selected = [r for data in train for r in data["selected"]]
    _require(any(r["period"] == 1 for r in selected) and any(r["period"] > 1 for r in selected), "solid and dotted winners needed")
    _require(any(r["sites"][0] < r["a"] and r["sites"][-1] > r["b"] for r in selected), "need extrapolation in both directions")
    twins = any(a["right"] - a["left"] == b["right"] - b["left"]
                and a["period"] == b["period"] and a["capacity"] != b["capacity"]
                for data in train for family in data["families"].values() for a in family for b in family)
    _require(twins, "need a phase-sensitive capacity comparison")
    for shortcut in _selection_shortcuts(train[0]):
        _require(not all(_selection_shortcuts(data)[shortcut] for data in train), "surviving shortcut: " + shortcut)
    for data in analyses["test"]:
        _require(set(data["keys"].values()) <= ranks, "test rank not demonstrated")
    tests = analyses["test"]
    population_shift = (max(len(family) for data in train for family in data["families"].values()) <= 4
                        and all(max(map(len, data["families"].values())) == 5 for data in tests))
    train_axes = {data["transposed"] for data in train}
    test_axes = {data["transposed"] for data in tests}
    orientation_shift = len(train_axes) == len(test_axes) == 1 and train_axes != test_axes
    mixing_shift = all(_layout(data)[0] for data in train) and all(_layout(data)[1] >= 2 for data in tests)
    _require(population_shift or orientation_shift or mixing_shift, "no demonstrated train/test transfer regime")
    return None


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42-seed streaming checks; no files or unbounded retained payloads."""
    # Hand-calculated semantic fixture: the two 13-cell rods have the same
    # period but capacities four and three. The 16-cell rod holds only two.
    literal = ("2000000000000000", "0000000000000000",
               "2000500500002000", "2005005000002000", "2500000005000002")
    inp = np.array([list(map(int, row)) for row in literal], dtype=np.int8)
    expected = inp.copy()
    expected[2] = np.array(list(map(int, "2200200200202000")), dtype=np.int8)
    original = inp.copy()
    _require(np.array_equal(solve(inp), expected), "hand-calculated phase fixture")
    _require(np.array_equal(inp, original), "solve mutated its input")
    # Swapping only the two interior phases swaps the winning rod.
    shifted = inp.copy()
    shifted[[2, 3]] = shifted[[3, 2]]
    shifted_expected = shifted.copy()
    shifted_expected[3] = expected[2]
    _require(np.array_equal(solve(shifted), shifted_expected), "phase counterfactual")
    # Higher ranks remain the same rule beyond the sampler's usual 1..3 keys.
    for rank, chosen_capacity in ((4, 3), (5, 2)):
        inp = np.zeros((7, 15), dtype=np.int8)
        inp[0, :rank] = 2
        for row, capacity in enumerate(range(2, 7), 2):
            inp[row, [0, 2 * capacity + 1]] = 2
            inp[row, [1, 3]] = 5
        expected = inp.copy()
        expected[chosen_capacity, list(range(1, 2 * chosen_capacity, 2))] = 2
        _require(np.array_equal(solve(inp), expected), "higher rank or last-ranked selection")
    # A full-width rod, period 13, strict clipping and physical edge caps.
    inp = np.zeros((4, 30), dtype=np.int8)
    inp[0, :2] = 2
    inp[2, [0, 29]] = 2
    inp[2, [1, 14]] = 5
    inp[3, [5, 10]] = 2
    inp[3, [6, 7]] = 5
    expected = inp.copy()
    expected[2, [1, 14, 27]] = 2
    _require(np.array_equal(solve(inp), expected), "full-width boundary clipping")
    signatures, structures, shapes, backgrounds, periods = set(), set(), set(), set(), set()
    populations, ranks, family_counts, axes, capacities = set(), set(), set(), set(), set()
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                data = _analyze(pair)
                grid = pair["input"]
                shapes.add(grid.shape)
                backgrounds.add(data["background"])
                axes.add(data["transposed"])
                family_counts.add(len(data["families"]))
                ranks.update(data["keys"].values())
                populations.update(map(len, data["families"].values()))
                periods.update(r["period"] for r in data["rods"])
                capacities.update(r["capacity"] for r in data["rods"])
                # Hash normalized structure instead of keeping entire grids.
                colors = {}
                norm = bytes(colors.setdefault(int(c), len(colors)) for c in grid.flat)
                structures.add(hashlib.sha256(bytes(grid.shape) + norm).hexdigest())
        # All symmetries and an arbitrary color permutation preserve the rule.
        pair = puzzle["test"][0]
        permutation = np.array([8, 3, 0, 6, 1, 9, 4, 2, 7, 5], dtype=np.int8)
        for turn in range(4):
            for flip in (False, True):
                a = np.rot90(np.fliplr(pair["input"]) if flip else pair["input"], turn)
                b = np.rot90(np.fliplr(pair["output"]) if flip else pair["output"], turn)
                _require(np.array_equal(solve(permutation[a]), permutation[b]), "symmetry/color equivariance failed")
    _require(len(signatures) == 42 and len(structures) >= 180, "insufficient structural diversity")
    _require(len(shapes) >= 40 and backgrounds == set(range(10)) and len(axes) == 2, "geometry/palette diversity")
    _require({2, 3, 4, 5} <= populations and {1, 2, 3} <= ranks and {1, 2, 3} <= family_counts, "count/rank diversity")
    _require({1, 2, 3, 4, 5, 6} <= periods and len(capacities) >= 10, "cadence/capacity diversity")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(812 + a * 10 + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "explicit count mismatch")
    _require(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)), "nondeterministic")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    pair = bad["test"][0]
    row, col = np.argwhere(pair["input"] != pair["output"])[0]
    pair["output"][row, col] = pair["input"][row, col]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "backgrounds": sorted(backgrounds), "periods": sorted(periods),
            "capacities": sorted(capacities), "populations": sorted(populations),
            "ranks": sorted(ranks), "family_counts": sorted(family_counts)}
