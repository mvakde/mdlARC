"""Threaded loom: periodic stencil recovery and alternation along its holes.

Only NumPy and the standard library are used. The generator renders from its
sampled stencil; solve reconstructs that stencil from the input alone.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _primitive(tile):
    """Circular primitive period, allowing individual rows shorter periods."""
    p = tile.shape[1]
    return all(not np.array_equal(tile, tile[:, np.arange(p) % q])
               for q in range(1, p) if p % q == 0)


def _decode(input_grid):
    """Read roles, erase the thread launchers mentally, and infer the stencil."""
    a = _grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    _require(h >= 4 and np.all(a[0] == bg) and np.all(a[-1] == bg),
             "blank top and bottom margins required")
    counts = Counter(int(x) for x in a.flat if int(x) != bg)
    _require(len(counts) >= 2, "stencil and at least one launcher required")
    scaffold = max(counts, key=counts.get)
    _require(counts[scaffold] > sum(n for c, n in counts.items() if c != scaffold),
             "stencil must outnumber all launcher cells together")
    observed = a == scaffold
    visible = int(np.flatnonzero(np.any(observed, axis=0))[-1]) + 1
    candidates = [p for p in range(1, visible // 2 + 1)
                  if np.array_equal(observed[:, p:visible], observed[:, :visible - p])]
    _require(candidates, "at least two complete stencil periods required")
    period = candidates[0]
    mask = observed[:, np.arange(w) % period]
    seeds = [(int(r), int(c), int(a[r, c]))
             for r, c in np.argwhere((a != bg) & (a != scaffold))]
    _require(len({r for r, _, _ in seeds}) == len(seeds), "one launcher per row")
    active_rows = tuple(int(r) for r in np.flatnonzero(np.any(mask, axis=1)))
    _require(all(r in active_rows and not mask[r, c] for r, c, _ in seeds),
             "launchers must occupy stencil holes in active rows")
    return {"array": a, "bg": bg, "scaffold": scaffold, "period": period,
            "visible": visible, "mask": mask, "observed": observed,
            "seeds": seeds, "rows": active_rows}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the periodic mask, then count holes rather than physical cells."""
    d = _decode(input_grid)
    out = np.full(d["array"].shape, d["bg"], dtype=np.int8)
    out[d["mask"]] = d["scaffold"]
    for row, start, color in d["seeds"]:
        slots = np.flatnonzero(~d["mask"][row])
        slots = slots[slots >= start]
        out[row, slots[::2]] = color
        out[row, slots[1::2]] = d["scaffold"]
    return out


def _render(mask, bg, scaffold, seeds, *, alternate=True, physical=False, reset=None):
    """Scalar renderer also used for explicit counterfactual rules in validation."""
    out = np.full(mask.shape, bg, dtype=np.int8)
    out[mask] = scaffold
    for row, start, color in seeds:
        rank = 0
        last_tile = start // reset if reset else 0
        for col in range(start, mask.shape[1]):
            if reset and col // reset != last_tile:
                rank = 0
                last_tile = col // reset
            if mask[row, col]:
                continue
            phase = (col - start) % 2 if physical else rank % 2
            out[row, col] = color if not alternate or phase == 0 else scaffold
            rank += 1
    return out


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape must be preserved")
    d = _decode(inp)
    p, v, mask = d["period"], d["visible"], d["mask"]
    _require(3 <= p <= 8 and 2 * p <= v < inp.shape[1] - 2, "stencil recurrence/suffix")
    _require(3 <= len(d["rows"]) <= 10, "active row count")
    _require(np.all(~d["observed"][:, v:]), "stencil suffix must be erased")
    _require(np.all(d["observed"][:, :v] == mask[:, :v]), "prefix exactness")
    _require(_primitive(mask[:, :p]), "primitive stencil")
    holes = [int(np.count_nonzero(~mask[r, :p])) for r in d["rows"]]
    _require(all(1 <= n < p for n in holes), "every active row has holes and solids")
    _require(any(n % 2 for n in holes) and any(n % 2 == 0 for n in holes),
             "both odd and even hole counts required")
    _require(len(d["seeds"]) >= 1, "no launchers")
    expected = _render(mask, d["bg"], d["scaffold"], d["seeds"])
    _require(np.array_equal(expected, out), "wrong slot-ranked thread or stencil")
    _require(not np.array_equal(inp, out), "nonidentity pair")
    _require(np.all(out[inp != d["bg"]] == inp[inp != d["bg"]]), "input ink changed")
    _require(set(np.unique(out)) == set(np.unique(inp)), "palette changed")
    no_a = _render(mask, d["bg"], d["scaffold"], d["seeds"], alternate=False)
    no_b = _render(d["observed"], d["bg"], d["scaffold"], d["seeds"])
    physical = _render(mask, d["bg"], d["scaffold"], d["seeds"], physical=True)
    reset = _render(mask, d["bg"], d["scaffold"], d["seeds"], reset=p)
    # The B ablation must change an actual hole's thread phase, not merely
    # omit solid cells. This is the important interaction check.
    future_holes = ~mask & (np.arange(inp.shape[1])[None, :] >= v)
    b_phase_error = bool(np.any((no_b != out) & future_holes))
    _require(not np.array_equal(no_a, out), "alternation is dispensable")
    _require(b_phase_error, "period recovery does not affect thread phase")
    _require(not np.array_equal(physical, out), "physical-distance shortcut survives")
    _require(not np.array_equal(reset, out), "reset-at-tile-boundary shortcut survives")
    row_periods = [next(q for q in range(1, p + 1) if p % q == 0
                        and np.array_equal(mask[r, :p], mask[r, np.arange(p) % q]))
                   for r in d["rows"]]
    uneven = False
    for r in d["rows"]:
        positions = np.flatnonzero(~mask[r, :p])
        gaps = np.diff(np.r_[positions, positions[0] + p])
        uneven |= len(set(int(x) for x in gaps)) > 1
    return {"period": p, "visible": v, "shape": inp.shape, "bg": d["bg"],
            "scaffold": d["scaffold"], "rows": len(d["rows"]),
            "seeds": len(d["seeds"]), "hole_counts": tuple(holes),
            "cutoff": "complete" if v % p == 0 else "partial",
            "blank_phase": bool(np.any(~np.any(mask[:, :p], axis=0))),
            "joint_period": all(q < p for q in row_periods),
            "uneven_spacing": uneven,
            "unseeded_rows": len(d["rows"]) > len(d["seeds"]),
            "left_edge_seed": any(c == 0 for _, c, _ in d["seeds"]),
            "late_seed": any(c >= v for _, c, _ in d["seeds"]),
            "repeated_color": len({c for _, _, c in d["seeds"]}) < len(d["seeds"]),
            "seed_colors": tuple(c for _, _, c in d["seeds"]),
            "layout": (tuple(d["rows"]), tuple(tuple(bool(x) for x in row)
                                              for row in mask[list(d["rows"]), :p]),
                       tuple((r, c) for r, c, _ in d["seeds"]))}


def _sample_tile(rng, rows, period):
    """Runs, staggered slits, irregular perforations, and interlocking rhythms."""
    mode = int(rng.integers(0, 4))
    if period == 6 and mode == 3:
        # No individual row has period six: two and three jointly define it.
        tile = np.zeros((rows, period), dtype=bool)
        for r in range(rows):
            q = 2 if r == 0 else 3 if r == 1 else int(rng.choice([2, 3]))
            tile[r] = np.arange(period) % q == int(rng.integers(q))
        return tile
    shared_gap = int(rng.integers(period)) if rng.random() < .24 else None
    tile = np.ones((rows, period), dtype=bool)
    base = int(rng.integers(period))
    for r in range(rows):
        options = list(range(1, period))
        if r < 2:
            options = [n for n in options if n % 2 == (1 if r == 0 else 0)]
        n_holes = int(rng.choice(options))
        if mode in (0, 1):
            shift = (base + (r if mode == 0 else abs(2 * r - rows + 1))) % period
            slots = [(shift + j) % period for j in range(n_holes)]
        else:
            slots = [int(x) for x in rng.choice(period, n_holes, replace=False)]
        if shared_gap is not None and shared_gap not in slots:
            slots[0] = shared_gap
        tile[r, slots] = False
    return tile


def _sample_pair(rng, period, scaffold, palette, *, split, cutoff=None,
                 force_late=False, color_offset=0, introductory=False):
    for _ in range(400):
        n_rows = int(rng.integers(3, 8 if split == "train" else 11))
        local_tile = _sample_tile(rng, n_rows, period)
        # Preserve coherent neighbors, but do not tie the odd/even witnesses
        # or early/late launchers to particular vertical positions.
        local_tile = np.roll(local_tile, int(rng.integers(n_rows)), axis=0)
        if rng.random() < .5:
            local_tile = local_tile[::-1].copy()
        if not _primitive(local_tile):
            continue
        # More than three stencil cycles give an early odd-hole launcher
        # room to expose the phase flip, including for the largest period.
        width = int(rng.integers(max(13, 3 * period + 2), 31))
        height = int(rng.integers(n_rows + 2, min(30, 2 * n_rows + 8) + 1))
        if rng.random() < .5:
            top = int(rng.integers(1, height - n_rows))
            row_ids = list(range(top, top + n_rows))
        else:
            row_ids = sorted(int(x) for x in rng.choice(np.arange(1, height - 1), n_rows, replace=False))
        tile = np.zeros((height, period), dtype=bool)
        tile[row_ids] = local_tile
        mask = tile[:, np.arange(width) % period]
        cutoffs = [v for v in range(2 * period, min(3 * period + 1, width - 3) + 1)
                   if np.any(mask[:, v - 1])
                   and (cutoff is None or (v % period == 0) == (cutoff == "complete"))]
        if not cutoffs:
            continue
        visible = int(rng.choice(cutoffs))
        available_bg = [c for c in range(10) if c != scaffold and c not in palette]
        bg = int(rng.choice(available_bg))
        minimum_seeds = 2 if introductory or force_late or period == 3 else 1
        count = int(rng.integers(minimum_seeds, min(n_rows, 4) + 1 if introductory else n_rows + 1))
        hole_counts = np.count_nonzero(~local_tile, axis=1)
        early_index = int(rng.choice(np.flatnonzero(hole_counts % 2 == 1)))
        late_index = int(rng.choice(np.flatnonzero(hole_counts % 2 == 0)))
        selected = [early_index] + ([late_index] if count >= 2 else [])
        if count > 2:
            selected += [int(x) for x in rng.choice([r for r in range(n_rows) if r not in selected],
                                                   count - 2, replace=False)]
        seeds = []
        for i, index in enumerate(selected):
            row = row_ids[index]
            slots = np.flatnonzero(~mask[row])
            if i == 0:
                choices = slots[slots < period]
            elif force_late and i == 1:
                # At least two holes at/after this launcher, so it is informative.
                choices = slots[(slots >= visible) & (slots < slots[-1])]
            else:
                choices = slots[:-1]
            if not len(choices):
                break
            start = int(rng.choice(choices))
            color = int(palette[(color_offset + i) % len(palette)]) if introductory else int(rng.choice(palette))
            seeds.append((row, start, color))
        if len(seeds) != count:
            continue
        inp = np.full((height, width), bg, dtype=np.int8)
        inp[:, :visible][mask[:, :visible]] = scaffold
        for row, col, color in seeds:
            inp[row, col] = color
        out = _render(mask, bg, scaffold, seeds)
        pair = {"input": inp, "output": out}
        try:
            info = _inspect_pair(pair)
            _require(info["period"] == period, "sampled and visible periods disagree")
        except AssertionError:
            continue
        return pair
    raise RuntimeError("unable to sample an informative threaded stencil")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    _require(len({i["scaffold"] for i in all_infos}) == 1, "shared separator/stencil color")
    _require(len({i["period"] for i in infos["train"]}) >= 2, "training period contrast")
    _require({i["cutoff"] for i in infos["train"]} == {"complete", "partial"}, "cutoff contrast")
    _require(any(i["late_seed"] for i in infos["train"]), "late launcher demonstration")
    _require(len({c for i in infos["train"] for c in i["seed_colors"]}) >= 3,
             "at least three demonstrated seed colors")
    train_periods = {i["period"] for i in infos["train"]}
    _require(all(i["period"] not in train_periods and i["late_seed"] for i in infos["test"]),
             "test must combine an unseen stencil period with a late launcher")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    scaffold = int(rng.integers(10))
    palette = tuple(int(x) for x in rng.choice([c for c in range(10) if c != scaffold],
                                              int(rng.integers(3, 7)), replace=False))
    held_out = [int(x) for x in rng.choice(np.arange(3, 9), 2, replace=False)]
    train_pool = [p for p in range(3, 9) if p not in held_out]
    first_two = [int(x) for x in rng.choice(train_pool, 2, replace=False)]
    cutoffs = ["complete", "partial"]
    rng.shuffle(cutoffs)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            period = first_two[i] if split == "train" and i < 2 else int(rng.choice(train_pool if split == "train" else held_out))
            for _ in range(30):
                pair = _sample_pair(rng, period, scaffold, palette, split=split,
                                    cutoff=cutoffs[i] if split == "train" and i < 2 else None,
                                    force_late=split == "test" or i == 1,
                                    introductory=split == "train" and i < 2,
                                    color_offset=i)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded grid-derived coverage, ablations, metamorphisms and API checks."""
    shapes, layouts, signatures = set(), set(), set()
    periods, backgrounds, scaffold_colors, seed_counts = set(), set(), set(), set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "determinism")
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _inspect_pair(pair)
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagrees")
                shapes.add(i["shape"])
                layouts.add((i["shape"], i["layout"]))
                periods.add(i["period"])
                backgrounds.add(i["bg"])
                scaffold_colors.add(i["scaffold"])
                seed_counts.add(i["seeds"])
                for name in ("blank_phase", "late_seed", "repeated_color", "joint_period",
                             "uneven_spacing", "unseeded_rows", "left_edge_seed"):
                    counts[name] += i[name]
                counts[i["cutoff"]] += 1
        # Color-role permutation must commute with solving, including moving
        # zero into a nonbackground role. Lists are accepted, too.
        pair = puzzle["test"][0]
        permutation = np.random.default_rng(seed + 10000).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]].tolist()),
                               permutation[pair["output"]]), "color equivariance")
    _require(len(signatures) == 42, "42 distinct puzzles")
    _require(len(shapes) >= 60 and len(layouts) >= 180, "structural variation")
    _require(periods == set(range(3, 9)), "period coverage")
    _require(len(backgrounds) >= 9 and len(scaffold_colors) >= 8, "palette role coverage")
    _require({1, 2, 3, 4, 5}.issubset(seed_counts), "launcher-count coverage")
    _require(all(counts[k] >= 5 for k in ("blank_phase", "late_seed", "repeated_color",
                                         "complete", "partial")), "missing structural cases")
    _require(counts["joint_period"] >= 1 and counts["uneven_spacing"] >= 50
             and counts["unseeded_rows"] >= 20 and counts["left_edge_seed"] >= 5,
             "missing interacting geometric cases")
    # Hand-computed fixture: three holes in the first thread's period means
    # its next block starts in the opposite color. The second launcher is
    # in the erased region; the third row is an unthreaded stencil.
    fixture_in = ("00000000000000", "52005000000000", "50505050030000",
                  "05550555000000", "00000000000000")
    fixture_out = ("00000000000000", "52525525525255", "50505050535553",
                   "05550555055505", "00000000000000")
    a = np.array([[int(c) for c in row] for row in fixture_in], dtype=np.int8)
    b = np.array([[int(c) for c in row] for row in fixture_out], dtype=np.int8)
    _require(np.array_equal(solve(a), b), "worked fixture")
    _inspect_pair({"input": a, "output": b})
    # A launcher on its row's final hole has no extension.
    edge = a.copy()
    edge[3, 12] = 4  # The final *hole* of this row; column 13 is solid.
    expected_edge = b.copy()
    expected_edge[3, 12] = 4
    _require(np.array_equal(solve(edge), expected_edge), "last-hole boundary")
    last = a.copy()
    last[2, 9] = 0
    last[2, 13] = 3
    expected_last = b.copy()
    expected_last[2] = [5 if c % 2 == 0 else 0 for c in range(14)]
    expected_last[2, 13] = 3
    _require(np.array_equal(solve(last), expected_last), "last-column launcher")
    # Padding is spatially irrelevant; no remembered active row indices.
    padded = np.insert(a, 2, np.zeros(a.shape[1], dtype=np.int8), axis=0)
    _require(np.array_equal(solve(padded), np.insert(b, 2, 0, axis=0)), "blank row insertion")
    # All explicit legal counts are inexpensive and part of the public API.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            task = generate_puzzle(20261016, a, b)
            _require((len(task["train"]), len(task["test"])) == (a, b), "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "layouts": len(layouts), "periods": sorted(periods),
            "background_colors": sorted(backgrounds), "stencil_colors": sorted(scaffold_colors),
            "launcher_counts": sorted(seed_counts), "cases": dict(counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
