"""Bead census bridges: geometry supplies an ordinal and selection routes emission.

Only standard-library modules and NumPy are used. solve() reads pixels alone.
The intentionally fixed bead color is 5; every other role is input-visible.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np

BEAD = 5


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors outside 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _span(a, b):
    if a[0] > b[0]:
        a, b = b, a
    _require(0 < b[0] - a[0] == abs(b[1] - a[1]), "not a diagonal pair")
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k, a[1] + slope * k) for k in range(b[0] - a[0] + 1))


def _line(p, slope, shape):
    h, w = shape
    r, c = p
    while r > 0 and 0 <= c - slope < w:
        r, c = r - 1, c - slope
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r, c = r + 1, c + slope
    return tuple(cells)


def _decode(value):
    """Recover families, bead ownership, ordinal choices, and emission masks."""
    grid = _grid(value)
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > grid.size // 2 and bg != BEAD, "background must dominate")
    caps = sorted(set(map(int, np.unique(grid))) - {bg, BEAD})
    _require(bool(caps), "no endpoint families")
    beads = {tuple(map(int, p)) for p in np.argwhere(grid == BEAD)}
    bridges, endpoints = [], set()
    families = {}
    for cap in caps:
        points = [tuple(map(int, p)) for p in np.argwhere(grid == cap)]
        degrees = Counter()
        family = []
        for i, a in enumerate(points):
            for b in points[i + 1:]:
                if 0 < abs(a[0] - b[0]) == abs(a[1] - b[1]):
                    cells = _span(a, b)
                    _require(len(cells) >= 3, "degenerate bridge")
                    family.append({"cap": cap, "cells": cells, "set": set(cells),
                                   "length": len(cells),
                                   "slope": 1 if cells[-1][1] > cells[0][1] else -1})
                    degrees[a] += 1
                    degrees[b] += 1
        _require(len(family) >= 2, "each family needs competitors")
        _require(all(degrees[p] == 1 for p in points), "ambiguous or unpaired endpoints")
        _require(len({b["length"] for b in family}) == len(family), "length ties")
        families[cap] = family
        bridges.extend(family)
        endpoints.update(points)
    for b in bridges:
        _require(not (set(b["cells"][1:-1]) & endpoints), "endpoint inside another bridge")
        b["beads"] = set()
    inactive = set()
    for p in beads:
        containing = [b for b in bridges if p in b["set"]]
        _require(len(containing) <= 1, "bead at an ambiguous bridge crossing")
        if containing:
            containing[0]["beads"].add(p)
        else:
            inactive.add(p)
    selected, ranks = [], {}
    for cap, family in families.items():
        rank = sum(len(b["beads"]) for b in family)
        _require(1 <= rank <= len(family), "bead census must name an existing rank")
        ranks[cap] = rank
        selected.append(sorted(family, key=lambda b: b["length"], reverse=True)[rank - 1])
    selected_union = set()
    for b in selected:
        _require(not selected_union & b["set"], "completed bridges must not cross")
        selected_union.update(b["set"])
    launches, beam_union = [], set()
    for b in selected:
        for p in sorted(b["beads"]):
            line = set(_line(p, -b["slope"], grid.shape))
            _require(not line & endpoints, "beam would overwrite endpoints")
            _require(line & selected_union == {p}, "beam/bridge color conflict")
            launches.append((p, -b["slope"], line))
            beam_union.update(line)
    return {"grid": grid, "bg": bg, "caps": caps, "beads": beads,
            "endpoints": endpoints, "bridges": bridges, "families": families,
            "ranks": ranks, "selected": selected, "inactive": inactive,
            "launches": launches, "beam_union": beam_union,
            "selected_union": selected_union}


def _draw(a):
    out = a["grid"].copy()
    for b in a["selected"]:
        for p in b["cells"]:
            if p not in a["beads"]:
                out[p] = b["cap"]
    for p in a["beam_union"]:
        out[p] = BEAD
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from original input geometry, with no seed or training lookup."""
    return _draw(_decode(input_grid))


def _counts(num_train, num_test):
    for name, n, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(n) <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _place(rng, shape, lengths, slope_for_selected, ranks):
    """Free endpoint placement; enforce intended pairing, not a rack template."""
    h, w = shape
    all_bridges, endpoints = [], set()
    family_endpoints = [[] for _ in lengths]
    tasks = [(f, length) for f, ls in enumerate(lengths) for length in ls]
    # Longer spans have fewer available placements, so place those first.
    tasks.sort(key=lambda t: t[1], reverse=True)
    chosen_lengths = [sorted(ls, reverse=True)[r - 1] for ls, r in zip(lengths, ranks)]
    for f, length in tasks:
        for _ in range(90):
            slope = (slope_for_selected[f] if length == chosen_lengths[f]
                     and slope_for_selected[f] is not None else int(rng.choice((-1, 1))))
            steps = length - 1
            r = int(rng.integers(h - steps))
            c = int(rng.integers(w - steps)) + (steps if slope == -1 else 0)
            cells = _span((r, c), (r + steps, c + slope * steps))
            a, b = cells[0], cells[-1]
            if any(abs(p[0] - q[0]) == abs(p[1] - q[1])
                   for p in (a, b) for q in family_endpoints[f]):
                continue
            cell_set = set(cells)
            if cell_set & endpoints or any(a in x["set"] or b in x["set"] for x in all_bridges):
                continue
            selected = length == chosen_lengths[f]
            if selected and any(x["selected"] and cell_set & x["set"] for x in all_bridges):
                continue
            all_bridges.append({"family": f, "cells": cells, "set": cell_set,
                                "length": length, "slope": slope, "selected": selected})
            endpoints.update((a, b))
            family_endpoints[f].extend((a, b))
            break
        else:
            return None
    return all_bridges, endpoints


def _construct(rng, caps, ranks, populations, emit_counts, slopes, bg, witness=None):
    """Build a scene with a prescribed visible census, then decode it afresh."""
    witness = witness or {}
    total = sum(populations)
    lower = 13 if total <= 5 else 19 if total <= 9 else 23
    for _ in range(140):
        shape = tuple(map(int, rng.integers(lower, 31, size=2)))
        maximum = min(shape)
        lengths = []
        for n, rank, emit in zip(populations, ranks, emit_counts):
            low = max(3, emit + 3)
            upper = min(maximum, int(rng.integers(max(low + n - 1, 9), maximum + 1)))
            style = int(rng.integers(3))
            if style == 0:
                start = int(rng.integers(low, upper - n + 2))
                ls = list(range(start, start + n))
            elif style == 1:
                ls = list(map(int, np.rint(np.linspace(low, upper, n))))
            else:
                ls = list(map(int, rng.choice(np.arange(low, upper + 1), n, replace=False)))
            lengths.append(ls)
        placed = _place(rng, shape, lengths, slopes, ranks)
        if placed is None:
            continue
        bridges, endpoints = placed
        selected = [next(b for b in bridges if b["family"] == f and b["selected"])
                    for f in range(len(caps))]
        if witness.get("middle"):
            b = selected[0]
            family = [x for x in bridges if x["family"] == 0]
            # Neither spatial extreme explains the longest selection.
            if any(sorted(family, key=lambda x: x["cells"][0][axis]).index(b)
                   in (0, len(family) - 1) for axis in (0, 1)):
                continue
        selected_union = set().union(*(b["set"] for b in selected))
        incidence = Counter(p for b in bridges for p in b["cells"][1:-1])
        beads, launches = set(), []
        failed = False
        for f, b in enumerate(selected):
            eligible = []
            for p in b["cells"][1:-1]:
                if incidence[p] != 1:
                    continue
                line = set(_line(p, -b["slope"], shape))
                if not line & endpoints and line & selected_union == {p}:
                    eligible.append((p, line))
            if len(eligible) < emit_counts[f]:
                failed = True
                break
            rng.shuffle(eligible)
            for p, line in eligible[:emit_counts[f]]:
                beads.add(p)
                launches.append(line)
            # The remaining census is routed through losing bridges.
            losing = [p for x in bridges if x["family"] == f and not x["selected"]
                      for p in x["cells"][1:-1] if incidence[p] == 1]
            needed = ranks[f] - emit_counts[f]
            if len(losing) < needed:
                failed = True
                break
            for i in rng.choice(len(losing), needed, replace=False):
                beads.add(losing[int(i)])
        if failed:
            continue
        latent_union = set().union(*(b["set"] for b in bridges))
        beams = set().union(*launches) if launches else set()
        inactive = set()
        if witness.get("beyond"):
            b = selected[0]
            options = []
            for end, sign in ((b["cells"][0], -1), (b["cells"][-1], 1)):
                p = (end[0] + sign, end[1] + sign * b["slope"])
                if (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
                        and p not in latent_union and p not in beams):
                    options.append(p)
            if not options:
                continue
            inactive.add(options[int(rng.integers(len(options)))])
        free = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                if (r, c) not in latent_union and (r, c) not in inactive]
        number = int(rng.integers(0, 5))
        for i in rng.choice(len(free), number, replace=False):
            inactive.add(free[int(i)])
        grid = np.full(shape, bg, dtype=np.int8)
        for b in bridges:
            grid[b["cells"][0]] = grid[b["cells"][-1]] = caps[b["family"]]
        for p in beads | inactive:
            grid[p] = BEAD
        # Translation and slope are supplied by the free endpoint placement.
        try:
            decoded = _decode(grid)
        except AssertionError:
            continue
        out = _draw(decoded)
        if sum(emit_counts) and not any(grid[p] == bg for p in decoded["beam_union"]):
            continue  # An active scene must visibly teach something beyond finite fills.
        if any(b not in decoded["selected"] and not any(
                grid[p] == bg and out[p] == bg for p in b["cells"][1:-1])
               for b in decoded["bridges"]):
            continue
        return {"input": grid, "output": out}
    return None


def _fallback(rng, caps, ranks, populations, emit_counts, slopes, bg, witness):
    """Constructive fallback in disjoint row+column bands on a 30x30 canvas.

    Distinct endpoint sums and distinct diagonal lanes guarantee unique pairs.
    The bands keep emitted diagonals away from other families. No saved inputs
    or outputs are used, and solve() does not know this placement scheme.
    """
    small = [max(r, 3 if r == 2 else 2) for r in ranks]
    if witness.get("middle"):
        small[0] = 4
    grid = np.full((30, 30), bg, dtype=np.int8)
    bridges, endpoints = [], set()
    for f, n in enumerate(small):
        lanes = set()
        for j in range(n):
            length = 5 + n - j
            start_sum = 2 + 18 * f + j
            candidates = [(r, start_sum - r) for r in range(30)
                          if 0 <= start_sum - r < 30
                          and r + length - 1 < 30 and start_sum - r + length - 1 < 30
                          and 2 * r - start_sum not in lanes]
            if witness.get("middle") and f == 0:
                top = ((1, 1), (0, 3), (3, 1), (5, 0))[j]
            else:
                top = candidates[int(rng.integers(len(candidates)))]
            lanes.add(top[0] - top[1])
            cells = _span(top, (top[0] + length - 1, top[1] + length - 1))
            bridges.append({"family": f, "cells": cells, "selected": j == ranks[f] - 1})
            endpoints.update((cells[0], cells[-1]))
            grid[cells[0]] = grid[cells[-1]] = caps[f]
    selected = [next(b for b in bridges if b["family"] == f and b["selected"])
                for f in range(len(caps))]
    endpoint_sums = {r + c for r, c in endpoints}
    active, beams = set(), set()
    for f, b in enumerate(selected):
        eligible = [p for p in b["cells"][1:-1] if sum(p) not in endpoint_sums]
        rng.shuffle(eligible)
        for p in eligible[:emit_counts[f]]:
            active.add(p)
            beams.update(_line(p, -1, grid.shape))
    # Reserve one unchanged background cell on each losing bridge before
    # assigning the remainder of each family's distributed unary instruction.
    reserved = set()
    for b in bridges:
        if not b["selected"]:
            reserved.add(next(p for p in b["cells"][1:-1] if p not in beams))
    for f, rank in enumerate(ranks):
        eligible = [p for b in bridges if b["family"] == f and not b["selected"]
                    for p in b["cells"][1:-1] if p not in reserved]
        rng.shuffle(eligible)
        active.update(eligible[:rank - emit_counts[f]])
    latent = {p for b in bridges for p in b["cells"]}
    inactive = set()
    if witness.get("beyond"):
        b = selected[0]
        for end, sign in ((b["cells"][0], -1), (b["cells"][-1], 1)):
            p = (end[0] + sign, end[1] + sign)
            if (0 <= p[0] < 30 and 0 <= p[1] < 30 and p not in latent and p not in beams):
                inactive.add(p)
                break
        _require(bool(inactive), "constructive beyond-endpoint witness")
    free = [(r, c) for r in range(30) for c in range(30) if (r, c) not in latent]
    for i in rng.choice(len(free), int(rng.integers(1, 5)), replace=False):
        inactive.add(free[int(i)])
    for p in active | inactive:
        grid[p] = BEAD
    required_slopes = {s for s in slopes if s is not None}
    _require(len(required_slopes) <= 1, "fallback expects one forced slope")
    if required_slopes == {-1}:
        grid = np.fliplr(grid).copy()
    return {"input": grid, "output": solve(grid)}


def _pair(rng, palette, spec, used):
    caps = palette[:len(spec["ranks"])]
    background_choices = [c for c in range(10) if c != BEAD and c not in caps]
    for _ in range(8):
        bg = int(rng.choice(background_choices))
        pair = _construct(rng, caps, spec["ranks"], spec["populations"],
                          spec["emit"], spec["slopes"], bg, spec.get("witness"))
        if pair is None:
            pair = _fallback(rng, caps, spec["ranks"], spec["populations"],
                             spec["emit"], spec["slopes"], bg, spec.get("witness", {}))
        if _key(pair["input"]) not in used:
            return pair
    raise AssertionError("distinct-input placement exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one deterministic task; all conventions are visible or fixed."""
    train_count, test_count = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != BEAD])))
    shift = str(rng.choice(("ordinal", "families", "recombine")))
    first_slope = int(rng.choice((-1, 1)))
    anchors = [
        {"ranks": [1], "populations": [int(rng.integers(4, 7))],
         "emit": [1], "slopes": [first_slope], "witness": {"middle": True, "beyond": True}},
        {"ranks": [2, 2], "populations": list(map(int, rng.integers(3, 6, size=2))),
         "emit": [0, 2], "slopes": [None, -first_slope]},
    ]
    specs = {"train": anchors, "test": []}
    for split, count in (("train", train_count - 2), ("test", test_count)):
        for _ in range(count):
            f = 3 if split == "test" and shift == "families" else int(rng.integers(1, 3))
            max_rank = 3 if split == "test" and shift == "ordinal" else 2
            ranks = list(map(int, rng.integers(1, max_rank + 1, size=f)))
            if split == "test" and shift == "ordinal":
                ranks[0] = 3
            populations = [int(rng.integers(max(2, r), (5 if f == 3 else 7))) for r in ranks]
            emit = [int(rng.integers(r + 1)) for r in ranks]
            if shift == "recombine":
                if split == "train":
                    emit = [int(rng.choice((0, 2))) if r == 2 else e
                            for r, e in zip(ranks, emit)]
                else:
                    # Recombine voting and emitting: one vote on the winner,
                    # one vote on a loser, after both extreme allocations.
                    ranks[0] = 2
                    emit[0] = 1
                    populations[0] = max(2, populations[0])
            specs[split].append({"ranks": ranks, "populations": populations,
                                 "emit": emit, "slopes": [None] * f})
    puzzle, used = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for spec in specs[split]:
            pair = _pair(rng, palette, spec, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _beyond(a):
    return any(abs(p[0] - b["cells"][0][0]) == abs(p[1] - b["cells"][0][1])
               and abs(p[0] - b["cells"][-1][0]) == abs(p[1] - b["cells"][-1][1])
               for p in a["inactive"] for b in a["bridges"])


def validate_generated_puzzle(puzzle) -> None:
    """Check actual returned pixels, family geometry, labels, and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _decode(inp)
            _require(np.array_equal(out, _draw(a)), "wrong bridge census output")
            _require(np.any(inp != out), "example must change")
            _require(np.array_equal(out[inp != a["bg"]], inp[inp != a["bg"]]), "original marker changed")
            allowed = a["selected_union"] | a["beam_union"]
            changed = {tuple(map(int, p)) for p in np.argwhere(inp != out)}
            _require(changed <= allowed, "change outside chosen constructions")
            # Each losing bridge has an unpainted interior witness. This defeats
            # a B-only completion of every recoverable endpoint pair.
            for b in a["bridges"]:
                if b in a["selected"]:
                    continue
                _require(any(inp[p] == a["bg"] and out[p] == a["bg"]
                             for p in b["cells"][1:-1]), "losing bridge lacks visible witness")
            if split == "train":
                training.append(a)
            else:
                testing.append(a)
    histories = {}
    for a in training:
        for cap, rank in a["ranks"].items():
            histories.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= h for h in histories.values()), "same-color ordinal contrast")
    _require(any(a["ranks"][b["cap"]] == 2 and len(a["families"][b["cap"]]) >= 3
                 for a in training for b in a["selected"]), "non-extreme selection")
    _require(any(not b["beads"] for a in training for b in a["selected"]), "dark selected bridge")
    _require(any(len(b["beads"]) >= 2 for a in training for b in a["selected"]), "multiple emission")
    _require(any(_beyond(a) for a in training), "finite-span bead distractor")
    _require({s for a in training for _, s, _ in a["launches"]} == {-1, 1}, "both emitted slopes")
    _require(any(a["grid"][p] == a["bg"] for a in training for p in a["beam_union"]),
             "training must visibly require perpendicular emission")
    _require(any(len(a["families"]) >= 2 for a in training), "per-family census evidence")
    taught_ranks = {r for a in training for r in a["ranks"].values()}
    taught_routes = {(a["ranks"][b["cap"]], len(b["beads"]))
                     for a in training for b in a["selected"]}
    largest_family_count = max(len(a["families"]) for a in training)
    for a in testing:
        _require(all(r in taught_ranks or set(range(1, r)) <= taught_ranks
                     for r in a["ranks"].values()), "unsupported ordinal extrapolation")
        _require(len(a["families"]) > largest_family_count
                 or bool(set(a["ranks"].values()) - taught_ranks)
                 or any(0 < len(b["beads"]) < a["ranks"][b["cap"]]
                        and (a["ranks"][b["cap"]], len(b["beads"])) not in taught_routes
                        for b in a["selected"]), "missing structural train/test shift")


def measure_pair(pair):
    """Compact features reconstructed from pixels for variation review."""
    a = _decode(pair["input"])
    return {"shape": list(a["grid"].shape), "background": a["bg"], "caps": a["caps"],
            "families": len(a["caps"]), "ranks": list(a["ranks"].values()),
            "populations": [len(f) for f in a["families"].values()],
            "lengths": [b["length"] for b in a["bridges"]],
            "slopes": sorted({b["slope"] for b in a["bridges"]}),
            "emitters": [len(b["beads"]) for b in a["selected"]],
            "inactive": len(a["inactive"]), "beyond": _beyond(a),
            "latent_crossing": any(x["set"] & y["set"] for i, x in enumerate(a["bridges"])
                                   for y in a["bridges"][i + 1:]),
            "beam_crossing": any(x[2] & y[2] for i, x in enumerate(a["launches"])
                                 for y in a["launches"][i + 1:]),
            "beam_through_loser": any(b not in a["selected"] and b["set"] & a["beam_union"]
                                      for b in a["bridges"]),
            "beam_through_inert": bool(a["inactive"] & a["beam_union"]),
            "boundary_endpoint": any(r in (0, a["grid"].shape[0] - 1)
                                      or c in (0, a["grid"].shape[1] - 1) for r, c in a["endpoints"])}


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s]) for s in ("train", "test"))


def _parent_ablation_differences(a):
    """Compare two explicit omissions against the answer, using input pixels.

    Without A, complete every bridge and emit from every voting bead. Without
    B's emission, grant the correct ordinal choices but draw finite spans only.
    These are counterfactual checks, not alternate modes of the family solver.
    """
    without_a = a["grid"].copy()
    without_b = a["grid"].copy()
    for b in a["bridges"]:
        for p in b["cells"][1:-1]:
            if p not in a["beads"]:
                without_a[p] = b["cap"]
    for b in a["bridges"]:
        for bead in b["beads"]:
            for p in _line(bead, -b["slope"], a["grid"].shape):
                if p not in a["endpoints"]:
                    without_a[p] = BEAD
    for b in a["selected"]:
        for p in b["cells"][1:-1]:
            if p not in a["beads"]:
                without_b[p] = b["cap"]
    answer = _draw(a)
    return bool(np.any(without_a != answer)), bool(np.any(without_b != answer))


def self_test() -> None:
    """Bounded 42-seed coverage plus metamorphic and corruption checks."""
    # Hand-specified regression: two active beads on a losing length-8 bridge
    # choose an empty length-7 bridge; another family's two beads choose and
    # emit from its length-7 bridge. The two extra beads are inert, including
    # one just beyond an endpoint and one on a beam from a different bead.
    fixture = np.zeros((30, 30), dtype=np.int8)
    for p in ((0, 2), (7, 9), (0, 3), (6, 9), (0, 4), (5, 9)):
        fixture[p] = 2
    for p in ((10, 10), (17, 17), (9, 12), (15, 18), (12, 10), (17, 15)):
        fixture[p] = 8
    for p in ((2, 4), (3, 5), (10, 13), (12, 15), (7, 10), (13, 10)):
        fixture[p] = BEAD
    expected = fixture.copy()
    for r in range(1, 6):
        expected[r, r + 3] = 2
    for r in range(10, 15):
        if fixture[r, r + 3] != BEAD:
            expected[r, r + 3] = 8
    for total in (23, 27):
        for r in range(total + 1):
            expected[r, total - r] = BEAD
    _require(np.array_equal(solve(fixture), expected), "hand-calculated census and emission")
    _require(solve(fixture)[7, 10] == BEAD and solve(fixture)[8, 9] == 0,
             "a bead beyond the endpoint cannot emit")
    _require(all(_parent_ablation_differences(_decode(fixture))),
             "manual fixture must require both parent mechanisms")
    # A bead at a crossing has two owners; an extra collinear endpoint has two
    # possible partners. Reject these inputs before any ordinal is interpreted.
    crossing = np.zeros((14, 14), dtype=np.int8)
    for p in ((0, 0), (6, 6), (0, 8), (4, 12)):
        crossing[p] = 2
    for p in ((0, 6), (6, 0), (8, 10), (10, 12)):
        crossing[p] = 8
    crossing[3, 3] = BEAD
    ambiguous = fixture.copy()
    ambiguous[4, 6] = 2
    for invalid, reason in ((crossing, "bead at an ambiguous bridge crossing"),
                            (ambiguous, "ambiguous or unpaired endpoints")):
        try:
            solve(invalid)
        except AssertionError as error:
            _require(str(error) == reason, "geometry regression failed for the wrong reason")
        else:
            raise AssertionError("ambiguous geometry was accepted")
    # Exercise the constructive emergency path separately from normal sampling.
    for index in range(42):
        rng = np.random.default_rng(8000 + index)
        f = 1 + index % 3
        ranks = [1 + (index + j) % 3 for j in range(f)]
        emit = [(index // 3 + j) % (r + 1) for j, r in enumerate(ranks)]
        p = _fallback(rng, [2, 4, 8][:f], ranks, [6] * f, emit,
                      [None] * f, 0, {})
        a = _decode(p["input"])
        _require(sorted(a["ranks"].values()) == sorted(ranks), "fallback census")
        _require(np.array_equal(solve(p["input"]), p["output"]), "fallback output")
    for slope in (-1, 1):
        p = _fallback(np.random.default_rng(88), [2], [1], [6], [1], [slope], 0,
                      {"middle": True, "beyond": True})
        _require(_beyond(_decode(p["input"])), "fallback beyond-endpoint evidence")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [measure_pair(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    for puzzle in puzzles:
        contrasts = [_parent_ablation_differences(_decode(pair["input"]))
                     for pair in puzzle["train"]]
        _require(all(a for a, _ in contrasts), "training must distinguish removing A")
        _require(any(b for _, b in contrasts), "training must distinguish removing B")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed uniqueness")
    _require(len({tuple(r["shape"]) for r in rows}) >= 50, "shape variation")
    _require({1, 2, 3} <= {r["families"] for r in rows}, "family variation")
    _require({1, 2, 3} <= {n for r in rows for n in r["ranks"]}, "ordinal variation")
    _require({2, 3, 4, 5, 6} <= {n for r in rows for n in r["populations"]}, "population variation")
    _require({0, 1, 2, 3} <= {n for r in rows for n in r["emitters"]}, "emitter variation")
    _require(len({r["background"] for r in rows}) >= 7, "background variation")
    for flag in ("latent_crossing", "beam_crossing", "beam_through_loser", "beam_through_inert", "boundary_endpoint"):
        _require(any(r[flag] for r in rows), f"missing {flag}")
    _require(max(n for r in rows for n in r["lengths"]) >= 24, "long spans")
    _require(min(n for r in rows for n in r["lengths"]) == 3, "short spans")
    for p in puzzles[:3]:
        for s in ("train", "test"):
            for pair in p[s]:
                for rotation in range(4):
                    for reflected in (False, True):
                        x, y = pair["input"], pair["output"]
                        if reflected:
                            x, y = np.fliplr(x), np.fliplr(y)
                        x, y = np.rot90(x, rotation), np.rot90(y, rotation)
                        _require(np.array_equal(solve(x), y), "dihedral invariance")
                # Any recoloring that fixes the visibly special bead color works.
                mapping = np.arange(10, dtype=np.int8)
                colors = [c for c in range(10) if c != BEAD]
                for i, c in enumerate(colors):
                    mapping[c] = colors[(i + 3) % len(colors)]
                _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]), "role recoloring")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(123456 + train * 10 + test, train, test)
            _require(len(p["train"]) == train and len(p["test"]) == test, "explicit counts")
    _require(_puzzle_key(generate_puzzle(901, 2, 1)) == _puzzle_key(generate_puzzle(901, 2, 1)), "determinism")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    pair = bad["test"][0]
    p = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][p] = pair["input"][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
