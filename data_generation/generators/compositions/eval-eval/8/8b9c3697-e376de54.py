"""Self-calibrating comb: engagement selects the exemplar that sets docking tips.

Standard library + NumPy only. No files, examples, seeds, or generation metadata
are consulted by solve(). All generated grids are at most 30 by 30.
"""

import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    cells.add(point)
                    stack.append(point)
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return int(min(rows)), int(max(rows)) + 1, int(min(cols)), int(max(cols)) + 1


def _rectangular(cells):
    r0, r1, c0, c1 = _box(cells)
    return len(cells) == (r1 - r0) * (c1 - c0)


def _comb_frame(mask):
    """Find the unique full base and the positive, separated rectangular teeth."""
    options = []
    for turns in range(4):
        m = np.rot90(mask, turns)
        cells = set(map(tuple, np.argwhere(m)))
        r0, r1, c0, c1 = _box(cells)
        if not all(m[r0, c] for c in range(c0, c1)):
            continue
        depths = [int(m[r0 + 1:r1, c].sum()) for c in range(c0, c1)]
        expected = {(r0, c) for c in range(c0, c1)}
        expected.update((r0 + k, c0 + q)
                        for q, d in enumerate(depths) for k in range(1, d + 1))
        if cells != expected:
            continue
        runs, q = [], 0
        while q < len(depths):
            if depths[q] == 0:
                q += 1
                continue
            start = q
            while q < len(depths) and depths[q] > 0:
                q += 1
            if len(set(depths[start:q])) != 1:
                break
            runs.append({"c0": c0 + start, "c1": c0 + q, "depth": depths[start]})
        else:
            if (5 <= len(runs) <= 9 and runs[0]["c0"] == c0
                    and runs[-1]["c1"] == c1):
                options.append({"turns": turns, "base": r0, "left": c0,
                                "right": c1, "runs": runs})
    if len(options) != 1:
        raise ValueError("expected one comb with 3..7 interior pins and two guards")
    return options[0]


def _analyze(input_grid):
    original = _grid(input_grid)
    values, counts = np.unique(original, return_counts=True)
    if 0 in values or counts.max() * 2 <= original.size:
        raise ValueError("input needs a strict majority background and reserves black")
    background = int(values[np.argmax(counts)])
    objects, structures = [], []
    for color in values:
        if int(color) == background:
            continue
        for cells in _components(original == color):
            obj = {"color": int(color), "cells": cells}
            (objects if _rectangular(cells) else structures).append(obj)
    if len(structures) != 1:
        raise ValueError("one connected comb must be the only nonrectangular object")
    structure = structures[0]
    a = _comb_frame(original == structure["color"])
    a.update({"background": background, "structure": structure["color"],
              "original": original, "grid": np.rot90(original, a["turns"]).copy()})
    g, base = a["grid"], a["base"]
    pins = a["runs"][1:-1]
    tokens = []
    for obj in objects:
        mask = np.zeros(original.shape, dtype=bool)
        for cell in obj["cells"]:
            mask[cell] = True
        cells = set(map(tuple, np.argwhere(np.rot90(mask, a["turns"]))))
        r0, r1, c0, c1 = _box(cells)
        if obj["color"] == structure["color"]:
            raise ValueError("payload and comb colors must differ")
        tokens.append({"color": obj["color"], "r0": r0, "r1": r1,
                       "c0": c0, "c1": c1})
    if not tokens:
        raise ValueError("missing tokens")
    maximum = max(run["depth"] for run in a["runs"])
    # All loose tokens start beyond all teeth, with at least one blank row.
    if any(t["r0"] < base + maximum + 2 for t in tokens):
        raise ValueError("tokens must be separated from every tooth tip")
    matches = {}
    for i, pin in enumerate(pins):
        candidates = [j for j, t in enumerate(tokens)
                      if (t["c0"], t["c1"]) == (pin["c0"], pin["c1"])]
        if len(candidates) > 1:
            raise ValueError("more than one width-matching token in a lane")
        if candidates:
            j = candidates[0]
            t = tokens[j]
            corridor = g[base + pin["depth"] + 1:t["r0"], pin["c0"]:pin["c1"]]
            if np.any(corridor != background):
                raise ValueError("blocked matching route")
            matches[i] = j
    active = sorted(matches)
    if len(active) not in (3, 5, 7):
        raise ValueError("there must be 3, 5 or 7 engaged pins")
    a.update({"pins": pins, "tokens": tokens, "matches": matches, "active": active,
              "reference_index": active[len(active) // 2],
              "reference": pins[active[len(active) // 2]]["depth"]})
    return a


def _render(a, targets=None, dock=True):
    """Render from recovered input geometry; optional targets are ablation aids."""
    g, base, background = a["grid"].copy(), a["base"], a["background"]
    for t in a["tokens"]:
        g[t["r0"]:t["r1"], t["c0"]:t["c1"]] = background
    for i, j in a["matches"].items():
        pin, token = a["pins"][i], a["tokens"][j]
        target = a["reference"] if targets is None else targets[i]
        q = slice(pin["c0"], pin["c1"])
        g[base + 1:base + pin["depth"] + 1, q] = background
        g[base + 1:base + target + 1, q] = a["structure"]
        if dock:
            stop = base + target + 1
            g[stop:token["r1"], q] = 0
            g[stop:stop + token["r1"] - token["r0"], q] = token["color"]
        else:
            g[token["r0"]:token["r1"], q] = token["color"]
    return np.rot90(g, -a["turns"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Calibrate only engaged pins to their ordinal middle, then dock tokens."""
    return _render(_analyze(input_grid))


def _sample_cross_section(rng, n, witness):
    if witness:
        # Equal spacing makes the *engaged* subset's geometric midpoint be the
        # next member, not its ordinal middle. Width two admits a false token.
        return [int(rng.integers(1, 3)), *([2] * n), int(rng.integers(1, 3))], [1] * (n + 1)
    widths = [1] * (n + 2)
    gaps = [1] * (n + 1)
    budget = int(rng.integers(0, min(11, 29 - sum(widths) - sum(gaps)) + 1))
    for _ in range(budget):
        eligible = [(0, i) for i, x in enumerate(widths) if x < (2 if i in (0, n + 1) else 3)]
        eligible += [(1, i) for i, x in enumerate(gaps) if x < 3]
        which, i = eligible[int(rng.integers(len(eligible)))]
        (widths if which == 0 else gaps)[i] += 1
    return widths, gaps


def _make_pair(rng, split, index, first_reference, shift, base_turn):
    witness = split == "train" and index == 0
    if witness:
        n, active = 7, [0, 1, 2, 3, 6]
        if rng.integers(2):
            active = sorted(n - 1 - i for i in active)
    elif split == "test" and shift == "more_engaged" and index == 0:
        n, active = 7, list(range(7))
    else:
        n = int(rng.integers(3, 8))
        allowed = [x for x in (3, 5, 7) if x <= n and (split != "train" or x <= 5)]
        count = 3 if split == "train" and index == 1 else int(rng.choice(allowed))
        active = sorted(int(x) for x in rng.choice(n, size=count, replace=False))
    widths, gaps = _sample_cross_section(rng, n, witness)
    starts, q = [], 0
    for j, width in enumerate(widths):
        starts.append(q)
        q += width + (gaps[j] if j < len(gaps) else 0)
    width = q
    choices = [x for x in range(2, 6 if split == "train" else 8)
               if not (split == "train" and index == 1 and x == first_reference)]
    reference = int(rng.choice(choices))
    middle = active[len(active) // 2]
    depths = [int(rng.integers(1, reference + 4)) for _ in range(n)]
    depths[middle] = reference
    others = [i for i in active if i != middle]
    rng.shuffle(others)
    mode = "mixed" if witness or (split == "train" and index == 1) else str(rng.choice(("mixed", "extend", "trim", "one")))
    if witness:
        for i in others:
            depths[i] = int(rng.integers(1, reference))
        depths[others[0]] = int(rng.integers(reference + 1, reference + 4))
    elif mode == "one":
        for i in others:
            depths[i] = reference
        depths[others[0]] = 1 if rng.integers(2) else reference + int(rng.integers(1, 4))
    else:
        for i in others:
            if rng.random() < .20:
                depths[i] = reference
            elif mode == "extend" or (mode == "mixed" and rng.integers(2)):
                depths[i] = int(rng.integers(1, reference))
            else:
                depths[i] = int(rng.integers(reference + 1, reference + 4))
        if mode == "mixed":
            depths[others[0]], depths[others[1]] = 1, reference + int(rng.integers(1, 4))
        elif all(depths[i] == reference for i in others):
            depths[others[0]] = 1 if mode == "extend" else reference + 1
    guard_depths = [int(rng.integers(1, max(depths) + 2)) for _ in range(2)]
    all_depths = [guard_depths[0], *depths, guard_depths[1]]
    maximum = max(all_depths)
    # Token descriptors contain only direct construction geometry, not a latent
    # solution convention. Output is drawn separately from this construction.
    tokens = []
    for i in active:
        travel_gap = int(rng.integers(6, 10)) if split == "test" and shift == "longer_travel" else int(rng.integers(1, 5))
        start = maximum + 2 + travel_gap
        length = int(rng.integers(1, 4 if split == "test" else 3))
        tokens.append({"pin": i, "start": start, "length": length,
                       "q": starts[i + 1], "width": widths[i + 1], "valid": True})
    if witness:
        # Show rigid rectangles of different lengths and different travel spans,
        # so aligning the cargo's *outer* faces cannot fit the demonstration.
        tokens[0]["length"], tokens[1]["length"] = 1, 2
        tokens[0]["start"], tokens[1]["start"] = maximum + 3, maximum + 6
    inactive = [i for i in range(n) if i not in active]
    rng.shuffle(inactive)
    for i in inactive[:1 if witness else int(rng.integers(0, min(2, len(inactive)) + 1))]:
        pw, tq = widths[i + 1], starts[i + 1]
        if pw > 1:
            tw = pw - 1
        elif gaps[i + 1] >= 2:
            tw = 2
        else:
            continue
        tokens.append({"pin": i, "start": maximum + 3 + int(rng.integers(1, 5)),
                       "length": int(rng.integers(1, 3)), "q": tq, "width": tw, "valid": False})
    if witness:
        # The other empty lane also gets a correct-width but displaced token.
        # It is shifted toward the neighboring inactive lane, away from routes.
        # Its separate launch row prevents contact with the narrow distractor.
        narrow_pin = tokens[-1]["pin"]
        other = next(i for i in inactive if i != narrow_pin)
        offset = 1 if other < narrow_pin else -1
        tokens.append({"pin": other, "start": maximum + 10, "length": 1,
                       "q": starts[other + 1] + offset, "width": widths[other + 1], "valid": False})
    colors = [int(x) for x in rng.permutation(np.arange(1, 10))]
    background, structural = colors[:2]
    minimum_palette = 2 if split == "train" and index == 1 else 1
    palette_size = 1 if witness else int(rng.integers(minimum_palette, min(7, len(tokens)) + 1))
    palette = colors[2:2 + palette_size]
    for ti, token in enumerate(tokens):
        token["color"] = palette[ti % palette_size] if ti < palette_size else int(rng.choice(palette))
    used_h = max(maximum + 1, max(t["start"] + t["length"] for t in tokens))
    foreground = width + sum(pw * depth for pw, depth in zip(widths, all_depths))
    foreground += sum(t["width"] * t["length"] for t in tokens)
    w = int(rng.integers(width, min(30, width + 5) + 1))
    minimum_h = max(used_h, (2 * foreground) // w + 1)
    assert minimum_h <= 30
    h = int(rng.integers(minimum_h, min(30, max(minimum_h, used_h + 6)) + 1))
    top, left = int(rng.integers(h - used_h + 1)), int(rng.integers(w - width + 1))
    inp = np.full((h, w), background, dtype=np.int8)
    out = inp.copy()
    for grid in (inp, out):
        grid[top, left:left + width] = structural
    for j, (start, pw, depth) in enumerate(zip(starts, widths, all_depths)):
        target = reference if j - 1 in active else depth
        inp[top + 1:top + depth + 1, left + start:left + start + pw] = structural
        out[top + 1:top + target + 1, left + start:left + start + pw] = structural
    for token in tokens:
        r, c = top + token["start"], left + token["q"]
        length, tw = token["length"], token["width"]
        inp[r:r + length, c:c + tw] = token["color"]
        if token["valid"]:
            stop = top + reference + 1
            out[stop:r + length, c:c + tw] = 0
            out[stop:stop + length, c:c + tw] = token["color"]
    if shift == "new_axis":
        turns = (base_turn + (2 * (index % 2) if split == "train" else 1 + 2 * (index % 2))) % 4
    elif split == "train" and index < 2:
        turns = (base_turn + index) % 4
    else:
        turns = int(rng.integers(4))
    if rng.integers(2):
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": np.rot90(inp, turns).copy(), "output": np.rot90(out, turns).copy()}, reference


def _count(value, lower, upper):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and lower <= int(value) <= upper)


def _key(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _count(num_train, 2, 10) or not _count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    shift = str(rng.choice(("more_engaged", "longer_travel", "new_axis")))
    base_turn = int(rng.integers(4))
    puzzle, seen, first_reference = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(int(count)):
            for _ in range(32):
                pair, reference = _make_pair(rng, split, index, first_reference, shift, base_turn)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_reference = reference
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _selection_features(a):
    active = a["active"]
    lengths = [a["pins"][i]["depth"] for i in active]
    centers = [a["pins"][i]["c0"] + a["pins"][i]["c1"] - 1 for i in active]
    distances = [abs(2 * p - centers[0] - centers[-1]) for p in centers]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    geometric = (len(nearest) == 1 and lengths[nearest[0]] != a["reference"])
    return {"nonmedian": sorted(lengths)[len(lengths) // 2] != a["reference"],
            "geometric": geometric,
            "membership": (len(a["pins"]) % 2 == 1 and
                           a["pins"][len(a["pins"]) // 2]["depth"] != a["reference"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert _count(len(puzzle["train"]), 2, 10) and _count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            analyses[split].append(a)
            assert np.array_equal(_render(a), out), "incorrect calibrated docking"
            assert not np.array_equal(inp, out) and np.any(out == 0)
            canonical = np.rot90(out, a["turns"])
            b = a["base"]
            assert np.array_equal(canonical[b], a["grid"][b]), "base changed"
            inactive = [a["runs"][0], a["runs"][-1]]
            inactive += [p for i, p in enumerate(a["pins"]) if i not in a["matches"]]
            for pin in inactive:
                assert np.all(canonical[b + 1:b + pin["depth"] + 1, pin["c0"]:pin["c1"]] == a["structure"])
            for i, j in a["matches"].items():
                pin, t = a["pins"][i], a["tokens"][j]
                q = slice(pin["c0"], pin["c1"])
                stop, length = b + a["reference"] + 1, t["r1"] - t["r0"]
                assert np.all(canonical[b + 1:stop, q] == a["structure"])
                assert np.all(canonical[stop:stop + length, q] == t["color"])
                assert np.all(canonical[stop + length:t["r1"], q] == 0)
                assert (stop - 1) == b + a["reference"]
            for j, t in enumerate(a["tokens"]):
                if j not in a["matches"].values():
                    assert np.all(canonical[t["r0"]:t["r1"], t["c0"]:t["c1"]] == a["background"])
    train = analyses["train"]
    assert len({a["reference"] for a in train}) >= 2
    assert {3, 5} <= {len(a["active"]) for a in train}
    assert any(all(_selection_features(a).values()) for a in train)
    assert any(len(a["tokens"]) > len(a["active"]) for a in train)
    witness = train[0]
    matched = [witness["tokens"][j] for j in witness["matches"].values()]
    assert {1, 2} <= {t["r1"] - t["r0"] for t in matched}
    assert len({t["r0"] for t in matched}) >= 2
    false_tokens = [t for j, t in enumerate(witness["tokens"]) if j not in witness["matches"].values()]
    assert any(t["c1"] - t["c0"] == 1 for t in false_tokens)
    assert any(t["c1"] - t["c0"] == 2 for t in false_tokens)
    assert any(any(a["pins"][i]["depth"] < a["reference"] for i in a["active"]) for a in train)
    assert any(any(a["pins"][i]["depth"] > a["reference"] for i in a["active"]) for a in train)


def self_test():
    """42 puzzles, 210 pairs, bounded aggregates; no payloads or traces saved."""
    shapes, layouts, palettes, references, counts, widths, gaps, directions = (set() for _ in range(8))
    hashes, shifts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        train_a, test_a = [], []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                a = _analyze(inp)
                (train_a if split == "train" else test_a).append(a)
                shapes.add(inp.shape)
                references.add(a["reference"])
                counts.add((len(a["pins"]), len(a["active"])))
                widths.update(p["c1"] - p["c0"] for p in a["pins"])
                gaps.update(y["c0"] - x["c1"] for x, y in zip(a["runs"], a["runs"][1:]))
                directions.add(a["turns"])
                palettes.add((a["background"], a["structure"], tuple(sorted(t["color"] for t in a["tokens"]))))
                layout = (len(a["pins"]), tuple(p["depth"] for p in a["pins"]),
                          tuple(p["c0"] - a["left"] for p in a["pins"]), tuple(a["active"]),
                          tuple((t["r0"] - a["base"], t["r1"] - t["r0"]) for t in a["tokens"]))
                layouts.add(layout)
                digest.update(bytes(inp.shape))
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                if index == 0:
                    # Literal black is fixed; every other color is freely relabeled.
                    rng = np.random.default_rng(seed + 1000)
                    perm = np.array([0, *map(int, rng.permutation(np.arange(1, 10)))], dtype=np.int8)
                    assert np.array_equal(solve(perm[inp]), perm[out])
                    for turns in range(4):
                        for mirror in (False, True):
                            x, y = np.rot90(inp, turns), np.rot90(out, turns)
                            if mirror:
                                x, y = np.fliplr(x), np.fliplr(y)
                            assert np.array_equal(solve(x), y)
        hashes.add(digest.hexdigest())
        a = train_a[0]
        # Omit B: dock at each pin's original, unequal stopping surface.
        local = {i: a["pins"][i]["depth"] for i in a["active"]}
        assert not np.array_equal(_render(a, targets=local), puzzle["train"][0]["output"])
        # Omit A's eligibility filter: take the middle of all interior pins.
        all_middle = a["pins"][len(a["pins"]) // 2]["depth"]
        assert not np.array_equal(_render(a, targets={i: all_middle for i in a["active"]}), puzzle["train"][0]["output"])
        assert not np.array_equal(_render(a, dock=False), puzzle["train"][0]["output"])
        # An input-only intervention on the exemplar moves the common target.
        changed = a["grid"].copy()
        pin = a["pins"][a["reference_index"]]
        changed[a["base"] + a["reference"] + 1, pin["c0"]:pin["c1"]] = a["structure"]
        changed = np.rot90(changed, -a["turns"]).copy()
        changed_a = _analyze(changed)
        assert changed_a["reference"] == a["reference"] + 1
        assert not np.array_equal(solve(changed), puzzle["train"][0]["output"])
        if max(len(x["active"]) for x in test_a) == 7:
            shifts.add("more_engaged")
        if {x["turns"] % 2 for x in train_a}.isdisjoint({x["turns"] % 2 for x in test_a}):
            shifts.add("new_axis")
        if any(min(x["tokens"][j]["r0"] - x["base"] - max(p["depth"] for p in x["runs"])
                   for j in x["matches"].values()) >= 8 for x in test_a):
            shifts.add("longer_travel")
    assert len(hashes) == 42 and len(layouts) >= 180 and len(shapes) >= 50
    assert widths == {1, 2, 3} and gaps == {1, 2, 3} and directions == {0, 1, 2, 3}
    assert references == set(range(2, 8)) and {a for _, a in counts} == {3, 5, 7}
    assert len(palettes) >= 150 and shifts == {"more_engaged", "longer_travel", "new_axis"}
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(789, 2, 1), generate_puzzle(789, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in p for x, y in zip(p[split], q[split]) for k in x)
    a = _analyze(p["test"][0]["input"])
    broken = np.rot90(p["test"][0]["output"], a["turns"]).copy()
    pin = a["pins"][a["reference_index"]]
    broken[a["base"] + a["reference"] + 1, pin["c0"]] = a["background"]
    p["test"][0]["output"] = np.rot90(broken, -a["turns"]).copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted docked token accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "shapes": len(shapes), "structural_layouts": len(layouts),
            "palettes": len(palettes), "references": sorted(references),
            "pin_active_counts": sorted(counts), "widths": sorted(widths),
            "gaps": sorted(gaps), "directions": sorted(directions), "shifts": sorted(shifts),
            "legal_count_combinations": 30, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
