"""Repair periodic strips and equalize their number of primitive repeats.

No file access, generated-example cache, or seed is used by solve().
All campaigns run in memory; importing this module writes nothing.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _noise_limit(size):
    return max(1, min(2, size // 24))


def _recover(motif, capacity):
    """All sparse, strict-majority explanations must have one continuation.

    Enumerate even periods outside the sampled 2..5 range, so rejecting an
    ambiguous sample does not depend on knowing its construction period.
    Redundant periods are harmless; the smallest supplies the measuring unit.
    """
    cross, length = motif.shape
    colors = np.unique(motif)
    _require(len(colors) == 2, "a strip must contain two colors")
    candidates = []
    for period in range(1, length // 3 + 1):
        template = np.empty((cross, period), dtype=np.int8)
        errors, supported = 0, True
        for phase in range(period):
            samples = motif[:, phase::period]
            counts = np.sum(samples == colors[1], axis=1)
            if np.any(2 * counts == samples.shape[1]):
                supported = False
                break
            template[:, phase] = np.where(2 * counts > samples.shape[1], colors[1], colors[0])
            errors += int(np.minimum(counts, samples.shape[1] - counts).sum())
        if not supported or errors > _noise_limit(motif.size):
            continue
        continuation = template[:, np.arange(capacity) % period]
        candidates.append((period, continuation))
    _require(bool(candidates), "no supported periodic repair")
    first = candidates[0][1]
    _require(all(np.array_equal(first, c) for _, c in candidates), "ambiguous periodic repair")
    period = candidates[0][0]
    _require(2 <= period <= 5, "primitive unit must vary along the strip and have period 2..5")
    return period, first


def _parse(input_grid):
    original = _grid(input_grid)
    structure = int(original[0, 0])
    _require(all(np.all(edge == structure) for edge in
                 (original[0], original[-1], original[:, 0], original[:, -1])), "missing frame")
    full_rows = np.flatnonzero(np.all(original == structure, axis=1))
    full_cols = np.flatnonzero(np.all(original == structure, axis=0))
    horizontal = len(full_rows) in (4, 6, 8) and len(full_cols) == 2
    vertical = len(full_cols) in (4, 6, 8) and len(full_rows) == 2
    _require(horizontal != vertical, "ambiguous compartment orientation")
    transposed = bool(vertical)
    a = original.T if transposed else original
    dividers = np.flatnonzero(np.all(a == structure, axis=1)).tolist()
    _require(all(b - s >= 4 for s, b in zip(dividers, dividers[1:])), "missing lane padding")
    capacity = a.shape[1] - 4
    _require(6 <= capacity <= 26, "invalid strip capacity")
    padding = int(a[1, 1])
    _require(padding != structure, "padding and structure must differ")
    rectangles, spans, accents = [], [], []
    for top, bottom in zip(dividers, dividers[1:]):
        _require(np.all(a[top + 1, 1:-1] == padding) and
                 np.all(a[bottom - 1, 1:-1] == padding) and
                 np.all(a[top + 1:bottom, 1] == padding) and
                 np.all(a[top + 1:bottom, -2] == padding), "invalid one-cell margin")
        rectangle = (slice(top + 2, bottom - 1), slice(2, a.shape[1] - 2))
        area = a[rectangle]
        used = np.flatnonzero(np.any(area != padding, axis=0))
        _require(len(used) > 0 and np.all(np.diff(used) == 1), "strip has a gap")
        start, stop = int(used[0]), int(used[-1]) + 1
        motif = area[:, start:stop]
        _require(np.all(motif != padding), "ragged strip boundary")
        palette = [int(c) for c in np.unique(motif)]
        _require(len(palette) == 2 and structure in palette, "strip must use structure and one accent")
        rectangles.append(rectangle)
        spans.append((start, stop))
        accents.append(next(c for c in palette if c != structure))
    left = all(start == 0 for start, _ in spans)
    right = all(stop == capacity for _, stop in spans)
    _require(left != right, "strips must have exactly one shared anchor side")
    reverse = bool(right)
    if reverse:
        a = np.fliplr(a)
    lengths = [stop - start for start, stop in spans]
    periods, continuations = [], []
    for rectangle, length in zip(rectangles, lengths):
        motif = a[rectangle][:, :length]
        period, continuation = _recover(motif, capacity)
        periods.append(period)
        continuations.append(continuation)
    middle = len(rectangles) // 2
    _require(lengths[middle] % periods[middle] == 0, "reference ends inside a unit")
    repeats = lengths[middle] // periods[middle]
    _require(3 <= repeats <= 6, "reference must contain 3..6 units")
    targets = [repeats * p for p in periods]
    _require(max(targets) <= capacity, "target goes beyond compartment")
    crosses = [r[0].stop - r[0].start for r in rectangles]
    _require(all(1 <= c <= 10 for c in crosses), "invalid motif thickness")
    return {"a": a, "transposed": transposed, "reverse": reverse,
            "structure": structure, "padding": padding, "rectangles": rectangles,
            "lengths": lengths, "periods": periods, "targets": targets,
            "continuations": continuations, "repeats": repeats, "accents": accents,
            "capacity": capacity, "crosses": crosses}


def _restore(a, parsed):
    if parsed["reverse"]:
        a = np.fliplr(a)
    return (a.T if parsed["transposed"] else a).copy()


def _render(parsed, targets):
    answer = parsed["a"].copy()
    for rectangle, target, continuation in zip(parsed["rectangles"], targets, parsed["continuations"]):
        area = answer[rectangle]
        area[:] = parsed["padding"]
        area[:, :target] = continuation[:, :target]
    return _restore(answer, parsed)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer frames, anchors, primitive units and reference count from input."""
    parsed = _parse(input_grid)
    return _render(parsed, parsed["targets"])


def _geometric_member(crosses):
    centers, divider = [], 0
    for cross in crosses:
        centers.append(2 * divider + cross + 3)  # Twice the strip center.
        divider += cross + 3
    distances = [abs(2 * c - centers[0] - centers[-1]) for c in centers]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    return nearest[0] if len(nearest) == 1 else None


def _crosses(rng, count, witness):
    budget = 29 - 3 * count
    for _ in range(80):
        crosses = np.ones(count, dtype=int)
        total = int(rng.integers(count, budget + 1))
        if not witness and rng.random() < .20:
            crosses[:] = int(rng.integers(1, min(10, budget // count) + 1))
        else:
            for _ in range(total - count):
                eligible = np.flatnonzero(crosses < 10)
                crosses[int(rng.choice(eligible))] += 1
        result = [int(c) for c in crosses]
        if not witness or _geometric_member(result) not in (None, count // 2):
            return result
    # A geometric construction, not a saved puzzle; ordinal and metric centers
    # differ for all these fitting five-lane arrangements.
    x = int(rng.integers(1, 11))
    result = [1, 1, 1, x, 11 - x]
    return result if rng.random() < .5 else result[::-1]


def _primitive(bits):
    period = bits.shape[1]
    return all(not np.array_equal(bits, bits[:, np.arange(period) % q])
               for q in range(1, period) if period % q == 0)


def _sample_motif(rng, cross, length, period, target, capacity, structure, accent):
    for attempt in range(160):
        if attempt < 96:
            density = float(rng.choice((.25, .5, .75)))
            bits = (rng.random((cross, period)) < density).astype(np.int8)
            if not _primitive(bits):
                continue
        else:
            bits = np.zeros((cross, period), dtype=np.int8)
            for row in range(cross):
                bits[row, (row + attempt) % period] = 1
        template = np.where(bits, accent, structure).astype(np.int8)
        clean = template[:, np.arange(capacity) % period]
        noisy = clean[:, :length].copy()
        errors = int(rng.integers(1, _noise_limit(noisy.size) + 1)) if attempt < 96 else 1
        # Keep the first error inside the eventual surviving prefix. This is a
        # visible witness that normalization alone cannot explain the output.
        row = int(rng.integers(cross))
        col = int(rng.integers(min(length, target)))
        locations = [(row, col)]
        if errors == 2:
            groups = [(r, phase) for r in range(cross) for phase in range(period)
                      if (r, phase) != (row, col % period)]
            r, phase = groups[int(rng.integers(len(groups)))]
            positions = list(range(phase, length, period))
            locations.append((r, positions[int(rng.integers(len(positions)))]))
        for r, c in locations:
            noisy[r, c] = accent if noisy[r, c] == structure else structure
        try:
            recovered_period, recovered = _recover(noisy, capacity)
        except ValueError:
            continue
        if recovered_period == period and np.array_equal(recovered, clean):
            return noisy, clean
    raise RuntimeError("bounded motif construction exhausted")


def _make_pair(rng, count, repeats, witness, direction, roles):
    structure, padding = roles
    pmax = min(5, 25 // repeats)
    periods = [int(rng.integers(2, pmax + 1)) for _ in range(count)]
    if len(set(periods)) == 1:
        periods[int(rng.integers(count))] = next(p for p in range(2, pmax + 1) if p != periods[0])
    capacity = int(rng.integers(max(3, repeats) * max(periods) + 1, 27))
    targets = [repeats * p for p in periods]
    middle = count // 2
    others = [int(i) for i in rng.permutation(count) if i != middle]
    lengths = targets.copy()

    def shorter(i):
        return int(rng.integers(3 * periods[i], targets[i]))

    def longer(i):
        return int(rng.integers(targets[i] + 1, capacity + 1))

    if witness:
        for i in others:
            lengths[i] = shorter(i)
        lengths[others[0]] = longer(others[0])
    else:
        mode = str(rng.choice(("mixed", "extend", "trim", "one"))) if repeats > 3 else "trim"
        for i in others:
            if rng.random() < .25 or mode == "one":
                continue
            lengths[i] = shorter(i) if mode == "extend" or (mode == "mixed" and rng.random() < .5) else longer(i)
        if mode == "mixed":
            lengths[others[0]], lengths[others[1]] = shorter(others[0]), longer(others[1])
        elif all(x == y for x, y in zip(lengths, targets)):
            i = others[0]
            lengths[i] = shorter(i) if mode == "extend" or (mode == "one" and rng.random() < .5) else longer(i)
    crosses = _crosses(rng, count, witness)
    available = [c for c in range(10) if c not in roles]
    palette_size = 1 if rng.random() < .25 else int(rng.integers(2, min(count, 8) + 1))
    palette = [int(c) for c in rng.choice(available, palette_size, replace=False)]
    accents = palette + [int(rng.choice(palette)) for _ in range(count - palette_size)]
    rng.shuffle(accents)
    shape = (1 + sum(c + 3 for c in crosses), capacity + 4)
    inp = np.full(shape, padding, dtype=np.int8)
    out = inp.copy()
    inp[:, (0, -1)] = structure
    out[:, (0, -1)] = structure
    divider = 0
    for cross, period, length, target, accent in zip(crosses, periods, lengths, targets, accents):
        inp[divider] = out[divider] = structure
        rectangle = (slice(divider + 2, divider + 2 + cross), slice(2, 2 + capacity))
        noisy, clean = _sample_motif(rng, cross, length, period, target, capacity, structure, accent)
        inp[rectangle][:, :length] = noisy
        out[rectangle][:, :target] = clean[:, :target]
        divider += cross + 3
    inp[-1] = out[-1] = structure
    if direction % 2:
        inp, out = np.fliplr(inp), np.fliplr(out)
    if direction // 2:
        inp, out = inp.T, out.T
    return {"input": inp.copy(), "output": out.copy()}


def _integer(value, name, lower, upper=None):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), name + " must be an integer")
    _require(lower <= value and (upper is None or value <= upper), "illegal " + name)
    return int(value)


def _fingerprint(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _signature(puzzle):
    return tuple(_fingerprint(p[k]) for split in ("train", "test") for p in puzzle[split] for k in ("input", "output"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    witness_k = int(rng.integers(4, 7))
    other_k = int(rng.choice([k for k in range(3, 7) if k != witness_k]))
    familiar = [witness_k, other_k]
    unseen = [k for k in range(3, 7) if k not in familiar]
    first_counts = [5, 3] if rng.random() < .5 else [3, 5]
    regime = int(rng.integers(3))
    base_direction = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    first_roles = None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            count = first_counts[index] if split == "train" and index < 2 else int(rng.choice((3, 5, 7)))
            witness = split == "train" and index < 2 and count == 5
            repeats = witness_k if witness else other_k if split == "train" and index < 2 else int(rng.choice(familiar if split == "train" else unseen))
            direction = base_direction if regime == 0 else int(rng.integers(4)) if regime == 1 else (base_direction if split == "train" else base_direction ^ 2)
            for attempt in range(12):
                palette = [int(c) for c in rng.permutation(10)]
                roles = tuple(palette[:2])
                if split == "train" and index == 1:
                    structure = next(c for c in palette if c != first_roles[0])
                    padding = next(c for c in palette if c not in (first_roles[1], structure))
                    roles = (structure, padding)
                pair = _make_pair(rng, count, repeats, witness, direction, roles)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            if split == "train" and index == 0:
                first_roles = roles
            puzzle[split].append(pair)
            seen.add(key)
    return puzzle


def _pair_checks(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    d = _parse(inp)
    assert np.array_equal(out, _render(d, d["targets"])), "incorrect cycle normalization"
    assert len(set(d["periods"])) >= 2, "missing independent measuring units"
    assert any(x != y for x, y in zip(d["lengths"], d["targets"])), "missing resizing"
    normalized_out = out.T if d["transposed"] else out
    if d["reverse"]:
        normalized_out = np.fliplr(normalized_out)
    mask = np.zeros_like(d["a"], dtype=bool)
    for rectangle, length, target, period, continuation in zip(d["rectangles"], d["lengths"], d["targets"], d["periods"], d["continuations"]):
        mask[rectangle] = True
        motif = d["a"][rectangle][:, :length]
        errors = motif != continuation[:, :length]
        assert length >= 3 * period
        assert 1 <= int(errors.sum()) <= _noise_limit(motif.size)
        assert np.any(errors[:, :min(length, target)]), "no surviving periodic-repair witness"
        for r in range(motif.shape[0]):
            assert all(int(errors[r, phase::period].sum()) <= 1 for phase in range(period))
        result = normalized_out[rectangle]
        assert np.array_equal(result[:, :target], continuation[:, :target])
        assert np.all(result[:, target:] == d["padding"])
    assert np.array_equal(d["a"][~mask], normalized_out[~mask]), "changed frames or padding"
    # Ablation 1: recover periodic content but keep the original lengths.
    assert not np.array_equal(out, _render(d, d["lengths"])), "B has no effect"
    # Ablation 2: use the center's physical length for every repaired strip.
    raw_target = d["lengths"][len(d["lengths"]) // 2]
    assert not np.array_equal(out, _render(d, [raw_target] * len(d["lengths"]))), "periods do not affect measurement"
    return d


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _fingerprint(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            analyses[split].append(_pair_checks(pair))
    train = analyses["train"]
    assert {len(d["lengths"]) for d in train} >= {3, 5}
    assert len({d["repeats"] for d in train}) == 2
    assert len({d["structure"] for d in train}) >= 2 and len({d["padding"] for d in train}) >= 2
    assert all(d["repeats"] not in {t["repeats"] for t in train} for d in analyses["test"])
    # The five-strip witness defeats numerical medians and geometric centers,
    # and demonstrates both tail extension and tail removal.
    witnesses = []
    for d in train:
        n = len(d["lengths"])
        ratios = [length / period for length, period in zip(d["lengths"], d["periods"])]
        geometric = _geometric_member(d["crosses"])
        if (n == 5 and geometric not in (None, n // 2) and
                ratios[geometric] != d["repeats"] and
                sorted(ratios)[n // 2] != d["repeats"] and
                min(ratios) < d["repeats"] < max(ratios)):
            witnesses.append(d)
    assert witnesses, "missing central-reference counterexamples"


def self_test():
    """42 puzzles, bounded hashes/feature sets, metamorphic and mutation checks."""
    shapes, structures, signatures, counts, periods, crosses, capacities = set(), set(), set(), set(), set(), set(), set()
    directions, repeat_counts, role_colors, partial = set(), set(), set(), 0
    retained_errors = 0
    rng = np.random.default_rng(812738)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                d = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                counts.add(len(d["lengths"]))
                periods.update(d["periods"])
                crosses.update(d["crosses"])
                capacities.add(d["capacity"])
                directions.add((d["transposed"], d["reverse"]))
                repeat_counts.add(d["repeats"])
                role_colors.update((d["structure"], d["padding"]))
                partial += any(length % p for length, p in zip(d["lengths"], d["periods"]))
                retained_errors += int(np.count_nonzero(pair["input"] != pair["output"]))
                structural = (tuple(d["crosses"]), tuple(d["periods"]), tuple(d["lengths"]), d["repeats"], d["capacity"])
                structures.add(structural)
                if index == 0:
                    palette = rng.permutation(10).astype(np.int8)
                    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
                    for turns in range(4):
                        for mirror in (False, True):
                            inp, out = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                            if mirror:
                                inp, out = np.fliplr(inp), np.fliplr(out)
                            assert np.array_equal(solve(inp), out)
        # Deliberately damage a frame cell in an otherwise correct output.
        broken = {s: [{k: g.copy() for k, g in p.items()} for p in puzzle[s]] for s in puzzle}
        broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("damaged frame accepted")
    assert len(signatures) == 42 and len(structures) >= 180 and len(shapes) >= 25
    assert counts == {3, 5, 7} and periods == {2, 3, 4, 5}
    assert len(crosses) >= 7 and len(capacities) >= 8 and len(directions) == 4
    assert repeat_counts == {3, 4, 5, 6} and role_colors == set(range(10)) and partial >= 100
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(97000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "distinct_structures": len(structures), "shapes": len(shapes),
            "strip_counts": sorted(counts), "periods": sorted(periods),
            "thicknesses": sorted(crosses), "capacities": sorted(capacities),
            "directed_axes": len(directions), "repeat_counts": sorted(repeat_counts),
            "partial_end_pairs": partial, "changed_cells": retained_errors}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
