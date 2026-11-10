"""Joint periodic repair and overlap assembly of shuffled ribbon fragments.

No files, seeds, or examples enter solve(). All searches are bounded by six
fragments, two directed terminals, and a 30-cell finished ribbon.
"""
from __future__ import annotations

from itertools import permutations
import hashlib
import json

import numpy as np


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid using colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _place(panels, order, layout, divider):
    nr, nc = layout
    h, w = panels[0].shape
    result = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        result[r * (h + 1):r * (h + 1) + h,
               c * (w + 1):c * (w + 1) + w] = panels[index]
    return result


def _partition(grid):
    candidates = []
    h, w = grid.shape
    for divider in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == divider, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [h], [-1] + cols + [w]
        heights = [b - a - 1 for a, b in zip(rr, rr[1:])]
        widths = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(heights + widths) < 3 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        panels = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 3 <= len(panels) <= 6 or any(np.any(p == divider) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        background = next(iter(corners))
        inks = set(int(c) for p in panels for c in np.unique(p)) - {background}
        if len(inks) != 1:
            continue
        candidates.append(dict(panels=panels, background=background,
                               ink=inks.pop(), divider=int(divider),
                               layout=(len(heights), len(widths))))
    if len(candidates) != 1:
        raise ValueError("fragments must have one unambiguous divider partition")
    return candidates[0]


def _orientations(parsed):
    bg = parsed["background"]
    result = []
    for axis in ("H", "V"):
        panels = [p if axis == "H" else p.T for p in parsed["panels"]]
        c, w = panels[0].shape[0] - 2, panels[0].shape[1]
        if not 2 <= c <= 7 or not 3 <= w <= 8:
            continue
        if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in panels):
            continue
        bits = np.stack([p[1:-1] != bg for p in panels])
        left, right = bits[:, :, 0].any(axis=1), bits[:, :, -1].any(axis=1)
        starts, ends = np.flatnonzero(~left), np.flatnonzero(~right)
        if len(starts) != 1 or len(ends) != 1 or starts[0] == ends[0]:
            continue
        if not right[starts[0]] or not left[ends[0]]:
            continue
        length = len(panels) * (w - 1) + 1
        if length > 30:
            continue
        result.append(dict(axis=axis, bits=bits, cross=c, width=w,
                           length=length, start=int(starts[0]), end=int(ends[0])))
    if len(result) != 1:
        raise ValueError("blank caps and clear sides must specify one directed axis")
    return result[0]


def _raw_orders(bits, start, end):
    """Unrepaired B ablation: require exactly equal observed edge signatures."""
    middle = [i for i in range(len(bits)) if i not in (start, end)]
    result = []
    for mid in permutations(middle):
        order = (start,) + mid + (end,)
        if all(np.array_equal(bits[a, :, -1], bits[b, :, 0])
               for a, b in zip(order, order[1:])):
            result.append(order)
    return result


def _fit_order(bits, order, length):
    """Enumerate strict-majority periodic explanations of one overlap order."""
    n, cross, width = bits.shape
    coords = np.array([i * (width - 1) + j for i in range(n) for j in range(width)])
    keep = (coords > 0) & (coords < length - 1)
    coords = coords[keep]
    observations = bits[list(order)].transpose(1, 0, 2).reshape(cross, -1)[:, keep]
    cap = max(2, observations.size // 16)
    fits = []
    for period in range(1, (length - 2) // 3 + 1):
        phases = coords % period
        counts = np.bincount(phases, minlength=period)
        ones = np.stack([observations[:, phases == p].sum(axis=1) for p in range(period)], axis=1)
        if np.any(2 * ones == counts):
            continue
        unit = 2 * ones > counts
        # Blank signatures are reserved for the two terminal caps.
        if not np.all(unit.any(axis=0)):
            continue
        errors = int(np.minimum(ones, counts - ones).sum())
        if errors > cap:
            continue
        clean = np.zeros((cross + 2, length), dtype=bool)
        clean[1:-1, 1:-1] = unit[:, np.arange(1, length - 1) % period]
        fits.append((errors, period, clean))
    return fits


def _decode(input_grid):
    grid = _grid(input_grid)
    parsed = _partition(grid)
    d = _orientations(parsed)
    middle = [i for i in range(len(d["bits"])) if i not in (d["start"], d["end"])]
    best_cost, winners = None, []
    for mid in permutations(middle):
        order = (d["start"],) + mid + (d["end"],)
        for errors, period, clean in _fit_order(d["bits"], order, d["length"]):
            if best_cost is None or errors < best_cost:
                best_cost, winners = errors, [(order, period, clean)]
            elif errors == best_cost:
                winners.append((order, period, clean))
    if not winners:
        raise ValueError("no sparse periodic overlap assembly")
    order, period, clean = min(winners, key=lambda item: item[1])
    if any(other_order != order or not np.array_equal(other, clean)
           for other_order, _, other in winners):
        raise ValueError("equally economical assemblies disagree")
    answer = np.where(clean, parsed["ink"], parsed["background"]).astype(np.int8)
    parsed.update(d)
    parsed.update(order=order, period=period, errors=best_cost, clean=clean,
                  answer=answer if d["axis"] == "H" else answer.T.copy())
    return parsed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique least-change periodic ribbon from its input alone."""
    return _decode(input_grid)["answer"].copy()


def _layouts(n, h, w):
    return [(r, n // r) for r in range(1, n + 1) if n % r == 0
            and r * (h + 1) - 1 <= 30 and (n // r) * (w + 1) - 1 <= 30]


def _configurations(n, witness):
    options = []
    for cross in range(3 if witness else 2, 8):
        for width in range(3, 9):
            length = n * (width - 1) + 1
            if length > 30 or not _layouts(n, cross + 2, width):
                continue
            for period in range(2, (length - 2) // 3 + 1):
                if width >= 3 * period:
                    continue
                # Distinct internal starting phases avoid interchangeable clean
                # fragments; this is a sampling choice, not a solver assumption.
                phases = [(i * (width - 1)) % period for i in range(1, n - 1)]
                if len(set(phases)) != n - 2:
                    continue
                options.append((cross, width, period))
    return options


def _unit(rng, cross, period, rich):
    if rich:
        bits = rng.random((cross, period)) < rng.uniform(.30, .70)
        bits[int(rng.integers(cross)), :] = True
    else:
        spine = int(rng.integers(cross))
        bits = np.zeros((cross, period), dtype=bool)
        for p in range(period):
            lo, hi = int(rng.integers(spine + 1)), int(rng.integers(spine, cross))
            bits[lo:hi + 1, p] = True
    return bits


def _join_witness(clean, n, width):
    signatures = [tuple(np.flatnonzero(clean[1:-1, i * (width - 1)])) for i in range(1, n)]
    return any(a != b and len(a) == len(b) for a in signatures for b in signatures)


def _attempt_pair(rng, n, axis, witness, palette, force_rich=None):
    configurations = _configurations(n, witness)
    for _ in range(240):
        cross, width, period = configurations[int(rng.integers(len(configurations)))]
        length = n * (width - 1) + 1
        rich = bool(rng.integers(2)) if force_rich is None else force_rich
        unit = _unit(rng, cross, period, rich)
        if not np.any(~unit) or any(np.array_equal(unit, unit[:, np.arange(period) % p])
                                    for p in range(1, period) if period % p == 0):
            continue
        clean = np.zeros((cross + 2, length), dtype=bool)
        clean[1:-1, 1:-1] = unit[:, np.arange(1, length - 1) % period]
        if witness and not _join_witness(clean, n, width):
            continue
        bits = np.stack([clean[1:-1, i * (width - 1):i * (width - 1) + width]
                         for i in range(n)])
        noisy = bits.copy()
        # Each pair shows both binary flip directions: an interior erasure and
        # a false contact on one seam copy. Distinct phase-coordinate groups
        # retain strict majority across the entire reassembled ribbon.
        interior_ones = [(i, r, j) for i in range(n) for r in range(cross)
                         for j in range(1, width - 1) if bits[i, r, j]]
        seam_zeros = [(i, r, j) for i in range(n) for r in range(cross)
                      for j in (0, width - 1) if not bits[i, r, j]
                      and not (i == 0 and j == 0 or i == n - 1 and j == width - 1)]
        if not interior_ones or not seam_zeros:
            continue
        first = interior_ones[int(rng.integers(len(interior_ones)))]
        group = lambda q: (q[1], (q[0] * (width - 1) + q[2]) % period)
        seam_zeros = [q for q in seam_zeros if group(q) != group(first)]
        if not seam_zeros:
            continue
        second = seam_zeros[int(rng.integers(len(seam_zeros)))]
        selected, groups = [first, second], {group(first), group(second)}
        noise_cap = max(2, cross * (n * width - 2) // 16)
        desired = int(rng.integers(2, min(4, noise_cap) + 1))
        available = [(i, r, j) for i in range(n) for r in range(cross) for j in range(width)
                     if not (i == 0 and j == 0 or i == n - 1 and j == width - 1)]
        rng.shuffle(available)
        for q in available:
            if len(selected) == desired:
                break
            if group(q) not in groups:
                selected.append(q)
                groups.add(group(q))
        for q in selected:
            noisy[q] = ~noisy[q]
        left, right = noisy[:, :, 0].any(axis=1), noisy[:, :, -1].any(axis=1)
        if np.count_nonzero(~left) != 1 or np.count_nonzero(~right) != 1:
            continue
        if _raw_orders(noisy, 0, n - 1):
            continue
        bg, ink, divider = palette
        panels = []
        for b in noisy:
            p = np.full((cross + 2, width), bg, dtype=np.int8)
            p[1:-1] = np.where(b, ink, bg)
            panels.append(p if axis == "H" else p.T.copy())
        layouts = _layouts(n, *panels[0].shape)
        if not layouts:
            continue
        layout = layouts[int(rng.integers(len(layouts)))]
        order = list(map(int, rng.permutation(n)))
        # At least the geometry witness defeats the cap-then-reading-order rule.
        middle = [i for i in order if i not in (0, n - 1)]
        if witness and middle == list(range(1, n - 1)):
            a, b = order.index(1), order.index(2)
            order[a], order[b] = order[b], order[a]
        inp = _place(panels, order, layout, divider)
        out = np.where(clean, ink, bg).astype(np.int8)
        if axis == "V":
            out = out.T.copy()
        try:
            decoded = _decode(inp)
        except ValueError:
            continue
        if (decoded["errors"] != len(selected) or decoded["period"] != period
                or not np.array_equal(decoded["answer"], out)):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError(f"bounded fragment sampling exhausted for {n}/{axis}/{witness}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate all arrays in memory; no retained runs, traces, or disk writes."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    train_n = list(map(int, rng.integers(3, 6, size=num_train)))
    witness_index = int(rng.integers(num_train))
    train_n[witness_index] = int(rng.integers(4, 6))
    # A second demonstration differs in count, even for a two-example puzzle.
    other = (witness_index + 1) % num_train
    train_n[other] = int(rng.choice([n for n in range(3, 6) if n != train_n[witness_index]]))
    test_n = list(map(int, rng.integers(5, 7, size=num_test)))
    total = num_train + num_test
    axes = list(rng.choice(["H", "V"], size=total))
    # Some puzzles remain homogeneous; otherwise expose both axes in training.
    if rng.random() < .20:
        axes = [axes[0]] * total
    elif len(set(axes[:num_train])) == 1:
        axes[other] = "V" if axes[witness_index] == "H" else "H"
    role_pool = list(permutations(range(10), 3))
    role_indices = rng.choice(len(role_pool), size=total, replace=False)
    puzzle = {"train": [], "test": []}
    for serial, n in enumerate(train_n + test_n):
        split = "train" if serial < num_train else "test"
        pair = _attempt_pair(rng, n, str(axes[serial]), serial == witness_index,
                             role_pool[int(role_indices[serial])])
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    d = _decode(inp)
    assert np.array_equal(out, d["answer"]), "output differs from joint periodic assembly"
    n, width, cross = len(d["bits"]), d["width"], d["cross"]
    clean = d["clean"][1:-1]
    repaired = np.stack([clean[:, i * (width - 1):i * (width - 1) + width]
                         for i in range(n)])
    observed = d["bits"][list(d["order"])]
    changed = observed != repaired
    assert np.count_nonzero(changed) == d["errors"] >= 2
    assert np.any(changed[:, :, 1:-1]), "interior repair must require repetition"
    assert np.any(changed[:, :, (0, width - 1)]), "repair must restore a broken seam"
    assert np.any(observed & ~repaired) and np.any(~observed & repaired)
    assert not _raw_orders(d["bits"], d["start"], d["end"]), "raw port matching must fail"
    assert width < 3 * d["period"], "fragments must not independently show three periods"
    # Every source observation, including both copies of seams, is accounted for.
    assert out.shape == ((cross + 2, d["length"]) if d["axis"] == "H"
                         else (d["length"], cross + 2))
    assert not np.any(out == d["divider"])
    for i in range(n - 1):
        assert np.array_equal(repaired[i, :, -1], repaired[i + 1, :, 0])
    joins = [tuple(np.flatnonzero(repaired[i, :, -1])) for i in range(n - 1)]
    equal_size = any(a != b and len(a) == len(b) for a in joins for b in joins)
    natural_middle = tuple(i for i in range(n) if i not in (d["start"], d["end"]))
    moved_middle = d["order"][1:-1] != natural_middle
    return dict(n=n, width=width, cross=cross, period=d["period"], axis=d["axis"],
                layout=d["layout"], errors=d["errors"], length=d["length"],
                equal_size=equal_size, moved_middle=moved_middle,
                palette=(d["background"], d["ink"], d["divider"]),
                separated=any(len(s) > 1 and s[-1] - s[0] + 1 != len(s) for s in joins),
                partial=(d["length"] - 2) % d["period"] != 0)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, records = set(), {}
    for split in ("train", "test"):
        records[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            records[split].append(_analyze(pair))
    assert len({r["n"] for r in records["train"]}) >= 2
    assert any(r["equal_size"] and r["moved_middle"] for r in records["train"])
    assert all(r["n"] >= 5 for r in records["test"])
    assert len({r["palette"] for split in records.values() for r in split}) == a + b


def _puzzle_digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
              for s in ("train", "test")}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Sequential 42-seed audit; retain descriptors/digests rather than grids."""
    records, digests, shapes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                records.append(_analyze(pair))
                shapes.add(np.asarray(pair["input"]).shape)
                original = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solver mutated its input"
    assert len(digests) == 42
    assert {r["n"] for r in records} == {3, 4, 5, 6}
    assert {r["axis"] for r in records} == {"H", "V"}
    assert len({r["period"] for r in records}) >= 4
    assert len({r["width"] for r in records}) >= 3
    assert len({r["cross"] for r in records}) >= 5
    assert len({r["layout"] for r in records}) >= 6
    assert len(shapes) >= 30
    assert any(r["separated"] for r in records)
    assert any(r["partial"] for r in records) and any(not r["partial"] for r in records)
    assert _puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Wrong pixels, retained dividers, and inconsistent overlaps are rejected.
    p = generate_puzzle(91, 2, 1)
    for pair in p["train"]:
        # Palette and input-layout changes cannot alter geometric semantics.
        color_map = np.array([7, 2, 8, 1, 9, 3, 0, 6, 4, 5], dtype=np.int8)
        assert np.array_equal(solve(color_map[pair["input"]]), color_map[pair["output"]])
        assert np.array_equal(solve(pair["input"].T), pair["output"].T)
        d = _decode(pair["input"])
        reordered = _place(d["panels"], list(reversed(range(len(d["panels"])))),
                           d["layout"], d["divider"])
        assert np.array_equal(solve(reordered), pair["output"])
        if d["order"][1:-1] != tuple(i for i in range(len(d["bits"]))
                                     if i not in (d["start"], d["end"])):
            forced = (d["start"],) + tuple(i for i in range(len(d["bits"]))
                                          if i not in (d["start"], d["end"])) + (d["end"],)
            alternatives = _fit_order(d["bits"], forced, d["length"])
            assert all(cost > d["errors"] for cost, _, _ in alternatives)
        bad = {k: v.copy() for k, v in pair.items()}
        bad["output"].flat[0] = (int(bad["output"].flat[0]) + 1) % 10
        try:
            _analyze(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted answer accepted")
    # Two identical intermediate fragments admit two orders even though the
    # rendered ribbons agree. The stated unique-order contract rejects this.
    clean = np.zeros((5, 13), dtype=bool)
    unit = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=bool)
    clean[1:-1, 1:-1] = unit[:, np.arange(1, 12) % 3]
    panels = [clean[:, 3 * i:3 * i + 4].astype(np.int8) for i in range(4)]
    ambiguous = _place(panels, list(range(4)), (1, 4), 2)
    try:
        solve(ambiguous)
    except ValueError as exc:
        assert "equally economical" in str(exc)
    else:
        raise AssertionError("interchangeable-fragment ambiguity accepted")
    return {"seeds": 42, "pairs": len(records), "input_shapes": len(shapes),
            "periods": sorted({r["period"] for r in records}),
            "layouts": sorted({r["layout"] for r in records}),
            "counts": sorted({r["n"] for r in records})}
