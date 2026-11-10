"""Repair six periodic masks, then label their unique strict union relation.

All solving conventions are local to the input or fixed by this family:
8 highlights operands and 3 highlights the result. No evidence files are read.
Generation, validation and self_test write no files and retain no traces.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


OPERAND_COLOR = 8
RESULT_COLOR = 3
_LAYOUTS = {(1, 6), (2, 3), (3, 2), (6, 1)}


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, numbers.Integral) or not lo <= value <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape) or
            not np.issubdtype(grid.dtype, np.integer) or
            np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid


def _parse(value):
    grid = _grid(value)
    structure = int(grid[0, 0])
    rows = np.flatnonzero(np.all(grid == structure, axis=1)).tolist()
    cols = np.flatnonzero(np.all(grid == structure, axis=0)).tolist()
    h, w = grid.shape
    if (not rows or not cols or rows[0] != 0 or rows[-1] != h - 1 or
            cols[0] != 0 or cols[-1] != w - 1 or
            (len(rows) - 1, len(cols) - 1) not in _LAYOUTS or
            len(set(np.diff(rows))) != 1 or len(set(np.diff(cols))) != 1):
        raise ValueError("expected six equal framed compartments")
    mh, mw = rows[1] - rows[0] - 3, cols[1] - cols[0] - 3
    if min(mh, mw) < 1 or mh == mw:
        raise ValueError("motifs must have a visible long axis")
    rectangles = []
    motif_cells = np.zeros(grid.shape, dtype=bool)
    frame = np.zeros(grid.shape, dtype=bool)
    frame[rows, :] = True
    frame[:, cols] = True
    for top, bottom in zip(rows[:-1], rows[1:]):
        for left, right in zip(cols[:-1], cols[1:]):
            rect = (slice(top + 2, bottom - 1), slice(left + 2, right - 1))
            rectangles.append(rect)
            motif_cells[rect] = True
    padding_values = np.unique(grid[~(frame | motif_cells)])
    if len(padding_values) != 1:
        raise ValueError("padding must have one common color")
    padding = int(padding_values[0])
    if structure == padding or {structure, padding} & {OPERAND_COLOR, RESULT_COLOR}:
        raise ValueError("structural colors collide with highlights")
    accents, raw_masks = [], []
    for rect in rectangles:
        block = grid[rect]
        colors = set(int(c) for c in np.unique(block))
        if len(colors) != 2 or structure not in colors:
            raise ValueError("each motif must use structure and one accent")
        accent = next(iter(colors - {structure}))
        if accent in (padding, OPERAND_COLOR, RESULT_COLOR):
            raise ValueError("invalid motif accent")
        accents.append(accent)
        raw_masks.append(block == accent)
    return {"structure": structure, "padding": padding,
            "rows": len(rows) - 1, "cols": len(cols) - 1,
            "shape": (mh, mw), "vertical": mh > mw,
            "rectangles": rectangles, "motif_cells": motif_cells,
            "accents": accents, "raw_masks": raw_masks}


def _repair(mask, vertical):
    """Accept only a unique repair among all supported periodic explanations.

    A phase needs at least three observations and a strict binary majority.
    Every period up to length//3 is considered; not just a construction period.
    """
    seq = mask.T if vertical else mask
    cross, length = seq.shape
    limit = max(1, seq.size // 8)
    candidates = {}
    for period in range(1, length // 3 + 1):
        template = np.empty((cross, period), dtype=bool)
        supported = True
        for phase in range(period):
            copies = seq[:, phase::period]
            votes = np.count_nonzero(copies, axis=1)
            if np.any(2 * votes == copies.shape[1]):
                supported = False
                break
            template[:, phase] = 2 * votes > copies.shape[1]
        if not supported:
            continue
        repaired = template[:, np.arange(length) % period]
        errors = int(np.count_nonzero(repaired != seq))
        if errors <= limit and np.any(repaired) and not np.all(repaired):
            key = repaired.tobytes()
            if key not in candidates:
                candidates[key] = (repaired.copy(), period, errors)
    if len(candidates) != 1:
        raise ValueError("periodic repair is absent or ambiguous")
    repaired, period, errors = next(iter(candidates.values()))
    return (repaired.T if vertical else repaired), period, errors


def _mask_int(mask):
    return int.from_bytes(np.packbits(np.asarray(mask).ravel()).tobytes(), "big")


def _relations(masks):
    """Exact OR, with a nonempty exclusive contribution from each operand."""
    bits = [_mask_int(mask) for mask in masks]
    return [(a, b, c) for a in range(6) for b in range(a + 1, 6)
            if bits[a] & ~bits[b] and bits[b] & ~bits[a]
            for c in range(6) if c not in (a, b) and bits[a] | bits[b] == bits[c]]


def _chosen(masks):
    triples = _relations(masks)
    signatures = {(tuple(sorted((_mask_int(masks[a]), _mask_int(masks[b])))),
                   _mask_int(masks[c])) for a, b, c in triples}
    if len(signatures) != 1 or len({c for _, _, c in triples}) != 1:
        raise ValueError("expected one mask-level union and one result slot")
    return min(triples)


def _analyze(input_grid):
    parsed = _parse(input_grid)
    repairs = [_repair(mask, parsed["vertical"]) for mask in parsed["raw_masks"]]
    parsed["masks"] = [r[0] for r in repairs]
    parsed["periods"] = [r[1] for r in repairs]
    parsed["errors"] = [r[2] for r in repairs]
    parsed["relation"] = _chosen(parsed["masks"])
    parsed["ties"] = len(_relations(parsed["masks"]))
    return parsed


def _paint(input_grid, parsed, highlight=True):
    answer = np.asarray(input_grid).copy()
    a, b, c = parsed["relation"]
    for slot, (rect, mask, accent) in enumerate(zip(
            parsed["rectangles"], parsed["masks"], parsed["accents"])):
        color = accent
        if highlight:
            color = (RESULT_COLOR if slot == c else
                     OPERAND_COLOR if slot in (a, b) else accent)
        answer[rect] = np.where(mask, color, parsed["structure"])
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from a single input, without puzzle state, random numbers or files."""
    return _paint(input_grid, _analyze(input_grid))


def _base(rng, cross, period, style):
    base = np.zeros((cross, period), dtype=bool)
    if style == 0:
        # Short bars: repeated long-axis units make stripes and ladders.
        row = int(rng.integers(cross))
        start = int(rng.integers(period))
        base[row, start:start + int(rng.integers(1, period + 1))] = True
    elif style == 1:
        # Staircase fragments in the repeating unit.
        for r in range(cross):
            base[r, (r + int(rng.integers(2))) % period] = True
    elif style == 2:
        r, c = int(rng.integers(cross)), int(rng.integers(period))
        base[r:min(cross, r + 2), c:min(period, c + 2)] = True
    else:
        base = rng.random((cross, period)) < float(rng.uniform(0.15, 0.5))
    # Keep shared foreground and enough blank coordinates for both additions
    # and distractors larger than the result. This also works for thin lanes.
    flat = base.ravel()
    if not np.any(flat):
        flat[int(rng.integers(flat.size))] = True
    if np.count_nonzero(flat) > flat.size - 4:
        chosen = rng.choice(np.flatnonzero(flat), size=flat.size - 4, replace=False)
        flat[:] = False
        flat[chosen] = True
    return base


def _units(rng, cross, period, style, duplicate):
    n = cross * period
    for _ in range(300):
        base = _base(rng, cross, period, style)
        empty = np.flatnonzero(~base.ravel())
        rng.shuffle(empty)
        spare = len(empty) - 2
        na = int(rng.integers(1, min(3, spare - 1) + 1))
        nb = int(rng.integers(1, min(3, spare - na) + 1))
        a, b = base.copy(), base.copy()
        a.flat[empty[:na]] = True
        b.flat[empty[na:na + nb]] = True
        union = a | b
        units = [a, b, union]
        # A bigger distractor makes foreground count an insufficient rule.
        big = np.zeros((cross, period), dtype=bool)
        size = int(rng.integers(np.count_nonzero(union) + 1, n))
        big.flat[rng.choice(n, size=size, replace=False)] = True
        units.append(big)
        if duplicate == "operand":
            units.append(units[int(rng.integers(2))].copy())
        else:
            near = units[int(rng.integers(4))].copy()
            flip = rng.choice(n, size=int(rng.integers(1, min(3, n) + 1)), replace=False)
            near.flat[flip] = ~near.flat[flip]
            units.append(near)
        if duplicate == "distractor":
            units.append(big.copy())
        else:
            other = rng.random((cross, period)) < float(rng.uniform(0.2, 0.7))
            units.append(other)
        if any(not np.any(u) or np.all(u) for u in units):
            continue
        unique_count = len({u.tobytes() for u in units})
        if unique_count != (6 if duplicate == "none" else 5):
            continue
        if len(_relations(units)) != (2 if duplicate == "operand" else 1):
            continue
        rng.shuffle(units)
        try:
            _chosen(units)
        except ValueError:
            continue
        return units
    raise RuntimeError("bounded unit-mask search exhausted")


def _noisy(rng, clean, period, direction=None, first_cycle=False):
    noisy = clean.copy()
    cross, length = clean.shape
    amount = int(rng.integers(1, min(3, max(1, clean.size // 10)) + 1))
    used = set()
    for index in range(amount):
        positions = [(r, c) for r in range(cross) for c in range(length)
                     if (r, c % period) not in used and
                     (index != 0 or not first_cycle or c < period) and
                     (index != 0 or direction is None or bool(clean[r, c]) == direction)]
        r, c = positions[int(rng.integers(len(positions)))]
        noisy[r, c] = not noisy[r, c]
        used.add((r, c % period))
    return noisy


def _compose(masks, vertical, structure, padding, accents, strip):
    rows, cols = ((1, 6) if vertical else (6, 1)) if strip else (
        (2, 3) if vertical else (3, 2))
    physical = [m.T if vertical else m for m in masks]
    mh, mw = physical[0].shape
    grid = np.full((rows * (mh + 3) + 1, cols * (mw + 3) + 1),
                   padding, dtype=np.int8)
    grid[::mh + 3, :] = structure
    grid[:, ::mw + 3] = structure
    for slot, mask in enumerate(physical):
        r, c = divmod(slot, cols)
        top, left = r * (mh + 3) + 2, c * (mw + 3) + 2
        grid[top:top + mh, left:left + mw] = np.where(mask, accents[slot], structure)
    return grid


def _make_pair(rng, *, strip, vertical, length, period, cross, duplicate,
               forbidden_relations, separated):
    # A maximum of 160 live attempts, discarded immediately. Nothing is logged.
    colors = [c for c in range(10) if c not in (OPERAND_COLOR, RESULT_COLOR)]
    structure, padding = [int(c) for c in rng.choice(colors, size=2, replace=False)]
    available = [c for c in colors if c not in (structure, padding)]
    palette_mode = int(rng.integers(3))
    accents = ([int(rng.choice(available))] * 6 if palette_mode == 0 else
               [int(c) for c in rng.choice(available, size=6, replace=palette_mode == 1)])
    columns = 6 if strip and vertical else 1 if strip else 3 if vertical else 2
    for _ in range(160):
        try:
            units = _units(rng, cross, period, int(rng.integers(4)), duplicate)
        except RuntimeError:
            continue
        clean = [u[:, np.arange(length) % period] for u in units]
        relation = _chosen(clean)
        a, b, _ = relation
        ar, ac = divmod(a, columns)
        br, bc = divmod(b, columns)
        if relation in forbidden_relations or (separated and abs(ar - br) + abs(ac - bc) < 2):
            continue
        noisy = [_noisy(rng, m, period, True if i == 0 else False if i == 1 else None,
                        first_cycle=i == 0)
                 for i, m in enumerate(clean)]
        if _relations(noisy):
            continue
        try:
            recoveries = [_repair(m, False) for m in noisy]
        except ValueError:
            continue
        if not all(np.array_equal(m, r[0]) for m, r in zip(clean, recoveries)):
            continue
        if not any(np.any(n[:, :r[1]] != r[0][:, :r[1]]) for n, r in zip(noisy, recoveries)):
            continue
        input_grid = _compose(noisy, vertical, structure, padding, accents, strip)
        physical_clean = [m.T if vertical else m for m in clean]
        # Build labels from the intended clean construction, then verify that
        # input-only recovery finds those masks (never generate labels via solve).
        parsed = _parse(input_grid)
        parsed.update(masks=physical_clean, relation=relation)
        output_grid = _paint(input_grid, parsed)
        assert np.array_equal(solve(input_grid), output_grid)
        return {"input": input_grid, "output": output_grid}, relation
    raise RuntimeError("bounded pair search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    strip = bool(rng.random() < 0.25)
    tie_enabled = bool(rng.random() < 0.45)
    first_axis = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen, training_relations = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            vertical = (first_axis if index == 0 else not first_axis) if (
                split == "train" and index < 2) else bool(rng.integers(2))
            if split == "train":
                if strip:
                    length = int(rng.integers(15, 18)) if index == 0 else int(rng.integers(18, 21))
                    period = 5 if index == 0 else 6
                else:
                    length = 8 if index == 0 else 9
                    period = 2 if index == 0 else 3
            else:
                length = int(rng.integers(22, 27)) if strip else int(rng.integers(10, 12))
                period = int(rng.integers(5, length // 3 + 1)) if strip else int(rng.integers(2, 4))
            cross = 1 if strip else int(rng.integers(3 if period == 2 else 2, 7))
            duplicate = ("operand" if tie_enabled and index == 0 else
                         ("operand" if tie_enabled and rng.random() < 0.3 else
                          "distractor" if rng.random() < 0.3 else "none"))
            for _ in range(12):
                pair, relation = _make_pair(
                    rng, strip=strip, vertical=vertical, length=length, period=period,
                    cross=cross, duplicate=duplicate,
                    forbidden_relations=training_relations,
                    separated=split == "train" and index == 0)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input search exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                training_relations.add(relation)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Structural, repair, relational, preservation and demonstration checks."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, descriptions = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            d = _analyze(inp)
            assert all(e > 0 for e in d["errors"]), "all six compartments require repair"
            assert any(np.any((raw.T if d["vertical"] else raw)[:, :p] !=
                                 (mask.T if d["vertical"] else mask)[:, :p])
                       for raw, mask, p in zip(d["raw_masks"], d["masks"], d["periods"]))
            assert not _relations(d["raw_masks"]), "raw masks must not solve the union"
            assert np.array_equal(out, _paint(inp, d))
            assert np.array_equal(inp[~d["motif_cells"]], out[~d["motif_cells"]])
            a, b, c = d["relation"]
            masks = d["masks"]
            assert np.any(masks[a] & masks[b]), "expose OR versus XOR"
            assert np.any(masks[a] & ~masks[b]) and np.any(masks[b] & ~masks[a])
            assert any(np.count_nonzero(m) > np.count_nonzero(masks[c]) for m in masks)
            clean_grid = _paint(inp, d, highlight=False)
            assert not np.array_equal(clean_grid, out), "B must affect labels"
            assert any(np.any(inp[r] != clean_grid[r]) for i, r in enumerate(d["rectangles"])
                       if i not in (a, b, c)), "repair unrelated motifs too"
            # Explicitly check motif zeros after repair: recoloring never fills them.
            for rect, mask in zip(d["rectangles"], masks):
                assert np.all(out[rect][~mask] == d["structure"])
            descriptions[split].append(d)
    train, test = descriptions["train"], descriptions["test"]
    assert len({d["vertical"] for d in train}) == 2
    assert len({d["relation"] for d in train}) == len(train)
    train_relations = {d["relation"] for d in train}
    assert all(d["relation"] not in train_relations for d in test)
    assert min(max(d["shape"]) for d in test) > max(max(d["shape"]) for d in train)
    if any(d["ties"] > 1 for d in test):
        assert any(d["ties"] > 1 for d in train)
    return None


def self_test():
    """Bounded 42-seed diversity audit; retain only sets of small signatures."""
    # Hand-constructed tests establish strictness, overlap and duplicate order
    # independently of random mask construction.
    bits = [np.array([bool(int(c)) for c in f"{n:05b}"])[None, :]
            for n in (5, 6, 7, 27, 24, 20)]
    assert _relations(bits) == [(0, 1, 2)]
    tied = bits[:5] + [bits[0].copy()]
    assert _relations(tied) == [(0, 1, 2), (1, 5, 2)]
    assert _chosen(tied) == (0, 1, 2)
    template = np.array([[True, False, True]])
    clean = template[:, np.arange(13) % 3]
    noisy = clean.copy()
    noisy[0, 0] = False
    recovered, period, errors = _repair(noisy, False)
    assert np.array_equal(recovered, clean) and (period, errors) == (3, 1)
    assert not np.array_equal(noisy[:, :3], template), "copying the first tile fails"
    shapes, units, palettes, relations, signatures, periods = set(), set(), set(), set(), set(), set()
    ties, strips, partials, pair_count = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                d = _analyze(inp)
                assert np.array_equal(solve(inp.tolist()), pair["output"])
                digest.update(bytes(inp.shape))
                digest.update(inp.tobytes())
                shapes.add(inp.shape)
                units.add(tuple(_mask_int(m) for m in d["masks"]))
                palettes.add((d["structure"], d["padding"], tuple(d["accents"])))
                relations.add(d["relation"])
                periods.update(d["periods"])
                ties += d["ties"] > 1
                strips += min(d["shape"]) == 1
                partials += any(max(d["shape"]) % p for p in d["periods"])
                pair_count += 1
        signatures.add(digest.digest())
    assert len(signatures) == 42 and pair_count == 210
    assert len(shapes) >= 25 and len(units) >= 180 and len(palettes) >= 100
    assert len(relations) >= 40 and {2, 3, 5, 6} <= periods
    assert 0 < ties < pair_count and 0 < strips < pair_count and partials > 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(123, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(42, 5, 3), generate_puzzle(42, 5, 3)
    assert all(np.array_equal(a[r], b[r]) for s in ("train", "test")
               for a, b in zip(first[s], second[s]) for r in ("input", "output"))


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus all legal counts)")
