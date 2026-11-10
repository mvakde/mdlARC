"""Capped echo bars: a tip word echoes into the baseline-nearest floor half.

All geometry and color roles used by solve are visible in its one input.  No
parent code, seed, puzzle demonstrations, or generation metadata is consulted.
"""

from __future__ import annotations

import numpy as np


SIDES = ("left", "right", "top", "bottom")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _key(a):
    return a.shape, tuple(int(x) for x in a.flat)


def _to_left(a, side):
    """Rows become bars, with distance zero at the common endpoint."""
    if side == "left":
        return a
    if side == "right":
        return a[:, ::-1]
    if side == "top":
        return a.T
    if side == "bottom":
        return a[::-1, :].T
    raise ValueError("unknown baseline")


def _from_left(a, side):
    if side == "left":
        return a.copy()
    if side == "right":
        return a[:, ::-1].copy()
    if side == "top":
        return a.T.copy()
    if side == "bottom":
        return a.T[::-1, :].copy()
    raise ValueError("unknown baseline")


def _analyze_input(value):
    """Infer baseline, common body color, and each cap from actual cells."""
    a = _grid(value)
    bg = int(a[0, 0])
    if any(int(a[r, c]) != bg for r, c in
           ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("corners must show the background")
    edges = (a[:, 0], a[:, -1], a[0], a[-1])
    touched = [s for s, edge in zip(SIDES, edges) if np.any(edge != bg)]
    if len(touched) != 1:
        raise ValueError("bars must touch exactly one common canvas side")
    side = touched[0]
    c = _to_left(a, side)
    rows = [int(r) for r in np.flatnonzero(c[:, 0] != bg)]
    if (not 2 <= len(rows) <= 6 or rows[0] == 0 or rows[-1] == c.shape[0] - 1
            or any(y - x < 2 for x, y in zip(rows, rows[1:]))):
        raise ValueError("expected 2..6 separated bars and blank lateral margins")
    bodies = set(int(c[r, 0]) for r in rows)
    if len(bodies) != 1:
        raise ValueError("all bar bodies must have one common color")
    body = bodies.pop()
    expected = np.full(c.shape, bg, dtype=c.dtype)
    bars = []
    for row in rows:
        positions = np.flatnonzero(c[row] != bg)
        length = len(positions)
        if (length >= c.shape[1] or not np.array_equal(positions, np.arange(length))):
            raise ValueError("bar must be contiguous from the baseline and leave headroom")
        prefix = 0
        while prefix < length and c[row, prefix] == body:
            prefix += 1
        cap = c[row, prefix:length].copy()
        k = len(cap)
        if (not 2 <= k <= 5 or length < 2 * k - 1
                or np.any(cap == bg) or np.any(cap == body)):
            raise ValueError("a 2..5-cell cap must fit wholly in the far ceil half")
        expected[row, :length] = c[row, :length]
        bars.append({"row": row, "length": length, "prefix": prefix, "cap": cap})
    if not np.array_equal(c, expected):
        raise ValueError("unexpected cells outside the bars")
    return {"canonical": c, "background": bg, "body": body,
            "side": side, "bars": bars}


def _echo(cap, count):
    period = np.concatenate((cap[::-1], cap))
    return period[np.arange(count) % len(period)]


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(input_grid)
    result = info["canonical"].copy()
    for bar in info["bars"]:
        q = bar["length"] // 2
        result[bar["row"], :q] = _echo(bar["cap"], q)
    return _from_left(result, info["side"])


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _roles(rng):
    colors = rng.permutation(10)
    return int(colors[0]), int(colors[1]), colors[2:2 + int(rng.integers(2, 7))]


def _word(rng, k, colors, asymmetric=False):
    style = int(rng.integers(5))
    if style == 0:
        cap = np.full(k, rng.choice(colors), dtype=np.int8)
    elif style == 1:
        cap = np.resize(rng.choice(colors, size=2, replace=False), k).astype(np.int8)
    elif style == 2:
        cap = np.asarray(rng.choice(colors, size=k), dtype=np.int8)
        for j in range(k // 2):
            cap[-1 - j] = cap[j]
    elif style == 3:
        cap = np.full(k, rng.choice(colors), dtype=np.int8)
        cap[int(rng.integers(1, k)):] = rng.choice(colors)
    else:
        cap = np.asarray(rng.choice(colors, size=k), dtype=np.int8)
    if asymmetric:
        # Distinct ends guarantee that mirror-first and original-first differ.
        cap[0], cap[-1] = rng.choice(colors, size=2, replace=False)
    return cap


def _make_pair(rng, side, roles, training_index=None):
    bg, body, colors = roles
    is_test = training_index is None
    count = int(rng.integers(3 if training_index == 0 else 2, 7))
    limit = 29 if is_test else 25
    bars = []
    for _ in range(count):
        k = int(rng.integers(2, 6 if is_test else 4))
        length = int(rng.integers(2 * k - 1, limit + 1))
        bars.append((length, _word(rng, k, colors)))

    if training_index == 0:
        # A complete M,I,M,I witness, and matched floor-half ties (6 and 7).
        common = _word(rng, 2, colors, asymmetric=True)
        bars[:3] = [(int(rng.integers(18, 26)), _word(rng, 2, colors, True)),
                    (6, common.copy()), (7, common.copy())]
    elif training_index == 1:
        # A second tile width with a partial final tile, rather than a fixed
        # two-color ornament that could be memorized from the first example.
        bars[0] = (int(rng.choice((9, 11, 15, 17, 21, 23))),
                   _word(rng, 3, colors, asymmetric=True))
    elif is_test:
        # Train caps have width 2 or 3.  Generalize to a wider word and stop
        # inside a tile after at least one full mirrored/original pair.
        k = int(rng.integers(4, 6))
        choices = [v for v in range(4 * k + 2, 30) if (v // 2) % k]
        bars[0] = (int(rng.choice(choices)), _word(rng, k, colors, True))

    rng.shuffle(bars)
    # Equal lengths are allowed, as are equal words and wholly symmetric words.
    gaps = [int(rng.integers(1, 4)) for _ in range(count - 1)]
    if rng.random() < 0.22:
        gaps = [int(rng.integers(1, 4))] * (count - 1)
    margins = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
    cross = count + sum(gaps) + sum(margins)  # at most 27
    maximum = max(length for length, _ in bars)
    depth = maximum + int(rng.integers(1, min(4, 30 - maximum) + 1))
    inp = np.full((cross, depth), bg, dtype=np.int8)
    out = inp.copy()
    row = margins[0]
    for index, (length, cap) in enumerate(bars):
        inp[row, :length] = body
        inp[row, length - len(cap):length] = cap
        out[row] = inp[row]
        # Construction uses a separate explicit panel loop; solve parses cells.
        q = length // 2
        for t in range(q):
            panel, offset = divmod(t, len(cap))
            out[row, t] = cap[-1 - offset] if panel % 2 == 0 else cap[offset]
        if index < count - 1:
            row += gaps[index] + 1
    return {"input": _from_left(inp, side), "output": _from_left(out, side)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    fixed_roles = _roles(rng)
    variable_roles = bool(rng.integers(2))
    sides = [str(s) for s in rng.choice(SIDES, size=int(rng.integers(1, 3)), replace=False)]
    seen = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            side = sides[index] if split == "train" and index < len(sides) else str(rng.choice(sides))
            for _ in range(100):
                roles = _roles(rng) if variable_roles else fixed_roles
                pair = _make_pair(rng, side, roles, index if split == "train" else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not sample distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, mode):
    """Deliberate simpler rules used only as diagnostic counterfactuals."""
    result = info["canonical"].copy()
    global_half = max(b["length"] for b in info["bars"]) // 2
    for bar in info["bars"]:
        cap, length = bar["cap"], bar["length"]
        q = length // 2
        if mode == "no_half":
            q = bar["prefix"]
        elif mode == "ceil":
            q = min((length + 1) // 2, bar["prefix"])
        elif mode == "global":
            q = min(global_half, bar["prefix"])
        elif mode == "body_only":
            q = bar["prefix"] // 2
        if mode == "no_reflection":
            values = cap[np.arange(q) % len(cap)]
        elif mode == "mirror_only":
            values = cap[::-1][np.arange(q) % len(cap)]
        else:
            values = _echo(cap, q)
        result[bar["row"], :q] = values
    return _from_left(result, info["side"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal counts")
    seen, infos = set(), {"train": [], "test": []}
    ruled_out = {m: False for m in ("no_reflection", "mirror_only", "no_half", "ceil", "global", "body_only")}
    floor_tie = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze_input(inp)
            infos[split].append(info)
            canonical_out = _to_left(out, info["side"])
            expected_mask = np.zeros(inp.shape, dtype=bool)
            mask = _to_left(expected_mask, info["side"])
            for bar in info["bars"]:
                r, length, cap = bar["row"], bar["length"], bar["cap"]
                q, k = length // 2, len(cap)
                mask[r, :q] = True
                for t in range(q):
                    panel, offset = divmod(t, k)
                    wanted = cap[k - 1 - offset] if panel % 2 == 0 else cap[offset]
                    _require(canonical_out[r, t] == wanted, "wrong mirrored/forward echo or phase")
                _require(np.array_equal(canonical_out[r, q:], info["canonical"][r, q:]), "changed the far half or cap")
            _require(np.array_equal(out[~expected_mask], inp[~expected_mask]), "changed background or far half")
            _require(np.array_equal(out != inp, expected_mask), "changed cells are not exactly the per-bar floor halves")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "introduced a color")
            _require(np.array_equal(solve(inp), out), "input-only solver disagrees")
            if split == "train":
                for mode in ruled_out:
                    ruled_out[mode] |= not np.array_equal(_alternative(info, mode), out)
                for a in info["bars"]:
                    for b in info["bars"]:
                        floor_tie |= (a["length"] % 2 == 0 and b["length"] == a["length"] + 1
                                      and np.array_equal(a["cap"], b["cap"]))
    _require(all(ruled_out.values()), f"training lacks counterexamples: {ruled_out}")
    _require(floor_tie, "training needs equal-word adjacent even/odd length evidence")
    train_sides = {i["side"] for i in infos["train"]}
    _require(all(i["side"] in train_sides for i in infos["test"]), "untrained test orientation")
    train_ks = {len(b["cap"]) for i in infos["train"] for b in i["bars"]}
    _require({2, 3} <= train_ks, "training must demonstrate more than one word width")
    _require(all(any(len(b["cap"]) > max(train_ks) for b in i["bars"])
                 for i in infos["test"]), "missing wider-word test shift")


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s]) for s in ("train", "test"))


def self_test():
    """Bounded structural, semantic, counterfactual, and 42-seed coverage checks."""
    shapes, sides, counts, lengths, widths, residues = set(), set(), set(), set(), set(), set()
    layouts, roles, signatures, gaps, headrooms = set(), set(), set(), set(), set()
    symmetric = asymmetric = clipped = equal_lengths = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                bars, c = info["bars"], info["canonical"]
                shapes.add(pair["input"].shape)
                sides.add(info["side"])
                counts.add(len(bars))
                roles.add((info["background"], info["body"]))
                gaps.add(tuple(b["row"] - a["row"] - 1 for a, b in zip(bars, bars[1:])))
                headrooms.add(c.shape[1] - max(b["length"] for b in bars))
                equal_lengths += len({b["length"] for b in bars}) < len(bars)
                layouts.add((info["side"], c.shape, tuple((b["row"], b["length"], len(b["cap"])) for b in bars)))
                for bar in bars:
                    k, length = len(bar["cap"]), bar["length"]
                    lengths.add(length)
                    widths.add(k)
                    residues.add((k, (length // 2) % (2 * k)))
                    sym = np.array_equal(bar["cap"], bar["cap"][::-1])
                    symmetric += bool(sym)
                    asymmetric += not sym
                    clipped += (length // 2) % k != 0
    _require(len(signatures) == 42, "seed diversity failed")
    _require(sides == set(SIDES) and counts == set(range(2, 7)), "orientation/count coverage failed")
    _require(widths == {2, 3, 4, 5} and len(lengths) >= 24, "word/length coverage failed")
    _require(len(shapes) >= 60 and len(layouts) >= 180, "geometry diversity failed")
    _require(len(roles) >= 30 and len(gaps) >= 40 and headrooms == {1, 2, 3, 4}, "role/spacing coverage failed")
    _require(min(symmetric, asymmetric, clipped) >= 80 and equal_lengths >= 15, "word/partial-tile/tie diversity failed")
    _require(len(residues) >= 24, "echo phase endpoints lack variety")

    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "count ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    for seed in (0, 17, 41):
        _require(_puzzle_key(generate_puzzle(seed)) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")

    # A hand-constructed limit case is outside the training sampling regime:
    # a length-three bar with a two-cell cap and a length-29 bar with a
    # five-cell palindrome. Test input-only solving in every orientation.
    base = np.zeros((5, 30), dtype=np.int8)
    base[1, :3] = [8, 2, 3]
    base[3, :29] = 8
    base[3, 24:29] = [1, 2, 4, 2, 1]
    for side in SIDES:
        inp = _from_left(base, side)
        out = _to_left(solve(inp), side)
        _require(out[1, 0] == 3 and np.array_equal(out[1, 1:], base[1, 1:]), "short bar boundary failed")
        _require(np.all(out[3, 14:24] == 8) and np.array_equal(out[3, 24:], base[3, 24:]), "odd midpoint or cap changed")
    damaged = generate_puzzle(0)
    info = _analyze_input(damaged["train"][0]["input"])
    damaged["train"][0]["output"] = _alternative(info, "no_reflection")
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a reflection ablation")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "geometric_layouts": len(layouts), "baseline_sides": sorted(sides),
            "bar_counts": sorted(counts), "bar_lengths": sorted(lengths),
            "cap_widths": sorted(widths), "phase_endpoints": len(residues),
            "background_body_roles": len(roles), "gap_patterns": len(gaps),
            "symmetric_caps": symmetric, "asymmetric_caps": asymmetric,
            "clipped_echoes": clipped, "pairs_with_equal_lengths": equal_lengths}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
