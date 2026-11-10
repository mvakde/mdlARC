"""Count complete pluses in two equal windows, conditionally recolor, advance.

Only standard-library modules and NumPy are needed. No parent code, saved
examples, random state, or generation metadata enters solve().
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


_CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _pluses(mask):
    """Simultaneous center mask and five-cell union, without edge wrapping."""
    centers = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                     & mask[2:, 1:-1] & mask[1:-1, :-2]
                                     & mask[1:-1, 2:])
    union = centers.copy()
    union[1:] |= centers[:-1]
    union[:-1] |= centers[1:]
    union[:, 1:] |= centers[:, :-1]
    union[:, :-1] |= centers[:, 1:]
    return centers, union


def _visible(grid):
    """Recover all conventions from one input, including a possibly empty tail."""
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("four colors and a unique most frequent background required")
    bg = int(colors[np.argmax(counts)])
    lines = [(c, int(a[0, c])) for c in range(a.shape[1])
             if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    if len(lines) != 1:
        raise ValueError("one full-height nonbackground delimiter required")
    d, line = lines[0]
    dest = 2 * d + 1
    if d < 3 or dest >= a.shape[1] or a.shape[0] < 3:
        raise ValueError("both complete plus windows and the destination must fit")
    if not np.all(a[:, dest] == bg) or np.count_nonzero(a == line) != a.shape[0]:
        raise ValueError("destination is blank and delimiter color occurs only on its line")
    near = set(map(int, np.unique(a[:, :d]))) - {bg}
    far = set(map(int, np.unique(a[:, d + 1:]))) - {bg}
    if len(near) != 1 or len(far) != 1:
        raise ValueError("each side must reveal exactly one marker role")
    ref, cand = near.pop(), far.pop()
    if len({bg, line, ref, cand}) != 4:
        raise ValueError("the four spatial color roles must be distinct")
    return bg, line, ref, cand, d, dest


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Execute one window step from the input alone; do not mutate the input."""
    a = _grid(input_grid)
    bg, line, ref, cand, d, dest = _visible(a)
    ref_centers, _ = _pluses(a[:, :d] == ref)
    cand_centers, union = _pluses(a[:, d + 1:dest] == cand)
    out = a.copy()
    if int(cand_centers.sum()) >= int(ref_centers.sum()):
        block = out[:, d + 1:dest]
        block[union] = ref
    out[:, d] = bg
    out[:, dest] = line
    return out


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _count_arg(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _count(mask):
    return int(_pluses(mask)[0].sum())


def _block(rng, h, w, n, *, overlap=False, marks=None):
    """Build a variable exact-count field; all clutter is checked after painting.

    Bounded attempts retain only the current tiny masks. A failed proposal is
    discarded in memory. No retry artifacts or execution traces are written.
    """
    cap = int(0.48 * h * w)
    for _ in range(24):
        mask = np.zeros((h, w), dtype=bool)
        if overlap and n >= 2:
            # Filled 3x4 or 4x3: two adjacent centers; eight union cells.
            ph, pw = (3, 4) if w >= 4 and (h < 4 or rng.random() < .5) else (4, 3)
            r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
            mask[r:r + ph, c:c + pw] = True
        sites = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        # Random center positions and incidental overlaps replace a fixed tiling.
        for i in rng.permutation(len(sites)):
            if _count(mask) == n:
                break
            r, c = sites[int(i)]
            proposed = mask.copy()
            for dr, dc in _CROSS:
                proposed[r + dr, c + dc] = True
            old, new = _count(mask), _count(proposed)
            if (old < new <= n and int(proposed.sum()) <= cap
                    and not np.any(proposed.all(axis=0))):
                mask = proposed
        if _count(mask) != n:
            continue
        minimum = max(1, int(mask.sum()))
        if marks is None:
            # Clean/sparse, moderate and dense fields all use the same rule.
            density = float(rng.choice((.08, .18, .32, .44)))
            target = max(minimum, min(cap, int(h * w * density)))
        else:
            target = max(minimum, int(marks))
        if target > cap:
            continue

        # Straight/bent strokes, rings, diagonal Xs, Ts, and compact patches.
        # They may touch existing motifs but cannot change the requested count.
        for _ in range(min(24, h * w)):
            if int(mask.sum()) >= target:
                break
            r, c = int(rng.integers(h)), int(rng.integers(w))
            kind = int(rng.integers(6))
            length = int(rng.integers(2, 7))
            if kind == 0:
                points = [(r, c + k) for k in range(length)]
            elif kind == 1:
                points = [(r + k, c) for k in range(length)] + [(r, c + 1)]
            elif kind == 2:
                points = [(r + dr, c + dc) for dr in (-1, 0, 1)
                          for dc in (-1, 0, 1) if dr or dc]
            elif kind == 3:
                points = [(r + dr, c + dc) for dr, dc in
                          ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))]
            elif kind == 4:
                points = [(r, c - 1), (r, c), (r, c + 1), (r + 1, c)]
            else:
                points = [(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)]
            proposed = mask.copy()
            for rr, cc in points:
                if 0 <= rr < h and 0 <= cc < w:
                    proposed[rr, cc] = True
            if (int(proposed.sum()) <= target and _count(proposed) == n
                    and not np.any(proposed.all(axis=0))):
                mask = proposed
        # Monotone single-cell completion has a strict h*w bound.
        for i in rng.permutation(h * w):
            if int(mask.sum()) == target:
                break
            r, c = divmod(int(i), w)
            if not mask[r, c]:
                mask[r, c] = True
                if _count(mask) != n or np.any(mask.all(axis=0)):
                    mask[r, c] = False
        # A marker column must never masquerade as the full-height delimiter.
        # Preserve at least one background cell in every sampled marker column.
        if int(mask.sum()) == target and not np.any(mask.all(axis=0)):
            return mask
    raise RuntimeError(f"bounded construction failed for {h}x{w}, {n} pluses, {marks} marks")


def _assemble(rng, h, d, tail, nr, nc, roles, *, witness=None):
    bg, line, ref, cand = roles
    if witness == "equal":
        # Acceptance despite fewer raw candidate cells; overlap and selective
        # union recoloring are visible even with only two demonstrations.
        cm = _block(rng, h, d, 2, overlap=True, marks=12)
        rm = _block(rng, h, d, 2, marks=int(rng.integers(15, 19)))
    elif witness == "below":
        rm = _block(rng, h, d, 3, marks=16)
        cm = _block(rng, h, d, 2, overlap=True, marks=int(rng.integers(19, 23)))
    else:
        rm = _block(rng, h, d, nr, overlap=nr >= 2 and rng.random() < .35)
        cm = _block(rng, h, d, nc, overlap=nc >= 2 and rng.random() < .45)
    if tail:
        max_tail = min(7, h * tail // 14) if tail >= 3 else 0
        nt = (1 if witness == "below" else int(rng.integers(max_tail + 1)))
        tm = _block(rng, h, tail, nt)
    else:
        tm = np.zeros((h, 0), dtype=bool)
    dest = 2 * d + 1
    inp = np.full((h, dest + 1 + tail), bg, dtype=np.int8)
    inp[:, :d][rm] = ref
    inp[:, d] = line
    inp[:, d + 1:dest][cm] = cand
    inp[:, dest + 1:][tm] = cand
    # Labels use the sampled, verified counts and pre-recolor candidate mask.
    # solve() separately recovers these roles and windows from the finished grid.
    out = inp.copy()
    if nc >= nr:
        out[:, d + 1:dest][_pluses(cm)[1]] = ref
    out[:, d] = bg
    out[:, dest] = line
    return {"input": inp, "output": out}


def _random_spec(rng, d, outcome):
    h = int(rng.integers(8, 31))
    limit = min(10, max(2, h * d // 14))
    if outcome == "equal":
        nr = nc = int(rng.integers(1, limit + 1))
    elif outcome == "above":
        nc = int(rng.integers(2, limit + 1))
        nr = int(rng.integers(1, nc))
    else:
        nr = int(rng.integers(1, limit + 1))
        nc = int(rng.integers(nr))
    # Make counts comfortably feasible even in three-column windows.
    h = max(h, (max(12, 5 * max(nr, nc)) * 100 + 47 * d - 1) // (47 * d))
    h = min(30, h)
    tail = int(rng.integers(0, 30 - (2 * d + 2) + 1))
    return h, d, tail, nr, nc


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one coherent puzzle, including mandatory causal counterexamples."""
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count_arg(num_train, "num_train", 2, 10)
    num_test = _count_arg(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    mixed_palette = bool(rng.integers(2))
    held_width = int(rng.integers(3, 14))
    training_widths = [d for d in range(3, 14) if d != held_width]
    core_width = int(rng.choice([d for d in range(4, 11) if d != held_width]))
    specs = []
    for witness, nr, nc in (("equal", 2, 2), ("below", 3, 2)):
        h = int(rng.integers(12, 26))
        max_tail = 30 - (2 * core_width + 2)
        tail = int(rng.integers(3 if witness == "below" else 0, max_tail + 1))
        specs.append(("train", (h, core_width, tail, nr, nc), witness))
    for i in range(2, num_train):
        outcome = "above" if i == 2 else str(rng.choice(("equal", "above", "below")))
        d = int(rng.choice(training_widths))
        specs.append(("train", _random_spec(rng, d, outcome), None))
    allowed = ("equal", "below", "above") if num_train >= 3 else ("equal", "below")
    test_outcomes = [str(rng.choice(allowed)) for _ in range(num_test)]
    if num_test >= 2:
        test_outcomes[0], test_outcomes[1] = "below", "equal"
        rng.shuffle(test_outcomes)
    for outcome in test_outcomes:
        specs.append(("test", _random_spec(rng, held_width, outcome), None))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, spec, witness in specs:
        pair_roles = tuple(map(int, rng.choice(10, 4, replace=False))) if mixed_palette else roles
        for _ in range(16):
            pair = _assemble(rng, *spec, pair_roles, witness=witness)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("bounded uniqueness retries exhausted")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _enumerate(mask):
    """Direct motif enumeration for validation, rather than boolean dilation."""
    centers, union = set(), set()
    for r in range(1, mask.shape[0] - 1):
        for c in range(1, mask.shape[1] - 1):
            points = {(r + dr, c + dc) for dr, dc in _CROSS}
            if all(mask[p] for p in points):
                centers.add((r, c))
                union.update(points)
    return centers, union


def _facts(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    assert a.shape == b.shape
    bg, line, ref, cand, d, dest = _visible(a)
    rm, cm = a[:, :d] == ref, a[:, d + 1:dest] == cand
    tm = a[:, dest + 1:] == cand
    rc, _ = _enumerate(rm)
    cc, union = _enumerate(cm)
    tc, _ = _enumerate(tm)
    nr, nc = len(rc), len(cc)
    # The semantic rule also works for zero reference pluses. The generated
    # distribution uses a positive threshold so every puzzle has real gating.
    assert nr >= 1
    expected = a.copy()
    if nc >= nr:
        for r, c in union:
            expected[r, d + 1 + c] = ref
    expected[:, d] = bg
    expected[:, dest] = line
    assert np.array_equal(b, expected), "incorrect comparison, union, delimiter move, or preservation"
    assert np.array_equal(b[:, :d], a[:, :d])
    assert np.array_equal(b[:, dest + 1:], a[:, dest + 1:])
    assert np.all(b[:, d] == bg) and np.all(b[:, dest] == line)
    predicted = solve(a)
    assert np.array_equal(predicted, b)
    # Destructive scanning loses one of two adjacent motifs.
    remaining = cm.copy()
    sequential = set()
    for r in range(1, cm.shape[0] - 1):
        for c in range(1, cm.shape[1] - 1):
            points = {(r + dr, c + dc) for dr, dc in _CROSS}
            if all(remaining[p] for p in points):
                sequential.update(points)
                for p in points:
                    remaining[p] = False
    return {"shape": a.shape, "roles": (bg, line, ref, cand), "d": d,
            "reference": nr, "candidate": nc, "tail": len(tc),
            "raw_reference": int(rm.sum()), "raw_candidate": int(cm.sum()),
            "union": len(union), "accepted": nc >= nr,
            "overlap": len(union) < 5 * nc,
            "destructive_differs": nc >= nr and union != sequential,
            "selective": nc >= nr and 0 < len(union) < int(cm.sum()),
            "raw_wrong": (int(cm.sum()) >= int(rm.sum())) != (nc >= nr),
            "far_wrong": nc < nr <= nc + len(tc),
            "edge": any(r in (1, cm.shape[0] - 2) or c in (1, d - 2) for r, c in cc)}


def validate_generated_puzzle(puzzle) -> None:
    """Check visible family invariants, labels, and demonstration sufficiency."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, facts = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            facts[split].append(_facts(pair))
    train = facts["train"]
    assert any(f["reference"] == f["candidate"] == 2 and f["raw_wrong"]
               and f["selective"] and f["destructive_differs"] for f in train)
    assert any(f["reference"] == 3 and f["candidate"] == 2
               and f["raw_wrong"] and f["far_wrong"] for f in train)
    assert any(a["candidate"] == b["candidate"] and a["accepted"] != b["accepted"]
               for a in train for b in train)
    if any(f["candidate"] > f["reference"] for f in facts["test"]):
        assert any(f["candidate"] > f["reference"] for f in train)
    assert {f["d"] for f in train}.isdisjoint(f["d"] for f in facts["test"])


def _digest(puzzle):
    payload = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test() -> dict:
    """42 seeds, all legal counts, boundary motifs, determinism, and bad labels.

    Stream one puzzle at a time; retain only small sets and aggregate counters.
    Return descriptive measurements. Do not write diagnostic files implicitly.
    """
    signatures, shapes, widths, roles, counts, normalized = set(), set(), set(), set(), set(), set()
    totals = dict(pairs=0, overlap=0, selective=0, destructive_differs=0,
                  raw_wrong=0, far_wrong=0, edge=0)
    test_outcomes = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _facts(pair)
                shapes.add(f["shape"])
                widths.add(f["d"])
                roles.add(f["roles"])
                counts.add((f["reference"], f["candidate"], f["tail"]))
                norm = np.zeros(pair["input"].shape, dtype=np.uint8)
                for i, color in enumerate(f["roles"]):
                    norm[pair["input"] == color] = i
                normalized.add(hashlib.sha256(norm.tobytes() + str(norm.shape).encode()).hexdigest())
                totals["pairs"] += 1
                for name in totals.keys() - {"pairs"}:
                    totals[name] += int(f[name])
                if split == "test":
                    test_outcomes.add("below" if f["candidate"] < f["reference"] else
                                      "equal" if f["candidate"] == f["reference"] else "above")
    assert len(signatures) == 42 and len(normalized) >= 195
    assert len(shapes) >= 65 and len(widths) >= 9 and len(roles) >= 30 and len(counts) >= 35
    assert test_outcomes == {"below", "equal", "above"}
    assert all(totals[n] >= 42 for n in ("overlap", "selective", "destructive_differs", "far_wrong"))
    assert totals["raw_wrong"] >= 84 and totals["edge"] > 0
    assert _digest(generate_puzzle(17)) == _digest(generate_puzzle(17))
    # Regression: this explicit count request once created a second full-height
    # candidate column in an 8-row scene and made visible roles ambiguous.
    validate_generated_puzzle(generate_puzzle(0, 5, 2))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(91, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exhaust all local neighborhoods and compare the two motif enumerations.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        cs, us = _enumerate(mask)
        cm, um = _pluses(mask)
        assert cs == set(map(tuple, np.argwhere(cm)))
        assert us == set(map(tuple, np.argwhere(um)))
    cs, us = _enumerate(np.ones((3, 4), dtype=bool))
    assert len(cs) == 2 and len(us) == 8
    clipped = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool)
    assert _count(clipped) == 0
    p = generate_puzzle(7)
    for pair in p["train"] + p["test"]:
        a = pair["input"].copy()
        solve(pair["input"])
        assert np.array_equal(a, pair["input"]), "solver mutated input"
    bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    a = bad["test"][0]["input"]
    bg, line, ref, cand, d, dest = _visible(a)
    bad["test"][0]["output"][:, d] = line  # Incorrectly leave the old delimiter.
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a stationary old delimiter")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "widths": len(widths), "palettes": len(roles), "count_triples": len(counts),
            "color_normalized_structures": len(normalized), "features": totals,
            "test_outcomes": sorted(test_outcomes), "independent_semantic_approval": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
