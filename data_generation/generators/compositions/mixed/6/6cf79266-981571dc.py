"""Repair a reflected field, then union all neutral 3x3 window footprints.

Only NumPy and the standard library are used. No file access, examples, seed,
or generation metadata enter solve(). Blue (1) is reserved for highlighting;
all other color roles are identifiable from each input independently.
"""
from functools import lru_cache
from hashlib import sha256
import numpy as np


HIGHLIGHT = 1


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


@lru_cache(maxsize=30)
def _fold_indices(length):
    """Folds with reflected overlap covering at least three quarters of a side."""
    result = []
    for total in range(1, 2 * length - 2):
        overlap = length - abs(total - (length - 1))
        if 4 * overlap < 3 * length:
            continue
        first = np.array([i for i in range(length) if i < total - i < length], dtype=int)
        if len(first):
            result.append((total, first, total - first))
    return tuple(result)


def _axis_candidates(grid, erase, dimension):
    a = grid if dimension == 0 else grid.T
    result = []
    for total, first, second in _fold_indices(a.shape[0]):
        x, y = a[first], a[second]
        visible = (x != erase) & (y != erase)
        if (int(visible.sum()) >= max(8, a.shape[1])
                and np.all(x[visible] == y[visible])):
            result.append(total)
    return result


def _orbits(shape, axes):
    h, w = shape
    a, b = axes
    seen = np.zeros(shape, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            rows = sorted({r, a - r}) if 0 <= a - r < h else [r]
            cols = sorted({c, b - c}) if 0 <= b - c < w else [c]
            orbit = tuple((y, x) for y in rows for x in cols)
            for p in orbit:
                seen[p] = True
            result.append(orbit)
    return result


def _repair(grid, erase, orbits):
    restored = grid.copy()
    for orbit in orbits:
        colors = {int(grid[p]) for p in orbit if grid[p] != erase}
        if len(colors) != 1:
            raise ValueError("a reflection orbit is empty or contradictory")
        color = colors.pop()
        for p in orbit:
            if grid[p] == erase:
                restored[p] = color
    return restored


def _windows(mask):
    h, w = mask.shape
    if min(h, w) < 3:
        return []
    counts = np.zeros((h - 2, w - 2), dtype=np.int8)
    for dr in range(3):
        for dc in range(3):
            counts += mask[dr:h - 2 + dr, dc:w - 2 + dc]
    return [tuple(map(int, p)) for p in np.argwhere(counts == 9)]


def _union(shape, starts):
    mask = np.zeros(shape, dtype=bool)
    for r, c in starts:
        mask[r:r + 3, c:c + 3] = True
    return mask


def _interpret(grid):
    """Infer the unique erasure marker, both folds, and the majority neutral role."""
    if np.any(grid == HIGHLIGHT):
        raise ValueError("highlight color 1 is reserved and absent from inputs")
    candidates = []
    for erase in map(int, np.unique(grid)):
        rows = _axis_candidates(grid, erase, 0)
        cols = _axis_candidates(grid, erase, 1)
        # The family requires input-visible unique folds, not a hidden tie-break.
        if len(rows) != 1 or len(cols) != 1:
            continue
        axes = rows[0], cols[0]
        orbits = _orbits(grid.shape, axes)
        try:
            clean = _repair(grid, erase, orbits)
        except ValueError:
            continue
        colors, counts = np.unique(clean, return_counts=True)
        neutral = int(colors[int(counts.argmax())])
        if len(colors) < 3 or 2 * int(counts.max()) <= clean.size:
            continue
        starts = _windows(clean == neutral)
        if not starts:
            continue
        candidates.append((erase, neutral, axes, clean, starts))
    if len(candidates) != 1:
        raise ValueError("input does not have a unique erasure/fold interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution for every field satisfying the family specification."""
    grid = _grid(input_grid)
    _, neutral, _, clean, starts = _interpret(grid)
    output = clean.copy()
    output[_union(clean.shape, starts)] = HIGHLIGHT
    return output


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _counts(rng, num_train, num_test):
    # None is a convenience for the supplied inspection utility; defaults are 4/1.
    nt = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    ns = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if nt is None:
        nt = int(rng.integers(max(2, (ns or 1) + 1), 11))
    if ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _geometry(rng, kind, test):
    low, high = (20, 31) if test else ((18, 27) if kind == "single" else (20, 29))
    h, w = map(int, rng.integers(low, high, size=2))
    if rng.random() < 0.22:
        w = h
    if test:
        offsets = list(map(int, rng.integers(-3, 4, size=2)))
        offsets[int(rng.integers(2))] = int(rng.choice([-3, 3]))
    elif rng.random() < 0.30:
        offsets = [0, 0]
    else:
        offsets = list(map(int, rng.integers(-2, 3, size=2)))
    return (h, w), (h - 1 + offsets[0], w - 1 + offsets[1])


def _draw_field(rng, shape, orbits, neutral, payload):
    probability = float(rng.uniform(0.59, 0.72))
    representatives = np.array([o[0] for o in orbits], dtype=int)
    size = len(orbits)
    values = rng.choice(payload, size=size).astype(np.int8)
    values[rng.random(size) < probability] = neutral
    style = int(rng.integers(3))
    if style:
        # Coherent coarse patches or staggered bands, retaining irregular detail.
        bh, bw = map(int, rng.integers(2, 5, size=2))
        coarse = rng.choice(payload, size=shape).astype(np.int8)
        coarse[rng.random(shape) < probability] = neutral
        r, c = representatives.T
        cells = coarse[r // bh, c // bw] if style == 1 else coarse[(r + c // bw) // bh, c // bw]
        use = rng.random(size) < 0.62
        values[use] = cells[use]
    clean = np.empty(shape, dtype=np.int8)
    for value, orbit in zip(values, orbits):
        for p in orbit:
            clean[p] = value
    return clean


def _template(rng, kind):
    if kind == "single":
        return np.ones((3, 3), dtype=bool)
    mode = int(rng.integers(5))
    if mode < 4:
        shape = ((3, 4), (4, 3), (4, 4), (3, 5))[mode]
        return np.ones(shape, dtype=bool)
    mask = np.zeros((5, 5), dtype=bool)
    mask[:3, :3] = mask[2:, 2:] = True
    return mask


def _build_clean(rng, shape, axes, orbits, neutral, payload, kind):
    h, w = shape
    a, b = axes
    ids = np.empty(shape, dtype=np.int16)
    for index, orbit in enumerate(orbits):
        for p in orbit:
            ids[p] = index
    clean = _draw_field(rng, shape, orbits, neutral, payload)
    protected = np.zeros(shape, dtype=bool)

    def paint(points, color, protect=True):
        for index in {int(ids[p]) for p in points}:
            for p in orbits[index]:
                clean[p] = color
                if protect:
                    protected[p] = True

    # A main motif in a fully paired quadrant has four complete reflected copies.
    # Leave a separating row/column at a between-cell fold. In particular the
    # simple demonstration contains genuinely isolated 3x3 blank components.
    rl, rh = max(0, a - h + 1), (a - 2) // 2
    cl, ch = max(0, b - w + 1), (b - 2) // 2
    motif = _template(rng, kind)
    mh, mw = motif.shape
    if rh - rl + 1 < mh or ch - cl + 1 < mw:
        return None
    r0 = int(rng.integers(rl, rh - mh + 2))
    c0 = int(rng.integers(cl, ch - mw + 2))
    points = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(motif)}
    ring = {(y, x) for r, c in points for y, x in
            ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= y < h and 0 <= x < w and (y, x) not in points}
    # Paint border before core so a reflected border cannot damage its own motif.
    for p in sorted(ring):
        paint([p], int(rng.choice(payload)))
    paint(points, neutral)
    if kind != "single":
        # A narrow blank appendage is connected to a target but is not a window.
        core_ids = {int(ids[p]) for p in points}
        tails = [p for p in ((r0 - 1, c0 + 1), (r0 + mh, c0 + mw - 2),
                            (r0 + 1, c0 - 1), (r0 + mh - 2, c0 + mw))
                 if p in ring and int(ids[p]) not in core_ids]
        if not tails:
            return None
        paint([tails[int(rng.integers(len(tails)))]], neutral)

    options = [(r, c) for r in range(rl, rh - 1) for c in range(cl, ch - 1)
               if not protected[r:r + 3, c:c + 3].any()]
    if not options:
        return None
    r1, c1 = options[int(rng.integers(len(options)))]
    near = [(r, c) for r in range(r1, r1 + 3) for c in range(c1, c1 + 3)]
    paint(near, neutral)
    blocker = (r1 + 1, c1 + 1)
    paint([blocker], int(rng.choice(payload)))

    # Further reflected windows can be interior, edge-touching, or cross a fold.
    extras = 0 if kind == "single" else int(rng.integers(0, 4))
    for _ in range(extras):
        for _ in range(12):
            ph, pw = map(int, rng.integers(3, 6, size=2))
            r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
            patch = [(y, x) for y in range(r, r + ph) for x in range(c, c + pw)]
            if any(protected[p] for p in patch):
                continue
            # Reflection-closed protection is already represented by protected.
            paint(patch, neutral)
            break
    return clean, ids, (r0, c0), blocker


def _make_mask(rng, clean, axes, orbits, ids, main, blocker, hard, test, neutral):
    h, w = clean.shape
    density = float(rng.uniform(0.045, 0.12) if test else rng.uniform(0.01, 0.045))
    mask = rng.random(clean.shape) < density
    # Shape and location vary; tests combine larger patches with scattered gaps.
    for _ in range(int(rng.integers(1, 4))):
        ph, pw = map(int, rng.integers(1 if not test else 2, 4 if not test else 7, size=2))
        r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        if rng.random() < 0.5:
            mask[r:r + ph, c:c + pw] = True
        else:
            for dr in range(ph):
                mask[r + dr, c:c + 1 + dr * (pw - 1) // max(1, ph - 1)] = True
    r0, c0 = main
    visible_window = np.zeros_like(mask)
    visible_window[r0:r0 + 3, c0:c0 + 3] = True
    forced = np.zeros_like(mask)

    def force(orbit, survivor=None, triple=False):
        if survivor is None:
            survivor = orbit[int(rng.integers(len(orbit)))]
        for p in orbit:
            mask[p] = False
        choices = [p for p in orbit if p != survivor]
        if not triple:
            choices = [choices[int(rng.integers(len(choices)))]]
        for p in choices:
            mask[p] = forced[p] = True

    center = r0 + 1, c0 + 1
    force(orbits[int(ids[center])], center, hard)
    force(orbits[int(ids[blocker])], triple=hard)
    # Also repair neutral cells that must remain unhighlighted.
    footprint = _union(clean.shape, _windows(clean == neutral))
    outside = [o for o in orbits if len(o) > 1 and clean[o[0]] == neutral
               and all(not footprint[p] and not visible_window[p] for p in o)
               and not any(forced[p] for p in o)]
    if outside:
        force(outside[int(rng.integers(len(outside)))])
    mask[visible_window] = False
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            choices = [p for p in orbit if not forced[p]]
            if not choices:
                return None
            mask[choices[int(rng.integers(len(choices)))]] = False
    return mask


def _make_pair(rng, neutral, erase, pool, kind, test, hard):
    for _ in range(160):
        shape, axes = _geometry(rng, kind, test)
        k = int(rng.integers(2, min(5, len(pool)) + 1))
        payload = list(map(int, rng.choice(pool, size=k, replace=False)))
        orbits = _orbits(shape, axes)
        built = _build_clean(rng, shape, axes, orbits, neutral, payload, kind)
        if built is None:
            continue
        clean, ids, main, blocker = built
        colors, counts = np.unique(clean, return_counts=True)
        if (len(colors) < 3 or int(colors[int(counts.argmax())]) != neutral
                or 2 * int(counts.max()) <= clean.size):
            continue
        starts = _windows(clean == neutral)
        footprint = _union(shape, starts)
        if not np.any((clean == neutral) & ~footprint):
            continue
        mask = _make_mask(rng, clean, axes, orbits, ids, main, blocker,
                          hard, test, neutral)
        if mask is None:
            continue
        if not np.any(mask & ~footprint & (clean == neutral)):
            continue
        inp = clean.copy()
        inp[mask] = erase
        try:
            inferred_erase, inferred_neutral, inferred_axes, repaired, _ = _interpret(inp)
        except ValueError:
            continue
        if ((inferred_erase, inferred_neutral, inferred_axes) != (erase, neutral, axes)
                or not np.array_equal(repaired, clean)):
            continue
        output = clean.copy()
        output[footprint] = HIGHLIGHT
        return {"input": inp, "output": output}
    raise RuntimeError("bounded construction exhausted; no ambiguous fallback is emitted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    palette = list(map(int, rng.permutation([c for c in range(10) if c != HIGHLIGHT])))
    neutral, erase = palette[:2]
    pool = palette[2:]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            kind = "single" if split == "train" and i == 0 else "overlap"
            if split == "train" and i > 1 and rng.random() < 0.35:
                kind = "single"
            hard = split == "test" or i == 1 or (i > 1 and rng.random() < 0.5)
            for _ in range(8):
                pair = _make_pair(rng, neutral, erase, pool, kind, split == "test", hard)
                signature = pair["input"].shape, pair["input"].tobytes()
                if signature not in seen:
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
            seen.add(signature)
            puzzle[split].append(pair)
    # Demonstration order is not part of the rule.
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    return puzzle


def _features(inp):
    erase, neutral, axes, clean, starts = _interpret(inp)
    footprint = _union(inp.shape, starts)
    erased = inp == erase
    blocked = _union(inp.shape, _windows(inp == neutral))
    pretend = inp.copy()
    pretend[erased] = neutral
    false_blank = _union(inp.shape, _windows(pretend == neutral))
    h, w = inp.shape
    a, b = axes
    diagonal = 0
    for r, c in map(tuple, np.argwhere(erased)):
        rr, cc = a - r, b - c
        lr = 0 <= cc < w and cc != c and not erased[r, cc]
        tb = 0 <= rr < h and rr != r and not erased[rr, c]
        diag = 0 <= rr < h and 0 <= cc < w and not erased[rr, cc]
        diagonal += int(not lr and not tb and diag)
    adjacent = np.zeros_like(footprint)
    adjacent[:-1] |= footprint[1:]
    adjacent[1:] |= footprint[:-1]
    adjacent[:, :-1] |= footprint[:, 1:]
    adjacent[:, 1:] |= footprint[:, :-1]
    isolated_square = False
    for r, c in starts:
        perimeter = ([(r - 1, x) for x in range(c, c + 3)] if r else [])
        perimeter += ([(r + 3, x) for x in range(c, c + 3)] if r + 3 < h else [])
        perimeter += ([(y, c - 1) for y in range(r, r + 3)] if c else [])
        perimeter += ([(y, c + 3) for y in range(r, r + 3)] if c + 3 < w else [])
        if all(clean[p] != neutral for p in perimeter):
            isolated_square = True
            break
    return {
        "erase": erase, "neutral": neutral, "axes": axes, "shape": inp.shape,
        "offsets": (a - h + 1, b - w + 1), "windows": len(starts),
        "highlighted": int(footprint.sum()), "overlap": int(footprint.sum()) < 9 * len(starts),
        "hidden_highlight_cells": int(np.count_nonzero(footprint & ~blocked)),
        "false_highlight_cells": int(np.count_nonzero(false_blank & ~footprint)),
        "diagonal_only": diagonal, "erased": int(erased.sum()),
        "neutral_fraction": float(np.mean(clean == neutral)),
        "neutral_tail": bool(np.any(adjacent & ~footprint & (clean == neutral))),
        "isolated_square": isolated_square,
        "visible_window": bool(_windows(inp == neutral)),
        "neutral_repaired_outside": int(np.count_nonzero(erased & ~footprint & (clean == neutral))),
        "payload_repaired": int(np.count_nonzero(erased & (clean != neutral))),
        "palette_size": len(np.unique(clean)),
        "edge_highlight": bool(footprint[0].any() or footprint[-1].any()
                               or footprint[:, 0].any() or footprint[:, -1].any()),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, roles, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = inp.shape, inp.tobytes()
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            erase, neutral, axes, clean, starts = _interpret(inp)
            roles.add((erase, neutral))
            footprint = _union(inp.shape, starts)
            expected = clean.copy()
            expected[footprint] = HIGHLIGHT
            assert np.array_equal(out, expected), "wrong repair or window union"
            assert not np.any(out == erase)
            assert np.array_equal(out == HIGHLIGHT, footprint)
            preserved = (inp != erase) & ~footprint
            assert np.array_equal(out[preserved], inp[preserved]), "visible payload changed"
            assert np.array_equal(inp != out, (inp == erase) | footprint)
            for orbit in _orbits(inp.shape, axes):
                values = {int(inp[p]) for p in orbit if inp[p] != erase}
                assert len(values) == 1, "orbit does not have unanimous surviving evidence"
            f = _features(inp)
            assert f["hidden_highlight_cells"] > 0, "repair never enables highlighting"
            assert f["false_highlight_cells"] > 0, "missing blockers do not reject false windows"
            assert f["payload_repaired"] > 0 and f["neutral_repaired_outside"] > 0
            assert f["visible_window"], "all visible windows were unnecessarily destroyed"
            if split == "train":
                training.append(f)
            else:
                assert f["diagonal_only"] > 0, "test admits a one-fold-only shortcut"
    assert len(roles) == 1, "roles must be consistent within one puzzle"
    assert any(f["overlap"] for f in training), "no all-window-union demonstration"
    assert any(f["isolated_square"] for f in training), "no isolated 3x3 unit demonstration"
    assert any(f["diagonal_only"] > 0 for f in training), "both folds not demonstrated"
    assert any(f["neutral_tail"] for f in training), "component filling not distinguished"


def _digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed study, negative validation and geometric metamorphisms.

    Returns small descriptive metrics; performs no file I/O and retains only one
    puzzle at a time. This is generator-owned validation, not external approval.
    """
    shapes, roles, offsets, palettes, structures, signatures = set(), set(), set(), set(), set(), set()
    windows, erased_counts, fractions = [], [], []
    totals = {"pairs": 0, "overlapping_pairs": 0, "diagonal_only_pairs": 0,
              "edge_highlight_pairs": 0, "neutral_tail_pairs": 0,
              "hidden_highlight_cells": 0, "false_highlight_cells": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                f = _features(inp)
                shapes.add(inp.shape)
                roles.add((f["neutral"], f["erase"]))
                offsets.add(tuple(int(np.sign(x)) for x in f["offsets"]))
                palettes.add(f["palette_size"])
                windows.append(f["windows"])
                erased_counts.append(f["erased"])
                fractions.append(f["neutral_fraction"])
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, sha256(normalized).hexdigest()))
                totals["pairs"] += 1
                for key, feature in (("overlapping_pairs", "overlap"),
                                     ("diagonal_only_pairs", "diagonal_only"),
                                     ("edge_highlight_pairs", "edge_highlight"),
                                     ("neutral_tail_pairs", "neutral_tail")):
                    totals[key] += int(bool(f[feature]))
                for key in ("hidden_highlight_cells", "false_highlight_cells"):
                    totals[key] += f[key]
        # Reflections/quarter-turns preserve semantics, including fringe handling.
        pair = puzzle["test"][0]
        turns = seed % 4
        assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
        # Permuting all non-blue colors also preserves the rule and role inference.
        recolor = np.arange(10, dtype=np.int8)
        nonblue = [c for c in range(10) if c != HIGHLIGHT]
        recolor[nonblue] = np.random.default_rng(seed + 900).permutation(nonblue)
        assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
    assert len(signatures) == 42
    assert len(shapes) >= 65 and len(roles) >= 20 and len(structures) == totals["pairs"]
    assert len(offsets) == 9 and len(palettes) >= 4
    assert len(set(windows)) >= 15 and min(windows) >= 4
    assert totals["overlapping_pairs"] >= 84 and totals["diagonal_only_pairs"] >= 84
    assert totals["edge_highlight_pairs"] > 0 and totals["neutral_tail_pairs"] >= 42
    assert _digest(generate_puzzle(12345)) == _digest(generate_puzzle(12345))
    for nt, ns in ((2, 1), (10, 4), (None, 4), (2, None)):
        p = generate_puzzle(98765, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, "1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(42, 2, 1)
    for kind in ("repair", "highlight", "visible"):
        bad = {s: [{k: a.copy() for k, a in q.items()} for q in p[s]] for s in p}
        pair = bad["test"][0]
        erase, _, _, clean, starts = _interpret(pair["input"])
        footprint = _union(clean.shape, starts)
        masks = {"repair": (pair["input"] == erase) & ~footprint,
                 "highlight": footprint,
                 "visible": (pair["input"] != erase) & ~footprint}
        point = tuple(np.argwhere(masks[kind])[0])
        pair["output"][point] = (int(pair["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"validator accepted corrupted {kind}")
    return {"seeds": 42, **totals, "distinct_shapes": len(shapes), "role_assignments": len(roles),
            "normalized_input_structures": len(structures), "signed_offset_combinations": len(offsets),
            "clean_palette_sizes": sorted(palettes), "window_range": [min(windows), max(windows)],
            "erased_range": [min(erased_counts), max(erased_counts)],
            "neutral_fraction_range": [min(fractions), max(fractions)],
            "independent_semantic_verification": False}
