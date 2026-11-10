"""An isolated-dot checksum selects a reflected completion by cavity count.

Only standard library and NumPy are used. Public prediction is input-only.
The sampler constructs competing visible hypotheses; none is a parent generator.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))
_NAMES = ("horizontal", "vertical", "diagonal", "antidiagonal")


def _components(mask, diagonal=False):
    """Deterministic connected components of true cells, including empty masks."""
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(cells)
    return result


def _bbox(mask):
    rr, cc = np.nonzero(mask)
    assert len(rr), "Empty bounding box"
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer), "Noninteger grid"
    assert np.all((0 <= a) & (a <= 9)), "Invalid color"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _maps(n):
    rr, cc = np.indices((n, n))
    return ((n - 1 - rr, cc), (rr, n - 1 - cc),
            (cc, rr), (n - 1 - cc, n - 1 - rr))


def _hypotheses(panel, background, foreground, mask_color):
    """Complete each directly recoverable mirror and check all visible pairs.

    Equal crops are consolidated: extra agreeing symmetries are harmless.
    Counting is done on the entire completed panel, never the source patch.
    """
    hidden = panel == mask_color
    box = _bbox(hidden)
    r0, r1, c0, c1 = box
    assert hidden.sum() == (r1 - r0) * (c1 - c0)
    predictions = {}
    for code, (rr, cc) in enumerate(_maps(len(panel))):
        if np.any(hidden[rr, cc][hidden]):
            continue  # A fixed erased cell or an entirely erased mirror pair.
        visible = ~hidden & ~hidden[rr, cc]
        if not np.array_equal(panel[visible], panel[rr, cc][visible]):
            continue
        # The intact border alone must not constitute all the mirror evidence.
        if np.count_nonzero(visible & (panel == background)) < 2:
            continue
        full = panel.copy()
        full[hidden] = panel[rr, cc][hidden]
        if len(_components(full == foreground)) != 1:
            continue
        cavities = _components(full == background)
        crop = full[r0:r1, c0:c1].copy()
        key = _key(crop)
        if key in predictions:
            predictions[key]["axes"].append(code)
            continue
        sr0, sr1, sc0, sc1 = _bbox(hidden[rr, cc])
        native = panel[sr0:sr1, sc0:sc1]
        joining = False
        crossing = False
        for cells in cavities:
            inside = sum(bool(hidden[q]) for q in cells)
            crossing |= 0 < inside < len(cells)
            if 0 < inside < len(cells):
                remainder = np.zeros(panel.shape, dtype=bool)
                for q in cells:
                    if not hidden[q]:
                        remainder[q] = True
                joining |= len(_components(remainder)) >= 2
        predictions[key] = {
            "full": full, "crop": crop, "axes": [code],
            "count": len(cavities), "area": int((full == background).sum()),
            "eight_count": len(_components(full == background, True)),
            "sizes": tuple(sorted(map(len, cavities))),
            "crossing": crossing, "joining": joining,
            "orientation_visible": native.shape != crop.shape or not np.array_equal(native, crop),
            "support": int(visible.sum()),
        }
    return list(predictions.values())


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert np.all(_perimeter(a) == bg), "The canvas perimeter identifies background"
    colors = list(map(int, np.unique(a)))
    assert len(colors) == 4, "Four distinct local color roles are required"
    interpretations = []
    for fg in colors:
        if fg == bg:
            continue
        pr0, pr1, pc0, pc1 = _bbox(a == fg)
        n = pr1 - pr0
        if not (7 <= n <= 26 and pc1 - pc0 == n):
            continue
        panel = a[pr0:pr1, pc0:pc1]
        if not np.all(_perimeter(panel) == fg):
            continue
        outside = np.ones(a.shape, dtype=bool)
        outside[pr0:pr1, pc0:pc1] = False
        for mask in colors:
            if mask in (bg, fg) or np.any((a == mask) & outside):
                continue
            r0, r1, c0, c1 = _bbox(panel == mask)
            if not (0 < r0 < r1 < n and 0 < c0 < c1 < n):
                continue
            if (r1-r0)*(c1-c0) < 2 or not np.all(panel[r0:r1, c0:c1] == mask):
                continue
            if np.count_nonzero(panel == mask) != (r1-r0)*(c1-c0):
                continue
            tally = next(c for c in colors if c not in (bg, fg, mask))
            if np.any(panel == tally) or np.any(outside & ~np.isin(a, (bg, tally))):
                continue
            dots = a == tally
            if np.any(dots[:-1] & dots[1:]) or np.any(dots[:, :-1] & dots[:, 1:]):
                continue
            # A background gap separates every dot from the object's border.
            near = np.zeros(a.shape, dtype=bool)
            near[max(0, pr0-1):min(a.shape[0], pr1+1),
                 max(0, pc0-1):min(a.shape[1], pc1+1)] = True
            if np.any(near & dots):
                continue
            interpretations.append({"panel": panel.copy(), "background": bg,
                "foreground": fg, "mask": mask, "tally_color": tally,
                "tally": int(dots.sum()), "panel_box": (pr0, pr1, pc0, pc1),
                "mask_box": (r0, r1, c0, c1), "shape": a.shape})
    assert len(interpretations) == 1, "Ambiguous object or color roles"
    return interpretations[0]


def _analyze(input_grid):
    record = _parse(input_grid)
    hypotheses = _hypotheses(record["panel"], record["background"],
                              record["foreground"], record["mask"])
    assert len(hypotheses) >= 2, "The tally must resolve a real mirror ambiguity"
    matched = [h for h in hypotheses if h["count"] == record["tally"]]
    assert len(matched) == 1, "Cavity checksum must select one answer"
    record["hypotheses"] = hypotheses
    record["selected"] = matched[0]
    return matched[0]["crop"].copy(), record


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the erased crop from pixels alone; does not modify the input."""
    return _analyze(input_grid)[0]


def _equivalence_labels(n, hidden):
    """Constraints of horizontal and diagonal mirrors on VISIBLE cells only.

    This constructs incomplete evidence, not a completed reflected template.
    The solver does not call this function or receive its labels.
    """
    parent = list(range(n * n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for code in (0, 2):
        rr, cc = _maps(n)[code]
        for r, c in np.argwhere(~hidden):
            s, d = int(rr[r, c]), int(cc[r, c])
            if not hidden[s, d]:
                x, y = find(int(r*n+c)), find(s*n+d)
                parent[x] = y
    labels = np.array([find(i) for i in range(n*n)]).reshape(n, n)
    protected = set(map(int, _perimeter(labels)))
    groups = [int(x) for x in np.unique(labels[~hidden]) if x not in protected]
    representatives = [tuple(map(int, np.argwhere(labels == x)[0])) for x in groups]
    return labels, groups, representatives


def _sample_panel(rng, n, hidden, labels, groups, reps):
    mode = int(rng.integers(4))
    chosen = []
    if mode <= 1:
        p = float(rng.uniform(.09, .68))
        chosen = [g for g in groups if rng.random() < p]
    elif mode == 2:
        scale = int(rng.integers(2, 5))
        coarse = rng.random(((n+scale-1)//scale, (n+scale-1)//scale))
        threshold = float(rng.uniform(.3, .75))
        chosen = [g for g, (r, c) in zip(groups, reps) if coarse[r//scale, c//scale] > threshold]
    else:
        # Connected patches and long chambers, alongside the independent-cell style.
        region = np.zeros((n, n), dtype=bool)
        for _ in range(int(rng.integers(1, 7))):
            r, c = map(int, rng.integers(1, n-1, size=2))
            h, w = map(int, rng.integers(1, max(3, n//2), size=2))
            region[r:min(n-1, r+h), c:min(n-1, c+w)] = True
        chosen = [g for g, q in zip(groups, reps) if region[q]]
    panel = np.ones((n, n), dtype=np.int8)
    panel[np.isin(labels, chosen)] = 0
    panel[hidden] = 2
    return panel


def _geometry(rng, n):
    mid = n//2
    h, w = map(int, rng.integers(2, mid, size=2))
    r = int(rng.integers(1, mid-h+1))
    # Overlap of folded row and column intervals admits genuinely competing
    # axial/diagonal completions. The shift frees aspect ratio and exact offset.
    low, high = max(mid, r+h), n-w-1
    c = int(np.clip(n-r-w + int(rng.integers(-2, 3)), low, high))
    hidden = np.zeros((n, n), dtype=bool)
    hidden[r:r+h, c:c+w] = True
    return hidden


def _place(rng, panel, checksum):
    n = len(panel)
    # Every side may hold tally dots. Unequal margins change canvas proportions.
    margins = [int(rng.integers(3, min(6, (30-n)//2)+1)) for _ in range(4)]
    top, bottom, left, right = margins
    a = np.zeros((n+top+bottom, n+left+right), dtype=np.int8)
    a[top:top+n, left:left+n] = panel
    allowed = np.zeros(a.shape, dtype=bool)
    allowed[1:-1, 1:-1] = True
    allowed[top-1:top+n+1, left-1:left+n+1] = False
    cells = np.argwhere(allowed)
    # A parity-independent random independent set, with a capacity fallback.
    for attempt in range(8):
        dots = np.zeros(a.shape, dtype=bool)
        positions = cells[rng.permutation(len(cells))]
        if attempt == 7:
            positions = positions[(positions[:, 0]+positions[:, 1]) % 2 == 0]
        for r, c in positions:
            r, c = int(r), int(c)
            if any(dots[r+dr, c+dc] for dr, dc in _FOUR):
                continue
            dots[r, c] = True
            if int(dots.sum()) == checksum:
                a[dots] = 3
                palette = rng.choice(10, 4, replace=False).astype(np.int8)
                return palette[a]
    raise AssertionError("Insufficient exterior tally space")


def _rotate_for_corner(a, corner):
    # Canonical erased rectangle is top-right. The order of corners is
    # top-right, top-left, bottom-left, bottom-right under counterclockwise turns.
    return np.rot90(a, corner).copy()


def _finish_pair(rng, panel, full, hidden, checksum, corner, seen):
    inp = _place(rng, _rotate_for_corner(panel, corner), checksum)
    full = _rotate_for_corner(full, corner)
    hidden = _rotate_for_corner(hidden, corner)
    r0, r1, c0, c1 = _bbox(hidden)
    # Label is the true completed object's target-oriented crop.
    rec = _parse(inp)
    lookup = np.array([rec["background"], rec["foreground"]], dtype=np.int8)
    pair = {"input": inp, "output": lookup[full[r0:r1, c0:c1]].copy()}
    if seen is not None and _key(inp) in seen:
        # Ninety distinct background/foreground assignments suffice against at
        # most thirteen earlier inputs. This is a finite uniqueness guarantee.
        source = [rec[k] for k in ("background", "foreground", "mask", "tally_color")]
        for bg in range(10):
            for fg in range(10):
                if bg == fg:
                    continue
                targets = [bg, fg] + [c for c in range(10) if c not in (bg, fg)][:2]
                perm = np.arange(10, dtype=np.int8)
                perm[source] = targets
                fresh = {k: perm[v] for k, v in pair.items()}
                if _key(fresh["input"]) not in seen:
                    return fresh
        raise AssertionError("Finite palette uniqueness bound violated")
    return pair


def _make_pair(rng, sizes, corner, axis_class=None, rank=None, strong=False, seen=None,
               allow_fallback=True, geometry_limit=36):
    for geometry_attempt in range(geometry_limit):
        n = int(rng.choice(sizes))
        hidden = _geometry(rng, n)
        labels, groups, reps = _equivalence_labels(n, hidden)
        for _ in range(100):
            panel = _sample_panel(rng, n, hidden, labels, groups, reps)
            hypotheses = _hypotheses(panel, 0, 1, 2)
            if len(hypotheses) < 2:
                continue
            counts = Counter(h["count"] for h in hypotheses)
            choices = [h for h in hypotheses if 1 <= h["count"] <= 16 and counts[h["count"]] == 1
                       and h["orientation_visible"] and min(h["sizes"], default=0) >= 1]
            if axis_class is not None:
                choices = [h for h in choices if any((code < 2) == (axis_class == "axial") for code in h["axes"])]
            if rank is not None:
                wanted = (min if rank == "min" else max)(h["count"] for h in hypotheses)
                choices = [h for h in choices if h["count"] == wanted]
            if strong:
                choices = [h for h in choices if h["eight_count"] != h["count"]
                           and h["area"] != h["count"] and max(h["sizes"]) >= 3
                           and h["joining"]]
            if not choices:
                continue
            chosen = choices[int(rng.integers(len(choices)))]
            return _finish_pair(rng, panel, chosen["full"], hidden, chosen["count"], corner, seen)
    if not allow_fallback:
        raise RuntimeError("Fixed core construction failed")
    # A reproducibly checked small core guarantees completion of a rare exhausted
    # random search. Centered foreground padding preserves all four mirrors,
    # cavity counts, joining witnesses, and the requested size/parity shift.
    # No grid bank is stored and prediction never sees this construction seed.
    parity = int(rng.choice(sizes)) % 2
    core_size = 9 if parity else 10
    core = _make_pair(np.random.default_rng(123), [core_size], 0, axis_class, rank,
                      strong, None, allow_fallback=False)
    rec = _analyze(core["input"])[1]
    panel = np.where(rec["panel"] == rec["background"], 0,
                     np.where(rec["panel"] == rec["foreground"], 1, 2)).astype(np.int8)
    full = np.where(rec["selected"]["full"] == rec["background"], 0, 1).astype(np.int8)
    n = int(rng.choice([x for x in sizes if x % 2 == parity and x >= core_size]))
    pad = (n-core_size)//2
    return _finish_pair(rng, np.pad(panel, pad, constant_values=1),
                        np.pad(full, pad, constant_values=1),
                        np.pad(panel == 2, pad), rec["tally"], corner, seen)


def _check_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} through {high}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    # Default_rng accepts nonnegative integers; signed integers have a fixed map.
    rng = np.random.default_rng(int(seed) % (1 << 64) if seed < 0 else int(seed))
    relationship = str(rng.choice(("held_corner", "growth")))
    held = int(rng.integers(4))
    rank_order = ("min", "max") if rng.random() < .5 else ("max", "min")
    classes = ("axial", "diagonal") if rng.random() < .5 else ("diagonal", "axial")
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(number):
            training = split == "train"
            corners = [x for x in range(4) if x != held] if training and relationship == "held_corner" else list(range(4))
            corner = int(rng.choice(corners)) if training or relationship == "growth" else held
            sizes = list(range(9, 17)) if training and relationship == "growth" else list(range(9, 22))
            if not training and relationship == "growth":
                sizes = list(range(17, 22))
            witness = training and i < 2
            # The joining witness uses the lower-count completion: restoring a
            # bridge merges visible chamber fragments. The other witness picks
            # the larger count, so "always merge" cannot explain the puzzle.
            strong = witness and rank_order[i] == "min"
            pair = _make_pair(rng, sizes, corner, classes[i] if witness else None,
                              rank_order[i] if witness else None, strong, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, checksum semantics, real ambiguity, and training evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            expected, record = _analyze(inp)
            assert np.array_equal(expected, out), "Incorrect mirror reconstruction or crop"
            assert record["mask"] not in out and record["tally_color"] not in out
            assert record["selected"]["orientation_visible"], "Unoriented copying happens to suffice"
            if split == "train":
                records.append(record)
    selected = [r["selected"] for r in records]
    assert any(any(c < 2 for c in h["axes"]) for h in selected), "No axial demonstration"
    assert any(any(c >= 2 for c in h["axes"]) for h in selected), "No diagonal demonstration"
    assert any(h["count"] == min(x["count"] for x in r["hypotheses"])
               for h, r in zip(selected, records)), "Always choosing more cavities is not refuted"
    assert any(h["count"] == max(x["count"] for x in r["hypotheses"])
               for h, r in zip(selected, records)), "Always choosing fewer cavities is not refuted"
    assert any(h["eight_count"] != h["count"] and h["area"] != h["count"]
               and max(h["sizes"]) >= 3 and h["joining"] for h in selected), \
        "Missing whole-object topology, area, or four/eight-connectivity witness"


def self_test():
    """Bounded 42-seed diversity audit, ablations, mutations and count checks."""
    shapes, masks, sizes, colors, axes, signatures, chamber_shapes = (set() for _ in range(7))
    largest = 0
    joining = diagonal_contact = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                predicted, rec = _analyze(pair["input"])
                assert np.array_equal(predicted, pair["output"])
                assert np.array_equal(before, pair["input"])
                sel = rec["selected"]
                # Without B there is a visibly consistent DIFFERENT answer.
                assert any(not np.array_equal(h["crop"], predicted) for h in rec["hypotheses"])
                # Without A, directly taking a donor's native orientation differs.
                assert sel["orientation_visible"]
                shapes.add(pair["input"].shape)
                masks.add((pair["output"].shape, rec["mask_box"]))
                sizes.add(len(rec["panel"]))
                colors.add((rec["background"], rec["foreground"], rec["mask"], rec["tally_color"]))
                axes.update(sel["axes"])
                joining += sel["joining"]
                diagonal_contact += sel["eight_count"] != sel["count"]
                for cells in _components(sel["full"] == rec["background"]):
                    largest = max(largest, len(cells))
                    r0, r1, c0, c1 = _bbox(_cells_mask(cells, sel["full"].shape))
                    shape = _cells_mask(cells, sel["full"].shape)[r0:r1, c0:c1]
                    variants = [np.rot90(shape, k) for k in range(4)] + [np.rot90(shape.T, k) for k in range(4)]
                    chamber_shapes.add(min((x.shape, x.tobytes()) for x in variants))
        if seed in (0, 13, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(signatures) == 42 and len(shapes) >= 30 and len(masks) >= 60
    assert len(sizes) >= 10 and len(colors) >= 100 and axes == {0, 1, 2, 3}
    assert largest >= 8 and len(chamber_shapes) >= 25
    assert joining >= 42 and diagonal_contact >= 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(600+nt*10+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(5, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted answer accepted")
    # Color bijections and D4 transforms preserve the rule, including dot roles.
    p = generate_puzzle(12, 2, 1)
    rng = np.random.default_rng(918)
    for s in p:
        for pair in p[s]:
            perm = rng.permutation(10).astype(np.int8)
            for k in pair:
                pair[k] = perm[np.rot90(pair[k], 1)]
    validate_generated_puzzle(p)
    nested = {s: [{k: v.tolist() for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(nested)
    assert np.array_equal(solve(nested["test"][0]["input"]), nested["test"][0]["output"])
    # Exercise the finite fallback directly, including both core parities,
    # both mirror classes and both witness count rankings.
    for cls in ("axial", "diagonal"):
        for rank in ("min", "max"):
            for n in (19, 20):
                pair = _make_pair(np.random.default_rng(817), [n], 3, cls, rank,
                                  rank == "min", geometry_limit=0)
                expected, rec = _analyze(pair["input"])
                assert len(rec["panel"]) == n and np.array_equal(expected, pair["output"])
                assert any((code < 2) == (cls == "axial") for code in rec["selected"]["axes"])
                if rank == "min":
                    assert rec["selected"]["joining"]
    # Change ONLY the tally, keeping the square and mask pixel-for-pixel equal.
    pair = generate_puzzle(0)["train"][0]
    answer, rec = _analyze(pair["input"])
    other = next(h for h in rec["hypotheses"] if not np.array_equal(h["crop"], answer))
    changed = _retally(pair["input"], rec, other["count"])
    assert np.array_equal(solve(changed), other["crop"])
    assert not np.array_equal(solve(changed), answer)
    missing = next(k for k in range(1, 20) if k not in {h["count"] for h in rec["hypotheses"]})
    invalid = _retally(pair["input"], rec, missing)
    try:
        solve(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("An unsupported cavity checksum was accepted")
    return {"seeds": 42, "puzzles": len(signatures), "input_shapes": len(shapes),
            "mask_geometries": len(masks), "panel_sizes": sorted(sizes),
            "palettes": len(colors), "mirror_directions": sorted(axes),
            "cavity_shapes_up_to_D4": len(chamber_shapes), "largest_cavity": largest,
            "joining_examples": joining, "four_eight_witnesses": diagonal_contact}


def _cells_mask(cells, shape):
    result = np.zeros(shape, dtype=bool)
    for q in cells:
        result[q] = True
    return result


def _retally(a, rec, number):
    """Counterfactual fixture helper; unused by generation and prediction."""
    changed = a.copy()
    changed[changed == rec["tally_color"]] = rec["background"]
    allowed = np.ones(a.shape, dtype=bool)
    allowed[[0, -1], :] = False
    allowed[:, [0, -1]] = False
    r0, r1, c0, c1 = rec["panel_box"]
    allowed[r0-1:r1+1, c0-1:c1+1] = False
    rr, cc = np.indices(a.shape)
    positions = np.argwhere(allowed & ((rr+cc) % 2 == 0))
    assert len(positions) >= number
    for r, c in positions[:number]:
        changed[r, c] = rec["tally_color"]
    return changed


if __name__ == "__main__":
    # Local audit output is compact and confined to scratch/.
    report = self_test()
    Path("scratch").mkdir(exist_ok=True)
    Path("scratch/self_test.json").write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report, separators=(",", ":")))
