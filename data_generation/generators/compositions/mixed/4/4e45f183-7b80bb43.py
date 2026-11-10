"""Repair-addressed panel permutation. Only NumPy and the standard library.

The decoder reads its argument only. Construction keeps its own clean panels,
then checks that the visible evidence recovers them before returning a pair.
No parent generator or saved puzzle is imported.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not ((0 <= a) & (a <= 9)).all()):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _strong(mask):
    """Membership in an intact horizontal or vertical run of length >= 3."""
    strong = np.zeros_like(mask)
    for m, s in ((mask, strong), (mask.T, strong.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            s[:, k:k + triples.shape[1]] |= triples
    return strong


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _evidence(mask):
    """Three-cell gap, retained diagonal anchor + two weak off-axis pixels."""
    strong = _strong(mask)
    result = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r - dr, c - dc) or _at(mask, r + 5*dr, c + 5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar + k*sr, ac + k*sc) for k in (1, 2))
                tr, tc = ar + 3*sr, ac + 3*sc
                if (all(_at(mask, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                        and (not _at(mask, tr, tc) or strong[tr, tc])):
                    matches.append((endpoint, side, spur))
        if matches:
            result.append(((r, c, dr, dc), matches))
    return result


def _repair(mask):
    evidence = _evidence(mask)
    if len(evidence) != 1 or len(evidence[0][1]) != 1:
        raise ValueError("a panel needs exactly one uniquely supported gap")
    (r, c, dr, dc), matches = evidence[0]
    added = np.zeros_like(mask)
    for k in (1, 2, 3):
        added[r + k*dr, c + k*dc] = True
    anchors = np.zeros_like(mask)
    anchors[r, c] = anchors[r + 4*dr, c + 4*dc] = True
    # Repaired orthogonal runs protect their cells. Weak diagonally linked cells
    # disappear, including spurs beside an orthogonal branch and free decoys.
    weak = mask & ~_strong(mask | added) & ~anchors
    erased = np.zeros_like(mask)
    for dc2 in (-1, 1):
        a = weak[:-1, :-1] if dc2 == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc2 == 1 else weak[1:, :-1]
        linked = a & b
        if dc2 == 1:
            erased[:-1, :-1] |= linked
            erased[1:, 1:] |= linked
        else:
            erased[:-1, 1:] |= linked
            erased[1:, :-1] |= linked
    clean = (mask & ~erased) | added
    junctions = []
    for rr, cc in np.argwhere(added):
        rr, cc = int(rr), int(cc)
        if any(all(_at(clean, rr + k*side*dc, cc + k*side*dr)
                       for k in (1, 2)) for side in (-1, 1)):
            junctions.append((rr, cc))
    if len(junctions) != 1:
        raise ValueError("exactly one repaired cell must join a perpendicular arm")
    # A spur is evidence, never retained cargo.
    if any(clean[rr, cc] for rr, cc in matches[0][2]):
        raise ValueError("spur collides with retained linework")
    return clean, junctions[0], evidence[0], added, erased


def _axis_sector(coordinate, size):
    # Classify cell centers into equal continuous thirds; exact boundaries go
    # to the following sector. Integer arithmetic avoids floating-point ties.
    return min(2, (3 * (2*coordinate + 1)) // (2*size))


def _address(junction, shape):
    return 3*_axis_sector(junction[0], shape[0]) + _axis_sector(junction[1], shape[1])


def _centroid_address(mask):
    points = np.argwhere(mask)
    return (3*min(2, int(3*(float(points[:, 0].mean()) + .5)/mask.shape[0]))
            + min(2, int(3*(float(points[:, 1].mean()) + .5)/mask.shape[1])))


def _false_locators(details):
    """Wrong destinations produced by closing unsupported gaps at side stumps."""
    result = []
    for d in details:
        for r, c, dr, dc in _gaps(d["clean"]):
            for k in (1, 2, 3):
                rr, cc = r + k*dr, c + k*dc
                if any(all(_at(d["clean"], rr + t*s*dc, cc + t*s*dr)
                           for t in (1, 2)) for s in (-1, 1)):
                    wrong = _address((rr, cc), d["clean"].shape)
                    if wrong != d["target"]:
                        result.append((d["source"], d["target"], wrong, (rr, cc)))
    return result


def _render(panels, separator):
    h, w = panels[0].shape
    grid = np.full((3*h + 2, 3*w + 2), separator, dtype=np.int8)
    for index, panel in enumerate(panels):
        r, c = divmod(index, 3)
        grid[r*(h + 1):r*(h + 1) + h, c*(w + 1):c*(w + 1) + w] = panel
    return grid


def _parse(grid):
    a = _grid(grid)
    h, w = (a.shape[0] - 2)//3, (a.shape[1] - 2)//3
    if not (7 <= h <= 9 and 7 <= w <= 9 and a.shape == (3*h + 2, 3*w + 2)):
        raise ValueError("expected nine equal 7..9-cell panels")
    separator = int(a[h, w])
    lattice = np.zeros_like(a, dtype=bool)
    lattice[[h, 2*h + 1], :] = True
    lattice[:, [w, 2*w + 1]] = True
    if not np.all(a[lattice] == separator) or np.any(a[~lattice] == separator):
        raise ValueError("invalid separator lattice")
    colors, counts = np.unique(a[~lattice], return_counts=True)
    fill = int(colors[np.argmax(counts)])
    if len(colors) < 2 or 2*int(max(counts)) <= int(sum(counts)):
        raise ValueError("panel fill must be the pooled strict majority")
    panels = []
    for index in range(9):
        r, c = divmod(index, 3)
        panel = a[r*(h + 1):r*(h + 1) + h, c*(w + 1):c*(w + 1) + w].copy()
        ink = np.unique(panel[panel != fill])
        if len(ink) != 1:
            raise ValueError("each complete panel motif has one ink color")
        panels.append((panel, int(ink[0])))
    return a, separator, fill, panels, lattice


def _decode(input_grid):
    a, separator, fill, panels, lattice = _parse(input_grid)
    destinations = [None]*9
    details = []
    for source, (panel, ink) in enumerate(panels):
        mask = panel != fill
        clean, junction, evidence, added, erased = _repair(mask)
        target = _address(junction, panel.shape)
        if destinations[target] is not None:
            raise ValueError("duplicate local address")
        cleaned_panel = np.where(clean, ink, fill).astype(np.int8)
        destinations[target] = cleaned_panel
        details.append({"source": source, "target": target, "junction": junction,
                        "evidence": evidence, "added": added, "erased": erased,
                        "clean": clean, "raw": mask, "panel": cleaned_panel,
                        "ink": ink, "unsupported": len(_gaps(clean))})
    return _render(destinations, separator), details, (a, separator, fill, lattice)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair and erase within panels, then translate each whole cleaned panel."""
    return _decode(input_grid)[0]


def _core(rng, h, w, label):
    """Construct ground truth first, independently of the input decoder."""
    rows = [r for r in range(1, h - 1) if _axis_sector(r, h) == label//3]
    cols = [c for c in range(1, w - 1) if _axis_sector(c, w) == label % 3]
    for _ in range(192):
        jr, jc = int(rng.choice(rows)), int(rng.choice(cols))
        vertical = bool(rng.integers(2))
        n, p, q = (h, jr, jc) if vertical else (w, jc, jr)
        perpendicular = w if vertical else h
        start = p - int(rng.integers(1, 4))
        if start < 0 or start + 4 >= n:
            continue
        lo, hi = int(rng.integers(start + 1)), int(rng.integers(start + 4, n))
        if lo == start and hi == start + 4:
            continue
        clean = np.zeros((h, w), dtype=bool)

        # Use direct indexing to keep horizontal/vertical coordinates explicit.
        if vertical:
            clean[lo:hi + 1, jc] = True
        else:
            clean[jr, lo:hi + 1] = True
        sides = [s for s in (-1, 1) if (q if s == -1 else perpendicular - 1 - q) >= 2]
        if not sides:
            continue
        side = int(rng.choice(sides))
        chosen = sides if len(sides) == 2 and rng.random() < .25 else [side]
        for s in chosen:
            room = q if s == -1 else perpendicular - 1 - q
            length = int(rng.integers(2, room + 1))
            for k in range(1, length + 1):
                if vertical:
                    clean[jr, jc + k*s] = True
                else:
                    clean[jr + k*s, jc] = True
        # Extra intact branches are payload, not additional address markers.
        if rng.random() < .40:
            t = int(rng.choice([lo, hi]))
            s = int(rng.choice(sides))
            room = q if s == -1 else perpendicular - 1 - q
            for k in range(1, int(rng.integers(2, room + 1)) + 1):
                if vertical:
                    clean[t, jc + k*s] = True
                else:
                    clean[jr + k*s, t] = True
        dr, dc = (1, 0) if vertical else (0, 1)
        r, c = (start, jc) if vertical else (jr, start)
        inp = clean.copy()
        for k in (1, 2, 3):
            inp[r + k*dr, c + k*dc] = False
        choices = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for s in (-1, 1):
                spur = [(ar + k*(toward*dr + s*dc), ac + k*(toward*dc + s*dr)) for k in (1, 2)]
                if all(0 <= rr < h and 0 <= cc < w and not clean[rr, cc] for rr, cc in spur):
                    choices.append(spur)
        if not choices:
            continue
        for rr, cc in choices[int(rng.integers(len(choices)))]:
            inp[rr, cc] = True
        try:
            decoded, j, _, _, _ = _repair(inp)
        except ValueError:
            continue
        if j == (jr, jc) and np.array_equal(decoded, clean):
            return inp, clean, j
    # A finite, simple construction covers every sector at every allowed size.
    # This fallback varies neither the rule nor the required evidence.
    for jr in rows:
        for jc in cols:
            for dr, dc in ((0, 1), (1, 0)):
                n, p = (w, jc) if dc else (h, jr)
                for offset in (1, 3, 2):
                    start = p - offset
                    if start < 0 or start + 4 >= n or n == 5:
                        continue
                    for branch_side in (-1, 1):
                        clean = np.zeros((h, w), dtype=bool)
                        if dc:
                            clean[jr, :] = True
                        else:
                            clean[:, jc] = True
                        branch = [(jr + k*branch_side*dc, jc + k*branch_side*dr) for k in (1, 2, 3)]
                        if not all(0 <= r < h and 0 <= c < w for r, c in branch):
                            continue
                        for cell in branch:
                            clean[cell] = True
                        r, c = (jr, start) if dc else (start, jc)
                        for endpoint in (0, 4):
                            for side in (-1, 1):
                                toward = 1 if endpoint == 0 else -1
                                spur = [(r + endpoint*dr + k*(toward*dr + side*dc),
                                         c + endpoint*dc + k*(toward*dc + side*dr)) for k in (1, 2)]
                                if not all(0 <= rr < h and 0 <= cc < w and not clean[rr, cc] for rr, cc in spur):
                                    continue
                                inp = clean.copy()
                                for k in (1, 2, 3):
                                    inp[r + k*dr, c + k*dc] = False
                                for cell in spur:
                                    inp[cell] = True
                                try:
                                    rebuilt, j, _, _, _ = _repair(inp)
                                except ValueError:
                                    continue
                                if j == (jr, jc) and np.array_equal(rebuilt, clean):
                                    return inp, clean, j
    raise RuntimeError(f"no core at {h}/{w}/{label}")


def _augment(rng, inp, clean, junction, kind, attempts=18):
    """Try one compatible payload or decoy; no rejected arrays are retained."""
    h, w = inp.shape
    for _ in range(attempts):
        piece = np.zeros_like(clean)
        holes = []
        if kind == "gap":
            vertical = bool(rng.integers(2))
            n, p = (h, w) if vertical else (w, h)
            start, fixed = int(rng.integers(n - 4)), int(rng.integers(p))
            lo, hi = int(rng.integers(start + 1)), int(rng.integers(start + 4, n))
            for t in range(lo, hi + 1):
                r, c = (t, fixed) if vertical else (fixed, t)
                if start < t < start + 4:
                    holes.append((r, c))
                else:
                    piece[r, c] = True
            # A disconnected perpendicular stump makes filling every gap a
            # plausible *wrong* locator, instead of just an extra straight line.
            t = start + int(rng.integers(1, 4))
            s = int(rng.choice([-1, 1]))
            if not 0 <= fixed + 2*s < p:
                continue
            for k in (1, 2):
                r, c = (t, fixed + k*s) if vertical else (fixed + k*s, t)
                piece[r, c] = True
        elif kind == "cargo":
            r, c = int(rng.integers(h - 2)), int(rng.integers(w - 2))
            form = int(rng.integers(4))
            bh = int(rng.integers(3, min(5, h - r) + 1))
            bw = int(rng.integers(3, min(5, w - c) + 1))
            if form == 0:
                piece[r, c:c + bw] = True
            elif form == 1:
                piece[r:r + bh, c] = piece[r, c:c + bw] = True
            elif form == 2:
                piece[r:r + bh, c:c + bw] = True
                piece[r + 1:r + bh - 1, c + 1:c + bw - 1] = False
            else:
                piece[r:r + 2, c:c + bw] = True
        else:
            length = int(rng.choice([2, 3]))
            r, c = int(rng.integers(h)), int(rng.integers(w))
            dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
            cells = [(r + k*dr, c + k*dc) for k in range(length)]
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                continue
            for rr, cc in cells:
                piece[rr, cc] = True
        if (piece & (inp | clean)).any() or any(inp[x] or clean[x] for x in holes):
            continue
        # Never overwrite the genuine missing segment, even with retained cargo.
        original_gap = clean & ~inp
        if (piece & original_gap).any():
            continue
        trial = inp | piece
        target = clean if kind == "decoy" else clean | piece
        if int(trial.sum()) >= .46*trial.size or int(target.sum()) >= .46*target.size:
            continue
        try:
            rebuilt, j, _, _, _ = _repair(trial)
        except ValueError:
            continue
        if j == junction and np.array_equal(rebuilt, target):
            return trial, target
    return inp, clean


def _permutation(rng, index, test):
    order = np.arange(9)
    if index == 1 and not test:
        chosen = rng.choice(9, size=int(rng.choice([2, 3, 4])), replace=False)
        order[chosen] = np.roll(order[chosen], 1)
    else:
        order = rng.permutation(9)
        if np.array_equal(order, np.arange(9)):
            order = np.roll(order, 1)
    return order


def _make_pair(rng, shape, index, test, palette):
    h, w = shape
    separator, fill, primary = palette
    extra_inks = [c for c in range(10) if c not in palette]
    inks = [primary]*9
    if index > 0 or test:
        if rng.random() < .65:
            for label in rng.choice(9, size=int(rng.integers(2, 6)), replace=False):
                inks[int(label)] = int(rng.choice(extra_inks))
    for _ in range(24):
        raw, target = [], []
        for label in range(9):
            inp, clean, j = _core(rng, h, w, label)
            inp, clean = _augment(rng, inp, clean, j, "gap", attempts=24)
            if index > 0 or test or rng.random() < .30:
                inp, clean = _augment(rng, inp, clean, j, "cargo")
            if index > 0 or test:
                inp, clean = _augment(rng, inp, clean, j, "decoy")
            raw.append(np.where(inp, inks[label], fill).astype(np.int8))
            target.append(np.where(clean, inks[label], fill).astype(np.int8))
        order = _permutation(rng, index, test)
        pair = {"input": _render([raw[int(i)] for i in order], separator),
                "output": _render(target, separator)}
        predicted, details, _ = _decode(pair["input"])
        if (np.array_equal(predicted, pair["output"])
                and {bool(d["evidence"][0][2]) for d in details} == {False, True}
                and sum(d["unsupported"] for d in details) >= 1
                and _false_locators(details)
                and sum(_centroid_address(d["raw"]) != d["target"] for d in details) >= 2
                and sum(_centroid_address(d["clean"]) != d["target"] for d in details) >= 2):
            return pair
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    n = _integer(num_train, "num_train", 2, 10)
    t = _integer(num_test, "num_test", 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shapes = [(h, w) for h in range(7, 10) for w in range(7, 10)]
    held_out = shapes[int(rng.integers(len(shapes)))]
    train_shapes = [s for s in shapes if s != held_out]
    rng.shuffle(train_shapes)
    base_palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
    simple_palette = bool(rng.random() < .30)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            shape = held_out if split == "test" else train_shapes[index % len(train_shapes)]
            if simple_palette:
                palette = base_palette
            elif split == "train" and index == 1:
                palette = base_palette[1:] + base_palette[:1]
            else:
                palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
            for _ in range(8):
                pair = _make_pair(rng, shape, index, split == "test", palette)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate avoidance exhausted")
    return puzzle


def _inspect_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise AssertionError("pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    expected, details, (_, separator, fill, lattice) = _decode(inp)
    assert np.array_equal(expected, out), "incorrect repair, erasure, cargo or permutation"
    assert np.all(out[lattice] == separator)
    assert not np.array_equal(inp, out)
    restored = [d["panel"] for d in details]
    in_place = _render(restored, separator)
    assert not np.array_equal(in_place, out), "A ablation: panels must move"
    # Removing B leaves the raw payloads rather than a repaired address decoder.
    _, _, _, original, _ = _parse(inp)
    raw_sorted = [None]*9
    for d, (panel, _) in zip(details, original):
        raw_sorted[d["target"]] = panel
    assert not np.array_equal(_render(raw_sorted, separator), out), "B ablation"
    assert all(int(d["added"].sum()) == 3 for d in details)
    assert all(int(d["erased"].sum()) >= 2 for d in details)
    assert all(not d["raw"][d["junction"]] and d["clean"][d["junction"]] for d in details)
    assert {d["target"] for d in details} == set(range(9))
    raw_bad = sum(_centroid_address(d["raw"]) != d["target"] for d in details)
    clean_bad = sum(_centroid_address(d["clean"]) != d["target"] for d in details)
    assert raw_bad >= 2 and clean_bad >= 2, "centroid shortcut not distinguished"
    assert sum(d["unsupported"] for d in details) >= 1, "no unsupported-gap contrast"
    assert _false_locators(details), "wrong repair must imply a wrong destination"
    # Color accounting happens after cleanup, since repair changes ink counts.
    expected_counts = Counter({separator: int(lattice.sum())})
    for d in details:
        expected_counts.update({d["ink"]: int(d["clean"].sum()),
                                fill: int(d["clean"].size - d["clean"].sum())})
    assert Counter(map(int, out.flat)) == expected_counts
    return details


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, t = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, sizes = set(), {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            details = _inspect_pair(pair)
            a = _grid(pair["input"])
            key = (a.shape, a.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            sizes[split].add(a.shape)
            if split == "train":
                assert {bool(d["evidence"][0][2]) for d in details} == {False, True}, "demonstrate both axes"
    assert sizes["train"].isdisjoint(sizes["test"]), "missing geometry shift"
    assert len(sizes["train"]) >= 2


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Streaming 42-seed invariants, diversity and deliberate wrong-label tests."""
    signatures, shapes, layouts, colors, structures = set(), set(), set(), set(), set()
    line_counts, erased_counts, junction_offsets, node_degrees = set(), set(), set(), set()
    pairs = centroid_failures = unsupported = fixed = extra_decoys = false_locators = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                predicted, details, _ = _decode(pair["input"])
                assert np.array_equal(predicted, pair["output"])
                shapes.add(pair["input"].shape)
                colors.update(map(int, np.unique(pair["input"])))
                layouts.add(tuple(d["target"] for d in details))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
                structures.add(hashlib.sha256(bytes(pair["input"].shape) + normalized).hexdigest())
                pairs += 1
                false_locators += len(_false_locators(details))
                for d in details:
                    line_counts.add(int(d["clean"].sum()))
                    erased_counts.add(int(d["erased"].sum()))
                    centroid_failures += _centroid_address(d["clean"]) != d["target"]
                    unsupported += d["unsupported"]
                    fixed += d["source"] == d["target"]
                    extra_decoys += int(d["erased"].sum()) > 2
                    r, c, dr, dc = d["evidence"][0]
                    jr, jc = d["junction"]
                    junction_offsets.add((jr - r)*dr + (jc - c)*dc)
                    node_degrees.add(sum(_at(d["clean"], jr + rr, jc + cc)
                                         for rr, cc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
            pair = puzzle["train"][0]
            before = pair["input"].copy()
            assert np.array_equal(solve(before.tolist()), pair["output"])
            assert np.array_equal(before, pair["input"]), "solver mutated input"
            for turns in range(4):
                for flip in (False, True):
                    im, om = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                    if flip:
                        im, om = im[:, ::-1], om[:, ::-1]
                    assert np.array_equal(solve(im), om), "geometric equivariance"
            color_map = np.array([3, 8, 0, 7, 1, 9, 4, 2, 6, 5], dtype=np.int8)
            assert np.array_equal(solve(color_map[pair["input"]]), color_map[pair["output"]])
            # Full-output validation catches an omitted repair, lost cargo,
            # wrong destination, and a retained diagonal pixel alike.
            out, details, (_, separator, fill, _) = _decode(pair["input"])
            d = details[0]
            ph, pw = d["clean"].shape
            r0, c0 = divmod(d["target"], 3)
            rr, cc = r0*(ph + 1), c0*(pw + 1)
            jr, jc = d["junction"]
            mutations = []
            bad = out.copy()
            bad[rr + jr, cc + jc] = fill
            mutations.append(bad)
            bad = out.copy()
            retained = d["clean"] & ~d["added"]
            cr, cc2 = map(int, np.argwhere(retained)[0])
            bad[rr + cr, cc + cc2] = fill
            mutations.append(bad)
            bad = out.copy()
            er, ec = map(int, np.argwhere(d["erased"])[0])
            bad[rr + er, cc + ec] = d["ink"]
            mutations.append(bad)
            mutations.append(_render([x["panel"] for x in details], separator))
            for bad in mutations:
                try:
                    _inspect_pair({"input": pair["input"], "output": bad})
                except AssertionError:
                    pass
                else:
                    raise AssertionError("bad label accepted")
    assert len(signatures) == 42 and len(structures) == pairs
    assert len(shapes) == 9 and len(colors) == 10 and len(layouts) >= 100
    assert len(line_counts) >= 15 and len(erased_counts) >= 3
    assert junction_offsets == {1, 2, 3} and node_degrees == {3, 4}
    assert centroid_failures >= 2*pairs and unsupported >= pairs and fixed > 100
    assert extra_decoys >= 42 and false_locators >= pairs
    for n, t in ((2, 1), (10, 4), (10, 1), (5, 4)):
        validate_generated_puzzle(generate_puzzle(9173, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "grid_shapes": sorted(shapes), "normalized_inputs": len(structures),
            "permutations": len(layouts), "clean_ink_counts": sorted(line_counts),
            "erasure_counts": sorted(erased_counts), "junction_offsets": sorted(junction_offsets),
            "node_degrees": sorted(node_degrees), "centroid_disagreements": centroid_failures,
            "unsupported_gaps": unsupported, "fixed_panels": fixed,
            "panels_with_extra_decoys": extra_decoys, "wrong_gap_locators": false_locators}
