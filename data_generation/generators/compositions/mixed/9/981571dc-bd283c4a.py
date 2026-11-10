"""Recover a mirror field's inventory, then pack its frequency-ranked colors.

Zero always means erasure. No parent module, file, seed, or demonstration is
consulted by solve. Generation and checking retain no execution traces or caches.
"""
from hashlib import sha256

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _orbits(shape, axes):
    h, w = shape
    a, b = axes
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = {r, a - r} if 0 <= a - r < h else {r}
            cols = {c, b - c} if 0 <= b - c < w else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            yield orbit


def _axis_candidates(grid, dimension):
    """All nontrivial folds, including extreme offsets; no center prior."""
    field = grid if dimension == 0 else grid.T
    n = len(field)
    candidates = []
    for axis in range(1, 2 * n - 2):
        left = np.arange(max(0, axis - n + 1), (axis + 1) // 2)
        first, second = field[left], field[axis - left]
        known = (first != 0) & (second != 0)
        if known.any() and np.all(first[known] == second[known]):
            candidates.append(axis)
    return candidates


def _axes(grid):
    rows, cols = (_axis_candidates(grid, d) for d in (0, 1))
    if len(rows) != 1 or len(cols) != 1:
        raise ValueError("surviving cells must identify exactly one fold per direction")
    return rows[0], cols[0]


def _restore(grid):
    axes = _axes(grid)
    restored = grid.copy()
    for orbit in _orbits(grid.shape, axes):
        values = {int(grid[p]) for p in orbit if grid[p] != 0}
        if len(values) != 1:
            raise ValueError("each reflection orbit needs unanimous nonzero witnesses")
        color = next(iter(values))
        for p in orbit:
            restored[p] = color
    return restored, axes


def _hist(grid):
    return np.bincount(grid.ravel(), minlength=10)


def _rank(grid, visible_only=False):
    counts = _hist(grid)
    colors = [c for c in range(1 if visible_only else 0, 10) if counts[c]]
    return tuple(sorted(colors, key=lambda c: (-int(counts[c]), c)))


def _pack(restored):
    counts = _hist(restored)
    colors = _rank(restored)
    if len({int(counts[c]) for c in colors}) != len(colors):
        raise ValueError("tied repaired frequencies are outside this family")
    stream = np.repeat(np.array(colors, dtype=np.int8), [counts[c] for c in colors])
    h, w = restored.shape
    return stream.reshape(w, h)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repair black gaps by two inferred folds, then sort the repaired inventory."""
    grid = _grid(input_grid)
    restored, _ = _restore(grid)
    return _pack(restored)


def _count(name, value, lower, upper):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _geometry(rng, test):
    # Size extrapolation is the deliberate train/test shift, not a solving cue.
    lo, hi = (19, 31) if test else (8, 19)
    h, w = (int(x) for x in rng.integers(lo, hi, size=2))
    if rng.random() < .2:
        w = h
    offsets = []
    for size in (h, w):
        limit = min(3, size // 4)
        offsets.append(0 if rng.random() < .35 else int(rng.integers(-limit, limit + 1)))
    return (h, w), (h - 1 + offsets[0], w - 1 + offsets[1])


def _draw_clean(rng, shape, axes, colors, style):
    orbits = list(_orbits(shape, axes))
    n, k = len(orbits), len(colors)
    rc = np.array([o[0] for o in orbits])
    r, c = rc.T
    weights = rng.dirichlet(np.full(k, float(rng.uniform(1.5, 9.))))
    if rng.random() < .18:
        weights[int(rng.integers(k))] *= 4
        weights /= weights.sum()
    amounts = rng.multinomial(n - k, weights) + 1
    if style == 0:
        key = rng.random(n)
    elif style == 1:
        bh, bw = (int(x) for x in rng.integers(2, 5, size=2))
        patches = rng.random(shape)
        key = patches[r // bh, c // bw] + rng.random(n) * .12
    elif style == 2:
        key = r + rng.uniform(.6, 2.) * c + rng.normal(0, .7, n)
    elif style == 3:
        key = np.maximum(abs(r - rng.uniform(0, shape[0])),
                         abs(c - rng.uniform(0, shape[1]))) + rng.random(n) * .8
    else:
        key = r // 2 + c // 3 + rng.normal(0, .8, n)
    labels = np.empty(n, dtype=np.int8)
    labels[np.argsort(key, kind="stable")] = np.repeat(colors, amounts)
    # Speckles are payload. They also break accidental additional folds in motifs.
    for _ in range(int(n * rng.uniform(.06, .20))):
        i, j = rng.integers(n, size=2)
        labels[i], labels[j] = labels[j], labels[i]
    clean = np.empty(shape, dtype=np.int8)
    for color, orbit in zip(labels, orbits):
        for p in orbit:
            clean[p] = color
    return clean


def _raw_mask(rng, shape, test):
    h, w = shape
    mode = int(rng.integers(5))
    mask = np.zeros(shape, dtype=bool)
    if mode == 0:
        mask = rng.random(shape) < rng.uniform(.06, .18 if not test else .24)
    else:
        for _ in range(int(rng.integers(1, 5))):
            bh = int(rng.integers(1, max(3, h // 3) + 1))
            bw = int(rng.integers(1, max(3, w // 3) + 1))
            if mode == 2:
                bh = 1
            elif mode == 3:
                bw = 1
            r = int(rng.integers(h - bh + 1))
            c = int(rng.integers(w - bw + 1))
            if mode == 4:
                for dr in range(bh):
                    mask[r + dr, c:c + 1 + dr * (bw - 1) // max(1, bh - 1)] = True
            else:
                mask[r:r + bh, c:c + bw] = True
    return mask


def _make_input(rng, clean, axes, test, inversion, hard):
    orbits = list(_orbits(clean.shape, axes))
    counts = _hist(clean)
    ranks = _rank(clean)
    # An adjacent pair with a small count gap is economical to reverse visibly.
    high, low = min(zip(ranks, ranks[1:]), key=lambda ab: counts[ab[0]] - counts[ab[1]])
    target_four = [o for o in orbits if len(o) == 4 and (not inversion or clean[o[0]] == high)]
    if hard and not target_four:
        return None
    mask = _raw_mask(rng, clean.shape, test)
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    if hard:
        orbit = target_four[int(rng.integers(len(target_four)))]
        survivor = int(rng.integers(4))
        for i, p in enumerate(orbit):
            mask[p] = i != survivor
    if inversion:
        mask[clean == low] = False
        need = int(counts[high] - counts[low] + 1)
        available = [o for o in orbits if clean[o[0]] == high]
        for oi in rng.permutation(len(available)):
            orbit = available[int(oi)]
            for pi in rng.permutation(len(orbit)):
                if int(np.count_nonzero(mask & (clean == high))) >= need:
                    break
                p = orbit[int(pi)]
                if not mask[p] and sum(not mask[q] for q in orbit) > 1:
                    mask[p] = True
        if int(np.count_nonzero(mask & (clean == high))) < need:
            return None
    if not mask.any() or mask.mean() > .42:
        return None
    inp = clean.copy()
    inp[mask] = 0
    try:
        recovered, inferred = _restore(inp)
    except ValueError:
        return None
    if inferred != axes or not np.array_equal(recovered, clean):
        return None
    if inversion and not _has_inversion(inp, clean):
        return None
    return inp


def _has_inversion(inp, restored):
    visible, full = _hist(inp), _hist(restored)
    colors = [c for c in range(1, 10) if full[c]]
    return any(full[a] > full[b] and visible[a] < visible[b]
               for a in colors for b in colors)


def _diagonal_gaps(inp, axes):
    h, w = inp.shape
    a, b = axes
    total = 0
    for r, c in np.argwhere(inp == 0):
        rr, cc = a - int(r), b - int(c)
        if (0 <= rr < h and 0 <= cc < w and rr != r and cc != c
                and inp[rr, c] == inp[r, cc] == 0 and inp[rr, cc] != 0):
            total += 1
    return total


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _new_pair(rng, test, inversion=False, hard=False, forbidden_top=None, clean_from=None):
    for attempt in range(160):
        if clean_from is None:
            shape, axes = _geometry(rng, test)
            k = int(rng.integers(3, 7 if min(shape) < 19 else 8))
            colors = rng.choice(np.arange(1, 10, dtype=np.int8), size=k, replace=False)
            clean = _draw_clean(rng, shape, axes, colors, int(rng.integers(5)) if attempt < 50 else 0)
            counts = _hist(clean)[np.unique(clean)]
            if len(set(counts.tolist())) != len(counts):
                continue
            if forbidden_top is not None and _rank(clean)[0] == forbidden_top:
                a, b = forbidden_top, int(rng.choice([c for c in colors if c != forbidden_top]))
                old = clean.copy()
                clean[old == a], clean[old == b] = b, a
            if _axis_candidates(clean, 0) != [axes[0]] or _axis_candidates(clean, 1) != [axes[1]]:
                continue
            output = _pack(clean)
            # Every training example reveals an internal partial-column boundary.
            if not test:
                ranked_counts = [_hist(clean)[c] for c in _rank(clean)]
                if not any(2 <= int(n) % shape[0] < shape[0] for n in np.cumsum(ranked_counts)[:-1]):
                    continue
        else:
            clean, axes = clean_from
            output = _pack(clean)
        for _ in range(12):
            inp = _make_input(rng, clean, axes, test, inversion, hard)
            if inp is not None and not np.array_equal(output, clean):
                return {"input": inp, "output": output}, clean, axes
    raise RuntimeError("bounded construction exhausted; no ambiguous fallback is emitted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_clean = first_axes = first_top = None
    contrast = nt >= 3 and rng.random() < .55
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            test = split == "test"
            equal_inventory = not test and index == 2 and contrast
            for _ in range(20):
                pair, clean, axes = _new_pair(
                    rng, test, inversion=test or index == 0,
                    hard=test or index == 0 or rng.random() < .3,
                    forbidden_top=first_top if not test and index == 1 else None,
                    clean_from=(first_clean, first_axes) if equal_inventory else None)
                sig = _signature(pair["input"])
                if sig not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            if first_clean is None:
                first_clean, first_axes, first_top = clean.copy(), axes, _rank(clean)[0]
            seen.add(sig)
            puzzle[split].append(pair)
    # The evidence-bearing demonstration has no fixed displayed index.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _traversals(restored):
    """Eight basic raster alternatives, used only to check demonstration evidence."""
    colors = _rank(restored)
    counts = _hist(restored)
    stream = [c for c in colors for _ in range(int(counts[c]))]
    h, w = restored.shape
    for major in (0, 1):
        for reverse_major in (False, True):
            for reverse_minor in (False, True):
                ys = list(range(h))
                xs = list(range(w))
                if major == 0:
                    if reverse_major:
                        xs.reverse()
                    if reverse_minor:
                        ys.reverse()
                    positions = [(r, c) for c in xs for r in ys]
                else:
                    if reverse_major:
                        ys.reverse()
                    if reverse_minor:
                        xs.reverse()
                    positions = [(r, c) for r in ys for c in xs]
                result = np.empty_like(restored)
                for p, color in zip(positions, stream):
                    result[p] = color
                yield (major, reverse_major, reverse_minor), result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, candidate_traversals = set(), None
    train_inversion = train_diagonal = False
    tops, train_sizes, test_sizes = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape and (inp == 0).any() and not (out == 0).any()
            sig = _signature(inp)
            assert sig not in seen, "repeated input"
            seen.add(sig)
            restored, axes = _restore(inp)
            assert np.array_equal(restored[inp != 0], inp[inp != 0])
            assert set(np.unique(inp)) - {0} == set(np.unique(out))
            assert 3 <= len(np.unique(restored)) <= 7
            assert np.array_equal(_hist(restored), _hist(out)), "repaired inventory lost"
            assert not np.array_equal(restored, out), "packing concept has no effect"
            assert np.array_equal(_pack(restored), out), "wrong sorted raster"
            counts = _hist(restored)
            assert len(set(int(counts[c]) for c in np.unique(restored))) == len(np.unique(restored))
            # Direct input counting includes zero, hence cannot produce this answer.
            assert not np.array_equal(_hist(inp), _hist(out))
            inversion = _has_inversion(inp, restored)
            diagonal = _diagonal_gaps(inp, axes) > 0
            if split == "train":
                train_inversion |= inversion
                train_diagonal |= diagonal
                tops.add(_rank(restored)[0])
                train_sizes.append(max(inp.shape))
                compatible = {key for key, candidate in _traversals(restored)
                              if np.array_equal(candidate, out)}
                candidate_traversals = compatible if candidate_traversals is None else candidate_traversals & compatible
            else:
                assert inversion and diagonal, "test must exercise both inherited mechanisms"
                test_sizes.append(min(inp.shape))
    assert train_inversion and train_diagonal
    assert len(tops) >= 2, "training must change the literal dominant color"
    assert candidate_traversals == {(0, False, True)}, "packing direction underdetermined"
    assert min(test_sizes) > max(train_sizes), "intended size extrapolation missing"


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = np.asarray(pair[key], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed study; at most one puzzle plus scalar statistics retained."""
    shapes, parities, offsets, palettes, dominant, signatures = (set() for _ in range(6))
    erased_sizes, boundaries, masks, clean_structures = set(), set(), set(), set()
    equal_inventory = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        train_outputs = [_signature(p["output"]) for p in puzzle["train"]]
        equal_inventory += len(set(train_outputs)) < len(train_outputs)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                restored, axes = _restore(inp)
                h, w = inp.shape
                # Check each gap's original three lookups, without iterative filling.
                for r, c in np.argwhere(inp == 0):
                    points = ((axes[0] - r, c), (r, axes[1] - c),
                              (axes[0] - r, axes[1] - c))
                    witnesses = {int(inp[y, x]) for y, x in points
                                 if 0 <= y < h and 0 <= x < w and inp[y, x] != 0}
                    assert witnesses == {int(restored[r, c])}
                shapes.add(inp.shape)
                parities.add(tuple(x % 2 for x in axes))
                offsets.add(tuple(int(np.sign(a - d + 1)) for a, d in zip(axes, inp.shape)))
                palettes.add(len(np.unique(out)))
                dominant.add(_rank(restored)[0])
                erased_sizes.add(int((inp == 0).sum()))
                boundaries.update(int(n) % h for n in np.cumsum([_hist(restored)[c] for c in _rank(restored)])[:-1])
                mask = inp == 0
                masks.add((bool(mask[0].any()), bool(mask[-1].any()), bool(mask[:, 0].any()), bool(mask[:, -1].any())))
                # Hash normalized structures; do not retain reconstructed grids.
                mapping = {c: i for i, c in enumerate(dict.fromkeys(restored.ravel().tolist()))}
                normalized = bytes(mapping[c] for c in restored.ravel().tolist())
                clean_structures.add((inp.shape, sha256(normalized).digest()))
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(clean_structures) >= 150
    assert parities == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(offsets) >= 7 and len(palettes) >= 4 and len(dominant) >= 8
    assert len(erased_sizes) >= 40 and len(boundaries) >= 12 and len(masks) >= 10
    assert equal_inventory >= 5
    assert _puzzle_digest(generate_puzzle(123, 7, 3)) == _puzzle_digest(generate_puzzle(123, 7, 3))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4., 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and every legal explicit count)")
