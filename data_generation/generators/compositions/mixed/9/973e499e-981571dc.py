"""Repair perpendicular reflection orbits, then expand complete color supports.

Zero is the fixed erasure marker in inputs and the mismatch blank in outputs.
No parent generator, file, random state, or demonstration enters solve().
"""

from functools import lru_cache

import numpy as np


_SELF_TEST_SEEDS = (
    0, 259883205, 548090802, 397527792, 707073766, 801147464,
    707882604, 81529349, 282886044, 54114400, 356786048, 750302456,
    316064493, 950266905, 284801040, 645576977, 337796145, 989722362,
    328966470, 152750593, 124997730, 70676777, 337531894, 918625301,
    632552546, 204528712, 525050602, 523222022, 660140174, 838526367,
    4458641, 643541471, 10685252, 979499475, 35583660, 694987797,
    866722300, 993687156, 229533475, 656542223, 828987030, 771827007,
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _counts(seed, num_train, num_test):
    # Explicit defaults are 4/1; None also supports the inspection tool's API.
    tr = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    te = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if tr is None and te is None:
        te = 1 + seed % 4
        tr = te + 1 + (seed // 4) % (10 - te)
    elif tr is None:
        tr = te + 1 + seed % (10 - te)
    elif te is None:
        te = 1 + seed % min(4, tr - 1)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= d <= 30 for d in g.shape)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 2D ARC integer grid")
    return g.astype(np.int8, copy=True)


@lru_cache(maxsize=96)
def _orbits(n, a, b):
    """Fold axes are at a/2 and b/2 in zero-based cell coordinates."""
    seen, result = set(), []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            rr, cc = a - r, b - c
            rows = {r, rr} if 0 <= rr < n else {r}
            cols = {c, cc} if 0 <= cc < n else {c}
            orbit = tuple(sorted((u, v) for u in rows for v in cols))
            result.append(orbit)
            seen.update(orbit)
    return tuple(result)


def _axis_candidates(g, dimension):
    """All nontrivial folds, with no match score or evidence cutoff."""
    t = g if dimension == 0 else g.T
    n = len(t)
    candidates = []
    for a in range(1, 2 * n - 2):
        valid = True
        for r in range(n):
            s = a - r
            if r < s < n:
                both = (t[r] != 0) & (t[s] != 0)
                if np.any(t[r, both] != t[s, both]):
                    valid = False
                    break
        if valid:
            candidates.append(a)
    return candidates


def _fold_candidates(g):
    """A candidate must give one surviving value in EVERY in-bounds orbit."""
    candidates = []
    for a in _axis_candidates(g, 0):
        for b in _axis_candidates(g, 1):
            if all(len({int(g[p]) for p in orbit if g[p] != 0}) == 1
                   for orbit in _orbits(len(g), a, b)):
                candidates.append((a, b))
    return candidates


def _complete(g):
    if g.shape[0] != g.shape[1] or not 3 <= len(g) <= 5:
        raise ValueError("categorical input must be a 3..5 square")
    if not np.any(g == 0) or len(np.unique(g[g != 0])) < 2:
        raise ValueError("input needs erasures and at least two categories")
    candidates = _fold_candidates(g)
    if len(candidates) != 1:
        raise ValueError("input does not identify one repairable fold pair")
    axes = candidates[0]
    completed = g.copy()
    for orbit in _orbits(len(g), *axes):
        color = next(int(g[p]) for p in orbit if g[p] != 0)
        for p in orbit:
            completed[p] = color
    return completed, axes


def _expand(selectors, reference=None):
    """Also accepts a separate reference for explicit ablation checks."""
    if reference is None:
        reference = selectors
    n = len(selectors)
    macro = np.repeat(np.repeat(selectors, n, axis=0), n, axis=1)
    micro = np.tile(reference, (n, n))
    return np.where(macro == micro, macro, 0).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution for every member of the specified family."""
    completed, _ = _complete(_grid(input_grid))
    return _expand(completed)


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def _nearest_fill(g):
    """Concrete local-copy alternative; row-major ties are deterministic."""
    visible = [tuple(int(v) for v in p) for p in np.argwhere(g != 0)]
    result = g.copy()
    for r, c in np.argwhere(g == 0):
        p = min(visible, key=lambda q: (abs(q[0] - r) + abs(q[1] - c), q))
        result[r, c] = g[p]
    return result


def _features(g, completed, axes):
    n = len(g)
    a, b = axes
    triple = diagonal = fringe = 0
    for orbit in _orbits(n, a, b):
        triple += len(orbit) == 4 and sum(g[p] == 0 for p in orbit) == 3
    for r, c in np.argwhere(g == 0):
        rr, cc = a - r, b - c
        horizontal = 0 <= rr < n and g[rr, c] != 0
        vertical = 0 <= cc < n and g[r, cc] != 0
        diagonal += (not horizontal and not vertical
                     and 0 <= rr < n and 0 <= cc < n and g[rr, cc] != 0)
        fringe += not 0 <= rr < n or not 0 <= cc < n
    return {
        "n": n, "offsets": (a - n + 1, b - n + 1),
        "parity": (a % 2, b % 2), "categories": len(np.unique(completed)),
        "erased": int(np.count_nonzero(g == 0)), "triple": int(triple),
        "diagonal": int(diagonal), "fringe": int(fringe),
        "erased_colors": len(np.unique(completed[g == 0])),
        "local_fails": not np.array_equal(_nearest_fill(g), completed),
        "asymmetric_transpose": not np.array_equal(completed, completed.T),
        "edges": tuple(bool(x) for x in ((g[0] == 0).any(), (g[-1] == 0).any(),
                                          (g[:, 0] == 0).any(), (g[:, -1] == 0).any())),
    }


def _draw_field(rng, n, axes, k, style):
    orbits = _orbits(n, *axes)
    reps = np.array([o[0] for o in orbits])
    r, c = reps.T
    if style == 0:
        values = rng.integers(k, size=len(orbits))
    elif style == 1:  # Stepped diagonal/band fragments, reflected by the orbits.
        u, v = (int(x) for x in rng.integers(1, 3, size=2))
        values = (u * r + v * c + int(rng.integers(k))) % k
    elif style == 2:  # Compact patches by proximity to representative centers.
        centers = reps[rng.choice(len(reps), k, replace=False)]
        distances = abs(r[:, None] - centers[:, 0]) + abs(c[:, None] - centers[:, 1])
        values = np.argmin(distances, axis=1)
    else:  # Nested corner-distance fragments.
        r0, c0 = (int(x) for x in rng.integers(n, size=2))
        values = (np.maximum(abs(r - r0), abs(c - c0)) + int(rng.integers(k))) % k
    # All requested colors participate. This may modify a simple motif locally.
    values[rng.permutation(len(orbits))[:k]] = np.arange(k)
    palette = rng.choice(np.arange(1, 10, dtype=np.int8), size=k, replace=False)
    clean = np.empty((n, n), dtype=np.int8)
    for orbit, label in zip(orbits, values):
        for p in orbit:
            clean[p] = palette[int(label)]
    return clean


def _mask_field(rng, clean, axes, hard):
    orbits = _orbits(len(clean), *axes)
    eligible = [i for i, o in enumerate(orbits) if len(o) > 1]
    if hard:
        four = [i for i, o in enumerate(orbits) if len(o) == 4]
        i = int(rng.choice(four))
        orbit = orbits[i]
        survivor = orbit[int(rng.integers(4))]
        erased = {p for p in orbit if p != survivor}
        other = [j for j in eligible if j != i and clean[orbits[j][0]] != clean[survivor]]
        if not other:
            return None
        j = int(rng.choice(other))
        erased.add(orbits[j][int(rng.integers(len(orbits[j])))])
        protected_orbits = {i, j}
    else:
        i = int(rng.choice(eligible))
        orbit = orbits[i]
        erased = {orbit[int(rng.integers(len(orbit)))]}
        protected_orbits = {i}
    # Additional masks range from isolated points to small connected runs.
    mode = int(rng.integers(3))
    anchor = next(iter(sorted(erased)))
    for j in rng.permutation(len(orbits)):
        j = int(j)
        orbit = orbits[j]
        if j in protected_orbits or len(orbit) == 1:
            continue
        if rng.random() >= (0.25 if hard else 0.18):
            continue
        if mode == 0:
            count = int(rng.integers(1, len(orbit))) if hard else 1
            indices = rng.choice(len(orbit), count, replace=False)
            erased.update(orbit[int(t)] for t in indices)
        else:
            near = sorted(orbit, key=lambda p: (abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]), p))
            # Keep a witness, even when a strip/patch crosses an axis.
            count = 1 if mode == 1 else min(2, len(orbit) - 1)
            erased.update(near[:count])
    g = clean.copy()
    for p in erased:
        g[p] = 0
    return g


def _sample_pair(rng, n, offsets, hard, seen):
    axes = tuple(n - 1 + d for d in offsets)
    orbit_count = len(_orbits(n, *axes))
    for attempt in range(256):
        k = int(rng.integers(2, min(6, orbit_count) + 1))
        style = int(rng.integers(4)) if attempt < 96 else 0
        completed = _draw_field(rng, n, axes, k, style)
        if _fold_candidates(completed) != [axes]:
            continue
        g = _mask_field(rng, completed, axes, hard)
        if g is None or _signature(g) in seen or _fold_candidates(g) != [axes]:
            continue
        features = _features(g, completed, axes)
        if hard and (not features["local_fails"] or not features["asymmetric_transpose"]):
            continue
        seen.add(_signature(g))
        return {"input": g, "output": _expand(completed)}
    # No hidden exceptional rule or unbounded retry: failure remains explicit.
    raise RuntimeError("bounded sampling could not expose unique folds")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    tr, te = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    larger_test = bool(rng.random() < 0.40)
    sign = int(rng.choice([-1, 1]))
    training_sizes = (3, 4) if larger_test else (3, 4, 5)
    # Two complementary demonstrations survive even at the minimum 2/1 count.
    specs = [(int(rng.choice(training_sizes)), (0, 0), False)]
    n = int(rng.choice((4,) if larger_test else (3, 4, 5)))
    axis = int(rng.integers(2))
    offset = [0, 0]
    offset[axis] = sign
    if rng.random() < 0.5:
        offset[1 - axis] = sign
    specs.append((n, tuple(offset), True))
    for _ in range(2, tr):
        n = int(rng.choice(training_sizes))
        offsets = tuple(sign * int(rng.integers(2)) for _ in range(2))
        specs.append((n, offsets, bool(rng.random() < 0.5)))
    rng.shuffle(specs)
    seen = set()
    puzzle = {"train": [], "test": []}
    for n, offsets, hard in specs:
        puzzle["train"].append(_sample_pair(rng, n, offsets, hard, seen))
    for _ in range(te):
        n = 5 if larger_test else int(rng.choice((3, 4, 5)))
        limit = 2 if n == 5 and larger_test else 1
        offsets = [int(rng.integers(-limit, limit + 1)) for _ in range(2)]
        # A previously unseen opposite-signed offset also accompanies size shift.
        offsets[int(rng.integers(2))] = -sign * int(rng.integers(1, limit + 1))
        puzzle["test"].append(_sample_pair(rng, n, tuple(offsets), True, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "invalid counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            completed, axes = _complete(g)
            n = len(g)
            _require(_fold_candidates(completed) == [axes], "clean field has incidental fold ties")
            _require(out.shape == (n * n, n * n), "wrong expansion size")
            _require(2 <= len(np.unique(completed)) <= 9, "wrong category count")
            _require(np.all(completed[g != 0] == g[g != 0]), "visible payload changed")
            _require(np.all(completed != 0), "incomplete repair")
            _require(_signature(g) not in seen, "duplicate input")
            seen.add(_signature(g))
            # Recover the repaired categorical field independently from each
            # tile's own-coordinate diagonal; this field must be exactly B's repair.
            decoded = np.array([[out[r * n + r, c * n + c]
                                 for c in range(n)] for r in range(n)], dtype=np.int8)
            _require(np.array_equal(decoded, completed), "wrong repaired tile selectors")
            # Scalar tile checks do not call the vectorized expansion helper.
            for r in range(n):
                for c in range(n):
                    for u in range(n):
                        for v in range(n):
                            expected = completed[r, c] if completed[r, c] == completed[u, v] else 0
                            _require(out[r * n + u, c * n + v] == expected, "wrong support-mask cell")
            # Check both causal paths: restored selectors AND restored reference.
            raw = _expand(g)
            expected_changes = sum(int(np.count_nonzero(completed == color)) ** 2
                                   - int(np.count_nonzero(g == color)) ** 2
                                   for color in np.unique(completed))
            _require(expected_changes > 0 and np.count_nonzero(out != raw) == expected_changes,
                     "repair has no correct nonlocal expansion effect")
            _require(not np.array_equal(out, _expand(completed, g)), "raw-reference shortcut survives")
            _require(not np.array_equal(out, _expand(g, completed)), "raw-selector shortcut survives")
            _require(np.any(out == 0), "missing mismatch blanks")
            features[split].append(_features(g, completed, axes))
    # Evidence requirements are checks, never additional solving conventions.
    _require(any(f["offsets"] == (0, 0) for f in features["train"]), "no centered demonstration")
    _require(any(f["triple"] and f["diagonal"] and any(f["offsets"])
                 and f["erased_colors"] >= 2 and f["local_fails"]
                 and f["asymmetric_transpose"] for f in features["train"]),
             "training does not distinguish both folds and global masks")
    train_signs = [{int(np.sign(f["offsets"][d])) for f in features["train"]} for d in (0, 1)]
    for f in features["test"]:
        _require(f["triple"] and f["erased_colors"] >= 2, "test lacks joint repair")
        _require(any(int(np.sign(f["offsets"][d])) not in train_signs[d] for d in (0, 1)),
                 "test lacks the geometric shift")


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed study; retain only small counters and signatures in RAM."""
    structures, signatures, palettes = set(), set(), set()
    sizes, offsets, parities, categories, edges = set(), set(), set(), set(), set()
    totals = {"pairs": 0, "triple_pairs": 0, "fringe_pairs": 0, "larger_test_puzzles": 0}
    for seed in _SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed)), "nondeterminism")
        signatures.add(_puzzle_signature(puzzle))
        totals["larger_test_puzzles"] += min(len(p["input"]) for p in puzzle["test"]) > max(len(p["input"]) for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                completed, axes = _complete(g)
                _require(np.array_equal(solve(g), out), "solver disagreement")
                f = _features(g, completed, axes)
                sizes.add((split, len(g)))
                offsets.add(tuple(int(np.sign(d)) for d in f["offsets"]))
                parities.add(f["parity"])
                categories.add(f["categories"])
                palettes.add(tuple(int(c) for c in np.unique(completed)))
                edges.add(f["edges"])
                roles = {0: 0}
                structures.add((g.shape, tuple(roles.setdefault(int(c), len(roles)) for c in g.flat)))
                totals["pairs"] += 1
                totals["triple_pairs"] += f["triple"] > 0
                totals["fringe_pairs"] += f["fringe"] > 0
    _require(len(signatures) == 42, "cross-seed puzzle collision")
    _require(len(structures) >= 160 and len(palettes) >= 50, "insufficient structural variation")
    _require(all((split, n) in sizes for split in ("train", "test") for n in (3, 4, 5)), "size coverage")
    _require(len(offsets) == 9 and len(parities) == 4, "fold variation missing")
    _require(set(range(2, 7)) <= categories and len(edges) >= 10, "palette/mask variation missing")
    _require(totals["fringe_pairs"] > 10 and totals["larger_test_puzzles"] > 3, "boundary/shift variation missing")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(700 + tr * 10 + te, tr, te)
            _require(len(p["train"]) == tr and len(p["test"]) == te, "explicit count ignored")
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        generate_puzzle(71, tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, False)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    for coordinate in ((0, 0), (2, 3)):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in sample[s]] for s in sample}
        out = bad["test"][0]["output"]
        out[coordinate] = (int(out[coordinate]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted a damaged output")
    # A tied, empty, or contradictory input must not silently choose a fold.
    for g in (np.zeros((3, 3), dtype=np.int8), np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])):
        try:
            solve(g)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid repair accepted")
    totals.update(seeds=42, structures=len(structures), palettes=len(palettes),
                  signed_offset_combinations=len(offsets), parity_combinations=len(parities),
                  category_counts=sorted(categories), edge_contact_patterns=len(edges),
                  split_sizes=sorted(sizes))
    return totals
