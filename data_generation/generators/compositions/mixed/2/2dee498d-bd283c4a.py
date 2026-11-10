"""Stable column deduplication supplies the inventory for frequency packing.

No files, parent generators, or puzzle-specific state are used at runtime.
All generated arrays are small; self_test streams its 42-seed study in memory.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _columns(grid):
    return [tuple(int(c) for c in column) for column in grid.T]


def _deduplicate(grid):
    columns = list(dict.fromkeys(_columns(grid)))
    return np.asarray(columns, dtype=np.int8).T.copy()


def _rank(grid, tie="first"):
    # The insertion order is the first color occurrence in top-left row order.
    histogram = Counter(int(c) for c in grid.flat)
    first = {c: i for i, c in enumerate(histogram)}
    colors = sorted(histogram, key=lambda c: (
        -histogram[c], c if tie == "numeric" else first[c]))
    return colors, [histogram[c] for c in colors]


def _pack(grid, tie="first"):
    colors, counts = _rank(grid, tie)
    stream = np.repeat(colors, counts).astype(np.int8)
    h, w = grid.shape
    return stream.reshape(w, h)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Deduplicate columns, rank their cells, and pack bottom-up by columns.

    The family fixes the horizontal axis and a square latent result. Repeated
    rows are legitimate content and are never removed. Zero is an ordinary color.
    """
    source = _grid(input_grid)
    square = _deduplicate(source)
    if square.shape[0] != square.shape[1]:
        raise ValueError("the number of distinct columns must equal the height")
    return _pack(square)


def _signature(grid):
    return grid.shape, grid.tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _histogram(grid):
    return Counter(int(c) for c in grid.flat)


def _strict_counts(rng, n, k, profile):
    # Incrementing prefixes of k,k-1,...,1 always preserves strict ordering.
    counts = np.arange(k, 0, -1, dtype=int)
    remaining = n * n - int(counts.sum())
    if profile == "balanced":
        counts += remaining // k
        counts[:remaining % k] += 1
    else:
        if profile == "skewed":
            extra = min(remaining, int(n * n * .57))
            counts[0] += extra
            remaining -= extra
        while remaining:
            prefix = int(rng.integers(1, min(k, remaining) + 1))
            counts[:prefix] += 1
            remaining -= prefix
    return counts


def _counts(rng, n, tied, profile):
    total = n * n
    if tied:
        k = int(rng.integers(3, min(8, total - 1) + 1))
        t = int(rng.integers(1, (total - (k - 2)) // 2 + 1))
        tail = rng.multinomial(total - 2 * t - (k - 2),
                               rng.dirichlet(np.ones(k - 2))) + 1
        counts = np.concatenate(([t, t], tail))
    else:
        max_k = max(k for k in range(2, 11) if k * (k + 1) // 2 <= total)
        k = int(rng.integers(2, max_k + 1))
        counts = _strict_counts(rng, n, k, profile)
    return counts.astype(int)


def _arrange(rng, n, colors, counts, style):
    stream = np.repeat(colors, counts).astype(np.int8)
    r, c = np.indices((n, n))
    if style == "shuffle":
        rng.shuffle(stream)
        return stream.reshape(n, n).copy()
    if style == "row_bands":
        key = r + rng.normal(0, .9, (n, n))
    elif style == "diagonal":
        key = r + c * rng.uniform(.5, 1.5) + rng.normal(0, .8, (n, n))
    elif style == "patches":
        fields = rng.random((3, 3))
        key = fields[r * 3 // n, c * 3 // n] + rng.normal(0, .2, (n, n))
    else:
        raise ValueError("unknown arrangement style")
    result = np.empty(n * n, dtype=np.int8)
    result[np.argsort(key.ravel(), kind="stable")] = stream
    return result.reshape(n, n)


_STYLES = ("shuffle", "row_bands", "diagonal", "patches")


def _motif(rng, n, tied=False, strict_witness=False):
    for _ in range(256):
        profile = str(rng.choice(("balanced", "ordinary", "skewed")))
        if strict_witness:
            max_k = max(j for j in range(3, 6) if j * (j + 1) // 2 <= n * n)
            k = int(rng.integers(3, max_k + 1))
            counts = _strict_counts(rng, n, k, "balanced")
        else:
            counts = _counts(rng, n, tied, profile)
        # Extremely sparse minority inventories cannot support n distinct columns.
        if max(counts) > max(3, int(.74 * n * n)):
            continue
        colors = rng.choice(10, len(counts), replace=False)
        style = str(rng.choice(_STYLES))
        for attempt in range(24):
            square = _arrange(rng, n, colors, counts,
                              style if attempt < 12 else "shuffle")
            if len(set(_columns(square))) != n:
                continue
            if np.array_equal(square, _pack(square)):
                continue
            return square
    raise RuntimeError("bounded motif sampling exhausted")


def _multiplicities(rng, n, block=False):
    if block:
        return np.full(n, int(rng.integers(2, min(5, 30 // n) + 1)), dtype=int)
    result = np.full(n, 2, dtype=int)
    extra = int(rng.integers(0, min(30 - 2 * n, 6 * n) + 1))
    for _ in range(extra):
        candidates = np.flatnonzero(result < 8)
        result[int(rng.choice(candidates))] += 1
    return result


def _indices(rng, multiplicities, interleaved=True):
    n = len(multiplicities)
    result = list(range(n))
    if interleaved:
        # New columns still first appear in canonical order; their first
        # positions need not lie in the first n input columns.
        for _ in range(int(rng.integers(1, n + 1))):
            last = result.index(n - 1)
            choices = [i for i in range(n - 1)
                       if result.count(i) < multiplicities[i]]
            if not choices:
                break
            chosen = int(rng.choice(choices))
            position = int(rng.integers(result.index(chosen) + 1, last + 1))
            result.insert(position, chosen)
    remaining = [i for i in range(n)
                 for _ in range(int(multiplicities[i]) - result.count(i))]
    tail_style = int(rng.integers(3))
    if tail_style == 0:
        rng.shuffle(remaining)
    elif tail_style == 1:
        remaining.sort(reverse=bool(rng.integers(2)))
    else:
        left = Counter(remaining)
        remaining = []
        for wave in range(max(left.values(), default=0)):
            order = range(n) if wave % 2 == 0 else range(n - 1, -1, -1)
            remaining.extend(i for i in order if left[i] > wave)
    return result + remaining


def _inverts_rank(source, square):
    raw, clean = _histogram(source), _histogram(square)
    return any((clean[a] - clean[b]) * (raw[a] - raw[b]) < 0
               for a in clean for b in clean)


def _has_separated_copies(source):
    columns = _columns(source)
    runs = [c for i, c in enumerate(columns) if i == 0 or c != columns[i - 1]]
    return len(runs) > len(set(runs))


def _inverting_multiplicities(rng, square):
    n = square.shape[0]
    # Search a small bounded set of one-column over-representations first.
    for col in rng.permutation(n):
        for extra in rng.permutation(min(6, 30 - 2 * n)) + 1:
            copies = np.full(n, 2, dtype=int)
            copies[col] += int(extra)
            if _inverts_rank(square[:, np.repeat(np.arange(n), copies)], square):
                return copies
    return None


def _pair(square, rng, copies=None, interleaved=True):
    n = square.shape[0]
    if copies is None:
        copies = _multiplicities(rng, n, block=not interleaved)
    source = square[:, _indices(rng, copies, interleaved)].copy()
    return {"input": source, "output": _pack(square)}


def _tie_witness(square):
    target = _pack(square)
    numeric_fails = not np.array_equal(target, _pack(square, "numeric"))
    # Passing square.T changes only the color-first-occurrence order here.
    colors, counts = _rank(square.T)
    column_first = np.repeat(colors, counts).reshape(square.shape)[:, ::-1].T
    sorted_square = np.asarray(sorted(_columns(square)), dtype=np.int8).T
    return (numeric_fails and not np.array_equal(target, column_first)
            and not np.array_equal(target, _pack(sorted_square)))


def _strict_witness(pair):
    source, target = pair["input"], pair["output"]
    square = _deduplicate(source)
    counts = list(_histogram(square).values())
    return (len(counts) >= 3 and len(set(counts)) == len(counts)
            and len(set(int(c) for c in square[0])) < len(square)
            and _rank(square)[0] != list(_histogram(square))
            and _has_separated_copies(source)
            and _inverts_rank(source, square)
            and not np.array_equal(_pack(source[:, :len(square)]), target))


def _new_pair(rng, n, seen, kind="free"):
    for _ in range(256):
        square = _motif(rng, n, tied=(kind == "tie" or (
            kind == "free" and rng.random() < .35)), strict_witness=kind == "strict")
        if kind == "tie" and not _tie_witness(square):
            continue
        copies = _inverting_multiplicities(rng, square) if kind == "strict" else None
        if kind == "strict" and copies is None:
            continue
        interleaved = kind != "free" or rng.random() < .75
        pair = _pair(square, rng, copies, interleaved)
        if kind == "strict" and not _strict_witness(pair):
            continue
        key = _signature(pair["input"])
        if key not in seen:
            seen.add(key)
            return pair
    raise RuntimeError(f"bounded {kind} pair sampling exhausted")


def _repeat_contrast(rng, base, seen):
    square = _deduplicate(base["input"])
    columns = _columns(base["input"])
    copies = np.array([columns.count(c) for c in _columns(square)], dtype=int)
    for donor in rng.permutation(len(square)):
        if copies[donor] <= 2:
            continue
        for receiver in rng.permutation(len(square)):
            if donor == receiver or copies[receiver] == 8:
                continue
            changed = copies.copy()
            changed[donor] -= 1
            changed[receiver] += 1
            pair = _pair(square, rng, changed)
            if (_histogram(pair["input"]) != _histogram(base["input"])
                    and _signature(pair["input"]) not in seen):
                seen.add(_signature(pair["input"]))
                return pair
    raise RuntimeError("could not construct a changed-redundancy contrast")


def _scatter_contrast(rng, base, seen):
    square = _deduplicate(base["input"])
    n = len(square)
    colors, counts = _rank(square)
    for _ in range(256):
        changed = _arrange(rng, n, colors, counts, str(rng.choice(_STYLES)))
        if (len(set(_columns(changed))) != n or np.array_equal(changed, square)
                or np.array_equal(changed, _pack(changed))):
            continue
        pair = _pair(changed, rng, interleaved=bool(rng.integers(2)))
        if _signature(pair["input"]) not in seen:
            seen.add(_signature(pair["input"]))
            return pair
    raise RuntimeError("could not construct an equal-inventory contrast")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    seen = set()
    n0 = int(rng.integers(3, 7))
    n1 = int(rng.choice([n for n in range(3, 7) if n != n0]))
    train = [_new_pair(rng, n0, seen, "strict"), _new_pair(rng, n1, seen, "tie")]
    if num_train >= 3:
        train.append(_repeat_contrast(rng, train[0], seen))
    if num_train >= 4:
        train.append(_scatter_contrast(rng, train[0], seen))
    while len(train) < num_train:
        train.append(_new_pair(rng, int(rng.integers(2, 9)), seen))
    max_train = max(p["output"].shape[0] for p in train)
    test = [_new_pair(rng, int(rng.integers(max_train + 1, 16)), seen)
            for _ in range(num_test)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _traversal_alternatives(square):
    """Seven explicit packing shortcuts, used only to check demonstrations."""
    stream = np.repeat(*_rank(square))
    n = len(square)
    results = []
    for major in ("column", "row"):
        for reverse_major in (False, True):
            for reverse_minor in (False, True):
                if (major, reverse_major, reverse_minor) == ("column", False, True):
                    continue
                outer = list(range(n))[::(-1 if reverse_major else 1)]
                inner = list(range(n))[::(-1 if reverse_minor else 1)]
                positions = [(b, a) if major == "column" else (a, b)
                             for a in outer for b in inner]
                result = np.empty_like(square)
                for position, color in zip(positions, stream):
                    result[position] = color
                results.append(result)
    return results


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input/output")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            columns = _columns(source)
            retained = list(dict.fromkeys(columns))
            n = source.shape[0]
            _require(len(retained) == n and target.shape == (n, n),
                     "latent square dimensions disagree")
            copies = [columns.count(c) for c in retained]
            _require(2 <= n <= 15 and all(2 <= c <= 8 for c in copies),
                     "generated vector multiplicities out of range")
            square = np.array(retained, dtype=np.int8).T
            _require(_histogram(square) == _histogram(target),
                     "deduplicated exact cell inventory was not conserved")
            # Check the labeled traversal directly: one run per color, with
            # precisely the deduplicated counts and the specified ordering.
            colors, counts = _rank(square)
            positions = [int(target[r, c]) for c in range(n) for r in range(n - 1, -1, -1)]
            expected = [color for color, count in zip(colors, counts) for _ in range(count)]
            _require(positions == expected, "frequency order, tie, or packing violated")
            _require(np.array_equal(target, solve(source)), "input-only solve mismatch")
            _require(not np.array_equal(square, target), "repacking has no effect")
            key = _signature(source)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training.append((square, target))
    pairs = [{k: np.asarray(v) for k, v in p.items()} for p in puzzle["train"]]
    _require(any(_strict_witness(p) for p in pairs),
             "training needs crop and raw-count rank counterexamples with distinct counts")
    _require(any(_tie_witness(square) for square, _ in training),
             "training needs a first-occurrence tie counterexample")
    alternatives = [_traversal_alternatives(square) for square, _ in training]
    _require(all(any(not np.array_equal(options[i], target)
                     for options, (_, target) in zip(alternatives, training))
                 for i in range(7)), "a basic traversal alternative fits every demonstration")
    _require(min(np.asarray(p["input"]).shape[0] for p in puzzle["test"])
             > max(square.shape[0] for square, _ in training),
             "generated test must extrapolate the latent square size")


def self_test():
    """Bounded, deterministic, in-memory diversity and semantic regression study."""
    signatures, shapes, sizes, palettes, structures = set(), set(), set(), set(), set()
    multiplicities, palette_sizes, ranks = set(), set(), set()
    tied = strict = nonadjacent = unequal = partial = aligned = near = 0
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = tuple(_signature(p[r]) for s in ("train", "test")
                            for p in puzzle[s] for r in ("input", "output"))
        signatures.add(fingerprint)
        again = generate_puzzle(seed)
        _require(fingerprint == tuple(_signature(p[r]) for s in ("train", "test")
                                      for p in again[s] for r in ("input", "output")),
                 "seed is not reproducible")
        _require(np.array_equal(puzzle["train"][0]["output"], puzzle["train"][2]["output"])
                 and np.array_equal(puzzle["train"][0]["output"], puzzle["train"][3]["output"]),
                 "controlled inventory contrasts disagree")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                square = _deduplicate(source)
                cols = _columns(source)
                copies = [cols.count(c) for c in _columns(square)]
                multiplicities.update(copies)
                unequal += len(set(copies)) > 1
                colors, counts = _rank(square)
                ranks.add(tuple(colors))
                tied += len(set(counts)) < len(counts)
                strict += len(set(counts)) == len(counts)
                boundaries = np.cumsum(counts)[:-1] % len(square)
                partial += bool(np.any(boundaries != 0))
                aligned += bool(np.any(boundaries == 0))
                nonadjacent += _has_separated_copies(source)
                near += any(sum(a != b for a, b in zip(x, y)) == 1
                            for i, x in enumerate(_columns(square))
                            for y in _columns(square)[i + 1:])
                shapes.add(source.shape)
                sizes.add(len(square))
                palettes.add(tuple(sorted(colors)))
                palette_sizes.add(len(colors))
                normalized, roles = [], {}
                for c in source.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structures.add((source.shape, bytes(normalized)))
                # Mutating a returned answer cannot change an input-only solve.
                original = source.copy()
                predicted = solve(source)
                _require(np.array_equal(source, original) and not np.shares_memory(source, predicted),
                         "solve mutated or aliased its input")
                _require(np.array_equal(predicted, target), "solve failed")
                pairs_checked += 1
    _require(len(signatures) == 42 and len(shapes) >= 25 and len(structures) >= 150,
             "shape or structural variation too narrow")
    _require(len(sizes) >= 9 and 15 in sizes and len(palette_sizes) >= 5,
             "size or palette complexity variation too narrow")
    _require(tied >= 42 and strict >= 42 and unequal >= 42 and nonadjacent >= 150,
             "frequency or repetition variation too narrow")
    _require(partial >= 150 and aligned >= 5 and near >= 1,
             "boundary or exact-vector contrasts too narrow")
    _require({2, 3, 4, 5, 6, 7, 8} <= multiplicities, "copy-count coverage too narrow")
    # Every legal explicit count combination, plus type and boundary errors.
    legal_count_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1234 + nt * 10 + ne, nt, ne)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ne),
                     "explicit pair counts ignored")
            legal_count_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4., 1), (4, 1.), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Equal column histograms do not imply equal full-height vectors.
    square = np.array([[2, 2, 7], [2, 9, 7], [9, 2, 7]], dtype=np.int8)
    source = square[:, [0, 0, 1, 2, 2, 1, 2]]
    _require(np.array_equal(solve(source), np.array([[2, 7, 9], [2, 7, 9], [2, 2, 7]])),
             "same-histogram columns were merged or raw frequencies were counted")
    _require(not _has_separated_copies(np.array([[1, 1, 1, 2, 2, 2]]))
             and _has_separated_copies(np.array([[1, 1, 2, 1]])),
             "adjacent runs were mistaken for separated copies")
    # Hand-calculated tiny tie case also separates row-first from column-first.
    square = np.array([[1, 9, 2], [2, 9, 2], [2, 9, 9]], dtype=np.int8)
    source = square[:, [0, 1, 0, 2, 2, 1]]
    _require(np.array_equal(solve(source), np.array([[9, 2, 1], [9, 2, 2], [9, 9, 2]])),
             "hand-worked frequency/tie/packing regression failed")
    # Near-equal columns and repeated rows must survive until the counting step.
    square = np.array([[4, 4, 4], [4, 4, 4], [4, 7, 2]], dtype=np.int8)
    source = square[:, [0, 0, 1, 2, 1, 2]]
    _require(np.array_equal(solve(source), np.array([[4, 4, 2], [4, 4, 7], [4, 4, 4]])),
             "near-match columns or legitimate repeated rows were lost")
    _require(np.array_equal(solve(np.array([[0, 0, 0]])), np.array([[0]])),
             "one-cell latent boundary failed")
    return {"seeds": 42, "pairs_checked": pairs_checked, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "latent_sizes": sorted(sizes),
            "palette_sizes": sorted(palette_sizes), "palettes": len(palettes),
            "color_normalized_inputs": len(structures), "rank_orders": len(ranks),
            "tied_pairs": tied, "distinct_count_pairs": strict, "unequal_copy_pairs": unequal,
            "nonadjacent_repeat_pairs": nonadjacent, "partial_boundary_pairs": partial,
            "aligned_boundary_pairs": aligned, "near_match_pairs": near,
            "legal_count_calls": legal_count_calls,
            "note": "Generator-owned checks; no independent solver or blind prediction."}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
