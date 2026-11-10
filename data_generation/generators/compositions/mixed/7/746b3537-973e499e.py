"""Run-compressed categorical support tensors. No external data or state."""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _collapse(a):
    """Compute BOTH run partitions on the original full pixel vectors."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return a[np.ix_(rows, cols)].copy()


def _support(logical):
    n = logical.shape[0]
    # Axes are macro-row, within-row, macro-column, within-column.
    selectors = logical[:, None, :, None]
    references = logical[None, :, None, :]
    return np.where(selectors == references, selectors, 0).reshape(n*n, n*n).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    if np.any(a == 0):
        raise ValueError("zero is the fixed output blank, not an input category")
    logical = _collapse(a)
    h, w = logical.shape
    if h != w or not 2 <= h <= 5 or len(np.unique(logical)) < 2:
        raise ValueError("compressed mosaic must be square, side 2..5, with 2+ categories")
    return _support(logical)


def _minimal(a):
    return (np.all(np.any(a[1:] != a[:-1], axis=1))
            and np.all(np.any(a[:, 1:] != a[:, :-1], axis=0)))


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in
               (a.T, np.flipud(a), np.fliplr(a), np.rot90(a)))


def _local_run_traps(a):
    # A local equal pair need not mean equality of the entire vectors.
    return bool(np.any(a[1:] == a[:-1]) and np.any(a[:, 1:] == a[:, :-1]))


def _recurs(a, axis):
    vectors = a if axis == 0 else a.T
    return any(np.array_equal(vectors[i], vectors[j])
               for i in range(len(vectors)) for j in range(i+2, len(vectors)))


def _labels(rng, n, k, style, recurrence=False, anchor=False):
    """Bounded sampling of different spatial category mechanisms."""
    side = n-1 if recurrence else n
    y, x = np.indices((side, side))
    for attempt in range(512):
        mode = style if attempt < 96 else 0
        if mode == 0:  # arbitrary frequency-skewed scatter
            p = rng.dirichlet(np.full(k, 1.4))
            a = rng.choice(k, size=(side, side), p=p)
        elif mode == 1:  # Manhattan color domains
            centers = rng.integers(side, size=(k, 2))
            distance = (abs(y[None]-centers[:, 0, None, None])
                        + abs(x[None]-centers[:, 1, None, None]))
            a = np.argmin(distance+rng.random(distance.shape)*0.8, axis=0)
        elif mode == 2:  # periodic offset rows, including checkerboards
            a = (y + int(rng.integers(1, k))*x + int(rng.integers(k))) % k
        elif mode == 3:  # reflected arrangements, odd side avoids adjacent duplicates
            a = rng.integers(k, size=(side, side))
            a[:, (side+1)//2:] = a[:, :side//2][:, ::-1]
        else:  # broad regions with independent cell perturbations
            a = ((y*2//side)*2 + x*2//side) % k
            mask = rng.random(a.shape) < 0.35
            a[mask] = rng.integers(k, size=int(mask.sum()))
        # Put missing categories in cells whose current category can spare one.
        flat = a.reshape(-1)
        for label in range(k):
            if not np.any(flat == label):
                counts = np.bincount(flat, minlength=k)
                donors = np.flatnonzero(counts[flat] > 1)
                flat[int(rng.choice(donors))] = label
        if anchor:
            # A singleton, plus repeated categories, is visible in the logical grid.
            locations = np.flatnonzero(flat == k-1)
            keep = int(rng.choice(locations))
            for loc in locations:
                if loc != keep:
                    flat[loc] = int(rng.integers(k-1))
        a = flat.reshape(side, side)
        if recurrence:
            a = np.concatenate((a, a[:1]), axis=0)
            a = np.concatenate((a, a[:, :1]), axis=1)
        if not _minimal(a) or len(np.unique(a)) != k:
            continue
        if anchor and (not _asymmetric(a) or not _local_run_traps(a)):
            continue
        return a.astype(np.int8)
    raise RuntimeError("bounded logical-grid sampling exhausted")


def _widths(rng, n, mode, large=False):
    if mode == 0:
        widths = np.ones(n, dtype=int)
    elif mode == 1:
        widths = np.full(n, int(rng.integers(2, 5)), dtype=int)
    elif mode == 2:
        widths = np.ones(n, dtype=int)
        widths[int(rng.integers(n))] = int(rng.integers(2, 6))
    else:
        widths = rng.integers(1, 5, size=n)
        if np.all(widths == widths[0]):
            widths[0] = 1 if widths[0] > 1 else 2
    if large:
        widths[int(rng.integers(n))] = int(rng.integers(6, 13))
    # All widths remain positive, with a maximum physical side of 30.
    while widths.sum() > 30:
        index = int(np.argmax(widths))
        widths[index] -= 1
    return widths


def _key(a):
    return (a.shape, a.tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    tr = _integer(num_train, "num_train", 2, 10)
    te = _integer(num_test, "num_test", 1, 4)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    category_cap = int(rng.integers(3, 10))
    size_shift = bool(rng.integers(2))
    result = {"train": [], "test": []}
    seen = set()

    def add(split, n, k, recurrence=False, anchor=False, thick=False, demo=False):
        for _ in range(80):
            labels = _labels(rng, n, k, int(rng.integers(5)), recurrence, anchor)
            palette = rng.choice(np.arange(1, 10), size=k, replace=False).astype(np.int8)
            logical = palette[labels]
            if demo:
                rw = _widths(rng, n, 3)
                cw = _widths(rng, n, 3)
            else:
                geometry = int(rng.integers(4))
                # Include one-axis stretching, and uniform and nonuniform scales.
                rw = _widths(rng, n, 0 if geometry == 0 else int(rng.integers(1, 4)))
                cw = _widths(rng, n, 0 if geometry == 1 else int(rng.integers(1, 4)))
            if thick:
                if rng.integers(2):
                    rw = _widths(rng, n, 3, large=True)
                else:
                    cw = _widths(rng, n, 3, large=True)
            inp = np.repeat(np.repeat(logical, rw, axis=0), cw, axis=1)
            if _key(inp) in seen:
                continue
            seen.add(_key(inp))
            result[split].append({"input": inp, "output": _support(logical)})
            return
        raise RuntimeError("bounded distinct-input sampling exhausted")

    # Two complementary witnesses suffice even for the legal 2/1 count case.
    n = int(rng.integers(3, 5))
    add("train", n, int(rng.integers(2, min(category_cap, (n-1)**2)+1)),
        recurrence=True, demo=True)
    n = 4 if category_cap >= 8 else int(rng.integers(3, 5))
    add("train", n, category_cap, anchor=True)
    for _ in range(tr-2):
        n = int(rng.integers(2, 5))
        recurrence = n >= 3 and rng.random() < 0.30
        cap = min(category_cap, (n-1)**2 if recurrence else n*n-1)
        add("train", n, int(rng.integers(2, cap+1)), recurrence=recurrence)
    rng.shuffle(result["train"])
    for index in range(te):
        n = 5 if index == 0 and size_shift else int(rng.integers(2, 6))
        recurrence = n >= 3 and rng.random() < 0.42
        cap = min(category_cap, (n-1)**2 if recurrence else n*n-1)
        add("test", n, int(rng.integers(2, cap+1)), recurrence=recurrence,
            thick=index == 0 and not size_shift)
    return result


def _scan(a):
    """Scalar run scan for validation; no call to the solver's run partition."""
    starts, lengths = [], []
    for vectors in (a, a.T):
        positions, sizes = [0], [1]
        for i in range(1, len(vectors)):
            if np.array_equal(vectors[i], vectors[i-1]):
                sizes[-1] += 1
            else:
                positions.append(i)
                sizes.append(1)
        starts.append(positions)
        lengths.append(sizes)
    return a[np.ix_(*starts)], lengths


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert np.all(a > 0)
            logical, (rw, cw) = _scan(a)
            n = len(logical)
            assert logical.shape == (n, n) and 2 <= n <= 5
            assert _minimal(logical) and len(np.unique(logical)) >= 2
            assert np.array_equal(a, np.repeat(np.repeat(logical, rw, axis=0), cw, axis=1))
            assert _key(a) not in seen
            seen.add(_key(a))
            assert a.shape != logical.shape, "every generated pair must need compression"
            assert out.shape == (n*n, n*n)
            assert set(np.unique(out)) == set(np.unique(a)) | {0}
            # Exact scalar relation at all four coordinates, independent of _support.
            for r in range(n):
                for c in range(n):
                    for u in range(n):
                        for v in range(n):
                            color = logical[r, c]
                            expected = color if color == logical[u, v] else 0
                            assert out[r*n+u, c*n+v] == expected
            colors, counts = np.unique(logical, return_counts=True)
            for color, count in zip(colors, counts):
                assert np.count_nonzero(out == color) == int(count)**2
            # A same-sized solid-block substitute omits B and must fail.
            solid = np.repeat(np.repeat(logical, n, axis=0), n, axis=1)
            assert not np.array_equal(out, solid)
            if split == "train":
                train.append((logical, rw, cw))
    assert any(_recurs(l, 0) and _recurs(l, 1)
               and len(set(r)) > 1 and len(set(c)) > 1 for l, r, c in train)
    assert any(_asymmetric(l) and _local_run_traps(l)
               and np.any(np.unique(l, return_counts=True)[1] == 1) for l, _, _ in train)
    max_train_k = max(len(np.unique(l)) for l, _, _ in train)
    assert all(len(np.unique(p["input"])) <= max_train_k for p in puzzle["test"])
    # First test shifts logical side or reaches a new, thicker pixel-run regime.
    test_l, test_widths = _scan(_grid(puzzle["test"][0]["input"]))
    assert len(test_l) > max(len(l) for l, _, _ in train) or max(map(max, test_widths)) >= 6


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    # A hand-computed mask example, not generated labels checked against solve.
    raw = np.array([[1, 1, 2], [1, 1, 2], [2, 2, 1]])
    expected = np.array([[1, 0, 0, 2], [0, 1, 2, 0],
                         [0, 2, 1, 0], [2, 0, 0, 1]])
    assert np.array_equal(solve(raw), expected)
    # Full-vector boundaries: the first row starts with a local equal-color run,
    # yet columns 0 and 1 differ elsewhere and must both survive.
    l = np.array([[1, 1, 2], [3, 2, 3], [1, 1, 2]], dtype=np.int8)
    a = np.repeat(np.repeat(l, [2, 1, 3], 0), [1, 3, 2], 1)
    assert np.array_equal(_collapse(a), l)
    assert _recurs(_collapse(a), 0)
    # Expansion thickness is a nuisance, never a frequency weight in the masks.
    b = np.repeat(np.repeat(l, [1, 5, 1], 0), [4, 1, 1], 1)
    assert np.array_equal(solve(a), solve(b))
    # Recoloring and transposition commute with the equality construction.
    colors = np.arange(10, dtype=np.int8)
    colors[1:] = colors[1:][::-1]
    assert np.array_equal(solve(colors[a]), colors[solve(a)])
    assert np.array_equal(solve(a.T), solve(a).T)
    signatures, layouts, shapes, widths, categories, sides = set(), set(), set(), set(), set(), set()
    one_axis, singleton, recurring, asymmetric = False, False, False, False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        assert signature == _digest(generate_puzzle(seed))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                l, (rw, cw) = _scan(a)
                mapping = {}
                norm = tuple(mapping.setdefault(int(x), len(mapping)) for x in l.flat)
                layouts.add((l.shape, norm))
                shapes.add(a.shape)
                widths.add((tuple(rw), tuple(cw)))
                categories.add(len(np.unique(l)))
                sides.add((split, len(l)))
                one_axis |= all(w == 1 for w in rw) or all(w == 1 for w in cw)
                singleton |= np.any(np.unique(l, return_counts=True)[1] == 1)
                recurring |= _recurs(l, 0) and _recurs(l, 1)
                asymmetric |= _asymmetric(l)
                assert np.array_equal(solve(a.copy()), pair["output"])
    assert len(signatures) == 42 and len(layouts) >= 100
    assert len(shapes) >= 40 and len(widths) >= 120
    assert categories == set(range(2, 10))
    assert all(("test", n) in sides for n in range(2, 6))
    assert one_axis and singleton and recurring and asymmetric
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(999+tr*10+te, tr, te)
            validate_generated_puzzle(p)
            assert len(p["train"]) == tr and len(p["test"]) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "logical_color_normalized_layouts": len(layouts),
            "input_shapes": len(shapes), "stretch_profiles": len(widths),
            "category_counts": sorted(categories), "one_axis_stretch": bool(one_axis)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
